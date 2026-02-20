import os
import sys
import importlib.util
from pathlib import Path

import torch
import pytorch_lightning as pl

from utils.coco_eval_utils import (
    convert_preds_to_coco,
    gather_outputs_across_processes,
    broadcast_object,
    compute_coco_metrics,
)


def _import_from_yolo_repo(repo_path: str, module_name: str):
    """Import a module from the YOLOv5 repository using normal module resolution.

    Args:
        repo_path: Path to YOLOv5 repository
        module_name: Module name to import (e.g., 'models.yolo', 'utils.loss')
    """
    repo = Path(repo_path).expanduser().resolve()
    if not repo.exists():
        raise FileNotFoundError(
            f"YOLOv5 repo not found at: {repo}. Set model.yolov5.repo_path to official YOLOv5 source."
        )

    repo_str = str(repo)

    # Temporarily manipulate sys.path and sys.modules to isolate YOLOv5 imports
    # This prevents YOLOv5's "from models.X import ..." from finding our project's models/
    original_path = sys.path.copy()
    original_modules = {}

    try:
        # Remove current directory and project paths from sys.path
        sys.path = [p for p in sys.path if p not in ("", ".", str(Path.cwd()))]

        # Add YOLOv5 repo to the front
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        # Cache any existing modules that might conflict
        for key in list(sys.modules.keys()):
            if key.startswith(("models", "utils", "detect", "export")):
                original_modules[key] = sys.modules.pop(key)

        # Import the module
        module = importlib.import_module(module_name)
        return module

    except ImportError as e:
        raise ImportError(
            f"Could not import '{module_name}' from YOLOv5 at {repo}. "
            f"Make sure the YOLOv5 repository structure is correct.\n{e}"
        )
    finally:
        # Restore sys.path
        sys.path = original_path

        # Don't restore modules - we want to keep the imported yolov5 modules
        # but exclude them from being re-imported


def _ensure_repo_import(repo_path: str):
    repo = Path(repo_path).expanduser().resolve()
    if not repo.exists():
        raise FileNotFoundError(
            f"YOLOv5 repo not found at: {repo}. Set model.yolov5.repo_path to official YOLOv5 source."
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


class YOLOv5LightningModule(pl.LightningModule):
    """Lightning module for official YOLOv5 training with COCO metrics."""

    def __init__(
        self,
        config,
        yolo_repo_path: str,
        model_to_coco: dict,
        val_coco_gt=None,
        test_coco_gt=None,
    ):
        super().__init__()
        self.config = config
        self.yolo_repo_path = str(_ensure_repo_import(yolo_repo_path))
        self.model_to_coco = {int(k): int(v) for k, v in model_to_coco.items()}
        self.val_coco_gt = val_coco_gt
        self.test_coco_gt = test_coco_gt

        # Import modules explicitly from YOLOv5 repo
        yolo_module = _import_from_yolo_repo(self.yolo_repo_path, "models.yolo")
        Model = yolo_module.Model

        utils_loss = _import_from_yolo_repo(self.yolo_repo_path, "utils.loss")
        ComputeLoss = utils_loss.ComputeLoss

        utils_general = _import_from_yolo_repo(self.yolo_repo_path, "utils.general")
        non_max_suppression = utils_general.non_max_suppression

        self._non_max_suppression = non_max_suppression

        model_cfg = self.config.model.yolov5
        nc = len(self.config.model.label_map)
        model_def = model_cfg.model_cfg
        if not os.path.isabs(model_def):
            model_def = str(Path(self.yolo_repo_path) / model_def)
        model = Model(model_def, ch=3, nc=nc)
        self._load_weights_if_available(model, model_cfg.weights)

        # Set hyperparameters on the model (required by ComputeLoss)
        model.hyp = dict(model_cfg.hyp)

        self.model = model
        self.compute_loss = ComputeLoss(self.model)
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def _load_weights_if_available(self, model, weights_path: str):
        if not weights_path:
            return
        w = Path(weights_path).expanduser()
        if not w.is_absolute():
            w = Path(self.yolo_repo_path) / w
        if not w.exists():
            return
        ckpt = torch.load(str(w), map_location="cpu")
        state_dict = ckpt["model"].float().state_dict() if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)

    @property
    def stride(self):
        s = int(max(self.model.stride)) if hasattr(self.model, "stride") else 32
        return max(s, 32)

    def forward(self, images):
        return self.model(images)

    def _extract_model_outputs(self, model_output):
        # YOLOv5 eval forward returns (pred, train_out); train forward returns train_out.
        if isinstance(model_output, tuple) and len(model_output) == 2:
            return model_output[0], model_output[1]
        if isinstance(model_output, list):
            return None, model_output
        return model_output, None

    def _map_label_ids(self, yolo_label_tensor):
        mapped = []
        for label in yolo_label_tensor.tolist():
            mapped.append(self.model_to_coco.get(int(label), int(label)))
        return torch.tensor(mapped, device=yolo_label_tensor.device, dtype=torch.int64)

    def _undo_letterbox(self, boxes_xyxy: torch.Tensor, shape_meta):
        """Map boxes from letterboxed model-input coordinates back to original image coordinates."""
        (h0, w0), (ratio, (dw, dh)) = shape_meta
        boxes = boxes_xyxy.clone()
        boxes[:, [0, 2]] -= float(dw)
        boxes[:, [1, 3]] -= float(dh)
        boxes[:, :4] /= float(ratio)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, float(w0))
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, float(h0))
        return boxes

    def _log_loss_items(self, split: str, loss_items):
        """
        Log all available YOLO loss components.
        Official YOLOv5 typically returns [box, obj, cls], but we keep this dynamic
        so extra terms are also logged if present.
        """
        if loss_items is None:
            return

        if isinstance(loss_items, torch.Tensor):
            values = loss_items.detach().float().view(-1).tolist()
        elif isinstance(loss_items, (list, tuple)):
            values = []
            for item in loss_items:
                if isinstance(item, torch.Tensor):
                    values.append(float(item.detach().float().item()))
                else:
                    values.append(float(item))
        else:
            return

        default_names = ["box_loss", "obj_loss", "cls_loss"]
        for idx, value in enumerate(values):
            if idx < len(default_names):
                name = default_names[idx]
            else:
                name = f"loss_{idx}"
            self.log(f"{split}/{name}", value, on_step=False, on_epoch=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        images, targets = batch[0], batch[1]
        images = images.to(self.device, non_blocking=True).float() / 255.0

        # Ensure targets and model are on the same device as model
        targets = targets.to(self.device, non_blocking=True).float()
        self.model = self.model.to(self.device)

        _, train_out = self._extract_model_outputs(self.model(images))
        if train_out is None:
            # If model returns only train outputs in train mode, use raw output.
            train_out = self.model(images)

        loss, loss_items = self.compute_loss(train_out, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self._log_loss_items("train", loss_items)
        return loss

    @torch.no_grad()
    def _run_eval_step(self, batch, split_name: str):
        images, targets, _, shapes, batch_image_ids = batch
        images = images.to(self.device, non_blocking=True).float() / 255.0

        # Ensure targets and model are on the same device
        targets = targets.to(self.device, non_blocking=True).float()
        self.model = self.model.to(self.device)

        infer_out, train_out = self._extract_model_outputs(self.model(images))
        if infer_out is None and train_out is not None:
            infer_out = train_out

        if train_out is not None:
            val_loss, loss_items = self.compute_loss(train_out, targets)
            self.log(f"{split_name}/loss", val_loss, on_step=False, on_epoch=True, sync_dist=True)
            self._log_loss_items(split_name, loss_items)

        pred_list = self._non_max_suppression(
            infer_out,
            conf_thres=float(self.config.model.detection_threshold),
            iou_thres=float(self.config.model.yolov5.iou_threshold),
            max_det=int(self.config.model.max_detections),
        )

        result_map = {}
        image_ids = []
        for sample_idx, pred in enumerate(pred_list):
            image_id = int(batch_image_ids[sample_idx])
            image_ids.append(int(image_id))
            if pred is None or len(pred) == 0:
                result_map[int(image_id)] = {
                    "boxes": torch.empty((0, 4), dtype=torch.float32),
                    "scores": torch.empty((0,), dtype=torch.float32),
                    "labels": torch.empty((0,), dtype=torch.int64),
                }
                continue

            predn = pred.clone()
            predn[:, :4] = self._undo_letterbox(predn[:, :4], shapes[sample_idx])
            mapped_labels = self._map_label_ids(predn[:, 5].to(torch.int64))

            result_map[int(image_id)] = {
                "boxes": predn[:, :4].detach().cpu(),
                "scores": predn[:, 4].detach().cpu(),
                "labels": mapped_labels.detach().cpu(),
            }

        return {"predictions": result_map, "image_ids": image_ids}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        output = self._run_eval_step(batch, split_name="val")
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        all_outputs = gather_outputs_across_processes(self.validation_step_outputs)
        metrics = {}
        if self.trainer.is_global_zero:
            predictions = []
            image_ids = []
            for batch_out in all_outputs:
                predictions.extend(convert_preds_to_coco(batch_out["predictions"]))
                image_ids.extend(batch_out["image_ids"])

            if len(predictions) > 0:
                metrics = compute_coco_metrics(
                    coco_gt=self.val_coco_gt,
                    predictions=predictions,
                    image_ids=list(set(image_ids)),
                    max_detections=int(self.config.model.max_detections),
                    label_map=self.config.model.label_map,
                )
            else:
                metrics = {"map": 0.0, "map_50": 0.0, "map_75": 0.0}

        metrics = broadcast_object(metrics, src=0)
        for key, value in metrics.items():
            self.log(f"val/{key}", value, prog_bar=(key in {"map", "map_50"}), sync_dist=True)
            if key == "map":
                self.log("val_map", value, prog_bar=False, sync_dist=True)

        self.validation_step_outputs.clear()

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        output = self._run_eval_step(batch, split_name="test")
        self.test_step_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        all_outputs = gather_outputs_across_processes(self.test_step_outputs)
        metrics = {}
        if self.trainer.is_global_zero:
            predictions = []
            image_ids = []
            for batch_out in all_outputs:
                predictions.extend(convert_preds_to_coco(batch_out["predictions"]))
                image_ids.extend(batch_out["image_ids"])

            if len(predictions) > 0:
                metrics = compute_coco_metrics(
                    coco_gt=self.test_coco_gt,
                    predictions=predictions,
                    image_ids=list(set(image_ids)),
                    max_detections=int(self.config.model.max_detections),
                    label_map=self.config.model.label_map,
                )
            else:
                metrics = {"map": 0.0, "map_50": 0.0, "map_75": 0.0}

        metrics = broadcast_object(metrics, src=0)
        for key, value in metrics.items():
            self.log(f"test/{key}", value, prog_bar=(key in {"map", "map_50"}), sync_dist=True)

        self.test_step_outputs.clear()

    @torch.no_grad()
    def predict_batch(self, images, conf_threshold=None, iou_threshold=None):
        """Simple inference helper for external inference scripts."""
        if conf_threshold is None:
            conf_threshold = float(self.config.model.detection_threshold)
        if iou_threshold is None:
            iou_threshold = float(self.config.model.yolov5.iou_threshold)

        images = images.to(self.device).float() / 255.0
        infer_out, _ = self._extract_model_outputs(self.model(images))
        preds = self._non_max_suppression(
            infer_out,
            conf_thres=float(conf_threshold),
            iou_thres=float(iou_threshold),
            max_det=int(self.config.model.max_detections),
        )
        return preds

    def configure_optimizers(self):
        opt_cfg = self.config.optimizer.optimizer
        sch_cfg = self.config.scheduler

        params = [p for p in self.model.parameters() if p.requires_grad]

        # Determine optimizer type based on available config keys
        # If momentum and nesterov are present, use SGD; otherwise use AdamW
        if "momentum" in opt_cfg and "nesterov" in opt_cfg:
            optimizer = torch.optim.SGD(
                params,
                lr=float(opt_cfg.lr),
                momentum=float(opt_cfg.momentum),
                weight_decay=float(opt_cfg.weight_decay),
                nesterov=bool(opt_cfg.nesterov),
            )
        else:
            optimizer = torch.optim.AdamW(
                params,
                lr=float(opt_cfg.lr),
                weight_decay=float(opt_cfg.weight_decay),
            )

        if sch_cfg.type == "onecycle":
            total_steps = max(1, self.trainer.estimated_stepping_batches)
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=float(opt_cfg.lr),
                    total_steps=total_steps,
                    pct_start=float(sch_cfg.pct_start),
                    anneal_strategy="cos",
                ),
                "interval": "step",
            }
        elif sch_cfg.type == "cosine":
            total_steps = max(1, self.trainer.estimated_stepping_batches)
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=total_steps,
                    eta_min=float(sch_cfg.eta_min),
                ),
                "interval": "step",
            }
        else:
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=max(1, int(getattr(sch_cfg, "step_size", 10))),
                    gamma=float(getattr(sch_cfg, "gamma", 0.1)),
                ),
                "interval": "epoch",
            }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
