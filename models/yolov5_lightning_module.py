import os
import sys
import importlib.util
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

import torch
import pytorch_lightning as pl

from utils.coco_eval_utils import (
    convert_preds_to_coco,
    gather_outputs_across_processes,
    broadcast_object,
    compute_coco_metrics,
)
from utils.ema import EMACallback


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
        # Store ComputeLoss class for later instantiation on device
        self._ComputeLossClass = ComputeLoss
        self._compute_loss = None
        self._compute_loss_device = None
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        if hasattr(self.config.model, 'ema') and self.config.model.ema.enabled:
            self.validation_step_outputs_ema = []
            self.test_step_outputs_ema = []
            
        self.PALETTE = [
            (255, 64, 64), (64, 255, 64), (64, 64, 255), (255, 255, 64), (255, 64, 255),
            (64, 255, 255), (255, 128, 64), (128, 64, 255), (64, 255, 128), (255, 64, 128),
            (128, 255, 64), (64, 128, 255), (255, 128, 128), (128, 255, 128), (128, 128, 255)
        ]
        try:
            self.font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            self.font = ImageFont.load_default()
            
        self.val_viz_counter = 0
        self.test_viz_counter = 0
        self._train_image_ids_epoch = []

    @property
    def compute_loss(self):
        """Lazy initialization of compute_loss, recreating if model device changes."""
        # Get current model device
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = None

        # Recreate if device changed or not initialized yet
        if self._compute_loss is None or self._compute_loss_device != model_device:
            self._compute_loss = self._ComputeLossClass(self.model)
            self._compute_loss_device = model_device

        return self._compute_loss

    @compute_loss.setter
    def compute_loss(self, value):
        self._compute_loss = value
        self._compute_loss_device = None

    def _load_weights_if_available(self, model, weights_path: str):
        if not weights_path:
            return
        w = Path(weights_path).expanduser()
        
        # Try finding the file in multiple common locations
        search_paths = [
            w,                                    # As-is (usually relative to CWD)
            Path(self.yolo_repo_path) / w,        # Relative to YOLOv5 repo
            Path(self.yolo_repo_path).parent / w, # Relative to repo parent (our project root)
        ]
        
        found_path = None
        for p in search_paths:
            if p.exists() and p.is_file():
                found_path = p
                break
                
        if found_path is None:
            print(f"[WARNING] YOLOv5 weights not found. Searched: {[str(p) for p in search_paths]}")
            return
            
        print(f"[INFO] YOLOv5 Loading pre-trained weights from: {found_path}")
        # weights_only=False is required for YOLOv5 checkpoints as they contain custom classes
        ckpt = torch.load(str(found_path), map_location="cpu", weights_only=False)

        state_dict = ckpt["model"].float().state_dict() if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        
        # Filter state_dict to handle class mismatch (Detection head layers)
        # In YOLOv5s, the detection head is typically layer 24.
        exclude = ["model.24.m.0.weight", "model.24.m.0.bias", 
                   "model.24.m.1.weight", "model.24.m.1.bias", 
                   "model.24.m.2.weight", "model.24.m.2.bias"]
        
        state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and k not in exclude}
        
        model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] YOLOv5 Weights loaded successfully (backbone and neck only).")

    def on_sanity_check_start(self):
        """Move model to device before sanity check."""
        self.model = self.model.to(self.device)

    def on_train_start(self):
        """Move model to device at training start."""
        self.model = self.model.to(self.device)

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
        # Now that both GT and Predictions use model IDs (0, 1, 2...)
        # we can return the labels as is.
        yolo_labels = yolo_label_tensor.tolist()
        mapped = [int(x) for x in yolo_labels]
        
        # if self.config.debug and self.trainer.is_global_zero and len(yolo_labels) > 0:
        #      # Look for this in the logs to verify both match (e.g. [0, 0] -> [0, 0])
        #      self.print(f"[DEBUG] Evaluation labels (model idx): {yolo_labels[:5]}")

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
        targets = targets.to(self.device, non_blocking=True).float()

        # Track image IDs for debug overfitting verification
        if self.config.debug and len(batch) >= 5:
            batch_image_ids = batch[4]
            self._train_image_ids_epoch.extend([int(x) for x in batch_image_ids])

        # Ensure model is in training mode
        self.model.train()
        
        outputs = self.model(images)
        _, train_out = self._extract_model_outputs(outputs)
        if train_out is None:
            train_out = outputs

        loss, loss_items = self.compute_loss(train_out, targets)

        # Diagnostic: what are we actually training on?
        # if self.config.debug and self.trainer.is_global_zero and batch_idx % 5 == 0:
        #     self.print(f"[DEBUG] Training Step {batch_idx}: batch={images.shape}, targets={targets.shape}")
        #     if targets.numel() > 0:
        #         # Target format: [batch_idx, cls, cx, cy, w, h]
        #         unique_ids = torch.unique(targets[:, 1]).tolist()
        #         id2name = self.config.model.label_map
        #         # Ensure we handle mapping regardless of whether keys are int or str
        #         names = [id2name.get(int(i)) or id2name.get(str(int(i))) or f"ID_{int(i)}" for i in unique_ids]
                
        #         self.print(f"[DEBUG] Target IDs in Batch: {unique_ids}")
        #         self.print(f"[DEBUG] Target Names in Batch: {names}")
                
        #         counts = torch.unique(targets[:, 1], return_counts=True)[1].tolist()
        #         self.print(f"[DEBUG] Target Distribution: {dict(zip(names, counts))}")
            
        #     # Anchor Diagnostic
        #     if hasattr(self.model, 'model') and isinstance(self.model.model[-1], torch.nn.Module):
        #         m = self.model.model[-1]
        #         if hasattr(m, 'anchors'):
        #             self.print(f"[DEBUG] Model Anchors (pixels/stride): {m.anchors}")
            
        #     # Loss items: obj_loss, box_loss, cls_loss? (order depends on YOLOv5 version)
        #     # Standard YOLOv5 ComputeLoss returns [box, obj, cls]
        #     if isinstance(loss_items, torch.Tensor):
        #         l_vals = loss_items.tolist()
        #         self.print(f"[DEBUG] Loss Items ([box, obj, cls]): {['%.4f' % v for v in l_vals]}")
        #         if l_vals[0] < 1e-4:  # box loss is near zero
        #             self.print("[WARNING] BOX LOSS IS ZERO! This means NO ANCHORS MATCHED the ground truth.")

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self._log_loss_items("train", loss_items)
        return loss

    @torch.no_grad()
    def _run_eval_step(self, batch, split_name: str):
        images, targets, paths, shapes, batch_image_ids = batch
        images = images.to(self.device, non_blocking=True).float() / 255.0
        targets = targets.to(self.device, non_blocking=True).float()

        # Ensure model is in eval mode for inference
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(images)
            infer_out, train_out = self._extract_model_outputs(outputs)

        # Diagnostic: what did the model actually return?
        # if self.config.debug and self.trainer.is_global_zero:
        #     self.print(f"[DEBUG] Model forward mode={self.model.training}, "
        #               f"infer_out type={type(infer_out)}, "
        #               f"train_out type={type(train_out)}")
        #     if torch.is_tensor(infer_out):
        #         # Shape: [batch, num_anchors, 5 + nc]
        #         max_obj = float(infer_out[..., 4].sigmoid().max())
        #         self.print(f"[DEBUG] Raw max objectness (sigmoid): {max_obj:.6f}")
        #         if infer_out.ndim == 3 and infer_out.shape[-1] > 5:
        #             max_cls = float(infer_out[..., 5:].sigmoid().max())
        #             self.print(f"[DEBUG] Raw max class score (sigmoid): {max_cls:.6f}")

        if infer_out is None and train_out is not None:
            # Fallback for models that might not have custom multi-head forward
            infer_out = train_out

        if train_out is not None:
            # Note: ComputeLoss expects the raw training output (usually a list of tensors)
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
            
            # DIAGNOSTIC: Show GT info for the first sample in the batch regardless of predictions
            # if self.config.debug and self.trainer.is_global_zero and sample_idx == 0:
            #     coco_gt = self.val_coco_gt if split_name == "val" else self.test_coco_gt
            #     if coco_gt is not None:
            #         gt_ann_ids = coco_gt.getAnnIds(imgIds=[image_id])
            #         gt_anns = coco_gt.loadAnns(gt_ann_ids)
            #         if len(gt_anns) > 0:
            #             self.print(f"[DEBUG] Image {image_id} COMPARISON (Split: {split_name}):")
            #             self.print(f"  GT (first 2 boxes xywh): {[a['bbox'] for a in gt_anns[:2]]}")
            #             self.print(f"  GT (first 2 categories): {[a['category_id'] for a in gt_anns[:2]]}")

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

            # DIAGNOSTIC (continued): Add prediction info if it exists
            # if self.config.debug and self.trainer.is_global_zero and sample_idx == 0:
            #     # Convert first pred to xywh for direct comparison
            #     p_box = predn[0, :4].tolist()
            #     p_xywh = [p_box[0], p_box[1], p_box[2]-p_box[0], p_box[3]-p_box[1]]
            #     self.print(f"  PRED (first 1 box xywh): {[round(x, 2) for x in p_xywh]}")
            #     self.print(f"  PRED Category: {int(mapped_labels[0])}")

            result_map[int(image_id)] = {
                "boxes": predn[:, :4].detach().cpu(),
                "scores": predn[:, 4].detach().cpu(),
                "labels": mapped_labels.detach().cpu(),
            }

        # Diagnostic: Print info about predictions in this batch
        # if self.config.debug and self.trainer.is_global_zero:
        #     total_preds = sum(len(p["boxes"]) for p in result_map.values())
        #     if total_preds > 0:
        #         all_scores = torch.cat([p["scores"] for p in result_map.values() if len(p["boxes"]) > 0])
        #         self.print(f"[DEBUG] Batch {split_name} predictions: {total_preds} boxes, "
        #                   f"score range: [{all_scores.min():.4f}, {all_scores.max():.4f}], "
        #                   f"mean: {all_scores.mean():.4f}")
        #     else:
        #         self.print(f"[DEBUG] Batch {split_name}: NO predictions above detection_threshold ({self.config.model.detection_threshold})")
                
        #         # Extreme diagnostic: try NMS with 0.001 threshold to see if ANY boxes exist
        #         peek_preds = self._non_max_suppression(
        #             infer_out,
        #             conf_thres=0.001,
        #             iou_thres=0.45,
        #             max_det=10,
        #         )
        #         peek_count = sum(len(x) for x in peek_preds if x is not None)
        #         if peek_count > 0:
        #             peek_max = max(x[:, 4].max().item() for x in peek_preds if x is not None and len(x) > 0)
        #             self.print(f"[DEBUG] PEEK at 0.001 threshold: found {peek_count} boxes (max score: {peek_max:.6f})")
        #             # Log first peeked box in XYWH for direct comparison with GT
        #             for idx, x in enumerate(peek_preds):
        #                 if x is not None and len(x) > 0:
        #                     p_box = x[0, :4].clone()
        #                     # Undo letterbox for the peeked box
        #                     p_box = self._undo_letterbox(p_box.unsqueeze(0), shapes[idx])[0]
        #                     p_xywh = [p_box[0], p_box[1], p_box[2]-p_box[0], p_box[3]-p_box[1]]
        #                     self.print(f"  PEEK PRED (Image {batch_image_ids[idx]} best box xywh): {[round(float(v), 1) for v in p_xywh]}")
        #                     self.print(f"  PEEK PRED Category: {int(x[0, 5])} (score: {x[0, 4]:.4f})")
        #                     break
        #         else:
        #             self.print(f"[DEBUG] PEEK at 0.001 threshold: still NOTHING found.")

        return {"predictions": result_map, "image_ids": image_ids, "paths": paths}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        output = self._run_eval_step(batch, split_name="val")
        self.validation_step_outputs.append(output)
        
        # EMA validation
        ema_callback = next((cb for cb in self.trainer.callbacks if isinstance(cb, EMACallback)), None)
        if ema_callback and ema_callback.ema_model:
            # Temporarily swap model explicitly for EMA eval using helper
            original_model = self.model
            self.model = ema_callback.ema_model.module
            ema_output = self._run_eval_step(batch, split_name="val")
            self.model = original_model
            
            self.validation_step_outputs_ema.append(ema_output)
            
        # Draw Visualizations occasionally
        if (self.current_epoch + 1) % self.config.checkpointing.visualize_every_n_epochs == 0 and \
           self.trainer.is_global_zero and \
           (self.val_viz_counter < self.config.checkpointing.visualize_samples or self.config.checkpointing.visualize_samples == -1):
            
            visualizer_out = ema_output if (ema_callback and ema_callback.ema_model) else output
            save_dir = os.path.join(
                self.config.checkpointing.save_dir,
                self.config.checkpointing.visualization_dir, 
                f"epoch_{(self.current_epoch+1):03d}", 
                "val"
            )
            self.val_viz_counter = self._visualize_batch(
                save_dir, 
                visualizer_out["predictions"], 
                visualizer_out["paths"],
                visualizer_out["image_ids"],
                self.val_viz_counter,
                split="val"
            )

        return output

    def on_validation_epoch_start(self):
        """Reset validation visualization counter."""
        self.val_viz_counter = 0
        if self.config.checkpointing.visualize_samples == -1:
            self.config.checkpointing.visualize_samples = float('inf')

    def on_validation_epoch_end(self):
        # Debug: print image IDs used during training and validation
        if self.config.debug and self.trainer.is_global_zero:
            train_ids = sorted(set(self._train_image_ids_epoch))
            self.print(f"\n{'='*60}")
            self.print(f"[DEBUG] Epoch {self.current_epoch} - TRAIN image IDs ({len(train_ids)}): {train_ids}")
            val_ids = sorted(set(
                img_id for batch_out in self.validation_step_outputs for img_id in batch_out["image_ids"]
            ))
            self.print(f"[DEBUG] Epoch {self.current_epoch} - VAL image IDs ({len(val_ids)}): {val_ids}")
            overlap = set(train_ids) & set(val_ids)
            self.print(f"[DEBUG] Overlap (train ∩ val): {len(overlap)} IDs: {sorted(overlap)}")
            self.print(f"{'='*60}\n")
            self._train_image_ids_epoch = []  # Reset for next epoch

        all_outputs = gather_outputs_across_processes(self.validation_step_outputs)
        metrics = {}
        if self.trainer.is_global_zero:
            predictions = []
            image_ids = []
            for batch_out in all_outputs:
                predictions.extend(convert_preds_to_coco(batch_out["predictions"]))
                image_ids.extend(batch_out["image_ids"])

            # if self.config.debug and len(predictions) > 0:
            #     self.print(f"[DEBUG] Final COCO preds for eval (first 3): {predictions[:3]}")

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
        
        # Compute EMA metrics
        all_ema_outputs = gather_outputs_across_processes(getattr(self, 'validation_step_outputs_ema', []))
        if hasattr(self, 'validation_step_outputs_ema'):
            ema_metrics = {}
            if self.trainer.is_global_zero:
                ema_predictions = []
                ema_image_ids = []
                for batch_out in all_ema_outputs:
                    ema_predictions.extend(convert_preds_to_coco(batch_out["predictions"]))
                    ema_image_ids.extend(batch_out["image_ids"])
                    
                if len(ema_predictions) > 0:
                    ema_metrics = compute_coco_metrics(
                        coco_gt=self.val_coco_gt,
                        predictions=ema_predictions,
                        image_ids=list(set(ema_image_ids)),
                        max_detections=int(self.config.model.max_detections),
                        label_map=self.config.model.label_map,
                    )
                else:
                    ema_metrics = {"map": 0.0, "map_50": 0.0, "map_75": 0.0}
                    
            ema_metrics = broadcast_object(ema_metrics, src=0)
            for key, value in ema_metrics.items():
                self.log(f"val/{key}_ema", value, prog_bar=True, sync_dist=True)
                if key == "map":
                    self.log("val_map_ema", value, prog_bar=False, sync_dist=True)
                    
            self.validation_step_outputs_ema.clear()
            
        self.validation_step_outputs.clear()

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        output = self._run_eval_step(batch, split_name="test")
        self.test_step_outputs.append(output)

        # EMA test
        ema_callback = next((cb for cb in self.trainer.callbacks if isinstance(cb, EMACallback)), None)
        if ema_callback and ema_callback.ema_model:
            original_model = self.model
            self.model = ema_callback.ema_model.module
            ema_output = self._run_eval_step(batch, split_name="test")
            self.model = original_model
            self.test_step_outputs_ema.append(ema_output)
            
        if self.trainer.is_global_zero and \
           (self.test_viz_counter < self.config.checkpointing.visualize_samples or self.config.checkpointing.visualize_samples == -1):
            
            visualizer_out = ema_output if (ema_callback and ema_callback.ema_model) else output
            save_dir = os.path.join(
                self.config.checkpointing.save_dir,
                self.config.checkpointing.visualization_dir, 
                "test"
            )
            self.test_viz_counter = self._visualize_batch(
                save_dir, 
                visualizer_out["predictions"], 
                visualizer_out["paths"],
                visualizer_out["image_ids"],
                self.test_viz_counter,
                split="test"
            )

        return output

    def on_test_epoch_start(self):
        """Reset validation visualization counter."""
        self.test_viz_counter = 0
        if self.config.checkpointing.visualize_samples == -1:
            self.config.checkpointing.visualize_samples = float('inf')

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
        
        # Compute EMA metrics for Test
        all_ema_outputs = gather_outputs_across_processes(getattr(self, 'test_step_outputs_ema', []))
        if hasattr(self, 'test_step_outputs_ema'):
            ema_metrics = {}
            if self.trainer.is_global_zero:
                ema_predictions = []
                ema_image_ids = []
                for batch_out in all_ema_outputs:
                    ema_predictions.extend(convert_preds_to_coco(batch_out["predictions"]))
                    ema_image_ids.extend(batch_out["image_ids"])
                    
                if len(ema_predictions) > 0:
                    ema_metrics = compute_coco_metrics(
                        coco_gt=self.test_coco_gt,
                        predictions=ema_predictions,
                        image_ids=list(set(ema_image_ids)),
                        max_detections=int(self.config.model.max_detections),
                        label_map=self.config.model.label_map,
                    )
                else:
                    ema_metrics = {"map": 0.0, "map_50": 0.0, "map_75": 0.0}
                    
            ema_metrics = broadcast_object(ema_metrics, src=0)
            for key, value in ema_metrics.items():
                self.log(f"test/{key}_ema", value, prog_bar=True, sync_dist=True)
                    
            self.test_step_outputs_ema.clear()
            
        self.test_step_outputs.clear()

    def draw_boxes(self, image, boxes, labels, scores=None, id2label=None, color_override=None, label_prefix="", threshold_override=None):
        """Draws bounding boxes on a PIL image."""
        draw = ImageDraw.Draw(image)
        threshold = threshold_override if threshold_override is not None else self.config.model.draw_threshold
        
        if id2label is None:
            id2label = self.config.model.label_map

        for i in range(len(boxes)):
            box = boxes[i]
            label = labels[i]
            score = scores[i] if scores is not None else 1.0
            
            if score < threshold:
                continue
            
            if torch.is_tensor(box):
                box = box.tolist()
            
            label_id = label.item() if torch.is_tensor(label) else int(label)
            
            if color_override:
                color = color_override
            else:
                color = self.PALETTE[label_id % len(self.PALETTE)]
            
            draw.rectangle(box, outline=color, width=3)
            
            class_name = id2label.get(label_id) or id2label.get(str(label_id)) or f"class_{label_id}"
            label_text = f"{label_prefix}{class_name}"
            if scores is not None:
                label_text += f": {score:.2f}"
                
            text_box = draw.textbbox((box[0], box[1]), label_text, font=self.font)
            draw.rectangle(text_box, fill=color)
            draw.text((box[0], box[1]), label_text, fill="white", font=self.font)
            
        return image

    def _visualize_batch(self, save_dir, predictions_map, paths, image_ids, counter, split="val"):
        """Saves visualizations for a batch showing both GT and Predictions."""
        os.makedirs(save_dir, exist_ok=True)
        id2label = self.config.model.label_map
        max_samples = self.config.checkpointing.visualize_samples
        
        coco_gt = self.test_coco_gt if split == "test" else self.val_coco_gt
        
        # Use detection_threshold for drawing predictions (not draw_threshold)
        # since predictions have already been NMS-filtered at detection_threshold
        viz_threshold = float(self.config.model.detection_threshold)
        
        for i, (path, img_id) in enumerate(zip(paths, image_ids)):
            if counter >= max_samples and max_samples != -1:
                break
                
            image_id = int(img_id)
            
            # Resolve full path: paths from dataloader are just filenames
            split_name = self.config.test_name if split == "test" else self.config.val_name
            full_path = os.path.join(self.config.data.path, "images", split_name, os.path.basename(path))
            
            try:
                image_arr = cv2.imread(full_path)
                image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image_arr)
            except Exception as e:
                self.print(f"[WARNING] Could not load {path}. Skipping viz.")
                continue
                
            # --- 1. Scaled Ground Truth Boxes ---
            if coco_gt:
                try:
                    gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=image_id))
                    gt_boxes = []
                    gt_labels = []
                    for ann in gt_anns:
                        x, y, w, h = ann['bbox'] # xywh
                        gt_boxes.append([x, y, x + w, y + h]) # xyxy
                        gt_labels.append(ann['category_id'])

                    image = self.draw_boxes(
                        image, 
                        gt_boxes, 
                        gt_labels, 
                        scores=None, 
                        id2label=self.config.model.label_map,
                        color_override=(0, 255, 0), # Green
                        label_prefix=""
                    )
                except Exception:
                    pass
            
            # --- 2. Prediction Boxes ---
            if image_id in predictions_map:
                preds = predictions_map[image_id]
                n_preds = len(preds['boxes'])
                if n_preds > 0:
                    max_score = float(preds['scores'].max()) if n_preds > 0 else 0.0
                    above_thresh = int((preds['scores'] >= viz_threshold).sum())
                    self.print(f"[VIZ] image_id={image_id}: {n_preds} preds, max_score={max_score:.4f}, above_thresh({viz_threshold})={above_thresh}")
                image = self.draw_boxes(
                    image, 
                    preds['boxes'], 
                    preds['labels'], 
                    preds['scores'], 
                    id2label=self.config.model.label_map,
                    color_override=(255, 0, 0), # Red
                    label_prefix="",
                    threshold_override=viz_threshold,
                )
            
            filename = os.path.basename(path)
            save_path = os.path.join(save_dir, filename)
            image.save(save_path)
            counter += 1
            
        return counter
        
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
        # Try to use YOLOv5-specific optimizer settings first, fall back to global
        yolo_cfg = getattr(self.config.model, "yolov5", None)
        if yolo_cfg and hasattr(yolo_cfg, "optimizer"):
            opt_cfg = yolo_cfg.optimizer
            self.print(f"🚀 Using YOLOv5-specific optimizer settings: {opt_cfg.type}, lr={opt_cfg.lr}")
        else:
            opt_cfg = self.config.optimizer.optimizer
            self.print(f"🚀 Using global optimizer settings: lr={opt_cfg.lr}")
            
        sch_cfg = self.config.scheduler

        params = [p for p in self.model.parameters() if p.requires_grad]

        if getattr(opt_cfg, "type", "adamw").lower() == "sgd" or ("momentum" in opt_cfg and "nesterov" in opt_cfg):
            optimizer = torch.optim.SGD(
                params,
                lr=float(opt_cfg.lr),
                momentum=float(opt_cfg.get("momentum", 0.937)),
                weight_decay=float(opt_cfg.get("weight_decay", 0.0005)),
                nesterov=bool(opt_cfg.get("nesterov", True)),
            )
        else:
            optimizer = torch.optim.AdamW(
                params,
                lr=float(opt_cfg.lr),
                weight_decay=float(opt_cfg.get("weight_decay", 1e-6)),
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
