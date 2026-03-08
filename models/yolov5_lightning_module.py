import os
import sys
import importlib.util
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

import torch
import pytorch_lightning as pl
from tqdm import tqdm

from utils.coco_eval_utils import (
    convert_preds_to_coco,
    gather_outputs_across_processes,
    broadcast_object,
    compute_coco_metrics,
    to_cpu_device,
)
from utils.ema import EMACallback
from utils.sahi_eval import run_sahi_sliced_eval
import numpy as np

# import yolo models from ultralytics
# from ultralytics import yolov5n, yolov5s, yolov5m, yolov5l, yolov5x

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
        self.ema_validation_step_outputs = []
        self.test_step_outputs = []
        
        self.validation_step_outputs_sliced = []
        self.test_step_outputs_sliced = []
        
        if hasattr(self.config.model, 'ema') and self.config.model.ema.enabled:
            self.validation_step_outputs_ema = []
            self.test_step_outputs_ema = []
            self.validation_step_outputs_sliced_ema = []
            self.test_step_outputs_sliced_ema = []
            
        self.PALETTE = [
            (255, 64, 64), (64, 255, 64), (64, 64, 255), (255, 255, 64), (255, 64, 255),
            (64, 255, 255), (255, 128, 64), (128, 64, 255), (64, 255, 128), (255, 64, 128),
            (128, 255, 64), (64, 128, 255), (255, 128, 128), (128, 255, 128), (128, 128, 255)
        ]
        try:
            self.font = ImageFont.truetype("arial.ttf", 17)
        except IOError:
            self.font = ImageFont.load_default()
            
        self.val_viz_counter = 0
        self.test_viz_counter = 0
        self._train_image_ids_epoch = []
        self.save_hyperparameters()

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

        # Check if weights_path looks like a torch hub model name (e.g., 'yolov5s', 'yolov5m')
        # and not a file path (doesn't end in .pt)
        is_hub_model = not weights_path.endswith('.pt') and not Path(weights_path).exists()
        
        if is_hub_model:
            try:
                print(f"[INFO] YOLOv5 Loading pre-trained weights from torch.hub: {weights_path}")
                # Load model from torch hub to get weights
                # trusting repo since we are loading official yolov5
                hub_model = torch.hub.load('ultralytics/yolov5', weights_path, pretrained=True, trust_repo=True)
                
                # Extract state dict
                if hasattr(hub_model, 'model'):
                    state_dict = hub_model.model.float().state_dict()
                else:
                    state_dict = hub_model.float().state_dict()
                    
                print(f"[INFO] YOLOv5 Successfully downloaded weights for {weights_path} from torch.hub")
            except Exception as e:
                print(f"[WARNING] Failed to load from torch.hub: {e}")
                print("[INFO] Falling back to local file search...")
                state_dict = None
        else:
            state_dict = None

        if state_dict is None:
            # Fallback to local file search
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

            print(f"[INFO] YOLOv5 Loading pre-trained weights from local file: {found_path}")
            # weights_only=False is required for YOLOv5 checkpoints as they contain custom classes
            ckpt = torch.load(str(found_path), map_location="cpu", weights_only=False)
            state_dict = ckpt["model"].float().state_dict() if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        
        # Filter state_dict by shape to handle class mismatch (Detection head layers)
        # The detection head weights will have different shapes due to num_classes differences
        # (e.g. 80 classes in COCO vs 4 classes in this dataset).
        # This dynamic filtering works for all YOLOv5 variants (n, s, m, l, x) without hardcoding layer indices.
        model_state_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() 
                     if k in model_state_dict and v.shape == model_state_dict[k].shape}
        
        model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] YOLOv5 Weights loaded successfully (backbone and neck only).")

    def on_sanity_check_start(self):
        """Move model to device before sanity check."""
        self.model = self.model.to(self.device)

    def on_train_start(self):
        """Move model to device at training start."""
        self.model = self.model.to(self.device)

    def on_test_start(self):
        """
        Ensure dtype/device are compatible with the configured precision mode.
        For mixed precision, keep FP32 weights and rely on autocast.
        For true precision modes, cast model weights explicitly.
        """
        self.model = self.model.to(self.device)
        precision_mode = str(self.trainer.precision).lower()

        # Mixed precision keeps model weights in FP32.
        if precision_mode in {"16-mixed", "bf16-mixed"}:
            self.model = self.model.float()
            return

        target_dtype = None
        if precision_mode in {"16-true", "16"}:
            target_dtype = torch.float16
        elif precision_mode in {"bf16-true", "bf16"}:
            target_dtype = torch.bfloat16

        if target_dtype is not None:
            self.model = self.model.to(dtype=target_dtype)
            self.print(f"[INFO] Cast model weights to {target_dtype} for precision={self.trainer.precision}.")

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

    def _process_predictions(self, model_output, shapes, batch_image_ids):
        """Helper to process model predictions (called by both standard and EMA models)."""
        pred_list = self._non_max_suppression(
            model_output,
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

    def _log_loss_items(self, split: str, loss_items, batch_size: int):
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
            self.log(
                f"{split}/{name}",
                value,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

    def training_step(self, batch, batch_idx):
        images, targets = batch[0], batch[1]
        batch_size = int(images.shape[0]) if isinstance(images, torch.Tensor) else len(targets)
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

        self.log(
            "train/loss",
            loss,
            batch_size=batch_size,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self._log_loss_items("train", loss_items, batch_size=batch_size)
        return loss

    @torch.no_grad()
    def _run_eval_step(self, batch, split_name: str):
        images, targets, paths, shapes, batch_image_ids = batch
        batch_size = int(images.shape[0]) if isinstance(images, torch.Tensor) else len(batch_image_ids)
        images = images.to(self.device, non_blocking=True).float() / 255.0
        targets = targets.to(self.device, non_blocking=True).float()

        # Ensure model is in eval mode for inference
        self.model.eval()
        
        eval_mode = self.config.get("eval_inference", {}).get("mode", "whole")
        ret_dict = {}

        if eval_mode in ["whole", "both"]:
            with torch.no_grad():
                outputs = self.model(images)
                infer_out, train_out = self._extract_model_outputs(outputs)

            if infer_out is None and train_out is not None:
                infer_out = train_out

            if train_out is not None:
                val_loss, loss_items = self.compute_loss(train_out, targets)
                self.log(
                    f"{split_name}/loss",
                    val_loss,
                    batch_size=batch_size,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
                self._log_loss_items(split_name, loss_items, batch_size=batch_size)

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
            ret_dict["whole"] = {"predictions": result_map, "image_ids": image_ids, "paths": paths}

        if eval_mode in ["sliced", "both"]:
            sliced_batch_predictions = {}
            sliced_batch_image_ids = []
            
            for sample_idx, img_path in enumerate(paths):
                image_id = int(batch_image_ids[sample_idx])
                sliced_batch_image_ids.append(image_id)
                
                # Resolve full path from ultralytics format paths if needed
                split_name_for_path = self.config.test_name if "test" in split_name else self.config.val_name
                full_path = os.path.join(self.config.data.path, "images", split_name_for_path, os.path.basename(img_path))
                
                if not full_path or not os.path.exists(full_path):
                    self.print(f"[{split_name.upper()}] WARNING: Cannot find image {full_path} for SAHI. Falling back to whole image.")
                    if "whole" in ret_dict:
                         sliced_batch_predictions[image_id] = ret_dict["whole"]["predictions"][image_id]
                    continue
                
                def predict_fn(image_np):
                    from data.yolov5_data_module import _letterbox
                    # Ensure dimensions match what the model expects
                    im, _, _ = _letterbox(image_np, int(self.config.model.input_size))
                    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                    im = np.ascontiguousarray(im)
                    
                    im_tensor = torch.from_numpy(im).to(self.device).float() / 255.0
                    if len(im_tensor.shape) == 3:
                        im_tensor = im_tensor[None]
                    
                    if next(self.model.parameters()).dtype == torch.float16:
                        im_tensor = im_tensor.half()
                    elif next(self.model.parameters()).dtype == torch.bfloat16:
                        im_tensor = im_tensor.bfloat16()
                    
                    with torch.no_grad():
                        out = self.model(im_tensor)
                        infer_out, _ = self._extract_model_outputs(out)
                        
                    conf_thres = float(self.config.model.detection_threshold)
                    iou_thres = float(self.config.model.yolov5.iou_threshold)
                    max_det = int(self.config.model.max_detections)
                    
                    preds = self._non_max_suppression(
                        infer_out, conf_thres=conf_thres, iou_thres=iou_thres, 
                        max_det=max_det
                    )[0]
                    
                    boxes = torch.empty((0, 4), device=self.device)
                    scores = torch.empty((0,), device=self.device)
                    labels = torch.empty((0,), dtype=torch.long, device=self.device)
                    
                    if preds is not None and len(preds):
                        # Use _letterbox ratio/pad to undo letterbox correctly
                        _, ratio, (dw, dh) = _letterbox(image_np, int(self.config.model.input_size))
                        h0, w0 = image_np.shape[:2]
                        shape_meta = ((h0, w0), ((ratio, ratio), (dw, dh)))
                        
                        preds[:, :4] = self._undo_letterbox(preds[:, :4], shape_meta)
                        boxes = preds[:, :4]
                        scores = preds[:, 4]
                        mapped_labels = self._map_label_ids(preds[:, 5].to(torch.int64))
                        labels = mapped_labels
                    
                    return {
                        "boxes": to_cpu_device(boxes),
                        "scores": to_cpu_device(scores),
                        "labels": to_cpu_device(labels)
                    }

                from PIL import Image
                img_pil = Image.open(full_path).convert("RGB")
                sahi_cfg = self.config.get("eval_inference", {}).get("sahi", {})
                input_size = int(self.config.model.input_size)
                
                preds = run_sahi_sliced_eval(img_pil, predict_fn, sahi_cfg, input_size)
                sliced_batch_predictions[image_id] = preds
                
            ret_dict["sliced"] = {"predictions": sliced_batch_predictions, "image_ids": sliced_batch_image_ids, "paths": paths}

        return ret_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        output = self._run_eval_step(batch, split_name="val")
        if "whole" in output:
            self.validation_step_outputs.append(output["whole"])
        if "sliced" in output:
            self.validation_step_outputs_sliced.append(output["sliced"])
        
        # EMA validation
        ema_callback = next((cb for cb in self.trainer.callbacks if isinstance(cb, EMACallback)), None)
        if ema_callback and ema_callback.ema_model:
            original_model = self.model
            self.model = ema_callback.ema_model.module
            ema_output = self._run_eval_step(batch, split_name="val")
            self.model = original_model
            
            if "whole" in ema_output:
                self.validation_step_outputs_ema.append(ema_output["whole"])
            if "sliced" in ema_output:
                self.validation_step_outputs_sliced_ema.append(ema_output["sliced"])
            
        # Draw Visualizations occasionally
        if (self.current_epoch + 1) % max(1, self.config.checkpointing.visualize_every_n_epochs) == 0 and \
           self.trainer.is_global_zero and \
           (self.val_viz_counter < self.config.checkpointing.visualize_samples or self.config.checkpointing.visualize_samples == -1):
            
            visualizer_out = ema_output["whole"] if (ema_callback and ema_callback.ema_model and "whole" in ema_output) else output.get("whole", output.get("sliced", {}))
            if "predictions" in visualizer_out:
                save_dir = os.path.join(
                    self.config.checkpointing.save_dir,
                    self.config.checkpointing.visualization_dir, 
                    f"epoch_{(self.current_epoch+1):03d}", 
                    "val"
                )
                self.val_viz_counter = self._visualize_batch(
                    save_dir, 
                    visualizer_out["predictions"], 
                    visualizer_out.get("paths", batch[2]),
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
        def _compute_and_log(outputs_list, prefix_name, log_prefix):
            all_outputs = gather_outputs_across_processes(outputs_list)
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
                        prefix=f"{prefix_name} performance"
                    )
                else:
                    metrics = {"map": 0.0, "map_50": 0.0, "map_75": 0.0}

            metrics = broadcast_object(metrics, src=0)
            for key, value in metrics.items():
                self.log(f"{log_prefix}/{key}", value, prog_bar=(key in {"map", "map_50"}), sync_dist=True)
                if key == "map":
                    self.log(f"{log_prefix}_map", value, prog_bar=False, sync_dist=True)

            outputs_list.clear()

        if self.validation_step_outputs:
            _compute_and_log(self.validation_step_outputs, "Val", "val")
            
        if self.validation_step_outputs_sliced:
            _compute_and_log(self.validation_step_outputs_sliced, "Val Sliced", "val_sliced")

        if hasattr(self, "validation_step_outputs_ema") and self.validation_step_outputs_ema:
            _compute_and_log(self.validation_step_outputs_ema, "Val EMA", "val_ema")
            
        if hasattr(self, "validation_step_outputs_sliced_ema") and self.validation_step_outputs_sliced_ema:
            _compute_and_log(self.validation_step_outputs_sliced_ema, "Val Sliced EMA", "val_sliced_ema")

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        output = self._run_eval_step(batch, split_name="test")
        if "whole" in output:
            self.test_step_outputs.append(output["whole"])
        if "sliced" in output:
            self.test_step_outputs_sliced.append(output["sliced"])

        # EMA test
        ema_callback = next((cb for cb in self.trainer.callbacks if isinstance(cb, EMACallback)), None)
        if ema_callback and ema_callback.ema_model:
            original_model = self.model
            self.model = ema_callback.ema_model.module
            ema_output = self._run_eval_step(batch, split_name="test")
            self.model = original_model
            
            if "whole" in ema_output:
                self.test_step_outputs_ema.append(ema_output["whole"])
            if "sliced" in ema_output:
                self.test_step_outputs_sliced_ema.append(ema_output["sliced"])
            
        if self.trainer.is_global_zero and \
           (self.test_viz_counter < self.config.checkpointing.visualize_samples or self.config.checkpointing.visualize_samples == -1):
            
            visualizer_out = ema_output["whole"] if (ema_callback and ema_callback.ema_model and "whole" in ema_output) else output.get("whole", output.get("sliced", {}))
            if "predictions" in visualizer_out:
                save_dir = os.path.join(
                    self.config.checkpointing.save_dir,
                    self.config.checkpointing.visualization_dir, 
                    "test"
                )
                self.test_viz_counter = self._visualize_batch(
                    save_dir, 
                    visualizer_out["predictions"], 
                    visualizer_out.get("paths", batch[2]),
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
        def _compute_and_log(outputs_list, prefix_name, log_prefix):
            all_outputs = gather_outputs_across_processes(outputs_list)
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
                        prefix=f"{prefix_name} performance"
                    )
                else:
                    metrics = {"map": 0.0, "map_50": 0.0, "map_75": 0.0}

            metrics = broadcast_object(metrics, src=0)
            for key, value in metrics.items():
                self.log(f"{log_prefix}/{key}", value, prog_bar=(key in {"map", "map_50"}), sync_dist=True)

            outputs_list.clear()

        if self.test_step_outputs:
            _compute_and_log(self.test_step_outputs, "Test", "test")
            
        if self.test_step_outputs_sliced:
            _compute_and_log(self.test_step_outputs_sliced, "Test Sliced", "test_sliced")

        if hasattr(self, "test_step_outputs_ema") and self.test_step_outputs_ema:
            _compute_and_log(self.test_step_outputs_ema, "Test EMA", "test_ema")
            
        if hasattr(self, "test_step_outputs_sliced_ema") and self.test_step_outputs_sliced_ema:
            _compute_and_log(self.test_step_outputs_sliced_ema, "Test Sliced EMA", "test_sliced_ema")

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
        
        if counter == 0:
            self.print(f"[VIZ] Saving visualizations to: {save_dir}")
            self.print(f"[VIZ] Max samples: {max_samples}")
            if max_samples == -1 or max_samples == float("inf"):
                self.print(
                    f"[VIZ] WARNING: Unlimited visualization enabled for {split}. "
                    "This can be very slow on large datasets."
                )

        coco_gt = self.test_coco_gt if split == "test" else self.val_coco_gt
        
        # Use draw_threshold for visualization to show only "clean" detections.
        # detection_threshold is still used for the actual metrics/COCO eval.
        viz_threshold = float(self.config.model.draw_threshold)
        
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
            gt_labels = []
            if coco_gt:
                try:
                    gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=image_id))
                    gt_boxes = []
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
            pred_class_names = []
            if image_id in predictions_map:
                preds = predictions_map[image_id]
                n_preds = len(preds['boxes'])
                if n_preds > 0:
                    max_score = float(preds['scores'].max()) if n_preds > 0 else 0.0
                    above_thresh = int((preds['scores'] >= viz_threshold).sum())
                    # self.print(f"[VIZ] image_id={image_id}: {n_preds} preds, max_score={max_score:.4f}, above_thresh({viz_threshold})={above_thresh}")
                    
                    # Collect names for counts (filtered by threshold)
                    valid_indices = preds['scores'] >= viz_threshold
                    valid_labels = preds['labels'][valid_indices]
                    label_map = self.config.model.label_map
                    for l in valid_labels:
                        l_item = l.item() if torch.is_tensor(l) else int(l)
                        name = label_map.get(int(l_item)) or label_map.get(str(l_item)) or str(l_item)
                        pred_class_names.append(name)
                        
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
            
            # --- 3. Visualization Counts & Filename ---
            from collections import Counter
            label_map = self.config.model.label_map
            
            # GT Counts
            gt_counts = Counter([label_map.get(int(l)) or label_map.get(str(l)) or str(l) for l in gt_labels])
            
            # Pred Counts
            pred_counts = Counter(pred_class_names)
            
            # Draw Counts on Image (Top Right)
            draw = ImageDraw.Draw(image)
            text_y = 10
            line_height = 24
            
            all_classes = set(gt_counts.keys()) | set(pred_counts.keys())
            
            for cls_name in sorted(all_classes):
                # Parts to draw: (Text, Color)
                parts = [
                    (f"{cls_name}: ", "white"),
                    (f"{pred_counts[cls_name]}", "red"),
                    ("/", "white"),
                    (f"{gt_counts[cls_name]}", "green")
                ]
                
                # Calculate total width to align right
                total_width = 0
                for text, _ in parts:
                    bbox = draw.textbbox((0, 0), text, font=self.font)
                    total_width += bbox[2] - bbox[0]
                
                current_x = image.width - total_width - 10
                
                for text, color in parts:
                    # Draw shadow
                    draw.text((current_x + 1, text_y + 1), text, fill="black", font=self.font)
                    # Draw text
                    draw.text((current_x, text_y), text, fill=color, font=self.font)
                    
                    # Advance cursor
                    bbox = draw.textbbox((0, 0), text, font=self.font)
                    current_x += bbox[2] - bbox[0]
                
                text_y += line_height

            # Construct new filename
            detected_classes = sorted(list(set(pred_class_names)))
            if detected_classes:
                class_str = "_".join(detected_classes)
                prefix = f"image_{class_str}_"
            else:
                prefix = "image_no_detections_"
            
            original_filename = os.path.basename(path)
            new_filename = f"{prefix}{original_filename}"
            new_filename = new_filename.replace("image_image_", "image_")
            
            save_path = os.path.join(save_dir, new_filename)
            image.save(save_path)
            counter += 1
            # if counter % 500 == 0:
            #     self.print(f"[VIZ] {split.upper()} progress: saved {counter} images...")
            
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

    def lr_scheduler_step(self, scheduler, metric):
        """
        Keep scheduler progression aligned with real optimizer updates.
        This avoids stepping LR when AMP/overflow skips optimizer.step().
        """
        optimizer = getattr(scheduler, "optimizer", None)
        optimizer_has_stepped = optimizer is None or getattr(optimizer, "_step_count", 0) > 0
        if not optimizer_has_stepped:
            return

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if metric is not None:
                scheduler.step(metric)
            return

        scheduler.step()
