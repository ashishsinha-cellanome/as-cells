import os
import torch
import pytorch_lightning as pl
import pycocotools.mask as mask_utils
import numpy as np

from utils.coco_eval_utils import convert_preds_to_coco, compute_coco_metrics, gather_outputs_across_processes, to_cpu_device

class DetailedCocoEvalCallback(pl.Callback):
    """
    Computes class-wise YOLO-style COCO metrics (P, R, mAP@.5) as an additional callback.
    It does not override or interfere with the default callback or visualizations.
    It logs two tables: one for bbox (Detection) and one for segm (Segmentation).
    """
    def __init__(self):
        super().__init__()
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.val_coco_gt = None
        self.test_coco_gt = None
        self.label_map = None

    def _ensure_metadata(self, trainer, pl_module):
        dm = getattr(trainer, "datamodule", None)
        if self.val_coco_gt is None and dm:
            self.val_coco_gt = getattr(dm, "val_coco_gt", None)
        if self.test_coco_gt is None and dm:
            self.test_coco_gt = getattr(dm, "test_coco_gt", None)
            
        if self.label_map is None:
            if hasattr(pl_module, "config") and hasattr(pl_module.config.model, "label_map"):
                self.label_map = pl_module.config.model.label_map
            elif dm and hasattr(dm, "class_names"):
                self.label_map = {i: name for i, name in enumerate(dm.class_names)}

    def on_validation_epoch_start(self, trainer, pl_module):
        self.validation_step_outputs.clear()
        self._ensure_metadata(trainer, pl_module)

    def on_test_epoch_start(self, trainer, pl_module):
        self.test_step_outputs.clear()
        self._ensure_metadata(trainer, pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._accumulate_batch(outputs, self.validation_step_outputs)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._accumulate_batch(outputs, self.test_step_outputs)

    def _accumulate_batch(self, outputs, storage_list):
        if not outputs or "predictions" not in outputs or "image_ids" not in outputs:
            return
            
        # The lightning module's validation_step already returns a dict of {image_id: pred_dict}
        # and has already converted masks to RLE format.
        predictions = outputs["predictions"]
        image_ids = outputs["image_ids"]
        
        storage_list.append({
            "predictions": predictions,
            "image_ids": image_ids
        })

    def on_validation_epoch_end(self, trainer, pl_module):
        self._compute_and_log(trainer, pl_module, self.validation_step_outputs, self.val_coco_gt, "val")

    def on_test_epoch_end(self, trainer, pl_module):
        self._compute_and_log(trainer, pl_module, self.test_step_outputs, self.test_coco_gt, "test")

    def _compute_and_log(self, trainer, pl_module, step_outputs, coco_gt, split):
        all_outputs = gather_outputs_across_processes(step_outputs)
        
        metrics_bbox = {}
        metrics_segm = {}
        
        if trainer.is_global_zero:
            predictions = []
            image_ids = []
            for batch_out in all_outputs:
                predictions_map = batch_out.get("predictions", {})
                # Use the lightning module's model_to_coco mapping for accurate class IDs
                model_to_coco = getattr(pl_module, "model_to_coco", None)
                predictions.extend(convert_preds_to_coco(predictions_map, model_to_coco=model_to_coco))
                image_ids.extend(batch_out.get("image_ids", []))
                
            if predictions and coco_gt is not None:
                # Calculate BBOX metrics
                metrics_bbox = compute_coco_metrics(
                    coco_gt=coco_gt,
                    predictions=predictions,
                    image_ids=sorted(set(image_ids)),
                    max_detections=int(pl_module.config.model.max_detections),
                    label_map=self.label_map,
                    prefix=f"Detailed YOLO-Style Performance ({split.upper()}) - BBOX",
                    iou_type="bbox",
                    metric_prefix="detailed_bbox"
                )
                
                # Check if segmentation is present
                is_seg = any("segmentation" in p for p in predictions)
                if is_seg:
                    # Calculate SEGM metrics
                    metrics_segm = compute_coco_metrics(
                        coco_gt=coco_gt,
                        predictions=predictions,
                        image_ids=sorted(set(image_ids)),
                        max_detections=int(pl_module.config.model.max_detections),
                        label_map=self.label_map,
                        prefix=f"Detailed YOLO-Style Performance ({split.upper()}) - SEGM",
                        iou_type="segm",
                        metric_prefix="detailed_segm"
                    )
                
        if torch.distributed.is_initialized() and torch.distributed.is_available():
            import torch.distributed as dist
            obj_list = [{"bbox": metrics_bbox, "segm": metrics_segm}]
            if dist.get_world_size() > 1:
                dist.broadcast_object_list(obj_list, src=0)
            metrics_bbox = obj_list[0]["bbox"]
            metrics_segm = obj_list[0]["segm"]
            
        for key, value in metrics_bbox.items():
            pl_module.log(f"{split}/{key}", value, sync_dist=True)
            
        for key, value in metrics_segm.items():
            pl_module.log(f"{split}/{key}", value, sync_dist=True)
            
        step_outputs.clear()
