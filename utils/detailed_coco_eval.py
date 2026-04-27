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
            if hasattr(dm, "val_coco_gt"):
                self.val_coco_gt = getattr(dm, "val_coco_gt")
            elif hasattr(dm, "_dataset_val") and hasattr(dm._dataset_val, "coco"):
                self.val_coco_gt = dm._dataset_val.coco
                
        if self.test_coco_gt is None and dm:
            if hasattr(dm, "test_coco_gt"):
                self.test_coco_gt = getattr(dm, "test_coco_gt")
            elif hasattr(dm, "_dataset_test") and hasattr(dm._dataset_test, "coco"):
                self.test_coco_gt = dm._dataset_test.coco
            elif hasattr(dm, "_dataset_val") and hasattr(dm._dataset_val, "coco"):
                self.test_coco_gt = dm._dataset_val.coco # fallback to val for testing if test not available
            
        if self.label_map is None:
            if hasattr(pl_module, "config") and hasattr(pl_module.config.model, "label_map"):
                self.label_map = pl_module.config.model.label_map
            elif hasattr(pl_module, "model_config") and hasattr(pl_module.model_config, "label_map"):
                self.label_map = pl_module.model_config.label_map
            elif dm and hasattr(dm, "class_names") and dm.class_names:
                self.label_map = {i: name for i, name in enumerate(dm.class_names)}
            elif self.val_coco_gt and hasattr(self.val_coco_gt, "cats"):
                # If we have coco_gt but no label_map yet, build it from coco_gt.cats
                if hasattr(self.val_coco_gt, "label2cat"):
                    self.label_map = {label: self.val_coco_gt.cats[cat_id]["name"] for label, cat_id in self.val_coco_gt.label2cat.items()}
                else:
                    self.label_map = {k: v["name"] for k, v in self.val_coco_gt.cats.items()}

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
        if not outputs or "results" not in outputs or "targets" not in outputs:
            return
            
        results = outputs["results"]
        targets = outputs["targets"]
        
        preds_for_metric = [to_cpu_device(res) for res in results]
            
        for pred in preds_for_metric:
            if "masks" in pred:
                masks = pred["masks"]
                if masks.ndim == 4 and masks.shape[1] == 1:
                    masks = masks.squeeze(1)
                segmentations = []
                for i in range(masks.shape[0]):
                    mask_np = np.asfortranarray(masks[i].numpy().astype(np.uint8))
                    rle = mask_utils.encode(mask_np)
                    rle["counts"] = rle["counts"].decode("utf-8")
                    segmentations.append(rle)
                pred["segmentation"] = segmentations
                del pred["masks"]
                
        result_map = {
            int(target["image_id"].item()): pred for target, pred in zip(targets, preds_for_metric)
        }
        image_ids = [int(target["image_id"].item()) for target in targets]
        
        storage_list.append({
            "predictions": result_map,
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
                # Find max_detections
                if hasattr(pl_module, "config") and hasattr(pl_module.config.model, "max_detections"):
                    max_dets = int(pl_module.config.model.max_detections)
                elif hasattr(pl_module, "train_config") and hasattr(pl_module.train_config, "eval_max_dets"):
                    max_dets = int(pl_module.train_config.eval_max_dets)
                else:
                    max_dets = 100

                # Calculate BBOX metrics
                metrics_bbox = compute_coco_metrics(
                    coco_gt=coco_gt,
                    predictions=predictions,
                    image_ids=sorted(set(image_ids)),
                    max_detections=max_dets,
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
                        max_detections=max_dets,
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
