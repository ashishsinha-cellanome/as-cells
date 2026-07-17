
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
    It also computes metrics for the EMA model if one is active.
    """
    def __init__(self):
        super().__init__()
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.validation_step_outputs_ema = []
        self.test_step_outputs_ema = []
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
                self.test_coco_gt = dm._dataset_val.coco
            
        if self.label_map is None:
            if hasattr(pl_module, "config") and hasattr(pl_module.config.model, "label_map"):
                self.label_map = pl_module.config.model.label_map
            elif hasattr(pl_module, "model_config") and hasattr(pl_module.model_config, "label_map"):
                self.label_map = pl_module.model_config.label_map
            elif dm and hasattr(dm, "class_names") and dm.class_names:
                self.label_map = {i: name for i, name in enumerate(dm.class_names)}
            elif self.val_coco_gt and hasattr(self.val_coco_gt, "cats"):
                if hasattr(self.val_coco_gt, "label2cat"):
                    self.label_map = {label: self.val_coco_gt.cats[cat_id]["name"] for label, cat_id in self.val_coco_gt.label2cat.items()}
                else:
                    self.label_map = {k: v["name"] for k, v in self.val_coco_gt.cats.items()}

    def _get_ema_callback(self, trainer):
        for callback in getattr(trainer, "callbacks", []):
            if callable(getattr(callback, "get_ema_model_state_dict", None)) or hasattr(callback, "_average_model") or hasattr(callback, "ema_model"):
                return callback
        return None

    def on_validation_epoch_start(self, trainer, pl_module):
        self.validation_step_outputs.clear()
        self.validation_step_outputs_ema.clear()
        self._ensure_metadata(trainer, pl_module)

    def on_test_epoch_start(self, trainer, pl_module):
        self.test_step_outputs.clear()
        self.test_step_outputs_ema.clear()
        self._ensure_metadata(trainer, pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._accumulate_batch(outputs, self.validation_step_outputs)
        
        ema_cb = self._get_ema_callback(trainer)
        if ema_cb is not None:
            self._evaluate_ema(ema_cb, pl_module, batch, self.validation_step_outputs_ema)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._accumulate_batch(outputs, self.test_step_outputs)
        
        ema_cb = self._get_ema_callback(trainer)
        if ema_cb is not None:
            self._evaluate_ema(ema_cb, pl_module, batch, self.test_step_outputs_ema)

    def _evaluate_ema(self, ema_cb, pl_module, batch, storage_list):
        ema_model = getattr(ema_cb, "_average_model", None) or getattr(ema_cb, "ema_model", None)
        if ema_model is not None:
            samples, targets = batch
            if hasattr(ema_model, "module") and hasattr(ema_model.module, "model"):
                ema_underlying = ema_model.module.model
            elif hasattr(ema_model, "module"):
                ema_underlying = ema_model.module
            else:
                ema_underlying = ema_model
                
            orig_sizes = torch.stack([t["orig_size"] for t in targets]).to(pl_module.device)
            
            with torch.no_grad():
                ema_underlying.eval()
                ema_outputs = ema_underlying(samples)
                # Some pipelines use pl_module.postprocess directly
                if hasattr(pl_module, "postprocess"):
                    ema_results = pl_module.postprocess(ema_outputs, orig_sizes)
                else:
                    ema_results = ema_outputs

            ema_outputs_dict = {
                "results": ema_results,
                "targets": targets
            }
            self._accumulate_batch(ema_outputs_dict, storage_list)

    def _accumulate_batch(self, outputs, storage_list):
        if not outputs or "results" not in outputs or "targets" not in outputs:
            if outputs and "predictions" in outputs and "image_ids" in outputs:
                storage_list.append({
                    "predictions": outputs["predictions"],
                    "image_ids": outputs["image_ids"]
                })
            return
            
        results = outputs["results"]
        targets = outputs["targets"]
        
        preds_for_metric = [to_cpu_device(res) for res in results]
            
        for pred in preds_for_metric:
            if "scores" in pred and len(pred["scores"]) > 100:
                topk = torch.topk(pred["scores"], 100)
                indices = topk.indices
                pred["scores"] = pred["scores"][indices]
                pred["labels"] = pred["labels"][indices]
                pred["boxes"] = pred["boxes"][indices]
                if "masks" in pred:
                    pred["masks"] = pred["masks"][indices]
            if "masks" in pred:
                masks = pred["masks"]
                if masks.ndim == 4 and masks.shape[1] == 1:
                    masks = masks.squeeze(1)
                # Vectorized batch encoding
                # Pycocotools encode expects Fortran-contiguous array of shape (H, W, N)
                # Input masks is shape (N, H, W)
                mask_np = masks.numpy().astype(np.uint8)
                mask_np_hw_n = np.asfortranarray(np.transpose(mask_np, (1, 2, 0)))
                
                # Batch encode all masks at once
                encoded_masks = mask_utils.encode(mask_np_hw_n)
                
                segmentations = []
                for i in range(masks.shape[0]):
                    rle = encoded_masks[i]
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
        print("\n[DetailedCocoEvalCallback] Computing Validation Metrics...")
        self._compute_and_log(trainer, pl_module, self.validation_step_outputs, self.val_coco_gt, "val", "")
        if len(self.validation_step_outputs_ema) > 0:
            print("\n[DetailedCocoEvalCallback] Computing Validation EMA Metrics...")
            self._compute_and_log(trainer, pl_module, self.validation_step_outputs_ema, self.val_coco_gt, "val", "_ema")

    def on_test_epoch_end(self, trainer, pl_module):
        print("\n[DetailedCocoEvalCallback] Computing Test Metrics...")
        self._compute_and_log(trainer, pl_module, self.test_step_outputs, self.test_coco_gt, "test", "")
        if len(self.test_step_outputs_ema) > 0:
            print("\n[DetailedCocoEvalCallback] Computing Test EMA Metrics...")
            self._compute_and_log(trainer, pl_module, self.test_step_outputs_ema, self.test_coco_gt, "test", "_ema")

    def _compute_and_log(self, trainer, pl_module, step_outputs, coco_gt, split, suffix):
        from tqdm import tqdm
        import torch.distributed as dist
        
        is_distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_distributed else 0
        
        predictions = []
        image_ids = []
        processed_image_ids = set()
        
        # Only iterate over LOCAL step_outputs
        for batch_out in tqdm(step_outputs, desc=f"Converting {split}{suffix} preds", disable=rank!=0):
            predictions_map = batch_out.get("predictions", {})
            filtered_predictions_map = {}
            for img_id, pred in predictions_map.items():
                if img_id not in processed_image_ids:
                    filtered_predictions_map[img_id] = pred
                    processed_image_ids.add(img_id)
                    image_ids.append(img_id)
                    
            model_to_coco = getattr(pl_module, "model_to_coco", None)
            predictions.extend(convert_preds_to_coco(filtered_predictions_map, model_to_coco=model_to_coco))
            
        max_dets = 100
        ema_label = " EMA" if suffix else ""
        
        metrics_bbox = compute_coco_metrics(
            coco_gt=coco_gt,
            predictions=predictions,
            image_ids=sorted(set(image_ids)),
            max_detections=max_dets,
            label_map=self.label_map,
            prefix=f"Detailed YOLO-Style Performance ({split.upper()}{ema_label}) - BBOX",
            iou_type="bbox",
            metric_prefix="detailed_bbox"
        )
        
        local_has_seg = any("segmentation" in p for p in predictions)
        if is_distributed:
            import torch
            has_seg_tensor = torch.tensor([local_has_seg], dtype=torch.uint8, device=pl_module.device)
            dist.all_reduce(has_seg_tensor, op=dist.ReduceOp.MAX)
            global_has_seg = bool(has_seg_tensor.item())
        else:
            global_has_seg = local_has_seg
            
        metrics_segm = {}
        if global_has_seg:
            metrics_segm = compute_coco_metrics(
                coco_gt=coco_gt,
                predictions=predictions,
                image_ids=sorted(set(image_ids)),
                max_detections=max_dets,
                label_map=self.label_map,
                prefix=f"Detailed YOLO-Style Performance ({split.upper()}{ema_label}) - SEGM",
                iou_type="segm",
                metric_prefix="detailed_segm"
            )
            
        if is_distributed:
            obj_list = [{"bbox": metrics_bbox, "segm": metrics_segm}]
            if dist.get_world_size() > 1:
                dist.broadcast_object_list(obj_list, src=0)
            metrics_bbox = obj_list[0]["bbox"]
            metrics_segm = obj_list[0]["segm"]
            
        for key, value in metrics_bbox.items():
            if key == "_markdown_table": continue
            pl_module.log(f"{split}/{key}{suffix}", value, sync_dist=True)
            
        for key, value in metrics_segm.items():
            if key == "_markdown_table": continue
            pl_module.log(f"{split}/{key}{suffix}", value, sync_dist=True)
            
        step_outputs.clear()
        return metrics_bbox, metrics_segm
