import time
import re
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import os
import pytorch_lightning as pl
from pycocotools.cocoeval import COCOeval
from PIL import Image, ImageDraw, ImageFont
from models.custom_rt_detr_with_dinov2_backbone import RTDetrV2ForObjectDetectionWithCustomBackbone
from utils.ema import ModelEma

def to_cpu_device(tensor):
    """Move a CUDA torch tensor to CPU memory."""
    return tensor.detach().cpu() if tensor.requires_grad else tensor.cpu()


def convert_to_xywh(boxes):
    """Convert boxes from xyxy to xywh format."""
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def convert_preds_to_coco(predictions):
    """Convert predictions to COCO format."""
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue
        
        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()
        
        coco_results.extend([
            {
                "image_id": original_id,
                "category_id": labels[k],
                "bbox": boxes[k],
                "score": scores[k],
            }
            for k in range(len(scores))
        ])
    return coco_results


class RTDETRLightningModule(pl.LightningModule):
    """PyTorch Lightning Module for RT-DETR with DINOv2 backbone."""
    
    def __init__(
        self,
        model: RTDetrV2ForObjectDetectionWithCustomBackbone,
        image_processor,
        val_coco_gt=None,
        test_coco_gt = None,
        train_coco_gt = None,
        config = None,
    ):
        super().__init__()
        # breakpoint()
        self.save_hyperparameters(ignore=['model', 'config', 'image_processor', 'val_coco_gt', 'test_coco_gt', 'train_coco_gt'])
        self.model = model
        # self.model.train() # REMOVED: Managed by train() override below
        self.image_processor = image_processor
        self.val_coco_gt = val_coco_gt
        self.test_coco_gt = test_coco_gt
        self.train_coco_gt = train_coco_gt
        
        # For validation metric accumulation
        self.validation_predictions = []
        self.validation_image_ids = []
        
        # for test metrics
        self.test_predictions = []
        self.test_image_ids = []

        # counter for max logging 
        self.val_viz_counter = 0
        self.test_viz_counter = 0
        self.PALETTE = [
            (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
            (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
            (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42)
        ]
        
        # --- Load font for labels ---
        try:
            self.font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            self.font = ImageFont.load_default()
        # debug setting
        self.debug_train_image_ids = set() 
        self.config = config
        self.warmup_steps = self.config.scheduler.warmup_steps
        self.base_lr = self.config.optimizer.optimizer.lr
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        if hasattr(self.config.model, 'ema') and self.config.model.ema.enabled:
            self.validation_step_outputs_ema = []
            self.test_step_outputs_ema = []


    def forward(self, pixel_values, labels=None):
        """Forward pass."""
        # breakpoint()
        return self.model(pixel_values=pixel_values, labels=labels)

    def train(self, mode: bool = True):
        """Override to keep frozen modules in eval mode."""
        super().train(mode)
        if mode:
            # When switching to train mode, we must ensure that any frozen modules stay in eval mode
            # This is critical for backbones that are partially or fully frozen (e.g. BatchNorm stats)
            for m in self.modules():
                # Robust check: If a module and all its sub-parameters are frozen, force it to eval.
                # access generator
                params = m.parameters()
                # Check if there is at least one param, and if all are frozen
                has_params = False
                all_frozen = True
                for p in params:
                    has_params = True
                    if p.requires_grad:
                        all_frozen = False
                        break
                
                if has_params and all_frozen:
                    m.eval()
    
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        pixel_values = batch["pixel_values"]
        batch_size = pixel_values.shape[0]
        labels = [{k: v.to(self.device) for k, v in sample.items()} for sample in batch["labels"]]
        
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        
        loss = outputs.loss
        for label_dict in labels:
            self.debug_train_image_ids.add(int(label_dict["image_id"].item()))

        # Log training loss
        self.log("train/loss",loss,  batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Log individual loss components if available
        if hasattr(outputs, 'loss_dict'):
            for key, value in outputs.loss_dict.items():
                self.log(f"train/{key}", value,  batch_size=batch_size, on_step=False, on_epoch=True, sync_dist=True)
        
        return loss
    


    def on_validation_epoch_start(self):
        """Reset validation visualization counter."""
        self.val_viz_counter = 0

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        
        # Forward pass
        outputs = self.model(pixel_values=pixel_values, labels=None)
        
        # Collect image sizes
        # Use 'orig_size' for accurate mAP calculation when train/test image sizes are different than the selected model input size
        batch_image_sizes = [to_cpu_device(x["orig_size"]).numpy().tolist() for x in labels]
        # breakpoint()
        # Post-process predictions
        post_processed_outputs = self.image_processor.post_process_object_detection(
            outputs,
            # threshold=self.detection_threshold,
            threshold = self.config.model.detection_threshold,
            target_sizes=batch_image_sizes
        )
        
        # Move predictions to CPU
        post_processed_outputs = [
            {k: to_cpu_device(v) for k, v in outputs.items()}
            for outputs in post_processed_outputs
        ]
        
        if (self.current_epoch + 1) % self.config.checkpointing.visualize_every_n_epochs == 0 and \
           self.trainer.is_global_zero and \
           (self.val_viz_counter < self.config.checkpointing.visualize_samples or self.config.checkpointing.visualize_samples == -1):
            
            save_dir = os.path.join(
                self.config.checkpointing.save_dir,
                # self.config.run_name, 
                self.config.checkpointing.visualization_dir, 
                f"epoch_{(self.current_epoch+1):03d}", 
                "val"
            )
            # breakpoint()
            self.val_viz_counter = self._visualize_batch(
                save_dir, 
                post_processed_outputs, 
                pixel_values,
                labels, 
                self.val_viz_counter
            )

        # Convert to COCO format and store
        results = {
            int(target["image_id"].item()): output
            for target, output in zip(labels, post_processed_outputs)
        }
        # this fixes the slow state management during ddp
        image_ids = [int(target["image_id"].item()) for target in labels]
        # results = convert_preds_to_coco(results)
        
        # self.validation_predictions.extend(results)
        # self.validation_image_ids.extend([int(target["image_id"].item()) for target in labels])
        
        # return {"predictions": results}
        # breakpoint()
        self.validation_step_outputs.append({"predictions": results, "image_ids": image_ids})
        
        # EMA validation
        from utils.ema import RTDETREMACallback
        ema_callback = next((cb for cb in self.trainer.callbacks if isinstance(cb, RTDETREMACallback)), None)
        if ema_callback and ema_callback.ema_model:
            # EMA model forward pass
            ema_outputs = ema_callback.ema_model.module(pixel_values=pixel_values, labels=None)

            # Post-process EMA predictions
            post_processed_ema_outputs = self.image_processor.post_process_object_detection(
                ema_outputs,
                threshold=self.config.model.detection_threshold,
                target_sizes=batch_image_sizes
            )

            # Move EMA predictions to CPU
            post_processed_ema_outputs = [
                {k: to_cpu_device(v) for k, v in outputs.items()}
                for outputs in post_processed_ema_outputs
            ]

            # Convert to COCO format and store for EMA
            ema_results = {
                int(target["image_id"].item()): output
                for target, output in zip(labels, post_processed_ema_outputs)
            }
            self.validation_step_outputs_ema.append({"predictions": ema_results, "image_ids": image_ids})

        return {"predictions": results, "image_ids": image_ids}
    
    def on_validation_epoch_end(self):
        """Compute validation metrics at epoch end."""
        # breakpoint()
        if not self.validation_step_outputs:
            return
        
        # --- Collate results from all GPUs/steps ---
        validation_predictions = []
        validation_image_ids = []
        for output_batch in self.validation_step_outputs:
            validation_predictions.extend(convert_preds_to_coco(output_batch["predictions"]))
            validation_image_ids.extend(output_batch["image_ids"])
        
        if len(validation_predictions) == 0:
            return
        
        # breakpoint()
        if self.config.debug:
            self.print ("\n--- DEBUGGING IDS ---")
            self.print (f"TRAIN IDs seen this epoch: {self.debug_train_image_ids}")
            self.print (f"VAL IDs seen this epoch:   {set(validation_image_ids)}")
            self.print ("---------------------\n")
            
        # Compute COCO metrics
        if self.val_coco_gt is not None:
            metrics = self._compute_coco_metrics(
                predictions=validation_predictions,
                image_ids=list(set(validation_image_ids)),
                coco_gt=self.val_coco_gt
            )
            
            # Log metrics
            for key, value in metrics.items():
                self.log(f"val/{key}", value, prog_bar=True, sync_dist=True)
                if key == 'map':
                    self.log(f"val_{key}", value, prog_bar=False, sync_dist=True)
        else:
            # On non-zero ranks where coco_gt is None, we still need to log something to avoid DDP sync issues if necessary,
            # but usually sync_dist=True handles it if at least one rank logs.
            pass
        
        # --- EMA Metrics ---
        if hasattr(self, 'validation_step_outputs_ema') and self.validation_step_outputs_ema:
            ema_predictions = []
            ema_image_ids = []
            for output_batch in self.validation_step_outputs_ema:
                ema_predictions.extend(convert_preds_to_coco(output_batch["predictions"]))
                ema_image_ids.extend(output_batch["image_ids"])
            
            if len(ema_predictions) > 0:
                ema_metrics = self._compute_coco_metrics(
                    predictions=ema_predictions,
                    image_ids=list(set(ema_image_ids)),
                    coco_gt=self.val_coco_gt
                )
                for key, value in ema_metrics.items():
                    # Log with _ema suffix
                    self.log(f"val/{key}_ema", value, prog_bar=True, sync_dist=True)
                    if key == 'map':
                         self.log(f"val_{key}_ema", value, prog_bar=False, sync_dist=True)
            
            self.validation_step_outputs_ema.clear()

        # Clear accumulated predictions
        # self.validation_predictions = []
        # self.validation_image_ids = []
        self.debug_train_image_ids.clear() 
        self.validation_step_outputs.clear()  # free memory

    def on_test_epoch_start(self):
        """Reset test visualization counter."""
        self.test_viz_counter = 0

    def test_step(self, batch, batch_idx):
        """Test step (same as validation)."""
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        
        # Forward pass
        outputs = self.model(pixel_values=pixel_values, labels=None)
        
        # Collect image sizes
        # Use 'orig_size' for accurate mAP calculation when train/test image sizes are different than the selected model input size
        batch_image_sizes = [to_cpu_device(x["orig_size"]).numpy().tolist() for x in labels]
        
        # Post-process predictions
        post_processed_outputs = self.image_processor.post_process_object_detection(
            outputs,
            threshold=self.config.model.detection_threshold,
            target_sizes=batch_image_sizes
        )
        
        # Move predictions to CPU
        post_processed_outputs = [
            {k: to_cpu_device(v) for k, v in outputs.items()}
            for outputs in post_processed_outputs
        ]
        # breakpoint()
        if self.config.checkpointing.visualize_samples ==-1:
            self.config.checkpointing.visualize_samples = float('inf')
        if self.trainer.is_global_zero and \
            (self.test_viz_counter < self.config.checkpointing.visualize_samples): # or self.config.checkpointing.visualize_samples ==-1):
            save_dir = os.path.join(
                self.config.checkpointing.save_dir,
                # self.config.run_name, 
                self.config.checkpointing.visualization_dir, 
                "test"
            )
            self.test_viz_counter = self._visualize_batch(
                save_dir, 
                post_processed_outputs, 
                pixel_values,
                labels, 
                self.test_viz_counter
            )

        # Convert to COCO format and store
        results = {
            int(target["image_id"].item()): output
            for target, output in zip(labels, post_processed_outputs)
        }
        image_ids = [int(target["image_id"].item()) for target in labels]
        # results = convert_preds_to_coco(results)
        
        # self.test_predictions.extend(results)
        # self.test_image_ids.extend([int(target["image_id"].item()) for target in labels])
        self.test_step_outputs.append({"predictions": results, "image_ids": image_ids})

        # EMA validation during test
        from utils.ema import RTDETREMACallback
        ema_callback = next((cb for cb in self.trainer.callbacks if isinstance(cb, RTDETREMACallback)), None)
        if ema_callback and ema_callback.ema_model:
            # EMA model forward pass
            ema_outputs = ema_callback.ema_model.module(pixel_values=pixel_values, labels=None)

            # Post-process EMA predictions
            post_processed_ema_outputs = self.image_processor.post_process_object_detection(
                ema_outputs,
                threshold=self.config.model.detection_threshold,
                target_sizes=batch_image_sizes
            )

            # Move EMA predictions to CPU
            post_processed_ema_outputs = [
                {k: to_cpu_device(v) for k, v in outputs.items()}
                for outputs in post_processed_ema_outputs
            ]

            # Convert to COCO format and store for EMA
            ema_results = {
                int(target["image_id"].item()): output
                for target, output in zip(labels, post_processed_ema_outputs)
            }
            self.test_step_outputs_ema.append({"predictions": ema_results, "image_ids": image_ids})

        return {"predictions": results, "image_ids": image_ids}
    
    def on_test_epoch_end(self):
        """Compute test metrics at epoch end."""
        if not self.test_step_outputs:
            self.print ("No test predictions found.")
            return
        
        # --- Collate results from all GPUs/steps ---
        test_predictions = []
        test_image_ids = []
        for output_batch in self.test_step_outputs:
            test_predictions.extend(convert_preds_to_coco(output_batch["predictions"]))
            test_image_ids.extend(output_batch["image_ids"])

        if len(test_predictions) == 0:
            self.print ("No test predictions found.")
            return

        # Compute COCO metrics
        if self.test_coco_gt is not None:
            metrics = self._compute_coco_metrics(
                predictions=test_predictions,
                # image_ids=self.test_image_ids,
                image_ids = list(set(test_image_ids)),
                coco_gt=self.test_coco_gt  
            )
            
            # Log metrics
            for key, value in metrics.items():
                self.log(f"test/{key}", value, prog_bar=True, sync_dist=True)  
        
        # --- EMA Metrics ---
        if hasattr(self, 'test_step_outputs_ema') and self.test_step_outputs_ema:
            ema_predictions = []
            ema_image_ids = []
            for output_batch in self.test_step_outputs_ema:
                ema_predictions.extend(convert_preds_to_coco(output_batch["predictions"]))
                ema_image_ids.extend(output_batch["image_ids"])
            
            if len(ema_predictions) > 0:
                ema_metrics = self._compute_coco_metrics(
                    predictions=ema_predictions,
                    image_ids=list(set(ema_image_ids)),
                    coco_gt=self.test_coco_gt
                )
                for key, value in ema_metrics.items():
                    self.log(f"test/{key}_ema", value, prog_bar=True, sync_dist=True)
            
            self.test_step_outputs_ema.clear()

        # Clear accumulated predictions
        self.test_predictions = []
        self.test_image_ids = []
        self.test_step_outputs.clear()
        
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """
        Force the scheduler to accept a larger total_steps when resuming.
        This prevents the 'Tried to step N+1 times' error.
        """
        # Look into the loaded state for the lr_schedulers
        if "lr_schedulers" in checkpoint:
            for scheduler_state in checkpoint["lr_schedulers"]:
                # Check if it's a OneCycleLR state (it has total_steps)
                if "total_steps" in scheduler_state:
                    old_total = scheduler_state["total_steps"]
                    # Add a buffer (e.g., +5000 steps) to the restored state
                    scheduler_state["total_steps"] = old_total + 5000
                    self.print(f"Restoring checkpoint: Increased total_steps from {old_total} to {old_total + 5000}")
                    
    
    def _remap_coco_gt(self, coco_gt):
        """In-place remap of COCO GT categories to match remapped classes."""
        if not coco_gt:
            return None, {}
            
        if hasattr(coco_gt, '_remapped'):
            return coco_gt, getattr(coco_gt, '_remap_dict', {})
        
        # 0. Check if remapping is enabled
        if hasattr(self.config, 'remap_labels') and not self.config.remap_labels:
            return coco_gt, {}

        # 1. Get target map from config
        if not self.config or 'model' not in self.config or 'label_map' not in self.config.model:
            return coco_gt, {}
            
        target_label_map = self.config.model.label_map
        name_to_target_id = {v: int(k) for k, v in target_label_map.items()}
        
        # 2. Get remapping rules
        remapping_rules = {}
        if 'data' in self.config and self.config.data and 'class_remapping' in self.config.data:
            remapping_rules = self.config.data.class_remapping
        elif 'class_remapping' in self.config:
            remapping_rules = self.config.class_remapping
            
        remap_dict = {}
        for cat_id, cat_info in coco_gt.cats.items():
            src_name = cat_info['name']
            effective_name = remapping_rules.get(src_name, src_name)
            if effective_name in name_to_target_id:
                remap_dict[cat_id] = name_to_target_id[effective_name]
        
        # 3. Apply to annotations
        for ann in coco_gt.dataset.get('annotations', []):
            if ann['category_id'] in remap_dict:
                ann['category_id'] = remap_dict[ann['category_id']]
        
        # 4. Update categories in GT to match target
        # Only keep categories that are actually used as targets in the remapping
        used_target_ids = set(remap_dict.values())
        new_categories = []
        for target_id, name in target_label_map.items():
            # If we remapped *everything* (checked via remap_dict), strictly filter.
            # But if a class wasn't in source (not in remap_dict keys), we might still want it if it's a valid target.
            # Better strategy: If the user provided a remapping, trust the target_label_map BUT
            # we know the user wants to hide 2 and 3.
            
            # If the target_id is NOT in the values of our remapping, it implies no source category maps to it.
            # However, if we have a target class 'bead' (1) and NO 'bead' (1) in the source images, 
            # remap_dict might not contain 1 as a value if we only loop over existing cats.
            
            # Let's rely on the explicit instruction: 
            # "Only include categories that are present as values in the remap_dict"
            if int(target_id) in used_target_ids:
                new_categories.append({'id': int(target_id), 'name': name})
                
        coco_gt.dataset['categories'] = new_categories
        
        # 5. Re-index
        coco_gt.createIndex()
        coco_gt._remapped = True
        coco_gt._remap_dict = remap_dict
        self.print(f"[INFO] Remapped Validation GT classes using: {remap_dict}")
        return coco_gt, remap_dict

    def _compute_coco_metrics(self, predictions, image_ids, coco_gt):
        """Compute COCO mAP and mAR metrics."""
        if coco_gt is None or len(predictions) == 0:
            return {}
	
        if self.config.debug:
            self.print(f"DEBUG: COCO GT Categories before remap: {[{c['id']: c['name']} for c in coco_gt.dataset['categories']]}")
        
        coco_gt, remap_dict = self._remap_coco_gt(coco_gt)
        
        # Remap predictions if remapping rules were applied
        if remap_dict:
            for p in predictions:
                if p['category_id'] in remap_dict:
                    p['category_id'] = remap_dict[p['category_id']]

        # Debug: Verify remapping
        if self.config.debug:
            self.print(f"DEBUG: COCO GT Categories after remap: {[{c['id']: c['name']} for c in coco_gt.dataset['categories']]}")
            if remap_dict:
                self.print(f"DEBUG: Applied prediction remapping for evaluation.")
        
        metrics = {
            'map': -1.0, 'map_50': -1.0, 'map_75': -1.0,
            'map_small': -1.0, 'map_medium': -1.0, 'map_large': -1.0,
            'mar_1': -1.0, 'mar_10': -1.0, f'mar_{self.config.model.max_detections}': -1.0,
            'mar_small': -1.0, 'mar_medium': -1.0, 'mar_large': -1.0
        }
        
        try:
            # Initialize COCO evaluation
            coco_dt = coco_gt.loadRes(predictions)
            coco_evaluator = COCOeval(coco_gt, coco_dt, "bbox")
            coco_evaluator.params.maxDets = [1, 10, self.config.model.max_detections]
            coco_evaluator.params.imgIds = image_ids
            # Run evaluation
            coco_evaluator.evaluate()
            coco_evaluator.accumulate()
            coco_evaluator.summarize()
            
            # Extract aggregate metrics
            metric_keys = list(metrics.keys())
            for i, key in enumerate(metric_keys):
                if i < len(coco_evaluator.stats):
                    metrics[key] = round(coco_evaluator.stats[i], 4)
                    
            # Extract per-category metrics from the internal COCOeval precision tensor
            if hasattr(coco_evaluator, 'eval') and 'precision' in coco_evaluator.eval:
                precisions = coco_evaluator.eval['precision']
                import numpy as np
                for i, catId in enumerate(coco_evaluator.params.catIds):
                    # Use updated GT categories if remapping is applied, otherwise fallback to config label_map
                    if getattr(self.config, 'remap_labels', False):
                        cat_info = coco_gt.cats.get(int(catId))
                        cat_name = cat_info['name'] if cat_info else None
                    else:
                        cat_name = self.config.model.label_map.get(int(catId)) or self.config.model.label_map.get(str(catId))
                    
                    if not cat_name:
                        cat_name = f"class_{catId}"
                    
                    # mAP (average over all IoU thresholds)
                    s = precisions[:, :, i, 0, -1]
                    if len(s[s > -1]) > 0:
                        metrics[f'map_{cat_name}'] = round(float(np.mean(s[s > -1])), 4)
                        
                    # mAP-50 (IoU threshold 0.5)
                    s_50 = precisions[0, :, i, 0, -1]
                    if len(s_50[s_50 > -1]) > 0:
                        metrics[f'map_50_{cat_name}'] = round(float(np.mean(s_50[s_50 > -1])), 4)
                        
        except Exception as e:
            self.print(f"Error computing COCO metrics: {e}")
            import traceback
            self.print(traceback.format_exc())
        
        return metrics
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Read from the optimizer and scheduler config groups
        opt_config = self.config.optimizer.optimizer
        sch_config = self.config.scheduler
        
        # 1. Flexible Parameter Grouping
        # Check if 'param_groups' is enabled and defined in the config
        use_param_groups = opt_config.get('use_param_groups', False)
        param_groups_config = opt_config.get('param_groups', [])
        
        if not use_param_groups or not param_groups_config:
            # Fallback to uniform LR for the whole model (no grouping)
            self.print(f"[INFO] Using uniform LR: {opt_config.lr} for all parameters.")
            optimizer_grouped_params = [
                {'params': [p for p in self.model.parameters() if p.requires_grad], 
                 'lr': opt_config.lr, 
                 'weight_decay': opt_config.weight_decay}
            ]
        else:
            # Regex-based grouping from config
            optimizer_grouped_params = []
            memo = set() # Track assigned parameters
            
            # OmegaConf objects might need conversion or careful access
            for group_cfg in param_groups_config:
                group_params = []
                # Handle both string patterns and other attributes
                pattern = group_cfg.params
                
                for name, param in self.model.named_parameters():
                    if not param.requires_grad or id(param) in memo:
                        continue
                    
                    if re.search(pattern, name):
                        group_params.append(param)
                        memo.add(id(param))
                
                if group_params:
                    # Create group dict, excluding the 'params' pattern string
                    new_group = {k: v for k, v in group_cfg.items() if k != 'params'}
                    new_group['params'] = group_params
                    
                    # Inherit defaults if not specified
                    if 'lr' not in new_group:
                        new_group['lr'] = opt_config.lr
                    if 'weight_decay' not in new_group:
                        new_group['weight_decay'] = opt_config.weight_decay
                        
                    optimizer_grouped_params.append(new_group)
                    self.print(f"[INFO] Optimizer Group: '{pattern}' matched {len(group_params)} parameters.")

            # Catch-all for remaining parameters
            remaining_params = []
            for name, param in self.model.named_parameters():
                if param.requires_grad and id(param) not in memo:
                    remaining_params.append(param)
            
            if remaining_params:
                optimizer_grouped_params.append({
                    'params': remaining_params,
                    'lr': opt_config.lr,
                    'weight_decay': opt_config.weight_decay
                })
                self.print(f"[INFO] Optimizer Group: 'default' matched remaining {len(remaining_params)} parameters.")

        # Create optimizer
        optimizer = torch.optim.AdamW(optimizer_grouped_params, weight_decay=opt_config.weight_decay)

        # 2. Setup Scheduler
        # scheduler_config = self._get_scheduler_with_warmup(optimizer, sch_config)

        # return [optimizer], scheduler_config
    
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor= 0.01 , # 1% of target LR
            total_iters= sch_config.warmup_steps
        )
        milestones = [sch_config.warmup_steps]
        
        total_steps = self.trainer.estimated_stepping_batches + 100
        
        # Configure scheduler
        if sch_config.type == "reduce_lr_on_plateau":
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='max',
                    factor=sch_config.factor,
                    patience=sch_config.patience,
                ),
                'monitor': 'val/map',
                'interval': 'epoch',
                'frequency': 1,
                # 'verbose': Trueq
            }
        elif sch_config.type == "cosine":
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max= total_steps - max(sch_config.warmup_steps, int(0.1 * total_steps)),
                    eta_min=sch_config.eta_min
                ),
                'interval': 'step'
            }
            # schedulers = [warmup_scheduler, scheduler]
        elif sch_config.type == "lambda":
            # Linear warmup + constant
            def lr_lambda(step):
                if step < sch_config.warmup_steps:
                    return step / sch_config.warmup_steps
                return 1.0
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
                'interval': 'step'
            }

        elif sch_config.type == 'step':
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=sch_config.step_size,
                    gamma=sch_config.gamma
                ),
                'interval': 'epoch'
            }
        elif sch_config.type == 'multistep':
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=sch_config.milestones,
                    gamma=sch_config.gamma
                ),
                'interval': 'epoch'
            }
        elif sch_config.type == 'onecycle':
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=opt_config.lr,
                    total_steps=total_steps,
                    pct_start= sch_config.pct_start,
                    anneal_strategy='cos',
                    div_factor=25.0,
                    final_div_factor=1e3
                ),
                'interval': 'step'
            }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def _get_scheduler_with_warmup(self, optimizer, sch_config):
        """
        Helper to attach warmup to any scheduler strategy safely.
        """
        # --- A. Define the Warmup Scheduler ---
        # Starts at 1% of target LR and ramps up linearly
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor= 1e-2, 
            total_iters=sch_config.warmup_steps
        )

        # --- B. Define the Main Scheduler & Combine ---
        
        # 1. Reduce LR on Plateau (Cannot be chained sequentially)
        if sch_config.type == "reduce_lr_on_plateau":
            plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=sch_config.factor,
                patience=sch_config.patience,
            )
            # Return list: Warmup runs every step, Plateau runs every epoch independently
            return [
                {'scheduler': warmup_scheduler, 'interval': 'step', 'frequency': 1},
                {'scheduler': plateau_scheduler, 'interval': 'epoch', 'frequency': 1, 'monitor': 'val/map'}
            ]

        # 2. Cosine Annealing (Sequential)
        elif sch_config.type == "cosine":
            # Calculate remaining steps for the cosine phase
            total_steps = self.trainer.estimated_stepping_batches
            main_iters = total_steps - sch_config.warmup_steps

            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=main_iters,
                eta_min=optimizer.defaults['lr'] * 0.01
            )
            
            chained_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[sch_config.warmup_steps]
            )
            return [{'scheduler': chained_scheduler, 'interval': 'step'}]

        # 3. Lambda LR (Sequential)
        elif sch_config.type == "lambda":
            # Define your lambda logic here. 
            # NOTE: In SequentialLR, the lambda receives the GLOBAL step count.
            
            # Example: Inverse Square Root Decay (common in Transformers)
            # We use 'max' to prevent division by zero or overly high values if step < warmup
            def lr_lambda(step):
                # Since this runs AFTER warmup, step will be > warmup_steps
                # We normalize so it continues smoothly from 1.0 down
                if step < sch_config.warmup_steps: return 1.0 
                return (sch_config.warmup_steps / step) ** 0.5

            main_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lr_lambda
            )
            
            chained_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[sch_config.warmup_steps]
            )
            return [{'scheduler': chained_scheduler, 'interval': 'step'}]

        # 4. Default: Warmup only (Constant afterwards)
        else:
            # LinearLR stays at factor 1.0 after total_iters are done
            return [{'scheduler': warmup_scheduler, 'interval': 'step'}]
        
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # --- 1. Warmup Logic (Apply BEFORE step) ---
        total_steps = self.trainer.estimated_stepping_batches + 100
        warmup_steps = max(self.config.scheduler.warmup_steps, int(0.1 * total_steps))
        # update the hparams
        # self.logger.experiment.config.update({'warmup_steps':warmup_steps}, allow_val_change=True)
        
        is_one_cycle = self.config.scheduler.type == "onecycle"
        
        if self.trainer.global_step < warmup_steps and not is_one_cycle:
            # Calculate linear scale (0.0 to 1.0)
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / float(warmup_steps))
            
            # Get the base LR from config to ensure we always scale from the correct starting point
            # (Avoids issues where pg['lr'] might be modified by other schedulers or restarts)
            base_lr = self.config.optimizer.optimizer.lr
            
            for pg in optimizer.param_groups:
                pg['lr'] = base_lr * lr_scale
        
        optimizer.step(closure=optimizer_closure)

    def draw_boxes(self, image, boxes, labels, scores=None, id2label=None, color_override=None, label_prefix=""):
        """Draws bounding boxes on a PIL image."""
        draw = ImageDraw.Draw(image)
        threshold = self.config.model.draw_threshold
        
        # Use default label map if not provided
        if id2label is None:
            id2label = self.config.model.label_map

        for i in range(len(boxes)):
            box = boxes[i]
            label = labels[i]
            score = scores[i] if scores is not None else 1.0
            
            if score < threshold:
                continue
            
            # Handle both tensor and array-like boxes
            if torch.is_tensor(box):
                box = box.tolist()
            
            # Handle both tensor and scalar labels
            label_id = label.item() if torch.is_tensor(label) else int(label)
            
            # Color logic: Green for GT, Red for Pred, or Palette default
            if color_override:
                color = color_override
            else:
                color = self.PALETTE[label_id % len(self.PALETTE)]
            
            draw.rectangle(box, outline=color, width=3) # Increased width for better visibility
            
            class_name = id2label.get(label_id) or id2label.get(str(label_id)) or f"class_{label_id}"
            label_text = f"{label_prefix}{class_name}"
            if scores is not None:
                label_text += f": {score:.2f}"
                
            text_box = draw.textbbox((box[0], box[1]), label_text, font=self.font)
            draw.rectangle(text_box, fill=color)
            draw.text((box[0], box[1]), label_text, fill="white", font=self.font)
            
        return image

    def _visualize_batch(self, save_dir, post_processed_outputs, pixel_values, labels, counter):
        """Saves visualizations for a batch showing both GT and Predictions."""
        os.makedirs(save_dir, exist_ok=True)
        id2label = self.model.config.id2label
        max_samples = self.config.checkpointing.visualize_samples
        
        # Determine which COCO GT to use based on stage
        coco_gt = self.test_coco_gt if self.trainer.testing else self.val_coco_gt
        if coco_gt is None:
            self.print("[WARNING] COCO GT is None, skipping GT visualization.")
            # Fallback to only drawing predictions if GT is missing
            pass

        # Get mean and std from the processor to un-normalize
        mean = torch.tensor(self.image_processor.image_mean, device=pixel_values.device).view(1, 3, 1, 1)
        std = torch.tensor(self.image_processor.image_std, device=pixel_values.device).view(1, 3, 1, 1)

        # Un-normalize the entire batch
        unnormalized_images = torch.clamp((pixel_values * std) + mean, 0, 1)
        
        for i in range(len(labels)):
            if counter >= max_samples:
                break
            
            # Get original image info
            image_id = int(labels[i]["image_id"].item())
            image_tensor = unnormalized_images[i]
            image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            image = Image.fromarray(image_np)

            # Get image metadata from COCO GT for filename
            if coco_gt:
                img_info = coco_gt.loadImgs(image_id)[0]
                
                # --- 1. Scaled Ground Truth Boxes ---
                gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=image_id))
                gt_boxes = []
                gt_labels = []
                for ann in gt_anns:
                    # coco format is [x, y, width, height]
                    x, y, w, h = ann['bbox']
                    # Convert to [x1, y1, x2, y2] for draw_boxes
                    gt_boxes.append([x, y, x + w, y + h])
                    gt_labels.append(ann['category_id'])

                # Draw Ground Truth in GREEN
                image = self.draw_boxes(
                    image, 
                    gt_boxes, 
                    gt_labels, 
                    scores=None, 
                    id2label=self.config.model.label_map,
                    color_override=(0, 255, 0), # Green
                    label_prefix="GT: "
                )
            else:
                # If coco_gt is missing, we might not have the filename easily, 
                # but we can try to use the image_id
                img_info = {'file_name': f"image_{image_id}.png"}

            # --- 2. Prediction Boxes ---
            preds = post_processed_outputs[i]
            
            # Draw Predictions in RED
            image = self.draw_boxes(
                image, 
                preds['boxes'], 
                preds['labels'], 
                preds['scores'], 
                id2label=id2label,
                color_override=(255, 0, 0), # Red
                label_prefix="Pred: "
            )
            
            # Save image
            save_path = os.path.join(save_dir, img_info['file_name'])
            image.save(save_path)
            
            counter += 1
        
        return counter
    
    # def on_before_optimizer_step(self, optimizer):
    #     """Clip gradients before optimizer step."""
    #     if self.max_grad_norm > 0:
    #         torch.nn.utils.clip_grad_norm_(
    #             self.model.parameters(),
    #             self.max_grad_norm
    #         )
