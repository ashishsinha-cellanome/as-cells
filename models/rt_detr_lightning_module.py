import time
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import os
import pytorch_lightning as pl
from pycocotools.cocoeval import COCOeval
from PIL import Image, ImageDraw, ImageFont
from models.custom_rt_detr_with_dinov2_backbone import RTDetrV2ForObjectDetectionWithCustomBackbone


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
        self.save_hyperparameters(ignore=['model', 'config', 'image_processor', 'val_coco_gt', 'test_coco_gt', 'train_coco_gt'])
        # breakpoint()
        self.model = model
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
        # breakpoint()
        self.warmup_steps = self.config.optimizer.scheduler.warmup_steps
        self.base_lr = self.config.optimizer.optimizer.lr
        # breakpoint()
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, pixel_values, labels=None):
        """Forward pass."""
        # breakpoint()
        return self.model(pixel_values=pixel_values, labels=labels)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        pixel_values = batch["pixel_values"]
        batch_size = pixel_values.shape[0]
        labels = [{k: v.to(self.device) for k, v in sample.items()} for sample in batch["labels"]]
        
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        # breakpoint()
        loss = outputs.loss
        # batch_size = len(labels)
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
        batch_image_sizes = [to_cpu_device(x["size"]).numpy().tolist() for x in labels]
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
           self.val_viz_counter < self.config.checkpointing.visualize_samples:
            
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
        batch_image_sizes = [to_cpu_device(x["size"]).numpy().tolist() for x in labels]
        
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

        if self.trainer.is_global_zero and self.test_viz_counter < self.config.checkpointing.visualize_samples:
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
        metrics = self._compute_coco_metrics(
            predictions=test_predictions,
            # image_ids=self.test_image_ids,
            image_ids = list(set(test_image_ids)),
            coco_gt=self.test_coco_gt  
        )
        
        # Log metrics
        for key, value in metrics.items():
            self.log(f"test/{key}", value, prog_bar=True, sync_dist=True)  
        
        # Clear accumulated predictions
        self.test_predictions = []
        self.test_image_ids = []
        self.test_step_outputs.clear()
    
    def _compute_coco_metrics(self, predictions, image_ids, coco_gt):
        """Compute COCO mAP and mAR metrics."""
        if coco_gt is None or len(predictions) == 0:
            return {}
        
        metrics = {
            'map': -1.0, 'map_50': -1.0, 'map_75': -1.0,
            'map_small': -1.0, 'map_medium': -1.0, 'map_large': -1.0,
            'mar_1': -1.0, 'mar_10': -1.0, f'mar_{self.config.model.max_detections}': -1.0,
            'mar_small': -1.0, 'mar_medium': -1.0, 'mar_large': -1.0
        }
        # breakpoint()
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
            
            # Extract metrics
            #TODO: check here
            metric_keys = list(metrics.keys())
            for i, key in enumerate(metric_keys):
                # if i < len(coco_evaluator.stats):
                metrics[key] = round(coco_evaluator.stats[i], 4)
        except Exception as e:
            print(f"Error computing COCO metrics: {e}")
        
        return metrics
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Read from the optimizer and scheduler config groups
        # breakpoint()
        opt_config = self.config.optimizer.optimizer
        sch_config = self.config.optimizer.scheduler
        
        # Separate parameters: freeze DINOv2, train FPN and rest of model
        backbone_params = []
        other_params = []
        # breakpoint()
        # TODO: same as HF, comment below to change LR
        # trainable_params = [
        #     param for param in self.model.parameters() if param.requires_grad
        # ]
        # optimizer = torch.optim.AdamW(
        #     trainable_params,
        #     lr=self.learning_rate,
        #     weight_decay=self.weight_decay
        # )
        for name, param in self.model.named_parameters():
            if 'model.backbone.backbone' in name:
                # DINOv2 backbone - frozen
                param.requires_grad = False
            elif 'model.backbone' in name:
                # FPN part of backbone - trainable
                backbone_params.append(param)
            else:
                # Rest of model - trainable
                other_params.append(param)
        
        # Different learning rates for different parts
        # breakpoint()
        optimizer = torch.optim.AdamW([
            {'params': other_params, 'lr': opt_config.lr},
            {'params': backbone_params, 'lr': opt_config.lr}  # Lower LR for FPN if necessary
            ], 
            weight_decay= opt_config.weight_decay
        )
        
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
        elif sch_config.typ == "cosine":
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.config.trainer.max_epochs,
                    eta_min=opt_config.lr * 0.01
                ),
                'interval': 'epoch'
            }
        else:
            # Linear warmup + constant
            def lr_lambda(step):
                if step < sch_config.warmup_steps:
                    return step / sch_config.warmup_steps
                return 1.0
            
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
                'interval': 'step'
            }
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # 1. Execute the optimizer step (gradient update)
        optimizer.step(closure=optimizer_closure)

        # 2. Warmup Logic
        # We use 'global_step' which tracks total batches across all epochs
        if self.trainer.global_step < self.config.optimizer.scheduler.warmup_steps:
            
            # Calculate linear scale (0.0 to 1.0)
            # We add +1 so we don't start at exactly 0.0, which can cause issues for some optimizers
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.config.optimizer.scheduler.warmup_steps)
            
            # Apply to ALL param groups (handles your Backbone vs Head split automatically)
            for pg in optimizer.param_groups:
                # We save the 'initial_lr' in the param group when the optimizer is created.
                # If it's not there (first step), we use the current 'lr' as the initial.
                if 'initial_lr' not in pg:
                    pg['initial_lr'] = pg['lr']
                
                # Update the current LR based on the initial base
                pg['lr'] = pg['initial_lr'] * lr_scale

    def draw_boxes(self, image, boxes, labels, scores, id2label):
        """Draws bounding boxes on a PIL image."""
        draw = ImageDraw.Draw(image)
        threshold = self.config.model.draw_threshold
        for box, label, score in zip(boxes, labels, scores):
            if score < threshold:
                continue
            
            box = box.tolist()
            label_id = label.item()
            color = self.PALETTE[label_id % len(self.PALETTE)]
            
            draw.rectangle(box, outline=color, width=2)
            
            label_text = f"{id2label[label_id]}: {score:.2f}"
            text_box = draw.textbbox((box[0], box[1]), label_text, font=self.font)
            draw.rectangle(text_box, fill=color)
            draw.text((box[0], box[1]), label_text, fill="white", font=self.font)
            
        return image

    def _visualize_batch(self, save_dir, post_processed_outputs, pixel_values, labels, counter):
        """Saves visualizations for a batch."""
        os.makedirs(save_dir, exist_ok=True)
        id2label = self.model.config.id2label
        max_samples = self.config.checkpointing.visualize_samples
        # Get mean and std from the processor to un-normalize
        mean = torch.tensor(self.image_processor.image_mean, device=pixel_values.device).view(1, 3, 1, 1)
        std = torch.tensor(self.image_processor.image_std, device=pixel_values.device).view(1, 3, 1, 1)

        # Un-normalize the entire batch at once (faster)
        # 1. Multiply by std, 2. Add mean, 3. Clamp to [0, 1] range
        unnormalized_images = torch.clamp((pixel_values * std) + mean, 0, 1)
        for i in range(len(labels)):
            if counter >= max_samples:
                break
            
            # Get original image path from COCO GT
            image_id = int(labels[i]["image_id"].item())
            image_tensor = unnormalized_images[i]
            image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            image = Image.fromarray(image_np)

            img_info = self.val_coco_gt.loadImgs(image_id)[0]
            # img_path = os.path.join(self.config['checkpointing']['val_image_dir'], img_info['file_name'])
            # breakpoint()
            # try:
            #     image = Image.open(img_path).convert("RGB")
            # except FileNotFoundError:
            #     print(f"Warning: Could not find image {img_path} for visualization.")
            #     continue
                
            # Get predictions for this image
            preds = post_processed_outputs[i]
            
            # Draw boxes
            drawn_image = self.draw_boxes(
                image, 
                preds['boxes'], 
                preds['labels'], 
                preds['scores'], 
                id2label
            )
            
            # Save image
            save_path = os.path.join(save_dir, img_info['file_name'])
            drawn_image.save(save_path)
            
            counter += 1
        
        return counter
    
    # def on_before_optimizer_step(self, optimizer):
    #     """Clip gradients before optimizer step."""
    #     if self.max_grad_norm > 0:
    #         torch.nn.utils.clip_grad_norm_(
    #             self.model.parameters(),
    #             self.max_grad_norm
    #         )



class RTDETRLightningModuleDebug(pl.LightningModule):
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
        self.save_hyperparameters(ignore=['model', 'config', 'image_processor', 'val_coco_gt', 'test_coco_gt', 'train_coco_gt'])
        # breakpoint()
        self.model = model
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
        # self.
        # breakpoint()
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, pixel_values, labels=None):
        """Forward pass."""
        # breakpoint()
        return self.model(pixel_values=pixel_values, labels=labels)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        pixel_values = batch["pixel_values"]
        batch_size = pixel_values.shape[0]
        labels = [{k: v.to(self.device) for k, v in sample.items()} for sample in batch["labels"]]
        
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        # breakpoint()
        # sum([v for k,v in outputs.loss_dict.items() if 'vfl' in k])
        if 'auxiliary_outputs' in outputs:
            for k,v in outputs.loss_dict.items():
                print (f"{k}: {v.item():.5f}")
                if 'vfl_dn_2' in k:
                    outputs.loss -= outputs.loss_dict[k] 
                    outputs.loss_dict['loss_vfl'] -= outputs.loss_dict[k] 
                    outputs.loss_dict[k] *= 1e-2
                    outputs.loss += outputs.loss_dict[k]
                    outputs.loss_dict['loss_vfl'] += outputs.loss_dict[k]
                    # outputs.loss_dict[k] = 0
        # breakpoint()
        loss = outputs.loss
        # batch_size = len(labels)
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
        batch_image_sizes = [to_cpu_device(x["size"]).numpy().tolist() for x in labels]
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
           self.val_viz_counter < self.config.checkpointing.visualize_samples:
            
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
        batch_image_sizes = [to_cpu_device(x["size"]).numpy().tolist() for x in labels]
        
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

        if self.trainer.is_global_zero and self.test_viz_counter < self.config.checkpointing.visualize_samples:
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
        metrics = self._compute_coco_metrics(
            predictions=test_predictions,
            # image_ids=self.test_image_ids,
            image_ids = list(set(test_image_ids)),
            coco_gt=self.test_coco_gt  
        )
        
        # Log metrics
        for key, value in metrics.items():
            self.log(f"test/{key}", value, prog_bar=True, sync_dist=True)  
        
        # Clear accumulated predictions
        self.test_predictions = []
        self.test_image_ids = []
        self.test_step_outputs.clear()
    
    def _compute_coco_metrics(self, predictions, image_ids, coco_gt):
        """Compute COCO mAP and mAR metrics."""
        if coco_gt is None or len(predictions) == 0:
            return {}
        
        metrics = {
            'map': -1.0, 'map_50': -1.0, 'map_75': -1.0,
            'map_small': -1.0, 'map_medium': -1.0, 'map_large': -1.0,
            'mar_1': -1.0, 'mar_10': -1.0, f'mar_{self.config.model.max_detections}': -1.0,
            'mar_small': -1.0, 'mar_medium': -1.0, 'mar_large': -1.0
        }
        # breakpoint()
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
            
            # Extract metrics
            #TODO: check here
            metric_keys = list(metrics.keys())
            for i, key in enumerate(metric_keys):
                # if i < len(coco_evaluator.stats):
                metrics[key] = round(coco_evaluator.stats[i], 4)
        except Exception as e:
            print(f"Error computing COCO metrics: {e}")
        
        return metrics
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Read from the optimizer and scheduler config groups
        # breakpoint()
        opt_config = self.config.optimizer.optimizer
        sch_config = self.config.optimizer.scheduler
        
        # Separate parameters: freeze DINOv2, train FPN and rest of model
        backbone_params = []
        other_params = []
        # breakpoint()
        # TODO: same as HF, comment below to change LR
        # trainable_params = [
        #     param for param in self.model.parameters() if param.requires_grad
        # ]
        # optimizer = torch.optim.AdamW(
        #     trainable_params,
        #     lr=self.learning_rate,
        #     weight_decay=self.weight_decay
        # )
        for name, param in self.model.named_parameters():
            if 'model.backbone.backbone' in name:
                # DINOv2 backbone - frozen
                param.requires_grad = False
            elif 'model.backbone' in name:
                # FPN part of backbone - trainable
                backbone_params.append(param)
            else:
                # Rest of model - trainable
                other_params.append(param)
        
        # Different learning rates for different parts
        # breakpoint()
        optimizer = torch.optim.AdamW([
            {'params': other_params, 'lr': opt_config.lr},
            {'params': backbone_params, 'lr': opt_config.lr}  # Lower LR for FPN if necessary
        ], weight_decay= opt_config.weight_decay)
        
        # Configure scheduler
        if sch_config.type == "reduce_lr_on_plateau":
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='max',
                    factor=0.5,
                    patience=5,
                ),
                'monitor': 'val/map',
                'interval': 'epoch',
                'frequency': 1,
                # 'verbose': Trueq
            }
        elif sch_config.typ == "cosine":
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.config.trainer.max_epochs,
                    eta_min=opt_config.lr * 0.01
                ),
                'interval': 'epoch'
            }
        else:
            # Linear warmup + constant
            def lr_lambda(step):
                if step < sch_config.warmup_steps:
                    return step / sch_config.warmup_steps
                return 1.0
            
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
                'interval': 'step'
            }
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def draw_boxes(self, image, boxes, labels, scores, id2label):
        """Draws bounding boxes on a PIL image."""
        draw = ImageDraw.Draw(image)
        threshold = self.config.model.draw_threshold
        for box, label, score in zip(boxes, labels, scores):
            if score < threshold:
                continue
            box = box.tolist()
            label_id = label.item()
            color = self.PALETTE[label_id % len(self.PALETTE)]
            
            draw.rectangle(box, outline=color, width=2)
            
            label_text = f"{id2label[label_id]}: {score:.2f}"
            text_box = draw.textbbox((box[0], box[1]), label_text, font=self.font)
            draw.rectangle(text_box, fill=color)
            draw.text((box[0], box[1]), label_text, fill="white", font=self.font)
            
        return image

    def _visualize_batch(self, save_dir, post_processed_outputs, pixel_values, labels, counter):
        """Saves visualizations for a batch."""
        os.makedirs(save_dir, exist_ok=True)
        id2label = self.model.config.id2label
        max_samples = self.config.checkpointing.visualize_samples
        # Get mean and std from the processor to un-normalize
        mean = torch.tensor(self.image_processor.image_mean, device=pixel_values.device).view(1, 3, 1, 1)
        std = torch.tensor(self.image_processor.image_std, device=pixel_values.device).view(1, 3, 1, 1)

        # Un-normalize the entire batch at once (faster)
        # 1. Multiply by std, 2. Add mean, 3. Clamp to [0, 1] range
        unnormalized_images = torch.clamp((pixel_values * std) + mean, 0, 1)
        for i in range(len(labels)):
            if counter >= max_samples:
                break
            
            # Get original image path from COCO GT
            image_id = int(labels[i]["image_id"].item())
            image_tensor = unnormalized_images[i]
            image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            image = Image.fromarray(image_np)

            img_info = self.val_coco_gt.loadImgs(image_id)[0]
            # img_path = os.path.join(self.config['checkpointing']['val_image_dir'], img_info['file_name'])
            # breakpoint()
            # try:
            #     image = Image.open(img_path).convert("RGB")
            # except FileNotFoundError:
            #     print(f"Warning: Could not find image {img_path} for visualization.")
            #     continue
                
            # Get predictions for this image
            preds = post_processed_outputs[i]
            
            # Draw boxes
            drawn_image = self.draw_boxes(
                image, 
                preds['boxes'], 
                preds['labels'], 
                preds['scores'], 
                id2label
            )
            
            # Save image
            save_path = os.path.join(save_dir, img_info['file_name'])
            drawn_image.save(save_path)
            
            counter += 1
        
        return counter