import os
import cv2
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import torch
import pytorch_lightning as pl

from utils.coco_eval_utils import (
    to_cpu_device,
    convert_preds_to_coco,
    gather_outputs_across_processes,
    broadcast_object,
    compute_coco_metrics,
)


class RFDETRLightningModule(pl.LightningModule):
    """PyTorch Lightning module for RF-DETR training/evaluation."""

    def __init__(self, model, criterion, postprocess, config, val_coco_gt=None, test_coco_gt=None):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.postprocess = postprocess
        self.config = config
        self.val_coco_gt = val_coco_gt
        self.test_coco_gt = test_coco_gt

        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        if hasattr(self.config.model, 'ema') and self.config.model.ema.enabled:
            self.validation_step_outputs_ema = []
            self.test_step_outputs_ema = []

        # Visualization setup
        self.val_viz_counter = 0
        self.test_viz_counter = 0
        self.PALETTE = [
            (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
            (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
            (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42)
        ]
        try:
            self.font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            self.font = ImageFont.load_default()

    def forward(self, samples, targets=None):
        return self.model(samples, targets)

    def _move_targets(self, targets):
        return [{k: v.to(self.device) for k, v in target.items()} for target in targets]

    def _compute_loss(self, outputs, targets):
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
        return loss, loss_dict, weight_dict

    def _log_loss_dict(self, split: str, loss_dict, weight_dict):
        """
        Log all loss terms from RF-DETR criterion.
        For weighted terms we log both unscaled and scaled values.
        """
        for key, value in loss_dict.items():
            self.log(f"{split}/{key}_unscaled", value, on_step=False, on_epoch=True, sync_dist=True)
            if key in weight_dict:
                self.log(f"{split}/{key}", value * weight_dict[key], on_step=False, on_epoch=True, sync_dist=True)
            else:
                self.log(f"{split}/{key}", value, on_step=False, on_epoch=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        samples, targets = batch
        samples = samples.to(self.device)
        targets = self._move_targets(targets)

        outputs = self.model(samples, targets)
        loss, loss_dict, weight_dict = self._compute_loss(outputs, targets)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self._log_loss_dict("train", loss_dict, weight_dict)

        return loss

    def _collect_batch_predictions(self, outputs, targets):
        orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        post = self.postprocess(outputs, orig_sizes)
        post = [to_cpu_device(pred) for pred in post]
        result_map = {
            int(target["image_id"].item()): pred for target, pred in zip(targets, post)
        }
        image_ids = [int(target["image_id"].item()) for target in targets]
        return result_map, image_ids

    def on_validation_epoch_start(self):
        """Reset validation visualization counter."""
        self.val_viz_counter = 0

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        samples, targets = batch
        samples = samples.to(self.device)
        targets = self._move_targets(targets)

        outputs = self.model(samples)
        loss, loss_dict, weight_dict = self._compute_loss(outputs, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self._log_loss_dict("val", loss_dict, weight_dict)

        predictions, image_ids = self._collect_batch_predictions(outputs, targets)
        self.validation_step_outputs.append({"predictions": predictions, "image_ids": image_ids})
        
        # EMA validation
        from utils.ema import EMACallback
        ema_callback = next((cb for cb in self.trainer.callbacks if isinstance(cb, EMACallback)), None)
        if ema_callback and ema_callback.ema_model:
            ema_outputs = ema_callback.ema_model.module(samples)
            ema_predictions, ema_image_ids = self._collect_batch_predictions(ema_outputs, targets)
            self.validation_step_outputs_ema.append({"predictions": ema_predictions, "image_ids": ema_image_ids})
            
        # Visualization
        if (self.current_epoch) % self.config.checkpointing.visualize_every_n_epochs == 0 and \
           self.trainer.is_global_zero and \
           (self.val_viz_counter < self.config.checkpointing.visualize_samples or self.config.checkpointing.visualize_samples == -1):
            
            save_dir = os.path.join(
                self.config.checkpointing.save_dir,
                self.config.checkpointing.visualization_dir, 
                f"epoch_{(self.current_epoch+1):03d}", 
                "val"
            )
            # Prefer EMA predictions if available
            viz_preds = ema_predictions if (ema_callback and ema_callback.ema_model) else predictions
            
            self.val_viz_counter = self._visualize_batch(
                save_dir, 
                viz_preds, 
                samples,
                targets, 
                self.val_viz_counter,
                split="val"
            )

        return {"predictions": predictions, "image_ids": image_ids}

    def on_validation_epoch_end(self):
        all_outputs = gather_outputs_across_processes(self.validation_step_outputs)
        metrics = {}
        if self.trainer.is_global_zero:
            val_predictions = []
            val_image_ids = []
            for batch_out in all_outputs:
                val_predictions.extend(convert_preds_to_coco(batch_out["predictions"]))
                val_image_ids.extend(batch_out["image_ids"])

            if len(val_predictions) > 0:
                metrics = compute_coco_metrics(
                    coco_gt=self.val_coco_gt,
                    predictions=val_predictions,
                    image_ids=list(set(val_image_ids)),
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

    def on_test_epoch_start(self):
        """Reset test visualization counter."""
        self.test_viz_counter = 0

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        samples, targets = batch
        samples = samples.to(self.device)
        targets = self._move_targets(targets)

        outputs = self.model(samples)
        predictions, image_ids = self._collect_batch_predictions(outputs, targets)
        self.test_step_outputs.append({"predictions": predictions, "image_ids": image_ids})
        
        # EMA test
        from utils.ema import EMACallback
        ema_callback = next((cb for cb in self.trainer.callbacks if isinstance(cb, EMACallback)), None)
        if ema_callback and ema_callback.ema_model:
            ema_outputs = ema_callback.ema_model.module(samples)
            ema_predictions, ema_image_ids = self._collect_batch_predictions(ema_outputs, targets)
            self.test_step_outputs_ema.append({"predictions": ema_predictions, "image_ids": ema_image_ids})
        
        # Visualization
        if self.config.checkpointing.visualize_samples == -1:
             self.config.checkpointing.visualize_samples = float('inf')

        if self.trainer.is_global_zero and \
           (self.test_viz_counter < self.config.checkpointing.visualize_samples):
            
            save_dir = os.path.join(
                self.config.checkpointing.save_dir,
                self.config.checkpointing.visualization_dir, 
                "test"
            )
            # Prefer EMA predictions if available
            viz_preds = ema_predictions if (ema_callback and ema_callback.ema_model) else predictions
            
            self.test_viz_counter = self._visualize_batch(
                save_dir, 
                viz_preds, 
                samples,
                targets, 
                self.test_viz_counter,
                split="test"
            )

        return {"predictions": predictions, "image_ids": image_ids}

    def on_test_epoch_end(self):
        all_outputs = gather_outputs_across_processes(self.test_step_outputs)
        metrics = {}
        if self.trainer.is_global_zero:
            test_predictions = []
            test_image_ids = []
            for batch_out in all_outputs:
                test_predictions.extend(convert_preds_to_coco(batch_out["predictions"]))
                test_image_ids.extend(batch_out["image_ids"])

            if len(test_predictions) > 0:
                metrics = compute_coco_metrics(
                    coco_gt=self.test_coco_gt,
                    predictions=test_predictions,
                    image_ids=list(set(test_image_ids)),
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

    @torch.no_grad()
    def predict_batch(self, samples, score_threshold=0.25):
        """Simple inference helper used by external scripts."""
        self.model.eval()
        if hasattr(samples, "to"):
            samples = samples.to(self.device)
        outputs = self.model(samples)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if "pred_boxes" in outputs:
            # RF-DETR postprocess expects target sizes.
            if hasattr(samples, "tensors"):
                bsz = samples.tensors.shape[0]
                h, w = samples.tensors.shape[-2:]
            else:
                bsz = samples.shape[0]
                h, w = samples.shape[-2:]
            target_sizes = torch.tensor([[h, w]] * bsz, device=self.device)
            preds = self.postprocess(outputs, target_sizes)
            filtered = []
            for pred in preds:
                keep = pred["scores"] >= score_threshold
                filtered.append({k: v[keep] for k, v in pred.items()})
            return filtered
        return outputs

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

    def _visualize_batch(self, save_dir, predictions_map, samples, targets, counter, split="val"):
        """Saves visualizations for a batch showing both GT and Predictions."""
        os.makedirs(save_dir, exist_ok=True)
        max_samples = self.config.checkpointing.visualize_samples
        
        if counter == 0:
            self.print(f"[VIZ] Saving visualizations to: {save_dir}")
            self.print(f"[VIZ] Max samples: {max_samples}")

        # Determine which COCO GT to use based on stage
        coco_gt = self.test_coco_gt if split == "test" else self.val_coco_gt
        
        # Use draw_threshold for visualization
        viz_threshold = float(self.config.model.draw_threshold)
        
        # Un-normalize
        # Check if samples is NestedTensor or Tensor
        if hasattr(samples, 'tensors'):
            pixel_values = samples.tensors
        else:
            pixel_values = samples
            
        mean = torch.tensor([0.485, 0.456, 0.406], device=pixel_values.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pixel_values.device).view(1, 3, 1, 1)
        unnormalized_images = torch.clamp((pixel_values * std) + mean, 0, 1)

        for i, target in enumerate(tqdm(targets, desc="Visualizing Batch", leave=False)):
            if max_samples != -1 and counter >= max_samples:
                break
                
            image_id = int(target["image_id"].item())
            image_tensor = unnormalized_images[i]
            
            # Use original size if available in target, else tensor size
            if "orig_size" in target:
                orig_h, orig_w = target["orig_size"].tolist()
            else:
                orig_h, orig_w = image_tensor.shape[1], image_tensor.shape[2]
                
            image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            resized_image_np = cv2.resize(image_np, (int(orig_w), int(orig_h)), interpolation=cv2.INTER_LINEAR)
            image = Image.fromarray(resized_image_np)
            
             # Get image metadata from COCO GT for filename
            img_info = {'file_name': f"image_{image_id}.png"}
            if coco_gt:
                try:
                    loaded_imgs = coco_gt.loadImgs(image_id)
                    if loaded_imgs:
                         img_info = loaded_imgs[0]
                except (IndexError, AttributeError, KeyError):
                    pass

            # --- 1. GT ---
            gt_labels = []
            if coco_gt:
                try:
                    gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=image_id))
                    gt_boxes = []
                    for ann in gt_anns:
                         x, y, w, h = ann['bbox']
                         gt_boxes.append([x, y, x + w, y + h])
                         gt_labels.append(ann['category_id'])
                    
                    image = self.draw_boxes(
                        image, 
                        gt_boxes, 
                        gt_labels, 
                        scores=None, 
                        id2label=self.config.model.label_map,
                        color_override=(0, 255, 0), 
                        label_prefix=""
                    )
                except Exception:
                    pass

            # --- 2. Preds ---
            pred_class_names = []
            if image_id in predictions_map:
                preds = predictions_map[image_id]
                
                if 'boxes' in preds and len(preds['boxes']) > 0:
                     # Filter
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
                        color_override=(255, 0, 0), 
                        label_prefix="",
                        threshold_override=viz_threshold
                    )

            # --- 3. Counts & Filename ---
            from collections import Counter
            label_map = self.config.model.label_map
            
            gt_counts = Counter([label_map.get(int(l)) or label_map.get(str(l)) or str(l) for l in gt_labels])
            pred_counts = Counter(pred_class_names)
            
            draw = ImageDraw.Draw(image)
            text_x = image.width - 200 
            text_y = 10
            line_height = 20
            
            all_classes = set(gt_counts.keys()) | set(pred_counts.keys())
            for cls_name in sorted(all_classes):
                text = f"{cls_name}: {pred_counts[cls_name]}/{gt_counts[cls_name]}"
                text_bbox = draw.textbbox((text_x, text_y), text, font=self.font)
                text_width = text_bbox[2] - text_bbox[0]
                actual_x = image.width - text_width - 10
                draw.text((actual_x + 1, text_y + 1), text, fill="black", font=self.font) 
                draw.text((actual_x, text_y), text, fill="white", font=self.font)
                text_y += line_height

            detected_classes = sorted(list(set(pred_class_names)))
            if detected_classes:
                class_str = "_".join(detected_classes)
                prefix = f"image_{class_str}_"
            else:
                prefix = "image_no_detections_"
            
            original_filename = os.path.basename(img_info['file_name'])
            new_filename = f"{prefix}{original_filename}"
            new_filename = new_filename.replace("image_image_", "image_")
            
            save_path = os.path.join(save_dir, new_filename)
            image.save(save_path)
            counter += 1
            
        return counter

    def configure_optimizers(self):
        opt_config = self.config.optimizer.optimizer
        sch_config = self.config.scheduler

        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(opt_config.lr),
            weight_decay=float(opt_config.weight_decay),
        )

        if sch_config.type == "reduce_lr_on_plateau":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    factor=float(sch_config.factor),
                    patience=int(sch_config.patience),
                ),
                "monitor": "val/map",
                "interval": "epoch",
                "frequency": 1,
            }
        elif sch_config.type == "cosine":
            total_steps = max(1, self.trainer.estimated_stepping_batches)
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=total_steps,
                    eta_min=float(sch_config.eta_min),
                ),
                "interval": "step",
            }
        elif sch_config.type == "onecycle":
            total_steps = max(1, self.trainer.estimated_stepping_batches)
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=float(opt_config.lr),
                    total_steps=total_steps,
                    pct_start=float(sch_config.pct_start),
                    anneal_strategy="cos",
                    div_factor=25.0,
                    final_div_factor=1e3,
                ),
                "interval": "step",
            }
        else:
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=max(1, int(getattr(sch_config, "step_size", 10))),
                    gamma=float(getattr(sch_config, "gamma", 0.1)),
                ),
                "interval": "epoch",
            }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
