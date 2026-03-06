import os
import sys
import importlib
import torch
import torch.nn as nn
import torch.distributed as dist
import pytorch_lightning as pl
import numpy as np
import cv2
from pathlib import Path
from pycocotools.cocoeval import COCOeval
from PIL import Image, ImageDraw, ImageFont

from utils.distributed_utils import rank_zero_print
# from models.yolov5.utils.augmentations import normalize, denormalize

def _import_from_yolo_repo(repo_path: str, module_name: str):
    repo = Path(repo_path).expanduser().resolve()
    if not repo.exists():
        raise FileNotFoundError(f"YOLOv5 repo not found at: {repo}.")
    repo_str = str(repo)
    original_path = sys.path.copy()
    original_modules = {}
    try:
        sys.path = [p for p in sys.path if p not in ("", ".", str(Path.cwd()))]
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        for key in list(sys.modules.keys()):
            if key.startswith(("models", "utils", "detect", "export")):
                original_modules[key] = sys.modules.pop(key)
        return importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Could not import '{module_name}' from YOLOv5.\n{e}")
    finally:
        sys.path = original_path

def _ensure_repo_import(repo_path: str):
    repo = Path(repo_path).expanduser().resolve()
    if not repo.exists():
        raise FileNotFoundError(f"YOLOv5 repo not found at: {repo}.")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo

class YOLOv5LightningModule(pl.LightningModule):
    def __init__(self, config, yolo_repo_path: str, model_to_coco: dict, val_coco_gt=None, test_coco_gt=None):
        super().__init__()
        self.config = config
        self.yolo_repo_path = str(_ensure_repo_import(yolo_repo_path))
        self.model_to_coco = {int(k): int(v) for k, v in model_to_coco.items()}
        self.val_coco_gt = val_coco_gt
        self.test_coco_gt = test_coco_gt

        # Safe Imports
        yolo_module = _import_from_yolo_repo(self.yolo_repo_path, "models.yolo")
        utils_loss = _import_from_yolo_repo(self.yolo_repo_path, "utils.loss")
        utils_general = _import_from_yolo_repo(self.yolo_repo_path, "utils.general")
        utils_aug = _import_from_yolo_repo(self.yolo_repo_path, "utils.augmentations")
        
        self._ComputeLossClass = utils_loss.ComputeLoss
        self._non_max_suppression = utils_general.non_max_suppression
        self._scale_boxes = utils_general.scale_boxes
        self.normalize = utils_aug.normalize
        self.denormalize = utils_aug.denormalize


        model_cfg = self.config.model.yolov5
        nc = len(self.config.model.label_map)
        model_def = model_cfg.model_cfg
        if not os.path.isabs(model_def):
            model_def = str(Path(self.yolo_repo_path) / model_def)
            
        model = yolo_module.Model(model_def, ch=3, nc=nc)
        
        if model_cfg.weights:
            weights_path = model_cfg.weights if str(model_cfg.weights).endswith('.pt') else f"{model_cfg.weights}.pt"
            if os.path.exists(weights_path):
                ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
                csd = ckpt['model'].float().state_dict()
                csd = {k: v for k, v in csd.items() if model.state_dict()[k].shape == v.shape}
                model.load_state_dict(csd, strict=False)
                rank_zero_print(f"[INFO] Successfully loaded pretrained weights from: {weights_path}")
            else:
                rank_zero_print(f"[INFO] Pretrained weights not found at: {weights_path}")

        model.hyp = dict(model_cfg.hyp)
        self.model = model
        self._compute_loss = None
        
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
            self.font = ImageFont.truetype("arial.ttf", 17)
        except IOError:
            self.font = ImageFont.load_default()
            
        self.val_viz_counter = 0
        self.test_viz_counter = 0

        self.filename_to_img_id = {}
        if self.val_coco_gt is not None:
            for img_id, img_info in self.val_coco_gt.imgs.items():
                self.filename_to_img_id[Path(img_info['file_name']).name] = img_id
        if self.test_coco_gt is not None:
            for img_id, img_info in self.test_coco_gt.imgs.items():
                self.filename_to_img_id[Path(img_info['file_name']).name] = img_id
        self.save_hyperparameters(ignore=['val_coco_gt', 'test_coco_gt', 'model'])

    def setup(self, stage=None):
        pass

    def forward(self, x):
        return self.model(x)

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

    def _get_ema_model(self):
        """Safely extracts the EMA model instance from PL callbacks."""
        for cb in self.trainer.callbacks:
            if type(cb).__name__ == 'EMACallback' and hasattr(cb, 'ema_model'):
                return cb.ema_model.module if hasattr(cb.ema_model, 'module') else cb.ema_model
        return None

    def training_step(self, batch, batch_idx):
        if self._compute_loss is None:
            self._compute_loss = self._ComputeLossClass(self.model)
        # breakpoint()
        imgs, targets, paths, _ = batch
        # imgs = imgs.float() / 255.0  # Normalize to 0.0 - 1.0
        
        imgs = self.normalize(imgs)
        
        pred = self.model(imgs)
        loss, loss_items = self._compute_loss(pred, targets)
        
        batch_size = imgs.shape[0]
        self.log("train/loss", loss, batch_size=batch_size, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/box_loss", loss_items[0], batch_size=batch_size, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/obj_loss", loss_items[1], batch_size=batch_size, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/cls_loss", loss_items[2], batch_size=batch_size, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def _shared_eval_step(self, batch, batch_idx, model_instance, prefix="val", visualize=True):
        """Executes forward pass, NMS, and evaluation scaling. Can run on regular or EMA model."""
        imgs, targets, paths, shapes = batch
        # breakpoint()
        # imgs = imgs.float() / 255.0 

        imgs = self.normalize(imgs)
        
        preds, _ = model_instance(imgs)
        
        preds = self._non_max_suppression(
            preds, 
            conf_thres=self.config.model.detection_threshold, 
            iou_thres=self.config.model.yolov5.iou_threshold, 
            labels=[], multi_label=True, agnostic=False, 
            max_det=self.config.model.max_detections
        )
        
        results = []
        image_ids = []
        post_processed_outputs = []
        
        for i, det in enumerate(preds):
            filename = Path(paths[i]).name
            if filename in self.filename_to_img_id:
                img_id = self.filename_to_img_id[filename]
            else:
                # Fallback if somehow not in COCO GT
                try: 
                    img_id = int(''.join(filter(str.isdigit, filename)))
                except ValueError: 
                    img_id = hash(filename) % 1000000

            image_ids.append(img_id)
            orig_shape = shapes[i][0]
            
            img_preds = {'boxes': [], 'scores': [], 'labels': []}
            
            if len(det):
                det[:, :4] = self._scale_boxes(imgs.shape[2:], det[:, :4], orig_shape).round()
                
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = [float(coord) for coord in xyxy]
                    w, h = x2 - x1, y2 - y1
                    
                    coco_cat_id = self.model_to_coco.get(int(cls), int(cls))
                    
                    results.append({
                        "image_id": img_id,
                        "category_id": coco_cat_id,
                        "bbox": [x1, y1, w, h], 
                        "score": float(conf),
                    })
                    img_preds['boxes'].append([x1, y1, x2, y2])
                    img_preds['scores'].append(float(conf))
                    img_preds['labels'].append(int(cls))
                    
            post_processed_outputs.append(img_preds)
            
        if visualize:
            viz_counter = self.val_viz_counter if prefix == "val" else self.test_viz_counter
            if (self.current_epoch % self.config.checkpointing.visualize_every_n_epochs == 0) and \
               (viz_counter < self.config.checkpointing.visualize_samples or self.config.checkpointing.visualize_samples == -1):
                
                save_dir = os.path.join(self.config.checkpointing.save_dir, self.config.checkpointing.visualization_dir, f"epoch_{(self.current_epoch+1):03d}", prefix)
                new_counter = self._visualize_batch(save_dir, post_processed_outputs, imgs, paths, shapes, viz_counter, prefix)
                if prefix == "val": self.val_viz_counter = new_counter
                else: self.test_viz_counter = new_counter

        return {"predictions": results, "image_ids": image_ids}

    def validation_step(self, batch, batch_idx):
        # Regular Model
        res = self._shared_eval_step(batch, batch_idx, model_instance=self.model, prefix="val", visualize=True)
        self.validation_step_outputs.append(res)
        
        # EMA Model
        if hasattr(self.config.model, 'ema') and self.config.model.ema.enabled:
            ema_model = self._get_ema_model()
            if ema_model is not None:
                # visualize=False to avoid saving duplicate visualizations
                res_ema = self._shared_eval_step(batch, batch_idx, model_instance=ema_model, prefix="val", visualize=False)
                self.validation_step_outputs_ema.append(res_ema)
                
        return res

    def test_step(self, batch, batch_idx):
        # Regular Model
        res = self._shared_eval_step(batch, batch_idx, model_instance=self.model, prefix="test", visualize=True)
        self.test_step_outputs.append(res)
        
        # EMA Model
        if hasattr(self.config.model, 'ema') and self.config.model.ema.enabled:
            ema_model = self._get_ema_model()
            if ema_model is not None:
                res_ema = self._shared_eval_step(batch, batch_idx, model_instance=ema_model, prefix="test", visualize=False)
                self.test_step_outputs_ema.append(res_ema)
                
        return res

    def _gather_all_outputs(self, local_outputs):
        if not dist.is_available() or not dist.is_initialized(): return local_outputs
        world_size = dist.get_world_size()
        if world_size <= 1: return local_outputs
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_outputs)
        return [item for rank_outputs in gathered for item in rank_outputs]

    def on_validation_epoch_end(self):
        # Process regular outputs
        self._compute_and_log_metrics(self.validation_step_outputs, self.val_coco_gt, prefix="val", suffix="")
        self.validation_step_outputs.clear()
        
        # Process EMA outputs
        if hasattr(self, 'validation_step_outputs_ema') and len(self.validation_step_outputs_ema) > 0:
            self._compute_and_log_metrics(self.validation_step_outputs_ema, self.val_coco_gt, prefix="val", suffix="_ema")
            self.validation_step_outputs_ema.clear()
            
        self.val_viz_counter = 0

    def on_test_epoch_end(self):
        # Process regular outputs
        self._compute_and_log_metrics(self.test_step_outputs, self.test_coco_gt, prefix="test", suffix="")
        self.test_step_outputs.clear()
        
        # Process EMA outputs
        if hasattr(self, 'test_step_outputs_ema') and len(self.test_step_outputs_ema) > 0:
            self._compute_and_log_metrics(self.test_step_outputs_ema, self.test_coco_gt, prefix="test", suffix="_ema")
            self.test_step_outputs_ema.clear()
            
        self.test_viz_counter = 0

    def _compute_and_log_metrics(self, step_outputs, coco_gt, prefix, suffix=""):
        all_outputs = self._gather_all_outputs(step_outputs)
        metrics = {}
        
        if self.trainer.is_global_zero:
            predictions, image_ids = [], []
            for out in all_outputs:
                predictions.extend(out["predictions"])
                image_ids.extend(out["image_ids"])
                
            if len(predictions) > 0 and coco_gt is not None:
                try:
                    coco_dt = coco_gt.loadRes(predictions)
                    coco_evaluator = COCOeval(coco_gt, coco_dt, "bbox")
                    coco_evaluator.params.imgIds = list(set(image_ids))
                    coco_evaluator.params.maxDets = [1, 10, self.config.model.max_detections]
                    
                    coco_evaluator.evaluate()
                    coco_evaluator.accumulate()
                    model_type = "EMA Model" if "_ema" in suffix else "Standard Model"
                    print(f"\n{'='*50}")
                    print(f" COCO EVALUATION: {prefix.upper()} | {model_type}")
                    print(f"{'='*50}")
                    coco_evaluator.summarize()
                    
                    stats = coco_evaluator.stats
                    metrics = {
                        'map': round(stats[0], 4), 'map_50': round(stats[1], 4), 'map_75': round(stats[2], 4),
                        'mar_1': round(stats[6], 4), 'mar_10': round(stats[7], 4), f'mar_{self.config.model.max_detections}': round(stats[8], 4)
                    }
                    
                    if hasattr(coco_evaluator, 'eval') and 'precision' in coco_evaluator.eval:
                        precisions, recalls = coco_evaluator.eval['precision'], coco_evaluator.eval['recall']
                        
                        num_images = len(coco_evaluator.params.imgIds)
                        labels_per_class = {}
                        total_labels = 0
                        for ann in coco_gt.dataset.get('annotations', []):
                            if ann['image_id'] in coco_evaluator.params.imgIds:
                                c_id = ann['category_id']
                                labels_per_class[c_id] = labels_per_class.get(c_id, 0) + 1
                                total_labels += 1
                                
                        table_rows = []
                        
                        for i, catId in enumerate(coco_evaluator.params.catIds):
                            cat_name = self.config.model.label_map.get(int(catId), f"class_{catId}")
                            
                            s, s_50, r = precisions[:, :, i, 0, -1], precisions[0, :, i, 0, -1], recalls[:, i, 0, -1]
                            if len(s[s > -1]) > 0: metrics[f'map_{cat_name}'] = round(float(np.mean(s[s > -1])), 4)
                            if len(s_50[s_50 > -1]) > 0: metrics[f'map_50_{cat_name}'] = round(float(np.mean(s_50[s_50 > -1])), 4)
                            if len(r[r > -1]) > 0: metrics[f'mar_{cat_name}'] = round(float(np.mean(r[r > -1])), 4)

                            best_p, best_r = 0.0, 0.0
                            if len(s_50[s_50 > -1]) > 0:
                                p_curve = precisions[0, :, i, 0, -1]
                                r_curve = np.linspace(0.0, 1.0, len(p_curve))
                                valid_mask = p_curve > -1
                                if np.any(valid_mask):
                                    valid_p = p_curve[valid_mask]
                                    valid_r = r_curve[valid_mask]
                                    denominator = valid_p + valid_r + 1e-16
                                    f1_curve = 2 * valid_p * valid_r / denominator
                                    best_idx = np.argmax(f1_curve)
                                    best_p = valid_p[best_idx]
                                    best_r = valid_r[best_idx]
                                    
                            table_rows.append({
                                "Class": cat_name,
                                "Images": num_images,
                                "Labels": labels_per_class.get(catId, 0),
                                "P": best_p,
                                "R": best_r,
                                "mAP@.5": metrics.get(f"map_50_{cat_name}", 0.0),
                                "mAP@.5:.95": metrics.get(f"map_{cat_name}", 0.0)
                            })
                            
                        avg_p = np.mean([row["P"] for row in table_rows]) if table_rows else 0.0
                        avg_r = np.mean([row["R"] for row in table_rows]) if table_rows else 0.0
                        
                        all_row = {
                            "Class": "all",
                            "Images": num_images,
                            "Labels": total_labels,
                            "P": avg_p,
                            "R": avg_r,
                            "mAP@.5": metrics.get("map_50", 0.0),
                            "mAP@.5:.95": metrics.get("map", 0.0)
                        }
                        
                        print(f"\n{prefix.capitalize()} set performance" + (" (EMA)" if "_ema" in suffix else ""))
                        header = f"{'Class':<14}{'Images':<8}{'Labels':<11}{'P':<8}{'R':<9}{'mAP@.5':<11}{'mAP@.5:.95':<11}"
                        print(header)
                        
                        def format_row(r_dict):
                            return f"{r_dict['Class']:<14}{r_dict['Images']:<8}{r_dict['Labels']:<11}{r_dict['P']:<8.3f}{r_dict['R']:<9.3f}{r_dict['mAP@.5']:<11.3f}{r_dict['mAP@.5:.95']:<11.3f}"
                        
                        print(format_row(all_row))
                        for r_dict in table_rows:
                            print(format_row(r_dict))
                        print()
                        
                except Exception as e:
                    print(f"Error computing COCO metrics: {e}")
            else:
                metrics = {'map': 0.0, 'map_50': 0.0}

        if dist.is_available() and dist.is_initialized():
            obj_list = [metrics]
            dist.broadcast_object_list(obj_list, src=0)
            metrics = obj_list[0]

        for k, v in metrics.items():
            # Apply suffix (e.g., '_ema') to the logged key
            # Ensures ModelCheckpoint finds 'val/map_ema' correctly
            log_key = f"{prefix}/{k}{suffix}"
            
            # Show standard mAP on the progress bar, but ignore class-wise or EMA on the bar to avoid clutter
            show_on_prog_bar = (k == 'map' and suffix == "")
            self.log(log_key, v, prog_bar=show_on_prog_bar, sync_dist=True)

            if k == 'map':
                alias_key = f"{prefix}_{k}{suffix}" 
                self.log(alias_key, v, prog_bar=False, sync_dist=True)

    def draw_boxes(self, image, boxes, labels, scores=None, color_override=None):
        draw = ImageDraw.Draw(image)
        threshold = self.config.model.draw_threshold
        id2label = self.config.model.label_map
        
        for i in range(len(boxes)):
            score = scores[i] if scores is not None else 1.0
            if score < threshold: continue
            
            box = boxes[i]
            label_id = int(labels[i])
            color = color_override or self.PALETTE[label_id % len(self.PALETTE)]
            
            draw.rectangle(box, outline=color, width=3)
            class_name = id2label.get(label_id, str(label_id))
            label_text = f"{class_name}: {score:.2f}" if scores else class_name
            
            text_box = draw.textbbox((box[0], box[1]), label_text, font=self.font)
            draw.rectangle(text_box, fill=color)
            draw.text((box[0], box[1]), label_text, fill="white", font=self.font)
        return image

    def _visualize_batch(self, save_dir, post_processed_outputs, imgs, paths, shapes, counter, prefix):
        os.makedirs(save_dir, exist_ok=True)
        max_samples = self.config.checkpointing.visualize_samples
        coco_gt = self.test_coco_gt if prefix == "test" else self.val_coco_gt
        if counter == 0:
            self.print(f"[VIZ] Saving visualizations to: {save_dir}")
            self.print(f"[VIZ] Max samples: {max_samples}")
            if max_samples == -1 or max_samples == float("inf"):
                self.print(
                    f"[VIZ] WARNING: Unlimited visualization enabled for {prefix}. "
                    "This can be very slow on large datasets."
                )
        
        for i in range(len(paths)):
            if max_samples != -1 and counter >= max_samples: break
            
            orig_h, orig_w = shapes[i][0]
            filename = Path(paths[i]).name
            try: img_id = int(''.join(filter(str.isdigit, filename)))
            except ValueError: img_id = hash(filename) % 1000000
            
            image_np = (imgs[i].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            resized_image_np = cv2.resize(image_np, (int(orig_w), int(orig_h)), interpolation=cv2.INTER_LINEAR)
            image = Image.fromarray(resized_image_np)

            gt_labels = []
            if coco_gt and img_id in coco_gt.imgs:
                gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
                gt_boxes = []
                for ann in gt_anns:
                    x, y, w, h = ann['bbox']
                    gt_boxes.append([x, y, x + w, y + h])
                    gt_labels.append(ann['category_id'])
                image = self.draw_boxes(image, gt_boxes, gt_labels, color_override=(0, 255, 0))

            preds = post_processed_outputs[i]
            image = self.draw_boxes(image, preds['boxes'], preds['labels'], preds['scores'], color_override=(255, 0, 0))
            
            rank = self.global_rank if dist.is_available() and dist.is_initialized() else 0
            new_filename = f"rank{rank}_{filename}"
            image.save(os.path.join(save_dir, new_filename))
            counter += 1
            if counter % 500 == 0:
                self.print(f"[VIZ] {prefix.upper()} progress: saved {counter} images...")
        return counter

    def configure_optimizers(self):
        g0, g1, g2 = [], [], []
        for v in self.model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter): g2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d): g0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter): g1.append(v.weight)

        opt_cfg = self.config.model.yolov5.optimizer
        if opt_cfg.type == 'sgd': optimizer = torch.optim.SGD(g0, lr=opt_cfg.lr, momentum=opt_cfg.momentum, nesterov=opt_cfg.nesterov)
        else: optimizer = torch.optim.Adam(g0, lr=opt_cfg.lr, betas=(opt_cfg.momentum, 0.999))
            
        optimizer.add_param_group({'params': g1, 'weight_decay': opt_cfg.weight_decay})
        optimizer.add_param_group({'params': g2})
        
        lf = lambda x: (1 - x / self.trainer.max_epochs) * (1.0 - self.config.model.yolov5.hyp.lrf) + self.config.model.yolov5.hyp.lrf
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

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
