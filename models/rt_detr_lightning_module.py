import re
import torch
import torch.distributed as dist
import os
import cv2
import pytorch_lightning as pl
from collections import Counter
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
from PIL import Image, ImageDraw, ImageFont
from models.custom_rt_detr_with_dinov2_backbone import (
    RTDetrV2ForObjectDetectionWithCustomBackbone,
)
from utils.distributed_utils import rank_print
from utils.sahi_eval import run_sahi_sliced_eval
from utils.coco_eval_utils import compute_coco_metrics


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

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": boxes[k],
                    "score": scores[k],
                }
                for k in range(len(scores))
            ]
        )
    return coco_results


class RTDETRLightningModule(pl.LightningModule):
    """PyTorch Lightning Module for RT-DETR with DINOv2 backbone."""

    def __init__(
        self,
        model: RTDetrV2ForObjectDetectionWithCustomBackbone = None,
        image_processor=None,
        val_coco_gt=None,
        test_coco_gt=None,
        train_coco_gt=None,
        val_image_root=None,
        test_image_root=None,
        config=None,
    ):
        super().__init__()

        self.model = model
        # self.model.train() # REMOVED: Managed by train() override below
        self.image_processor = image_processor
        self.val_coco_gt = val_coco_gt
        self.test_coco_gt = test_coco_gt
        self.train_coco_gt = train_coco_gt
        self.val_image_root = val_image_root
        self.test_image_root = test_image_root

        # Allow loading checkpoints with extra keys (e.g. unused denoising weights)
        self.strict_loading = False

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
            (220, 20, 60),
            (119, 11, 32),
            (0, 0, 142),
            (0, 0, 230),
            (106, 0, 228),
            (0, 60, 100),
            (0, 80, 100),
            (0, 0, 70),
            (0, 0, 192),
            (250, 170, 30),
            (100, 170, 30),
            (220, 220, 0),
            (175, 116, 175),
            (250, 0, 30),
            (165, 42, 42),
        ]

        # --- Load font for labels ---
        try:
            self.font = ImageFont.truetype("arial.ttf", 17)
        except IOError:
            self.font = ImageFont.load_default()
        # debug setting
        self.debug_train_image_ids = set()
        self.config = config
        self.warmup_steps = self.config.scheduler.warmup_steps
        self.base_lr = self.config.optimizer.optimizer.lr
        self.validation_step_outputs = []
        self.test_step_outputs = []

        if hasattr(self.config.model, "ema") and self.config.model.ema.enabled:
            self.validation_step_outputs_ema = []
            self.test_step_outputs_ema = []

        self.validation_step_outputs_sliced = []
        self.test_step_outputs_sliced = []
        self.validation_step_outputs_sliced_ema = []
        self.test_step_outputs_sliced_ema = []

        self.save_hyperparameters(
            ignore=[
                "model",
                "image_processor",
                "val_coco_gt",
                "test_coco_gt",
                "train_coco_gt",
                "val_image_root",
                "test_image_root",
            ]
        )

    def forward(self, pixel_values, labels=None):
        """Forward pass."""
        return self.model(pixel_values=pixel_values, labels=labels)

    def train(self, mode: bool = True):
        """Override to keep frozen modules in eval mode."""
        super().train(mode)
        # breakpoint()
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
        labels = [
            {k: v.to(self.device) for k, v in sample.items()}
            for sample in batch["labels"]
        ]

        outputs = self.model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        # --- High-Stability NaN Sentinel ---
        # if not torch.isfinite(loss):
        #     rank_print("\n" + "!"*80 + f"\n🚨 [LOSS NAN] NaN/Inf detected in loss at epoch {self.current_epoch}!\n" + "!"*80)

        #     # Diagnostic logging
        #     # if hasattr(outputs, 'loss_dict'):
        #     #      rank_print(f"   -> Loss components: {outputs.loss_dict}")

        #     # Zero out gradient for this step to prevent model corruption
        #     loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))
        #     loss.requires_grad_(True) # Ensure it stays differentiable
        # # ----------------------------------

        for label_dict in labels:
            self.debug_train_image_ids.add(int(label_dict["image_id"].item()))

        # Log training loss
        self.log(
            "train/loss",
            loss,
            batch_size=batch_size,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # Log individual loss components if available
        if hasattr(outputs, "loss_dict"):
            for key, value in outputs.loss_dict.items():
                self.log(
                    f"train/{key}",
                    value,
                    batch_size=batch_size,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

        return loss

    def _get_image_path(self, image_id: int, split: str) -> str | None:
        """Resolve full image path using COCO annotations and configured roots."""
        coco_gt = self.val_coco_gt if split == "val" else self.test_coco_gt
        root = self.val_image_root if split == "val" else self.test_image_root

        if not coco_gt or not root:
            return None

        try:
            img_info = coco_gt.loadImgs(image_id)[0]
            file_name = img_info["file_name"]
            return os.path.join(root, file_name)
        except (IndexError, AttributeError, KeyError):
            return None

    def on_validation_epoch_start(self):
        """Reset validation visualization counter."""
        self.val_viz_counter = 0

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        # Collect true original image sizes from COCO metadata for accurate scaling
        batch_image_sizes = []
        for x in labels:
            image_id = int(x["image_id"].item())
            if self.val_coco_gt and image_id in self.val_coco_gt.imgs:
                img_info = self.val_coco_gt.imgs[image_id]
                batch_image_sizes.append([img_info["height"], img_info["width"]])
            else:
                batch_image_sizes.append(to_cpu_device(x["orig_size"]).numpy().tolist())

        eval_mode = self.config.get("eval_inference", {}).get("mode", "whole")

        # Whole Image Baseline Process (For Loss + Fallback)
        post_processed_outputs = []
        if eval_mode in ["whole", "both"]:
            outputs = self.model(pixel_values=pixel_values, labels=None)
            post_processed_outputs = self.image_processor.post_process_object_detection(
                outputs,
                threshold=self.config.model.detection_threshold,
                target_sizes=batch_image_sizes,
            )
            post_processed_outputs = [
                {k: to_cpu_device(v) for k, v in p.items()}
                for p in post_processed_outputs
            ]

            results = {
                int(t["image_id"].item()): out
                for t, out in zip(labels, post_processed_outputs)
            }
            image_ids = [int(t["image_id"].item()) for t in labels]
            self.validation_step_outputs.append(
                {"predictions": results, "image_ids": image_ids}
            )

            # if (self.current_epoch) % max(1, self.config.checkpointing.visualize_every_n_epochs) == 0 and \
            #    (self.val_viz_counter < self.config.checkpointing.visualize_samples or self.config.checkpointing.visualize_samples == -1):
            #     save_dir = os.path.join(self.config.checkpointing.save_dir, self.config.checkpointing.visualization_dir, f"epoch_{(self.current_epoch+1):03d}", "val_whole")
            #     self.val_viz_counter = self._visualize_batch(save_dir, post_processed_outputs, pixel_values, labels, batch_image_sizes, self.val_viz_counter)

        if eval_mode in ["sliced", "both"]:
            post_processed_outputs_sliced = []
            for i, target in enumerate(labels):
                image_id = int(target["image_id"].item())
                img_path = self._get_image_path(image_id, "val")

                if not img_path or not os.path.exists(img_path):
                    self.print(
                        f"[Val] WARNING: Cannot find image {img_path} for SAHI. Falling back to whole image."
                    )
                    # Fallback to whole image processing for this single image
                    single_pixel_values = pixel_values[i : i + 1]
                    single_output = self.model(
                        pixel_values=single_pixel_values, labels=None
                    )
                    single_post = self.image_processor.post_process_object_detection(
                        single_output,
                        threshold=self.config.model.detection_threshold,
                        target_sizes=[batch_image_sizes[i]],
                    )
                    post_processed_outputs.append(
                        {k: to_cpu_device(v) for k, v in single_post[0].items()}
                    )
                    continue

                # Define model-specific predict_fn
                def predict_fn(image_np):
                    # SAHI passes (H, W, 3) numpy array in RGB
                    pil_img = Image.fromarray(image_np)
                    # Processor expects PIL Image or list of PIL Images
                    inputs = self.image_processor(images=pil_img, return_tensors="pt")
                    pixel_val = inputs["pixel_values"].to(self.device)
                    # For half precision support if enabled
                    if next(self.model.parameters()).dtype == torch.float16:
                        pixel_val = pixel_val.half()
                    elif next(self.model.parameters()).dtype == torch.bfloat16:
                        pixel_val = pixel_val.bfloat16()

                    with torch.no_grad():
                        out = self.model(pixel_values=pixel_val, labels=None)

                    # Target size is the patch size, not the full image size
                    patch_size = [[image_np.shape[0], image_np.shape[1]]]
                    post_out = self.image_processor.post_process_object_detection(
                        out,
                        threshold=self.config.model.detection_threshold,
                        target_sizes=patch_size,
                    )[0]
                    return {k: to_cpu_device(v) for k, v in post_out.items()}

                # Run SAHI
                img_pil = Image.open(img_path).convert("RGB")
                sahi_cfg = self.config.get("eval_inference", {}).get("sahi", {})

                # Determine fallback slice size from model input
                input_size = (
                    self.config.data.model_input_size
                    if hasattr(self.config.data, "model_input_size")
                    else 640
                )

                preds = run_sahi_sliced_eval(
                    image=img_pil,
                    predict_fn=predict_fn,
                    sahi_config=sahi_cfg,
                    input_size=input_size,
                )
                post_processed_outputs_sliced.append(preds)

            results_sliced = {
                int(t["image_id"].item()): out
                for t, out in zip(labels, post_processed_outputs_sliced)
            }
            image_ids = [int(t["image_id"].item()) for t in labels]
            self.validation_step_outputs_sliced.append(
                {"predictions": results_sliced, "image_ids": image_ids}
            )

        # EMA validation
        from utils.ema import EMACallback

        ema_callback = next(
            (cb for cb in self.trainer.callbacks if isinstance(cb, EMACallback)), None
        )
        if ema_callback and ema_callback.ema_model:
            # DEBUG: Verify EMA is different from standard model (only log once per epoch)
            if batch_idx == 0 and self.trainer.is_global_zero:
                with torch.no_grad():
                    # Check first few parameters
                    params_equal = []
                    for (n1, p1), (n2, p2) in zip(
                        list(self.model.named_parameters())[:5],
                        list(ema_callback.ema_model.module.named_parameters())[:5],
                    ):
                        params_equal.append(torch.allclose(p1, p2, atol=1e-6))

                    if all(params_equal):
                        self.print(
                            f"⚠️  [Val] WARNING: EMA weights identical to model weights! EMA may not be updating."
                        )
                    else:
                        num_diff = sum(1 for eq in params_equal if not eq)
                        self.print(
                            f"✓ [Val] EMA weights differ from model ({num_diff}/5 params checked). EMA is working!"
                        )

            if eval_mode in ["whole", "both"]:
                ema_outputs = ema_callback.ema_model.module(
                    pixel_values=pixel_values, labels=None
                )
                post_processed_ema_outputs = (
                    self.image_processor.post_process_object_detection(
                        ema_outputs,
                        threshold=self.config.model.detection_threshold,
                        target_sizes=batch_image_sizes,
                    )
                )
                post_processed_ema_outputs = [
                    {k: to_cpu_device(v) for k, v in p.items()}
                    for p in post_processed_ema_outputs
                ]
                ema_results = {
                    int(t["image_id"].item()): out
                    for t, out in zip(labels, post_processed_ema_outputs)
                }
                self.validation_step_outputs_ema.append(
                    {"predictions": ema_results, "image_ids": image_ids}
                )

            if eval_mode in ["sliced", "both"]:
                post_processed_ema_outputs_sliced = []
                for i, target in enumerate(labels):
                    image_id = int(target["image_id"].item())
                    img_path = self._get_image_path(image_id, "val")

                    if not img_path or not os.path.exists(img_path):
                        single_pixel_values = pixel_values[i : i + 1]
                        single_output = ema_callback.ema_model.module(
                            pixel_values=single_pixel_values, labels=None
                        )
                        single_post = (
                            self.image_processor.post_process_object_detection(
                                single_output,
                                threshold=self.config.model.detection_threshold,
                                target_sizes=[batch_image_sizes[i]],
                            )
                        )
                        post_processed_ema_outputs.append(
                            {k: to_cpu_device(v) for k, v in single_post[0].items()}
                        )
                        continue

                    def predict_fn_ema(image_np):
                        pil_img = Image.fromarray(image_np)
                        inputs = self.image_processor(
                            images=pil_img, return_tensors="pt"
                        )
                        pixel_val = inputs["pixel_values"].to(self.device)
                        if (
                            next(ema_callback.ema_model.module.parameters()).dtype
                            == torch.float16
                        ):
                            pixel_val = pixel_val.half()
                        elif (
                            next(ema_callback.ema_model.module.parameters()).dtype
                            == torch.bfloat16
                        ):
                            pixel_val = pixel_val.bfloat16()

                        with torch.no_grad():
                            out = ema_callback.ema_model.module(
                                pixel_values=pixel_val, labels=None
                            )

                        patch_size = [[image_np.shape[0], image_np.shape[1]]]
                        post_out = self.image_processor.post_process_object_detection(
                            out,
                            threshold=self.config.model.detection_threshold,
                            target_sizes=patch_size,
                        )[0]
                        return {k: to_cpu_device(v) for k, v in post_out.items()}

                    img_pil = Image.open(img_path).convert("RGB")
                    sahi_cfg = self.config.get("eval_inference", {}).get("sahi", {})
                    input_size = (
                        self.config.data.model_input_size
                        if hasattr(self.config.data, "model_input_size")
                        else 640
                    )

                    preds = run_sahi_sliced_eval(
                        img_pil, predict_fn_ema, sahi_cfg, input_size
                    )
                    post_processed_ema_outputs_sliced.append(preds)

                ema_results_sliced = {
                    int(t["image_id"].item()): out
                    for t, out in zip(labels, post_processed_ema_outputs_sliced)
                }
                self.validation_step_outputs_sliced_ema.append(
                    {"predictions": ema_results_sliced, "image_ids": image_ids}
                )

        return {
            "predictions": {},
            "image_ids": image_ids,
        }  # Real return not heavily used

    def _gather_all_outputs(self, local_outputs):
        """Gather outputs from all devices in a distributed setting."""
        if not dist.is_available() or not dist.is_initialized():
            return local_outputs

        world_size = dist.get_world_size()
        if world_size <= 1:
            return local_outputs

        # Gather the list of objects from all ranks
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_outputs)

        # Flatten the list of lists into a single list
        return [item for rank_outputs in gathered for item in rank_outputs]

    def on_validation_epoch_end(self):
        viz_predictions = None

        def _compute_and_log(outputs_list, prefix_name, base_prefix, suffix=""):
            all_outputs = self._gather_all_outputs(outputs_list)
            merged = self._merge_predictions_map(all_outputs)
            metrics = {}
            if self.trainer.is_global_zero:
                predictions = []
                image_ids = []
                for output_batch in all_outputs:
                    predictions.extend(
                        convert_preds_to_coco(output_batch["predictions"])
                    )
                    image_ids.extend(output_batch["image_ids"])

                if len(predictions) > 0:
                    if self.val_coco_gt is not None:
                        # metrics = self._compute_coco_metrics(
                        #     predictions=predictions,
                        #     image_ids=list(set(image_ids)),
                        #     coco_gt=self.val_coco_gt,
                        #     prefix=f"{prefix_name} performance"
                        # )
                        metrics = compute_coco_metrics(
                            coco_gt=self.val_coco_gt,
                            predictions=predictions,
                            image_ids=sorted(list(set(image_ids))),
                            max_detections=int(self.config.model.max_detections),
                            label_map=self.config.model.label_map,
                            prefix=f"{prefix_name} performance",
                        )
                else:
                    metrics = {"map": 0.0, "map_50": 0.0, "map_75": 0.0}

            if dist.is_available() and dist.is_initialized():
                object_list = [metrics]
                dist.broadcast_object_list(object_list, src=0)
                metrics = object_list[0]

            for key, value in metrics.items():
                self.log(
                    f"{base_prefix}/{key}{suffix}",
                    value,
                    prog_bar=(key in {"map", "map_50"}),
                    sync_dist=True,
                )
                if key == "map":
                    self.log(
                        f"{base_prefix}_{key}{suffix}",
                        value,
                        prog_bar=False,
                        sync_dist=True,
                    )

            outputs_list.clear()
            return merged

        # 1. Standard Whole
        if self.validation_step_outputs:
            viz_predictions = _compute_and_log(
                self.validation_step_outputs, "Val", "val", ""
            )

        # 2. Standard Sliced
        if self.validation_step_outputs_sliced:
            viz_predictions = _compute_and_log(
                self.validation_step_outputs_sliced, "Val Sliced", "val", "_sliced"
            )

        # 3. EMA Whole
        if (
            hasattr(self, "validation_step_outputs_ema")
            and self.validation_step_outputs_ema
        ):
            viz_predictions = _compute_and_log(
                self.validation_step_outputs_ema, "Val EMA", "val", "_ema"
            )

        # 4. EMA Sliced
        if (
            hasattr(self, "validation_step_outputs_sliced_ema")
            and self.validation_step_outputs_sliced_ema
        ):
            viz_predictions = _compute_and_log(
                self.validation_step_outputs_sliced_ema,
                "Val Sliced EMA",
                "val",
                "_sliced_ema",
            )

        if self.trainer.is_global_zero and viz_predictions is not None:
            self._visualize_aggregated_predictions(viz_predictions, split="val")

        self.debug_train_image_ids.clear()

    def on_test_epoch_start(self):
        """Reset test visualization counter."""
        self.test_viz_counter = 0

    def _merge_predictions_map(self, gathered_outputs):
        merged = {}
        for batch_out in gathered_outputs:
            merged.update(batch_out.get("predictions", {}))
        return merged

    def _get_visualization_limit(self):
        raw_value = self.config.checkpointing.visualize_samples
        if raw_value == -1:
            return None
        try:
            numeric = float(raw_value)
        except (TypeError, ValueError):
            return None
        if numeric < 0 or numeric == float("inf"):
            return None
        return int(numeric)

    def _should_visualize(self, split: str) -> bool:
        if split == "val":
            every_n = max(1, int(self.config.checkpointing.visualize_every_n_epochs))
            return (self.current_epoch + 1) % every_n == 0
        return True

    def _visualize_aggregated_predictions(self, predictions_map, split="val"):
        if (
            not self.trainer.is_global_zero
            or not predictions_map
            or not self._should_visualize(split)
        ):
            return

        max_samples = self._get_visualization_limit()
        coco_gt = self.test_coco_gt if split == "test" else self.val_coco_gt
        label_map = self.config.model.label_map
        viz_threshold = float(self.config.model.draw_threshold)

        save_dir = os.path.join(
            self.config.checkpointing.save_dir,
            self.config.checkpointing.visualization_dir,
            f"epoch_{(self.current_epoch + 1):03d}" if split == "val" else "test",
            split,
        )

        os.makedirs(save_dir, exist_ok=True)
        saved_count = 0
        self.print(f"[VIZ] Saving {split.upper()} visualizations to: {save_dir}")

        for image_id in sorted(predictions_map.keys()):
            if max_samples is not None and saved_count >= max_samples:
                break

            # RT-DETR specific path resolver
            image_path = self._get_image_path(int(image_id), split)
            if not image_path or not os.path.exists(image_path):
                continue

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                continue

            gt_boxes, gt_labels = [], []
            if coco_gt:
                try:
                    gt_anns = coco_gt.loadAnns(
                        coco_gt.getAnnIds(imgIds=[int(image_id)])
                    )
                    for ann in gt_anns:
                        x, y, w, h = ann["bbox"]
                        gt_boxes.append([x, y, x + w, y + h])
                        gt_labels.append(ann["category_id"])
                except Exception:
                    pass

            if gt_boxes:
                image = self.draw_boxes(
                    image,
                    gt_boxes,
                    gt_labels,
                    scores=None,
                    id2label=label_map,
                    color_override=(0, 255, 0),
                    label_prefix="",
                )

            pred_class_names = []
            preds = predictions_map.get(image_id, {})
            if "boxes" in preds and len(preds["boxes"]) > 0:
                valid_indices = preds["scores"] >= viz_threshold
                valid_labels = preds["labels"][valid_indices]
                for label in valid_labels:
                    label_item = label.item() if torch.is_tensor(label) else int(label)
                    class_name = (
                        label_map.get(int(label_item))
                        or label_map.get(str(int(label_item)))
                        or str(label_item)
                    )
                    pred_class_names.append(class_name)

                image = self.draw_boxes(
                    image,
                    preds["boxes"],
                    preds["labels"],
                    preds["scores"],
                    id2label=label_map,
                    color_override=(255, 0, 0),
                    label_prefix="",
                )

            # Draw Counts on Image (Top Right)
            gt_counts = Counter(
                [
                    label_map.get(int(l)) or label_map.get(str(int(l))) or str(l)
                    for l in gt_labels
                ]
            )
            pred_counts = Counter(pred_class_names)

            draw = ImageDraw.Draw(image)
            text_y, line_height = 10, 24
            for cls_name in sorted(set(gt_counts.keys()) | set(pred_counts.keys())):
                parts = [
                    (f"{cls_name}: ", "white"),
                    (f"{pred_counts[cls_name]}", "red"),
                    ("/", "white"),
                    (f"{gt_counts[cls_name]}", "green"),
                ]
                total_width = sum(
                    (
                        draw.textbbox((0, 0), text, font=self.font)[2]
                        - draw.textbbox((0, 0), text, font=self.font)[0]
                    )
                    for text, _ in parts
                )
                current_x = image.width - total_width - 10
                for text, color in parts:
                    draw.text(
                        (current_x + 1, text_y + 1), text, fill="black", font=self.font
                    )
                    draw.text((current_x, text_y), text, fill=color, font=self.font)
                    current_x += (
                        draw.textbbox((0, 0), text, font=self.font)[2]
                        - draw.textbbox((0, 0), text, font=self.font)[0]
                    )
                text_y += line_height

            original_filename = os.path.basename(image_path)
            image.save(
                os.path.join(save_dir, f"image_{int(image_id)}_{original_filename}")
            )
            saved_count += 1

    def test_step(self, batch, batch_idx):
        """Test step (same as validation)."""
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        # Collect true original image sizes from COCO metadata
        batch_image_sizes = []
        for x in labels:
            image_id = int(x["image_id"].item())
            if self.test_coco_gt and image_id in self.test_coco_gt.imgs:
                img_info = self.test_coco_gt.imgs[image_id]
                batch_image_sizes.append([img_info["height"], img_info["width"]])
            else:
                batch_image_sizes.append(to_cpu_device(x["orig_size"]).numpy().tolist())

        eval_mode = self.config.get("eval_inference", {}).get("mode", "whole")

        post_processed_outputs = []
        if eval_mode in ["whole", "both"]:
            outputs = self.model(pixel_values=pixel_values, labels=None)
            post_processed_outputs = self.image_processor.post_process_object_detection(
                outputs,
                threshold=self.config.model.detection_threshold,
                target_sizes=batch_image_sizes,
            )
            post_processed_outputs = [
                {k: to_cpu_device(v) for k, v in p.items()}
                for p in post_processed_outputs
            ]

            results = {
                int(t["image_id"].item()): out
                for t, out in zip(labels, post_processed_outputs)
            }
            image_ids = [int(t["image_id"].item()) for t in labels]
            self.test_step_outputs.append(
                {"predictions": results, "image_ids": image_ids}
            )

        if eval_mode in ["sliced", "both"]:
            post_processed_outputs_sliced = []
            for i, target in enumerate(labels):
                image_id = int(target["image_id"].item())
                img_path = self._get_image_path(image_id, "test")

                if not img_path or not os.path.exists(img_path):
                    self.print(
                        f"[Test] WARNING: Cannot find image {img_path} for SAHI. Falling back to whole image."
                    )
                    single_pixel_values = pixel_values[i : i + 1]
                    single_output = self.model(
                        pixel_values=single_pixel_values, labels=None
                    )
                    single_post = self.image_processor.post_process_object_detection(
                        single_output,
                        threshold=self.config.model.detection_threshold,
                        target_sizes=[batch_image_sizes[i]],
                    )
                    post_processed_outputs.append(
                        {k: to_cpu_device(v) for k, v in single_post[0].items()}
                    )
                    continue

                def predict_fn(image_np):
                    pil_img = Image.fromarray(image_np)
                    inputs = self.image_processor(images=pil_img, return_tensors="pt")
                    pixel_val = inputs["pixel_values"].to(self.device)
                    if next(self.model.parameters()).dtype == torch.float16:
                        pixel_val = pixel_val.half()
                    elif next(self.model.parameters()).dtype == torch.bfloat16:
                        pixel_val = pixel_val.bfloat16()

                    with torch.no_grad():
                        out = self.model(pixel_values=pixel_val, labels=None)

                    patch_size = [[image_np.shape[0], image_np.shape[1]]]
                    post_out = self.image_processor.post_process_object_detection(
                        out,
                        threshold=self.config.model.detection_threshold,
                        target_sizes=patch_size,
                    )[0]
                    return {k: to_cpu_device(v) for k, v in post_out.items()}

                img_pil = Image.open(img_path).convert("RGB")
                sahi_cfg = self.config.get("eval_inference", {}).get("sahi", {})
                input_size = (
                    self.config.data.model_input_size
                    if hasattr(self.config.data, "model_input_size")
                    else 640
                )

                preds = run_sahi_sliced_eval(img_pil, predict_fn, sahi_cfg, input_size)
                post_processed_outputs_sliced.append(preds)

            results_sliced = {
                int(t["image_id"].item()): out
                for t, out in zip(labels, post_processed_outputs_sliced)
            }
            image_ids = [int(t["image_id"].item()) for t in labels]
            self.test_step_outputs_sliced.append(
                {"predictions": results_sliced, "image_ids": image_ids}
            )

        # EMA validation during test
        from utils.ema import EMACallback

        ema_callback = next(
            (cb for cb in self.trainer.callbacks if isinstance(cb, EMACallback)), None
        )
        if ema_callback and ema_callback.ema_model:
            if eval_mode in ["whole", "both"]:
                ema_outputs = ema_callback.ema_model.module(
                    pixel_values=pixel_values, labels=None
                )
                post_processed_ema_outputs = (
                    self.image_processor.post_process_object_detection(
                        ema_outputs,
                        threshold=self.config.model.detection_threshold,
                        target_sizes=batch_image_sizes,
                    )
                )
                post_processed_ema_outputs = [
                    {k: to_cpu_device(v) for k, v in p.items()}
                    for p in post_processed_ema_outputs
                ]
                ema_results = {
                    int(t["image_id"].item()): out
                    for t, out in zip(labels, post_processed_ema_outputs)
                }
                self.test_step_outputs_ema.append(
                    {"predictions": ema_results, "image_ids": image_ids}
                )

            if eval_mode in ["sliced", "both"]:
                post_processed_ema_outputs_sliced = []
                for i, target in enumerate(labels):
                    image_id = int(target["image_id"].item())
                    img_path = self._get_image_path(image_id, "test")

                    if not img_path or not os.path.exists(img_path):
                        single_pixel_values = pixel_values[i : i + 1]
                        single_output = ema_callback.ema_model.module(
                            pixel_values=single_pixel_values, labels=None
                        )
                        single_post = (
                            self.image_processor.post_process_object_detection(
                                single_output,
                                threshold=self.config.model.detection_threshold,
                                target_sizes=[batch_image_sizes[i]],
                            )
                        )
                        post_processed_ema_outputs.append(
                            {k: to_cpu_device(v) for k, v in single_post[0].items()}
                        )
                        continue

                    def predict_fn_ema(image_np):
                        pil_img = Image.fromarray(image_np)
                        inputs = self.image_processor(
                            images=pil_img, return_tensors="pt"
                        )
                        pixel_val = inputs["pixel_values"].to(self.device)
                        if (
                            next(ema_callback.ema_model.module.parameters()).dtype
                            == torch.float16
                        ):
                            pixel_val = pixel_val.half()
                        elif (
                            next(ema_callback.ema_model.module.parameters()).dtype
                            == torch.bfloat16
                        ):
                            pixel_val = pixel_val.bfloat16()

                        with torch.no_grad():
                            out = ema_callback.ema_model.module(
                                pixel_values=pixel_val, labels=None
                            )

                        patch_size = [[image_np.shape[0], image_np.shape[1]]]
                        post_out = self.image_processor.post_process_object_detection(
                            out,
                            threshold=self.config.model.detection_threshold,
                            target_sizes=patch_size,
                        )[0]
                        return {k: to_cpu_device(v) for k, v in post_out.items()}

                    img_pil = Image.open(img_path).convert("RGB")
                    sahi_cfg = self.config.get("eval_inference", {}).get("sahi", {})
                    input_size = (
                        self.config.data.model_input_size
                        if hasattr(self.config.data, "model_input_size")
                        else 640
                    )

                    preds = run_sahi_sliced_eval(
                        img_pil, predict_fn_ema, sahi_cfg, input_size
                    )
                    post_processed_ema_outputs_sliced.append(preds)

                ema_results_sliced = {
                    int(t["image_id"].item()): out
                    for t, out in zip(labels, post_processed_ema_outputs_sliced)
                }
                self.test_step_outputs_sliced_ema.append(
                    {"predictions": ema_results_sliced, "image_ids": image_ids}
                )

        return {"predictions": {}, "image_ids": image_ids}

    def on_test_epoch_end(self):
        viz_predictions = None

        def _compute_and_log(outputs_list, prefix_name, base_prefix, suffix=""):
            all_outputs = self._gather_all_outputs(outputs_list)
            merged = self._merge_predictions_map(all_outputs)
            metrics = {}
            if self.trainer.is_global_zero:
                predictions = []
                image_ids = []
                for output_batch in all_outputs:
                    predictions.extend(
                        convert_preds_to_coco(output_batch["predictions"])
                    )
                    image_ids.extend(output_batch["image_ids"])

                if len(predictions) > 0:
                    if self.test_coco_gt is not None:
                        # metrics = self._compute_coco_metrics(
                        #     predictions=predictions,
                        #     image_ids=list(set(image_ids)),
                        #     coco_gt=self.test_coco_gt,
                        #     prefix=f"{prefix_name} performance"
                        # )
                        metrics = compute_coco_metrics(
                            coco_gt=self.test_coco_gt,
                            predictions=predictions,
                            image_ids=sorted(list(set(image_ids))),
                            max_detections=int(self.config.model.max_detections),
                            label_map=self.config.model.label_map,
                            prefix=f"{prefix_name} performance",
                        )
                else:
                    metrics = {"map": 0.0, "map_50": 0.0, "map_75": 0.0}

            if dist.is_available() and dist.is_initialized():
                object_list = [metrics]
                dist.broadcast_object_list(object_list, src=0)
                metrics = object_list[0]

            for key, value in metrics.items():
                self.log(
                    f"{base_prefix}/{key}{suffix}",
                    value,
                    prog_bar=(key in {"map", "map_50"}),
                    sync_dist=True,
                )

            outputs_list.clear()
            return merged

        if self.test_step_outputs:
            viz_predictions = _compute_and_log(
                self.test_step_outputs, "Test", "test", ""
            )

        if self.test_step_outputs_sliced:
            viz_predictions = _compute_and_log(
                self.test_step_outputs_sliced, "Test Sliced", "test", "_sliced"
            )

        if hasattr(self, "test_step_outputs_ema") and self.test_step_outputs_ema:
            viz_predictions = _compute_and_log(
                self.test_step_outputs_ema, "Test EMA", "test", "_ema"
            )

        if (
            hasattr(self, "test_step_outputs_sliced_ema")
            and self.test_step_outputs_sliced_ema
        ):
            viz_predictions = _compute_and_log(
                self.test_step_outputs_sliced_ema,
                "Test Sliced EMA",
                "test",
                "_sliced_ema",
            )

        if self.trainer.is_global_zero and viz_predictions is not None:
            self._visualize_aggregated_predictions(viz_predictions, split="test")

    def on_test_start(self):
        """
        Ensure dtype/device are compatible with the configured precision mode.
        For mixed precision, keep FP32 weights and rely on autocast.
        For true precision modes, cast model weights explicitly.
        """
        precision_mode = str(self.trainer.precision).lower()

        # Mixed precision keeps model weights in FP32.
        if precision_mode in {"16-mixed", "bf16-mixed"}:
            return

        target_dtype = None
        if precision_mode in {"16-true", "16"}:
            target_dtype = torch.float16
        elif precision_mode in {"bf16-true", "bf16"}:
            target_dtype = torch.bfloat16

        if target_dtype is None:
            return

        if next(self.parameters()).dtype != target_dtype:
            self.to(dtype=target_dtype)
            if hasattr(self, "model"):
                self.model.to(dtype=target_dtype)
            self.print(
                f"[INFO] Cast model weights to {target_dtype} for precision={self.trainer.precision}."
            )

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
                    self.print(
                        f"Restoring checkpoint: Increased total_steps from {old_total} to {old_total + 5000}"
                    )

    def _remap_coco_gt(self, coco_gt):
        """In-place remap of COCO GT categories to match remapped classes."""
        if not coco_gt:
            return None, {}

        if hasattr(coco_gt, "_remapped"):
            return coco_gt, getattr(coco_gt, "_remap_dict", {})

        # 0. Check if remapping is enabled
        if hasattr(self.config, "remap_labels") and not self.config.remap_labels:
            return coco_gt, {}

        # 1. Get target map from config
        if (
            not self.config
            or "model" not in self.config
            or "label_map" not in self.config.model
        ):
            return coco_gt, {}

        target_label_map = self.config.model.label_map
        name_to_target_id = {v: int(k) for k, v in target_label_map.items()}

        # 2. Get remapping rules
        remapping_rules = {}
        if (
            "data" in self.config
            and self.config.data
            and "class_remapping" in self.config.data
        ):
            remapping_rules = self.config.data.class_remapping
        elif "class_remapping" in self.config:
            remapping_rules = self.config.class_remapping

        remap_dict = {}
        for cat_id, cat_info in coco_gt.cats.items():
            src_name = cat_info["name"]
            effective_name = remapping_rules.get(src_name, src_name)
            if effective_name in name_to_target_id:
                remap_dict[cat_id] = name_to_target_id[effective_name]

        # 3. Apply to annotations
        for ann in coco_gt.dataset.get("annotations", []):
            if ann["category_id"] in remap_dict:
                ann["category_id"] = remap_dict[ann["category_id"]]

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
                new_categories.append({"id": int(target_id), "name": name})

        coco_gt.dataset["categories"] = new_categories

        # 5. Re-index
        coco_gt.createIndex()
        coco_gt._remapped = True
        coco_gt._remap_dict = remap_dict
        self.print(f"[INFO] Remapped Validation GT classes using: {remap_dict}")
        return coco_gt, remap_dict

    def _compute_coco_metrics(
        self, predictions, image_ids, coco_gt, prefix="Performance"
    ):
        """Compute COCO mAP and mAR metrics."""
        if coco_gt is None or len(predictions) == 0:
            return {}

        if self.config.debug:
            self.print(
                f"DEBUG: COCO GT Categories before remap: {[{c['id']: c['name']} for c in coco_gt.dataset['categories']]}"
            )
        coco_gt, remap_dict = self._remap_coco_gt(coco_gt)
        # Remap predictions if remapping rules were applied
        if remap_dict:
            for p in predictions:
                if p["category_id"] in remap_dict:
                    p["category_id"] = remap_dict[p["category_id"]]

        # Debug: Verify remapping
        if self.config.debug:
            self.print(
                f"DEBUG: COCO GT Categories after remap: {[{c['id']: c['name']} for c in coco_gt.dataset['categories']]}"
            )
            if remap_dict:
                self.print("DEBUG: Applied prediction remapping for evaluation.")

        metrics = {
            "map": -1.0,
            "map_50": -1.0,
            "map_75": -1.0,
            "map_small": -1.0,
            "map_medium": -1.0,
            "map_large": -1.0,
            "mar_1": -1.0,
            "mar_10": -1.0,
            f"mar_{self.config.model.max_detections}": -1.0,
            "mar_small": -1.0,
            "mar_medium": -1.0,
            "mar_large": -1.0,
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

            self.print("[INFO] Computing Class-wise COCO metrics")
            # Extract per-category metrics from the internal COCOeval precision tensor
            if hasattr(coco_evaluator, "eval") and "precision" in coco_evaluator.eval:
                precisions = coco_evaluator.eval["precision"]
                import numpy as np

                for i, catId in enumerate(coco_evaluator.params.catIds):
                    # Use updated GT categories if remapping is applied, otherwise fallback to config label_map
                    if getattr(self.config, "remap_labels", False):
                        cat_info = coco_gt.cats.get(int(catId))
                        cat_name = cat_info["name"] if cat_info else None
                    else:
                        cat_name = self.config.model.label_map.get(
                            int(catId)
                        ) or self.config.model.label_map.get(str(catId))

                    if not cat_name:
                        cat_name = f"class_{catId}"

                    # mAP (average over all IoU thresholds)
                    s = precisions[:, :, i, 0, -1]
                    if len(s[s > -1]) > 0:
                        metrics[f"map_{cat_name}"] = round(float(np.mean(s[s > -1])), 4)

                    # mAP-50 (IoU threshold 0.5)
                    s_50 = precisions[0, :, i, 0, -1]
                    if len(s_50[s_50 > -1]) > 0:
                        metrics[f"map_50_{cat_name}"] = round(
                            float(np.mean(s_50[s_50 > -1])), 4
                        )

                    # compute recalls as well
                    recalls = coco_evaluator.eval["recall"][:, i, 0, -1]
                    if len(recalls[recalls > -1]) > 0:
                        metrics[f"mar_{cat_name}"] = round(
                            float(np.mean(recalls[recalls > -1])), 4
                        )
            self.print("[INFO] Logged Class-wise COCO metrics")
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
        use_param_groups = opt_config.get("use_param_groups", False)
        param_groups_config = opt_config.get("param_groups", [])

        if not use_param_groups or not param_groups_config:
            # Fallback to uniform LR for the whole model (no grouping)
            self.print(f"[INFO] Using uniform LR: {opt_config.lr} for all parameters.")
            optimizer_grouped_params = [
                {
                    "params": [p for p in self.model.parameters() if p.requires_grad],
                    "lr": opt_config.lr,
                    "weight_decay": opt_config.weight_decay,
                }
            ]
        else:
            # Regex-based grouping from config
            optimizer_grouped_params = []
            memo = set()  # Track assigned parameters

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
                    new_group = {k: v for k, v in group_cfg.items() if k != "params"}
                    new_group["params"] = group_params

                    # Inherit defaults if not specified
                    if "lr" not in new_group:
                        new_group["lr"] = opt_config.lr
                    if "weight_decay" not in new_group:
                        new_group["weight_decay"] = opt_config.weight_decay

                    optimizer_grouped_params.append(new_group)
                    self.print(
                        f"[INFO] Optimizer Group: '{pattern}' matched {len(group_params)} parameters."
                    )

            # Catch-all for remaining parameters
            remaining_params = []
            for name, param in self.model.named_parameters():
                if param.requires_grad and id(param) not in memo:
                    remaining_params.append(param)

            if remaining_params:
                optimizer_grouped_params.append(
                    {
                        "params": remaining_params,
                        "lr": opt_config.lr,
                        "weight_decay": opt_config.weight_decay,
                    }
                )
                self.print(
                    f"[INFO] Optimizer Group: 'default' matched remaining {len(remaining_params)} parameters."
                )

        # Create optimizer
        optimizer = torch.optim.AdamW(
            optimizer_grouped_params, weight_decay=opt_config.weight_decay
        )

        # 2. Setup Scheduler
        total_steps = self.trainer.estimated_stepping_batches
        # Ensure total_steps is at least 1 to avoid division by zero or weird logic
        total_steps = max(1, total_steps)

        if total_steps < sch_config.warmup_steps:
            # If total steps is less than requested warmup, reduce warmup to 10% of total
            warmup_steps = int(0.1 * total_steps)
            self.print(
                f"[WARN] Total steps ({total_steps}) < Warmup steps ({sch_config.warmup_steps}). Adjusting warmup to {warmup_steps}."
            )
        else:
            warmup_steps = sch_config.warmup_steps

        # Save effective warmup steps for optimizer_step
        self.effective_warmup_steps = warmup_steps

        # Configure scheduler
        if sch_config.type == "reduce_lr_on_plateau":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    factor=sch_config.factor,
                    patience=sch_config.patience,
                ),
                "monitor": "val/map",
                "interval": "epoch",
                "frequency": 1,
                # 'verbose': Trueq
            }
        elif sch_config.type == "cosine":
            # Ensure T_max is positive
            t_max = max(1, total_steps - warmup_steps)
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=t_max, eta_min=sch_config.eta_min
                ),
                "interval": "step",
            }
            # schedulers = [warmup_scheduler, scheduler]
        elif sch_config.type == "lambda":
            # Linear warmup + constant
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / max(1, warmup_steps)
                return 1.0

            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
                "interval": "step",
            }

        elif sch_config.type == "step":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=sch_config.step_size, gamma=sch_config.gamma
                ),
                "interval": "epoch",
            }
        elif sch_config.type == "multistep":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=sch_config.milestones, gamma=sch_config.gamma
                ),
                "interval": "epoch",
            }
        elif sch_config.type == "onecycle":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=opt_config.lr,
                    total_steps=total_steps,
                    pct_start=sch_config.pct_start,
                    anneal_strategy="cos",
                    div_factor=25.0,
                    final_div_factor=1e3,
                ),
                "interval": "step",
            }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # --- 1. Warmup Logic (Apply BEFORE step) ---

        # Use effective_warmup_steps if set in configure_optimizers, else default logic
        if hasattr(self, "effective_warmup_steps"):
            warmup_steps = self.effective_warmup_steps
        else:
            total_steps = self.trainer.estimated_stepping_batches + 100
            warmup_steps = max(
                self.config.scheduler.warmup_steps, int(0.1 * total_steps)
            )

        is_one_cycle = self.config.scheduler.type == "onecycle"

        if self.trainer.global_step < warmup_steps and not is_one_cycle:
            # Calculate linear scale (0.0 to 1.0)
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / float(max(1, warmup_steps))
            )

            # Get the base LR from config to ensure we always scale from the correct starting point
            # (Avoids issues where pg['lr'] might be modified by other schedulers or restarts)
            base_lr = self.config.optimizer.optimizer.lr

            for pg in optimizer.param_groups:
                pg["lr"] = base_lr * lr_scale

        optimizer.step(closure=optimizer_closure)

    def _get_scheduler_with_warmup(self, optimizer, sch_config):
        """
        Helper to attach warmup to any scheduler strategy safely.
        """
        # --- A. Define the Warmup Scheduler ---
        # Starts at 1% of target LR and ramps up linearly
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-2, total_iters=sch_config.warmup_steps
        )

        # --- B. Define the Main Scheduler & Combine ---

        # 1. Reduce LR on Plateau (Cannot be chained sequentially)
        if sch_config.type == "reduce_lr_on_plateau":
            plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=sch_config.factor,
                patience=sch_config.patience,
            )
            # Return list: Warmup runs every step, Plateau runs every epoch independently
            return [
                {"scheduler": warmup_scheduler, "interval": "step", "frequency": 1},
                {
                    "scheduler": plateau_scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": "val/map",
                },
            ]

        # 2. Cosine Annealing (Sequential)
        elif sch_config.type == "cosine":
            # Calculate remaining steps for the cosine phase
            total_steps = self.trainer.estimated_stepping_batches
            main_iters = total_steps - sch_config.warmup_steps

            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=main_iters, eta_min=optimizer.defaults["lr"] * 0.01
            )

            chained_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[sch_config.warmup_steps],
            )
            return [{"scheduler": chained_scheduler, "interval": "step"}]

        # 3. Lambda LR (Sequential)
        elif sch_config.type == "lambda":
            # Define your lambda logic here.
            # NOTE: In SequentialLR, the lambda receives the GLOBAL step count.

            # Example: Inverse Square Root Decay (common in Transformers)
            # We use 'max' to prevent division by zero or overly high values if step < warmup
            def lr_lambda(step):
                # Since this runs AFTER warmup, step will be > warmup_steps
                # We normalize so it continues smoothly from 1.0 down
                if step < sch_config.warmup_steps:
                    return 1.0
                return (sch_config.warmup_steps / step) ** 0.5

            main_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_lambda
            )

            chained_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[sch_config.warmup_steps],
            )
            return [{"scheduler": chained_scheduler, "interval": "step"}]

        # 4. Default: Warmup only (Constant afterwards)
        else:
            # LinearLR stays at factor 1.0 after total_iters are done
            return [{"scheduler": warmup_scheduler, "interval": "step"}]

    def draw_boxes(
        self,
        image,
        boxes,
        labels,
        scores=None,
        id2label=None,
        color_override=None,
        label_prefix="",
    ):
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

            draw.rectangle(
                box, outline=color, width=3
            )  # Increased width for better visibility

            class_name = (
                id2label.get(label_id)
                or id2label.get(str(label_id))
                or f"class_{label_id}"
            )
            label_text = f"{label_prefix}{class_name}"
            if scores is not None:
                label_text += f": {score:.2f}"

            text_box = draw.textbbox((box[0], box[1]), label_text, font=self.font)
            draw.rectangle(text_box, fill=color)
            draw.text((box[0], box[1]), label_text, fill="white", font=self.font)

        return image

    def _visualize_batch(
        self,
        save_dir,
        post_processed_outputs,
        pixel_values,
        labels,
        batch_image_sizes,
        counter,
    ):
        """Saves visualizations for a batch showing both GT and Predictions."""
        os.makedirs(save_dir, exist_ok=True)
        id2label = self.model.config.id2label
        max_samples = self.config.checkpointing.visualize_samples

        if counter == 0:
            self.print(f"[VIZ] Saving visualizations to: {save_dir}")
            self.print(f"[VIZ] Max samples: {max_samples}")

        # Determine which COCO GT to use based on stage
        coco_gt = self.test_coco_gt if self.trainer.testing else self.val_coco_gt
        if coco_gt is None:
            self.print("[WARNING] COCO GT is None, skipping GT visualization.")
            # Fallback to only drawing predictions if GT is missing
            pass

        # Get mean and std from the processor to un-normalize
        mean = torch.tensor(
            self.image_processor.image_mean, device=pixel_values.device
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            self.image_processor.image_std, device=pixel_values.device
        ).view(1, 3, 1, 1)

        # Un-normalize the entire batch
        unnormalized_images = torch.clamp((pixel_values * std) + mean, 0, 1)

        for i in range(len(labels)):
            if max_samples != -1 and counter >= max_samples:
                break

            # Get original image info
            image_id = int(labels[i]["image_id"].item())
            image_tensor = unnormalized_images[i]

            # Get true original size for this image (e.g., 672, 672)
            orig_h, orig_w = batch_image_sizes[i]

            # Resize the unnormalized model-input-sized image tensor (e.g., 640x640)
            # back to its original resolution (e.g., 672x672) so it matches the boxes.
            image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(
                "uint8"
            )
            resized_image_np = cv2.resize(
                image_np, (int(orig_w), int(orig_h)), interpolation=cv2.INTER_LINEAR
            )
            image = Image.fromarray(resized_image_np)

            # Get image metadata from COCO GT for filename
            if coco_gt:
                try:
                    img_info = coco_gt.loadImgs(image_id)[0]
                except (IndexError, AttributeError, KeyError):
                    img_info = {"file_name": f"image_{image_id}.png"}

                # --- 1. Scaled Ground Truth Boxes ---
                gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=image_id))
                gt_boxes = []
                gt_labels = []
                for ann in gt_anns:
                    # coco format is [x, y, width, height]
                    x, y, w, h = ann["bbox"]
                    # Convert to [x1, y1, x2, y2] for draw_boxes
                    gt_boxes.append([x, y, x + w, y + h])
                    gt_labels.append(ann["category_id"])

                # Draw Ground Truth in GREEN
                image = self.draw_boxes(
                    image,
                    gt_boxes,
                    gt_labels,
                    scores=None,
                    id2label=self.config.model.label_map,
                    color_override=(0, 255, 0),  # Green
                    label_prefix="",
                )
            else:
                # If coco_gt is missing, we might not have the filename easily,
                # but we can try to use the image_id
                img_info = {"file_name": f"image_{image_id}.png"}
                gt_labels = []  # No GT available

            # --- 2. Prediction Boxes ---
            preds = post_processed_outputs[i]

            # Draw Predictions in RED
            image = self.draw_boxes(
                image,
                preds["boxes"],
                preds["labels"],
                preds["scores"],
                id2label=id2label,
                color_override=(255, 0, 0),  # Red
                label_prefix="",
            )

            # --- 3. Visualization Counts & Filename ---
            # Compute counts
            label_map = self.config.model.label_map

            # GT Counts
            gt_counts = Counter(
                [
                    label_map.get(int(l)) or label_map.get(str(l)) or str(l)
                    for l in gt_labels
                ]
            )

            # Pred Counts (Apply threshold)
            viz_threshold = self.config.model.draw_threshold
            valid_indices = preds["scores"] >= viz_threshold
            valid_labels = preds["labels"][valid_indices]

            # Robust label mapping handling tensor or int
            pred_class_names = []
            for l in valid_labels:
                l_item = l.item() if torch.is_tensor(l) else int(l)
                name = (
                    label_map.get(int(l_item))
                    or label_map.get(str(l_item))
                    or str(l_item)
                )
                pred_class_names.append(name)

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
                    (f"{gt_counts[cls_name]}", "green"),
                ]

                # Calculate total width to align right
                total_width = 0
                for text, _ in parts:
                    bbox = draw.textbbox((0, 0), text, font=self.font)
                    total_width += bbox[2] - bbox[0]

                current_x = image.width - total_width - 10

                for text, color in parts:
                    # Draw shadow
                    draw.text(
                        (current_x + 1, text_y + 1), text, fill="black", font=self.font
                    )
                    # Draw text
                    draw.text((current_x, text_y), text, fill=color, font=self.font)

                    # Advance cursor
                    bbox = draw.textbbox((0, 0), text, font=self.font)
                    current_x += bbox[2] - bbox[0]

                text_y += line_height

            # Construct new filename
            # Format: "image_{detected_class_name1}_{detected_class_name2}_{image_name}.png"
            detected_classes = sorted(list(set(pred_class_names)))
            if detected_classes:
                class_str = "_".join(detected_classes)
                prefix = f"image_{class_str}_"
            else:
                prefix = "image_no_detections_"

            original_filename = os.path.basename(img_info["file_name"])
            # Avoid repeating "image_" if original already has it, but spec says "image_{class}_{name}"
            # Let's just prepend.
            new_filename = f"{prefix}{original_filename}"
            # Ensure unique valid filename
            new_filename = new_filename.replace(
                "image_image_", "image_"
            )  # Cleanup double prefix if needed

            # Save image
            save_path = os.path.join(save_dir, new_filename)
            image.save(save_path)

            counter += 1

        return counter
