import os
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils

from utils.coco_eval_utils import (
    broadcast_object,
    compute_coco_metrics,
    gather_outputs_across_processes,
)


def _to_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu() if isinstance(tensor, torch.Tensor) else tensor


def _encode_binary_mask(mask: np.ndarray) -> dict[str, Any]:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("utf-8")
    encoded["size"] = [int(x) for x in encoded["size"]]
    return encoded


class Mask2FormerLightningModule(pl.LightningModule):
    def __init__(
        self,
        *,
        model,
        image_processor,
        config,
        model_to_coco: dict[int, int] | None = None,
        val_coco_gt=None,
        test_coco_gt=None,
        val_segm_coco_gt=None,
        test_segm_coco_gt=None,
        val_image_root: str | None = None,
        test_image_root: str | None = None,
    ):
        super().__init__()
        self.model = model
        self.image_processor = image_processor
        self.config = config
        self.model_to_coco = {int(k): int(v) for k, v in (model_to_coco or {}).items()}
        self.detection_label_map = {
            int(v): str(name) for v, name in self.model_to_coco.items()
        }
        self.val_coco_gt = val_coco_gt
        self.test_coco_gt = test_coco_gt
        self.val_segm_coco_gt = val_segm_coco_gt
        self.test_segm_coco_gt = test_segm_coco_gt
        self.val_image_root = val_image_root
        self.test_image_root = test_image_root

        self.validation_step_outputs: list[dict[str, Any]] = []
        self.test_step_outputs: list[dict[str, Any]] = []
        if hasattr(self.config.model, "ema") and self.config.model.ema.enabled:
            self.validation_step_outputs_ema: list[dict[str, Any]] = []
            self.test_step_outputs_ema: list[dict[str, Any]] = []

        # Reverse mapping for GT category ID lookup
        self.coco_to_model = {int(v): int(k) for k, v in self.model_to_coco.items()}

        # Visualization setup
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
        try:
            self.font = ImageFont.truetype("arial.ttf", 17)
        except IOError:
            self.font = ImageFont.load_default()

        self.save_hyperparameters(
            ignore=[
                "model",
                "image_processor",
                "val_coco_gt",
                "test_coco_gt",
                "val_segm_coco_gt",
                "test_segm_coco_gt",
                "val_image_root",
                "test_image_root",
            ]
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_mask: torch.Tensor | None = None,
        mask_labels: list[torch.Tensor] | None = None,
        class_labels: list[torch.Tensor] | None = None,
    ):
        return self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )

    def _move_label_list(self, values: list[torch.Tensor]) -> list[torch.Tensor]:
        return [value.to(self.device) for value in values]

    def _should_visualize(self, split: str) -> bool:
        """Check whether visualization should run this epoch for the given split."""
        if split == "val":
            every_n = max(1, int(self.config.checkpointing.visualize_every_n_epochs))
            return (self.current_epoch + 1) % every_n == 0
        return True

    def _get_visualization_limit(self):
        """Return the maximum number of samples to visualize, or None for unlimited."""
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

    @staticmethod
    def _decode_rle_mask(rle, size):
        """Decode an RLE-encoded mask back to a binary numpy array (H, W)."""
        rle_copy = {"size": size, "counts": rle["counts"].encode("utf-8")}
        return mask_utils.decode(rle_copy).astype(np.uint8)

    @staticmethod
    def _mask_to_contours(mask_np):
        """Extract external contours from a binary mask for outline drawing."""
        contours, _ = cv2.findContours(
            mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    @staticmethod
    def _default_eval_metrics(metric_prefix: str = "") -> dict[str, float]:
        prefix = f"{metric_prefix}_" if metric_prefix else ""
        return {
            f"{prefix}map": 0.0,
            f"{prefix}map_50": 0.0,
            f"{prefix}map_75": 0.0,
        }

    def _post_process_batch(
        self,
        model,
        outputs,
        image_ids: list[int],
        orig_sizes: list[tuple[int, int]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        processed = self.image_processor.post_process_instance_segmentation(
            outputs,
            threshold=float(self.config.model.mask2former.threshold),
            mask_threshold=float(self.config.model.mask2former.mask_threshold),
            overlap_mask_area_threshold=float(
                self.config.model.mask2former.overlap_mask_area_threshold
            ),
            target_sizes=orig_sizes,
            return_binary_maps=True,
        )

        bbox_predictions: list[dict[str, Any]] = []
        segm_predictions: list[dict[str, Any]] = []
        max_detections = int(self.config.model.max_detections)

        for image_id, result in zip(image_ids, processed):
            segments = result.get("segments_info", [])
            segmentation = result.get("segmentation")
            if segmentation is None or len(segments) == 0:
                continue

            if isinstance(segmentation, torch.Tensor):
                mask_stack = segmentation.detach().cpu()
            else:
                mask_stack = torch.as_tensor(segmentation)

            keep_count = 0
            for idx, segment in enumerate(segments):
                if keep_count >= max_detections or idx >= mask_stack.shape[0]:
                    break

                binary_mask = mask_stack[idx].numpy().astype(np.uint8)
                if int(binary_mask.sum()) == 0:
                    continue

                encoded = _encode_binary_mask(binary_mask)
                bbox = mask_utils.toBbox(
                    {
                        "size": encoded["size"],
                        "counts": encoded["counts"].encode("utf-8"),
                    }
                ).tolist()
                category_id = int(
                    self.model_to_coco.get(
                        int(segment["label_id"]), int(segment["label_id"])
                    )
                )
                score = float(segment["score"])

                bbox_predictions.append(
                    {
                        "image_id": int(image_id),
                        "category_id": category_id,
                        "bbox": [float(x) for x in bbox],
                        "score": score,
                    }
                )
                segm_predictions.append(
                    {
                        "image_id": int(image_id),
                        "category_id": category_id,
                        "segmentation": encoded,
                        "score": score,
                    }
                )
                keep_count += 1

        return bbox_predictions, segm_predictions

    def training_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"].to(self.device)
        pixel_mask = batch["pixel_mask"].to(self.device)
        mask_labels = self._move_label_list(batch["mask_labels"])
        class_labels = self._move_label_list(batch["class_labels"])

        outputs = self(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )
        loss = outputs.loss
        batch_size = int(pixel_values.shape[0])

        self.log(
            "train/loss",
            loss,
            batch_size=batch_size,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        if getattr(outputs, "loss_dict", None):
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

    def _lookup_ema_model(self):
        if not (hasattr(self.config.model, "ema") and self.config.model.ema.enabled):
            return None
        from utils.ema import EMACallback

        ema_callback = next(
            (cb for cb in self.trainer.callbacks if isinstance(cb, EMACallback)), None
        )
        if ema_callback and ema_callback.ema_model:
            return ema_callback.ema_model.module
        return None

    def _run_eval_forward(self, model, pixel_values, pixel_mask):
        return model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
        )

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"].to(self.device)
        pixel_mask = batch["pixel_mask"].to(self.device)
        mask_labels = self._move_label_list(batch["mask_labels"])
        class_labels = self._move_label_list(batch["class_labels"])
        image_ids = [int(x) for x in batch["image_ids"].tolist()]
        orig_sizes = [tuple(map(int, x.tolist())) for x in batch["orig_sizes"]]

        outputs = self(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )
        batch_size = int(pixel_values.shape[0])

        self.log(
            "val/loss",
            outputs.loss,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        if getattr(outputs, "loss_dict", None):
            for key, value in outputs.loss_dict.items():
                self.log(
                    f"val/{key}",
                    value,
                    batch_size=batch_size,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

        bbox_predictions, segm_predictions = self._post_process_batch(
            model=self.model,
            outputs=outputs,
            image_ids=image_ids,
            orig_sizes=orig_sizes,
        )
        self.validation_step_outputs.append(
            {
                "bbox_predictions": bbox_predictions,
                "segm_predictions": segm_predictions,
                "image_ids": image_ids,
            }
        )

        ema_model = self._lookup_ema_model()
        if ema_model is not None:
            ema_outputs = self._run_eval_forward(ema_model, pixel_values, pixel_mask)
            ema_bbox_predictions, ema_segm_predictions = self._post_process_batch(
                model=ema_model,
                outputs=ema_outputs,
                image_ids=image_ids,
                orig_sizes=orig_sizes,
            )
            self.validation_step_outputs_ema.append(
                {
                    "bbox_predictions": ema_bbox_predictions,
                    "segm_predictions": ema_segm_predictions,
                    "image_ids": image_ids,
                }
            )
        return outputs.loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"].to(self.device)
        pixel_mask = batch["pixel_mask"].to(self.device)
        image_ids = [int(x) for x in batch["image_ids"].tolist()]
        orig_sizes = [tuple(map(int, x.tolist())) for x in batch["orig_sizes"]]

        outputs = self(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
        )
        bbox_predictions, segm_predictions = self._post_process_batch(
            model=self.model,
            outputs=outputs,
            image_ids=image_ids,
            orig_sizes=orig_sizes,
        )
        self.test_step_outputs.append(
            {
                "bbox_predictions": bbox_predictions,
                "segm_predictions": segm_predictions,
                "image_ids": image_ids,
            }
        )

        ema_model = self._lookup_ema_model()
        if ema_model is not None:
            ema_outputs = self._run_eval_forward(ema_model, pixel_values, pixel_mask)
            ema_bbox_predictions, ema_segm_predictions = self._post_process_batch(
                model=ema_model,
                outputs=ema_outputs,
                image_ids=image_ids,
                orig_sizes=orig_sizes,
            )
            self.test_step_outputs_ema.append(
                {
                    "bbox_predictions": ema_bbox_predictions,
                    "segm_predictions": ema_segm_predictions,
                    "image_ids": image_ids,
                }
            )
        return None

    def _evaluate_and_log_epoch(
        self,
        *,
        outputs_list: list[dict[str, Any]],
        bbox_gt,
        segm_gt,
        prefix: str,
        suffix: str = "",
        prefix_label: str | None = None,
    ) -> list[dict[str, Any]]:
        gathered = gather_outputs_across_processes(outputs_list)
        bbox_predictions: list[dict[str, Any]] = []
        segm_predictions: list[dict[str, Any]] = []
        image_ids: list[int] = []

        for batch_out in gathered:
            bbox_predictions.extend(batch_out.get("bbox_predictions", []))
            segm_predictions.extend(batch_out.get("segm_predictions", []))
            image_ids.extend(batch_out.get("image_ids", []))

        bbox_metrics = self._default_eval_metrics()
        segm_metrics = self._default_eval_metrics("segm")
        if self.trainer.is_global_zero:
            if bbox_predictions:
                bbox_metrics = compute_coco_metrics(
                    coco_gt=bbox_gt,
                    predictions=bbox_predictions,
                    image_ids=sorted(set(image_ids)),
                    max_detections=int(self.config.model.max_detections),
                    label_map=self.detection_label_map,
                    prefix=prefix_label
                    or f"{prefix.capitalize()} detection performance",
                    iou_type="bbox",
                )
            if segm_predictions:
                segm_metrics = compute_coco_metrics(
                    coco_gt=segm_gt,
                    predictions=segm_predictions,
                    image_ids=sorted(set(image_ids)),
                    max_detections=int(self.config.model.max_detections),
                    label_map=self.detection_label_map,
                    prefix=(
                        prefix_label.replace("detection", "segmentation")
                        if prefix_label
                        else f"{prefix.capitalize()} segmentation performance"
                    ),
                    iou_type="segm",
                    metric_prefix="segm",
                )

        bbox_metrics = broadcast_object(bbox_metrics, src=0)
        segm_metrics = broadcast_object(segm_metrics, src=0)

        for key, value in bbox_metrics.items():
            self.log(
                f"{prefix}/{key}{suffix}",
                value,
                prog_bar=(key in {"map", "map_50"}),
                sync_dist=True,
            )
            if key == "map":
                self.log(f"{prefix}_map{suffix}", value, prog_bar=False, sync_dist=True)

        for key, value in segm_metrics.items():
            self.log(
                f"{prefix}/{key}{suffix}",
                value,
                prog_bar=(key in {"segm_map", "segm_map_50"}),
                sync_dist=True,
            )

        gathered_copy = list(gathered)
        outputs_list.clear()
        return gathered_copy

    def on_validation_epoch_end(self):
        gathered_val = self._evaluate_and_log_epoch(
            outputs_list=self.validation_step_outputs,
            bbox_gt=self.val_coco_gt,
            segm_gt=self.val_segm_coco_gt,
            prefix="val",
        )

        if self.trainer.is_global_zero:
            viz_map: dict[int, list[dict[str, Any]]] = {}
            for batch_out in gathered_val:
                for img_id in batch_out.get("image_ids", []):
                    viz_map.setdefault(int(img_id), [])
                for pred in batch_out.get("segm_predictions", []):
                    viz_map[int(pred["image_id"])].append(pred)
            self._visualize_aggregated_predictions(viz_map, split="val")

        if (
            hasattr(self, "validation_step_outputs_ema")
            and self.validation_step_outputs_ema
        ):
            gathered_ema = self._evaluate_and_log_epoch(
                outputs_list=self.validation_step_outputs_ema,
                bbox_gt=self.val_coco_gt,
                segm_gt=self.val_segm_coco_gt,
                prefix="val",
                suffix="_ema",
                prefix_label="Val EMA detection performance",
            )

            if self.trainer.is_global_zero:
                viz_map_ema: dict[int, list[dict[str, Any]]] = {}
                for batch_out in gathered_ema:
                    for img_id in batch_out.get("image_ids", []):
                        viz_map_ema.setdefault(int(img_id), [])
                    for pred in batch_out.get("segm_predictions", []):
                        viz_map_ema[int(pred["image_id"])].append(pred)
                self._visualize_aggregated_predictions(
                    viz_map_ema, split="val", suffix="_ema"
                )

    def on_test_epoch_end(self):
        gathered_test = self._evaluate_and_log_epoch(
            outputs_list=self.test_step_outputs,
            bbox_gt=self.test_coco_gt,
            segm_gt=self.test_segm_coco_gt,
            prefix="test",
        )

        if self.trainer.is_global_zero:
            viz_map: dict[int, list[dict[str, Any]]] = {}
            for batch_out in gathered_test:
                for img_id in batch_out.get("image_ids", []):
                    viz_map.setdefault(int(img_id), [])
                for pred in batch_out.get("segm_predictions", []):
                    viz_map[int(pred["image_id"])].append(pred)
            self._visualize_aggregated_predictions(viz_map, split="test")

        if hasattr(self, "test_step_outputs_ema") and self.test_step_outputs_ema:
            gathered_ema = self._evaluate_and_log_epoch(
                outputs_list=self.test_step_outputs_ema,
                bbox_gt=self.test_coco_gt,
                segm_gt=self.test_segm_coco_gt,
                prefix="test",
                suffix="_ema",
                prefix_label="Test EMA detection performance",
            )

            if self.trainer.is_global_zero:
                viz_map_ema: dict[int, list[dict[str, Any]]] = {}
                for batch_out in gathered_ema:
                    for img_id in batch_out.get("image_ids", []):
                        viz_map_ema.setdefault(int(img_id), [])
                    for pred in batch_out.get("segm_predictions", []):
                        viz_map_ema[int(pred["image_id"])].append(pred)
                self._visualize_aggregated_predictions(
                    viz_map_ema, split="test", suffix="_ema"
                )

    def on_test_start(self):
        self.model = self.model.to(self.device)
        precision_mode = str(self.trainer.precision).lower()
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

    def configure_optimizers(self):
        opt_config = self.config.optimizer.optimizer
        sch_config = self.config.scheduler

        # Parameter Groups to protect pretrained Mask2Former decoder
        base_lr = float(opt_config.lr)
        weight_decay = float(opt_config.weight_decay)

        # Identify FPN/Backbone parameters vs Pretrained Decoder parameters
        fpn_params = []
        decoder_params = []
        backbone_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # If it's in the encoder but NOT the backbone, it's the FPN or Adapter
            if (
                "pixel_level_module.encoder" in name
                and "pixel_level_module.encoder.backbone" not in name
            ):
                fpn_params.append(param)
            # If it's the backbone
            elif "pixel_level_module.encoder.backbone" in name:
                backbone_params.append(param)
            # Everything else is the Mask2Former pretrained decoder
            else:
                decoder_params.append(param)

        param_groups = [
            {
                "params": fpn_params,
                "lr": base_lr,
            },  # FPN needs full LR to learn from scratch
            {
                "params": decoder_params,
                "lr": base_lr * 0.1,
            },  # Pretrained decoder needs much lower LR
        ]

        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": base_lr * 0.01})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

        if sch_config.type == "reduce_lr_on_plateau":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    factor=float(sch_config.factor),
                    patience=int(sch_config.patience),
                ),
                "monitor": (
                    "val/map_ema"
                    if hasattr(self.config.model, "ema")
                    and self.config.model.ema.enabled
                    else "val/map"
                ),
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

    def _resolve_image_path(
        self, image_root: str | None, file_name: str
    ) -> Path | None:
        if not image_root:
            return None
        candidate = Path(image_root) / file_name
        if candidate.exists():
            return candidate
        candidate = Path(image_root) / os.path.basename(file_name)
        if candidate.exists():
            return candidate
        return None

    def _load_image(self, image_path: Path) -> Image.Image:
        return Image.open(image_path).convert("RGB")

    def draw_outlines_and_boxes(
        self,
        image: Image.Image,
        boxes: list,
        labels: list,
        scores: list | None = None,
        contours: list | None = None,
        id2label: dict | None = None,
        color_override: tuple[int, int, int] | None = None,
        label_prefix: str = "",
        threshold_override: float | None = None,
        outline_width: int = 3,
        draw_boxes: bool = True,
        draw_contours: bool = True,
    ) -> Image.Image:
        """Draw bounding boxes and optional mask contour outlines on a PIL image.

        Args:
            image: PIL Image to draw on.
            boxes: List of [x1, y1, x2, y2] boxes.
            labels: List of class IDs (int or tensor).
            scores: Optional list of confidence scores.
            contours: Optional list of OpenCV contours (from cv2.findContours).
            id2label: Mapping from class ID to name.
            color_override: If set, use this color for all boxes/outlines.
            label_prefix: Prefix for label text.
            threshold_override: Override config draw_threshold.
            outline_width: Line width for contour outlines.
            draw_boxes: Whether to draw bounding box rectangles.
            draw_contours: Whether to draw mask contour outlines.

        Returns:
            The modified PIL Image.
        """
        draw = ImageDraw.Draw(image)
        threshold = (
            threshold_override
            if threshold_override is not None
            else float(self.config.model.draw_threshold)
        )

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
            x1, y1, x2, y2 = [int(v) for v in box]

            label_id = label.item() if torch.is_tensor(label) else int(label)

            if color_override:
                color = color_override
            else:
                color = self.PALETTE[label_id % len(self.PALETTE)]

            # Draw mask contour outlines (thick strokes)
            if (
                draw_contours
                and contours
                and i < len(contours)
                and contours[i] is not None
            ):
                for cnt in contours[i]:
                    pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
                    if len(pts) >= 2:
                        draw.line(pts + [pts[0]], fill=color, width=outline_width)

            # Draw bounding box
            if draw_boxes:
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # Draw label text
            class_name = (
                id2label.get(label_id)
                or id2label.get(str(label_id))
                or f"class_{label_id}"
            )
            label_text = f"{label_prefix}{class_name}"
            # Omit score when visualizing segmentation masks only
            if scores is not None and not (draw_contours and not draw_boxes):
                label_text += f": {score:.2f}"

            text_box = draw.textbbox((x1, y1), label_text, font=self.font)
            draw.rectangle(text_box, fill=color)
            draw.text((x1, y1), label_text, fill="white", font=self.font)

        return image

    def _visualize_aggregated_predictions(
        self, predictions_map, split: str = "val", suffix: str = ""
    ) -> None:
        """Save visualization images for aggregated epoch predictions.

        Draws GT and predicted segmentation outlines (thick contours) along
        with bounding boxes on the original images.  GT outlines come from
        ``val_segm_coco_gt`` / ``test_segm_coco_gt`` when available, falling
        back to bbox-only from ``val_coco_gt`` / ``test_coco_gt``.

        Args:
            predictions_map: Dict mapping image_id → list of prediction dicts
                with keys: image_id, category_id, bbox, score, segmentation (RLE).
            split: 'val' or 'test'.
            suffix: Appended to the output subdirectory (e.g. '_ema').
        """
        if not self.trainer.is_global_zero:
            return
        if not predictions_map:
            return
        if not self._should_visualize(split):
            return

        max_samples = self._get_visualization_limit()
        coco_gt = self.test_coco_gt if split == "test" else self.val_coco_gt
        segm_coco_gt = (
            self.test_segm_coco_gt if split == "test" else self.val_segm_coco_gt
        )
        image_root = self.test_image_root if split == "test" else self.val_image_root
        label_map = self.config.model.label_map
        viz_threshold = float(self.config.model.draw_threshold)

        if split == "val":
            save_dir = os.path.join(
                self.config.checkpointing.save_dir,
                self.config.checkpointing.visualization_dir,
                f"epoch_{(self.current_epoch + 1):03d}",
                f"val{suffix}",
            )
        else:
            save_dir = os.path.join(
                self.config.checkpointing.save_dir,
                self.config.checkpointing.visualization_dir,
                "test",
                f"test{suffix}",
            )

        os.makedirs(save_dir, exist_ok=True)
        saved_count = 0
        self.print(
            f"[VIZ] Saving {split.upper()}{suffix} visualizations to: {save_dir} "
            f"(max_samples={'all' if max_samples is None else max_samples})"
        )
        if max_samples is None:
            self.print(
                "[VIZ] WARNING: Unlimited visualization enabled. "
                "This can be very slow on large datasets."
            )

        for image_id in sorted(predictions_map.keys()):
            if max_samples is not None and saved_count >= max_samples:
                break

            pred_list = predictions_map[image_id]

            # Resolve image path
            img_info: dict[str, Any] = {"file_name": f"image_{image_id}.png"}
            if coco_gt and int(image_id) in getattr(coco_gt, "imgs", {}):
                img_info = coco_gt.imgs[int(image_id)]

            image_path = self._resolve_image_path(image_root, img_info.get("file_name"))
            if image_path is None:
                continue

            try:
                image_bbox = Image.open(image_path).convert("RGB")
                image_seg = Image.open(image_path).convert("RGB")
            except Exception:
                continue

            orig_h, orig_w = image_bbox.height, image_bbox.width

            # --- GT outlines from segm_coco_gt (preferred) ---
            gt_boxes: list[list[float]] = []
            gt_labels: list[int] = []
            gt_contours: list | None = None
            if segm_coco_gt:
                try:
                    gt_anns = segm_coco_gt.loadAnns(
                        segm_coco_gt.getAnnIds(imgIds=[int(image_id)])
                    )
                    gt_boxes = []
                    gt_labels = []
                    gt_contours = []
                    for ann in gt_anns:
                        x, y, w, h = ann["bbox"]
                        gt_boxes.append([x, y, x + w, y + h])
                        gt_labels.append(
                            self.coco_to_model.get(
                                int(ann["category_id"]), int(ann["category_id"])
                            )
                        )
                        rle = ann.get("segmentation")
                        if rle and isinstance(rle, dict) and "counts" in rle:
                            mask_np = self._decode_rle_mask(rle, size=(orig_h, orig_w))
                            gt_contours.append(self._mask_to_contours(mask_np))
                        else:
                            gt_contours.append(None)
                except Exception:
                    gt_boxes = []
                    gt_labels = []
                    gt_contours = None

            # Fallback: GT boxes only from coco_gt
            if not gt_boxes and coco_gt:
                try:
                    gt_anns = coco_gt.loadAnns(
                        coco_gt.getAnnIds(imgIds=[int(image_id)])
                    )
                    for ann in gt_anns:
                        x, y, w, h = ann["bbox"]
                        gt_boxes.append([x, y, x + w, y + h])
                        gt_labels.append(
                            self.coco_to_model.get(
                                int(ann["category_id"]), int(ann["category_id"])
                            )
                        )
                    gt_contours = [None] * len(gt_boxes)
                except Exception:
                    pass

            if gt_boxes:
                # Draw GT Bounding Boxes
                image_bbox = self.draw_outlines_and_boxes(
                    image_bbox,
                    gt_boxes,
                    gt_labels,
                    scores=None,
                    contours=None,
                    id2label=label_map,
                    color_override=(0, 255, 0),
                    outline_width=3,
                    draw_boxes=True,
                    draw_contours=False,
                )
                # Draw GT Segmentation Contours
                image_seg = self.draw_outlines_and_boxes(
                    image_seg,
                    gt_boxes,
                    gt_labels,
                    scores=None,
                    contours=gt_contours,
                    id2label=label_map,
                    color_override=(0, 255, 0),
                    outline_width=3,
                    draw_boxes=False,
                    draw_contours=True,
                )

            # --- Prediction outlines ---
            pred_boxes: list[list[float]] = []
            pred_labels: list[int] = []
            pred_scores: list[float] = []
            pred_contours: list = []
            pred_class_names: list[str] = []

            for pred in pred_list:
                score = float(pred["score"])
                if score < viz_threshold:
                    continue
                bx = pred["bbox"]  # [x, y, w, h]
                pred_boxes.append([bx[0], bx[1], bx[0] + bx[2], bx[1] + bx[3]])
                pred_labels.append(pred["category_id"])
                pred_scores.append(score)

                rle = pred.get("segmentation")
                if rle and isinstance(rle, dict) and "counts" in rle:
                    mask_np = self._decode_rle_mask(rle, size=(orig_h, orig_w))
                    pred_contours.append(self._mask_to_contours(mask_np))
                else:
                    pred_contours.append(None)

                class_name = (
                    label_map.get(int(pred["category_id"]))
                    or label_map.get(str(int(pred["category_id"])))
                    or str(pred["category_id"])
                )
                pred_class_names.append(class_name)

            if pred_boxes:
                # Draw Pred Bounding Boxes
                image_bbox = self.draw_outlines_and_boxes(
                    image_bbox,
                    pred_boxes,
                    pred_labels,
                    scores=pred_scores,
                    contours=None,
                    id2label=label_map,
                    color_override=(255, 0, 0),
                    outline_width=3,
                    draw_boxes=True,
                    draw_contours=False,
                )
                # Draw Pred Segmentation Contours
                image_seg = self.draw_outlines_and_boxes(
                    image_seg,
                    pred_boxes,
                    pred_labels,
                    scores=pred_scores,
                    contours=pred_contours,
                    id2label=label_map,
                    color_override=(255, 0, 0),
                    outline_width=3,
                    draw_boxes=False,
                    draw_contours=True,
                )

            # --- Count overlay (top-right): pred / gt per class ---
            gt_counts = Counter(
                label_map.get(int(lbl)) or label_map.get(str(int(lbl))) or str(lbl)
                for lbl in gt_labels
            )
            pred_counts = Counter(pred_class_names)

            def _draw_counts(img: Image.Image) -> None:
                draw = ImageDraw.Draw(img)
                text_y = 10
                line_height = 24
                all_classes = set(gt_counts.keys()) | set(pred_counts.keys())
                for cls_name in sorted(all_classes):
                    parts = [
                        (f"{cls_name}: ", "white"),
                        (f"{pred_counts[cls_name]}", "red"),
                        ("/", "white"),
                        (f"{gt_counts[cls_name]}", "green"),
                    ]
                    total_width = 0
                    for text, _ in parts:
                        bbox = draw.textbbox((0, 0), text, font=self.font)
                        total_width += bbox[2] - bbox[0]

                    current_x = img.width - total_width - 10
                    for text, color in parts:
                        draw.text(
                            (current_x + 1, text_y + 1),
                            text,
                            fill="black",
                            font=self.font,
                        )
                        draw.text((current_x, text_y), text, fill=color, font=self.font)
                        bbox = draw.textbbox((0, 0), text, font=self.font)
                        current_x += bbox[2] - bbox[0]
                    text_y += line_height

            _draw_counts(image_bbox)
            _draw_counts(image_seg)

            # --- Save with class-name-prefixed filename ---
            detected_classes = sorted(list(set(pred_class_names)))
            if detected_classes:
                class_prefix = "_".join(detected_classes)
                prefix = f"image_{class_prefix}_"
            else:
                prefix = "image_no_detections_"

            original_filename = os.path.basename(
                img_info.get("file_name", f"image_{image_id}.png")
            )
            new_filename = f"{prefix}{original_filename}"
            new_filename = new_filename.replace("image_image_", "image_")

            # Save separate bbox and seg images
            bbox_save_path = os.path.join(save_dir, f"bbox_{new_filename}")
            seg_save_path = os.path.join(save_dir, f"seg_{new_filename}")

            image_bbox.save(bbox_save_path)
            image_seg.save(seg_save_path)

            saved_count += 1

        self.print(
            f"[VIZ] {split.upper()}{suffix} saved {saved_count} visualizations to: {save_dir}"
        )
