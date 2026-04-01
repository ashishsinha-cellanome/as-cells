import os
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
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
                    {"size": encoded["size"], "counts": encoded["counts"].encode("utf-8")}
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
    ) -> None:
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
                    prefix=prefix_label or f"{prefix.capitalize()} detection performance",
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

        outputs_list.clear()

    def on_validation_epoch_end(self):
        self._evaluate_and_log_epoch(
            outputs_list=self.validation_step_outputs,
            bbox_gt=self.val_coco_gt,
            segm_gt=self.val_segm_coco_gt,
            prefix="val",
        )
        if hasattr(self, "validation_step_outputs_ema") and self.validation_step_outputs_ema:
            self._evaluate_and_log_epoch(
                outputs_list=self.validation_step_outputs_ema,
                bbox_gt=self.val_coco_gt,
                segm_gt=self.val_segm_coco_gt,
                prefix="val",
                suffix="_ema",
                prefix_label="Val EMA detection performance",
            )

    def on_test_epoch_end(self):
        self._evaluate_and_log_epoch(
            outputs_list=self.test_step_outputs,
            bbox_gt=self.test_coco_gt,
            segm_gt=self.test_segm_coco_gt,
            prefix="test",
        )
        if hasattr(self, "test_step_outputs_ema") and self.test_step_outputs_ema:
            self._evaluate_and_log_epoch(
                outputs_list=self.test_step_outputs_ema,
                bbox_gt=self.test_coco_gt,
                segm_gt=self.test_segm_coco_gt,
                prefix="test",
                suffix="_ema",
                prefix_label="Test EMA detection performance",
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

    def _resolve_image_path(self, image_root: str | None, file_name: str) -> Path | None:
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
