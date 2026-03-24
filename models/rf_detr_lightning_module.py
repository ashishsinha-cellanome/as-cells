import os
from collections import Counter
from pathlib import Path

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
from utils.sahi_eval import run_sahi_sliced_eval
import torchvision.transforms.functional as F


class RFDETRLightningModule(pl.LightningModule):
    """PyTorch Lightning module for RF-DETR training/evaluation."""

    def __init__(
        self,
        model,
        criterion,
        postprocess,
        config,
        model_to_coco=None,
        val_coco_gt=None,
        test_coco_gt=None,
        val_image_root=None,
        test_image_root=None,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.postprocess = postprocess
        self.config = config
        self.model_to_coco = {int(k): int(v) for k, v in (model_to_coco or {}).items()}
        self.coco_to_model = {int(v): int(k) for k, v in self.model_to_coco.items()}
        self.val_coco_gt = val_coco_gt
        self.test_coco_gt = test_coco_gt
        self.val_image_root = val_image_root
        self.test_image_root = test_image_root

        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.validation_step_outputs_sliced = []
        self.test_step_outputs_sliced = []

        if hasattr(self.config.model, "ema") and self.config.model.ema.enabled:
            self.validation_step_outputs_ema = []
            self.test_step_outputs_ema = []
            self.validation_step_outputs_sliced_ema = []
            self.test_step_outputs_sliced_ema = []

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

    def forward(self, samples, targets=None):
        return self.model(samples, targets)

    def _move_targets(self, targets):
        return [{k: v.to(self.device) for k, v in target.items()} for target in targets]

    def _compute_loss(self, outputs, targets):
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
        return loss, loss_dict, weight_dict

    def _log_loss_dict(self, split: str, loss_dict, weight_dict, batch_size: int):
        """
        Log all loss terms from RF-DETR criterion.
        For weighted terms we log both unscaled and scaled values.
        """
        for key, value in loss_dict.items():
            self.log(
                f"{split}/{key}_unscaled",
                value,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            if key in weight_dict:
                self.log(
                    f"{split}/{key}",
                    value * weight_dict[key],
                    batch_size=batch_size,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
            else:
                self.log(
                    f"{split}/{key}",
                    value,
                    batch_size=batch_size,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

    def training_step(self, batch, batch_idx):
        samples, targets = batch
        batch_size = (
            int(samples.shape[0]) if isinstance(samples, torch.Tensor) else len(targets)
        )
        samples = samples.to(self.device)
        targets = self._move_targets(targets)

        outputs = self.model(samples, targets)
        loss, loss_dict, weight_dict = self._compute_loss(outputs, targets)

        self.log(
            "train/loss",
            loss,
            batch_size=batch_size,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self._log_loss_dict("train", loss_dict, weight_dict, batch_size=batch_size)

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
        if self.trainer.is_global_zero:
            self.print(
                f"[VAL] Starting validation for epoch {self.current_epoch + 1}..."
            )

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        samples, targets = batch
        batch_size = (
            int(samples.shape[0]) if isinstance(samples, torch.Tensor) else len(targets)
        )
        samples = samples.to(self.device)
        targets = self._move_targets(targets)

        outputs = self.model(samples)
        loss, loss_dict, weight_dict = self._compute_loss(outputs, targets)
        self.log(
            "val/loss",
            loss,
            batch_size=batch_size,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self._log_loss_dict("val", loss_dict, weight_dict, batch_size=batch_size)

        eval_mode = self.config.get("eval_inference", {}).get("mode", "whole")

        # Whole Image Baseline Process (For Loss + Fallback)
        predictions, image_ids = self._collect_batch_predictions(outputs, targets)

        if eval_mode in ["whole", "both"]:
            self.validation_step_outputs.append(
                {"predictions": predictions, "image_ids": image_ids}
            )

        if eval_mode in ["sliced", "both"]:
            sliced_predictions = {}
            for i, target in enumerate(targets):
                img_id = int(target["image_id"].item())
                img_path = self._get_image_path(img_id, "val")

                if not img_path or not os.path.exists(img_path):
                    self.print(
                        f"[Val] WARNING: Cannot find image {img_path} for SAHI. Falling back to whole image."
                    )
                    sliced_predictions[img_id] = predictions[img_id]
                    continue

                def predict_fn(image_np):
                    # Convert numpy array to tensor (H, W, 3) -> (3, H, W)
                    img_tensor = (
                        torch.from_numpy(image_np).permute(2, 0, 1).contiguous()
                    )
                    img_tensor = F.convert_image_dtype(img_tensor, dtype=torch.float32)
                    img_tensor = img_tensor.to(self.device)
                    if next(self.model.parameters()).dtype == torch.float16:
                        img_tensor = img_tensor.half()
                    elif next(self.model.parameters()).dtype == torch.bfloat16:
                        img_tensor = img_tensor.bfloat16()

                    with torch.no_grad():
                        out = self.model([img_tensor])

                    orig_size = torch.tensor(
                        [[image_np.shape[0], image_np.shape[1]]], device=self.device
                    )
                    post = self.postprocess(out, orig_size)[0]
                    return to_cpu_device(post)

                img_pil = Image.open(img_path).convert("RGB")
                sahi_cfg = self.config.get("eval_inference", {}).get("sahi", {})
                # For RF-DETR we often pad/resize. Just use config input size or 640
                input_size = getattr(self.config.data, "model_input_size", 640)

                preds = run_sahi_sliced_eval(img_pil, predict_fn, sahi_cfg, input_size)
                sliced_predictions[img_id] = preds

            self.validation_step_outputs_sliced.append(
                {"predictions": sliced_predictions, "image_ids": image_ids}
            )

        # EMA validation
        # EMA validation
        from utils.ema import EMACallback

        ema_callback = next(
            (cb for cb in self.trainer.callbacks if isinstance(cb, EMACallback)), None
        )
        if ema_callback and ema_callback.ema_model:
            ema_outputs = ema_callback.ema_model.module(samples)
            ema_predictions, ema_image_ids = self._collect_batch_predictions(
                ema_outputs, targets
            )

            if eval_mode in ["whole", "both"]:
                self.validation_step_outputs_ema.append(
                    {"predictions": ema_predictions, "image_ids": ema_image_ids}
                )

            if eval_mode in ["sliced", "both"]:
                sliced_ema_predictions = {}
                for i, target in enumerate(targets):
                    img_id = int(target["image_id"].item())
                    img_path = self._get_image_path(img_id, "val")

                    if not img_path or not os.path.exists(img_path):
                        sliced_ema_predictions[img_id] = ema_predictions[img_id]
                        continue

                    def predict_fn_ema(image_np):
                        img_tensor = (
                            torch.from_numpy(image_np).permute(2, 0, 1).contiguous()
                        )
                        img_tensor = F.convert_image_dtype(
                            img_tensor, dtype=torch.float32
                        )
                        img_tensor = img_tensor.to(self.device)
                        if (
                            next(ema_callback.ema_model.module.parameters()).dtype
                            == torch.float16
                        ):
                            img_tensor = img_tensor.half()
                        elif (
                            next(ema_callback.ema_model.module.parameters()).dtype
                            == torch.bfloat16
                        ):
                            img_tensor = img_tensor.bfloat16()

                        with torch.no_grad():
                            out = ema_callback.ema_model.module([img_tensor])

                        orig_size = torch.tensor(
                            [[image_np.shape[0], image_np.shape[1]]], device=self.device
                        )
                        post = self.postprocess(out, orig_size)[0]
                        return to_cpu_device(post)

                    img_pil = Image.open(img_path).convert("RGB")
                    sahi_cfg = self.config.get("eval_inference", {}).get("sahi", {})
                    input_size = getattr(self.config.data, "model_input_size", 640)

                    preds = run_sahi_sliced_eval(
                        img_pil, predict_fn_ema, sahi_cfg, input_size
                    )
                    sliced_ema_predictions[img_id] = preds

                self.validation_step_outputs_sliced_ema.append(
                    {"predictions": sliced_ema_predictions, "image_ids": ema_image_ids}
                )

        return {"predictions": predictions, "image_ids": image_ids}

    def on_validation_epoch_end(self):
        viz_predictions = None

        # Added base_prefix and suffix parameters
        def _compute_and_log(outputs_list, prefix_name, base_prefix, suffix=""):
            all_outputs = gather_outputs_across_processes(outputs_list)
            merged = self._merge_predictions_map(all_outputs)
            metrics = {}
            if self.trainer.is_global_zero:
                predictions = []
                image_ids = []
                for batch_out in all_outputs:
                    predictions.extend(
                        convert_preds_to_coco(
                            batch_out["predictions"], model_to_coco=self.model_to_coco
                        )
                    )
                    image_ids.extend(batch_out["image_ids"])

                if predictions:
                    metrics = compute_coco_metrics(
                        coco_gt=self.val_coco_gt,
                        predictions=predictions,
                        image_ids=sorted(set(image_ids)),
                        max_detections=int(self.config.model.max_detections),
                        label_map=self.config.model.label_map,
                        prefix=f"{prefix_name} performance",
                    )
                else:
                    metrics = {"map": 0.0, "map_50": 0.0, "map_75": 0.0}

            metrics = broadcast_object(metrics, src=0)
            for key, value in metrics.items():
                # Formats as: val/map_ema instead of val_ema/map
                self.log(
                    f"{base_prefix}/{key}{suffix}",
                    value,
                    prog_bar=(key in {"map", "map_50"}),
                    sync_dist=True,
                )
                if key == "map":
                    # Formats as: val_map_ema instead of val_ema_map
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
            self.print(
                f"[VAL] Completed validation for epoch {self.current_epoch + 1}."
            )

    def on_test_epoch_start(self):
        """Reset test visualization counter."""
        self.test_viz_counter = 0

    def on_test_start(self):
        """
        Ensure dtype/device are compatible with the configured precision mode.
        For mixed precision, keep FP32 weights and rely on autocast.
        For true precision modes, cast model weights explicitly.
        """
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
            self.print(
                f"[INFO] Cast model weights to {target_dtype} for precision={self.trainer.precision}."
            )

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        samples, targets = batch
        samples = samples.to(self.device)
        targets = self._move_targets(targets)

        outputs = self.model(samples)
        predictions, image_ids = self._collect_batch_predictions(outputs, targets)

        eval_mode = self.config.get("eval_inference", {}).get("mode", "whole")

        if eval_mode in ["whole", "both"]:
            self.test_step_outputs.append(
                {"predictions": predictions, "image_ids": image_ids}
            )

        if eval_mode in ["sliced", "both"]:
            sliced_predictions = {}
            for i, target in enumerate(targets):
                img_id = int(target["image_id"].item())
                img_path = self._get_image_path(img_id, "test")

                if not img_path or not os.path.exists(img_path):
                    self.print(
                        f"[Test] WARNING: Cannot find image {img_path} for SAHI. Falling back to whole image."
                    )
                    sliced_predictions[img_id] = predictions[img_id]
                    continue

                def predict_fn(image_np):
                    img_tensor = (
                        torch.from_numpy(image_np).permute(2, 0, 1).contiguous()
                    )
                    img_tensor = F.convert_image_dtype(img_tensor, dtype=torch.float32)
                    img_tensor = img_tensor.to(self.device)
                    if next(self.model.parameters()).dtype == torch.float16:
                        img_tensor = img_tensor.half()
                    elif next(self.model.parameters()).dtype == torch.bfloat16:
                        img_tensor = img_tensor.bfloat16()

                    with torch.no_grad():
                        out = self.model([img_tensor])

                    orig_size = torch.tensor(
                        [[image_np.shape[0], image_np.shape[1]]], device=self.device
                    )
                    post = self.postprocess(out, orig_size)[0]
                    return to_cpu_device(post)

                img_pil = Image.open(img_path).convert("RGB")
                sahi_cfg = self.config.get("eval_inference", {}).get("sahi", {})
                input_size = getattr(self.config.data, "model_input_size", 640)

                preds = run_sahi_sliced_eval(img_pil, predict_fn, sahi_cfg, input_size)
                sliced_predictions[img_id] = preds

            self.test_step_outputs_sliced.append(
                {"predictions": sliced_predictions, "image_ids": image_ids}
            )

        # EMA validation during test
        from utils.ema import EMACallback

        ema_callback = next(
            (cb for cb in self.trainer.callbacks if isinstance(cb, EMACallback)), None
        )
        if ema_callback and ema_callback.ema_model:
            ema_outputs = ema_callback.ema_model.module(samples)
            ema_predictions, ema_image_ids = self._collect_batch_predictions(
                ema_outputs, targets
            )

            if eval_mode in ["whole", "both"]:
                self.test_step_outputs_ema.append(
                    {"predictions": ema_predictions, "image_ids": ema_image_ids}
                )

            if eval_mode in ["sliced", "both"]:
                sliced_ema_predictions = {}
                for i, target in enumerate(targets):
                    img_id = int(target["image_id"].item())
                    img_path = self._get_image_path(img_id, "test")

                    if not img_path or not os.path.exists(img_path):
                        sliced_ema_predictions[img_id] = ema_predictions[img_id]
                        continue

                    def predict_fn_ema(image_np):
                        img_tensor = (
                            torch.from_numpy(image_np).permute(2, 0, 1).contiguous()
                        )
                        img_tensor = F.convert_image_dtype(
                            img_tensor, dtype=torch.float32
                        )
                        img_tensor = img_tensor.to(self.device)
                        if (
                            next(ema_callback.ema_model.module.parameters()).dtype
                            == torch.float16
                        ):
                            img_tensor = img_tensor.half()
                        elif (
                            next(ema_callback.ema_model.module.parameters()).dtype
                            == torch.bfloat16
                        ):
                            img_tensor = img_tensor.bfloat16()

                        with torch.no_grad():
                            out = ema_callback.ema_model.module([img_tensor])

                        orig_size = torch.tensor(
                            [[image_np.shape[0], image_np.shape[1]]], device=self.device
                        )
                        post = self.postprocess(out, orig_size)[0]
                        return to_cpu_device(post)

                    img_pil = Image.open(img_path).convert("RGB")
                    sahi_cfg = self.config.get("eval_inference", {}).get("sahi", {})
                    input_size = getattr(self.config.data, "model_input_size", 640)

                    preds = run_sahi_sliced_eval(
                        img_pil, predict_fn_ema, sahi_cfg, input_size
                    )
                    sliced_ema_predictions[img_id] = preds

                self.test_step_outputs_sliced_ema.append(
                    {"predictions": sliced_ema_predictions, "image_ids": ema_image_ids}
                )

        return {"predictions": predictions, "image_ids": image_ids}

    def on_test_epoch_end(self):
        viz_predictions = None

        # Added base_prefix and suffix parameters
        def _compute_and_log(outputs_list, prefix_name, base_prefix, suffix=""):
            all_outputs = gather_outputs_across_processes(outputs_list)
            merged = self._merge_predictions_map(all_outputs)
            metrics = {}
            if self.trainer.is_global_zero:
                predictions = []
                image_ids = []
                for batch_out in all_outputs:
                    predictions.extend(
                        convert_preds_to_coco(
                            batch_out["predictions"], model_to_coco=self.model_to_coco
                        )
                    )
                    image_ids.extend(batch_out["image_ids"])

                if predictions:
                    metrics = compute_coco_metrics(
                        coco_gt=self.test_coco_gt,
                        predictions=predictions,
                        image_ids=sorted(set(image_ids)),
                        max_detections=int(self.config.model.max_detections),
                        label_map=self.config.model.label_map,
                        prefix=f"{prefix_name} performance",
                    )
                else:
                    metrics = {"map": 0.0, "map_50": 0.0, "map_75": 0.0}

            metrics = broadcast_object(metrics, src=0)
            for key, value in metrics.items():
                # Formats as: test/map_ema instead of test_ema/map
                self.log(
                    f"{base_prefix}/{key}{suffix}",
                    value,
                    prog_bar=(key in {"map", "map_50"}),
                    sync_dist=True,
                )

            outputs_list.clear()
            return merged

        # 1. Standard Whole
        if self.test_step_outputs:
            viz_predictions = _compute_and_log(
                self.test_step_outputs, "Test", "test", ""
            )

        # 2. Standard Sliced
        if self.test_step_outputs_sliced:
            viz_predictions = _compute_and_log(
                self.test_step_outputs_sliced, "Test Sliced", "test", "_sliced"
            )

        # 3. EMA Whole
        if hasattr(self, "test_step_outputs_ema") and self.test_step_outputs_ema:
            viz_predictions = _compute_and_log(
                self.test_step_outputs_ema, "Test EMA", "test", "_ema"
            )

        # 4. EMA Sliced
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

    def _resolve_image_path(self, image_root, file_name):
        if not file_name:
            return None

        candidate = Path(file_name)
        if candidate.is_absolute() and candidate.exists():
            return candidate

        if image_root:
            root_path = Path(image_root)
            candidate = root_path / file_name
            if candidate.exists():
                return candidate
            candidate = root_path / os.path.basename(file_name)
            if candidate.exists():
                return candidate

        return None

    def _visualize_aggregated_predictions(self, predictions_map, split="val"):
        if not self.trainer.is_global_zero:
            return
        if not predictions_map:
            return
        if not self._should_visualize(split):
            return

        max_samples = self._get_visualization_limit()
        coco_gt = self.test_coco_gt if split == "test" else self.val_coco_gt
        image_root = self.test_image_root if split == "test" else self.val_image_root
        label_map = self.config.model.label_map
        viz_threshold = float(self.config.model.draw_threshold)

        if split == "val":
            save_dir = os.path.join(
                self.config.checkpointing.save_dir,
                self.config.checkpointing.visualization_dir,
                f"epoch_{(self.current_epoch + 1):03d}",
                "val",
            )
        else:
            save_dir = os.path.join(
                self.config.checkpointing.save_dir,
                self.config.checkpointing.visualization_dir,
                "test",
            )

        os.makedirs(save_dir, exist_ok=True)
        saved_count = 0
        self.print(
            f"[VIZ] Saving {split.upper()} visualizations to: {save_dir} "
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

            img_info = {"file_name": f"image_{image_id}.png"}
            if coco_gt and int(image_id) in getattr(coco_gt, "imgs", {}):
                img_info = coco_gt.imgs[int(image_id)]

            image_path = self._resolve_image_path(image_root, img_info.get("file_name"))
            if image_path is None:
                continue

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                continue

            gt_boxes = []
            gt_labels = []
            if coco_gt:
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
                except Exception:
                    gt_boxes = []
                    gt_labels = []

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
                    threshold_override=viz_threshold,
                )

            gt_counts = Counter(
                [
                    label_map.get(int(l)) or label_map.get(str(int(l))) or str(l)
                    for l in gt_labels
                ]
            )
            pred_counts = Counter(pred_class_names)

            draw = ImageDraw.Draw(image)
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

                current_x = image.width - total_width - 10
                for text, color in parts:
                    draw.text(
                        (current_x + 1, text_y + 1), text, fill="black", font=self.font
                    )
                    draw.text((current_x, text_y), text, fill=color, font=self.font)
                    bbox = draw.textbbox((0, 0), text, font=self.font)
                    current_x += bbox[2] - bbox[0]
                text_y += line_height

            original_filename = os.path.basename(
                img_info.get("file_name", f"image_{image_id}.png")
            )
            save_path = os.path.join(
                save_dir, f"image_{int(image_id)}_{original_filename}"
            )
            image.save(save_path)
            saved_count += 1
            # if saved_count % 500 == 0:
            #     self.print(f"[VIZ] {split.upper()} progress: saved {saved_count} images...")

        self.print(
            f"[VIZ] {split.upper()} saved {saved_count} aggregated visualizations to: {save_dir}"
        )

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

    def draw_boxes(
        self,
        image,
        boxes,
        labels,
        scores=None,
        id2label=None,
        color_override=None,
        label_prefix="",
        threshold_override=None,
    ):
        """Draws bounding boxes on a PIL image."""
        draw = ImageDraw.Draw(image)
        threshold = (
            threshold_override
            if threshold_override is not None
            else self.config.model.draw_threshold
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

            label_id = label.item() if torch.is_tensor(label) else int(label)

            if color_override:
                color = color_override
            else:
                color = self.PALETTE[label_id % len(self.PALETTE)]

            draw.rectangle(box, outline=color, width=3)

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
        self, save_dir, predictions_map, samples, targets, counter, split="val"
    ):
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
        if hasattr(samples, "tensors"):
            pixel_values = samples.tensors
        else:
            pixel_values = samples

        mean = torch.tensor([0.485, 0.456, 0.406], device=pixel_values.device).view(
            1, 3, 1, 1
        )
        std = torch.tensor([0.229, 0.224, 0.225], device=pixel_values.device).view(
            1, 3, 1, 1
        )
        unnormalized_images = torch.clamp((pixel_values * std) + mean, 0, 1)

        for i, target in enumerate(
            tqdm(targets, desc="Visualizing Batch", leave=False)
        ):
            if max_samples != -1 and counter >= max_samples:
                break

            image_id = int(target["image_id"].item())
            image_tensor = unnormalized_images[i]

            # Use original size if available in target, else tensor size
            if "orig_size" in target:
                orig_h, orig_w = target["orig_size"].tolist()
            else:
                orig_h, orig_w = image_tensor.shape[1], image_tensor.shape[2]

            image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(
                "uint8"
            )
            resized_image_np = cv2.resize(
                image_np, (int(orig_w), int(orig_h)), interpolation=cv2.INTER_LINEAR
            )
            image = Image.fromarray(resized_image_np)

            # Get image metadata from COCO GT for filename
            img_info = {"file_name": f"image_{image_id}.png"}
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
                        x, y, w, h = ann["bbox"]
                        gt_boxes.append([x, y, x + w, y + h])
                        gt_labels.append(ann["category_id"])

                    image = self.draw_boxes(
                        image,
                        gt_boxes,
                        gt_labels,
                        scores=None,
                        id2label=self.config.model.label_map,
                        color_override=(0, 255, 0),
                        label_prefix="",
                    )
                except Exception:
                    pass

            # --- 2. Preds ---
            pred_class_names = []
            if image_id in predictions_map:
                preds = predictions_map[image_id]

                if "boxes" in preds and len(preds["boxes"]) > 0:
                    # Filter
                    valid_indices = preds["scores"] >= viz_threshold
                    valid_labels = preds["labels"][valid_indices]
                    label_map = self.config.model.label_map
                    for l in valid_labels:
                        l_item = l.item() if torch.is_tensor(l) else int(l)
                        name = (
                            label_map.get(int(l_item))
                            or label_map.get(str(l_item))
                            or str(l_item)
                        )
                        pred_class_names.append(name)

                    image = self.draw_boxes(
                        image,
                        preds["boxes"],
                        preds["labels"],
                        preds["scores"],
                        id2label=self.config.model.label_map,
                        color_override=(255, 0, 0),
                        label_prefix="",
                        threshold_override=viz_threshold,
                    )

            # --- 3. Counts & Filename ---
            from collections import Counter

            label_map = self.config.model.label_map

            gt_counts = Counter(
                [
                    label_map.get(int(l)) or label_map.get(str(l)) or str(l)
                    for l in gt_labels
                ]
            )
            pred_counts = Counter(pred_class_names)

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

            detected_classes = sorted(list(set(pred_class_names)))
            if detected_classes:
                class_str = "_".join(detected_classes)
                prefix = f"image_{class_str}_"
            else:
                prefix = "image_no_detections_"

            original_filename = os.path.basename(img_info["file_name"])
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

    def lr_scheduler_step(self, scheduler, metric):
        """
        Keep scheduler progression aligned with real optimizer updates.
        This avoids stepping LR when AMP/overflow skips optimizer.step().
        """
        optimizer = getattr(scheduler, "optimizer", None)
        optimizer_has_stepped = (
            optimizer is None or getattr(optimizer, "_step_count", 0) > 0
        )
        if not optimizer_has_stepped:
            return

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if metric is not None:
                scheduler.step(metric)
            return

        scheduler.step()
