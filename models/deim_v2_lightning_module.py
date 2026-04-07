import os
from collections import Counter
from pathlib import Path

import cv2
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import torch
import pytorch_lightning as pl
import torchvision.transforms.functional as F

from omegaconf import OmegaConf

from utils.coco_eval_utils import (
    to_cpu_device,
    convert_preds_to_coco,
    gather_outputs_across_processes,
    broadcast_object,
    compute_coco_metrics,
)
from utils.sahi_eval import run_sahi_sliced_eval

from DEIMv2.engine.backbone.dinov3_adapter import DINOv3STAs
from DEIMv2.engine.deim.hybrid_encoder import HybridEncoder
from DEIMv2.engine.deim.deim_decoder import DEIMTransformer
from DEIMv2.engine.deim.deim import DEIM
from DEIMv2.engine.deim.deim_criterion import DEIMCriterion
from DEIMv2.engine.deim.matcher import HungarianMatcher
from DEIMv2.engine.deim.postprocessor import PostProcessor


class DeimV2LightningModule(pl.LightningModule):
    """PyTorch Lightning module for DEIMv2 training/evaluation."""

    def __init__(
        self,
        config,
        model_to_coco=None,
        val_coco_gt=None,
        test_coco_gt=None,
        val_image_root=None,
        test_image_root=None,
    ):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=["config", "val_coco_gt", "test_coco_gt"])

        self.model_to_coco = {int(k): int(v) for k, v in (model_to_coco or {}).items()}
        self.coco_to_model = {int(v): int(k) for k, v in self.model_to_coco.items()}
        self.val_coco_gt = val_coco_gt
        self.test_coco_gt = test_coco_gt
        self.val_image_root = val_image_root
        self.test_image_root = test_image_root

        # Natively instantiate DEIMv2 components
        cfg = config.model.deimv2
        num_classes = len(config.model.label_map)

        backbone_kwargs = OmegaConf.to_container(cfg.backbone, resolve=True)
        self.backbone = DINOv3STAs(**backbone_kwargs)

        encoder_kwargs = OmegaConf.to_container(cfg.encoder, resolve=True)
        self.encoder = HybridEncoder(**encoder_kwargs)

        decoder_kwargs = OmegaConf.to_container(cfg.decoder, resolve=True)
        self.decoder = DEIMTransformer(num_classes=num_classes, **decoder_kwargs)

        self.model = DEIM(self.backbone, self.encoder, self.decoder)

        matcher_kwargs = OmegaConf.to_container(cfg.criterion.matcher, resolve=True)
        self.matcher = HungarianMatcher(use_focal_loss=True, **matcher_kwargs)

        crit_kwargs = OmegaConf.to_container(cfg.criterion, resolve=True)
        crit_kwargs.pop("matcher")
        self.criterion = DEIMCriterion(
            matcher=self.matcher, num_classes=num_classes, **crit_kwargs
        )

        post_kwargs = OmegaConf.to_container(cfg.postprocessor, resolve=True)
        self.postprocessor = PostProcessor(num_classes=num_classes, **post_kwargs)

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
        # We need to make sure targets have the format expected by DEIMv2
        # It expects labels to be int64 for example, but we'll let PyTorch handle it where possible
        return [
            {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in target.items()
            }
            for target in targets
        ]

    def _compute_loss(self, outputs, targets):
        loss_dict = self.criterion(outputs, targets, epoch=self.current_epoch)
        weight_dict = self.criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
        return loss, loss_dict, weight_dict

    def _log_loss_dict(self, split: str, loss_dict, weight_dict, batch_size: int):
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
        orig_sizes = torch.stack(
            [
                torch.tensor([t["orig_size"][1], t["orig_size"][0]])
                if isinstance(t["orig_size"], (list, tuple))
                or (
                    isinstance(t["orig_size"], torch.Tensor)
                    and t["orig_size"].dim() == 1
                )
                else t["orig_size"]
                for t in targets
            ],
            dim=0,
        )
        if orig_sizes.shape[-1] == 2 and hasattr(self, "device"):
            orig_sizes = orig_sizes.to(self.device)
        post = self.postprocessor(outputs, orig_sizes)
        post = [to_cpu_device(pred) for pred in post]
        result_map = {
            int(
                target["image_id"].item()
                if torch.is_tensor(target["image_id"])
                else target["image_id"]
            ): pred
            for target, pred in zip(targets, post)
        }
        image_ids = [
            int(
                target["image_id"].item()
                if torch.is_tensor(target["image_id"])
                else target["image_id"]
            )
            for target in targets
        ]
        return result_map, image_ids

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        samples, targets = batch
        batch_size = (
            int(samples.shape[0]) if isinstance(samples, torch.Tensor) else len(targets)
        )
        samples = samples.to(self.device)
        targets = self._move_targets(targets)

        outputs = self.model(samples)

        # Skip computing validation loss since DEIMCriterion expects aux_outputs which are not generated in eval() mode

        eval_mode = self.config.get("eval_inference", {}).get("mode", "whole")

        predictions, image_ids = self._collect_batch_predictions(outputs, targets)

        if eval_mode in ["whole", "both"]:
            self.validation_step_outputs.append(
                {"predictions": predictions, "image_ids": image_ids}
            )

        if eval_mode in ["sliced", "both"]:
            sliced_predictions = {}
            for i, target in enumerate(targets):
                img_id = int(
                    target["image_id"].item()
                    if torch.is_tensor(target["image_id"])
                    else target["image_id"]
                )
                img_path = self._get_image_path(img_id, "val")

                if not img_path or not os.path.exists(img_path):
                    self.print(
                        f"[Val] WARNING: Cannot find image {img_path} for SAHI. Falling back to whole image."
                    )
                    sliced_predictions[img_id] = predictions[img_id]
                    continue

                def predict_fn(image_np):
                    img_tensor = (
                        torch.from_numpy(image_np).permute(2, 0, 1).contiguous()
                    )
                    img_tensor = F.convert_image_dtype(
                        img_tensor, dtype=torch.float32
                    ).to(self.device)
                    if next(self.model.parameters()).dtype == torch.float16:
                        img_tensor = img_tensor.half()
                    elif next(self.model.parameters()).dtype == torch.bfloat16:
                        img_tensor = img_tensor.bfloat16()

                    with torch.no_grad():
                        out = self.model(img_tensor.unsqueeze(0))

                    orig_size = torch.tensor(
                        [[image_np.shape[0], image_np.shape[1]]], device=self.device
                    )
                    post = self.postprocessor(out, orig_size)[0]
                    return to_cpu_device(post)

                img_pil = Image.open(img_path).convert("RGB")
                sahi_cfg = self.config.get("eval_inference", {}).get("sahi", {})
                input_size = getattr(self.config.data, "model_input_size", 640)
                preds = run_sahi_sliced_eval(img_pil, predict_fn, sahi_cfg, input_size)
                sliced_predictions[img_id] = preds

            self.validation_step_outputs_sliced.append(
                {"predictions": sliced_predictions, "image_ids": image_ids}
            )

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

        return {"predictions": predictions, "image_ids": image_ids}

    def on_validation_epoch_end(self):
        def _compute_and_log(outputs_list, prefix_name, base_prefix, suffix=""):
            all_outputs = gather_outputs_across_processes(outputs_list)
            merged = {}
            for batch_out in all_outputs:
                merged.update(batch_out.get("predictions", {}))

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

        if self.validation_step_outputs:
            _compute_and_log(self.validation_step_outputs, "Val", "val", "")
        if self.validation_step_outputs_sliced:
            _compute_and_log(
                self.validation_step_outputs_sliced, "Val Sliced", "val", "_sliced"
            )
        if (
            hasattr(self, "validation_step_outputs_ema")
            and self.validation_step_outputs_ema
        ):
            _compute_and_log(self.validation_step_outputs_ema, "Val EMA", "val", "_ema")
        if (
            hasattr(self, "validation_step_outputs_sliced_ema")
            and self.validation_step_outputs_sliced_ema
        ):
            _compute_and_log(
                self.validation_step_outputs_sliced_ema,
                "Val Sliced EMA",
                "val",
                "_sliced_ema",
            )

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        samples, targets = batch
        batch_size = (
            int(samples.shape[0]) if isinstance(samples, torch.Tensor) else len(targets)
        )
        samples = samples.to(self.device)
        targets = self._move_targets(targets)

        outputs = self.model(samples)

        eval_mode = self.config.get("eval_inference", {}).get("mode", "whole")

        predictions, image_ids = self._collect_batch_predictions(outputs, targets)

        if eval_mode in ["whole", "both"]:
            self.test_step_outputs.append(
                {"predictions": predictions, "image_ids": image_ids}
            )

        if eval_mode in ["sliced", "both"]:
            sliced_predictions = {}
            for i, target in enumerate(targets):
                img_id = int(
                    target["image_id"].item()
                    if torch.is_tensor(target["image_id"])
                    else target["image_id"]
                )
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
                    img_tensor = F.convert_image_dtype(
                        img_tensor, dtype=torch.float32
                    ).to(self.device)
                    if next(self.model.parameters()).dtype == torch.float16:
                        img_tensor = img_tensor.half()
                    elif next(self.model.parameters()).dtype == torch.bfloat16:
                        img_tensor = img_tensor.bfloat16()

                    with torch.no_grad():
                        out = self.model(img_tensor.unsqueeze(0))

                    orig_size = torch.tensor(
                        [[image_np.shape[0], image_np.shape[1]]], device=self.device
                    )
                    post = self.postprocessor(out, orig_size)[0]
                    return to_cpu_device(post)

                img_pil = Image.open(img_path).convert("RGB")
                sahi_cfg = self.config.get("eval_inference", {}).get("sahi", {})
                input_size = getattr(self.config.data, "model_input_size", 640)
                preds = run_sahi_sliced_eval(img_pil, predict_fn, sahi_cfg, input_size)
                sliced_predictions[img_id] = preds

            self.test_step_outputs_sliced.append(
                {"predictions": sliced_predictions, "image_ids": image_ids}
            )

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

        return {"predictions": predictions, "image_ids": image_ids}

    def on_test_epoch_end(self):
        def _compute_and_log(outputs_list, prefix_name, base_prefix, suffix=""):
            all_outputs = gather_outputs_across_processes(outputs_list)
            merged = {}
            for batch_out in all_outputs:
                merged.update(batch_out.get("predictions", {}))

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

        if self.test_step_outputs:
            _compute_and_log(self.test_step_outputs, "Test", "test", "")
        if self.test_step_outputs_sliced:
            _compute_and_log(
                self.test_step_outputs_sliced, "Test Sliced", "test", "_sliced"
            )
        if hasattr(self, "test_step_outputs_ema") and self.test_step_outputs_ema:
            _compute_and_log(self.test_step_outputs_ema, "Test EMA", "test", "_ema")
        if (
            hasattr(self, "test_step_outputs_sliced_ema")
            and self.test_step_outputs_sliced_ema
        ):
            _compute_and_log(
                self.test_step_outputs_sliced_ema,
                "Test Sliced EMA",
                "test",
                "_sliced_ema",
            )

    def configure_optimizers(self):
        # We handle DEIM's discriminative learning rates natively here
        opt_config = self.config.optimizer.optimizer
        sch_config = self.config.scheduler

        # Backbone is fine-tuned at 1/20th the standard learning rate
        base_lr = float(opt_config.lr)
        backbone_lr = base_lr * 0.05

        params = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue

            is_backbone = "backbone" in name
            # No weight decay for norm/bias layers
            is_norm_bias = any(x in name for x in ["norm", "bn", "bias"])

            # Setup layer-specific hyperparams
            lr = backbone_lr if is_backbone else base_lr
            weight_decay = 0.0 if is_norm_bias else float(opt_config.weight_decay)

            params.append({"params": p, "lr": lr, "weight_decay": weight_decay})

        optimizer = torch.optim.AdamW(params)

        total_steps = max(1, self.trainer.estimated_stepping_batches)
        if sch_config.type == "cosine":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=total_steps,
                    eta_min=float(sch_config.eta_min),
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
