#!/usr/bin/env python3
from __future__ import annotations

"""
Training script for RT-DETR with DINOv2 backbone using PyTorch Lightning.
Powered by Hydra for flexible configuration.
"""

import os
import datetime
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import shutil
import time
import torch
from typing import Any, Dict, Literal, Optional, Tuple

from utils.distributed_utils import setup_cluster_env, rank_print

# --- Monkey-patch for generalized_box_iou to prevent crash on degenerate boxes ---
import transformers.loss.loss_for_object_detection as loss_utils


# 1. Zero-Overhead GIoU Patch
def patched_generalized_box_iou(boxes1, boxes2):
    try:
        return loss_utils.original_generalized_box_iou(boxes1, boxes2)
    except ValueError:
        # Repair only on failure
        boxes1 = torch.nan_to_num(boxes1, 0.0, 1.0, 0.0)
        boxes2 = torch.nan_to_num(boxes2, 0.0, 1.0, 0.0)
        boxes1 = torch.cat(
            [boxes1[..., :2], torch.maximum(boxes1[..., 2:], boxes1[..., :2])], dim=-1
        )
        boxes2 = torch.cat(
            [boxes2[..., :2], torch.maximum(boxes2[..., 2:], boxes2[..., :2])], dim=-1
        )
        return loss_utils.original_generalized_box_iou(boxes1, boxes2)


# 2. Matcher "NaN Sentinel" Patch
import transformers.loss.loss_rt_detr as loss_rt_detr
from transformers.image_transforms import center_to_corners_format
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F


def patched_matcher_forward(self, outputs, targets):
    try:
        return self.original_forward(outputs, targets)
    except ValueError as e:
        if "matrix contains invalid numeric entries" not in str(e):
            raise e

        from utils.distributed_utils import rank_zero_print

        # rank_zero_print("\n" + "!"*80 + "\n🚨 [NaN Sentinel] Invalid numeric entries detected in Matcher!\n" + "!"*80)

        with torch.no_grad():
            batch_size, num_queries = outputs["logits"].shape[:2]
            out_bbox = outputs["pred_boxes"].flatten(0, 1)
            target_ids = torch.cat([v["class_labels"] for v in targets])
            target_bbox = torch.cat([v["boxes"] for v in targets])

            # Diagnostic: where is the NaN?
            if not torch.isfinite(outputs["logits"]).all():
                rank_zero_print("   -> Source: NaNs found in LOGITS")
            if not torch.isfinite(out_bbox).all():
                rank_zero_print("   -> Source: NaNs found in PRED_BOXES")

            # Sanitization logic
            if self.use_focal_loss:
                out_prob = F.sigmoid(outputs["logits"].flatten(0, 1))
                out_prob = out_prob[:, target_ids]
                neg_cost_class = (
                    (1 - self.alpha)
                    * (out_prob**self.gamma)
                    * (-(1 - out_prob + 1e-8).log())
                )
                pos_cost_class = (
                    self.alpha
                    * ((1 - out_prob) ** self.gamma)
                    * (-(out_prob + 1e-8).log())
                )
                class_cost = pos_cost_class - neg_cost_class
            else:
                out_prob = outputs["logits"].flatten(0, 1).softmax(-1)
                class_cost = -out_prob[:, target_ids]

            bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)
            giou_cost = -patched_generalized_box_iou(
                center_to_corners_format(out_bbox),
                center_to_corners_format(target_bbox),
            )

            # Combine and sanitize the entire matrix
            cost_matrix = (
                self.bbox_cost * bbox_cost
                + self.class_cost * class_cost
                + self.giou_cost * giou_cost
            )
            cost_matrix = torch.nan_to_num(
                cost_matrix, nan=0.0, posinf=1e6, neginf=-1e6
            )
            cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            indices = [
                linear_sum_assignment(c[i])
                for i, c in enumerate(cost_matrix.split(sizes, -1))
            ]
            return [
                (
                    torch.as_tensor(i, dtype=torch.int64),
                    torch.as_tensor(j, dtype=torch.int64),
                )
                for i, j in indices
            ]


# Apply patches early
loss_utils.original_generalized_box_iou = loss_utils.generalized_box_iou
loss_utils.generalized_box_iou = patched_generalized_box_iou

try:
    loss_rt_detr.generalized_box_iou = patched_generalized_box_iou
    if not hasattr(loss_rt_detr.RTDetrHungarianMatcher, "original_forward"):
        loss_rt_detr.RTDetrHungarianMatcher.original_forward = (
            loss_rt_detr.RTDetrHungarianMatcher.forward
        )
        loss_rt_detr.RTDetrHungarianMatcher.forward = patched_matcher_forward
except (ImportError, AttributeError):
    pass

setup_cluster_env()


def _resolve_ckpt_path(
    config: DictConfig, run_save_dir: Optional[str] = None
) -> Optional[str]:
    """Resolve checkpoint path from manual override or auto-resume."""
    ckpt_path = config.initialization.get("load_from_checkpoint")
    if ckpt_path:
        raw = str(ckpt_path).strip().strip('"').strip("'")
        # Common copy/paste issue: trailing punctuation after .ckpt
        raw = raw.rstrip(".,;:)]}>")

        candidates = []
        for candidate in [
            raw,
            os.path.expanduser(raw),
            os.path.expandvars(raw),
            hydra.utils.to_absolute_path(raw),
        ]:
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        for candidate in candidates:
            if os.path.exists(candidate):
                rank_zero_print(f"🔄 Manual checkpoint path: {candidate}")
                return candidate

        rank_zero_print("WARNING: Provided checkpoint path was not found.")
        rank_zero_print(f"  raw: {raw}")
        for idx, candidate in enumerate(candidates):
            rank_zero_print(f"  candidate[{idx}]: {candidate}")
        return None

    if run_save_dir and config.initialization.get("auto_resume", False):
        last_ckpt = os.path.join(run_save_dir, "ckpts", "last.ckpt")
        if os.path.exists(last_ckpt):
            rank_zero_print(f"Auto-Resume: Found existing 'last.ckpt' at {last_ckpt}")
            return last_ckpt
        rank_zero_print(
            f"Auto-Resume: No 'last.ckpt' found in {last_ckpt}. Starting fresh."
        )

    return None


def _load_ckpt(ckpt_path: str) -> Dict[str, Any]:
    """Load checkpoint on CPU with trusted full object unpickling."""
    rank_zero_print(f"Loading checkpoint object from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint to be a dict, got {type(checkpoint)}")
    return checkpoint


def _set_nested_value(cfg: DictConfig, dotted_key: str, value: Any) -> None:
    """Set nested DictConfig value, creating intermediate dicts as needed."""
    if value is None:
        return
    node = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in node or node[part] is None:
            node[part] = OmegaConf.create({})
        node = node[part]
    node[parts[-1]] = value


def _merge_test_only_config_from_ckpt(
    current_cfg: DictConfig, ckpt: Dict[str, Any]
) -> DictConfig:
    """Use checkpoint config as base and re-apply runtime/test-only overrides."""
    hp = ckpt.get("hyper_parameters", {})
    ckpt_cfg_raw = hp.get("config") if isinstance(hp, dict) else None
    if ckpt_cfg_raw is None:
        rank_zero_print(
            "WARNING: No 'hyper_parameters.config' found in checkpoint. Using current CLI config."
        )
        return current_cfg

    ckpt_cfg = OmegaConf.create(
        OmegaConf.to_container(ckpt_cfg_raw, resolve=False)
        if OmegaConf.is_config(ckpt_cfg_raw)
        else ckpt_cfg_raw
    )
    merged_cfg = OmegaConf.create(OmegaConf.to_container(ckpt_cfg, resolve=False))
    OmegaConf.set_struct(merged_cfg, False)

    # Runtime/test-only override policy
    merged_cfg.test_only = current_cfg.test_only
    merged_cfg.debug = current_cfg.debug
    merged_cfg.seed = current_cfg.seed
    merged_cfg.run_name = current_cfg.run_name
    merged_cfg.val_name = current_cfg.val_name
    merged_cfg.test_name = current_cfg.test_name
    merged_cfg.train_name = current_cfg.train_name

    _set_nested_value(
        merged_cfg,
        "initialization.load_from_checkpoint",
        current_cfg.initialization.get("load_from_checkpoint"),
    )

    if hasattr(current_cfg, "trainer") and current_cfg.trainer is not None:
        merged_cfg.trainer = OmegaConf.create(
            OmegaConf.to_container(current_cfg.trainer, resolve=False)
        )

    if hasattr(current_cfg, "data") and current_cfg.data is not None:
        _set_nested_value(merged_cfg, "data.path", current_cfg.data.get("path"))
        _set_nested_value(
            merged_cfg,
            "data.limit_test_batches",
            current_cfg.data.get("limit_test_batches"),
        )
        _set_nested_value(
            merged_cfg, "data.num_workers", current_cfg.data.get("num_workers")
        )
        _set_nested_value(
            merged_cfg, "data.batch_size", current_cfg.data.get("batch_size")
        )

    if hasattr(current_cfg, "logging") and current_cfg.logging is not None:
        merged_cfg.logging = OmegaConf.create(
            OmegaConf.to_container(current_cfg.logging, resolve=False)
        )

    if (
        hasattr(current_cfg, "eval_inference")
        and current_cfg.eval_inference is not None
    ):
        merged_cfg.eval_inference = OmegaConf.create(
            OmegaConf.to_container(current_cfg.eval_inference, resolve=False)
        )

    if hasattr(current_cfg, "inference") and current_cfg.inference is not None:
        merged_cfg.inference = OmegaConf.create(
            OmegaConf.to_container(current_cfg.inference, resolve=False)
        )

    rank_zero_print(
        "Using checkpoint config as base for test_only; reapplied runtime/test CLI overrides."
    )
    return merged_cfg


def _extract_regular_state_dict(ckpt: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get regular model state dict from Lightning checkpoint or raw state dict checkpoint."""
    state_dict = ckpt.get("state_dict")
    if isinstance(state_dict, dict):
        return state_dict

    # Fallback: checkpoint may itself be a state_dict-like mapping
    tensor_items = [
        (k, v)
        for k, v in ckpt.items()
        if isinstance(k, str) and isinstance(v, torch.Tensor)
    ]
    if tensor_items and all(
        "." in k for k, _ in tensor_items[: min(8, len(tensor_items))]
    ):
        return ckpt
    return None


def _select_eval_weights_source(
    ckpt_path: str, ckpt: Dict[str, Any]
) -> Literal["ema", "regular"]:
    """Choose EMA vs regular weights for evaluation using path hint + key availability."""
    path_has_ema = "ema" in os.path.basename(ckpt_path).lower()
    has_ema = isinstance(ckpt.get("ema_state_dict"), dict)
    has_regular = _extract_regular_state_dict(ckpt) is not None

    if path_has_ema and has_ema:
        return "ema"
    if path_has_ema and not has_ema and has_regular:
        rank_zero_print(
            "WARNING: Checkpoint path suggests EMA but 'ema_state_dict' missing. Falling back to regular weights."
        )
        return "regular"
    if (not path_has_ema) and has_regular:
        return "regular"
    if (not path_has_ema) and (not has_regular) and has_ema:
        rank_zero_print("WARNING: Regular state_dict missing; using EMA weights.")
        return "ema"

    raise ValueError(
        "Checkpoint does not contain usable weights. Expected one of: "
        "'state_dict' (regular) or 'ema_state_dict' (EMA)."
    )


def _load_selected_weights(
    lightning_module: RTDETRLightningModule,
    ckpt: Dict[str, Any],
    source: Literal["ema", "regular"],
) -> Tuple[list, list]:
    """Load selected evaluation weights and return missing/unexpected keys."""
    if source == "ema":
        ema_state = ckpt.get("ema_state_dict")
        if not isinstance(ema_state, dict):
            raise ValueError(
                "Requested EMA weights but checkpoint has no valid 'ema_state_dict'."
            )
        result = lightning_module.model.load_state_dict(ema_state, strict=False)
    else:
        regular_state = _extract_regular_state_dict(ckpt)
        if regular_state is None:
            raise ValueError(
                "Requested regular weights but checkpoint has no valid 'state_dict'."
            )
        result = lightning_module.load_state_dict(regular_state, strict=False)

    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    return missing, unexpected


import torch

torch.set_float32_matmul_precision("medium")

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
    EarlyStopping,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler
from transformers import (
    RTDetrImageProcessor,
    RTDetrV2ForObjectDetection,
    RTDetrForObjectDetection,
    RTDetrConfig,
)
from torchvision.datasets import CocoDetection
import torch.distributed as dist

from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("extract_name", lambda path: path.split("/")[-1])
OmegaConf.register_new_resolver("oc.eval", eval)
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
import wandb

from models.custom_rt_detr_with_dinov2_backbone import (
    RTDetrV2ForObjectDetectionWithCustomBackbone,
    RTDetrV2ConfigWithCustomBackBone,
)
from models.rt_detr_lightning_module import RTDETRLightningModule
from data.coco_data_module import COCODataModule
from models.backbone_factory import (
    build_backbone,
    freeze_backbone_layers,
    get_backbone_unique_id,
)
from utils.train_utils import BackupToNASCallback
from utils.distributed_utils import (
    get_rank,
    rank_zero_print,
    rank_print,
    setup_cluster_env,
)

setup_cluster_env()


def create_initial_checkpoint(config: DictConfig) -> str:
    """
    Create initial RT-DETR checkpoint with DINOv2 backbone.
    """
    rank = get_rank()
    rank_zero_print("\n" + "=" * 80)
    rank_zero_print(
        f"Creating initial RT-DETR checkpoint with {config.model.backbone.type.upper()} backbone..."
    )
    rank_zero_print("=" * 80 + "\n")

    model_config = config.model
    checkpoint_config = config.checkpointing

    # Determine suffix WITHOUT loading the model
    unique_suffix = get_backbone_unique_id(
        model_config.backbone, model_config.rtdetr.model_name
    )

    if model_config.backbone.type == "resnet":
        full_suffix = f"{unique_suffix}"
    else:
        full_suffix = f"{unique_suffix}_{model_config.rtdetr.model_name}"
    # Include num_queries in the cache name so we don't accidentally load models with mismatched query sizes
    if hasattr(model_config.rtdetr, "num_queries"):
        full_suffix += f"_q{model_config.rtdetr.num_queries}"
    base_rtdetr_path = hydra.utils.to_absolute_path(
        checkpoint_config.rtdetr_initial_checkpoint
    )
    local_path = f"{base_rtdetr_path}{full_suffix}"

    # get nas path
    nas_base = checkpoint_config.get("nas_initial_checkpoint")
    # TODO: remove this when running on denvr
    nas_path = None
    if nas_base:
        nas_path = f"{hydra.utils.to_absolute_path(nas_base)}{full_suffix}"

    sentinel_file = os.path.join(local_path, ".done")

    # Step 1: Check if already exists (All ranks check)
    if os.path.exists(local_path) and len(os.listdir(local_path)) > 0:
        if os.path.exists(sentinel_file):
            rank_zero_print(f"✓ Found completed weights in SCRATCH: {local_path}")
            return local_path
        elif rank != 0:
            rank_print(f"Waiting for Rank 0 to finish writing {local_path}...")
            while not os.path.exists(sentinel_file):
                time.sleep(5)
            return local_path
        else:
            rank_zero_print(
                f"⚠ Found incomplete weights at {local_path}. Re-initializing..."
            )
            # Rank 0 continues to Step 2

    # Step 2: Handle Initialization (Only Rank 0)
    if rank == 0:
        os.makedirs(local_path, exist_ok=True)
        # Check NAS if available
        if nas_path and os.path.exists(nas_path) and len(os.listdir(nas_path)) > 0:
            rank_zero_print(f"✓ Found weights on NAS: {nas_path}")
            rank_zero_print("  -> Copying to scratch...")
            try:
                shutil.copytree(nas_path, local_path, dirs_exist_ok=True)
                # Touch sentinel
                open(sentinel_file, "w").close()
                rank_zero_print("  -> Copy complete.")
                return local_path
            except Exception as e:
                rank_zero_print(f"  -> Copy failed ({e}). Creating new...")

        rank_zero_print("! Weights not found. Starting Heavy Initialization...")

        backbone_model, backbone_config_obj, _ = build_backbone(
            model_config.backbone, model_config.rtdetr.model_name
        )

        if model_config.backbone.type in ["official", "resnet"]:
            base_model_name = model_config.backbone.name
        else:
            base_model_name = model_config.rtdetr.model_name

        rank_zero_print(f"Loading Base Weights: PekingU/{base_model_name}")

        id2label = {int(k): v for k, v in model_config.label_map.items()}
        label2id = {v: k for k, v in id2label.items()}
        overrides = OmegaConf.to_container(model_config.rtdetr, resolve=True)

        # Keys to remove that are Hydra-specific or training-specific
        for k in [
            "pretrained_name_or_path",
            "config_overrides",
            "model_name",
            "name",
            "freeze_backbone_batch_norms",
            "normalize_before",
            "input_size",
        ]:
            overrides.pop(k, None)

        if "rtdetr_v2" in model_config.rtdetr.model_name:
            model_cls = RTDetrV2ForObjectDetection
            config_cls = RTDetrV2ConfigWithCustomBackBone
            # v2 supports input_size, keep it in overrides
        else:
            rank_zero_print(
                f"Detected RT-DETRv1 model: {model_config.rtdetr.model_name}"
            )
            model_cls = RTDetrForObjectDetection
            config_cls = RTDetrConfig
            # v1 config doesn't support these parameters - remove them
            overrides.pop("input_size", None)
            overrides.pop("decoder_n_levels", None)
            overrides.pop("decoder_method", None)
            overrides.pop(
                "num_feature_levels", None
            )  # This is derived from decoder_n_levels
            # ensure we use 'auxiliary_loss' not 'use_auxiliary_loss' if it slipped in
            if "use_auxiliary_loss" in overrides:
                overrides["auxiliary_loss"] = overrides.pop("use_auxiliary_loss")
            rank_zero_print(f"Cleaned overrides for v1: {list(overrides.keys())}")

        model = model_cls.from_pretrained(
            f"PekingU/{base_model_name}",
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            **overrides,
        )
        # Inject Backbone
        if config.model.backbone.type == "resnet":
            if not model_config.backbone.train_backbone:
                # Freeze entire backbone
                rank_zero_print("[INFO] Freezing entire backbone (stage 4)...")
                freeze_backbone_layers(model, freeze_at_stage=4)
            elif model_config.backbone.freeze_at_stage > 0:
                # Partial freezing - freeze some stages, train others
                rank_zero_print(
                    f"[INFO] Applying partial backbone freezing at stage {model_config.backbone.freeze_at_stage}..."
                )
                freeze_backbone_layers(
                    model, freeze_at_stage=model_config.backbone.freeze_at_stage
                )
        else:
            pretrained_model_config_dict = model.config.to_dict()
            rt_detr_config = config_cls(**pretrained_model_config_dict)
            rt_detr_config.backbone_config = backbone_config_obj
            model.config = rt_detr_config
            # Handle difference in backbone attribute between V1 and V2
            if hasattr(model, "model") and hasattr(model.model, "backbone"):
                model.model.backbone = backbone_model
            else:  # V1 structure often has direct backbone or wrapped differently, handled by model class usually but here we inject
                # RTDetrForObjectDetection has model.backbone
                model.model.backbone = backbone_model

        # Save to Scratch (Always)
        rank_zero_print(f"Saving new model to Scratch: {local_path}")
        model.save_pretrained(local_path)

    # Backup to NAS (If available)
    if rank == 0:
        # Save sentinel to mark completion
        open(sentinel_file, "w").close()

        if nas_path:
            rank_zero_print(f"Mirroring new model to NAS: {nas_path}")
            try:
                if not os.path.exists(nas_path):
                    shutil.copytree(local_path, nas_path)
            except Exception as e:
                rank_zero_print(f"Warning: Backup to NAS failed: {e}")
    else:
        # Other ranks wait for the sentinel file to appear
        if not os.path.exists(sentinel_file):
            rank_print("Waiting for Rank 0 to finish initialization...")
            while not os.path.exists(sentinel_file):
                time.sleep(5)

    return local_path


def setup_model(config: DictConfig) -> RTDETRLightningModule:
    """Setup the RT-DETR model with DINOv2 backbone."""
    model_config = config.model
    init_config = config.initialization
    # breakpoint()
    if init_config.create_initial_checkpoint:
        model_checkpoint_path = create_initial_checkpoint(config)
    else:
        model_checkpoint_path = hydra.utils.to_absolute_path(
            config.checkpointing.rtdetr_initial_checkpoint
        )
        if not os.path.exists(model_checkpoint_path):
            rank_zero_print(f"WARNING: Checkpoint not found at {model_checkpoint_path}")
            rank_zero_print("Creating initial checkpoint...")
            model_checkpoint_path = create_initial_checkpoint(config)

    rank_zero_print(f"\nLoading RT-DETR model from: {model_checkpoint_path}")

    if "rtdetr_v2" in config.model.rtdetr.model_name:
        model_cls = RTDetrV2ForObjectDetectionWithCustomBackbone
    else:
        model_cls = RTDetrForObjectDetection

    model = model_cls.from_pretrained(
        model_checkpoint_path,
    )

    # Ensure model is on CUDA before casting to half
    # if torch.cuda.is_available():
    #     model.to("cuda")
    #     rank_zero_print("[INFO] Moved base model to CUDA device.")

    # if config.trainer.precision == "16-mixed":
    #     model.half()
    #     rank_zero_print("[INFO] Explicitly cast base model to Half precision for AMP compatibility.")

    # Explicitly set model to TRAIN mode initially.
    # This ensures that when we subsequently freeze the backbone (eval mode),
    # the rest of the model (decoder, etc.) remains in train mode, creating the correct mixed state.
    model.train()
    # breakpoint()

    if config.model.backbone.type == "resnet":
        if not config.model.backbone.train_backbone:
            # Freeze entire backbone
            rank_zero_print("[INFO] Freezing entire backbone (stage 4)...")
            freeze_backbone_layers(model, freeze_at_stage=4)
        elif config.model.backbone.freeze_at_stage > 0:
            # Partial freezing - freeze some stages, train others
            rank_zero_print(
                f"[INFO] Applying partial backbone freezing at stage {config.model.backbone.freeze_at_stage}..."
            )
            freeze_backbone_layers(model, config.model.backbone.freeze_at_stage)

    elif config.model.backbone.type == "dinov2":
        # Re-freeze DINOv2 if needed (usually handled by DINO class init,
        # but good to ensure if you are loading a full model checkpoint)
        rank_zero_print("[INFO] Ensuring DINOv2 backbone is frozen...")
        for name, param in model.named_parameters():
            if "model.backbone.backbone" in name:  # The ViT part
                param.requires_grad = False

    rtdetr_overrides = OmegaConf.to_container(config.model.rtdetr, resolve=True)
    rtdetr_overrides.pop("pretrained_name_or_path", None)
    rtdetr_overrides.pop("config_overrides", None)
    rtdetr_overrides.pop("model_name", None)
    rtdetr_overrides.pop("name", None)
    if rtdetr_overrides:
        rank_zero_print("Checking for model config overrides...")
        changes_made = False
        for key, value in rtdetr_overrides.items():
            if hasattr(model.config, key):
                current_value = getattr(model.config, key)
                # Only print and set if the value has changed
                if current_value != value:
                    if not changes_made:
                        rank_zero_print("Applying config overrides to loaded model:")
                        changes_made = True
                    rank_zero_print(
                        f"  > Setting model.config.{key}: {current_value} -> {value}"
                    )
                    setattr(model.config, key, value)
            else:
                rank_zero_print(
                    f"  > WARNING: model.config has no attribute '{key}' (cannot set)"
                )

        if not changes_made:
            rank_zero_print("...Loaded model config already matches overrides.")

    processor = RTDetrImageProcessor.from_pretrained(
        config.model.rtdetr.pretrained_name_or_path
    )
    processor.do_normalize = True
    processor.resample = 3
    processor.size = {
        "height": config.data.model_input_size,
        "width": config.data.model_input_size,
    }

    data_path = hydra.utils.to_absolute_path(config.data.path)
    val_annot_path = os.path.join(data_path, "images", config.val_name)
    val_json_path = os.path.join(data_path, f"{config.val_name}_annotations.json")
    val_coco_dataset = CocoDetection(
        root=val_annot_path, annFile=val_json_path, transforms=None
    )
    val_coco_gt = val_coco_dataset.coco
    val_coco_gt.dataset["info"] = {}

    test_annot_path = os.path.join(data_path, "images", config.test_name)
    test_json_path = os.path.join(data_path, f"{config.test_name}_annotations.json")
    test_coco_dataset = CocoDetection(
        root=test_annot_path, annFile=test_json_path, transforms=None
    )
    test_coco_gt = test_coco_dataset.coco
    test_coco_gt.dataset["info"] = {}

    lightning_model = RTDETRLightningModule(
        model=model,
        image_processor=processor,
        config=config,  # Pass the whole config
        val_coco_gt=val_coco_gt,
        test_coco_gt=val_coco_gt if config.debug else test_coco_gt,
        val_image_root=val_annot_path,
        test_image_root=val_annot_path if config.debug else test_annot_path,
    )

    rank_zero_print("✓ Model loaded successfully")
    return lightning_model, processor


def setup_data(config: DictConfig, processor) -> COCODataModule:
    """Setup the data module."""
    data_config = config.data

    data_module = COCODataModule(
        dataset_path=hydra.utils.to_absolute_path(
            data_config.path
        ),  # Use absolute path
        processor=processor,
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        model_input_size=data_config.model_input_size,
        min_random_scale=data_config.min_random_scale,
        max_random_scale=data_config.max_random_scale,
        p_noise=data_config.p_noise,
        org_images_in_model_input_size=data_config.org_images_in_model_input_size,
        config=config,
    )

    rank_zero_print(f"✓ Data module configured for: {data_config.path}")
    return data_module


def setup_profiler(config: DictConfig):
    # Note: Hydra changes CWD, profiler logs save to the hydra output dir
    profiler_config = config.training.profiler
    dir_name = "profiler_logs"  # Will be saved inside hydra's output dir

    if profiler_config.type == "simple":
        profiler = SimpleProfiler(dirpath=dir_name, filename="rtdetr_profile")
    elif profiler_config.type == "advanced":
        profiler = AdvancedProfiler(dirpath=dir_name, filename="rtdetr_profile")
    else:
        return None
    return profiler


def setup_callbacks(config: DictConfig):
    """Setup training callbacks."""
    checkpoint_config = config.checkpointing
    callbacks = []

    # 1. Standard Model Checkpoint
    # Tracks the standard validation metric (e.g. val/map)
    rank_zero_print(
        f"Configure ModelCheckpoint for Standard Model: {checkpoint_config.monitor}"
    )
    callbacks.append(
        ModelCheckpoint(
            dirpath=os.path.join(
                hydra.utils.to_absolute_path(checkpoint_config.save_dir), "ckpts"
            ),
            filename="rtdetr-regular-epoch{epoch:02d}-val_map{"
            + checkpoint_config.monitor.replace("/", "_")
            + ":.4f}",
            monitor=checkpoint_config.monitor,
            mode=checkpoint_config.mode,
            save_top_k=checkpoint_config.save_top_k,
            save_last=checkpoint_config.save_last,  # 'last.ckpt' will be managed by this one
            every_n_epochs=checkpoint_config.every_n_epochs,
            verbose=True,
            auto_insert_metric_name=False,
        )
    )

    # 2. EMA Callback and Checkpoint (If enabled)
    if hasattr(config.model, "ema") and config.model.ema.enabled:
        from utils.ema import EMACallback

        warmup_steps = config.model.ema.get("warmup_steps", 0)
        tau = config.model.ema.get("tau", 2000)
        rank_zero_print(
            f"💡 EMA enabled: Adding EMACallback with decay={config.model.ema.decay}, warmup_steps={warmup_steps}, tau={tau}"
        )
        callbacks.append(
            EMACallback(
                decay=config.model.ema.decay, warmup_steps=warmup_steps, tau=tau
            )
        )

        # Tracks the EMA validation metric (val/map_ema)
        rank_zero_print(
            "💡 EMA enabled: Adding second ModelCheckpoint for 'val/map_ema'"
        )
        ema_monitor = "val/map_ema"
        callbacks.append(
            ModelCheckpoint(
                dirpath=os.path.join(
                    hydra.utils.to_absolute_path(checkpoint_config.save_dir), "ckpts"
                ),
                filename="rtdetr-ema-{epoch:02d}-val_map{"
                + ema_monitor.replace("/", "_")
                + ":.4f}",
                monitor=ema_monitor,
                mode=checkpoint_config.mode,
                save_top_k=checkpoint_config.save_top_k,
                save_last=False,  # Don't duplicate 'last.ckpt' logic
                every_n_epochs=checkpoint_config.every_n_epochs,
                verbose=True,
                auto_insert_metric_name=False,
            )
        )

    callbacks.append(LearningRateMonitor(logging_interval="step"))
    callbacks.append(ModelSummary(max_depth=3))
    callbacks.append(
        EarlyStopping(
            monitor=checkpoint_config.monitor,
            mode=checkpoint_config.mode,
            patience=10,
            verbose=True,
        )
    )

    if "backup_dir" in checkpoint_config and checkpoint_config.backup_dir:
        # Resolve path (handle ${hydra...} if needed, though usually resolved by now)
        backup_path = hydra.utils.to_absolute_path(checkpoint_config.backup_dir)
        callbacks.append(BackupToNASCallback(backup_dir=backup_path))

    rank_zero_print("✓ Callbacks configured")
    return callbacks


def setup_logger(config: DictConfig):
    """Setup WandB logger."""
    wandb_config = config.logging.wandb

    if not wandb_config.enabled:
        rank_zero_print("✓ WandB logging disabled")
        return None

    # wandb_log_config = OmegaConf.to_container(config, resolve=True)
    # Manually construct log config to avoid resolution issues if any
    try:
        wandb_log_config = OmegaConf.to_container(config, resolve=True)
    except Exception as e:
        rank_zero_print(f"Warning: Failed to resolve config for WandB logging: {e}")
        wandb_log_config = OmegaConf.to_container(config, resolve=False)

    # Add top-level keys for easy filtering on WandB dashboard
    if config.model and config.model.rtdetr and config.model.rtdetr.model_name:
        wandb_log_config["model_type"] = (
            "rtdetrv2" if "v2" in config.model.rtdetr.model_name else "rtdetrv1"
        )
    else:
        wandb_log_config["model_type"] = "unknown"
    wandb_log_config["backbone"] = config.model.backbone.name
    wandb_log_config["backbone_type"] = config.model.backbone.type

    logger = WandbLogger(
        project=wandb_config.project,
        reinit="finish_previous",
        name=config.run_name,
        tags=list(wandb_config.tags),  # Convert OmegaConf list to plain list
        notes=wandb_config.notes,
        # group=wandb_config.get("group"),
        config=wandb_log_config,  # Log full config with extra filter keys
        # Hydra changes CWD, so we save logs to the new CWD
        save_dir=os.getcwd(),
    )

    rank_zero_print(f"✓ WandB logger configured - Project: {wandb_config.project}")
    # logger.watch(model, log='gradients', log_freq=100)
    # rank_zero_print("✓ WandB logger watching model for gradients")

    return logger


@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(config: DictConfig):
    # Unlock config to make changes
    OmegaConf.set_struct(config, False)
    test_only_checkpoint: Optional[Dict[str, Any]] = None
    test_only_weight_source: Optional[Literal["ema", "regular"]] = None
    early_ckpt_path: Optional[str] = None

    if config.test_only:
        early_ckpt_path = _resolve_ckpt_path(config)
        if not early_ckpt_path:
            raise ValueError(
                "test_only=true requires a valid checkpoint path. "
                "Check for typos and trailing punctuation (e.g., '.ckpt.'), and pass "
                "'initialization.load_from_checkpoint=/abs/path/model.ckpt'."
            )

        test_only_checkpoint = _load_ckpt(early_ckpt_path)
        config = _merge_test_only_config_from_ckpt(config, test_only_checkpoint)
        OmegaConf.set_struct(config, False)
        config.initialization.load_from_checkpoint = early_ckpt_path

        # Single selected-model evaluation in test_only mode (no EMA callback branch).
        if (
            hasattr(config, "model")
            and hasattr(config.model, "ema")
            and config.model.ema is not None
        ):
            config.model.ema.enabled = False
            rank_zero_print(
                "test_only: disabled EMA callback/branch for single selected-model evaluation."
            )

        test_only_weight_source = _select_eval_weights_source(
            early_ckpt_path, test_only_checkpoint
        )
        rank_zero_print(
            f"test_only: selected checkpoint weight source = {test_only_weight_source.upper()}"
        )

    # breakpoint()
    # --- 1. Handle Run Naming (Cluster Agnostic) ---
    # Try to find a shared ID from common cluster/launcher environment variables
    # This ensures all ranks in a distributed run agree on the run name/ID.
    unique_id = None
    id_candidates = [
        "SLURM_JOB_ID",  # SLURM
        "TORCHELASTIC_RUN_ID",  # torchrun / torch.distributed.launch
        "WANDB_RUN_ID",  # User-provided WandB ID
        "PBS_JOBID",  # PBS/Torque
        "LSB_JOBID",  # LSF
    ]

    for var in id_candidates:
        if os.environ.get(var):
            unique_id = os.environ.get(var)
            rank_zero_print(f"Found Job ID from {var}: {unique_id}")
            break

    if not unique_id:
        # Fallback to timestamp if no manager/launcher detected
        unique_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        rank_zero_print(f"No shared job ID found. Using timestamp: {unique_id}")
    else:
        # Append HH-MM so sequential runs in the same interactive session don't collide
        timestamp = datetime.datetime.now().strftime("%H-%M")
        unique_id = f"{unique_id}_{timestamp}"

    # Standardize run naming: {original_run_name_with_date_from_yaml}_{unique_id}
    config.run_name = f"{config.run_name}_{unique_id}"

    # --- 2. Handle Hydra Sweep Logic ---
    hydra_cfg = HydraConfig.get()

    # Convert overrides to WandB tags for easy identification
    if hasattr(hydra_cfg.overrides, "task"):
        job_overrides = hydra_cfg.overrides.task
        for override in job_overrides:
            if "=" in override:
                key, value = override.split("=", 1)
                short_key = key.split(".")[-1]

                # Skip adding load_from_checkpoint as a tag (it exceeds 64 char limit in wandb and is already logged in config)
                if short_key == "load_from_checkpoint":
                    continue

                tag = f"{short_key}={value}"
                config.logging.wandb.tags.append(tag)
                rank_zero_print(f"   -> Added WandB tag: {tag}")

    if hydra_cfg.mode == RunMode.MULTIRUN:
        rank_zero_print(f"Detected Hydra Sweep (Job {hydra_cfg.job.num})")

        # Append sweep job index to run_name for unique directories
        config.run_name = f"{config.run_name}_run{hydra_cfg.job.num}"

        # A. Set WandB Group
        # We group by the directory name Hydra created for this sweep (shared by all jobs)
        # or you can use a static string like "Sweep_Nov17"
        if not config.logging.wandb.get("group"):
            # Use the parent multirun folder timestamp as the group ID
            # This keeps all runs in this sweep together in the UI
            sweep_id = os.path.basename(os.path.normpath(hydra_cfg.sweep.dir))
            config.logging.wandb.group = f"sweep_{sweep_id}"

    # --- Hydra handles all config loading and merging ---
    # The 'config' object is already the final, merged config

    if config.debug:
        rank_zero_print("Running in DEBUG mode")
        OmegaConf.set_struct(config, False)  # Unlock config
        # Apply debug settings
        config.trainer.num_overfit_samples = 10
        config.data.batch_size = 1
        config.run_name = f"DEBUG_{config.run_name}"
        config.logging.wandb.project = f"{config.logging.wandb.project}"
        # config.checkpointing.save_dir = os.path.join({config.checkpointing.save_dir}, config.run_name)
        OmegaConf.set_struct(config, True)  # Re-lock config

    # logic for auto-resume training from lat ckpt
    base_save_dir = hydra.utils.to_absolute_path(config.checkpointing.save_dir)
    run_save_dir = os.path.join(base_save_dir, config.run_name)
    config.checkpointing.save_dir = run_save_dir

    # Resolve checkpoint path once (manual override or auto-resume).
    ckpt_path = _resolve_ckpt_path(config, run_save_dir=run_save_dir)

    OmegaConf.set_struct(config, True)  # Re-lock config
    # Set dynamic save_dir (relative to hydra's CWD)
    # This is now handled by ModelCheckpoint's dirpath

    rank_zero_print("\n" + "=" * 80)
    rank_zero_print("RT-DETR Training with DINOv2 Backbone (Hydra Edition)")
    rank_zero_print("=" * 80 + "\n")

    rank_zero_print("--- CWD (Hydra Output Dir) ---")
    rank_zero_print(f"{os.getcwd()}\n")

    rank_zero_print("--- Final Configuration ---")
    rank_zero_print(OmegaConf.to_yaml(config))
    rank_zero_print("---------------------------")

    eval_mode = config.get("eval_inference", {}).get("mode", "whole")
    rank_zero_print(f"--- Eval Inference Mode: {eval_mode.upper()} ---")
    if eval_mode == "sliced":
        rank_zero_print(OmegaConf.to_yaml(config.eval_inference.sahi))
        rank_zero_print("---------------------------")

    # Set seed
    pl.seed_everything(config.seed, workers=True)

    # Setup components
    rank = get_rank()
    if rank == 0:
        rank_zero_print("\n--- Distributed Environment ---")
        for var in [
            "MASTER_ADDR",
            "MASTER_PORT",
            "SLURM_PROCID",
            "SLURM_NNODES",
            "SLURM_NTASKS",
            "LOCAL_RANK",
            "RANK",
            "WORLD_SIZE",
        ]:
            rank_zero_print(f"{var}: {os.environ.get(var, 'NOT SET')}")
        rank_zero_print("-------------------------------\n")

    model, processor = setup_model(config)
    data_module = setup_data(config, processor)
    callbacks = setup_callbacks(config)
    logger = setup_logger(config)
    if logger:
        # Use absolute path to the source file
        source_path = os.path.join(
            hydra.utils.get_original_cwd(), "models/rt_detr_lightning_module.py"
        )
        if os.path.exists(source_path):
            logger.experiment.save(source_path)
        else:
            rank_zero_print(
                f"Warning: Could not find model source file at {source_path}"
            )
        logger.watch(model, log="gradients", log_freq=100)
        rank_zero_print("✓ WandB logger watching model for gradients")

    profiler = setup_profiler(config)
    # breakpoint()

    # Create trainer
    trainer_config = config.trainer
    data_config = config.data

    # Auto-detect number of nodes (default to 1)
    num_nodes = int(os.environ.get("SLURM_NNODES", 1))
    rank_zero_print(f"🌍 Detected Number of Nodes: {num_nodes}")

    trainer = pl.Trainer(
        accelerator=trainer_config.accelerator,
        devices=trainer_config.devices,
        num_nodes=num_nodes,
        precision=trainer_config.precision,
        strategy=trainer_config.strategy,
        max_epochs=trainer_config.max_epochs,
        log_every_n_steps=trainer_config.log_every_n_steps,
        val_check_interval=trainer_config.val_check_interval,
        gradient_clip_val=trainer_config.max_grad_norm,
        gradient_clip_algorithm=trainer_config.gradient_clip_algo,
        accumulate_grad_batches=trainer_config.accumulate_grad_batches,
        deterministic=trainer_config.deterministic,
        benchmark=trainer_config.benchmark,
        callbacks=callbacks,
        logger=logger,
        overfit_batches=trainer_config.num_overfit_samples,
        limit_test_batches=data_config.limit_test_batches if not config.debug else 10,
        limit_train_batches=data_config.limit_train_batches if not config.debug else 10,
        limit_val_batches=data_config.limit_val_batches,
        profiler=None if config.debug else profiler,
        plugins=(
            [SLURMEnvironment(auto_requeue=True)]
            if "SLURM_JOB_ID" in os.environ
            else None
        ),
    )

    if config.test_only:
        rank_zero_print("\n" + "=" * 80)
        rank_zero_print("Running in TEST-ONLY mode")
        rank_zero_print("=" * 80 + "\n")
        if not ckpt_path:
            raise ValueError(
                "Must provide a checkpoint path via 'initialization.load_from_checkpoint' for test-only mode."
            )

        if test_only_checkpoint is None:
            test_only_checkpoint = _load_ckpt(ckpt_path)
        if test_only_weight_source is None:
            test_only_weight_source = _select_eval_weights_source(
                ckpt_path, test_only_checkpoint
            )

        rank_zero_print(
            f"Loading {test_only_weight_source.upper()} weights for test-only evaluation..."
        )
        missing_keys, unexpected_keys = _load_selected_weights(
            model, test_only_checkpoint, test_only_weight_source
        )
        if missing_keys:
            rank_zero_print(
                f"⚠️  Missing keys during test-only load: {missing_keys[:10]} ..."
            )
        if unexpected_keys:
            rank_zero_print(
                f"⚠️  Unexpected keys during test-only load: {unexpected_keys[:10]} ..."
            )

        data_module.setup(stage="test")
        if getattr(data_module, "test_dataset", None) is None:
            raise RuntimeError(
                "Test dataset was not initialized before test-only evaluation."
            )
        rank_zero_print(f"Test set size: {len(data_module.test_dataset)}")
        test_loader = data_module.test_dataloader()
        rank_zero_print(f"Test dataloader batches: {len(test_loader)}")
        # Manual weight loading above; do not ask Lightning to restore checkpoint again.
        trainer.test(model, dataloaders=test_loader)
    else:
        rank_zero_print("\n" + "=" * 80)
        rank_zero_print("Starting Training")
        rank_zero_print("=" * 80 + "\n")

        # Logic to handle optimizer group changes or fine-tuning from a full checkpoint
        resume_weights_only = config.initialization.get("resume_weights_only", False)

        if ckpt_path and resume_weights_only:
            rank_zero_print(f"\n📢 [Warm Start] Loading weights ONLY from {ckpt_path}")
            rank_zero_print("   -> Optimizer and Scheduler will be re-initialized.\n")

            # Load checkpoint on the correct device (weights_only=False to allow custom classes)
            checkpoint = torch.load(
                ckpt_path, map_location=model.device, weights_only=False
            )

            # Extract state_dict (standard PL checkpoint format)
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            # Load state_dict into the model instance
            msg = model.load_state_dict(state_dict, strict=False)
            rank_zero_print(f"   -> Result: {msg}\n")

            # Clear ckpt_path so trainer.fit doesn't try a full resume of optimizer/epoch
            ckpt_path = None

        # Keep full checkpoint restore compatible with torch>=2.6.
        trainer.fit(
            model, datamodule=data_module, ckpt_path=ckpt_path, weights_only=False
        )

        rank_zero_print("waiting for syncing")
        # torch.cuda.synchronize()
        torch.distributed.barrier()

        rank_zero_print("\n" + "=" * 80)
        rank_zero_print("Training Complete!")
        rank_zero_print("=" * 80 + "\n")

        # Test the best model
        best_path = None
        best_score = None

        # 1. Try to find the EMA checkpoint callback first
        if hasattr(config.model, "ema") and config.model.ema.enabled:
            for cb in trainer.callbacks:
                if isinstance(cb, ModelCheckpoint) and cb.monitor == "val/map_ema":
                    if cb.best_model_path:
                        best_path = cb.best_model_path
                        best_score = cb.best_model_score
                        rank_zero_print(
                            f"🎯 Selected BEST EMA checkpoint (monitor: {cb.monitor})"
                        )
                    break

        # 2. Fallback to Regular checkpoint if EMA not found or not enabled
        if not best_path:
            for cb in trainer.callbacks:
                if (
                    isinstance(cb, ModelCheckpoint)
                    and cb.monitor == config.checkpointing.monitor
                ):
                    if cb.best_model_path:
                        best_path = cb.best_model_path
                        best_score = cb.best_model_score
                        rank_zero_print(
                            f"🎯 Selected BEST REGULAR checkpoint (monitor: {cb.monitor})"
                        )
                    break

        if best_path:
            rank_zero_print(f"Best model found at: {best_path}")
            if best_score is not None:
                rank_zero_print(f"Best score: {best_score:.4f}")

            rank_zero_print("\nLoading BEST checkpoint with strict=False...")

            # Actually, the safest way is to load state_dict into the CURRENT model structure
            try:
                checkpoint = torch.load(
                    best_path, map_location=model.device, weights_only=False
                )
                # If checkpoint has 'state_dict' key (PL format), use it
                state_dict = (
                    checkpoint["state_dict"]
                    if "state_dict" in checkpoint
                    else checkpoint
                )

                # Load with strict=False
                missing_keys, unexpected_keys = model.load_state_dict(
                    state_dict, strict=False
                )

                if missing_keys:
                    rank_zero_print(
                        f"⚠️  Missing keys during load: {missing_keys[:5]} ..."
                    )
                if unexpected_keys:
                    rank_zero_print(
                        f"⚠️  Unexpected keys during load: {unexpected_keys[:5]} ..."
                    )

                rank_zero_print("\nRunning test evaluation on BEST checkpoint...")
                data_module.setup(stage="test")
                if getattr(data_module, "test_dataset", None) is None:
                    raise RuntimeError(
                        "Test dataset was not initialized before best-checkpoint evaluation."
                    )
                rank_zero_print(f"Test set size: {len(data_module.test_dataset)}")
                test_loader = data_module.test_dataloader()
                rank_zero_print(f"Test dataloader batches: {len(test_loader)}")
                trainer.test(model, dataloaders=test_loader)

            except Exception as e:
                rank_zero_print(f"❌ Failed to load best checkpoint: {e}")
                import traceback

                rank_zero_print(traceback.format_exc())
        else:
            rank_zero_print("\nNo best model found. Testing disabled.")

    wandb.finish()


if __name__ == "__main__":
    main()
