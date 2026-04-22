#!/usr/bin/env python3

import datetime
import os
import warnings
from typing import Any, Dict, Optional

# Must be called BEFORE importing transformers or lightning modules that depend on it
from utils.distributed_utils import setup_cluster_env, get_rank, rank_zero_print

setup_cluster_env()

import hydra
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from hydra.utils import to_absolute_path
from lightning.pytorch.profilers import AdvancedProfiler, SimpleProfiler
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment

from data.mask2former_data_module import Mask2FormerDataModule
from models.mask2former_lightning_module import Mask2FormerLightningModule
from models.mask2former_model import (
    build_mask2former_with_dinov2_backbone,
    build_original_mask2former,
    get_mask2former_processor,
    summarize_trainable_parameters,
)
from utils.test_only_checkpoint_restore import (
    _load_ckpt,
    _load_selected_weights,
    _merge_test_only_config_from_ckpt,
    _resolve_ckpt_path,
    _select_eval_weights_source,
)
from utils.train_utils import BackupToNASCallback

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Grad strides do not match bucket view strides.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*The AccumulateGrad node's stream does not match the stream.*", category=UserWarning)

# --- MEMORY EFFICIENT PATCH FOR MASK2FORMER LOSS ---
# import numpy as np
# from scipy.optimize import linear_sum_assignment
# from transformers.models.mask2former.modeling_mask2former import (
#     Mask2FormerLoss, 
#     Mask2FormerHungarianMatcher,
#     sample_point, 
#     sigmoid_cross_entropy_loss, 
#     dice_loss,
#     pair_wise_sigmoid_cross_entropy_loss,
#     pair_wise_dice_loss
# )

# def memory_efficient_matcher_forward(
#     self,
#     masks_queries_logits: torch.Tensor,
#     class_queries_logits: torch.Tensor,
#     mask_labels: list[torch.Tensor],
#     class_labels: list[torch.Tensor],
# ) -> list[tuple[torch.Tensor]]:
#     indices = []
#     batch_size = masks_queries_logits.shape[0]
    
#     for i in range(batch_size):
#         pred_probs = class_queries_logits[i].softmax(-1)
#         pred_mask = masks_queries_logits[i]

#         cost_class = -pred_probs[:, class_labels[i]]
        
#         target_mask_bool = mask_labels[i]
#         num_gt = target_mask_bool.shape[0]
        
#         pred_mask = pred_mask[:, None]
#         point_coordinates = torch.rand(1, self.num_points, 2, device=pred_mask.device)
#         pred_coordinates = point_coordinates.repeat(pred_mask.shape[0], 1, 1)
#         pred_mask_sampled = sample_point(pred_mask, pred_coordinates, align_corners=False).squeeze(1)
        
#         CHUNK_SIZE = 256
#         cost_mask_list = []
#         cost_dice_list = []
        
#         target_coordinates_chunk = point_coordinates.repeat(CHUNK_SIZE, 1, 1)
        
#         for chunk_start in range(0, num_gt, CHUNK_SIZE):
#             chunk_end = min(num_gt, chunk_start + CHUNK_SIZE)
#             actual_chunk_size = chunk_end - chunk_start
            
#             target_mask_chunk = target_mask_bool[chunk_start:chunk_end].to(pred_mask.dtype)[:, None]
            
#             if actual_chunk_size == CHUNK_SIZE:
#                 coords = target_coordinates_chunk
#             else:
#                 coords = point_coordinates.repeat(actual_chunk_size, 1, 1)
                
#             target_mask_sampled_chunk = sample_point(target_mask_chunk, coords, align_corners=False).squeeze(1)
            
#             cost_mask_chunk = pair_wise_sigmoid_cross_entropy_loss(pred_mask_sampled, target_mask_sampled_chunk)
#             cost_dice_chunk = pair_wise_dice_loss(pred_mask_sampled, target_mask_sampled_chunk)
            
#             cost_mask_list.append(cost_mask_chunk)
#             cost_dice_list.append(cost_dice_chunk)
            
#             del target_mask_chunk, target_mask_sampled_chunk
            
#         if len(cost_mask_list) > 0:
#             cost_mask = torch.cat(cost_mask_list, dim=1)
#             cost_dice = torch.cat(cost_dice_list, dim=1)
#         else:
#             cost_mask = torch.empty((pred_mask.shape[0], 0), device=pred_mask.device)
#             cost_dice = torch.empty((pred_mask.shape[0], 0), device=pred_mask.device)
            
#         cost_matrix = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
#         cost_matrix = torch.minimum(cost_matrix, torch.tensor(1e10))
#         cost_matrix = torch.maximum(cost_matrix, torch.tensor(-1e10))
#         cost_matrix = torch.nan_to_num(cost_matrix, 0)
        
#         assigned_indices = linear_sum_assignment(cost_matrix.cpu())
#         indices.append(assigned_indices)

#     matched_indices = [
#         (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices
#     ]
#     return matched_indices

# Mask2FormerHungarianMatcher.forward = memory_efficient_matcher_forward

# def memory_efficient_loss_masks(
#     self,
#     masks_queries_logits: torch.Tensor,
#     mask_labels: list[torch.Tensor],
#     indices: tuple[np.ndarray],
#     num_masks: int,
# ) -> dict[str, torch.Tensor]:
#     src_idx = self._get_predictions_permutation_indices(indices)
#     tgt_idx = self._get_targets_permutation_indices(indices)
    
#     pred_masks = masks_queries_logits[src_idx]
    
#     batch_idx, target_idx = tgt_idx
#     if len(batch_idx) == 0:
#         target_masks = torch.empty((0, masks_queries_logits.shape[-2], masks_queries_logits.shape[-1]), device=masks_queries_logits.device)
#     else:
#         target_masks = torch.stack([mask_labels[b][i] for b, i in zip(batch_idx, target_idx)])

#     pred_masks = pred_masks[:, None]
#     target_masks = target_masks[:, None].to(pred_masks)

#     with torch.no_grad():
#         point_coordinates = self.sample_points_using_uncertainty(
#             pred_masks,
#             lambda logits: self.calculate_uncertainty(logits),
#             self.num_points,
#             self.oversample_ratio,
#             self.importance_sample_ratio,
#         )

#         point_labels = sample_point(target_masks, point_coordinates, align_corners=False).squeeze(1)

#     point_logits = sample_point(pred_masks, point_coordinates, align_corners=False).squeeze(1)

#     losses = {
#         "loss_mask": sigmoid_cross_entropy_loss(point_logits, point_labels, num_masks),
#         "loss_dice": dice_loss(point_logits, point_labels, num_masks),
#     }

#     del pred_masks
#     del target_masks
#     return losses

# Mask2FormerLoss.loss_masks = memory_efficient_loss_masks
# --------------------------------------------------

OmegaConf.register_new_resolver(
    "extract_name", lambda path: path.split("/")[-1], replace=True
)
OmegaConf.register_new_resolver("oc.eval", eval, replace=True)

torch.set_float32_matmul_precision("medium")


def _runtime_launch_summary(config: DictConfig) -> str:
    world_size = int(
        os.environ.get("WORLD_SIZE")
        or os.environ.get("SLURM_NTASKS")
        or os.environ.get("LOCAL_WORLD_SIZE")
        or 1
    )
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    accelerator = str(config.trainer.accelerator)
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return (
        "[Startup] Runtime launch:\n"
        f"  -> rank={get_rank()} local_rank={local_rank} world_size={world_size}\n"
        f"  -> accelerator={accelerator} devices={config.trainer.devices} "
        f"strategy={config.trainer.strategy}\n"
        f"  -> cuda_available={torch.cuda.is_available()} cuda_device_count={device_count}"
    )


def _warn_if_accelerate_launch() -> None:
    accelerate_env_vars = (
        "ACCELERATE_MIXED_PRECISION",
        "ACCELERATE_CONFIG_DS_FIELDS",
        "ACCELERATE_DYNAMO_BACKEND",
        "ACCELERATE_USE_FSDP",
    )
    if any(var in os.environ for var in accelerate_env_vars):
        rank_zero_print(
            "[Startup] Detected Accelerate launcher environment. "
            "Run Mask2Former directly with `uv run train_mask2former.py ...` "
            "so Lightning handles process launch."
        )


def _get_profiler(config: DictConfig):
    ptype = config.training.profiler.type
    if ptype == "simple":
        return SimpleProfiler(dirpath="profiler_logs", filename="mask2former_profile")
    if ptype == "advanced":
        return AdvancedProfiler(dirpath="profiler_logs", filename="mask2former_profile")
    return None


def _build_model_to_coco_map(coco_gt, label_map: Dict[int, str]) -> Dict[int, int]:
    name_to_model_id = {str(name): int(idx) for idx, name in label_map.items()}
    model_to_coco: Dict[int, int] = {}

    if coco_gt is None or not getattr(coco_gt, "cats", None):
        for model_id in sorted(name_to_model_id.values()):
            model_to_coco[model_id] = model_id
        return model_to_coco

    for coco_cat_id, cat_info in coco_gt.cats.items():
        cat_name = cat_info.get("name")
        if cat_name in name_to_model_id:
            model_to_coco[name_to_model_id[cat_name]] = int(coco_cat_id)

    for model_id in sorted(name_to_model_id.values()):
        model_to_coco.setdefault(model_id, model_id)

    rank_zero_print(f"[Startup] Mask2Former model_to_coco mapping: {model_to_coco}")
    return model_to_coco


def _setup_logger(config: DictConfig):
    wandb_cfg = config.logging.wandb
    if not wandb_cfg.enabled:
        return None
    cfg_for_log = OmegaConf.to_container(config, resolve=True)
    return WandbLogger(
        project=wandb_cfg.project,
        name=config.run_name,
        tags=list(wandb_cfg.tags),
        notes=wandb_cfg.notes,
        group=wandb_cfg.get("group"),
        config=cfg_for_log,
        save_dir=os.getcwd(),
        reinit="finish_previous",
    )


def _setup_callbacks(config: DictConfig):
    ckpt_cfg = config.checkpointing
    ckpt_dir = os.path.join(to_absolute_path(ckpt_cfg.save_dir), "ckpts")
    primary_early_stop_monitor = (
        "val/segm_map_ema"
        if hasattr(config.model, "ema") and config.model.ema.enabled
        else ckpt_cfg.monitor
    )
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="mask2former-epoch{epoch:02d}-val_segm_map{"
            + ckpt_cfg.monitor.replace("/", "_")
            + ":.4f}",
            monitor=ckpt_cfg.monitor,
            mode=ckpt_cfg.mode,
            save_top_k=ckpt_cfg.save_top_k,
            save_last=ckpt_cfg.save_last,
            every_n_epochs=ckpt_cfg.every_n_epochs,
            auto_insert_metric_name=False,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        ModelSummary(max_depth=3),
        EarlyStopping(
            monitor=primary_early_stop_monitor,
            patience=5,
            mode=ckpt_cfg.mode,
            verbose=True,
        ),
    ]

    if hasattr(config.model, "ema") and config.model.ema.enabled:
        from utils.ema import EMACallback

        warmup_steps = int(config.model.ema.get("warmup_steps", 0))
        tau = config.model.ema.get("tau", None)
        rank_zero_print(
            f"💡 EMA enabled: Adding EMACallback with decay={config.model.ema.decay}, "
            f"tau={tau}, warmup_steps={warmup_steps}"
        )
        callbacks.append(
            EMACallback(
                decay=float(config.model.ema.decay),
                tau=None if tau is None else float(tau),
                warmup_steps=warmup_steps,
            )
        )
        ema_monitor = "val/segm_map_ema"
        callbacks.append(
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="mask2former-ema-epoch{epoch:02d}-val_segm_map_ema{"
                + ema_monitor.replace("/", "_")
                + ":.4f}",
                monitor=ema_monitor,
                mode=ckpt_cfg.mode,
                save_top_k=ckpt_cfg.save_top_k,
                save_last=False,
                every_n_epochs=ckpt_cfg.every_n_epochs,
                auto_insert_metric_name=False,
                verbose=True,
            )
        )
    if "backup_dir" in ckpt_cfg and ckpt_cfg.backup_dir:
        callbacks.append(
            BackupToNASCallback(backup_dir=to_absolute_path(ckpt_cfg.backup_dir))
        )
    return callbacks


def _validate_lora_config(config: DictConfig) -> None:
    training_mode = str(config.model.backbone.training_mode).lower()
    lora_enabled = bool(config.model.lora.enabled)

    if training_mode == "lora" and not lora_enabled:
        raise ValueError(
            "Mask2Former LoRA requires model.backbone.training_mode=lora and "
            "model.lora.enabled=true."
        )

    if lora_enabled and training_mode != "lora":
        raise ValueError(
            "model.lora.enabled=true requires model.backbone.training_mode=lora."
        )

    if lora_enabled and not list(config.model.lora.target_modules):
        raise ValueError(
            "model.lora.target_modules must be non-empty when LoRA is enabled."
        )


def _resolve_run_name(config: DictConfig):
    unique_id = (
        os.environ.get("SLURM_JOB_ID")
        or os.environ.get("TORCHELASTIC_RUN_ID")
        or os.environ.get("WANDB_RUN_ID")
        or ""
    )
    timestamp = datetime.datetime.now().strftime("%H-%M")
    unique_id = (
        f"{unique_id}_{timestamp}"
        if unique_id
        else datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    )
    config.run_name = f"{config.run_name}_{unique_id}"


@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(config: DictConfig):
    test_only_checkpoint: Optional[Dict[str, Any]] = None
    test_only_weight_source: Optional[str] = None
    early_ckpt_path: Optional[str] = None

    OmegaConf.set_struct(config, False)
    _resolve_run_name(config)

    if hasattr(config.model, "checkpointing"):
        config.checkpointing.monitor = config.model.checkpointing.get(
            "monitor", "val/map"
        )
        config.checkpointing.mode = config.model.checkpointing.get("mode", "max")
        if "visualize_samples" in config.model.checkpointing:
            config.checkpointing.visualize_samples = (
                config.model.checkpointing.visualize_samples
            )
        if "visualize_every_n_epochs" in config.model.checkpointing:
            config.checkpointing.visualize_every_n_epochs = (
                config.model.checkpointing.visualize_every_n_epochs
            )
    else:
        config.checkpointing.monitor = "val/map"
        config.checkpointing.mode = "max"

    hydra_cfg = HydraConfig.get()
    job_overrides = hydra_cfg.overrides.task
    for override in job_overrides:
        if "=" in override:
            key, value = override.split("=", 1)
            short_key = key.split(".")[-1]
            if short_key == "load_from_checkpoint":
                continue
            tag = f"{short_key}={value}"
            config.logging.wandb.tags.append(tag)
            rank_zero_print(f"   -> Added WandB tag: {tag}")

    if hydra_cfg.mode == RunMode.MULTIRUN:
        rank_zero_print(f"Detected Hydra Sweep (Job {hydra_cfg.job.num})")
        config.run_name = f"{config.run_name}_run{hydra_cfg.job.num}"
        if not config.logging.wandb.get("group"):
            sweep_id = os.path.basename(os.path.normpath(hydra_cfg.sweep.dir))
            config.logging.wandb.group = f"sweep_{sweep_id}"

    if getattr(config.data, "num_overfit_samples", 0) > 0:
        config.debug = True

    if config.debug:
        rank_zero_print(
            f"{'!' * 80}\n[DEBUG] Running in DEBUG/OVERFIT mode\n{'!' * 80}"
        )
        overfit_samples = getattr(config.data, "num_overfit_samples", 1)
        config.trainer.num_overfit_samples = (
            overfit_samples if overfit_samples > 0 else 1
        )
        config.data.eval_batch_size = config.data.batch_size
        if not str(config.run_name).startswith("DEBUG_"):
            config.run_name = f"DEBUG_{config.run_name}"

    base_save_dir = to_absolute_path(config.checkpointing.save_dir)
    run_save_dir = os.path.join(base_save_dir, config.run_name)
    config.checkpointing.save_dir = run_save_dir

    if config.test_only:
        early_ckpt_path = _resolve_ckpt_path(config, run_save_dir=run_save_dir)
        if not early_ckpt_path:
            raise ValueError(
                "test_only=true requires a valid checkpoint path. "
                "Pass initialization.load_from_checkpoint=/abs/path/model.ckpt."
            )
        test_only_checkpoint = _load_ckpt(early_ckpt_path)
        config = _merge_test_only_config_from_ckpt(config, test_only_checkpoint)
        OmegaConf.set_struct(config, False)
        config.initialization.load_from_checkpoint = early_ckpt_path
        config.checkpointing.save_dir = run_save_dir
        test_only_weight_source = _select_eval_weights_source(
            early_ckpt_path, test_only_checkpoint, config
        )
        rank_zero_print(
            f"test_only: selected checkpoint weight source = {test_only_weight_source.upper()}"
        )

    OmegaConf.set_struct(config, True)
    _validate_lora_config(config)

    pl.seed_everything(config.seed, workers=True)
    _warn_if_accelerate_launch()

    dataset_path = to_absolute_path(config.data.path)
    rank_zero_print(f"[Startup] Using Mask2Former dataset path: {dataset_path}")
    rank_zero_print(_runtime_launch_summary(config))
    rank_zero_print(
        f"[Startup] Mask2Former backbone training_mode: "
        f"{config.model.backbone.training_mode}"
    )
    label_map = {int(k): v for k, v in config.model.label_map.items()}
    if hasattr(config.model, "ema"):
        rank_zero_print(
            "[Startup] Mask2Former EMA: "
            f"enabled={bool(config.model.ema.enabled)}, "
            f"decay={config.model.ema.decay}, "
            f"tau={config.model.ema.get('tau', None)}, "
            f"warmup_steps={config.model.ema.get('warmup_steps', 0)}"
        )

    primary_early_stop_monitor = (
        "val/segm_map_ema"
        if hasattr(config.model, "ema") and config.model.ema.enabled
        else config.checkpointing.monitor
    )
    test_monitors = (
        ["val/segm_map_ema", config.checkpointing.monitor]
        if hasattr(config.model, "ema") and config.model.ema.enabled
        else [config.checkpointing.monitor]
    )

    ema_enabled = hasattr(config.model, "ema") and config.model.ema.enabled
    ema_monitor_log = (
        "\n  -> EMA Checkpointing Monitor: val/segm_map_ema" if ema_enabled else ""
    )

    rank_zero_print(
        "[Startup] Metrics Tracking:\n"
        f"  -> Standard Checkpointing Monitor: {config.checkpointing.monitor}"
        f"{ema_monitor_log}\n"
        f"  -> Early Stopping Monitor: {primary_early_stop_monitor}\n"
        f"  -> Test Phase Checkpoint Selection: {test_monitors}"
    )

    image_processor = get_mask2former_processor(int(config.model.input_size))
    data_module = Mask2FormerDataModule(
        dataset_path=dataset_path,
        image_processor=image_processor,
        config=config,
    )
    data_module.setup("fit")
    data_module.setup("test")

    backbone_pretrained = config.model.backbone.get("pretrained_name_or_path")
    if backbone_pretrained is not None:
        # DINOv2 backbone path
        model = build_mask2former_with_dinov2_backbone(
            id2label=label_map,
            mask2former_pretrained_name_or_path=str(
                config.model.mask2former.pretrained_name_or_path
            ),
            backbone_pretrained_name_or_path=str(backbone_pretrained),
            training_mode=str(config.model.backbone.training_mode),
            lora_config=OmegaConf.to_container(config.model.lora, resolve=True),
            num_queries=int(config.model.mask2former.num_queries),
            out_indices=[int(x) for x in config.model.backbone.out_indices],
            local_files_only=bool(
                config.model.mask2former.get("local_files_only", False)
            ),
            fpn_type=str(config.model.backbone.get("fpn_type", "fused")),
            gradient_checkpointing=bool(
                config.model.backbone.get("gradient_checkpointing", False)
            ),
            adapter_config=OmegaConf.to_container(config.model.backbone, resolve=True)
            if str(config.model.backbone.get("fpn_type", "fused")).lower()
            == "adapter"
            else None,
            intermediate_resolutions=list(
                config.model.backbone.get("intermediate_resolutions", [168, 84, 42, 21])
            )
            if config.model.backbone.get("intermediate_resolutions")
            else None,
        )
    else:
        # Original Swin backbone path
        model = build_original_mask2former(
            id2label=label_map,
            mask2former_pretrained_name_or_path=str(
                config.model.mask2former.pretrained_name_or_path
            ),
            training_mode=str(config.model.backbone.training_mode),
            num_queries=int(config.model.mask2former.num_queries),
            out_indices=[int(x) for x in config.model.backbone.out_indices],
            local_files_only=bool(
                config.model.mask2former.get("local_files_only", False)
            ),
        )
    param_summary = summarize_trainable_parameters(model)
    rank_zero_print(
        "[Startup] Mask2Former params: "
        f"trainable={param_summary['trainable_params']:,} / "
        f"total={param_summary['total_params']:,} "
        f"(backbone trainable={param_summary['trainable_backbone_params']:,})"
    )
    if bool(config.model.lora.enabled):
        rank_zero_print(
            "[Startup] LoRA enabled: "
            f"rank={config.model.lora.rank}, "
            f"alpha={config.model.lora.alpha}, "
            f"dropout={config.model.lora.dropout}, "
            f"target_modules={list(config.model.lora.target_modules)}"
        )

    model_to_coco = _build_model_to_coco_map(data_module.val_coco_gt, label_map)
    lightning_model = Mask2FormerLightningModule(
        model=model,
        image_processor=image_processor,
        config=config,
        model_to_coco=model_to_coco,
        val_coco_gt=data_module.val_coco_gt,
        test_coco_gt=data_module.test_coco_gt,
        val_segm_coco_gt=data_module.val_segm_coco_gt,
        test_segm_coco_gt=data_module.test_segm_coco_gt,
        val_image_root=data_module.val_image_root,
        test_image_root=data_module.test_image_root,
    )

    logger = _setup_logger(config)
    if logger:
        logger.experiment.save("models/mask2former_lightning_module.py")
        logger.watch(lightning_model, log="gradients", log_freq=500)

    callbacks = _setup_callbacks(config)
    profiler = _get_profiler(config)
    
    gradient_clip_val = float(config.trainer.get("max_grad_norm", 0.01))
    if gradient_clip_val > 0.05:
        rank_zero_print(f"⚠️  WARNING: max_grad_norm is {gradient_clip_val}. Mask2Former is unstable with > 0.01. Forcing 0.01!")
        gradient_clip_val = 0.01

    trainer = pl.Trainer(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
        precision=config.trainer.precision,
        strategy=config.trainer.strategy,
        max_epochs=config.trainer.max_epochs,
        log_every_n_steps=config.trainer.log_every_n_steps,
        val_check_interval=config.trainer.val_check_interval,
        num_sanity_val_steps=int(config.trainer.num_sanity_val_steps),
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm=config.trainer.gradient_clip_algo,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        deterministic=config.trainer.deterministic,
        benchmark=config.trainer.benchmark,
        enable_progress_bar=True,
        callbacks=callbacks,
        logger=logger,
        overfit_batches=config.trainer.num_overfit_samples if config.debug else 0,
        limit_train_batches=config.data.limit_train_batches if not config.debug else 10,
        limit_val_batches=config.data.limit_val_batches if not config.debug else 10,
        limit_test_batches=config.data.limit_test_batches if not config.debug else 10,
        profiler=None if config.debug else profiler,
        plugins=[SLURMEnvironment(auto_requeue=True)]
        if "SLURM_JOB_ID" in os.environ
        else None,
    )

    ckpt_path = _resolve_ckpt_path(config, run_save_dir=config.checkpointing.save_dir)

    # Auto-load Mask2Former specific transforms into the global config for logging/printing
    transforms_path = os.path.join(
        hydra.utils.get_original_cwd(), "configs/data/mask2former_transforms.yaml"
    )
    if os.path.exists(transforms_path):
        m2f_cfg = OmegaConf.load(transforms_path)
        if hasattr(m2f_cfg, "transforms"):
            OmegaConf.set_struct(config, False)
            if not hasattr(config, "data"):
                config.data = OmegaConf.create({})
            config.data.transforms = m2f_cfg.transforms
            OmegaConf.set_struct(config, True)

    rank_zero_print(OmegaConf.to_yaml(config))

    if config.test_only:
        if not ckpt_path:
            raise ValueError(
                "test_only=true requires initialization.load_from_checkpoint"
            )
        if test_only_checkpoint is None:
            test_only_checkpoint = _load_ckpt(ckpt_path)
        if test_only_weight_source is None:
            test_only_weight_source = _select_eval_weights_source(
                ckpt_path, test_only_checkpoint
            )

        missing_keys, unexpected_keys = _load_selected_weights(
            lightning_model, test_only_checkpoint, test_only_weight_source
        )
        if missing_keys:
            rank_zero_print(
                f"⚠️ Missing keys during test-only load: {missing_keys[:10]} ..."
            )
        if unexpected_keys:
            rank_zero_print(
                f"⚠️ Unexpected keys during test-only load: {unexpected_keys[:10]} ..."
            )
        trainer.test(lightning_model, datamodule=data_module)
    else:
        trainer.fit(
            lightning_model,
            datamodule=data_module,
            ckpt_path=ckpt_path,
            weights_only=False,
        )

        best_path = None
        preferred_monitors = (
            ["val/segm_map_ema", config.checkpointing.monitor]
            if hasattr(config.model, "ema") and config.model.ema.enabled
            else [config.checkpointing.monitor]
        )
        for monitor_name in preferred_monitors:
            for cb in trainer.callbacks:
                if (
                    isinstance(cb, ModelCheckpoint)
                    and cb.monitor == monitor_name
                    and cb.best_model_path
                ):
                    best_path = cb.best_model_path
                    rank_zero_print(
                        f"🎯 Selected BEST checkpoint (monitor: {cb.monitor}): {best_path}"
                    )
                    break
            if best_path:
                break

        trainer.test(
            lightning_model,
            datamodule=data_module,
            ckpt_path=best_path if best_path else "best",
            weights_only=False,
        )

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if get_rank() == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
