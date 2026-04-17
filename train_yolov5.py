#!/usr/bin/env python3

import datetime
import os
import warnings
from typing import Any, Dict, Literal, Optional

warnings.filterwarnings("ignore", category=FutureWarning)

import hydra
import pytorch_lightning as pl
import torch
import wandb
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from lightning.pytorch.profilers import AdvancedProfiler, SimpleProfiler
from omegaconf import DictConfig, OmegaConf

# Register custom OmegaConf resolvers
OmegaConf.register_new_resolver("oc.eval", eval)
OmegaConf.register_new_resolver("extract_name", lambda path: path.split("/")[-1])

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment

from data.yolov5_data_module import YOLOv5DataModule
from models.yolov5_lightning_module import YOLOv5LightningModule
from utils.distributed_utils import rank_zero_print, setup_cluster_env
from utils.train_utils import BackupToNASCallback
from utils.ema import EMACallback
from utils.test_only_checkpoint_restore import (
    _load_ckpt,
    _load_selected_weights,
    _merge_test_only_config_from_ckpt,
    _resolve_ckpt_path,
    _select_eval_weights_source,
)

import torch.distributed as dist

# setup_cluster_env()  # Moved inside main()
torch.set_float32_matmul_precision("medium")


def _get_profiler(config: DictConfig):
    ptype = config.training.profiler.type
    if ptype == "simple":
        return SimpleProfiler(dirpath="profiler_logs", filename="yolov5_profile")
    if ptype == "advanced":
        return AdvancedProfiler(dirpath="profiler_logs", filename="yolov5_profile")
    return None


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
        config=cfg_for_log,
        save_dir=os.getcwd(),
        reinit="finish_previous",
    )


def _setup_callbacks(config: DictConfig):
    ckpt_cfg = config.checkpointing
    ckpt_dir = os.path.join(to_absolute_path(ckpt_cfg.save_dir), "ckpts")
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="yolov5-regular-epoch-{epoch:02d}-val_map-{"
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
            monitor="val/map_ema"
            if hasattr(config.model, "ema") and config.model.ema.enabled
            else ckpt_cfg.monitor,
            patience=10,
            mode=ckpt_cfg.mode,
            verbose=True,
        ),
    ]

    # EMA Callback and Checkpoint (If enabled)
    if hasattr(config.model, "ema") and config.model.ema.enabled:
        warmup_steps = config.model.ema.get("warmup_steps", 0)
        tau = config.model.ema.get("tau", 2000)
        rank_zero_print(
            f"💡 EMA enabled: Adding EMACallback with decay={config.model.ema.decay}, tau={tau}, warmup_steps={warmup_steps}"
        )
        callbacks.append(
            EMACallback(
                decay=config.model.ema.decay, tau=tau, warmup_steps=warmup_steps
            )
        )

        ema_monitor = "val/map_ema"
        callbacks.append(
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="yolov5-ema-epoch-{epoch:02d}-val_map_ema-{"
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
    test_only_weight_source: Optional[Literal["ema", "regular"]] = None
    early_ckpt_path: Optional[str] = None

    OmegaConf.set_struct(config, False)
    setup_cluster_env()
    # breakpoint()
    _resolve_run_name(config)
    rank_zero_print(f"{'*' * 80}\n[Startup] Run name: {config.run_name}\n{'*' * 80}")

    # --- Handle Hydra Sweep Logic ---
    hydra_cfg = HydraConfig.get()
    # Convert overrides to WandB tags for easy identification
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
        rank_zero_print(
            f"{'*' * 80}\n[Startup] Detected Hydra Sweep (Job {hydra_cfg.job.num})\n{'*' * 80}"
        )

        # Append sweep job index to run_name for unique directories
        config.run_name = f"{config.run_name}_run{hydra_cfg.job.num}"

        # Set WandB Group so all sweep runs are grouped together

        if not config.logging.wandb.get("group"):
            sweep_id = os.path.basename(os.path.normpath(hydra_cfg.sweep.dir))
            config.logging.wandb.group = f"sweep_{sweep_id}"

    # --- Debug Mode ---
    if config.debug:
        rank_zero_print(
            f"{'!' * 80}\n[DEBUG] Running in DEBUG/OVERFIT mode\n{'!' * 80}"
        )
        config.trainer.num_overfit_samples = 10  # Overfit on a single batch
        config.data.batch_size = 1
        config.data.eval_batch_size = config.data.batch_size  # Sync batch sizes
        config.run_name = f"DEBUG_{config.run_name}"

    base_save_dir = to_absolute_path(config.checkpointing.save_dir)
    run_save_dir = os.path.join(base_save_dir, config.run_name)
    config.checkpointing.save_dir = run_save_dir

    if config.test_only:
        early_ckpt_path = _resolve_ckpt_path(config, run_save_dir=run_save_dir)
        if not early_ckpt_path:
            raise ValueError(
                "test_only=true requires a valid checkpoint path. "
                "Check for typos and trailing punctuation (e.g., '.ckpt.'), and pass "
                "'initialization.load_from_checkpoint=/abs/path/model.ckpt'."
            )

        # Inject YOLOv5 repo into sys.path so torch.load can deserialize native YOLOv5 model classes
        import sys

        original_path = sys.path.copy()
        original_modules = {}
        try:
            yolo_repo_path = to_absolute_path(config.model.yolov5.repo_path)
            # Remove current directory and project paths from sys.path
            sys.path = [p for p in sys.path if p not in ("", ".", str(os.getcwd()))]
            if yolo_repo_path not in sys.path:
                sys.path.insert(0, yolo_repo_path)
                rank_zero_print(f"Injected YOLO repo path: {yolo_repo_path}")

            for key in list(sys.modules.keys()):
                if key.startswith(("models", "utils", "detect", "export")):
                    original_modules[key] = sys.modules.pop(key)
        except Exception as e:
            rank_zero_print(f"Warning: Could not isolate YOLO repo path. {e}")

        test_only_checkpoint = _load_ckpt(early_ckpt_path)

        # Restore sys.path
        sys.path = original_path
        # Keep imported yolov5 modules so models can be unpickled properly if used later
        config = _merge_test_only_config_from_ckpt(config, test_only_checkpoint)
        OmegaConf.set_struct(config, False)
        config.initialization.load_from_checkpoint = early_ckpt_path
        config.checkpointing.save_dir = run_save_dir

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
            early_ckpt_path, test_only_checkpoint, config
        )
        rank_zero_print(
            f"test_only: selected checkpoint weight source = {test_only_weight_source.upper()}"
        )

    OmegaConf.set_struct(config, True)

    eval_mode = config.get("eval_inference", {}).get("mode", "whole")
    rank_zero_print(f"--- Eval Inference Mode: {eval_mode.upper()} ---")
    if eval_mode in ["sliced", "both"]:
        rank_zero_print(OmegaConf.to_yaml(config.eval_inference.sahi))
        rank_zero_print("---------------------------")

    pl.seed_everything(config.seed, workers=True)

    raw_dataset_root = to_absolute_path(config.data.path)
    rank_zero_print(f"[Startup] Using YOLOv5 COCO dataset path: {raw_dataset_root}")

    yolo_repo_path = to_absolute_path(config.model.yolov5.repo_path)
    data_module = YOLOv5DataModule(
        dataset_root=raw_dataset_root,
        config=config,
    )

    # Setup data module once to prepare train/val/test splits and compute COCO annotations
    # Trainer will reuse this setup instead of calling it again
    data_module.setup(stage=None)

    lightning_model = YOLOv5LightningModule(
        config=config,
        yolo_repo_path=yolo_repo_path,
        model_to_coco=data_module.model_to_coco_map,
        val_coco_gt=data_module.val_coco_gt
        if config.debug
        else data_module.val_coco_gt,
        test_coco_gt=data_module.val_coco_gt
        if config.debug
        else data_module.test_coco_gt,
    )

    logger = _setup_logger(config)
    if logger:
        logger.experiment.save("models/yolov5_lightning_module.py")
        logger.watch(lightning_model, log='gradients', log_freq=500)
        rank_zero_print("✓ WandB logger watching model for gradients")

    callbacks = _setup_callbacks(config)
    profiler = _get_profiler(config)

    trainer = pl.Trainer(
        enable_progress_bar=True,
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
        precision=config.trainer.precision,
        strategy=config.trainer.strategy,
        max_epochs=config.trainer.max_epochs,
        log_every_n_steps=config.trainer.log_every_n_steps,
        val_check_interval=config.trainer.val_check_interval,
        gradient_clip_val=config.trainer.max_grad_norm,
        gradient_clip_algorithm=config.trainer.gradient_clip_algo,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        deterministic=config.trainer.deterministic,
        benchmark=config.trainer.benchmark,
        callbacks=callbacks,
        logger=logger,
        overfit_batches=config.trainer.num_overfit_samples if config.debug else 0,
        limit_train_batches=config.data.limit_train_batches if not config.debug else 10,
        limit_val_batches=config.data.limit_val_batches if not config.debug else 10,
        limit_test_batches=config.data.limit_test_batches if not config.debug else 10,
        profiler=None if config.debug else profiler,
        plugins=[SLURMEnvironment(auto_requeue=False)]
        if "SLURM_JOB_ID" in os.environ
        else None,
    )

    ckpt_path = _resolve_ckpt_path(config, run_save_dir=config.checkpointing.save_dir)

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

        rank_zero_print(
            f"Loading {test_only_weight_source.upper()} weights for test-only evaluation..."
        )
        missing_keys, unexpected_keys = _load_selected_weights(
            lightning_model, test_only_checkpoint, test_only_weight_source
        )
        if missing_keys:
            rank_zero_print(
                f"⚠️  Missing keys during test-only load: {missing_keys[:10]} ..."
            )
        if unexpected_keys:
            rank_zero_print(
                f"⚠️  Unexpected keys during test-only load: {unexpected_keys[:10]} ..."
            )

        # Manual weight loading above; do not ask Lightning to restore checkpoint again.
        trainer.test(lightning_model, datamodule=data_module)
    else:
        trainer.fit(
            lightning_model,
            datamodule=data_module,
            ckpt_path=ckpt_path,
            weights_only=False,
        )

        # Test the best model
        best_path = None

        # Priority 1: EMA checkpoint
        if hasattr(config.model, "ema") and config.model.ema.enabled:
            for cb in trainer.callbacks:
                if isinstance(cb, ModelCheckpoint) and cb.monitor == "val/map_ema":
                    if cb.best_model_path:
                        best_path = cb.best_model_path
                        rank_zero_print(
                            f"🎯 Selected BEST EMA checkpoint (monitor: {cb.monitor}): {best_path}"
                        )
                    break

        # Priority 2: Regular checkpoint
        if not best_path:
            for cb in trainer.callbacks:
                if (
                    isinstance(cb, ModelCheckpoint)
                    and cb.monitor == config.checkpointing.monitor
                ):
                    if cb.best_model_path:
                        best_path = cb.best_model_path
                        rank_zero_print(
                            f"🎯 Selected BEST REGULAR checkpoint (monitor: {cb.monitor}): {best_path}"
                        )
                    break

        eval_ckpt = best_path if best_path else "best"
        trainer.test(
            lightning_model,
            datamodule=data_module,
            ckpt_path=eval_ckpt,
            weights_only=False,
        )

    # 1. Stop all GPUs and ensure testing is 100% complete across the cluster
    if dist.is_initialized():
        dist.barrier()

    # 2. Safely close WandB ONLY on Rank 0
    # Lightning's global zero is the only one maintaining the WandB connection
    if trainer.is_global_zero:
        wandb.finish()


if __name__ == "__main__":
    main()
