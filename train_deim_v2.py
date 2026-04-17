#!/usr/bin/env python3

import datetime
import os
import warnings
from typing import Any, Dict, Literal, Optional

import hydra
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment

from data.deim_v2_data_module import DeimV2DataModule
from torchvision.datasets import CocoDetection
from models.deim_v2_lightning_module import DeimV2LightningModule
from utils.distributed_utils import rank_zero_print, setup_cluster_env
from utils.test_only_checkpoint_restore import (
    _load_ckpt,
    _load_selected_weights,
    _merge_test_only_config_from_ckpt,
    _resolve_ckpt_path,
    _select_eval_weights_source,
)
from utils.train_utils import BackupToNASCallback

warnings.filterwarnings("ignore", category=FutureWarning)
OmegaConf.register_new_resolver(
    "extract_name", lambda path: path.split("/")[-1], replace=True
)
OmegaConf.register_new_resolver("oc.eval", eval, replace=True)

setup_cluster_env()
torch.set_float32_matmul_precision("medium")


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
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="deimv2-epoch{epoch:02d}-val_map{"
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
            patience=20,
            mode=ckpt_cfg.mode,
            verbose=True,
        ),
    ]

    if hasattr(config.model, "ema") and config.model.ema.enabled:
        from utils.ema import EMACallback

        warmup_steps = config.model.ema.get("warmup_steps", 0)
        tau = config.model.ema.get("tau", 2000)
        rank_zero_print(
            f"💡 EMA enabled: Adding EMACallback with decay={config.model.ema.decay}, "
            f"tau={tau}, warmup_steps={warmup_steps}"
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
                filename="deimv2-ema-{epoch:02d}-val_map_ema{"
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
    _resolve_run_name(config)

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
                "Check for typos/trailing punctuation and pass "
                "'initialization.load_from_checkpoint=/abs/path/model.ckpt'."
            )
        test_only_checkpoint = _load_ckpt(early_ckpt_path)
        config = _merge_test_only_config_from_ckpt(config, test_only_checkpoint)
        OmegaConf.set_struct(config, False)
        config.initialization.load_from_checkpoint = early_ckpt_path
        config.checkpointing.save_dir = run_save_dir

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

    OmegaConf.set_struct(config, True)
    pl.seed_everything(config.seed, workers=True)

    dataset_path = to_absolute_path(config.data.path)
    rank_zero_print(f"[Startup] Using DEIMv2 dataset path: {dataset_path}")

    data_module = DeimV2DataModule(
        dataset_path=dataset_path,
        config=config,
    )
    data_module.setup()

    val_annot_path = os.path.join(dataset_path, "images", config.val_name)
    val_json_path = os.path.join(dataset_path, f"{config.val_name}_annotations.json")
    val_coco_dataset = CocoDetection(
        root=val_annot_path, annFile=val_json_path, transforms=None
    )
    val_coco_gt = val_coco_dataset.coco

    test_annot_path = os.path.join(dataset_path, "images", config.test_name)
    test_json_path = os.path.join(dataset_path, f"{config.test_name}_annotations.json")
    test_coco_dataset = CocoDetection(
        root=test_annot_path, annFile=test_json_path, transforms=None
    )
    test_coco_gt = test_coco_dataset.coco

    lightning_model = DeimV2LightningModule(
        config=config,
        val_coco_gt=val_coco_gt,
        test_coco_gt=val_coco_gt if config.debug else test_coco_gt,
        val_image_root=val_annot_path,
        test_image_root=val_annot_path if config.debug else test_annot_path,
    )

    logger = _setup_logger(config)
    if logger:
        logger.experiment.save("models/deim_v2_lightning_module.py")
        logger.watch(lightning_model, log="gradients", log_freq=500)
        rank_zero_print("✓ WandB logger watching model for gradients")

    callbacks = _setup_callbacks(config)

    trainer = pl.Trainer(
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
        enable_progress_bar=True,
        callbacks=callbacks,
        logger=logger,
        overfit_batches=config.trainer.num_overfit_samples if config.debug else 0,
        limit_train_batches=config.data.limit_train_batches if not config.debug else 10,
        limit_val_batches=config.data.limit_val_batches if not config.debug else 10,
        limit_test_batches=config.data.limit_test_batches if not config.debug else 10,
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

        trainer.test(lightning_model, datamodule=data_module)
    else:
        trainer.fit(lightning_model, datamodule=data_module, ckpt_path=ckpt_path)

        best_path = None
        if hasattr(config.model, "ema") and config.model.ema.enabled:
            for cb in trainer.callbacks:
                if isinstance(cb, ModelCheckpoint) and cb.monitor == "val/map_ema":
                    if cb.best_model_path:
                        best_path = cb.best_model_path
                        rank_zero_print(
                            f"🎯 Selected BEST EMA checkpoint (monitor: {cb.monitor}): {best_path}"
                        )
                    break

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
        trainer.test(lightning_model, datamodule=data_module, ckpt_path=eval_ckpt)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if trainer.is_global_zero:
        wandb.finish()


if __name__ == "__main__":
    main()
