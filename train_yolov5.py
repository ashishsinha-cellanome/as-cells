#!/usr/bin/env python3

import datetime
import os

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import to_absolute_path
from lightning.pytorch.profilers import AdvancedProfiler, SimpleProfiler
from omegaconf import DictConfig, OmegaConf

# Register custom OmegaConf resolvers
OmegaConf.register_new_resolver("oc.eval", eval)

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment

from data.yolov5_data_module import YOLOv5DataModule
from models.yolov5_lightning_module import YOLOv5LightningModule
from utils.distributed_utils import rank_zero_print, setup_cluster_env
from utils.train_utils import BackupToNASCallback

setup_cluster_env()
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
        group=wandb_cfg.get("group"),
        config=cfg_for_log,
        save_dir=os.getcwd(),
        reinit=True,
    )


def _setup_callbacks(config: DictConfig):
    ckpt_cfg = config.checkpointing
    ckpt_dir = os.path.join(to_absolute_path(ckpt_cfg.save_dir), "ckpts")
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="yolov5-epoch{epoch:02d}-val_map{" + ckpt_cfg.monitor.replace("/", "_") + ":.4f}",
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
    ]
    if "backup_dir" in ckpt_cfg and ckpt_cfg.backup_dir:
        callbacks.append(BackupToNASCallback(backup_dir=to_absolute_path(ckpt_cfg.backup_dir)))

    # Add EMA callback if enabled
    if getattr(config.model, "use_ema", True):
        from utils.ema import RTDETREMACallback
        ema_decay = getattr(config.model, "ema_decay", 0.9999)
        ema_warmup_steps = getattr(config.model, "ema_warmup_steps", 0)
        callbacks.append(RTDETREMACallback(decay=ema_decay, warmup_steps=ema_warmup_steps))

    return callbacks


def _resolve_run_name(config: DictConfig):
    unique_id = (
        os.environ.get("SLURM_JOB_ID")
        or os.environ.get("TORCHELASTIC_RUN_ID")
        or os.environ.get("WANDB_RUN_ID")
        or datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    config.run_name = f"{config.run_name}_{unique_id}"


@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(config: DictConfig):
    OmegaConf.set_struct(config, False)
    _resolve_run_name(config)
    base_save_dir = to_absolute_path(config.checkpointing.save_dir)
    config.checkpointing.save_dir = os.path.join(base_save_dir, config.run_name)
    OmegaConf.set_struct(config, True)

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
        val_coco_gt=data_module.val_coco_gt,
        test_coco_gt=data_module.test_coco_gt,
    )

    logger = _setup_logger(config)
    callbacks = _setup_callbacks(config)
    profiler = _get_profiler(config)

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
        callbacks=callbacks,
        logger=logger,
        overfit_batches=config.trainer.num_overfit_samples if config.debug else 0,
        limit_train_batches=config.data.limit_train_batches if not config.debug else 10,
        limit_val_batches=config.data.limit_val_batches if not config.debug else 10,
        limit_test_batches=config.data.limit_test_batches if not config.debug else 10,
        profiler=None if config.debug else profiler,
        plugins=[SLURMEnvironment(auto_requeue=False)] if "SLURM_JOB_ID" in os.environ else None,
    )

    ckpt_path = config.initialization.load_from_checkpoint
    if ckpt_path:
        ckpt_path = to_absolute_path(ckpt_path)
        if not os.path.exists(ckpt_path):
            rank_zero_print(f"Checkpoint not found: {ckpt_path}")
            ckpt_path = None
    elif config.initialization.auto_resume:
        candidate = os.path.join(config.checkpointing.save_dir, "ckpts", "last.ckpt")
        ckpt_path = candidate if os.path.exists(candidate) else None
    else:
        ckpt_path = None

    rank_zero_print(OmegaConf.to_yaml(config))
    if config.test_only:
        if not ckpt_path:
            raise ValueError("test_only=true requires initialization.load_from_checkpoint")
        trainer.test(lightning_model, datamodule=data_module, ckpt_path=ckpt_path)
    else:
        trainer.fit(lightning_model, datamodule=data_module, ckpt_path=ckpt_path)
        trainer.test(lightning_model, datamodule=data_module, ckpt_path="best")


if __name__ == "__main__":
    main()
