#!/usr/bin/env python3

import datetime
import os

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import to_absolute_path
from lightning.pytorch.profilers import AdvancedProfiler, SimpleProfiler
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
from rfdetr.models import build_criterion_and_postprocessors

from data.rf_detr_data_module import RFDETRDataModule
from models.rf_detr_lightning_module import RFDETRLightningModule
from utils.data_layout_utils import prepare_rfdetr_roboflow_layout
from utils.distributed_utils import rank_zero_print, setup_cluster_env
from utils.train_utils import BackupToNASCallback

setup_cluster_env()
torch.set_float32_matmul_precision("medium")


def _get_profiler(config: DictConfig):
    ptype = config.training.profiler.type
    if ptype == "simple":
        return SimpleProfiler(dirpath="profiler_logs", filename="rfdetr_profile")
    if ptype == "advanced":
        return AdvancedProfiler(dirpath="profiler_logs", filename="rfdetr_profile")
    return None


def _get_model_class(size_name: str):
    size_name = str(size_name).lower()
    if size_name == "base":
        return RFDETRBase
    if size_name == "small":
        return RFDETRSmall
    if size_name == "medium":
        return RFDETRMedium
    if size_name == "large":
        return RFDETRLarge
    if size_name == "nano":
        return RFDETRNano
    raise ValueError(f"Unsupported RF-DETR size: {size_name}")


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
            filename="rfdetr-epoch{epoch:02d}-val_map{" + ckpt_cfg.monitor.replace("/", "_") + ":.4f}",
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

    cache_dir = to_absolute_path(config.model.rfdetr.dataset_cache_dir)
    rank_zero_print(f"[Startup] Preparing RF-DETR layout cache at: {cache_dir}")
    staged_dataset_dir = prepare_rfdetr_roboflow_layout(
        dataset_path=to_absolute_path(config.data.path),
        cache_root=cache_dir,
        train_name=config.train_name,
        val_name=config.val_name,
        test_name=config.test_name,
    )
    rank_zero_print("[Startup] RF-DETR layout ready.")

    label_map = {int(k): v for k, v in config.model.label_map.items()}
    class_names = [label_map[idx] for idx in sorted(label_map.keys())]

    rf_model_cls = _get_model_class(config.model.rfdetr.size)
    rank_zero_print(f"[Startup] Building RF-DETR model ({config.model.rfdetr.size})...")
    rf_wrapper = rf_model_cls(
        pretrain_weights=config.model.rfdetr.pretrain_weights,
        resolution=int(config.model.input_size),
        num_classes=len(label_map),
        class_names=class_names,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    # Ensure detection head is resized for custom classes.
    rf_wrapper.model.reinitialize_detection_head(len(label_map))

    base_args = rf_wrapper.model.args
    data_module = RFDETRDataModule(dataset_dir=staged_dataset_dir, config=config, base_args=base_args)
    data_module.setup("fit")
    data_module.setup("test")

    criterion, postprocess = build_criterion_and_postprocessors(data_module.args)
    lightning_model = RFDETRLightningModule(
        model=rf_wrapper.model.model,
        criterion=criterion,
        postprocess=postprocess,
        config=config,
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
