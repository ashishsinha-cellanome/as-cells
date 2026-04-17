#!/usr/bin/env python3
import os
from typing import Any, Dict, Literal, Optional
import torch
import pytorch_lightning as pl
from pycocotools.coco import COCO
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver(
    "extract_name", lambda path: path.split("/")[-1], replace=True
)
OmegaConf.register_new_resolver("oc.eval", eval, replace=True)
import hydra

# Import your setup functions for callbacks, profilers, wandb, etc.
from utils.distributed_utils import setup_cluster_env, rank_zero_print
from data.yolo_data_module import YOLOv5DataModule
from models.yolo_lightning_module import YOLOv5LightningModule

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from pytorch_lightning.loggers import WandbLogger

from utils.train_utils import BackupToNASCallback
from utils.ema import EMACallback
from utils.test_only_checkpoint_restore import (
    _load_ckpt,
    _load_selected_weights,
    _merge_test_only_config_from_ckpt,
    _resolve_ckpt_path,
    _select_eval_weights_source,
)


# setup_cluster_env()  # Moved inside main()
torch.set_float32_matmul_precision("medium")


def setup_logger(config: DictConfig):
    wandb_cfg = config.logging.wandb
    if not wandb_cfg.enabled:
        return None
    cfg_for_log = OmegaConf.to_container(config, resolve=True)

    return WandbLogger(
        project=wandb_cfg.project,
        name=config.run_name,
        tags=list(wandb_cfg.tags),
        notes=wandb_cfg.notes,
        # group=wandb_cfg.get("group"),
        config=cfg_for_log,
        save_dir=os.getcwd(),
        log_model=False,
        reinit="finish_previous",
    )


def setup_callbacks(config: DictConfig):
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
        ModelSummary(max_depth=2),
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


@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(config: DictConfig):
    test_only_checkpoint: Optional[Dict[str, Any]] = None
    test_only_weight_source: Optional[Literal["ema", "regular"]] = None
    early_ckpt_path: Optional[str] = None

    pl.seed_everything(config.seed, workers=True)
    setup_cluster_env()

    OmegaConf.set_struct(config, False)
    base_save_dir = hydra.utils.to_absolute_path(config.checkpointing.save_dir)
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

        # Inject YOLO repo into sys.path so torch.load can deserialize native YOLO model classes
        import sys

        original_path = sys.path.copy()
        original_modules = {}
        try:
            if (
                hasattr(config, "model")
                and hasattr(config.model, "yolov5")
                and config.model.yolov5 is not None
            ):
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

    ckpt_path = _resolve_ckpt_path(config, run_save_dir=run_save_dir)
    OmegaConf.set_struct(config, True)

    data_path = hydra.utils.to_absolute_path(config.data.path)
    val_json = os.path.join(data_path, f"{config.data.val_name}_annotations.json")
    test_json = os.path.join(data_path, f"{config.data.test_name}_annotations.json")

    val_coco_gt = COCO(val_json)
    test_coco_gt = COCO(test_json)

    if "info" not in val_coco_gt.dataset:
        val_coco_gt.dataset["info"] = {}
    if "info" not in test_coco_gt.dataset:
        test_coco_gt.dataset["info"] = {}

    # Note: Map YOLO sequential IDs directly to your COCO Category IDs
    # Example: If YOLO class 0 is 'cell', and COCO uses ID 0 for 'cell', this map is {0:0, 1:1, etc.}
    # model_to_coco = {int(k): int(k) for k in config.model.label_map.keys()}
    # Dynamically build YOLO ID -> COCO ID mapping based on class names
    name_to_yolo_id = {v: int(k) for k, v in config.model.label_map.items()}
    model_to_coco = {}

    for coco_cat_id, cat_info in val_coco_gt.cats.items():
        cat_name = cat_info["name"]
        if cat_name in name_to_yolo_id:
            yolo_id = name_to_yolo_id[cat_name]
            model_to_coco[yolo_id] = coco_cat_id

    # Fallback just in case some classes aren't in the val set
    for yolo_id in config.model.label_map.keys():
        if int(yolo_id) not in model_to_coco:
            model_to_coco[int(yolo_id)] = int(yolo_id)

    print(f"[INFO] YOLO to COCO ID Mapping: {model_to_coco}")

    data_module = YOLOv5DataModule(config)
    model = YOLOv5LightningModule(
        config=config,
        yolo_repo_path=config.model.yolov5.repo_path,
        model_to_coco=model_to_coco,
        val_coco_gt=val_coco_gt,
        test_coco_gt=test_coco_gt,
    )

    logger = setup_logger(config)
    # logger.watch(model, log='gradients')

    callbacks = setup_callbacks(config)

    import inspect

    trainer_kwargs = dict(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
        strategy=config.trainer.strategy,
        max_epochs=config.trainer.max_epochs,
        gradient_clip_val=config.trainer.max_grad_norm,
        callbacks=callbacks,
        logger=logger,
        limit_train_batches=config.data.limit_train_batches,
        limit_val_batches=config.data.limit_val_batches,
        limit_test_batches=config.data.limit_test_batches,
        overfit_batches=config.data.num_overfit_samples,
    )
    trainer_sig = inspect.signature(pl.Trainer)
    if "use_distributed_sampler" in trainer_sig.parameters:
        trainer_kwargs["use_distributed_sampler"] = False
    if "replace_sampler_ddp" in trainer_sig.parameters:
        trainer_kwargs["replace_sampler_ddp"] = False

    trainer = pl.Trainer(**trainer_kwargs)
    # Extra guard: prevent Lightning from injecting a distributed sampler.
    if hasattr(trainer, "_data_connector"):
        if hasattr(trainer._data_connector, "_use_distributed_sampler"):
            trainer._data_connector._use_distributed_sampler = False
        if hasattr(trainer._data_connector, "_replace_sampler_ddp"):
            trainer._data_connector._replace_sampler_ddp = False

    if config.get("test_only", False):
        if not ckpt_path:
            raise ValueError("Must provide load_from_checkpoint for test-only.")
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

        # Manual weight loading above; do not ask Lightning to restore checkpoint again.
        trainer.test(model, datamodule=data_module)
    else:
        trainer.fit(
            model, datamodule=data_module, ckpt_path=ckpt_path, weights_only=False
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        trainer.test(
            model, datamodule=data_module, ckpt_path="best", weights_only=False
        )


if __name__ == "__main__":
    main()
