#!/usr/bin/env python3

import os
import sys
import datetime
import torch
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import numpy as np

# Setup distributed env if needed
from utils.distributed_utils import setup_cluster_env, get_rank, rank_zero_print
from utils.test_only_checkpoint_restore import (
    _load_ckpt,
    _select_eval_weights_source,
    _load_selected_weights,
)
setup_cluster_env()

# RF-DETR specific
from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
from rfdetr import RFDETRSegLarge, RFDETRSegMedium, RFDETRSegNano, RFDETRSegSmall
from rfdetr.models.lwdetr import build_criterion_and_postprocessors

# Custom Modules
from models.rf_detr_lightning_module import RFDETRLightningModule
from data.motif_data_module import MotifDataModule
from utils.motif_coco_eval import MotifCocoEvalCallback
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary
from pytorch_lightning.loggers import WandbLogger

torch.set_float32_matmul_precision("medium")
OmegaConf.register_new_resolver("oc.eval", eval, replace=True)

def _get_model_class(size_name: str, is_seg: bool = True):
    size_name = str(size_name).lower()
    if is_seg:
        if size_name == "small": return RFDETRSegSmall
        if size_name == "medium": return RFDETRSegMedium
        if size_name == "large": return RFDETRSegLarge
        if size_name == "nano": return RFDETRSegNano
    else:
        if size_name == "base": return RFDETRBase
        if size_name == "small": return RFDETRSmall
        if size_name == "medium": return RFDETRMedium
        if size_name == "large": return RFDETRLarge
        if size_name == "nano": return RFDETRNano
    raise ValueError(f"Unsupported RF-DETR size: {size_name}")

@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(config: DictConfig):
    OmegaConf.set_struct(config, False)
    
    # Run name and WandB tags
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    motif_name = getattr(config.data, "split_motif", "custom_split")
    finetune_mode = config.model.rfdetr.get("finetune_mode", "full")
    run_name = f"rfdetr_{motif_name}_{finetune_mode}_{timestamp}"
    config.run_name = run_name

    pl.seed_everything(config.seed, workers=True)

    label_map = {int(k): v for k, v in config.model.label_map.items()}
    class_names = [label_map[idx] for idx in sorted(label_map.keys())]
    num_classes = len(label_map)

    # 1. Initialize Model FIRST to get base_args
    is_seg = "seg" in config.model.name.lower()
    rf_model_cls = _get_model_class(config.model.rfdetr.size, is_seg=is_seg)
    kwargs = {
        "pretrain_weights": config.model.rfdetr.get("pretrain_weights", "rf-detr-large-2026.pth"),
        "resolution": int(config.model.input_size),
        "num_classes": num_classes,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "group_detr": getattr(config.model.rfdetr, "group_detr", 1),
        "compile": False
    }
    
    rank_zero_print(f"[Startup] Building model with args: {kwargs}")
    rf_wrapper = rf_model_cls(**kwargs)
    
    if hasattr(rf_wrapper.model, "class_names"):
        rf_wrapper.model.class_names = class_names
    rf_wrapper.model.reinitialize_detection_head(num_classes)

    inner_model = rf_wrapper.model.model
    base_args = rf_wrapper.model.args

    # 2. Initialize Dataset Module
    base_path = config.data.path
    if "ashish.sinha/cellanome/TRAINING_DATA" in base_path:
        base_path = "/mnt/direct-attached/PHASE2"
    data_module = MotifDataModule(base_path=base_path, config=config, base_args=base_args)
    data_module.setup("fit")
    data_module.setup("test")

    # 3. Finetuning Strategies
    if finetune_mode in ["decoder", "queries_decoder_head"]:
        rank_zero_print(f"[Startup] Freezing backbone & encoder (mode={finetune_mode}). Finetuning decoder & head only.")
        for name, param in inner_model.named_parameters():
            param.requires_grad = False
            allow_prefixes = [
                "refpoint_embed.", "query_feat.", "class_embed.", "bbox_embed.",
                "transformer.enc_out_class_embed.", "transformer.enc_out_bbox_embed.",
                "transformer.decoder.", "segmentation_head."
            ]
            if any(name.startswith(p) for p in allow_prefixes):
                param.requires_grad = True
                
    elif finetune_mode == "lora":
        rank_zero_print("[Startup] Applying LoRA to Transformer (Attention Projections)...")
        try:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["out_proj", "value_proj", "output_proj", "q_proj", "k_proj", "query", "key"], # Added q/k targets
                lora_dropout=0.05,
                bias="none"
            )
            inner_model.transformer = get_peft_model(inner_model.transformer, lora_config)
            # Ensure heads are trainable
            for name, param in inner_model.named_parameters():
                if "class_embed" in name or "bbox_embed" in name or "refpoint_embed" in name or "query_feat" in name:
                    param.requires_grad = True
        except ImportError:
            rank_zero_print("[ERROR] peft library not installed. Falling back to 'full' finetune mode.")
            finetune_mode = "full"
            
    trainable = sum(p.numel() for p in inner_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in inner_model.parameters())
    rank_zero_print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # 4. Debug Mode overrides
    trainer_kwargs = {}
    if getattr(config, "debug", False):
        rank_zero_print("--- RUNNING IN DEBUG MODE ---")
        trainer_kwargs["limit_train_batches"] = 10
        trainer_kwargs["limit_val_batches"] = 10
        trainer_kwargs["limit_test_batches"] = 10
        config.trainer.max_epochs = 1
    else:
        if "limit_train_batches" in config.trainer: trainer_kwargs["limit_train_batches"] = config.trainer.limit_train_batches
        if "limit_val_batches" in config.trainer: trainer_kwargs["limit_val_batches"] = config.trainer.limit_val_batches
        if "limit_test_batches" in config.trainer: trainer_kwargs["limit_test_batches"] = config.trainer.limit_test_batches

    # 5. Lightning Module
    criterion, postprocess = build_criterion_and_postprocessors(data_module._args)
    # model_to_coco is 1:1 for simplicity
    model_to_coco = {i: i for i in range(num_classes)}
    
    lightning_model = RFDETRLightningModule(
        model=inner_model,
        criterion=criterion,
        postprocess=postprocess,
        config=config,
        model_to_coco=model_to_coco,
        val_coco_gt=data_module.val_train_datasets_objs[0].coco, # Replaced by custom motif callback
        test_coco_gt=None,
        val_image_root=str(data_module.base_path),
        test_image_root=""
    )

    # 6. WandB Logger
    logger = None
    if config.logging.wandb.enabled and not getattr(config, "debug", False):
        tags = list(config.logging.wandb.tags)
        tags.extend([f"motif:{motif_name}", f"mode:{finetune_mode}"])
        try:
            cfg_for_log = OmegaConf.to_container(config, resolve=True)
        except Exception:
            cfg_for_log = OmegaConf.to_container(config, resolve=False)
        
        logger = WandbLogger(
            project="cell-detection-motifs",
            name=run_name,
            tags=tags,
            config=cfg_for_log
        )

    # 7. Callbacks
    # Determine the primary validation metric for checkpointing and early stopping
    primary_train_ds = data_module.train_dataset_names[0]
    
    # Check if EMA is enabled
    use_ema = getattr(config.model, "ema", None) and getattr(config.model.ema, "enabled", False)
    ema_suffix = "_ema" if use_ema else ""
    
    # We use the detailed segm map (mAP@0.5-0.95) of the merged validation dataset
    monitor_metric_segm = f"val/train_ds/merged/detailed_segm_map{ema_suffix}"
    monitor_metric_bbox = f"val/train_ds/merged/detailed_bbox_map{ema_suffix}"
    
    ckpt_dir = os.path.join(config.checkpointing.save_dir, config.run_name, "ckpts") if hasattr(config, "checkpointing") else "output/ckpts"
    
    from pytorch_lightning.callbacks import EarlyStopping
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="motif-epoch{epoch:02d}-" + monitor_metric_segm.replace("/", "_") + "={" + monitor_metric_segm + ":.4f}",
            monitor=monitor_metric_segm, 
            mode="max", 
            save_top_k=1, 
            auto_insert_metric_name=False
        ),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="motif-epoch{epoch:02d}-" + monitor_metric_bbox.replace("/", "_") + "={" + monitor_metric_bbox + ":.4f}",
            monitor=monitor_metric_bbox, 
            mode="max", 
            save_top_k=1, 
            auto_insert_metric_name=False
        ),
        EarlyStopping(
            monitor=monitor_metric_segm,
            patience=10,
            mode="max",
            verbose=True
        ),
        LearningRateMonitor(logging_interval="step"),
        ModelSummary(max_depth=3)
    ]
    
    if use_ema:
        from utils.ema import EMACallback
        warmup_steps = config.model.ema.get("warmup_steps", 0)
        tau = config.model.ema.get("tau", 2000)
        decay = config.model.ema.get("decay", 0.993)
        rank_zero_print(f"💡 EMA enabled: Adding EMACallback with decay={decay}, tau={tau}, warmup_steps={warmup_steps}")
        callbacks.append(EMACallback(decay=decay, tau=tau, warmup_steps=warmup_steps))
    
    # Custom COCO Eval for multiple dataloaders
    val_dataloader_names = ["train_ds/merged"]
    val_dataset_fns = [lambda ds=ds: ds.coco for ds in data_module.val_train_datasets_objs]
    
    test_dataloader_names = (
        [f"train_ds/{ds}/test" for ds in data_module.train_dataset_names] + 
        [f"test_ds/{ds}/test" for ds in data_module.test_dataset_names]
    )
    test_dataset_fns = (
        [lambda ds=ds: ds.coco for ds in getattr(data_module, "train_test_datasets_objs", [])] + 
        [lambda ds=ds: ds.coco for ds in data_module.test_datasets_objs]
    )
    
    callbacks.append(MotifCocoEvalCallback(
        val_dataloader_names=val_dataloader_names, 
        test_dataloader_names=test_dataloader_names,
        val_get_coco_gt_fns=val_dataset_fns,
        test_get_coco_gt_fns=test_dataset_fns,
        label_map=label_map
    ))

    # 8. Trainer
    try:
        num_devices = len(config.trainer.devices)
    except TypeError:
        num_devices = int(config.trainer.devices)
        
    trainer = pl.Trainer(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        strategy="auto" if num_devices == 1 else "ddp_find_unused_parameters_true",
        max_epochs=config.trainer.max_epochs,
        logger=logger,
        callbacks=callbacks,
        **trainer_kwargs
    )

    if getattr(config, "test_only", False):
        rank_zero_print(f"Skipping Training for Motif: {motif_name} (test_only mode)")
        ckpt_path = config.get("initialization", {}).get("load_from_checkpoint", None)
        if ckpt_path is None:
            raise ValueError("test_only requires initialization.load_from_checkpoint")
        
        test_only_checkpoint = _load_ckpt(ckpt_path)
        test_only_weight_source = _select_eval_weights_source(
            ckpt_path, test_only_checkpoint, config=config
        )
        
        rank_zero_print(f"Loading {test_only_weight_source.upper()} weights manually from {ckpt_path}")
        missing_keys, unexpected_keys = _load_selected_weights(
            lightning_model, test_only_checkpoint, test_only_weight_source
        )
        if missing_keys:
            rank_zero_print(f"⚠️  Missing keys during test-only load: {missing_keys[:10]} ...")
        if unexpected_keys:
            rank_zero_print(f"⚠️  Unexpected keys during test-only load: {unexpected_keys[:10]} ...")
            
        rank_zero_print(f"Starting Validation/Test eval with loaded weights...")
        
        orig_test = trainer.test
        def custom_test(*args, **kwargs):
            rank_zero_print("[Test] Injecting manual EMA evaluation wrapper...")
            from utils.ema import EMACallback
            ema_cb = next((cb for cb in trainer.callbacks if isinstance(cb, EMACallback)), None)
            
            if ema_cb is not None and getattr(ema_cb, "ema_model", None) is not None:
                orig_start = getattr(ema_cb, 'on_test_epoch_start', None)
                def patched_start(trainer, pl_module):
                    if hasattr(ema_cb.ema_model, "module"):
                        pl_module.model.load_state_dict(ema_cb.ema_model.module.state_dict())
                    if orig_start:
                        return orig_start(trainer, pl_module)
                ema_cb.on_test_epoch_start = patched_start
                
            return orig_test(*args, **kwargs)

        trainer.test = custom_test
        trainer.test(lightning_model, datamodule=data_module)
    else:
        rank_zero_print(f"Starting Training for Motif: {motif_name}")
        trainer.fit(lightning_model, datamodule=data_module)
        
        rank_zero_print("Training complete. Starting Validation/Test eval...")
        trainer.test(lightning_model, datamodule=data_module)

if __name__ == '__main__':
    main()