#!/usr/bin/env python3

import os
import sys
import datetime
import torch
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

# Setup distributed env if needed
from utils.distributed_utils import setup_cluster_env, get_rank, rank_zero_print
setup_cluster_env()

# RF-DETR specific
from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
from rfdetr.models.lwdetr import build_criterion_and_postprocessors

# Custom Modules
from models.rf_detr_lightning_module import RFDETRLightningModule
from data.motif_data_module import MotifDataModule
from utils.motif_coco_eval import MotifCocoEvalCallback
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary
from pytorch_lightning.loggers import WandbLogger

torch.set_float32_matmul_precision("medium")
OmegaConf.register_new_resolver("oc.eval", eval, replace=True)

def _get_model_class(size_name: str):
    size_name = str(size_name).lower()
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

    # 1. Initialize Dataset Module
    base_path = config.data.path
    data_module = MotifDataModule(base_path=base_path, config=config)
    data_module.setup("fit")
    data_module.setup("test")

    label_map = {int(k): v for k, v in config.model.label_map.items()}
    class_names = [label_map[idx] for idx in sorted(label_map.keys())]
    num_classes = len(label_map)

    # 2. Initialize Model
    rf_model_cls = _get_model_class(config.model.rfdetr.size)
    kwargs = {
        "pretrain_weights": config.model.rfdetr.get("pretrain_weights", "rf-detr-large-2026.pth"),
        "resolution": int(config.model.input_size),
        "num_classes": num_classes,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "group_detr": getattr(config.model.rfdetr, "group_detr", 1)
    }
    
    rank_zero_print(f"[Startup] Building model with args: {kwargs}")
    rf_wrapper = rf_model_cls(**kwargs)
    
    if hasattr(rf_wrapper.model, "class_names"):
        rf_wrapper.model.class_names = class_names
    rf_wrapper.model.reinitialize_detection_head(num_classes)

    inner_model = rf_wrapper.model.model

    # 3. Finetuning Strategies
    if finetune_mode == "decoder":
        rank_zero_print("[Startup] Freezing backbone & encoder. Finetuning decoder & head only.")
        for name, param in inner_model.named_parameters():
            param.requires_grad = False
            allow_prefixes = [
                "refpoint_embed.", "query_feat.", "class_embed.", "bbox_embed.",
                "transformer.enc_out_class_embed.", "transformer.enc_out_bbox_embed.",
                "transformer.decoder."
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
                target_modules=["out_proj", "value_proj", "output_proj"], # Supported linear layers in attention
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

    # 5. Lightning Module
    criterion, postprocess = build_criterion_and_postprocessors(data_module.args)
    # model_to_coco is 1:1 for simplicity
    model_to_coco = {i: i for i in range(num_classes)}
    
    lightning_model = RFDETRLightningModule(
        model=inner_model,
        criterion=criterion,
        postprocess=postprocess,
        config=config,
        model_to_coco=model_to_coco,
        val_coco_gt=None, # Replaced by custom motif callback
        test_coco_gt=None,
        val_image_root="",
        test_image_root=""
    )

    # 6. WandB Logger
    logger = None
    if config.logging.wandb.enabled and not getattr(config, "debug", False):
        tags = list(config.logging.wandb.tags)
        tags.extend([f"motif:{motif_name}", f"mode:{finetune_mode}"])
        cfg_for_log = OmegaConf.to_container(config, resolve=True)
        
        logger = WandbLogger(
            project="cell-detection-motifs",
            name=run_name,
            tags=tags,
            config=cfg_for_log
        )

    # 7. Callbacks
    callbacks = [
        ModelCheckpoint(monitor="val_map_50", mode="max", save_top_k=1, auto_insert_metric_name=False),
        LearningRateMonitor(logging_interval="step"),
        ModelSummary(max_depth=3)
    ]
    
    # Custom COCO Eval for multiple dataloaders
    test_datasets = data_module.test_dataset_names
    val_dataset_fns = [lambda ds=ds: ds.coco for ds in data_module.val_datasets_objs]
    
    callbacks.append(MotifCocoEvalCallback(dataloader_names=test_datasets, get_coco_gt_fns=val_dataset_fns))

    # 8. Trainer
    trainer = pl.Trainer(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        strategy="auto" if config.trainer.devices == 1 else "ddp_find_unused_parameters_true",
        max_epochs=config.trainer.max_epochs,
        logger=logger,
        callbacks=callbacks,
        **trainer_kwargs
    )

    rank_zero_print(f"Starting Training for Motif: {motif_name}")
    trainer.fit(lightning_model, datamodule=data_module)
    
    rank_zero_print("Training complete. Starting Validation/Test eval...")
    trainer.test(lightning_model, datamodule=data_module)

if __name__ == '__main__':
    main()