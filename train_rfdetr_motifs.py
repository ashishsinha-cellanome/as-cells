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

# Patch convert_coco_poly_to_mask to fix Pycocotools + Numpy 2.0 list bug AND lazily uncrop test RLEs
import rfdetr.datasets.coco as rfdetr_coco

_orig_convert_coco_call = rfdetr_coco.ConvertCoco.__call__

def patched_convert_coco_call(self, image, target):
    w, h = image.size
    anno = target["annotations"]
    anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]
    
    import pycocotools.mask as mask_util
    for obj in anno:
        seg = obj.get("segmentation")
        if isinstance(seg, dict) and list(seg.get('size', [])) != [h, w]:
            bbox = obj.get("bbox", [0, 0, 0, 0])
            x, y, bw, bh = [int(v) for v in bbox]
            
            if isinstance(seg['counts'], str):
                seg['counts'] = seg['counts'].encode('utf-8')
                
            try:
                cropped_mask = mask_util.decode(seg)
                full_mask = np.zeros((h, w), dtype=np.uint8)
                
                x_start = max(0, x)
                y_start = max(0, y)
                x_end = min(x + bw, w)
                y_end = min(y + bh, h)
                
                if x_start < x_end and y_start < y_end:
                    cx_start = 0 if x >= 0 else -x
                    cy_start = 0 if y >= 0 else -y
                    
                    true_cy_end = min(cy_start + (y_end - y_start), cropped_mask.shape[0])
                    true_cx_end = min(cx_start + (x_end - x_start), cropped_mask.shape[1])
                    
                    y_end = y_start + (true_cy_end - cy_start)
                    x_end = x_start + (true_cx_end - cx_start)
                    
                    full_mask[y_start:y_end, x_start:x_end] = cropped_mask[cy_start:true_cy_end, cx_start:true_cx_end]
                    
                full_mask_f = np.asfortranarray(full_mask)
                full_rle = mask_util.encode(full_mask_f)
                full_rle['counts'] = full_rle['counts'].decode('utf-8')
                
                obj['segmentation'] = full_rle
            except Exception as e:
                print(f"Error lazy decoding mask: {e}")
                
    return _orig_convert_coco_call(self, image, target)

rfdetr_coco.ConvertCoco.__call__ = patched_convert_coco_call

def patched_convert_coco_poly_to_mask(segmentations, height, width):
    import pycocotools.mask as coco_mask
    masks = []
    for polygons in segmentations:
        if polygons is None or len(polygons) == 0:
            masks.append(torch.zeros((height, width), dtype=torch.uint8))
            continue
        try:
            if isinstance(polygons, dict):
                # Ensure string counts are bytes
                if isinstance(polygons['counts'], str):
                    polygons['counts'] = polygons['counts'].encode('utf-8')
                mask = coco_mask.decode(polygons)
            else:
                rles = coco_mask.frPyObjects(polygons, height, width)
                mask = coco_mask.decode(rles)
            if mask.ndim < 3: mask = mask[..., None]
            mask = torch.as_tensor(mask, dtype=torch.uint8).any(dim=2)
            masks.append(mask)
        except Exception:
            masks.append(torch.zeros((height, width), dtype=torch.uint8))
    if len(masks) == 0: return torch.zeros((0, height, width), dtype=torch.uint8)
    return torch.stack(masks, dim=0)

rfdetr_coco.convert_coco_poly_to_mask = patched_convert_coco_poly_to_mask

# Setup distributed env if needed
from utils.distributed_utils import setup_cluster_env, get_rank, rank_zero_print
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
    val_dataloader_names = ["train_ds/merged"] + \
                           [f"test_ds/{ds}" for ds in data_module.test_dataset_names]
    val_dataset_fns = [lambda ds=ds: ds.coco for ds in (data_module.val_train_datasets_objs + data_module.val_test_datasets_objs)]
    
    test_dataloader_names = [f"{ds}" for ds in data_module.test_dataset_names]
    test_dataset_fns = [lambda ds=ds: ds.coco for ds in data_module.test_datasets_objs]
    
    callbacks.append(MotifCocoEvalCallback(
        val_dataloader_names=val_dataloader_names, 
        test_dataloader_names=test_dataloader_names,
        val_get_coco_gt_fns=val_dataset_fns,
        test_get_coco_gt_fns=test_dataset_fns,
        label_map=label_map
    ))

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