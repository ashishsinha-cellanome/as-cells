#!/usr/bin/env python3

import os
import math
import datetime
import torch
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import WeightedRandomSampler, DataLoader

from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
from rfdetr import RFDETRSegLarge, RFDETRSegMedium, RFDETRSegNano, RFDETRSegSmall

from rfdetr.training import RFDETRModelModule
from rfdetr._namespace import build_namespace
from rfdetr.models.lwdetr import build_criterion_and_postprocessors
from rfdetr.training.trainer import build_trainer
from rfdetr.training.callbacks.coco_eval import COCOEvalCallback
from rfdetr.training.callbacks import RFDETREMACallback
from rfdetr.util.misc import collate_fn

from pytorch_lightning.callbacks import ModelCheckpoint

from data.motif_data_module import MotifDataModule
from utils.motif_coco_eval import MotifCocoEvalCallback
from utils.distributed_utils import setup_cluster_env, rank_zero_print
from utils.test_only_checkpoint_restore import (
    _load_ckpt,
    _select_eval_weights_source,
    _load_selected_weights,
)

torch.set_float32_matmul_precision("medium")
OmegaConf.register_new_resolver("oc.eval", eval, replace=True)

# ---------------------------------------------------------------------------
# PreBuiltRFDETRModelModule
# ---------------------------------------------------------------------------
class PreBuiltRFDETRModelModule(RFDETRModelModule):
    """
    Subclass of RFDETRModelModule that bypasses internal model building,
    allowing a pre-constructed model with custom checkpoints to be injected.
    """
    def __init__(self, model_config, train_config, inner_model, lora_cfg=None):
        pl.LightningModule.__init__(self)  # Bypass parent __init__
        self.model_config = model_config
        self.train_config = train_config
        self.strict_loading = False
        self.model = inner_model
        
        # Build matching criterion and postprocessor natively
        ns = build_namespace(model_config, train_config)
        self.criterion, self.postprocess = build_criterion_and_postprocessors(ns)
        
        self._lora_cfg = lora_cfg or {}
        if model_config.backbone_lora:
            self._apply_lora()

    def _apply_lora(self) -> None:
        """Customizable LoRA injection that overrides the upstream hardcoded method."""
        from peft import LoraConfig, get_peft_model
        lc = self._lora_cfg
        lora_config = LoraConfig(
            r=lc.get("r", 16),
            lora_alpha=lc.get("alpha", 16),
            use_dora=lc.get("use_dora", False),
            target_modules=list(lc.get("target_modules", [
                "q_proj", "v_proj", "k_proj", "qkv", "dense",
                "query", "key", "value", "cls_token", "register_tokens"
            ])),
            lora_dropout=lc.get("dropout", 0.05),
            bias="none"
        )
        self.model.backbone[0].encoder = get_peft_model(
            self.model.backbone[0].encoder, 
            lora_config
        )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return super().test_step(batch, batch_idx)

# ---------------------------------------------------------------------------
# Phase2MotifDataModule
# ---------------------------------------------------------------------------
class Phase2MotifDataModule(MotifDataModule):
    """
    Subclass of MotifDataModule that computes inverse-frequency weights
    proportional to class counts to oversample rare classes.
    """
    def __init__(self, base_path, config, base_args):
        super().__init__(base_path, config, base_args)
        # Ensure default train split maps to "train_new"
        self.train_name = getattr(config.data, "train_name", "train_new")
        self.balance_class_sampling = getattr(config.data, "balance_class_sampling", False)
        self._train_sample_weights = None

    def setup(self, stage=None):
        # Build datasets using parent setup logic
        super().setup(stage)
        
        if stage in ("fit", None) and self.balance_class_sampling:
            # 1. Single pass to count classes and collect image compositions
            total_per_class = {}
            img_compositions = []
            
            for ds_obj in self.train_datasets_objs:
                coco = ds_obj.coco
                for img_id in coco.imgs.keys():
                    anns = coco.imgToAnns.get(img_id, [])
                    img_classes = {ann["category_id"] for ann in anns}
                    
                    for cat_id in img_classes:
                        total_per_class[cat_id] = total_per_class.get(cat_id, 0) + 1
                        
                    img_compositions.append(img_classes)
            
            total_imgs_with_classes = sum(total_per_class.values())
            num_classes = len(total_per_class)
            
            if total_imgs_with_classes > 0:
                # 2. Compute true inverse frequency
                class_weights = {
                    c: total_imgs_with_classes / (count * num_classes)
                    for c, count in total_per_class.items()
                }
                
                # 3. Assign weight to each image based on its rarest class (highest weight)
                sample_weights = [
                    max((class_weights[c] for c in classes), default=1e-4)
                    for classes in img_compositions
                ]
                
                self._train_sample_weights = torch.DoubleTensor(sample_weights)

        if stage in ("test", None):
            self.train_test_datasets_objs = [
                self._make_dataset(ds, self.test_name) for ds in self.train_dataset_names
            ]

    def train_dataloader(self):
        if self._train_sample_weights is not None and len(self._train_sample_weights) > 0:
            # DDP-safe custom sampling configuration
            num_samples = len(self.concat_train)
            
            if torch.distributed.is_initialized() and torch.distributed.is_available():
                world_size = torch.distributed.get_world_size()
                num_samples = max(1, num_samples // world_size)
                
            sampler = WeightedRandomSampler(
                self._train_sample_weights,
                num_samples=num_samples,
                replacement=True
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True

        return DataLoader(
            self.concat_train,
            batch_size=self._args.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self._args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    def test_dataloader(self):
        eval_batch_size = int(getattr(self.config.data, "eval_batch_size", 1))
        dataloaders = []
        
        for ds in getattr(self, "train_test_datasets_objs", []):
            dl = DataLoader(
                ds,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=self._args.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=False,
            )
            dataloaders.append(dl)
            
        for ds in self.test_datasets_objs:
            dl = DataLoader(
                ds,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=self._args.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=False,
            )
            dataloaders.append(dl)
            
        return dataloaders


def _get_model_class(size_name: str, is_seg: bool = True):
    size_name = str(size_name).lower()
    if is_seg:
        if size_name == "small":
            return RFDETRSegSmall
        if size_name == "medium":
            return RFDETRSegMedium
        if size_name == "large":
            return RFDETRSegLarge
        if size_name == "nano":
            return RFDETRSegNano
    else:
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


@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(config: DictConfig):
    OmegaConf.set_struct(config, False)
    setup_cluster_env()
    
    hc = hydra.core.hydra_config.HydraConfig.get()
    data_config_choice = hc.runtime.choices.data
    motif_config_name = os.path.basename(data_config_choice)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    finetune_mode = config.model.rfdetr.get("finetune_mode", "full")
    if finetune_mode == "decoder":
        finetune_mode = "queries_decoder_head"
    run_name = f"phase2_{motif_config_name}_{finetune_mode}_{timestamp}"
    config.run_name = run_name

    pl.seed_everything(config.get("seed", 42), workers=True)

    label_map = {int(k): v for k, v in config.model.label_map.items()}
    class_names = [label_map[idx] for idx in sorted(label_map.keys())]
    num_classes = len(label_map)

    # 1. Base Model Initialization
    is_seg = "seg" in config.model.name.lower()
    rf_model_cls = _get_model_class(config.model.rfdetr.size, is_seg=is_seg)
    kwargs = {
        "pretrain_weights": config.model.rfdetr.get("pretrain_weights", None),
        "resolution": int(config.model.input_size),
        "num_classes": num_classes,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "group_detr": getattr(config.model.rfdetr, "group_detr", 1),
        "compile": getattr(config.model.rfdetr, "compile", False),
        "backbone_lora": False # We apply it via the model_config later
    }
    
    if hasattr(config.model.rfdetr, "patch_size"):
        kwargs["patch_size"] = int(config.model.rfdetr.patch_size)
    if hasattr(config.model.rfdetr, "num_windows"):
        kwargs["num_windows"] = int(config.model.rfdetr.num_windows)
        
    rank_zero_print(f"[Startup] Building base model with args: {kwargs}")
    rf_wrapper = rf_model_cls(**kwargs)
    
    if hasattr(rf_wrapper.model, "class_names"):
        rf_wrapper.model.class_names = class_names

    inner_model = rf_wrapper.model.model
    base_args = rf_wrapper.model.args

    # 2. Extract and Prepare Configs
    model_config = rf_wrapper.model_config
    model_config.num_classes = num_classes
    model_config.backbone_lora = finetune_mode == "lora" or config.model.rfdetr.get("backbone_lora", False)
    
    lora_cfg_dict = getattr(config.model.rfdetr, "lora", {})
    if lora_cfg_dict:
        lora_cfg = OmegaConf.to_container(lora_cfg_dict, resolve=True)
    else:
        lora_cfg = {}

    batch_size = int(config.data.batch_size)
    grad_accum_steps = int(config.trainer.get("grad_accum_steps", getattr(config.model.rfdetr, "grad_accum_steps", 1)))
    
    train_config_kwargs = dict(
        dataset_dir=str(config.data.path),
        epochs=int(config.trainer.max_epochs),
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=float(config.optimizer.optimizer.lr),
        weight_decay=float(config.optimizer.optimizer.weight_decay),
        output_dir=os.path.join(config.checkpointing.save_dir, "phase2", f"{motif_config_name}_{timestamp}"),
        use_ema=bool(config.model.rfdetr.get("use_ema", True)),
        ema_decay=float(config.model.rfdetr.get("ema_decay", 0.993)),
        ema_tau=int(config.model.rfdetr.get("ema_tau", 1000)),
        eval_max_dets=int(config.model.get("max_detections", 100)),
        early_stopping=bool(getattr(config.trainer, "early_stopping", False)),
        num_workers=int(config.data.num_workers),
        seed=config.get("seed", 42),
        accelerator=config.trainer.get("accelerator", "auto"),
        log_per_class_metrics=True,
        train_log_sync_dist=True,
        compute_val_loss=True,
        compute_test_loss=True,
        fp16_eval=False,
        progress_bar="tqdm",
        wandb=True,
        project="cell-detection-motifs",
        run=run_name,
    )
    
    if hasattr(config.optimizer.optimizer, "lr_encoder"):
        train_config_kwargs["lr_encoder"] = float(config.optimizer.optimizer.lr_encoder)
    elif "lr_encoder" in config.model.rfdetr:
        train_config_kwargs["lr_encoder"] = float(config.model.rfdetr.lr_encoder)

    train_config = rf_wrapper.get_train_config(**train_config_kwargs)

    # 3. Create Module
    module = PreBuiltRFDETRModelModule(
        model_config=model_config, 
        train_config=train_config, 
        inner_model=inner_model, 
        lora_cfg=lora_cfg
    )
    module.config = config

    # 4. Finetuning Strategies (Freezing)
    if finetune_mode in ("decoder", "queries_decoder_head"):
        rank_zero_print(f"[Startup] Freezing backbone & encoder (mode={finetune_mode}). Finetuning decoder & head only.")
        for name, param in module.model.named_parameters():
            param.requires_grad = False
            allow_prefixes = [
                "class_embed.", "bbox_embed.", "transformer.decoder.", "segmentation_head.",
                "transformer.enc_out_class_embed.", "transformer.enc_out_bbox_embed."
            ]
            if finetune_mode == "queries_decoder_head":
                allow_prefixes.extend(["refpoint_embed.", "query_feat."])
                
            if any(name.startswith(p) for p in allow_prefixes):
                param.requires_grad = True

    # 5. Data Module
    data_module = Phase2MotifDataModule(
        base_path=str(config.data.path), 
        config=config, 
        base_args=base_args
    )
    data_module.setup("fit")
    data_module.setup("test")

    # 6. Build Trainer
    # Handle debug overrides natively
    trainer_kwargs = {}
    if getattr(config, "debug", False):
        rank_zero_print("--- RUNNING IN DEBUG MODE ---")
        trainer_kwargs["limit_train_batches"] = getattr(config.trainer, "limit_train_batches", 100)
        trainer_kwargs["limit_val_batches"] = getattr(config.trainer, "limit_val_batches", 100)
        trainer_kwargs["limit_test_batches"] = getattr(config.trainer, "limit_test_batches", 100)
        train_config.epochs = 1
    else:
        if "limit_train_batches" in config.trainer:
            trainer_kwargs["limit_train_batches"] = config.trainer.limit_train_batches
        if "limit_val_batches" in config.trainer:
            trainer_kwargs["limit_val_batches"] = config.trainer.limit_val_batches
        if "limit_test_batches" in config.trainer:
            trainer_kwargs["limit_test_batches"] = config.trainer.limit_test_batches

    devices = config.trainer.get("devices", 1)
    
    if str(devices) == "auto" or str(devices) == "-1":
        num_devices = torch.cuda.device_count()
    else:
        num_devices = int(devices)

    # --- Custom WandB Logger Injection ---
    if getattr(train_config, "wandb", False):
        from pytorch_lightning.loggers import WandbLogger
        try:
            cfg_for_log = OmegaConf.to_container(config, resolve=True)
        except Exception:
            cfg_for_log = OmegaConf.to_container(config, resolve=False)
            
        custom_logger = WandbLogger(
            project="cell-detection-motifs",
            name=run_name,
            save_dir=train_config.output_dir,
            config=cfg_for_log
        )
        trainer_kwargs["logger"] = custom_logger
    # ---------------------------------------
        
    trainer = build_trainer(
        train_config=train_config, 
        model_config=model_config, 
        strategy="auto" if num_devices <= 1 else "ddp_find_unused_parameters_true",
        devices=devices,
        **trainer_kwargs
    )

    # 7. Callbacks Swapping
    trainer.callbacks = [cb for cb in trainer.callbacks if not isinstance(cb, COCOEvalCallback)]
    
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

    motif_coco_eval = MotifCocoEvalCallback(
        val_dataloader_names=val_dataloader_names, 
        test_dataloader_names=test_dataloader_names,
        val_get_coco_gt_fns=val_dataset_fns,
        test_get_coco_gt_fns=test_dataset_fns,
        label_map=label_map
    )
    trainer.callbacks.append(motif_coco_eval)

    if is_seg:
        trainer.callbacks.append(
            ModelCheckpoint(
                dirpath=os.path.join(config.checkpointing.save_dir, "phase2", f"{motif_config_name}_{timestamp}"),
                filename="best-segm-epoch{epoch:02d}",
                monitor="val/segm_mAP_50_95",
                mode="max",
                save_top_k=1,
                auto_insert_metric_name=False,
                enable_version_counter=False,
            )
        )
        if bool(config.model.rfdetr.get("use_ema", True)):
            trainer.callbacks.append(
                ModelCheckpoint(
                    dirpath=os.path.join(config.checkpointing.save_dir, "phase2", f"{motif_config_name}_{timestamp}"),
                    filename="best-ema-segm-epoch{epoch:02d}",
                    monitor="val/ema_segm_mAP_50_95",
                    mode="max",
                    save_top_k=1,
                    auto_insert_metric_name=False,
                    enable_version_counter=False,
                )
            )

    # Port manual EMA test-eval logic directly by wrapping trainer.test
    orig_test = trainer.test
    def custom_test(*args, **kwargs):
        rank_zero_print("[Test] Injecting manual EMA evaluation wrapper...")
        
        # Look for EMA callback to manually inject weights if they exist in checkpoints
        ema_cb = next((cb for cb in trainer.callbacks if isinstance(cb, RFDETREMACallback)), None)
        
        if ema_cb is not None and getattr(ema_cb, "_average_model", None) is not None:
            orig_start = ema_cb.on_test_epoch_start
            def patched_start(trainer, pl_module):
                if getattr(ema_cb, "_pending_average_state_dict", None) is not None:
                    ema_cb._average_model.load_state_dict(ema_cb._pending_average_state_dict)
                    ema_cb._pending_average_state_dict = None
                return orig_start(trainer, pl_module)
            ema_cb.on_test_epoch_start = patched_start
            
        return orig_test(*args, **kwargs)

    trainer.test = custom_test

    if getattr(config, "test_only", False):
        rank_zero_print(f"Skipping Training for Motif: {motif_config_name} (test_only mode)")
        ckpt_path = config.get("initialization", {}).get("load_from_checkpoint", None)
        if ckpt_path is None:
            raise ValueError("test_only requires initialization.load_from_checkpoint")
        
        test_only_checkpoint = _load_ckpt(ckpt_path)
        test_only_weight_source = _select_eval_weights_source(
            ckpt_path, test_only_checkpoint, config=config
        )
        
        rank_zero_print(f"Loading {test_only_weight_source.upper()} weights manually from {ckpt_path}")
        missing_keys, unexpected_keys = _load_selected_weights(
            module, test_only_checkpoint, test_only_weight_source
        )
        if missing_keys:
            rank_zero_print(f"⚠️  Missing keys during test-only load: {missing_keys[:10]} ...")
        if unexpected_keys:
            rank_zero_print(f"⚠️  Unexpected keys during test-only load: {unexpected_keys[:10]} ...")
            
        rank_zero_print(f"Starting Validation/Test eval with loaded weights...")
        trainer.test(module, datamodule=data_module)
    else:
        rank_zero_print(f"Starting Training for Motif: {motif_config_name}")
        trainer.fit(module, datamodule=data_module)
        
        rank_zero_print("Training complete. Starting Validation/Test eval...")
        trainer.test(module, datamodule=data_module)
    
    # Sync optimized weights back so predictive APIs work post-training
    rf_wrapper.model.model = module.model
    rank_zero_print("Done!")

if __name__ == '__main__':
    main()
