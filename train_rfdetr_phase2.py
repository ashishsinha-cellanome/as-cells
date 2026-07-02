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
                "q_proj", "v_proj", "k_proj", "qkv", 
                "query", "key", "value", "cls_token", "register_tokens"
            ])),
            lora_dropout=lc.get("dropout", 0.05),
            bias="none"
        )
        self.model.backbone[0].encoder = get_peft_model(
            self.model.backbone[0].encoder, 
            lora_config
        )


# ---------------------------------------------------------------------------
# Phase2MotifDataModule
# ---------------------------------------------------------------------------
class Phase2MotifDataModule(MotifDataModule):
    """
    Subclass of MotifDataModule that computes inverse-frequency weights
    proportional to class counts, oversampling rare classes (with multiplier support).
    """
    def __init__(self, base_path, config, base_args, class_sample_multiplier=None):
        super().__init__(base_path, config, base_args)
        # Ensure default train split maps to "train_new"
        self.train_name = getattr(config.data, "train_name", "train_new")
        self.class_sample_multiplier = class_sample_multiplier or {}
        self._train_sample_weights = None

    def setup(self, stage=None):
        # Build datasets using parent setup logic
        super().setup(stage)
        
        if stage in ("fit", None) and self.class_sample_multiplier:
            # 1. Count total instances per class across all training sub-datasets
            total_per_class = {}
            for ds_obj in self.train_datasets_objs:
                coco = ds_obj.coco
                anns = coco.loadAnns(coco.getAnnIds())
                for ann in anns:
                    cat_id = ann["category_id"]
                    total_per_class[cat_id] = total_per_class.get(cat_id, 0) + 1
            
            total_instances = sum(total_per_class.values())
            num_classes = len(total_per_class)
            
            if total_instances > 0:
                # 2. Compute base inverse frequency weights: w_c = Total / (N_c * C)
                base_class_weight = {
                    cat_id: total_instances / (count * num_classes)
                    for cat_id, count in total_per_class.items() if count > 0
                }
                
                # 3. Apply custom multipliers
                for cat_id, multiplier in self.class_sample_multiplier.items():
                    cat_id_int = int(cat_id)
                    if cat_id_int in base_class_weight:
                        base_class_weight[cat_id_int] *= float(multiplier)
                
                # 4. Generate per-image sampling weights using log-scale count scaling
                sample_weights = []
                for ds_obj in self.train_datasets_objs:
                    coco = ds_obj.coco
                    anns = coco.loadAnns(coco.getAnnIds())
                    
                    img_counts = {}
                    for ann in anns:
                        img_id = ann["image_id"]
                        cat_id = ann["category_id"]
                        if img_id not in img_counts:
                            img_counts[img_id] = {}
                        img_counts[img_id][cat_id] = img_counts[img_id].get(cat_id, 0) + 1
                    
                    for img in coco.imgs.values():
                        img_id = img["id"]
                        counts = img_counts.get(img_id, {})
                        w = sum(math.log(1 + count) * base_class_weight.get(cat_id, 1.0) 
                                for cat_id, count in counts.items())
                        sample_weights.append(max(w, 1e-4))
                
                self._train_sample_weights = torch.DoubleTensor(sample_weights)

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
    class_multipliers = getattr(config.data, "class_sample_multiplier", None)
    if class_multipliers:
        class_multipliers = OmegaConf.to_container(class_multipliers, resolve=True)
    else:
        # Default to 3.0 for 2 and 3 if balance_class_sampling is set, else empty
        if getattr(config.data, "balance_class_sampling", False):
            class_multipliers = {2: 3.0, 3: 3.0}
        else:
            class_multipliers = {}
            
    data_module = Phase2MotifDataModule(
        base_path=str(config.data.path), 
        config=config, 
        base_args=base_args,
        class_sample_multiplier=class_multipliers
    )
    data_module.setup("fit")
    data_module.setup("test")

    # 6. Build Trainer
    # Handle debug overrides natively
    trainer_kwargs = {}
    if getattr(config, "debug", False):
        rank_zero_print("--- RUNNING IN DEBUG MODE ---")
        trainer_kwargs["limit_train_batches"] = 2
        trainer_kwargs["limit_val_batches"] = 2
        trainer_kwargs["limit_test_batches"] = 2
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
    
    test_dataloader_names = [f"{ds}" for ds in data_module.test_dataset_names]
    test_dataset_fns = [lambda ds=ds: ds.coco for ds in data_module.test_datasets_objs]

    motif_coco_eval = MotifCocoEvalCallback(
        val_dataloader_names=val_dataloader_names, 
        test_dataloader_names=test_dataloader_names,
        val_get_coco_gt_fns=val_dataset_fns,
        test_get_coco_gt_fns=test_dataset_fns,
        label_map=label_map
    )
    trainer.callbacks.append(motif_coco_eval)

    if hasattr(config.model, "segmentation_head") and config.model.segmentation_head:
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

    rank_zero_print(f"Starting Training for Motif: {motif_config_name}")
    trainer.fit(module, datamodule=data_module)
    
    rank_zero_print("Training complete. Starting Validation/Test eval...")
    trainer.test(module, datamodule=data_module)
    
    # Sync optimized weights back so predictive APIs work post-training
    rf_wrapper.model.model = module.model
    rank_zero_print("Done!")

if __name__ == '__main__':
    main()
