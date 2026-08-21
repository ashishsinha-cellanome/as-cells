#!/usr/bin/env python3

import os
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
from rfdetr.utilities import collate_fn

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
    def __init__(self, model_config, train_config, inner_model, lora_cfg=None, delay_lora=False):
        pl.LightningModule.__init__(self)  # Bypass parent __init__
        self.model_config = model_config
        self.train_config = train_config
        self.strict_loading = False
        self.model = inner_model
        
        # Build matching criterion and postprocessor natively
        ns = build_namespace(model_config, train_config)
        self.criterion, self.postprocess = build_criterion_and_postprocessors(ns)
        
        self._lora_cfg = lora_cfg or {}
        if model_config.backbone_lora and not delay_lora:
            self._apply_lora()

    def _apply_lora(self) -> None:
        """Customizable LoRA injection overriding the upstream hardcoded method."""
        from peft import LoraConfig, get_peft_model
        lc = self._lora_cfg
        
        # Exact regex matching leaf nodes for segmentation head, decoder object queries, and transformer attention/dense layers.
        # We explicitly exclude the backbone to keep the LoRA footprint low and focused on the decoder/heads.
        target_modules = lc.get("target_modules", r".*(pwconv1|spatial_features_proj|query_features_proj|refpoint_embed|query_feat|enc_out_bbox_embed\.\d+\.layers\.\d+|enc_out_class_embed\.\d+|(?<!enc_out_)class_embed|bbox_embed\.layers\.\d+|segmentation_head.*pwconv1|qkv|query|key|value|dense|q_proj|k_proj|v_proj|out_proj|output_proj|value_proj|sampling_offsets|attention_weights)$")
        exclude_modules = lc.get("exclude_modules", r".*(dwconv|norm|bn|act|relu|gelu|backbone).*")
        
        r_val = lc.get("r", 32)
        lora_config = LoraConfig(
            r=r_val,
            lora_alpha=lc.get("alpha", r_val * 2),
            use_dora=lc.get("use_dora", False),
            target_modules=target_modules,
            exclude_modules=exclude_modules,
            lora_dropout=lc.get("dropout", 0.05),
            bias="none"
        )
        
        # We wrap the ENTIRE model so PEFT can find the decoder/segmentation modules
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

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
        
        # --- NEW: Subsample crops to prevent catastrophic forgetting ---
        target_data_frac = getattr(self.config.data, "target_data_frac", None)
        # We also check for legacy 'lora_frac' for backward compatibility during transitions
        if target_data_frac is None:
            target_data_frac = getattr(self.config.data, "lora_frac", None)
            
        if stage in ("fit", None) and target_data_frac is not None:
            import random
            import math
            from collections import defaultdict
            from utils.distributed_utils import rank_zero_print
            
            # Subsampling must be fully deterministic without polluting global RNG
            target_datasets = list(getattr(self.config.data, "target_datasets", []))
            anchor_base_images = int(getattr(self.config.data, "anchor_base_images", getattr(self.config.data, "anchor_samples_per_dataset", 4)))
            
            if not target_datasets:
                rank_zero_print(f"[WARNING] No target_datasets specified! Applying target_data_frac ({target_data_frac}) to ALL datasets.")
            
            for ds_name, ds_obj in zip(self.train_dataset_names, self.train_datasets_objs):
                dataset_name = ds_name

                # Group crops by their original base 4k image filename (e.g. splitting on '_crp_')
                base_to_ids = defaultdict(list)
                # Sort keys to ensure absolute DDP determinism before grouping
                for img_id in sorted(list(ds_obj.coco.imgs.keys())):
                    file_name = ds_obj.coco.imgs[img_id]["file_name"]
                    # Usually formatted as `original_name_crp_N.jpg`
                    base_name = file_name.rsplit('_crp_', 1)[0]
                    base_to_ids[base_name].append(img_id)
                
                base_image_names = sorted(list(base_to_ids.keys()))
                total_base_imgs = len(base_image_names)
                
                if total_base_imgs == 0:
                    continue
                    
                # Isolate randomness per-dataset using a deterministic string seed 
                rng = random.Random(f"{self.config.get('seed', 42)}_{dataset_name}")
                
                # Use strict membership or fallback to matching if lists are identical
                if not target_datasets or dataset_name in target_datasets:
                    # Target dataset: apply fraction to BASE images to ensure diversity
                    keep_base_count = max(1, math.floor(total_base_imgs * target_data_frac))
                    sampled_bases = set(rng.sample(base_image_names, keep_base_count))
                    
                    # To minimize training images further while maintaining diversity, 
                    # sample only a few crops (e.g., 4) from each selected base image.
                    crops_per_base = getattr(self.config.data, "target_crops_per_base", 4)
                    
                    sampled_ids = set()
                    for base_name in sampled_bases:
                        available_crops = base_to_ids[base_name]
                        keep_crops = min(len(available_crops), crops_per_base)
                        sampled_ids.update(rng.sample(available_crops, keep_crops))
                        
                    rank_zero_print(f"[Data Subsample] Target Domain {dataset_name}: keeping {keep_base_count}/{total_base_imgs} diverse base images (frac={target_data_frac}), sampling {crops_per_base} crops per base -> {len(sampled_ids)} total crops.")
                else:
                    # Anchor dataset: apply replay buffer constraint to BASE images
                    keep_base_count = min(total_base_imgs, anchor_base_images)
                    sampled_bases = set(rng.sample(base_image_names, keep_base_count))
                    
                    crops_per_base = getattr(self.config.data, "anchor_crops_per_base", 4)
                    
                    sampled_ids = set()
                    for base_name in sampled_bases:
                        available_crops = base_to_ids[base_name]
                        keep_crops = min(len(available_crops), crops_per_base)
                        sampled_ids.update(rng.sample(available_crops, keep_crops))
                        
                    rank_zero_print(f"[Data Subsample] Anchor Domain {dataset_name}: keeping {keep_base_count}/{total_base_imgs} base images, {crops_per_base} crops per base -> {len(sampled_ids)} total crops (replay buffer).")
                    
                # Filter the internal coco dictionary inplace
                ds_obj.coco.imgs = {k: v for k, v in ds_obj.coco.imgs.items() if k in sampled_ids}
                if hasattr(ds_obj, 'ids'):
                    # Retain exact original list order to prevent DDP hash randomization crashes
                    ds_obj.ids = [i for i in ds_obj.ids if i in sampled_ids]
                
                # Filter annotations map
                ds_obj.coco.imgToAnns = {k: ds_obj.coco.imgToAnns.get(k, []) for k in sampled_ids}
                
            # DANGER: `super().setup(stage)` created `self.concat_train` (a ConcatDataset) *before* we shrank `ds_obj.ids`.
            # We MUST recreate the ConcatDataset here, otherwise its internal `cumulative_sizes` cache will cause IndexErrors!
            from torch.utils.data import ConcatDataset
            self.concat_train = ConcatDataset(self.train_datasets_objs)
        # ---------------------------------------------------------------
        
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
            if torch.distributed.is_initialized() and torch.distributed.is_available():
                sampler = torch.utils.data.distributed.DistributedSampler(
                    self.concat_train, shuffle=True, drop_last=True
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
        eval_batch_size = int(getattr(self.config.data, "eval_batch_size", 64))
        eval_num_workers = min(self._args.num_workers, 8)
        dataloaders = []
        
        for ds in getattr(self, "train_test_datasets_objs", []):
            sampler = None
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=False)
                
            dl = DataLoader(
                ds,
                batch_size=eval_batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=eval_num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=False,
            )
            dataloaders.append(dl)
            
        for ds in self.test_datasets_objs:
            sampler = None
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=False)
                
            dl = DataLoader(
                ds,
                batch_size=eval_batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=eval_num_workers,
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
    
    # Append target dataset names to make checkpoint and run names more descriptive
    # and prevent information loss when finetuning on multiple datasets
    target_ds = getattr(config.data, "target_datasets", [])
    if isinstance(target_ds, str):
        target_ds = [target_ds]
    else:
        target_ds = list(target_ds)
    if len(target_ds) > 0:
        motif_config_name = f"{motif_config_name}_" + "_".join(target_ds)
    
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
        "resolution": int(config.model.input_size),
        "num_classes": num_classes,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "group_detr": getattr(config.model.rfdetr, "group_detr", 1),
        "compile": getattr(config.model.rfdetr, "compile", False),
        "backbone_lora": False # We apply it via the model_config later
    }
    
    pretrain_weights = config.model.rfdetr.get("pretrain_weights", None)
    if pretrain_weights is not None:
        kwargs["pretrain_weights"] = pretrain_weights
        
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
        output_dir=os.path.join(config.checkpointing.save_dir, "phase2", f"{motif_config_name}_{finetune_mode}_{timestamp}"),
        use_ema=bool(config.model.rfdetr.get("use_ema", True)),
        ema_decay=float(config.model.rfdetr.get("ema_decay", 0.993)),
        ema_tau=int(config.model.rfdetr.get("ema_tau", 1000)),
        lr_scheduler=config.model.rfdetr.get("lr_scheduler", "step"),
        warmup_epochs=float(config.model.rfdetr.get("warmup_epochs", 3.0)),
        lr_drop=int(config.model.rfdetr.get("lr_drop", 100)),
        lr_min_factor=float(config.model.rfdetr.get("lr_min_factor", 0.0)),
        eval_max_dets=int(config.model.get("max_detections", 100)),
        early_stopping=bool(getattr(config.trainer, "early_stopping", True)),
        early_stopping_patience=int(getattr(config.trainer, "early_stopping_patience", 5)),
        eval_interval=int(getattr(config.trainer, "check_val_every_n_epoch", 1)),
        num_workers=int(config.data.num_workers),
        seed=config.get("seed", 42),
        accelerator=config.trainer.get("accelerator", "auto"),
        log_per_class_metrics=True,
        train_log_sync_dist=True,
        compute_val_loss=True,
        compute_test_loss=True,
        fp16_eval=True,
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
        lora_cfg=lora_cfg,
        delay_lora=True
    )
    module.config = config
    # Hook into the module to save the PEFT adapter alongside the main checkpoint
    def save_adapter_hook(filepath: str):
        if hasattr(module.model, "save_pretrained"):
            import os
            
            target_datasets = getattr(config.data, "target_datasets", ["unknown_target"])
            if isinstance(target_datasets, str):
                target_datasets = [target_datasets]
            else:
                target_datasets = list(target_datasets)
            target = "_".join(target_datasets) if len(target_datasets) > 0 else "unknown_target"
            frac = getattr(config.data, "target_data_frac", getattr(config.data, "lora_frac", "unknown"))
            adapter_dir = os.path.join(config.checkpointing.save_dir, "phase2", "adapters", f"{target}_r{lora_cfg.get('r', 32)}_frac{frac}_best")
            
            os.makedirs(adapter_dir, exist_ok=True)
            
            # Load the actual best weights into the model before extracting the PEFT adapter
            try:
                ckpt = torch.load(filepath, map_location="cpu", weights_only=False)
                state_dict = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
                module.model.load_state_dict(state_dict, strict=False)
                rank_zero_print(f"Loaded BEST checkpoint weights from {filepath} for adapter extraction.")
            except Exception as e:
                rank_zero_print(f"Warning: Could not load best checkpoint weights for adapter save. Using current model state. Error: {e}")
                
            module.model.save_pretrained(adapter_dir)
            rank_zero_print(f"Saved lightweight PEFT adapter to {adapter_dir}")
            
    # Subclass ModelCheckpoint so we save the adapter EXACTLY when the best checkpoint is evaluated and saved.
    class PEFTModelCheckpoint(ModelCheckpoint):
        def _save_checkpoint(self, trainer, filepath: str) -> None:
            super()._save_checkpoint(trainer, filepath)
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                if "best-bbox" in filepath and "ema" not in filepath:
                    save_adapter_hook(filepath)

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
    trainer_kwargs["use_distributed_sampler"] = False
    trainer_kwargs["num_nodes"] = int(os.environ.get("SLURM_NNODES", config.trainer.get("num_nodes", 1)))

    
    if getattr(config, "debug", False):
        rank_zero_print("--- RUNNING IN DEBUG MODE ---")
        trainer_kwargs["limit_train_batches"] = getattr(config.trainer, "limit_train_batches", 10)
        trainer_kwargs["limit_val_batches"] = getattr(config.trainer, "limit_val_batches", 10)
        trainer_kwargs["limit_test_batches"] = getattr(config.trainer, "limit_test_batches", 10)
        train_config.epochs = 1
    else:
        if "limit_train_batches" in config.trainer:
            trainer_kwargs["limit_train_batches"] = config.trainer.limit_train_batches
        if "limit_val_batches" in config.trainer:
            trainer_kwargs["limit_val_batches"] = config.trainer.limit_val_batches
        if "limit_test_batches" in config.trainer:
            trainer_kwargs["limit_test_batches"] = config.trainer.limit_test_batches
        if "check_val_every_n_epoch" in config.trainer:
            trainer_kwargs["check_val_every_n_epoch"] = int(config.trainer.check_val_every_n_epoch)

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
            
        # Ensure LoRA configs (like rank and alpha) are explicitly logged, including defaults
        if finetune_mode == "lora" or config.model.rfdetr.get("backbone_lora", False):
            r_val = lora_cfg.get("r", 32)
            lora_log_info = {
                "r": r_val,
                "alpha": lora_cfg.get("alpha", r_val * 2),
                "use_dora": lora_cfg.get("use_dora", False),
                "dropout": lora_cfg.get("dropout", 0.05),
            }
            if "model" in cfg_for_log and "rfdetr" in cfg_for_log["model"]:
                if "lora" not in cfg_for_log["model"]["rfdetr"] or not cfg_for_log["model"]["rfdetr"]["lora"]:
                    cfg_for_log["model"]["rfdetr"]["lora"] = {}
                cfg_for_log["model"]["rfdetr"]["lora"].update(lora_log_info)

        custom_logger = WandbLogger(
            project="cell-detection-motifs",
            name=run_name,
            save_dir=train_config.output_dir,
            config=cfg_for_log
        )
        trainer_kwargs["logger"] = custom_logger
    # ---------------------------------------
        
    if num_devices <= 1:
        strategy_obj = "auto"
    else:
        from pytorch_lightning.strategies import DDPStrategy
        from datetime import timedelta
        strategy_obj = DDPStrategy(find_unused_parameters=True, timeout=timedelta(hours=2))
        
    trainer = build_trainer(
        train_config=train_config, 
        model_config=model_config, 
        strategy=strategy_obj,
        devices=devices,
        **trainer_kwargs
    )

    if hasattr(trainer, "_data_connector"):
        if hasattr(trainer._data_connector, "_use_distributed_sampler"):
            trainer._data_connector._use_distributed_sampler = False

    # 7. Callbacks Swapping
    trainer.callbacks = [cb for cb in trainer.callbacks if not isinstance(cb, COCOEvalCallback)]
    
    val_dataloader_names = ["train_ds/merged"]
    val_dataset_fns = [lambda ds=ds: ds.coco for ds in data_module.val_train_datasets_objs]
    
    test_dataloader_names = (
        [f"train_ds/{ds}/test" for ds in data_module.train_dataset_names] 
        + 
        [f"test_ds/{ds}/test" for ds in data_module.test_dataset_names]
    )
    test_dataset_fns = (
        [lambda ds=ds: ds.coco for ds in getattr(data_module, "train_test_datasets_objs", [])] 
        + 
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
        CheckpointClass = PEFTModelCheckpoint if finetune_mode == "lora" else ModelCheckpoint
        trainer.callbacks.append(
            CheckpointClass(
                dirpath=os.path.join(config.checkpointing.save_dir, "phase2", f"{motif_config_name}_{finetune_mode}_{timestamp}"),
                filename="best-bbox",
                monitor="val/mAP_50_95",
                mode="max",
                save_top_k=1,
                auto_insert_metric_name=False,
                enable_version_counter=False,
            )
        )
        if bool(config.model.rfdetr.get("use_ema", True)):
            trainer.callbacks.append(
                CheckpointClass(
                    dirpath=os.path.join(config.checkpointing.save_dir, "phase2", f"{motif_config_name}_{finetune_mode}_{timestamp}"),
                    filename="best-ema-bbox",
                    monitor="val/ema_mAP_50_95",
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
        
        if getattr(module.model_config, "backbone_lora", False):
            module._apply_lora()
            
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
            
        rank_zero_print("Starting Validation/Test eval with loaded weights...")
        trainer.test(module, datamodule=data_module)
    else:
        rank_zero_print(f"Starting Training for Motif: {motif_config_name}")
        ckpt_path = config.get("initialization", {}).get("load_from_checkpoint", None)
        base_ckpt = config.get("initialization", {}).get("base_checkpoint", None)
        
        # 1. Manually load base model weights if specified (used for fresh LoRA runs to skip optimizer states)
        if base_ckpt and not ckpt_path:
            if finetune_mode != "lora":
                rank_zero_print(f"⚠️  WARNING: Using base_checkpoint in {finetune_mode} mode. Usually, full fine-tuning relies on the default rf-detr-seg.pth `pretrain_weights` behavior.")
                
            rank_zero_print(f"Initializing base weights manually from: {base_ckpt} (skipping optimizer state)")
            checkpoint = _load_ckpt(base_ckpt)
            weight_source = _select_eval_weights_source(base_ckpt, checkpoint, config=config)
            _load_selected_weights(module, checkpoint, weight_source)
            
        if getattr(module.model_config, "backbone_lora", False):
            module._apply_lora()
            
        # 2. Resume training state if load_from_checkpoint is provided (native Lightning behavior)
        if ckpt_path:
            rank_zero_print(f"Resuming full training state from checkpoint: {ckpt_path}")
            trainer.fit(module, datamodule=data_module, ckpt_path=ckpt_path)
        else:
            trainer.fit(module, datamodule=data_module)
        
        rank_zero_print("Training complete. Starting Validation/Test eval...")
        trainer.test(module, datamodule=data_module)
    
    # Sync optimized weights back so predictive APIs work post-training
    rf_wrapper.model.model = module.model
    rank_zero_print("Done!")

if __name__ == '__main__':
    main()
