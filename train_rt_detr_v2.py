#!/usr/bin/env python3
"""
Training script for RT-DETR with DINOv2 backbone using PyTorch Lightning.
Powered by Hydra for flexible configuration.
"""

import os
import datetime
import shutil
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
from torchvision.datasets import CocoDetection

# --- HYDRA & OMEGACONF ---
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
import wandb

from models.custom_rt_detr_with_dinov2_backbone import (
    RTDetrV2ForObjectDetectionWithCustomBackbone,
    RTDetrV2ConfigWithCustomBackBone
)
from models.dinov2_backbone_with_fpn import Dinov2BackBoneWithFPN, Dinov2BackBoneWithFPNConfig
from models.rt_detr_lightning_module import RTDETRLightningModule
from data.coco_data_module import COCODataModule
from models.backbone_factory import build_backbone,freeze_backbone_layers

def create_initial_checkpoint(config: DictConfig) -> str:
    """
    Create initial RT-DETR checkpoint with DINOv2 backbone.
    (This function is now Hydra-aware)
    """
    print("\n" + "="*80)
    print("Creating initial RT-DETR checkpoint with DINOv2 backbone...")
    print("="*80 + "\n")
    
    model_config = config.model
    checkpoint_config = config.checkpointing
    
    # choose the backbone model
    backbone_model, backbone_config_obj, unique_suffix = build_backbone(
        model_config.backbone, 
        model_config.rtdetr.model_name
    )
    print(f"DEBUG CHECK: backbone.type is '{model_config.backbone.type}'")
    print(f"DEBUG CHECK: backbone.name is '{model_config.backbone.name}'")
    
    # Check both "official" AND "resnet" to catch the mismatch
    if model_config.backbone.type in ["official", "resnet"]: 
         base_model_name = model_config.backbone.name
         print(f"DEBUG: Logic selected OFFICIAL name: {base_model_name}")
    else:
         base_model_name = model_config.rtdetr.model_name
         print(f"DEBUG: Logic selected DEFAULT name: {base_model_name}")

    print(f"Loading Base Weights: PekingU/{base_model_name}")
    
    base_rtdetr_path = hydra.utils.to_absolute_path(checkpoint_config.rtdetr_initial_checkpoint)
    # If using official backbone, we append the model name to ensure uniqueness
    if model_config.backbone.type == "resnet":
        full_suffix = f"{unique_suffix}" 
    else:
        full_suffix = f"{unique_suffix}_{model_config.rtdetr.model_name}"

    versioned_rtdetr_path = f"{base_rtdetr_path}{full_suffix}"

    # 3. Construct Versioned Paths
    # Base: .../dinov2_backbone_with_fpn
    # New:  .../dinov2_backbone_with_fpn_indices_4_8_12
    # base_backbone = hydra.utils.to_absolute_path(checkpoint_config.dinov2_backbone_checkpoint)
    # base_rtdetr = hydra.utils.to_absolute_path(checkpoint_config.rtdetr_initial_checkpoint)
    
    # # versioned_backbone_path = f"{base_backbone}{indices_suffix}"
    # # versioned_rtdetr_path = f"{base_rtdetr}{indices_suffix}"

    # # print(f"Target Architecture Indices: {target_indices}")
    # # print(f"Target Checkpoint Path:      {versioned_rtdetr_path}")
    # # The suffix now already contains the rtdetr model name, so we just append it
    # versioned_backbone_path = f"{base_backbone}{unique_suffix}"
    # versioned_rtdetr_path = f"{base_rtdetr}{unique_suffix}"

    # print(f"Backbone Type:          {model_config.backbone.type}")
    # print(f"RT-DETR Variant:        {model_config.rtdetr.model_name}")
    # print(f"Target Checkpoint Path: {versioned_rtdetr_path}")

    # # TODO: remove later
    # # force ckpt creation for every run 
    # if True: #not os.path.exists(versioned_backbone_path) or len(os.listdir(versioned_backbone_path)) == 0:
    #     print(f"\n[INFO] Cached backbone not found. Creating: {versioned_backbone_path}")
    #     os.makedirs(versioned_backbone_path, exist_ok=True)
    #     backbone_model.save_pretrained(versioned_backbone_path)
    #     print(f"✓ Backbone saved.")
        
    #     # dinov2_backbone = Dinov2BackBoneWithFPN.from_pretrained(
    #     #     model_config.dinov2.pretrained_name_or_path,
    #     #     output_indices_for_fpn=target_indices,
    #     #     first_layer_dims=first_layer_dims,
    #     #     fpn_type=model_config.dinov2.fpn_type,
    #     #     scale_factor=model_config.dinov2.scale_factor,
    #     #     upscale_method=model_config.dinov2.upscale_method,
    #     #     intermediate_channel_sizes=intermediate_channel_sizes,
    #     # )
    #     # dinov2_backbone.save_pretrained(versioned_backbone_path)
    #     # print(f"✓ DINOv2 backbone saved.")
    # else:
    #     print(f"✓ Found cached backbone at: {versioned_backbone_path}")

    # # TODO: remove later
    # if True: #not os.path.exists(versioned_rtdetr_path) or len(os.listdir(versioned_rtdetr_path)) == 0:
    #     print(f"\n[INFO] Cached RT-DETR not found. Creating: {versioned_rtdetr_path}")
    #     os.makedirs(versioned_rtdetr_path, exist_ok=True)
        
    #     id2label = {int(k): v for k, v in model_config.label_map.items()}
    #     label2id = {v: k for k, v in id2label.items()}
        
    #     # Get overrides
    #     # breakpoint()
    #     overrides = OmegaConf.to_container(model_config.rtdetr, resolve=True)
    #     # Clean up keys that aren't model arguments
    #     for k in ["pretrained_name_or_path", "config_overrides", "model_name"]:
    #         overrides.pop(k, None)

    #     print(f"Loading base RT-DETR to inject backbone...")
    #     pretrained_rt_detr = RTDetrV2ForObjectDetection.from_pretrained(
    #         model_config.rtdetr.pretrained_name_or_path,
    #         id2label=id2label,
    #         label2id=label2id,
    #         ignore_mismatched_sizes=True,
    #         **overrides
    #     )

    #     # TODO: check and remove later
    #     # Inject Custom Backbone Config
    #     pretrained_model_config_dict = pretrained_rt_detr.config.to_dict()
    #     rt_detr_config = RTDetrV2ConfigWithCustomBackBone(**pretrained_model_config_dict)
    #     rt_detr_config.backbone_config = backbone_config_obj
        
    #     # Inject Backbone Model
    #     pretrained_rt_detr.config = rt_detr_config
    #     pretrained_rt_detr.model.backbone = backbone_model
    #     pretrained_rt_detr.save_pretrained(versioned_rtdetr_path)
        
    #     # Load the SPECIFIC backbone version we just checked/created
    #     # dinov2_backbone_config = Dinov2BackBoneWithFPNConfig.from_pretrained(versioned_backbone_path)
    #     # dinov2_backbone = Dinov2BackBoneWithFPN.from_pretrained(versioned_backbone_path)
        
    #     # pretrained_model_config_dict = pretrained_rt_detr.config.to_dict()
    #     # rt_detr_config = RTDetrV2ConfigWithCustomBackBone(**pretrained_model_config_dict)
    #     # rt_detr_config.backbone_config = dinov2_backbone_config
        
    #     # pretrained_rt_detr.config = rt_detr_config
    #     # pretrained_rt_detr.model.backbone = dinov2_backbone
    #     # pretrained_rt_detr.save_pretrained(versioned_rtdetr_path)
        
    #     # print(f"✓ RT-DETR initialized and saved to: {versioned_rtdetr_path}")
    # else:
    #     print(f"✓ Found cached RT-DETR at: {versioned_rtdetr_path}")
    
    # # Return the specific versioned path
    # return versioned_rtdetr_path
    if True: # Replace with os.path.exists check
        print(f"\n[INFO] Creating/Loading model at: {versioned_rtdetr_path}")
        os.makedirs(versioned_rtdetr_path, exist_ok=True)
        
        id2label = {int(k): v for k, v in model_config.label_map.items()}
        label2id = {v: k for k, v in id2label.items()}
        
        # Determine which base model to load
        # If official, we force the base model to be the one specified in backbone config
        if model_config.backbone.type in ["official", "resnet"]:
             base_model_name = model_config.backbone.name 
        else:
             base_model_name = model_config.rtdetr.model_name

        print(f"Loading Base Weights: PekingU/{base_model_name}")
        
        # Load Model
        overrides = OmegaConf.to_container(model_config.rtdetr, resolve=True)
        for k in ["pretrained_name_or_path", "config_overrides", "model_name"]:
            overrides.pop(k, None)

        model = RTDetrV2ForObjectDetection.from_pretrained(
            f"PekingU/{base_model_name}",
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            **overrides
        )

        # --- LOGIC FORK ---
        if  config.model.backbone.type == "resnet":
            # Case A: Official Backbone
            print("✓ Using Official ResNet Backbone")
            # Freeze layers if requested
            if not model_config.backbone.train_backbone:
                freeze_backbone_layers(model, model_config.backbone.freeze_at_stage)
        else:
            # Case B: Custom (DINOv2) Backbone
            print("✓ Injecting Custom Backbone")
            
            # (Your existing injection logic)
            pretrained_model_config_dict = model.config.to_dict()
            rt_detr_config = RTDetrV2ConfigWithCustomBackBone(**pretrained_model_config_dict)
            rt_detr_config.backbone_config = backbone_config_obj
            model.config = rt_detr_config
            model.model.backbone = backbone_model
            
        model.save_pretrained(versioned_rtdetr_path)
        print(f"✓ Model saved to {versioned_rtdetr_path}")

    return versioned_rtdetr_path


def setup_model(config: DictConfig) -> RTDETRLightningModule:
    """Setup the RT-DETR model with DINOv2 backbone."""
    model_config = config.model
    init_config = config.initialization
    # breakpoint()
    if init_config.create_initial_checkpoint:
        model_checkpoint_path = create_initial_checkpoint(config)
    else:
        model_checkpoint_path = hydra.utils.to_absolute_path(config.checkpointing.rtdetr_initial_checkpoint)
        if not os.path.exists(model_checkpoint_path):
            print(f"WARNING: Checkpoint not found at {model_checkpoint_path}")
            print("Creating initial checkpoint...")
            model_checkpoint_path = create_initial_checkpoint(config)
    
    print(f"\nLoading RT-DETR model from: {model_checkpoint_path}")
    model = RTDetrV2ForObjectDetectionWithCustomBackbone.from_pretrained(
        model_checkpoint_path,
    )
    if config.model.backbone.type == "resnet":
        if not config.model.backbone.train_backbone:
            print("[INFO] Re-applying backbone freezing after load...")
            freeze_backbone_layers(model, config.model.backbone.freeze_at_stage)
    
    elif config.model.backbone.type == "dinov2":
         # Re-freeze DINOv2 if needed (usually handled by DINO class init, 
         # but good to ensure if you are loading a full model checkpoint)
         print("[INFO] Ensuring DINOv2 backbone is frozen...")
         for name, param in model.named_parameters():
             if "model.backbone.backbone" in name: # The ViT part
                 param.requires_grad = False

    # breakpoint()
    rtdetr_overrides = OmegaConf.to_container(config.model.rtdetr, resolve=True)
    rtdetr_overrides.pop("pretrained_name_or_path", None)
    rtdetr_overrides.pop("config_overrides", None)
    rtdetr_overrides.pop("model_name", None)
    if rtdetr_overrides:
        print("Checking for model config overrides...")
        changes_made = False
        for key, value in rtdetr_overrides.items():
            if hasattr(model.config, key):
                current_value = getattr(model.config, key)
                # Only print and set if the value has changed
                if current_value != value:
                    if not changes_made:
                        print("Applying config overrides to loaded model:")
                        changes_made = True
                    print(f"  > Setting model.config.{key}: {current_value} -> {value}")
                    setattr(model.config, key, value)
            else:
                print(f"  > WARNING: model.config has no attribute '{key}' (cannot set)")
        
        if not changes_made:
            print("...Loaded model config already matches overrides.")
    
    processor = RTDetrImageProcessor.from_pretrained(config.model.rtdetr.pretrained_name_or_path)
    processor.do_normalize = True
    processor.resample = 3
    processor.size = {
        "height": config.data.model_input_size,
        "width": config.data.model_input_size
    }
    
    data_path = hydra.utils.to_absolute_path(config.data.path)
    val_annot_path = os.path.join(data_path, 'images', config.val_name)
    val_json_path = os.path.join(data_path, f'{config.val_name}_annotations.json')
    val_coco_dataset = CocoDetection(root=val_annot_path, annFile=val_json_path, transforms=None)
    val_coco_gt = val_coco_dataset.coco
    val_coco_gt.dataset['info'] = {}

    test_annot_path = os.path.join(data_path, 'images', config.test_name)
    test_json_path = os.path.join(data_path, f'{config.test_name}_annotations.json')
    test_coco_dataset = CocoDetection(root=test_annot_path, annFile=test_json_path, transforms=None)
    test_coco_gt = test_coco_dataset.coco
    test_coco_gt.dataset['info'] = {}
    
    lightning_model = RTDETRLightningModule(
        model=model,
        image_processor=processor,
        config=config, # Pass the whole config
        val_coco_gt=val_coco_gt,
        test_coco_gt=val_coco_gt if config.debug else test_coco_gt,
    )
    
    print(f"✓ Model loaded successfully")
    return lightning_model, processor


def setup_data(config: DictConfig, processor) -> COCODataModule:
    """Setup the data module."""
    data_config = config.data
    
    data_module = COCODataModule(
        dataset_path=hydra.utils.to_absolute_path(data_config.path), # Use absolute path
        processor=processor,
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        model_input_size=data_config.model_input_size,
        min_random_scale=data_config.min_random_scale,
        max_random_scale=data_config.max_random_scale,
        p_noise=data_config.p_noise,
        org_images_in_model_input_size=data_config.org_images_in_model_input_size,
        config = config,
    )
    
    print(f"✓ Data module configured for: {data_config.path}")
    return data_module


def setup_profiler(config: DictConfig):
    # Note: Hydra changes CWD, profiler logs save to the hydra output dir
    profiler_config = config.training.profiler
    dir_name = "profiler_logs" # Will be saved inside hydra's output dir

    if profiler_config.type == 'simple':
        profiler = SimpleProfiler(dirpath=dir_name, filename="rtdetr_profile")
    elif profiler_config.type == 'advanced':
        profiler = AdvancedProfiler(dirpath=dir_name, filename="rtdetr_profile")
    else:
        return None
    return profiler

def setup_callbacks(config: DictConfig):
    """Setup training callbacks."""
    checkpoint_config = config.checkpointing
    
    callbacks = [
        ModelCheckpoint(
            # dirpath=os.path.join(hydra.utils.to_absolute_path(checkpoint_config.save_dir), config.run_name, 'ckpts'), # Use absolute path
            dirpath=os.path.join(hydra.utils.to_absolute_path(checkpoint_config.save_dir), 'ckpts'), # Use absolute path
            filename='rtdetr-{epoch:02d}-{val_map:.4f}',
            monitor=checkpoint_config.monitor,
            mode=checkpoint_config.mode,
            save_top_k=checkpoint_config.save_top_k,
            save_last=checkpoint_config.save_last,
            every_n_epochs=checkpoint_config.every_n_epochs,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval='epoch'),
        ModelSummary(max_depth=2),
    ]
    
    print(f"✓ Callbacks configured")
    return callbacks

def setup_logger(config: DictConfig):
    """Setup WandB logger."""
    wandb_config = config.logging.wandb
    
    if not wandb_config.enabled:
        print("✓ WandB logging disabled")
        return None
    
    logger = WandbLogger(
        project=wandb_config.project,
        name=config.run_name,
        tags=list(wandb_config.tags), # Convert OmegaConf list to plain list
        notes=wandb_config.notes,
        group=wandb_config.get("group"),
        config=OmegaConf.to_container(config, resolve=True), # Log full config
        # Hydra changes CWD, so we save logs to the new CWD
        save_dir=os.getcwd(), 
        reinit=True,
        
    )
    
    print(f"✓ WandB logger configured - Project: {wandb_config.project}")
    return logger

@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(config: DictConfig):
    # breakpoint()
    # Unlock config to make changes
    OmegaConf.set_struct(config, False)

    # --- 1. Handle Run Naming (Datetime) ---
    # We regenerate the timestamp here so that EVERY job in the sweep 
    # gets its own unique time-based ID (e.g., job 1 starts at :01, job 2 at :05)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # If the user didn't provide a specific name in CLI, use the timestamp format
    if config.run_name.startswith("rtdetrv2_dinov2"): 
        config.run_name = f"rtdetrv2_dinov2_{timestamp}"
    else:
        # If user passed a custom name (e.g. "my_sweep"), append timestamp to keep it unique
        config.run_name = f"{config.run_name}_{timestamp}"

    # --- 2. Handle Hydra Sweep Logic ---
    hydra_cfg = HydraConfig.get()
    
    if hydra_cfg.mode == RunMode.MULTIRUN:
        print(f"🚀 Detected Hydra Sweep (Job {hydra_cfg.job.num})")
        
        # A. Set WandB Group
        # We group by the directory name Hydra created for this sweep (shared by all jobs)
        # or you can use a static string like "Sweep_Nov17"
        if not config.logging.wandb.get("group"):
            # Use the parent multirun folder timestamp as the group ID
            # This keeps all runs in this sweep together in the UI
            sweep_id = os.path.basename(os.path.normpath(hydra_cfg.sweep.dir))
            config.logging.wandb.group = f"sweep_{sweep_id}"

        # B. Convert Overrides to Tags
        # Get list of overrides for this specific job (e.g. ["model.x=1", "data.y=2"])
        job_overrides = hydra_cfg.overrides.task
        
        for override in job_overrides:
            # Split "key=value"
            if "=" in override:
                key, value = override.split("=", 1)
                # Shorten the key (e.g. "model.rtdetr.config_overrides.aux_loss" -> "aux_loss")
                short_key = key.split(".")[-1] 
                tag = f"{short_key}={value}"
                
                # Add to WandB tags
                config.logging.wandb.tags.append(tag)
                print(f"   -> Added WandB tag: {tag}")

    # --- Hydra handles all config loading and merging ---
    # The 'config' object is already the final, merged config
    
    if config.debug:
        print ('Running in DEBUG mode')
        OmegaConf.set_struct(config, False) # Unlock config
        # Apply debug settings
        config.trainer.num_overfit_samples = 10
        config.run_name = f"DEBUG_{config.run_name}"
        config.logging.wandb.project = f"{config.logging.wandb.project}"
        # config.checkpointing.save_dir = os.path.join({config.checkpointing.save_dir}, config.run_name)
        OmegaConf.set_struct(config, True) # Re-lock config

    # logic for auto-resume training from lat ckpt
    base_save_dir = hydra.utils.to_absolute_path(config.checkpointing.save_dir)
    run_save_dir = os.path.join(base_save_dir, config.run_name)
    config.checkpointing.save_dir = run_save_dir
    
    # breakpoint()
    ckpt_path = config.initialization.load_from_checkpoint
    if ckpt_path:
        print(f"🔄 Manual Resume: Loading specified checkpoint: {ckpt_path}")
        ckpt_path = hydra.utils.to_absolute_path(ckpt_path)
    
    elif config.initialization.auto_resume:
        # FIX: Look inside the 'ckpts' subdirectory
        last_ckpt = os.path.join(run_save_dir, 'ckpts', "last.ckpt")
        
        if os.path.exists(last_ckpt):
            print (f"🔄 Auto-Resume: Found existing 'last.ckpt' at {last_ckpt}")
            ckpt_path = last_ckpt
        else:
            print (f"ℹ️  Auto-Resume: No 'last.ckpt' found in {last_ckpt}. Starting fresh.")
            ckpt_path = None
    else:
        ckpt_path = None

    OmegaConf.set_struct(config, True) # Re-lock config
    # Set dynamic save_dir (relative to hydra's CWD)
    # This is now handled by ModelCheckpoint's dirpath
    
    print("\n" + "="*80)
    print("RT-DETR Training with DINOv2 Backbone (Hydra Edition)")
    print("="*80 + "\n")
    
    print("--- CWD (Hydra Output Dir) ---")
    print(f"{os.getcwd()}\n")
    
    print("--- Final Configuration ---")
    print(OmegaConf.to_yaml(config))
    print("---------------------------")
    
    # Set seed
    pl.seed_everything(config.seed, workers=True)
    
    # Setup components
    model, processor = setup_model(config)
    data_module = setup_data(config, processor)
    callbacks = setup_callbacks(config)
    logger = setup_logger(config)
    profiler = setup_profiler(config)
    # breakpoint()

    # Create trainer
    trainer_config = config.trainer
    data_config = config.data
    
    trainer = pl.Trainer(
        accelerator=trainer_config.accelerator,
        devices=trainer_config.devices,
        precision=trainer_config.precision,
        strategy=trainer_config.strategy,
        max_epochs=trainer_config.max_epochs,
        log_every_n_steps=trainer_config.log_every_n_steps,
        val_check_interval=trainer_config.val_check_interval,
        gradient_clip_val=trainer_config.max_grad_norm,
        gradient_clip_algorithm=trainer_config.gradient_clip_algo,
        accumulate_grad_batches=trainer_config.accumulate_grad_batches,
        deterministic=trainer_config.deterministic,
        benchmark=trainer_config.benchmark,
        callbacks=callbacks,
        logger=logger,
        overfit_batches = trainer_config.num_overfit_samples,
        limit_test_batches = data_config.limit_test_batches,
        limit_train_batches = data_config.limit_train_batches,
        limit_val_batches = data_config.limit_val_batches,
        profiler = None if config.debug else profiler,
    )
    
    # Handle checkpoint path (must be absolute)
    ckpt_path = config.initialization.load_from_checkpoint
    if ckpt_path:
        ckpt_path = hydra.utils.to_absolute_path(ckpt_path)
        if not os.path.exists(ckpt_path):
            print(f"WARNING: Checkpoint path not found: {ckpt_path}")
            ckpt_path = None
    
    if config.test_only:
        print("\n" + "="*80)
        print("Running in TEST-ONLY mode")
        print("="*80 + "\n")
        if not ckpt_path:
            raise ValueError("Must provide a checkpoint path via 'initialization.load_from_checkpoint' for test-only mode.")
        
        trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)
    else:
        print("\n" + "="*80)
        print("Starting Training")
        print("="*80 + "\n")

        trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
        
        print("\n" + "="*80)
        print("Training Complete!")
        print("="*80 + "\n")
        
        # Test the best model
        if trainer.checkpoint_callback.best_model_path:
            print(f"Best model: {trainer.checkpoint_callback.best_model_path}")
            print(f"Best val_map: {trainer.checkpoint_callback.best_model_score:.4f}")
            print("\nRunning test evaluation on BEST checkpoint...")
            trainer.test(model, datamodule=data_module, ckpt_path='best')
        else:
            print("\nNo best model found. Testing disabled.")
    
    wandb.finish()

if __name__ == "__main__":
    main()