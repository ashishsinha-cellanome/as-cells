#!/usr/bin/env python3
"""
Training script for RT-DETR with DINOv2 backbone using PyTorch Lightning.
Powered by Hydra for flexible configuration.
"""

import os
import datetime
import shutil
import torch
torch.set_float32_matmul_precision('medium')

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
from torchvision.datasets import CocoDetection
import torch.distributed as dist

from omegaconf import DictConfig, OmegaConf
OmegaConf.register_new_resolver("extract_name", lambda path: path.split("/")[-1])
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
import wandb

from models.custom_rt_detr_with_dinov2_backbone import (
    RTDetrV2ForObjectDetectionWithCustomBackbone,
    RTDetrV2ConfigWithCustomBackBone
)
from models.rt_detr_v1_model import RTDetrV1Model
from models.dinov2_backbone_with_fpn import Dinov2BackBoneWithFPN, Dinov2BackBoneWithFPNConfig
from models.rt_detr_lightning_module import RTDETRLightningModule
from data.coco_data_module import COCODataModule
from models.backbone_factory import build_backbone,freeze_backbone_layers, get_backbone_unique_id
from utils.train_utils import BackupToNASCallback

def rank_zero_print(*args, **kwargs):
    """Print only on the main process (Rank 0)."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)

def create_initial_checkpoint(config: DictConfig) -> str:
    """
    Create initial RT-DETR checkpoint with DINOv2 backbone.
    """
    print("\n" + "="*80)
    print(f"Creating initial RT-DETR checkpoint with {config.model.backbone.type.upper()} backbone...")
    print("="*80 + "\n")
    
    model_config = config.model
    checkpoint_config = config.checkpointing
    
    # Determine suffix WITHOUT loading the model
    unique_suffix = get_backbone_unique_id(model_config.backbone, model_config.rtdetr.model_name)
    
    if model_config.backbone.type == "resnet":
        full_suffix = f"{unique_suffix}" 
    else:
        full_suffix = f"{unique_suffix}_{model_config.rtdetr.model_name}"
    base_rtdetr_path = hydra.utils.to_absolute_path(checkpoint_config.rtdetr_initial_checkpoint)
    local_path = f"{base_rtdetr_path}{full_suffix}"

    # get nas path
    nas_base = checkpoint_config.get("nas_initial_checkpoint")
    nas_path = None
    if nas_base:
        nas_path = f"{hydra.utils.to_absolute_path(nas_base)}{full_suffix}"

    # TODO: remove later. best to be used only on denvr
    # TODO: change to true, when running on DENVR
    if False: # os.path.exists(local_path) and len(os.listdir(local_path)) > 0:
        # print(f"✓ Found weights in SCRATCH: {local_path}")
        rank_zero_print( f"✓ Found weights in SCRATCH: {local_path}")

        return local_path

    # Only if NAS is configured and file exists there
    if nas_path and os.path.exists(nas_path) and len(os.listdir(nas_path)) > 0:
        rank_zero_print(f"✓ Found weights on NAS: {nas_path}")
        rank_zero_print(f"  -> Copying to scratch...")
        try:
            shutil.copytree(nas_path, local_path)
            rank_zero_print("  -> Copy complete.")
            return local_path
        except Exception as e:
            rank_zero_print(f"  -> Copy failed ({e}). Creating new...")
        
    rank_zero_print("! Weights not found. Starting Heavy Initialization...")

    backbone_model, backbone_config_obj, _ = build_backbone(
        model_config.backbone, 
        model_config.rtdetr.model_name
    )
    
    if model_config.backbone.type in ["official", "resnet"]:
         base_model_name = model_config.backbone.name 
    else:
         base_model_name = model_config.rtdetr.model_name
    
    rank_zero_print(f"Loading Base Weights: PekingU/{base_model_name}")

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

    # # Return the specific versioned path
    # # return versioned_rtdetr_path
    id2label = {int(k): v for k, v in model_config.label_map.items()}
    label2id = {v: k for k, v in id2label.items()}
    overrides = OmegaConf.to_container(model_config.rtdetr, resolve=True)
    for k in ["pretrained_name_or_path", "config_overrides", "model_name", 'name']:
        overrides.pop(k, None)
    
    if "rtdetr_v2" in model_config.rtdetr.model_name:
        model_cls = RTDetrV2ForObjectDetection
        config_cls = RTDetrV2ConfigWithCustomBackBone
    else:
        rank_zero_print(f"Detected RT-DETRv1 model: {model_config.rtdetr.model_name}")
        model_cls = RTDetrV1Model
        # RTDetrV1Model uses RTDetrConfigWithCustomBackBone internally
        from models.rt_detr_v1_model import RTDetrConfigWithCustomBackBone 
        config_cls = RTDetrConfigWithCustomBackBone

    model = model_cls.from_pretrained(
        f"PekingU/{base_model_name}",
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        **overrides
    )
    # Inject Backbone
    if config.model.backbone.type == "resnet":
        if not model_config.backbone.train_backbone:
            freeze_backbone_layers(model, freeze_at_stage=5)
        else:
            freeze_backbone_layers(model, freeze_at_stage=model_config.backbone.freeze_at_stage)
    else:
        pretrained_model_config_dict = model.config.to_dict()
        rt_detr_config = config_cls(**pretrained_model_config_dict)
        rt_detr_config.backbone_config = backbone_config_obj
        model.config = rt_detr_config
        # Handle difference in backbone attribute between V1 and V2
        if hasattr(model, 'model') and hasattr(model.model, 'backbone'):
            model.model.backbone = backbone_model
        else: # V1 structure often has direct backbone or wrapped differently, handled by model class usually but here we inject
             # RTDetrV1Model wraps RTDetrForObjectDetection which has model.backbone
             model.model.backbone = backbone_model

    # Save to Scratch (Always)
    rank_zero_print(f"Saving new model to Scratch: {local_path}")
    model.save_pretrained(local_path)

    #     # --- LOGIC FORK ---
    #     if  config.model.backbone.type == "resnet":
    #         # Case A: Official Backbone
    #         print("✓ Using Official ResNet Backbone")
    #         # Freeze layers if requested
    #         if not model_config.backbone.train_backbone:
    #             freeze_backbone_layers(model, model_config.backbone.freeze_at_stage)
    #     else:
    #         # Case B: Custom (DINOv2) Backbone
    #         print("✓ Injecting Custom Backbone")
            
    #         # (Your existing injection logic)
    #         pretrained_model_config_dict = model.config.to_dict()
    #         rt_detr_config = RTDetrV2ConfigWithCustomBackBone(**pretrained_model_config_dict)
    #         rt_detr_config.backbone_config = backbone_config_obj
    #         model.config = rt_detr_config
    #         model.model.backbone = backbone_model
            
    #     model.save_pretrained(versioned_rtdetr_path)
    #     print(f"✓ Model saved to {versioned_rtdetr_path}")

    # # return versioned_rtdetr_path

    # Backup to NAS (If available)
    if nas_path:
        rank_zero_print(f"Mirroring new model to NAS: {nas_path}")
        try:
            if not os.path.exists(nas_path):
                shutil.copytree(local_path, nas_path)
        except Exception as e:
            rank_zero_print(f"Warning: Backup to NAS failed: {e}")

    return local_path


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
            rank_zero_print(f"WARNING: Checkpoint not found at {model_checkpoint_path}")
            rank_zero_print("Creating initial checkpoint...")
            model_checkpoint_path = create_initial_checkpoint(config)
    
    rank_zero_print(f"\nLoading RT-DETR model from: {model_checkpoint_path}")
    
    if "rtdetr_v2" in config.model.rtdetr.model_name:
        model_cls = RTDetrV2ForObjectDetectionWithCustomBackbone
    else:
        model_cls = RTDetrV1Model

    model = model_cls.from_pretrained(
        model_checkpoint_path,
    )
    if config.model.backbone.type == "resnet":
        if not config.model.backbone.train_backbone:
            rank_zero_print("[INFO] Re-applying backbone freezing after load...")
            freeze_backbone_layers(model, config.model.backbone.freeze_at_stage)
    
    elif config.model.backbone.type == "dinov2":
         # Re-freeze DINOv2 if needed (usually handled by DINO class init, 
         # but good to ensure if you are loading a full model checkpoint)
         rank_zero_print("[INFO] Ensuring DINOv2 backbone is frozen...")
         for name, param in model.named_parameters():
             if "model.backbone.backbone" in name: # The ViT part
                 param.requires_grad = False

    # Check for EMA keys if EMA is enabled (just a warning)
    if hasattr(config.model, 'ema') and config.model.ema.enabled:
        # We can't easily check the checkpoint file here without reloading it, 
        # but the LightningModule init will set up self.ema_model.
        # If loading from a PL checkpoint later, strict=False handles it.
        pass

    # breakpoint()
    rtdetr_overrides = OmegaConf.to_container(config.model.rtdetr, resolve=True)
    rtdetr_overrides.pop("pretrained_name_or_path", None)
    rtdetr_overrides.pop("config_overrides", None)
    rtdetr_overrides.pop("model_name", None)
    rtdetr_overrides.pop("name", None)
    if rtdetr_overrides:
        rank_zero_print("Checking for model config overrides...")
        changes_made = False
        for key, value in rtdetr_overrides.items():
            if hasattr(model.config, key):
                current_value = getattr(model.config, key)
                # Only print and set if the value has changed
                if current_value != value:
                    if not changes_made:
                        rank_zero_print("Applying config overrides to loaded model:")
                        changes_made = True
                    rank_zero_print(f"  > Setting model.config.{key}: {current_value} -> {value}")
                    setattr(model.config, key, value)
            else:
                rank_zero_print(f"  > WARNING: model.config has no attribute '{key}' (cannot set)")
        
        if not changes_made:
            rank_zero_print("...Loaded model config already matches overrides.")
    
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
    
    rank_zero_print(f"✓ Model loaded successfully")
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
    
    rank_zero_print(f"✓ Data module configured for: {data_config.path}")
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
    callbacks = []
    
    # 1. Standard Model Checkpoint
    # Tracks the standard validation metric (e.g. val/map)
    rank_zero_print(f"Configure ModelCheckpoint for Standard Model: {checkpoint_config.monitor}")
    callbacks.append(
        ModelCheckpoint(
            dirpath=os.path.join(hydra.utils.to_absolute_path(checkpoint_config.save_dir), 'ckpts'),
            filename='rtdetr-regular-{epoch:02d}-{' + checkpoint_config.monitor.replace('/', '_') + ':.4f}',
            monitor=checkpoint_config.monitor,
            mode=checkpoint_config.mode,
            save_top_k=checkpoint_config.save_top_k,
            save_last=checkpoint_config.save_last, # 'last.ckpt' will be managed by this one
            every_n_epochs=checkpoint_config.every_n_epochs,
            verbose=True,
        )
    )

    # 2. EMA Model Checkpoint (If enabled)
    # Tracks the EMA validation metric (val/map_ema)
    if hasattr(config.model, 'ema') and config.model.ema.enabled:
        rank_zero_print("💡 EMA enabled: Adding second ModelCheckpoint for 'val/map_ema'")
        ema_monitor = "val/map_ema"
        callbacks.append(
            ModelCheckpoint(
                dirpath=os.path.join(hydra.utils.to_absolute_path(checkpoint_config.save_dir), 'ckpts'),
                filename='rtdetr-ema-{epoch:02d}-{' + ema_monitor.replace('/', '_') + ':.4f}',
                monitor=ema_monitor,
                mode=checkpoint_config.mode,
                save_top_k=checkpoint_config.save_top_k,
                save_last=False, # Don't duplicate 'last.ckpt' logic
                every_n_epochs=checkpoint_config.every_n_epochs,
                verbose=True,
            )
        )

    callbacks.append(LearningRateMonitor(logging_interval='step'))
    callbacks.append(ModelSummary(max_depth=2))

    if "backup_dir" in checkpoint_config and checkpoint_config.backup_dir:
        # Resolve path (handle ${hydra...} if needed, though usually resolved by now)
        backup_path = hydra.utils.to_absolute_path(checkpoint_config.backup_dir)
        callbacks.append(BackupToNASCallback(backup_dir=backup_path))

    rank_zero_print(f"✓ Callbacks configured")
    return callbacks

def setup_logger(config: DictConfig):
    """Setup WandB logger."""
    wandb_config = config.logging.wandb
    
    if not wandb_config.enabled:
        rank_zero_print("✓ WandB logging disabled")
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
    
    rank_zero_print(f"✓ WandB logger configured - Project: {wandb_config.project}")
    return logger

@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(config: DictConfig):
    # breakpoint()
    # Unlock config to make changes
    OmegaConf.set_struct(config, False)
    # breakpoint()
    # --- 1. Handle Run Naming (Cluster Agnostic) ---
    # Try to find a shared ID from common cluster/launcher environment variables
    # This ensures all ranks in a distributed run agree on the run name/ID.
    unique_id = None
    id_candidates = [
        "SLURM_JOB_ID",          # SLURM
        "TORCHELASTIC_RUN_ID",   # torchrun / torch.distributed.launch
        "WANDB_RUN_ID",          # User-provided WandB ID
        "PBS_JOBID",             # PBS/Torque
        "LSB_JOBID",             # LSF
    ]
    
    for var in id_candidates:
        if os.environ.get(var):
            unique_id = os.environ.get(var)
            rank_zero_print(f"🚀 Found Job ID from {var}: {unique_id}")
            break
            
    if not unique_id:
        # Fallback to timestamp if no manager/launcher detected
        # Note: In a raw SSH-loop launch without a shared env var, this might still differ by seconds.
        # Ideally use torchrun or srun for multi-node.
        unique_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        rank_zero_print(f"⚠️  No shared job ID found. Using timestamp: {unique_id}")
    
    if config.run_name.startswith("rtdetrv2_dinov2"):
        config.run_name = f"rtdetrv2_dinov2_{unique_id}"
    else:
        config.run_name = f"{config.run_name}_{unique_id}"

    # --- 2. Handle Hydra Sweep Logic ---
    hydra_cfg = HydraConfig.get()
    
    if hydra_cfg.mode == RunMode.MULTIRUN:
        rank_zero_print(f"🚀 Detected Hydra Sweep (Job {hydra_cfg.job.num})")
        
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
                rank_zero_print(f"   -> Added WandB tag: {tag}")

    # --- Hydra handles all config loading and merging ---
    # The 'config' object is already the final, merged config
    
    if config.debug:
        rank_zero_print ('Running in DEBUG mode')
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
        rank_zero_print(f"🔄 Manual Resume: Loading specified checkpoint: {ckpt_path}")
        ckpt_path = hydra.utils.to_absolute_path(ckpt_path)
    
    elif config.initialization.auto_resume:
        # FIX: Look inside the 'ckpts' subdirectory
        last_ckpt = os.path.join(run_save_dir, 'ckpts', "last.ckpt")
        
        if os.path.exists(last_ckpt):
            rank_zero_print (f"🔄 Auto-Resume: Found existing 'last.ckpt' at {last_ckpt}")
            ckpt_path = last_ckpt
        else:
            rank_zero_print (f"ℹ️  Auto-Resume: No 'last.ckpt' found in {last_ckpt}. Starting fresh.")
            ckpt_path = None
    else:
        ckpt_path = None

    OmegaConf.set_struct(config, True) # Re-lock config
    # Set dynamic save_dir (relative to hydra's CWD)
    # This is now handled by ModelCheckpoint's dirpath
    
    rank_zero_print("\n" + "="*80)
    rank_zero_print("RT-DETR Training with DINOv2 Backbone (Hydra Edition)")
    rank_zero_print("="*80 + "\n")
    
    rank_zero_print("--- CWD (Hydra Output Dir) ---")
    rank_zero_print(f"{os.getcwd()}\n")
    
    rank_zero_print("--- Final Configuration ---")
    rank_zero_print(OmegaConf.to_yaml(config))
    rank_zero_print("---------------------------")
    
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
    
    # Auto-detect number of nodes (default to 1)
    num_nodes = int(os.environ.get("SLURM_NNODES", 1))
    rank_zero_print(f"🌍 Detected Number of Nodes: {num_nodes}")

    trainer = pl.Trainer(
        accelerator=trainer_config.accelerator,
        devices=trainer_config.devices,
        num_nodes=num_nodes,
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
        limit_test_batches = data_config.limit_test_batches if not config.debug else 10,
        limit_train_batches = data_config.limit_train_batches if not config.debug else 10,
        limit_val_batches = data_config.limit_val_batches,
        profiler = None if config.debug else profiler,
    )
    
    # Handle checkpoint path (must be absolute)
    ckpt_path = config.initialization.load_from_checkpoint
    if ckpt_path:
        ckpt_path = hydra.utils.to_absolute_path(ckpt_path)
        if not os.path.exists(ckpt_path):
            rank_zero_print(f"WARNING: Checkpoint path not found: {ckpt_path}")
            ckpt_path = None
    
    if config.test_only:
        rank_zero_print("\n" + "="*80)
        rank_zero_print("Running in TEST-ONLY mode")
        rank_zero_print("="*80 + "\n")
        if not ckpt_path:
            raise ValueError("Must provide a checkpoint path via 'initialization.load_from_checkpoint' for test-only mode.")
        
        trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)
    else:
        rank_zero_print("\n" + "="*80)
        rank_zero_print("Starting Training")
        rank_zero_print("="*80 + "\n")

        trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
        
        rank_zero_print("\n" + "="*80)
        rank_zero_print("Training Complete!")
        rank_zero_print("="*80 + "\n")
        
        # Test the best model
        best_path = None
        best_score = None
        
        # 1. Try to find the EMA checkpoint callback first
        if hasattr(config.model, 'ema') and config.model.ema.enabled:
            for cb in trainer.callbacks:
                if isinstance(cb, ModelCheckpoint) and cb.monitor == "val/map_ema":
                    if cb.best_model_path:
                        best_path = cb.best_model_path
                        best_score = cb.best_model_score
                        rank_zero_print(f"🎯 Selected BEST EMA checkpoint (monitor: {cb.monitor})")
                    break
        
        # 2. Fallback to Regular checkpoint if EMA not found or not enabled
        if not best_path:
            for cb in trainer.callbacks:
                if isinstance(cb, ModelCheckpoint) and cb.monitor == config.checkpointing.monitor:
                    if cb.best_model_path:
                        best_path = cb.best_model_path
                        best_score = cb.best_model_score
                        rank_zero_print(f"🎯 Selected BEST REGULAR checkpoint (monitor: {cb.monitor})")
                    break

        if best_path:
            rank_zero_print(f"Best model found at: {best_path}")
            if best_score is not None:
                rank_zero_print(f"Best score: {best_score:.4f}")
            
            rank_zero_print("\nLoading BEST checkpoint with strict=False...")
            
            # Actually, the safest way is to load state_dict into the CURRENT model structure
            try:
                checkpoint = torch.load(best_path, map_location=model.device)
                # If checkpoint has 'state_dict' key (PL format), use it
                state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
                
                # Load with strict=False
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    rank_zero_print(f"⚠️  Missing keys during load: {missing_keys[:5]} ...")
                if unexpected_keys:
                    rank_zero_print(f"⚠️  Unexpected keys during load: {unexpected_keys[:5]} ...")
                
                rank_zero_print("\nRunning test evaluation on BEST checkpoint...")
                trainer.test(model, datamodule=data_module)
                
            except Exception as e:
                rank_zero_print(f"❌ Failed to load best checkpoint: {e}")
        else:
            rank_zero_print("\nNo best model found. Testing disabled.")
    
    wandb.finish()

if __name__ == "__main__":
    main()
