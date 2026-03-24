#!/usr/bin/env python3
"""
Training script for RT-DETR with DINOv2 backbone using PyTorch Lightning.
Supports COCO format datasets with train/valid/test splits.
"""

import os
import argparse
from typing import Dict, Any
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
)
from pytorch_lightning.loggers import WandbLogger
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
from torchvision.datasets import CocoDetection

from models.custom_rt_detr_with_dinov2_backbone import (
    RTDetrV2ForObjectDetectionWithCustomBackbone,
    RTDetrV2ConfigWithCustomBackBone,
)
from models.dinov2_backbone_with_fpn import (
    Dinov2BackBoneWithFPN,
    Dinov2BackBoneWithFPNConfig,
)
from models.rt_detr_lightning_module import RTDETRLightningModule
from data.coco_data_module import COCODataModule


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(
    base_config: Dict[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    """Merge CLI arguments with config file, CLI args take precedence."""
    config = base_config.copy()

    # Override with CLI arguments if provided
    if args.dataset_path:
        config["dataset"]["path"] = args.dataset_path
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.num_epochs:
        config["training"]["num_epochs"] = args.num_epochs
        config["trainer"]["max_epochs"] = args.num_epochs
    if args.checkpoint_dir:
        config["checkpointing"]["save_dir"] = args.checkpoint_dir
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers
    if args.wandb_project:
        config["wandb"]["project"] = args.wandb_project
    if args.wandb_name:
        config["wandb"]["name"] = args.wandb_name
    if args.no_wandb:
        config["wandb"]["enabled"] = False
    if args.create_initial_checkpoint:
        config["initialization"]["create_initial_checkpoint"] = True
    if args.resume_from:
        config["initialization"]["load_from_checkpoint"] = args.resume_from

    return config


def create_initial_checkpoint(config: Dict[str, Any]) -> str:
    """
    Create initial RT-DETR checkpoint with DINOv2 backbone.
    Returns the path to the created checkpoint.
    """
    print("\n" + "=" * 80)
    print("Creating initial RT-DETR checkpoint with DINOv2 backbone...")
    print("=" * 80 + "\n")

    model_config = config["model"]
    checkpoint_config = config["checkpointing"]

    # Step 1: Create DINOv2 backbone with FPN
    backbone_checkpoint_path = checkpoint_config["dinov2_backbone_checkpoint"]
    if not os.path.exists(backbone_checkpoint_path):
        print(f"Creating DINOv2 backbone checkpoint at: {backbone_checkpoint_path}")
        os.makedirs(backbone_checkpoint_path, exist_ok=True)

        dinov2_backbone = Dinov2BackBoneWithFPN.from_pretrained(
            model_config["dinov2_backbone"],
            output_indices_for_fpn=model_config["output_indices_for_fpn"],
        )
        dinov2_backbone.save_pretrained(backbone_checkpoint_path)
        print(f"✓ DINOv2 backbone saved to: {backbone_checkpoint_path}")
    else:
        print(
            f"✓ DINOv2 backbone checkpoint already exists at: {backbone_checkpoint_path}"
        )

    # Step 2: Create RT-DETR with custom backbone
    rtdetr_checkpoint_path = checkpoint_config["rtdetr_initial_checkpoint"]
    if not os.path.exists(rtdetr_checkpoint_path):
        print(f"\nCreating RT-DETR with DINOv2 backbone at: {rtdetr_checkpoint_path}")
        os.makedirs(rtdetr_checkpoint_path, exist_ok=True)

        # Load label mapping
        id2label = {int(k): v for k, v in model_config["label_map"].items()}
        label2id = {v: k for k, v in id2label.items()}

        # Load pretrained RT-DETR
        # from transformers import RTDetrV2ForObjectDetection, RTDetrV2Config
        pretrained_rt_detr = RTDetrV2ForObjectDetection.from_pretrained(
            model_config["rtdetr_pretrained"],
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        # check the backnone output dims
        print(
            "Pretrained models encoder input dims: ",
            [
                pretrained_rt_detr.model.encoder_input_proj[i][0].weight.shape[1]
                for i in range(len(pretrained_rt_detr.model.encoder_input_proj))
            ],
        )
        print(
            "Pretrained config encoder input dims: ",
            pretrained_rt_detr.model.backbone.intermediate_channel_sizes,
        )

        # Load custom DINOv2 backbone with FPN from the initial checkpoint
        dinov2_backbone_config = Dinov2BackBoneWithFPNConfig.from_pretrained(
            backbone_checkpoint_path
        )
        print(
            f"DINOv2 backbone: {dinov2_backbone_config.dinov2_pretrained_backbone_name_or_path}"
        )

        # Load backbone
        dinov2_backbone = Dinov2BackBoneWithFPN.from_pretrained(
            backbone_checkpoint_path
        )

        # update RT-DETR's config with custom backbone (FPN)
        pretrained_model_config_dict = pretrained_rt_detr.config.to_dict()
        rt_detr_config = RTDetrV2ConfigWithCustomBackBone(
            **pretrained_model_config_dict
        )
        rt_detr_config.backbone_config = dinov2_backbone_config

        # Replace backbone and save
        pretrained_rt_detr.config = rt_detr_config
        pretrained_rt_detr.model.backbone = dinov2_backbone
        pretrained_rt_detr.save_pretrained(rtdetr_checkpoint_path)

        print(f"✓ RT-DETR with DINOv2 backbone saved to: {rtdetr_checkpoint_path}")
    else:
        print(f"✓ RT-DETR checkpoint already exists at: {rtdetr_checkpoint_path}")

    print("\n" + "=" * 80)
    print("Initial checkpoint creation complete!")
    print("=" * 80 + "\n")

    return rtdetr_checkpoint_path


def setup_model(config: Dict[str, Any]) -> RTDETRLightningModule:
    """Setup the RT-DETR model with DINOv2 backbone."""
    model_config = config["model"]
    training_config = config["training"]
    checkpoint_config = config["checkpointing"]
    init_config = config["initialization"]

    # Create or load initial checkpoint
    if init_config["create_initial_checkpoint"]:
        model_checkpoint_path = create_initial_checkpoint(config)
    else:
        model_checkpoint_path = checkpoint_config["rtdetr_initial_checkpoint"]
        if not os.path.exists(model_checkpoint_path):
            print(f"WARNING: Checkpoint not found at {model_checkpoint_path}")
            print("Creating initial checkpoint...")
            model_checkpoint_path = create_initial_checkpoint(config)

    # Load the model
    print(f"\nLoading RT-DETR model from: {model_checkpoint_path}")
    model = RTDetrV2ForObjectDetectionWithCustomBackbone.from_pretrained(
        model_checkpoint_path
    )

    # Setup image processor
    processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
    processor.do_normalize = True
    processor.resample = 3
    processor.size = {
        "height": config["dataset"]["model_input_size"],
        "width": config["dataset"]["model_input_size"],
    }

    # Load validation COCO ground truth
    val_annot_path = os.path.join(config["dataset"]["path"], "images", "valid")
    val_json_path = os.path.join(config["dataset"]["path"], "valid_annotations.json")
    val_coco_dataset = CocoDetection(
        root=val_annot_path, annFile=val_json_path, transforms=None
    )
    val_coco_gt = val_coco_dataset.coco
    val_coco_gt.dataset["info"] = {}

    # Create Lightning module
    lightning_model = RTDETRLightningModule(
        model=model,
        image_processor=processor,
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        max_grad_norm=training_config["max_grad_norm"],
        warmup_steps=training_config["warmup_steps"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        detection_threshold=model_config["detection_threshold"],
        max_detections=model_config["max_detections"],
        val_coco_gt=val_coco_gt,
    )

    print("✓ Model loaded successfully")
    return lightning_model, processor


def setup_data(config: Dict[str, Any], processor) -> COCODataModule:
    """Setup the data module."""
    dataset_config = config["dataset"]
    training_config = config["training"]

    data_module = COCODataModule(
        dataset_path=dataset_config["path"],
        processor=processor,
        batch_size=training_config["batch_size"],
        num_workers=training_config["num_workers"],
        model_input_size=dataset_config["model_input_size"],
        min_random_scale=dataset_config["min_random_scale"],
        max_random_scale=dataset_config["max_random_scale"],
        p_noise=dataset_config["p_noise"],
        org_images_in_model_input_size=dataset_config["org_images_in_model_input_size"],
    )

    print(f"✓ Data module configured for: {dataset_config['path']}")
    return data_module


def setup_callbacks(config: Dict[str, Any]):
    """Setup training callbacks."""
    checkpoint_config = config["checkpointing"]

    callbacks = [
        # Model checkpointing
        ModelCheckpoint(
            dirpath=checkpoint_config["save_dir"],
            filename="rtdetr-{epoch:02d}-{val_map:.4f}",
            monitor=checkpoint_config["monitor"],
            mode=checkpoint_config["mode"],
            save_top_k=checkpoint_config["save_top_k"],
            save_last=checkpoint_config["save_last"],
            every_n_epochs=checkpoint_config["every_n_epochs"],
            verbose=True,
        ),
        # Learning rate monitoring
        LearningRateMonitor(logging_interval="epoch"),
        ModelSummary(max_depth=2),
    ]

    print("✓ Callbacks configured")
    return callbacks


def setup_logger(config: Dict[str, Any]):
    """Setup WandB logger."""
    wandb_config = config["wandb"]

    if not wandb_config["enabled"]:
        print("✓ WandB logging disabled")
        return None

    logger = WandbLogger(
        project=wandb_config["project"],
        name=wandb_config["name"],
        tags=wandb_config["tags"],
        notes=wandb_config["notes"],
        config=config,  # Log the full configuration
    )

    print(f"✓ WandB logger configured - Project: {wandb_config['project']}")
    return logger


def main():
    parser = argparse.ArgumentParser(
        description="Train RT-DETR with DINOv2 backbone on COCO format dataset"
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rt_detr_dinov2_config.yaml",
        help="Path to configuration file",
    )

    # Dataset
    parser.add_argument("--dataset_path", type=str, help="Path to dataset directory")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument(
        "--num_workers", type=int, help="Number of data loading workers"
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint_dir", type=str, help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--resume_from", type=str, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--create_initial_checkpoint",
        action="store_true",
        help="Create initial checkpoint with DINOv2 backbone",
    )

    # WandB
    parser.add_argument("--wandb_project", type=str, help="WandB project name")
    parser.add_argument("--wandb_name", type=str, help="WandB run name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Load and merge configurations
    print("\n" + "=" * 80)
    print("RT-DETR Training with DINOv2 Backbone")
    print("=" * 80 + "\n")

    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    config = merge_configs(config, args)

    print(f"Dataset: {config['dataset']['path']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Epochs: {config['training']['num_epochs']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print()

    # Set seed
    pl.seed_everything(args.seed, workers=True)

    # Setup components
    model, processor = setup_model(config)
    data_module = setup_data(config, processor)
    callbacks = setup_callbacks(config)
    logger = setup_logger(config)

    # Create trainer
    trainer_config = config["trainer"]
    trainer = pl.Trainer(
        accelerator=trainer_config["accelerator"],
        devices=trainer_config["devices"],
        precision=trainer_config["precision"],
        strategy=trainer_config["strategy"],
        max_epochs=trainer_config["max_epochs"],
        log_every_n_steps=trainer_config["log_every_n_steps"],
        val_check_interval=trainer_config["val_check_interval"],
        gradient_clip_val=trainer_config["gradient_clip_val"],
        accumulate_grad_batches=trainer_config["accumulate_grad_batches"],
        deterministic=trainer_config["deterministic"],
        benchmark=trainer_config["benchmark"],
        callbacks=callbacks,
        logger=logger,
        # limit_train_batches = trainer_config['limit_train_batches'],
        # limit_val_batches = trainer_config['limit_val_batches'],
        fast_dev_run=trainer_config["fast_dev_run"],
    )

    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80 + "\n")

    # Start training
    ckpt_path = config["initialization"].get("load_from_checkpoint")
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80 + "\n")

    # Run test evaluation
    if trainer.checkpoint_callback.best_model_path:
        print(f"Best model: {trainer.checkpoint_callback.best_model_path}")
        print(f"Best val_map: {trainer.checkpoint_callback.best_model_score:.4f}")

        print("\nRunning test evaluation...")
        trainer.test(model, datamodule=data_module, ckpt_path="best")


if __name__ == "__main__":
    main()
