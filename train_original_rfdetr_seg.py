#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

# 1. Monkey-patch the rfdetr dataset builder to support the custom directory layout
import rfdetr.datasets
import rfdetr.datasets.coco

# Bypass format detection to always assume coco
rfdetr.datasets.__dict__['detect_roboflow_format'] = lambda x: "coco"

def custom_build_roboflow_from_coco(image_set, args, resolution):
    from rfdetr.datasets.coco import CocoDetection, make_coco_transforms, make_coco_transforms_square_div_64
    from rfdetr.utilities.logger import get_logger
    logger = get_logger()
    root = Path(args.dataset_dir)
    
    PATHS = {
        "train": (root / "images" / "train", root / "train_annotations.json"),
        "val": (root / "images" / "valid_no300", root / "valid_no300_annotations.json"),
        "test": (root / "images" / "test_no300", root / "test_no300_annotations.json"),
    }
    
    # Identify split ('train', 'val', or 'test')
    split_key = image_set.split("_")[0]
    img_folder, ann_file = PATHS[split_key]
    
    square_resize_div_64 = getattr(args, "square_resize_div_64", False)
    include_masks = getattr(args, "segmentation_head", False)
    multi_scale = getattr(args, "multi_scale", False)
    expanded_scales = getattr(args, "expanded_scales", False)
    do_random_resize_via_padding = getattr(args, "do_random_resize_via_padding", False)
    patch_size = getattr(args, "patch_size", 16)
    num_windows = getattr(args, "num_windows", 4)
    aug_config = getattr(args, "aug_config", None)

    transform_fn = make_coco_transforms_square_div_64 if square_resize_div_64 else make_coco_transforms

    dataset = CocoDetection(
        img_folder,
        ann_file,
        transforms=transform_fn(
            image_set,
            resolution,
            multi_scale=multi_scale,
            expanded_scales=expanded_scales,
            skip_random_resize=not do_random_resize_via_padding,
            patch_size=patch_size,
            num_windows=num_windows,
            aug_config=aug_config,
        ),
        include_masks=include_masks,
        remap_category_ids=True,
    )
    return dataset

# Apply the patches
rfdetr.datasets.coco.build_roboflow_from_coco = custom_build_roboflow_from_coco
rfdetr.datasets.build_roboflow_from_coco = custom_build_roboflow_from_coco

# 2. Setup training
from rfdetr import RFDETRSegLarge
import pytorch_lightning.loggers as pl_loggers

orig_wandb_logger = pl_loggers.WandbLogger

def custom_wandb_logger(*args, **kwargs):
    config = {
        "model": {
            "name": "rf_detr_seg_large",
            "rfdetr": {
                "size": "large",
                "finetune_mode": "queries_decoder_head",
                "pretrain_weights": "rf-detr-seg-large.pt",
            }
        },
        "data": {
            "path": "/mnt/direct-attached/TRAINING_DATA",
        },
        "trainer": {
            "batch_size": 2,
            "grad_accum_steps": 8,
            "max_epochs": 50,
        }
    }
    kwargs['config'] = config
    return orig_wandb_logger(*args, **kwargs)

import rfdetr.training.trainer
rfdetr.training.trainer.WandbLogger = custom_wandb_logger

class DummyCSVLogger:
    def __init__(self, *args, **kwargs): pass
    def save(self): pass
    def log_metrics(self, *args, **kwargs): pass
    def log_hyperparams(self, *args, **kwargs): pass
    def finalize(self, *args, **kwargs): pass
    def log_graph(self, *args, **kwargs): pass
    @property
    def name(self): return "dummy"
    @property
    def version(self): return "0"
    @property
    def save_dir(self): return "output"

rfdetr.training.trainer.CSVLogger = DummyCSVLogger

# 3. Inject Custom DetailedCocoEvalCallback
import pytorch_lightning as pl
orig_trainer_init = pl.Trainer.__init__

def custom_trainer_init(self, *args, **kwargs):
    from utils.detailed_coco_eval import DetailedCocoEvalCallback
    if 'callbacks' in kwargs and kwargs['callbacks'] is not None:
        kwargs['callbacks'].append(DetailedCocoEvalCallback())
    orig_trainer_init(self, *args, **kwargs)

pl.Trainer.__init__ = custom_trainer_init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="/mnt/direct-attached/TRAINING_DATA", help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--devices", type=int, default=4)
    args = parser.parse_args()

    print(f"Initializing RFDETRSegLarge...")
    model = RFDETRSegLarge(group_detr=1)
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"rf_detr_seg_large_{timestamp}"
    
    print(f"Starting training on {args.dataset_dir} for {args.epochs} epochs...")
    
    trainer_kwargs = {}
    if args.debug:
        trainer_kwargs["limit_train_batches"] = 1
        trainer_kwargs["limit_val_batches"] = 1
        trainer_kwargs["limit_test_batches"] = 0
        args.epochs = 1
        
    model.train(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        devices=args.devices,
        strategy="auto" if args.devices == 1 else "ddp_find_unused_parameters_true",
        wandb=False if args.debug else True,
        project="cell-detection",
        run=run_name,
        **trainer_kwargs
    )
    
if __name__ == "__main__":
    main()
