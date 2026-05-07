#!/usr/bin/env python3
import faulthandler
import signal
faulthandler.register(signal.SIGUSR1)
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
        remap_category_ids=False,
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
                "finetune_mode": "full",
                "pretrain_weights": "rf-detr-seg-large.pt",
            }
        },
        "data": {
            "path": "/mnt/direct-attached/TRAINING_DATA",
        },
        "trainer": {
            "batch_size": 16,
            "grad_accum_steps": 1,
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
    def after_save_checkpoint(self, *args, **kwargs): pass
    @property
    def name(self): return "dummy"
    @property
    def version(self): return "0"
    @property
    def save_dir(self): return "output"

rfdetr.training.trainer.CSVLogger = DummyCSVLogger

# 3. Inject Custom DetailedCocoEvalCallback and kwargs
import pytorch_lightning as pl
import sys
orig_trainer_init = pl.Trainer.__init__

def custom_trainer_init(self, *args, **kwargs):
    from utils.detailed_coco_eval import DetailedCocoEvalCallback
    if 'callbacks' in kwargs and kwargs['callbacks'] is not None:
        kwargs['callbacks'].append(DetailedCocoEvalCallback())
        
    # Inject fraction logic globally since we can't easily pass it through RFDETR's config
    fraction = 1.0
    test_only = False
    for i, arg in enumerate(sys.argv):
        if arg == '--fraction' and i + 1 < len(sys.argv):
            fraction = float(sys.argv[i + 1])
        if arg == '--test_only':
            test_only = True
            
    self.test_only = test_only
    if fraction < 1.0:
        kwargs['limit_train_batches'] = fraction
        kwargs['limit_val_batches'] = fraction
        
    orig_trainer_init(self, *args, **kwargs)

pl.Trainer.__init__ = custom_trainer_init

orig_fit = pl.Trainer.fit
def custom_fit(self, model, datamodule=None, ckpt_path=None):
    if getattr(self, "test_only", False):
        print("Running validation ONLY (test_only mode).")
        # Remove original COCOEvalCallback to prevent massive memory leak from torchmetrics keeping masks in RAM
        self.callbacks = [cb for cb in self.callbacks if type(cb).__name__ != 'COCOEvalCallback']
        
        if ckpt_path and ckpt_path.endswith('.pth'):
            import torch
            print(f"Manually loading raw weights from {ckpt_path} to avoid PTL validate loop errors")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if "state_dict" in ckpt:
                model.load_state_dict(ckpt["state_dict"], strict=False)
            elif "model" in ckpt:
                model.load_state_dict(ckpt["model"], strict=False)
            ckpt_path = None
        
        # Measure GFLOPS and Throughput BEFORE validation so we see it immediately
        import time
        import torch
        
        if self.is_global_zero:
            try:
                from fvcore.nn import FlopCountAnalysis, flop_count_table
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                if hasattr(model, 'model'):
                    inner_model = model.model
                else:
                    inner_model = model
                    
                inner_model.to(device)
                inner_model.eval()
                
                datamodule.setup(stage="validate")
                dl = datamodule.val_dataloader()
                batch = next(iter(dl))
                images, targets = batch
                images = images.to(device)
                
                # NestedTensor handling
                if hasattr(images, 'tensors'):
                    tensor_images = images.tensors
                else:
                    tensor_images = images
                    
                print(f"Input shape for profiling: {tensor_images.shape}")
                
                try:
                    flops = FlopCountAnalysis(inner_model, images)
                    print(flop_count_table(flops))
                    print(f"Total GFLOPS: {flops.total() / 1e9:.2f}")
                except Exception as e:
                    print(f"Could not compute GFLOPS directly with NestedTensor, trying raw tensors... {e}")
                    try:
                        flops = FlopCountAnalysis(inner_model, tensor_images)
                        print(flop_count_table(flops))
                        print(f"Total GFLOPS: {flops.total() / 1e9:.2f}")
                    except Exception as e2:
                        print(f"Failed again: {e2}")
                    
                print("Warming up for throughput measurement...")
                with torch.no_grad():
                    for _ in range(5):
                        _ = inner_model(images)
                        
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.time()
                num_runs = 50
                with torch.no_grad():
                    for _ in range(num_runs):
                        _ = inner_model(images)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.time()
                
                total_time = end - start
                time_per_batch = total_time / num_runs
                time_per_image = time_per_batch / tensor_images.shape[0]
                
                print(f"Inference time per batch ({tensor_images.shape[0]} images): {time_per_batch*1000:.2f} ms")
                print(f"Inference time per image: {time_per_image*1000:.2f} ms")
                print(f"Throughput: {1.0 / time_per_image:.2f} FPS")
            except ImportError:
                print("fvcore not installed, skipping GFLOPS calculation.")
                
        self.validate(model, datamodule=datamodule, ckpt_path=ckpt_path)
        
    else:
        orig_fit(self, model, datamodule=datamodule, ckpt_path=ckpt_path)

pl.Trainer.fit = custom_fit

# 4. Monkey-patch COCOEvalCallback to limit predictions
import rfdetr.training.callbacks.coco_eval as orig_coco_eval
import torch

orig_convert_preds = orig_coco_eval.COCOEvalCallback._convert_preds

def custom_convert_preds(self, preds):
    out = orig_convert_preds(self, preds)
    for p in out:
        if "scores" in p and len(p["scores"]) > 100:
            topk = torch.topk(p["scores"], 100)
            indices = topk.indices
            p["scores"] = p["scores"][indices]
            p["labels"] = p["labels"][indices]
            p["boxes"] = p["boxes"][indices]
            if "masks" in p:
                p["masks"] = p["masks"][indices]
    return out

orig_coco_eval.COCOEvalCallback._convert_preds = custom_convert_preds

import rfdetr.training.callbacks.ema as orig_ema

orig_ema_load_state_dict = orig_ema.RFDETREMACallback.load_state_dict

def custom_ema_load_state_dict(self, state_dict):
    orig_ema_load_state_dict(self, state_dict)
    if getattr(self, "_average_model", None) is not None and getattr(self, "_pending_average_state_dict", None) is not None:
        self._average_model.load_state_dict(self._pending_average_state_dict)
        self._pending_average_state_dict = None

orig_ema.RFDETREMACallback.load_state_dict = custom_ema_load_state_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="/mnt/direct-attached/TRAINING_DATA", help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of dataset to use for training/val")
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--test_only", action="store_true", help="Only run validation/test on the validation set")
    args = parser.parse_args()

    print(f"Initializing RFDETRSegLarge...")
    model = RFDETRSegLarge(group_detr=1, compile=True, num_classes=4)
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"rf_detr_seg_large_{timestamp}"
    
    print(f"Starting training on {args.dataset_dir} for {args.epochs} epochs with fraction {args.fraction}...")
    
    trainer_kwargs = {}
    if args.debug:
        trainer_kwargs["limit_train_batches"] = 1
        trainer_kwargs["limit_val_batches"] = 1
        trainer_kwargs["limit_test_batches"] = 0
        args.epochs = 1
    elif args.fraction < 1.0:
        trainer_kwargs["limit_train_batches"] = args.fraction
        trainer_kwargs["limit_val_batches"] = args.fraction
        
    if args.resume:
        trainer_kwargs["resume"] = args.resume
        
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
        early_stopping=True,
        progress_bar="rich",
        num_workers=args.num_workers,
        use_ema=True,
        **trainer_kwargs
    )
    
if __name__ == "__main__":
    main()
