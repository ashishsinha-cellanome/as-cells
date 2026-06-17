#!/usr/bin/env python3
import faulthandler
import signal
faulthandler.register(signal.SIGUSR1)
import argparse
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
        "val": (root / "images" / "val", root / "val_annotations.json"),
        "test": (root / "images" / "test", root / "test_annotations.json"),
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
    import os
    
    if 'callbacks' in kwargs and kwargs['callbacks'] is not None:
        kwargs['callbacks'].append(DetailedCocoEvalCallback())
        
    # Inject fraction and test_only logic globally
    fraction = float(os.environ.get("RFDETR_FRACTION", "1.0"))
    test_only = os.environ.get("RFDETR_TEST_ONLY") == "1"
            
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
        
        import os
        manual_load_path = os.environ.get('TEST_ONLY_MANUAL_LOAD_PATH')
        if manual_load_path:
            import torch
            print(f"[Rank {self.global_rank}] Manually loading raw weights from {manual_load_path} to avoid PTL validate loop errors")
            ckpt = torch.load(manual_load_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
            model.load_state_dict(state_dict, strict=False)
            
            # Setup EMA callback if it exists to ensure both regular and EMA are evaluated
            ema_cb = None
            for cb in self.callbacks:
                if type(cb).__name__ == 'RFDETREMACallback':
                    ema_cb = cb
                    break
                    
            if ema_cb:
                print(f"[Rank {self.global_rank}] Initializing EMA model for dual evaluation")
                from torch.optim.swa_utils import AveragedModel
                # The model is on CPU here, PTL moves it later. We will patch the EMA callback to move it.
                ema_cb._average_model = AveragedModel(
                    model=model,
                    device=torch.device('cpu'),
                    use_buffers=ema_cb._use_buffers,
                    avg_fn=ema_cb._avg_fn,
                )
                ema_cb._average_model.eval()
                
                ema_state_dict = None
                if 'callbacks' in ckpt and 'RFDETREMACallback' in ckpt['callbacks']:
                    ema_state_dict = ckpt['callbacks']['RFDETREMACallback'].get('average_model_state_dict')
                    
                if ema_state_dict is not None:
                    ema_cb._average_model.load_state_dict(ema_state_dict)
                    print(f"[Rank {self.global_rank}] Successfully loaded distinct EMA weights from checkpoint")
                else:
                    print(f"[Rank {self.global_rank}] No distinct EMA weights found. EMA metrics will mirror regular metrics.")
                
                # Patch on_validation_start to ensure the EMA model gets moved to the correct device
                orig_on_val_start = ema_cb.on_validation_start
                def patched_on_val_start(trainer, pl_module):
                    if hasattr(ema_cb, '_average_model') and ema_cb._average_model is not None:
                        ema_cb._average_model.to(pl_module.device)
                    if callable(orig_on_val_start):
                        orig_on_val_start(trainer, pl_module)
                ema_cb.on_validation_start = patched_on_val_start
            
        # Measure GFLOPS and Throughput BEFORE validation so we see it immediately
        import time
        import torch
        
        if self.is_global_zero and not getattr(self, "_gflops_measured", False):
            self._gflops_measured = True
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
                    print(f"Total GFLOPS per batch ({tensor_images.shape[0]} images): {flops.total() / 1e9:.2f}")
                    print(f"Total GFLOPS per image: {(flops.total() / 1e9) / tensor_images.shape[0]:.2f}")
                except Exception as e:
                    print(f"Could not compute GFLOPS directly with NestedTensor, trying raw tensors... {e}")
                    try:
                        flops = FlopCountAnalysis(inner_model, tensor_images)
                        print(flop_count_table(flops))
                        print(f"Total GFLOPS per batch ({tensor_images.shape[0]} images): {flops.total() / 1e9:.2f}")
                        print(f"Total GFLOPS per image: {(flops.total() / 1e9) / tensor_images.shape[0]:.2f}")
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

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

@hydra.main(config_path="configs", config_name="config_rfdetr_seg.yaml", version_base=None)
def main(config: DictConfig):
    import datetime
    import os
    import sys
    
    dataset_dir = to_absolute_path(config.data.path)
    epochs = config.trainer.max_epochs
    batch_size = config.data.batch_size
    grad_accum_steps = config.trainer.accumulate_grad_batches
    devices = config.trainer.devices
    num_workers = config.data.num_workers
    debug = config.debug
    
    fraction = config.data.get("limit_train_batches", 1.0)
    if isinstance(fraction, int) and fraction > 1:
        fraction = 1.0
        
    test_only = config.get("test_only", False)
    
    os.environ["RFDETR_FRACTION"] = str(fraction)
    if test_only:
        os.environ["RFDETR_TEST_ONLY"] = "1"
        
    weights = config.model.rfdetr.get("pretrain_weights", None)
    resume = None
    
    if config.initialization.get("load_from_checkpoint"):
        if test_only:
            resume = to_absolute_path(config.initialization.load_from_checkpoint)
        else:
            weights = to_absolute_path(config.initialization.load_from_checkpoint)

    print("Initializing RFDETRSegLarge...")
    if weights:
        model = RFDETRSegLarge(group_detr=1, compile=True, num_classes=4, pretrain_weights=weights)
        print(f"Loaded custom pretrain weights from: {weights}")
    else:
        model = RFDETRSegLarge(group_detr=1, compile=True, num_classes=4)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = config.get("run_name", f"rf_detr_seg_large_{timestamp}")
    if debug:
        run_name = f"DEBUG_{run_name}"
    
    print(f"Starting training on {dataset_dir} for {epochs} epochs with fraction {fraction}...")
    
    trainer_kwargs = {}
    if debug:
        os.environ["RFDETR_FRACTION"] = "1.0"
        trainer_kwargs["limit_train_batches"] = 1
        trainer_kwargs["limit_val_batches"] = 1
        trainer_kwargs["limit_test_batches"] = 0
        epochs = 1
        
    if test_only and resume and resume.endswith('.pth'):
        os.environ['TEST_ONLY_MANUAL_LOAD_PATH'] = resume
        resume = None

    if resume:
        trainer_kwargs["resume"] = resume
        
    model.train(
        dataset_dir=dataset_dir,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        devices=devices,
        strategy="auto" if devices == 1 else "ddp_find_unused_parameters_true",
        wandb=False if debug else True,
        project="cell-detection",
        run=run_name,
        early_stopping=True,
        progress_bar="rich",
        num_workers=num_workers,
        use_ema=True,
        **trainer_kwargs
    )
    
if __name__ == "__main__":
    main()
