#!/usr/bin/env python3

import os
import hydra
from omegaconf import DictConfig
from ultralytics import YOLO
from utils.yolo_utils import convert_coco_to_yolo_seg, create_data_yaml, visualize_predictions
from utils.distributed_utils import setup_cluster_env, get_rank

setup_cluster_env()

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    yolo_cfg = config.model.yolov26
    data_cfg = config.data
    dp = data_cfg.path
    cache_dir = yolo_cfg.dataset_cache_dir
    
    train_fraction = yolo_cfg.get("train_fraction", 1.0)
    val_fraction = yolo_cfg.get("val_fraction", 1.0)
    if train_fraction != 1.0 or val_fraction != 1.0:
        cache_dir = f"{cache_dir}_train_{train_fraction}_val_{val_fraction}"
    
    yaml_path = os.path.join(cache_dir, "data.yaml")
    if get_rank() == 0:
        if not os.path.exists(yaml_path):
            print(f"[INFO] YOLO dataset not found. Generating cached dataset at {cache_dir}...")
            # 1. Convert Data to YOLO format
            convert_coco_to_yolo_seg(
                data_cfg.train_name,
                os.path.join(dp, "images", data_cfg.train_name),
                os.path.join(dp, f"{data_cfg.train_name}_annotations.json"),
                cache_dir,
                config.model.label_map,
                fraction=train_fraction
            )
            convert_coco_to_yolo_seg(
                data_cfg.val_name,
                os.path.join(dp, "images", data_cfg.val_name),
                os.path.join(dp, f"{data_cfg.val_name}_annotations.json"),
                cache_dir,
                config.model.label_map,
                fraction=val_fraction
            )
            convert_coco_to_yolo_seg(
                data_cfg.test_name,
                os.path.join(dp, "images", data_cfg.test_name),
                os.path.join(dp, f"{data_cfg.test_name}_annotations.json"),
                cache_dir,
                config.model.label_map,
                fraction=1.0 # Test set is always full
            )
            
            create_data_yaml(
                cache_dir, 
                data_cfg.train_name, 
                data_cfg.val_name, 
                data_cfg.test_name, 
                config.model.label_map
            )
        else:
            print(f"[INFO] Using cached YOLO dataset from {cache_dir}")
    
    # Wait for Rank 0 to finish creating the data.yaml
    if get_rank() > 0:
        import time
        while not os.path.exists(yaml_path):
            time.sleep(1)
    
    # Initialize WandB if enabled in config
    if config.logging.wandb.enabled and get_rank() == 0:
        import wandb
        from omegaconf import OmegaConf
        wandb.init(
            project=config.logging.wandb.project,
            name=config.run_name,
            tags=list(config.logging.wandb.tags),
            notes=config.logging.wandb.notes,
            config=OmegaConf.to_container(config, resolve=False),
        )

    # 2. Train YOLO
    model = YOLO(yolo_cfg.weights)
    
    # Custom callback to duplicate metrics for WandB
    def on_fit_epoch_end(trainer):
        if get_rank() == 0:
            if config.logging.wandb.enabled:
                metrics = trainer.metrics
                # Map Segmentation metrics
                if 'metrics/mAP50-95(M)' in metrics:
                    metrics['val/map'] = metrics['metrics/mAP50-95(M)']
                    metrics['val/map_ema'] = metrics['metrics/mAP50-95(M)']
                    metrics['val/map_segm'] = metrics['metrics/mAP50-95(M)']
                if 'metrics/mAP50(M)' in metrics:
                    metrics['val/map_50'] = metrics['metrics/mAP50(M)']
                    metrics['val/map_50_ema'] = metrics['metrics/mAP50(M)']
                    
                # Map Detection metrics too
                if 'metrics/mAP50-95(B)' in metrics:
                    metrics['val/map_box'] = metrics['metrics/mAP50-95(B)']
                if 'metrics/mAP50(B)' in metrics:
                    metrics['val/map_50_box'] = metrics['metrics/mAP50(B)']
            
            # Print detailed class-wise metrics
            if hasattr(trainer, "validator") and trainer.validator:
                val = trainer.validator
                print(f"\n[Epoch {trainer.epoch + 1}] Detailed Class-wise Metrics:")
                orig_training = getattr(val, 'training', True)
                val.training = False
                val.print_results()
                val.training = orig_training
            
            # NOTE: Custom visualizations during DDP training crash the reducer.
            # Only visualize during test_only mode or post-training.
            # base_save_dir = os.path.join(yolo_cfg.project, yolo_cfg.name)
            # visualize_predictions(model, config, trainer.epoch, split="val", base_save_dir=base_save_dir)
                    
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    # Move custom callback to the front so it executes BEFORE the native WandB logger
    if "on_fit_epoch_end" in model.callbacks:
        model.callbacks["on_fit_epoch_end"].insert(0, model.callbacks["on_fit_epoch_end"].pop())
    
    # Convert hyp DictConfig to dict
    hyp_dict = dict(yolo_cfg.hyp) if "hyp" in yolo_cfg else {}
    
    if getattr(config, "test_only", False):
        print(f"[INFO] Running in validation mode with weights: {yolo_cfg.weights}")
        metrics = model.val(
            data=yaml_path,
            split='val',
            batch=yolo_cfg.batch_size,
            imgsz=config.model.input_size,
            project=yolo_cfg.project,
            name=yolo_cfg.name + "_val",
            task=yolo_cfg.get("task", "segment"),
            **hyp_dict
        )
        if get_rank() == 0:
            print("\n--- Validation Completed ---")
            for k, v in metrics.results_dict.items():
                print(f"{k}: {v:.4f}")
            print("----------------------------\n")
            weights_path = os.path.abspath(yolo_cfg.weights)
            base_save_dir = os.path.dirname(os.path.dirname(weights_path))
            visualize_predictions(model, config, None, split="val", base_save_dir=base_save_dir)
    else:
        model.train(
            data=yaml_path,
            epochs=yolo_cfg.epochs,
            batch=yolo_cfg.batch_size,
            imgsz=config.model.input_size,
            project=yolo_cfg.project,
            name=yolo_cfg.name,
            task=yolo_cfg.get("task", "segment"),
            **hyp_dict
        )
        
        # Cleanup DDP process group to prevent deadlocks on teardown
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
