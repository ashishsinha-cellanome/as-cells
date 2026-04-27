#!/usr/bin/env python3

import os
import hydra
from omegaconf import DictConfig
from ultralytics import YOLO
from utils.yolo_utils import convert_coco_to_yolo, create_data_yaml
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
            convert_coco_to_yolo(
                data_cfg.train_name,
                os.path.join(dp, "images", data_cfg.train_name),
                os.path.join(dp, f"{data_cfg.train_name}_annotations.json"),
                cache_dir,
                config.model.label_map,
                fraction=train_fraction
            )
            convert_coco_to_yolo(
                data_cfg.val_name,
                os.path.join(dp, "images", data_cfg.val_name),
                os.path.join(dp, f"{data_cfg.val_name}_annotations.json"),
                cache_dir,
                config.model.label_map,
                fraction=val_fraction
            )
            convert_coco_to_yolo(
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
        if config.logging.wandb.enabled and get_rank() == 0:
            import wandb
            if wandb.run is not None:
                custom_metrics = {}
                metrics = trainer.metrics
                
                if 'metrics/mAP50-95(B)' in metrics:
                    custom_metrics['val/map'] = metrics['metrics/mAP50-95(B)']
                    custom_metrics['val/map_ema'] = metrics['metrics/mAP50-95(B)']
                if 'metrics/mAP50(B)' in metrics:
                    custom_metrics['val/map_50'] = metrics['metrics/mAP50(B)']
                    custom_metrics['val/map_50_ema'] = metrics['metrics/mAP50(B)']
                    
                if custom_metrics:
                    wandb.log(custom_metrics, commit=False)
    
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    
    # Convert hyp DictConfig to dict
    hyp_dict = dict(yolo_cfg.hyp) if "hyp" in yolo_cfg else {}
    
    model.train(
        data=yaml_path,
        epochs=yolo_cfg.epochs,
        batch=yolo_cfg.batch_size,
        imgsz=config.model.input_size,
        project=yolo_cfg.project,
        name=yolo_cfg.name,
        **hyp_dict
    )

if __name__ == "__main__":
    main()
