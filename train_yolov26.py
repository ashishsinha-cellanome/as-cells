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
    
    yaml_path = os.path.join(cache_dir, "data.yaml")
    if get_rank() == 0:
        # 1. Convert Data to YOLO format
        convert_coco_to_yolo(
            data_cfg.train_name,
            os.path.join(dp, "images", data_cfg.train_name),
            os.path.join(dp, f"{data_cfg.train_name}_annotations.json"),
            cache_dir,
            config.model.label_map
        )
        convert_coco_to_yolo(
            data_cfg.val_name,
            os.path.join(dp, "images", data_cfg.val_name),
            os.path.join(dp, f"{data_cfg.val_name}_annotations.json"),
            cache_dir,
            config.model.label_map
        )
        convert_coco_to_yolo(
            data_cfg.test_name,
            os.path.join(dp, "images", data_cfg.test_name),
            os.path.join(dp, f"{data_cfg.test_name}_annotations.json"),
            cache_dir,
            config.model.label_map
        )
        
        create_data_yaml(
            cache_dir, 
            data_cfg.train_name, 
            data_cfg.val_name, 
            data_cfg.test_name, 
            config.model.label_map
        )
    
    # Wait for Rank 0 to finish creating the data.yaml
    if get_rank() > 0:
        import time
        while not os.path.exists(yaml_path):
            time.sleep(1)
    
    # 2. Train YOLO
    model = YOLO(yolo_cfg.weights)
    
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
