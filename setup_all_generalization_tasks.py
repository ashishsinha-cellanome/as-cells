#!/usr/bin/env python3
import json
import os
from pathlib import Path
import yaml

def filter_coco(input_json, output_json, dataset_keywords):
    if not os.path.exists(input_json):
        return
    with open(input_json, 'r') as f:
        coco = json.load(f)
        
    new_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco.get("categories", []),
        "images": [],
        "annotations": []
    }
    
    valid_image_ids = set()
    for img in coco["images"]:
        file_name = img["file_name"].lower()
        if any(kw.lower() in file_name for kw in dataset_keywords):
            new_coco["images"].append(img)
            valid_image_ids.add(img["id"])
            
    for ann in coco["annotations"]:
        if ann["image_id"] in valid_image_ids:
            new_coco["annotations"].append(ann)
            
    with open(output_json, 'w') as f:
        json.dump(new_coco, f)


def setup_experiment(base_data_dir, exp_dir_name, train_keywords, test_keywords):
    base = Path(base_data_dir)
    exp_dir = base.parent / exp_dir_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    images_src = base / "images"
    images_dst = exp_dir / "images"
    if images_src.exists() and not images_dst.exists():
        os.symlink(images_src, images_dst)
        
    masks_src = base / "masks"
    masks_dst = exp_dir / "masks"
    if masks_src.exists() and not masks_dst.exists():
        os.symlink(masks_src, masks_dst)
        
    filter_coco(base / "train_annotations.json", exp_dir / "train_annotations.json", train_keywords)
    filter_coco(base / "valid_no300_annotations.json", exp_dir / "valid_no300_annotations.json", test_keywords)
    filter_coco(base / "test_no300_annotations.json", exp_dir / "test_no300_annotations.json", test_keywords)
    
    return str(exp_dir)


def write_hydra_yaml(yaml_name, data_path, run_prefix):
    yaml_path = Path("configs/data") / f"{yaml_name}.yaml"
    
    content = f"""# @package _global_

defaults:
  - default@data

data:
  path: "{data_path}"
  val_name: 'valid_no300'
  test_name: 'test_no300'
  train_name: 'train'

checkpointing:
  save_dir: "/project/aip-robsc/asinha/cellanome/DATA/checkpoints/"
  rtdetr_initial_checkpoint: "/project/aip-robsc/asinha/cellanome/DATA/checkpoints/${{checkpointing._folder_name}}"
  dinov2_backbone_checkpoint: "/project/aip-robsc/asinha/cellanome/DATA/checkpoints/backbones/dinov2"

hydra:
  run:
    dir: "/project/aip-robsc/asinha/cellanome/logs/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}_{run_prefix}"
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"Created config: {yaml_path}")


def main():
    BASE_DATA_DIR = "/project/aip-robsc/asinha/cellanome/DATA/TRAINING_DATA"
    
    if not os.path.exists(BASE_DATA_DIR):
        print(f"Error: {BASE_DATA_DIR} not found. Script must be run on Vulcan.")
        # We will still generate the yaml files locally for convenience
    else:
        print(f"Found {BASE_DATA_DIR}. Will generate data splits and yamls.")

    # Define the core experiments based on the chains
    experiments = {
        # Chain 1: Fibroblasts
        "c1_fibro_to_preadipo": {
            "train": ["20240509_hs675tfibroblasts"],
            "test": ["20241212_preadipocytes", "20250227_preadipocytes"]
        },
        "c1_preadipo_to_fibro": {
            "train": ["20241212_preadipocytes"],
            "test": ["20240509_hs675tfibroblasts", "imr90", "20250227_preadipocytes"]
        },
        
        # Chain 2: Epithelial
        "c2_mc38caged_to_broad": {
            "train": ["20240624_mc38__caged"],
            "test": ["20240624_mc38__uncaged", "20240905_u87", "20240509_hela", "20250820_c8d1a_astrocytes"]
        },
        "c2_mc38uncaged_to_narrow": {
            "train": ["20240624_mc38__uncaged"],
            "test": ["20240624_mc38__caged", "20240625_mc38__caged", "20240905_u87", "20240509_hela", "20250820_c8d1a_astrocytes"]
        },
        
        # Chain 3: Isolates (A549 to MOC22)
        "c3_mc38_to_moc22": {
            "train": ["20240624_mc38__caged"],
            "test": ["20260316_a549", "20250917_moc22"]
        },
        "c3_moc22_to_mc38": {
            "train": ["20250917_moc22"],
            "test": ["20260316_a549", "20240624_mc38__caged"]
        },
        
        # Chain 4: Dendritic Progression
        "c4_mc38_to_dc": {
            "train": ["20240624_mc38__caged"],
            "test": ["20240515_dc", "20240516_dc"]
        },
        "c4_dc_to_mc38": {
            "train": ["20240515_dc"],
            "test": ["20240624_mc38__caged", "20240516_dc"]
        },
        
        # 6-Centroids Final Generalization Test
        "centroids_to_heldout": {
            "train": ["20240624_mc38__caged", "20241212_preadipocytes", "20240515_dc", "20240905_u87", "20250917_moc22", "20240924_enteric"],
            "test": ["20240509_hela", "20250820_c8d1a_astrocytes", "20260316_a549", "20240509_hs675tfibroblasts", "imr90", "20240624_mc38__uncaged"]
        }
    }
    
    os.makedirs("configs/data", exist_ok=True)
    
    for exp_name, exp_data in experiments.items():
        exp_dir_name = f"TRAINING_DATA_{exp_name.upper()}"
        
        # Generate the data on Vulcan if it exists
        if os.path.exists(BASE_DATA_DIR):
            print(f"\nProcessing {exp_name}...")
            data_path = setup_experiment(
                BASE_DATA_DIR, 
                exp_dir_name, 
                exp_data["train"], 
                exp_data["test"]
            )
        else:
            # Fake path for local YAML generation
            data_path = f"/project/aip-robsc/asinha/cellanome/DATA/{exp_dir_name}"
            
        write_hydra_yaml(exp_name, data_path, exp_name.upper())

if __name__ == "__main__":
    main()
