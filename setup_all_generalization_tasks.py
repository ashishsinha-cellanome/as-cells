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


def write_hydra_yaml(yaml_name, train_datasets, test_datasets, run_prefix):
    yaml_path = Path("configs/data") / f"{yaml_name}.yaml"
    
    # Map the lists to actual absolute paths in /mnt/direct-attached/PHASE2
    base_phase2 = "/mnt/direct-attached/PHASE2"
    train_paths = [f"{base_phase2}/{ds}" for ds in train_datasets]
    test_paths = [f"{base_phase2}/{ds}" for ds in test_datasets]
    
    # We will use the MultiDataset pattern for Hydra configs
    # If the project doesn't have a specific multidataset dataloader configured in the yaml natively,
    # we construct the lists.
    
    content = f"""# @package _global_

defaults:
  - default@data

data:
  datasets:
    train:
"""
    for p in train_paths:
        content += f"      - {p}\n"
        
    content += "    val:\n"
    for p in test_paths:
        content += f"      - {p}\n"
        
    content += "    test:\n"
    for p in test_paths:
        content += f"      - {p}\n"

    content += """
  val_name: 'test'
  test_name: 'test'
  train_name: 'train'

initialization:
  load_from_checkpoint: "/mnt/personal/cellanome/checkpoints/ALL_CKPTS/RFDETR-Seg-ckpts/output/checkpoint_best_ema.pth"

checkpointing:
  save_dir: "/project/aip-robsc/asinha/cellanome/DATA/checkpoints/"
  rtdetr_initial_checkpoint: "/project/aip-robsc/asinha/cellanome/DATA/checkpoints/${checkpointing._folder_name}"
  dinov2_backbone_checkpoint: "/project/aip-robsc/asinha/cellanome/DATA/checkpoints/backbones/dinov2"

hydra:
  run:
    dir: "/project/aip-robsc/asinha/cellanome/logs/${now:%Y-%m-%d}/${now:%H-%M-%S}_""" + run_prefix + "\"\n"

    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"Created config: {yaml_path}")


def main():
    # We no longer need to filter JSONs or symlink! 
    # The dataloader natively supports taking a list of dataset folders from PHASE2.
    
    experiments = {
        # Chain 1: Fibroblasts
        "c1_fibro_to_preadipo": {
            "train": ["20240509_Hs675Tfibroblasts_10x_caged_4_class"],
            "test": ["20241212_preadipocytes-adhered_10x_uncaged_4_class", "20250227_preadipocytes-adhered_10x_caged_4_class"]
        },
        "c1_preadipo_to_fibro": {
            "train": ["20241212_preadipocytes-adhered_10x_uncaged_4_class"],
            "test": ["20240509_Hs675Tfibroblasts_10x_caged_4_class", "231212_imr90_multichannel_overlay_4_class", "240213_imr90_multichannel_overlay_4_class", "20250227_preadipocytes-adhered_10x_caged_4_class"]
        },
        
        # Chain 2: Epithelial
        "c2_mc38caged_to_broad": {
            "train": ["20240624_mc38_10x_caged_4_class"],
            "test": ["20240624_mc38_10x_uncaged_4_class", "20240905_u87-adhered_10x_caged_4_class", "20240509_hela-adhered_10x_caged_4_class", "20250820_c8d1a_astrocytes-adherent_10x_caged_4_class"]
        },
        "c2_mc38uncaged_to_narrow": {
            "train": ["20240624_mc38_10x_uncaged_4_class"],
            "test": ["20240624_mc38_10x_caged_4_class", "20240625_mc38_10x_caged_4_class", "20240905_u87-adhered_10x_caged_4_class", "20240509_hela-adhered_10x_caged_4_class", "20250820_c8d1a_astrocytes-adherent_10x_caged_4_class"]
        },
        
        # Chain 3: Isolates (A549 to MOC22)
        "c3_mc38_to_moc22": {
            "train": ["20240624_mc38_10x_caged_4_class"],
            "test": ["20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class", "20250917_moc22-adhered_10x_caged_4_class"]
        },
        "c3_moc22_to_mc38": {
            "train": ["20250917_moc22-adhered_10x_caged_4_class"],
            "test": ["20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class", "20240624_mc38_10x_caged_4_class"]
        },
        
        # Chain 4: Dendritic Progression
        "c4_mc38_to_dc": {
            "train": ["20240624_mc38_10x_caged_4_class"],
            "test": ["20240515_DC-adhered_10x_caged_4_class", "20240516_DC-adhered_10x_caged_4_class"]
        },
        "c4_dc_to_mc38": {
            "train": ["20240515_DC-adhered_10x_caged_4_class"],
            "test": ["20240624_mc38_10x_caged_4_class", "20240516_DC-adhered_10x_caged_4_class"]
        },
        
        # 6-Centroids Final Generalization Test
        "centroids_to_heldout": {
            "train": [
                "20240624_mc38_10x_caged_4_class", 
                "20241212_preadipocytes-adhered_10x_uncaged_4_class", 
                "20240515_DC-adhered_10x_caged_4_class", 
                "20240905_u87-adhered_10x_caged_4_class", 
                "20250917_moc22-adhered_10x_caged_4_class", 
                "20240924_enteric-glia-adhered_10x_uncaged_4_class"
            ],
            "test": [
                "20240509_hela-adhered_10x_caged_4_class", 
                "20250820_c8d1a_astrocytes-adherent_10x_caged_4_class", 
                "20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class", 
                "20240509_Hs675Tfibroblasts_10x_caged_4_class", 
                "231212_imr90_multichannel_overlay_4_class", 
                "20240624_mc38_10x_uncaged_4_class"
            ]
        }
    }
    
    os.makedirs("configs/data", exist_ok=True)
    
    for exp_name, exp_data in experiments.items():
        write_hydra_yaml(exp_name, exp_data["train"], exp_data["test"], exp_name.upper())

if __name__ == "__main__":
    main()
