#!/usr/bin/env python3
import json
import os
from pathlib import Path

def merge_coco_datasets(dataset_names, split_name, output_dir, base_phase2):
    output_dir = Path(output_dir)
    out_images_dir = output_dir / "images" / split_name
    out_images_dir.mkdir(parents=True, exist_ok=True)
    
    out_masks_dir = output_dir / "masks" / split_name
    out_masks_dir.mkdir(parents=True, exist_ok=True)
    
    merged_coco = {
        "info": {},
        "licenses": [],
        "categories": [{"id": 0, "name": "cell"}, {"id": 1, "name": "bead"}, {"id": 2, "name": "cell-adhered"}, {"id": 3, "name": "soma"}],
        "images": [],
        "annotations": []
    }
    
    image_id_offset = 0
    annotation_id_offset = 0
    
    for ds_name in dataset_names:
        ds_path = base_phase2 / ds_name
        json_path = ds_path / f"{split_name}_annotations.json"
        
        if not json_path.exists():
            print(f"Warning: {json_path} not found")
            continue
            
        with open(json_path, 'r') as f:
            coco = json.load(f)
            
        id_map = {}
        max_img_id = 0
        
        for img in coco.get("images", []):
            old_id = img["id"]
            new_id = old_id + image_id_offset
            id_map[old_id] = new_id
            
            img_copy = dict(img)
            img_copy["id"] = new_id
            
            new_file_name = f"{ds_name}_{img['file_name']}"
            img_copy["file_name"] = new_file_name
            merged_coco["images"].append(img_copy)
            
            max_img_id = max(max_img_id, old_id)
            
            # Symlink image
            src_img = ds_path / "images" / split_name / img["file_name"]
            dst_img = out_images_dir / new_file_name
            if src_img.exists() and not dst_img.exists():
                os.symlink(src_img, dst_img)
                
            # Symlink mask if exists (matches stem of original file_name)
            mask_src_dir = ds_path / "masks" / split_name
            if mask_src_dir.exists():
                stem = Path(img["file_name"]).stem
                for mask_file in mask_src_dir.glob(f"{stem}.*"):
                    new_mask_name = f"{ds_name}_{mask_file.name}"
                    dst_mask = out_masks_dir / new_mask_name
                    if not dst_mask.exists():
                        os.symlink(mask_file, dst_mask)
        
        max_ann_id = 0
        for ann in coco.get("annotations", []):
            old_ann_id = ann["id"]
            old_img_id = ann["image_id"]
            
            if old_img_id not in id_map:
                continue
                
            new_ann_id = old_ann_id + annotation_id_offset
            new_img_id = id_map[old_img_id]
            
            ann_copy = dict(ann)
            ann_copy["id"] = new_ann_id
            ann_copy["image_id"] = new_img_id
            merged_coco["annotations"].append(ann_copy)
            
            max_ann_id = max(max_ann_id, old_ann_id)
            
        image_id_offset += max_img_id + 1
        annotation_id_offset += max_ann_id + 1
        
    out_json = output_dir / f"{split_name}_annotations.json"
    with open(out_json, 'w') as f:
        json.dump(merged_coco, f)
        
    print(f"Merged {len(dataset_names)} datasets for {split_name} split into {out_json}")


def write_hydra_yaml(yaml_name, data_path, run_prefix):
    yaml_path = Path("configs/data") / f"{yaml_name}.yaml"
    
    content = f"""# @package _global_

defaults:
  - default@data

data:
  path: "{data_path}"
  val_name: 'test'
  test_name: 'test'
  train_name: 'train'

initialization:
  load_from_checkpoint: "/mnt/personal/cellanome/checkpoints/ALL_CKPTS/RFDETR-Seg-ckpts/output/checkpoint_best_ema.pth"

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
    BASE_PHASE2 = Path("/mnt/direct-attached/PHASE2")
    
    if not BASE_PHASE2.exists():
        print(f"Error: {BASE_PHASE2} not found. Script must be run on Vulcan.")
        
    experiments = {
        "c1_fibro_to_preadipo": {
            "train": ["20240509_Hs675Tfibroblasts_10x_caged_4_class"],
            "test": ["20241212_preadipocytes-adhered_10x_uncaged_4_class", "20250227_preadipocytes-adhered_10x_caged_4_class"]
        },
        "c1_preadipo_to_fibro": {
            "train": ["20241212_preadipocytes-adhered_10x_uncaged_4_class"],
            "test": ["20240509_Hs675Tfibroblasts_10x_caged_4_class", "231212_imr90_multichannel_overlay_4_class", "240213_imr90_multichannel_overlay_4_class", "20250227_preadipocytes-adhered_10x_caged_4_class"]
        },
        "c2_mc38caged_to_broad": {
            "train": ["20240624_mc38_10x_caged_4_class"],
            "test": ["20240624_mc38_10x_uncaged_4_class", "20240905_u87-adhered_10x_caged_4_class", "20240509_hela-adhered_10x_caged_4_class", "20250820_c8d1a_astrocytes-adherent_10x_caged_4_class"]
        },
        "c2_mc38uncaged_to_narrow": {
            "train": ["20240624_mc38_10x_uncaged_4_class"],
            "test": ["20240624_mc38_10x_caged_4_class", "20240625_mc38_10x_caged_4_class", "20240905_u87-adhered_10x_caged_4_class", "20240509_hela-adhered_10x_caged_4_class", "20250820_c8d1a_astrocytes-adherent_10x_caged_4_class"]
        },
        "c3_mc38_to_moc22": {
            "train": ["20240624_mc38_10x_caged_4_class"],
            "test": ["20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class", "20250917_moc22-adhered_10x_caged_4_class"]
        },
        "c3_moc22_to_mc38": {
            "train": ["20250917_moc22-adhered_10x_caged_4_class"],
            "test": ["20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class", "20240624_mc38_10x_caged_4_class"]
        },
        "c4_mc38_to_dc": {
            "train": ["20240624_mc38_10x_caged_4_class"],
            "test": ["20240515_DC-adhered_10x_caged_4_class", "20240516_DC-adhered_10x_caged_4_class"]
        },
        "c4_dc_to_mc38": {
            "train": ["20240515_DC-adhered_10x_caged_4_class"],
            "test": ["20240624_mc38_10x_caged_4_class", "20240516_DC-adhered_10x_caged_4_class"]
        },
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
        exp_dir_name = f"TRAINING_DATA_{exp_name.upper()}"
        out_dir = Path("/project/aip-robsc/asinha/cellanome/DATA") / exp_dir_name
        
        if BASE_PHASE2.exists():
            print(f"\nProcessing {exp_name}...")
            merge_coco_datasets(exp_data["train"], "train", out_dir, BASE_PHASE2)
            merge_coco_datasets(exp_data["test"], "test", out_dir, BASE_PHASE2)
            
        write_hydra_yaml(exp_name, out_dir, exp_name.upper())

if __name__ == "__main__":
    main()