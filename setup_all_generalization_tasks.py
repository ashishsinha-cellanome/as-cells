#!/usr/bin/env python3
import json
import os
import random
from pathlib import Path
import hashlib

def merge_train_datasets(dataset_names, output_dir, base_phase2, report_lines):
    split_name = "train"
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
    
    total_images_in_split = 0
    total_bboxes_in_split = 0
    total_excluded_in_split = 0
    
    for ds_name in dataset_names:
        ds_path = base_phase2 / ds_name
        json_path = ds_path / f"{split_name}_annotations.json"
        
        if not json_path.exists():
            print(f"Warning: {json_path} not found")
            continue
            
        with open(json_path, 'r') as f:
            coco = json.load(f)
            
        # Count annotations per image to filter >300
        ann_counts = {}
        for ann in coco.get("annotations", []):
            img_id = ann["image_id"]
            ann_counts[img_id] = ann_counts.get(img_id, 0) + 1
            
        id_map = {}
        max_img_id = 0
        
        kept_images_count = 0
        kept_bboxes_count = 0
        excluded_images_count = 0
        
        for img in coco.get("images", []):
            old_id = img["id"]
            
            # Check annotation count for this image
            if ann_counts.get(old_id, 0) > 300:
                excluded_images_count += 1
                continue
                
            new_id = old_id + image_id_offset
            id_map[old_id] = new_id
            
            img_copy = dict(img)
            img_copy["id"] = new_id
            
            new_file_name = str((ds_path / "images" / split_name / img['file_name']).resolve())
            img_copy["file_name"] = new_file_name
            merged_coco["images"].append(img_copy)
            
            kept_images_count += 1
            max_img_id = max(max_img_id, old_id)
        
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
            
            kept_bboxes_count += 1
            max_ann_id = max(max_ann_id, old_ann_id)
            
        image_id_offset += max_img_id + 1
        annotation_id_offset += max_ann_id + 1
        
        report_lines.append(f"| {split_name} | {ds_name} | {kept_images_count} | {kept_bboxes_count} | {excluded_images_count} |")
        total_images_in_split += kept_images_count
        total_bboxes_in_split += kept_bboxes_count
        total_excluded_in_split += excluded_images_count
        
    out_json = output_dir / f"{split_name}_annotations.json"
    with open(out_json, 'w') as f:
        json.dump(merged_coco, f)
        
    report_lines.append(f"| **TOTAL {split_name.upper()}** | | **{total_images_in_split}** | **{total_bboxes_in_split}** | **{total_excluded_in_split}** |")
    report_lines.append("")
    print(f"Merged {len(dataset_names)} datasets for {split_name} split into {out_json}")


def merge_and_split_test_datasets(dataset_names, output_dir, base_phase2, report_lines):
    output_dir = Path(output_dir)
    splits = ["val", "test"]
    
    out_dirs = {}
    merged_coco = {}
    for s in splits:
        out_images_dir = output_dir / "images" / s
        out_images_dir.mkdir(parents=True, exist_ok=True)
        
        out_masks_dir = output_dir / "masks" / s
        out_masks_dir.mkdir(parents=True, exist_ok=True)
        
        out_dirs[s] = {"images": out_images_dir, "masks": out_masks_dir}
        
        merged_coco[s] = {
            "info": {},
            "licenses": [],
            "categories": [{"id": 0, "name": "cell"}, {"id": 1, "name": "bead"}, {"id": 2, "name": "cell-adhered"}, {"id": 3, "name": "soma"}],
            "images": [],
            "annotations": []
        }
    
    image_id_offset = {"val": 0, "test": 0}
    annotation_id_offset = {"val": 0, "test": 0}
    
    total_images_in_split = {"val": 0, "test": 0}
    total_bboxes_in_split = {"val": 0, "test": 0}
    total_excluded_in_split = {"val": 0, "test": 0}
    
    for ds_name in dataset_names:
        ds_path = base_phase2 / ds_name
        json_path = ds_path / "test_annotations.json"
        
        if not json_path.exists():
            print(f"Warning: {json_path} not found")
            continue
            
        with open(json_path, 'r') as f:
            coco = json.load(f)
            
        ann_counts = {}
        for ann in coco.get("annotations", []):
            img_id = ann["image_id"]
            ann_counts[img_id] = ann_counts.get(img_id, 0) + 1
            
        valid_images = []
        excluded_images = []
        for img in coco.get("images", []):
            if ann_counts.get(img["id"], 0) <= 300:
                valid_images.append(img)
            else:
                excluded_images.append(img)
                
        base_name_groups = {}
        for img in valid_images:
            base_name = img["file_name"].split("_crp_")[0]
            if base_name not in base_name_groups:
                base_name_groups[base_name] = []
            base_name_groups[base_name].append(img)
            
        # Also group excluded images to track which split they "would" have belonged to
        for img in excluded_images:
            base_name = img["file_name"].split("_crp_")[0]
            if base_name not in base_name_groups:
                base_name_groups[base_name] = []
            
        base_names = list(base_name_groups.keys())
        base_names.sort()
        
        seed_val = int(hashlib.md5(ds_name.encode()).hexdigest(), 16) % (2**32)
        random.seed(seed_val)
        random.shuffle(base_names)
        
        split_idx = int(len(base_names) * 0.2)
        val_bases = set(base_names[:split_idx])
        test_bases = set(base_names[split_idx:])
        
        id_map = {}
        max_img_id = {"val": 0, "test": 0}
        
        kept_images_count = {"val": 0, "test": 0}
        kept_bboxes_count = {"val": 0, "test": 0}
        excluded_images_count = {"val": 0, "test": 0}
        
        for img in excluded_images:
            base_name = img["file_name"].split("_crp_")[0]
            target_split = "val" if base_name in val_bases else "test"
            excluded_images_count[target_split] += 1
            
        for img in valid_images:
            base_name = img["file_name"].split("_crp_")[0]
            target_split = "val" if base_name in val_bases else "test"
            
            old_id = img["id"]
            new_id = old_id + image_id_offset[target_split]
            id_map[old_id] = {"id": new_id, "split": target_split}
            
            img_copy = dict(img)
            img_copy["id"] = new_id
            
            new_file_name = str((ds_path / "images" / "test" / img['file_name']).resolve())
            img_copy["file_name"] = new_file_name
            merged_coco[target_split]["images"].append(img_copy)
            
            kept_images_count[target_split] += 1
            max_img_id[target_split] = max(max_img_id[target_split], old_id)
                        
        max_ann_id = {"val": 0, "test": 0}
        for ann in coco.get("annotations", []):
            old_ann_id = ann["id"]
            old_img_id = ann["image_id"]
            
            if old_img_id not in id_map:
                continue
                
            target_split = id_map[old_img_id]["split"]
            new_ann_id = old_ann_id + annotation_id_offset[target_split]
            new_img_id = id_map[old_img_id]["id"]
            
            ann_copy = dict(ann)
            ann_copy["id"] = new_ann_id
            ann_copy["image_id"] = new_img_id
            merged_coco[target_split]["annotations"].append(ann_copy)
            
            kept_bboxes_count[target_split] += 1
            max_ann_id[target_split] = max(max_ann_id[target_split], old_ann_id)
            
        for s in splits:
            image_id_offset[s] += max_img_id[s] + 1
            annotation_id_offset[s] += max_ann_id[s] + 1
            
            report_lines.append(f"| {s} | {ds_name} | {kept_images_count[s]} | {kept_bboxes_count[s]} | {excluded_images_count[s]} |")
            total_images_in_split[s] += kept_images_count[s]
            total_bboxes_in_split[s] += kept_bboxes_count[s]
            total_excluded_in_split[s] += excluded_images_count[s]
            
    for s in splits:
        out_json = output_dir / f"{s}_annotations.json"
        with open(out_json, 'w') as f:
            json.dump(merged_coco[s], f)
        
        report_lines.append(f"| **TOTAL {s.upper()}** | | **{total_images_in_split[s]}** | **{total_bboxes_in_split[s]}** | **{total_excluded_in_split[s]}** |")
        report_lines.append("")
        print(f"Merged {len(dataset_names)} datasets for {s} split into {out_json}")


def write_hydra_yaml(yaml_name, data_path, run_prefix):
    yaml_path = Path("configs/data") / f"{yaml_name}.yaml"
    
    content = f"""# @package _global_

defaults:
  - default@data

data:
  path: "{data_path}"
  val_name: 'val'
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
    
    report_lines = [
        "# Generalization Experiments Data Report (Filtered >300 Bboxes)",
        "",
        "This report details the number of images and bounding boxes for each merged data split.",
        "The Test sets were generated by taking 80% of unique base images, while 20% were reserved for the Val sets (preventing crop leakage).",
        ""
    ]
    
    for exp_name, exp_data in experiments.items():
        exp_dir_name = f"TRAINING_DATA_{exp_name.upper()}"
        out_dir = Path("/project/aip-robsc/asinha/cellanome/DATA") / exp_dir_name
        
        report_lines.append(f"## Experiment: `{exp_name}`")
        report_lines.append("| Split | Dataset | Kept Images | Kept Bboxes | Excluded Images (>300 Bboxes) |")
        report_lines.append("| :--- | :--- | ---: | ---: | ---: |")
        
        if BASE_PHASE2.exists():
            print(f"\nProcessing {exp_name}...")
            merge_train_datasets(exp_data["train"], out_dir, BASE_PHASE2, report_lines)
            merge_and_split_test_datasets(exp_data["test"], out_dir, BASE_PHASE2, report_lines)
        else:
            report_lines.append("| (Skipped) | Vulcan Storage Not Found | 0 | 0 |")
            
        write_hydra_yaml(exp_name, out_dir, exp_name.upper())

    report_path = Path("generalization_data_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport written to {report_path}")

if __name__ == "__main__":
    main()
