#!/usr/bin/env python3
import json
import os
import shutil
from pathlib import Path

def filter_coco(input_json, output_json, dataset_keywords):
    """
    Filters a COCO JSON to only include images (and their annotations) 
    that match any of the dataset_keywords in their file_name.
    """
    print(f"Loading {input_json}...")
    with open(input_json, 'r') as f:
        coco = json.load(f)
        
    new_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco.get("categories", []),
        "images": [],
        "annotations": []
    }
    
    # Filter images
    valid_image_ids = set()
    for img in coco["images"]:
        file_name = img["file_name"].lower()
        if any(kw.lower() in file_name for kw in dataset_keywords):
            new_coco["images"].append(img)
            valid_image_ids.add(img["id"])
            
    # Filter annotations
    for ann in coco["annotations"]:
        if ann["image_id"] in valid_image_ids:
            new_coco["annotations"].append(ann)
            
    print(f"Filtered {len(new_coco['images'])} images and {len(new_coco['annotations'])} annotations for {dataset_keywords}")
    
    with open(output_json, 'w') as f:
        json.dump(new_coco, f)
    print(f"Saved to {output_json}")


def setup_experiment(base_data_dir, exp_dir_name, train_keywords, test_keywords):
    """
    Sets up a directory structure mimicking the original dataset but with filtered JSONs.
    Symlinks the image directories to save space.
    """
    base = Path(base_data_dir)
    exp_dir = base.parent / exp_dir_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Symlink images dir
    images_src = base / "images"
    images_dst = exp_dir / "images"
    if images_src.exists() and not images_dst.exists():
        os.symlink(images_src, images_dst)
        
    # Symlink masks dir if it exists
    masks_src = base / "masks"
    masks_dst = exp_dir / "masks"
    if masks_src.exists() and not masks_dst.exists():
        os.symlink(masks_src, masks_dst)
        
    # Process Train (only include train_keywords)
    filter_coco(
        base / "train_annotations.json", 
        exp_dir / "train_annotations.json", 
        train_keywords
    )
    
    # Process Val and Test (only include test_keywords)
    filter_coco(
        base / "valid_no300_annotations.json", 
        exp_dir / "valid_no300_annotations.json", 
        test_keywords
    )
    
    filter_coco(
        base / "test_no300_annotations.json", 
        exp_dir / "test_no300_annotations.json", 
        test_keywords
    )
    
    print(f"Experiment {exp_dir_name} setup complete in {exp_dir}\n")


import argparse

def main():
    parser = argparse.ArgumentParser(description="Create dataset splits for generalization experiments")
    parser.add_argument("--base_dir", default="/project/aip-robsc/asinha/cellanome/DATA/TRAINING_DATA", help="Path to original data root")
    parser.add_argument("--exp_name", required=True, help="Name of the output experiment directory")
    parser.add_argument("--train", nargs='+', required=True, help="List of dataset keywords to include in train")
    parser.add_argument("--test", nargs='+', required=True, help="List of dataset keywords to include in val/test")
    args = parser.parse_args()
    
    if not os.path.exists(args.base_dir):
        print(f"Error: {args.base_dir} not found. Please run this script on the Vulcan server.")
        return

    print(f"Creating experiment: {args.exp_name}")
    print(f"Train datasets: {args.train}")
    print(f"Test datasets:  {args.test}")
    
    setup_experiment(
        args.base_dir,
        args.exp_name,
        train_keywords=args.train,
        test_keywords=args.test
    )

if __name__ == "__main__":
    main()
