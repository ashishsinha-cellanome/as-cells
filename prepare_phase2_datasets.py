import os
import json
import pickle
from pathlib import Path
from tqdm import tqdm

def process_dataset(dataset_path: Path):
    """Generates COCO JSON annotations for train, val, test splits from .pkl masks."""
    
    splits = ["train", "val", "test", "valid", "valid_no300", "test_no300"]
    
    for split in splits:
        img_dir = dataset_path / "images" / split
        mask_dir = dataset_path / "masks" / split
        
        if not img_dir.exists() or not mask_dir.exists():
            continue
            
        print(f"Processing {dataset_path.name} - {split}")
        
        seen_classes = set()
        annotations = []
        images = []
        
        ann_id_counter = 1
        
        for idx, img_name in enumerate(tqdm(sorted(os.listdir(img_dir)))):
            img_path = img_dir / img_name
            if not img_path.is_file():
                continue
                
            stem = img_path.stem
            mask_path = mask_dir / f"{stem}.pkl"
            
            if not mask_path.exists():
                print(f"Warning: No mask found for {img_name}")
                continue
            
            images.append({
                "id": idx + 1,
                "file_name": img_name,
                "width": 672,  # Default for the project
                "height": 672
            })
            
            with open(mask_path, "rb") as f:
                mask_data = pickle.load(f)
                
            for ann in mask_data.get("annotations", []):
                cat_id = int(ann["category_id"])
                seen_classes.add(cat_id)
                
                # xyxy -> xywh
                x1, y1, x2, y2 = [int(v) for v in ann["bbox"]]
                w = x2 - x1
                h = y2 - y1
                
                seg = ann.get("segmentation", [])
                if isinstance(seg, dict) and "counts" in seg and isinstance(seg["counts"], bytes):
                    seg["counts"] = seg["counts"].decode("utf-8")
                
                annotations.append({
                    "id": ann_id_counter,
                    "image_id": idx + 1,
                    "category_id": cat_id,
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "segmentation": seg
                })
                ann_id_counter += 1
                
        # Generate categories without remapping
        KNOWN_CLASSES = {
            0: "cell",
            1: "bead",
            2: "soma",
            3: "cell-adhered",
        }
        categories = [{"id": cat, "name": KNOWN_CLASSES.get(cat, f"class_{cat}"), "supercategory": "biology"} for cat in seen_classes]
        
        coco_dict = {
            "images": images,
            "annotations": annotations,
            "categories": categories
        }
        
        # Save JSON
        out_json = dataset_path / f"{split}_annotations.json"
        with open(out_json, "w") as f:
            json.dump(coco_dict, f, indent=2)
            
        # Save Class List
        classes_json = dataset_path / f"{split}_classes.json"
        with open(classes_json, "w") as f:
            json.dump(list(seen_classes), f, indent=2)
                
        print(f"Classes seen in {dataset_path.name}/{split}: {seen_classes}")

if __name__ == "__main__":
    phase2_dir = Path("/mnt/direct-attached/PHASE2")
    if not phase2_dir.exists():
        print("PHASE2 dir not found at /mnt/direct-attached/PHASE2")
        exit(1)
        
    for item in phase2_dir.iterdir():
        if item.is_dir():
            process_dataset(item)
