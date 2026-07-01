#!/usr/bin/env python3

import os
import json
import pickle
import random
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict
import numpy as np
import pycocotools.mask as mask_util
from tqdm import tqdm

# Apply NumPy 2.0 copy keyword deprecation patch for pycocotools stability
_orig_asarray = np.asarray
def _patched_asarray(*args, **kwargs):
    kwargs.pop("copy", None)
    return _orig_asarray(*args, **kwargs)
np.asarray = _patched_asarray

_orig_array = np.array
def _patched_array(*args, **kwargs):
    kwargs.pop("copy", None)
    return _orig_array(*args, **kwargs)
np.array = _patched_array


def decode_bytes(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, bytes):
        return obj.decode('utf-8')
    elif isinstance(obj, dict):
        return {k: decode_bytes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decode_bytes(v) for v in obj]
    return obj

def uncrop_rle_mask(seg, bbox, img_width=672, img_height=672):
    """Dynamically uncrops and pads any cropped RLE mask to the full image size [672, 672]."""
    if isinstance(seg, dict) and list(seg.get('size', [])) != [img_height, img_width]:
        x, y, bw, bh = [int(v) for v in bbox]
        if isinstance(seg.get('counts'), str):
            seg['counts'] = seg['counts'].encode('utf-8')
        elif isinstance(seg.get('counts'), bytes):
            pass # already bytes
            
        try:
            cropped_mask = mask_util.decode(seg)
            full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            
            x_start, y_start = max(0, x), max(0, y)
            x_end, y_end = min(x + bw, img_width), min(y + bh, img_height)
            
            if x_start < x_end and y_start < y_end:
                cx_start = 0 if x >= 0 else -x
                cy_start = 0 if y >= 0 else -y
                
                true_cy_end = min(cy_start + (y_end - y_start), cropped_mask.shape[0])
                true_cx_end = min(cx_start + (x_end - x_start), cropped_mask.shape[1])
                
                y_end = y_start + (true_cy_end - cy_start)
                x_end = x_start + (true_cx_end - cx_start)
                
                full_mask[y_start:y_end, x_start:x_end] = cropped_mask[cy_start:true_cy_end, cx_start:true_cx_end]
                
            full_mask_f = np.asfortranarray(full_mask)
            full_rle = mask_util.encode(full_mask_f)
            return decode_bytes(full_rle)
        except Exception as e:
            print(f"Error uncropping mask: {e}")
            return decode_bytes(seg)
            
    return decode_bytes(seg)


def process_dataset(dataset_path: Path):
    """Generates uncropped COCO JSON annotations, dynamically splitting train into 90/10."""
    splits = ["train", "val", "test", "valid_no300", "test_no300"]
    
    for split in splits:
        img_dir = dataset_path / "images" / split
        mask_dir = dataset_path / "masks" / split
        
        if not img_dir.exists() or not mask_dir.exists():
            continue
            
        seen_classes = set()
        annotations = []
        images = []
        ann_id_counter = 1
        
        valid_img_names = sorted(os.listdir(img_dir))
        
        for idx, img_name in enumerate(valid_img_names):
            img_path = img_dir / img_name
            if not img_path.is_file():
                continue
                
            stem = img_path.stem
            mask_path = mask_dir / f"{stem}.pkl"
            
            if not mask_path.exists():
                continue
            
            images.append({
                "id": idx + 1,
                "file_name": img_name,
                "width": 672,
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
                seg = uncrop_rle_mask(seg, ann["bbox"], img_width=672, img_height=672)
                
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
                
        # Define categories
        KNOWN_CLASSES = {0: "cell", 1: "bead", 2: "cell-adhered", 3: "soma"}
        categories = [{"id": cat, "name": KNOWN_CLASSES.get(cat, f"class_{cat}"), "supercategory": "biology"} for cat in seen_classes]
        
        # Split logic for train -> train_new (90%) and valid (10%)
        if split == "train":
            base_to_imgs = defaultdict(list)
            for img in images:
                fname = img['file_name']
                base_name = fname.split("_crp_")[0] if "_crp_" in fname else fname.rsplit(".", 1)[0]
                base_to_imgs[base_name].append(img)
                
            base_names = sorted(list(base_to_imgs.keys()))
            
            # STRICTLY USE SEED 42 FOR REPRODUCIBILITY ACROSS DIFFERENT SERVERS
            random.seed(42)
            random.shuffle(base_names)
            
            split_idx = max(1, int(len(base_names) * 0.9))
            train_bases = set(base_names[:split_idx])
            val_bases = set(base_names[split_idx:])
            
            train_imgs = []
            val_imgs = []
            for b in train_bases: train_imgs.extend(base_to_imgs[b])
            for b in val_bases: val_imgs.extend(base_to_imgs[b])
            
            train_img_ids = {img['id'] for img in train_imgs}
            val_img_ids = {img['id'] for img in val_imgs}
            
            train_anns = [ann for ann in annotations if ann['image_id'] in train_img_ids]
            val_anns = [ann for ann in annotations if ann['image_id'] in val_img_ids]
            
            with open(dataset_path / "train_new_annotations.json", "w") as f:
                json.dump(decode_bytes({"images": train_imgs, "annotations": train_anns, "categories": categories}), f, indent=2)
                
            with open(dataset_path / "valid_annotations.json", "w") as f:
                json.dump(decode_bytes({"images": val_imgs, "annotations": val_anns, "categories": categories}), f, indent=2)
        
        # Save standard non-train splits
        else:
            out_json = dataset_path / f"{split}_annotations.json"
            with open(out_json, "w") as f:
                json.dump(decode_bytes({"images": images, "annotations": annotations, "categories": categories}), f, indent=2)


def process_dataset_wrapper(dataset_path_str: str):
    """Multiprocessing worker wrapper."""
    dataset_path = Path(dataset_path_str)
    try:
        process_dataset(dataset_path)
    except Exception as e:
        print(f"Error processing {dataset_path.name}: {e}")


def main():
    phase2_dir = Path("/mnt/direct-attached/PHASE2")
    if not phase2_dir.exists():
        print(f"PHASE2 dir not found at: {phase2_dir}")
        return
        
    datasets = [str(item) for item in phase2_dir.iterdir() if item.is_dir()]
    
    # Scale workers safely (reserving 2 cores for system stability)
    num_workers = max(1, mp.cpu_count() - 2)
    print(f"[Phase2 Prep] Starting multiprocessing pool with {num_workers} workers to process {len(datasets)} datasets...")
    
    with mp.Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_dataset_wrapper, datasets), total=len(datasets), desc="Regenerating Phase 2 Datasets"))
    print("\n[Phase2 Prep] All datasets processed and split successfully!")


if __name__ == '__main__':
    main()
