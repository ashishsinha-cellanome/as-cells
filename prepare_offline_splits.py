#!/usr/bin/env python3

import os
import socket
import json
import random
from collections import defaultdict
import numpy as np
import pycocotools.mask as mask_util
from tqdm import tqdm
import multiprocessing as mp

# Apply numpy workaround for pycocotools if using numpy 2.0
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

if 'odin' in socket.gethostname():
    PHASE2_DIR = "/mnt/direct-attached/PHASE2"
elif 'vulcan' in socket.gethostname():
    PHASE2_DIR = "/project/aip-robsc/asinha/cellanome/DATA/PHASE2"
else:
    yield "PHASE2_DIR not declared or found"

suspension_keywords = ["suspension", "jurkat", "k562", "nk92", "pbmc", "mousepbmc", "tall104", "raji", "jerat", "tal104"]

def uncrop_rle(ann, img_width, img_height):
    seg = ann.get('segmentation')
    if isinstance(seg, dict) and list(seg.get('size', [])) != [img_height, img_width]:
        bbox = ann.get('bbox', [0, 0, 0, 0])
        x, y, bw, bh = [int(v) for v in bbox]
        
        if isinstance(seg['counts'], str):
            seg['counts'] = seg['counts'].encode('utf-8')
            
        try:
            cropped_mask = mask_util.decode(seg)
            full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            
            x_start = max(0, x)
            y_start = max(0, y)
            x_end = min(x + bw, img_width)
            y_end = min(y + bh, img_height)
            
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
            full_rle['counts'] = full_rle['counts'].decode('utf-8')
            
            ann['segmentation'] = full_rle
        except Exception as e:
            print(f"Error uncropping mask for annotation {ann['id']}: {e}")
    return ann

def process_file_in_place(filepath, desc=""):
    if not os.path.exists(filepath):
        return 0
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    img_id_to_dim = {img['id']: (img['width'], img['height']) for img in data.get('images', [])}
    uncropped_count = 0
    
    for ann in tqdm(data.get('annotations', []), desc=desc):
        img_id = ann['image_id']
        if img_id not in img_id_to_dim: continue
        w, h = img_id_to_dim[img_id]
        
        seg = ann.get('segmentation')
        if isinstance(seg, dict) and list(seg.get('size', [])) != [h, w]:
            uncrop_rle(ann, w, h)
            uncropped_count += 1
            
    if uncropped_count > 0:
        with open(filepath, 'w') as f:
            json.dump(data, f)
            
    return uncropped_count

def process_dataset(ds_name):
    if any(kw in ds_name.lower() for kw in suspension_keywords):
        return
        
    ds_path = os.path.join(PHASE2_DIR, ds_name)
    train_ann_path = os.path.join(ds_path, "train_annotations.json")
    out_train_path = os.path.join(ds_path, "train_new_annotations.json")
    out_val_path = os.path.join(ds_path, "valid_annotations.json")
    test_ann_path = os.path.join(ds_path, "test_annotations.json")
    
    print(f"\n[{ds_name}] Processing...")
    
    # 1. Process test_annotations.json in place
    test_fixed = process_file_in_place(test_ann_path, desc=f"{ds_name} Test")
    print(f"  -> Test fixed {test_fixed} RLEs")
    
    # 2. Re-create train_new and valid if we need to or to ensure they are padded
    if not os.path.exists(train_ann_path):
        print(f"  -> No train_annotations.json found. Skipping train/valid generation.")
        return
        
    with open(train_ann_path, 'r') as f:
        data = json.load(f)
        
    base_to_imgs = defaultdict(list)
    img_id_to_dim = {}
    for img in data.get('images', []):
        img_id_to_dim[img['id']] = (img['width'], img['height'])
        fname = img['file_name']
        if "_crp_" in fname:
            base_name = fname.split("_crp_")[0]
        else:
            base_name = fname.rsplit(".", 1)[0]
        base_to_imgs[base_name].append(img)
        
    base_names = sorted(list(base_to_imgs.keys()))
    random.seed(42)
    random.shuffle(base_names)
    
    split_idx = max(1, int(len(base_names) * 0.9))
    train_bases = set(base_names[:split_idx])
    val_bases = set(base_names[split_idx:])
    
    train_imgs = []
    val_imgs = []
    
    for b in train_bases:
        train_imgs.extend(base_to_imgs[b])
    for b in val_bases:
        val_imgs.extend(base_to_imgs[b])
        
    train_img_ids = {img['id'] for img in train_imgs}
    val_img_ids = {img['id'] for img in val_imgs}
    
    train_anns = []
    val_anns = []
    
    uncropped_count = 0
    for ann in tqdm(data.get('annotations', []), desc=f"{ds_name} Train"):
        img_id = ann['image_id']
        if img_id not in img_id_to_dim: continue
        w, h = img_id_to_dim[img_id]
        
        seg = ann.get('segmentation')
        if isinstance(seg, dict) and list(seg.get('size', [])) != [h, w]:
            uncrop_rle(ann, w, h)
            uncropped_count += 1
            
        if img_id in train_img_ids:
            train_anns.append(ann)
        elif img_id in val_img_ids:
            val_anns.append(ann)
            
    cats = data.get('categories', [])
    
    train_data = {'images': train_imgs, 'annotations': train_anns, 'categories': cats}
    val_data = {'images': val_imgs, 'annotations': val_anns, 'categories': cats}
    
    with open(out_train_path, 'w') as f:
        json.dump(train_data, f)
    with open(out_val_path, 'w') as f:
        json.dump(val_data, f)
        
    print(f"  -> Train uncropped {uncropped_count} RLEs. Train_new: {len(train_imgs)} imgs. Val: {len(val_imgs)} imgs.")

def main():
    if not os.path.exists(PHASE2_DIR):
        print(f"Directory not found: {PHASE2_DIR}")
        return
        
    datasets = [d for d in os.listdir(PHASE2_DIR) if os.path.isdir(os.path.join(PHASE2_DIR, d))]
    
    for ds in datasets:
        process_dataset(ds)
        
if __name__ == '__main__':
    main()
