import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pycocotools.mask as mask_util
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = "/mnt/direct-attached/PHASE2"
OUTPUT_DIR = "/mnt/direct-attached/PHASE2_EVAL_RESULTS/crop_visualizations"
TARGET_CLASSES = {0: "cell", 2: "cell-adhered", 3: "soma"}

def parse_dataset_name(ds_name):
    lower_name = ds_name.lower()
    suspension_keywords = ["suspension", "jurkat", "k562", "nk92", "pbmc", "mousepbmc", "tall104", "raji", "jerat", "tal104"]
    for kw in suspension_keywords:
        if kw in lower_name:
            return None
    clean_name = lower_name
    prefixes_suffixes_to_remove = ["-adhered", "-uncaged", "-caged", "10x", "1x", "20x"]
    for ps in prefixes_suffixes_to_remove:
        clean_name = clean_name.replace(ps, "")
    clean_name = clean_name.strip("-_")
    return clean_name

def save_crop_visualizations(ds_name, class_name, crops, output_dir):
    if not crops:
        return
    crops_np = np.array(crops, dtype=np.float32) / 255.0
    mean_img = np.mean(crops_np, axis=0)
    var_img = np.var(crops_np, axis=0)
    if np.max(var_img) > 0:
        var_img_norm = var_img / np.max(var_img)
    else:
        var_img_norm = var_img
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(mean_img)
    plt.title(f"{ds_name}\nMean Crop (n={len(crops)})")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    var_heatmap = np.mean(var_img_norm, axis=-1)
    im = plt.imshow(var_heatmap, cmap='hot')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Variance Heatmap")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{ds_name}_{class_name}_mean_var.png"), bbox_inches='tight')
    plt.close()
    
    n_samples = min(16, len(crops))
    if n_samples > 0:
        grid_size = int(np.ceil(np.sqrt(n_samples)))
        plt.figure(figsize=(grid_size * 2, grid_size * 2))
        for i in range(n_samples):
            plt.subplot(grid_size, grid_size, i + 1)
            plt.imshow(crops[i])
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{ds_name}_{class_name}_samples.png"), bbox_inches='tight')
        plt.close()

def extract_crops_cpu(img_path, mask_pkl_path, input_size=240, max_crops_per_img=16):
    img = cv2.imread(str(img_path))
    if img is None:
        return {}
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    with open(mask_pkl_path, 'rb') as f:
        mask_data = pickle.load(f)
        
    if not mask_data.get('annotations'):
        return {}
        
    crops_by_cat = {cat: [] for cat in TARGET_CLASSES}
    
    for ann in mask_data['annotations']:
        cat = int(ann['category_id'])
        if cat not in TARGET_CLASSES:
            continue
            
        if len(crops_by_cat[cat]) >= max_crops_per_img:
            continue
            
        x, y, x2, y2 = [int(float(v)) for v in ann['bbox']]
        w, h = x2 - x, y2 - y
        if w <= 0 or h <= 0:
            continue
            
        img_h, img_w = img.shape[:2]
        
        if isinstance(ann['segmentation'], list):
            rle = mask_util.frPyObjects(ann['segmentation'], img_h, img_w)
            mask = mask_util.decode(rle)
        else:
            mask = mask_util.decode(ann['segmentation'])
        if len(mask.shape) == 3:
            mask = mask.squeeze(2)
            
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        
        x1_clip, y1_clip = max(0, x), max(0, y)
        x2_clip, y2_clip = min(img_w, x + w), min(img_h, y + h)
        
        if mask.shape[:2] == (img_h, img_w):
            mask_3d_cropped = mask_3d[y1_clip:y2_clip, x1_clip:x2_clip]
        else:
            mask_y1 = y1_clip - y
            mask_y2 = mask_y1 + (y2_clip - y1_clip)
            mask_x1 = x1_clip - x
            mask_x2 = mask_x1 + (x2_clip - x1_clip)
            mask_3d_cropped = mask_3d[mask_y1:mask_y2, mask_x1:mask_x2]
        
        masked_img = np.zeros_like(img)
        masked_img[y1_clip:y2_clip, x1_clip:x2_clip] = img[y1_clip:y2_clip, x1_clip:x2_clip] * mask_3d_cropped
            
        size = max(w, h)
        cx, cy = x + w//2, y + h//2
        sq_x1, sq_y1 = cx - size//2, cy - size//2
        sq_x2, sq_y2 = sq_x1 + size, sq_y1 + size
        
        sq_x1_img, sq_y1_img = max(0, sq_x1), max(0, sq_y1)
        sq_x2_img, sq_y2_img = min(img.shape[1], sq_x2), min(img.shape[0], sq_y2)
        
        actual_crop = masked_img[sq_y1_img:sq_y2_img, sq_x1_img:sq_x2_img]
        if actual_crop.size == 0:
            continue
            
        crop = np.zeros((size, size, 3), dtype=np.uint8)
        offset_x, offset_y = sq_x1_img - sq_x1, sq_y1_img - sq_y1
        crop[offset_y:offset_y+actual_crop.shape[0], offset_x:offset_x+actual_crop.shape[1]] = actual_crop
        
        resized_crop = cv2.resize(crop, (input_size, input_size), interpolation=cv2.INTER_AREA)
        crops_by_cat[cat].append(resized_crop)
        
    return crops_by_cat

def main():
    base_dir = Path(DATA_DIR)
    if not base_dir.exists():
        print(f"Data dir {DATA_DIR} not found.")
        return
        
    for ds in tqdm(list(base_dir.iterdir()), desc="Processing Datasets"):
        if not ds.is_dir(): continue
        cell_line = parse_dataset_name(ds.name)
        if not cell_line: continue # skip suspension
        
        img_dir = ds / "images" / "test"
        mask_dir = ds / "masks" / "test"
        if not img_dir.exists() or not mask_dir.exists():
            img_dir = ds / "images" / "train"
            mask_dir = ds / "masks" / "train"
        if not img_dir.exists() or not mask_dir.exists(): continue
        
        imgs = sorted(list(img_dir.iterdir()))
        viz_saved = {cat: False for cat in TARGET_CLASSES}
        
        accumulated_crops = {cat: [] for cat in TARGET_CLASSES}
        
        for idx, img in enumerate(imgs):
            mask_path = mask_dir / f"{img.stem}.pkl"
            if mask_path.exists():
                crops_by_cat = extract_crops_cpu(img, mask_path)
                for cat, crops in crops_by_cat.items():
                    accumulated_crops[cat].extend(crops)
            
            all_done = True
            for cat in TARGET_CLASSES:
                if len(accumulated_crops[cat]) >= 64: 
                    if not viz_saved[cat]:
                        class_name = TARGET_CLASSES[cat]
                        out_d = os.path.join(OUTPUT_DIR, f"class_{class_name}")
                        save_crop_visualizations(ds.name, class_name, accumulated_crops[cat][:64], out_d)
                        viz_saved[cat] = True
                else:
                    all_done = False
                    
            if all_done or idx >= min(99, len(imgs)-1):
                for cat in TARGET_CLASSES:
                    if not viz_saved[cat] and len(accumulated_crops[cat]) > 0:
                        class_name = TARGET_CLASSES[cat]
                        out_d = os.path.join(OUTPUT_DIR, f"class_{class_name}")
                        save_crop_visualizations(ds.name, class_name, accumulated_crops[cat][:64], out_d)
                        viz_saved[cat] = True
                break

if __name__ == "__main__":
    main()
