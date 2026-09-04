import json
import os
import glob
from collections import defaultdict
import numpy as np

base_dir = '/mnt/direct-attached/PHASE2'

suspension_keywords = ["suspension", "jurkat", "k562", "nk92", "pbmc", "mousepbmc", "tall104", "raji", "jerat", "tal104"]

def is_adhered(dataset_name):
    name_lower = dataset_name.lower()
    for kw in suspension_keywords:
        if kw in name_lower:
            return False
    return True

stats_all = defaultdict(lambda: defaultdict(list))
total_images_all = defaultdict(int)

stats_adhered = defaultdict(lambda: defaultdict(list))
total_images_adhered = defaultdict(int)

# Classes mapping
class_names = {
    0: 'cell',
    1: 'bead',
    2: 'cell-adhered',
    3: 'soma'
}

def process_dataset(json_path, split_name, stats_dict, total_imgs_dict):
    if not os.path.exists(json_path):
        return
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    img_ids = [img['id'] for img in data['images']]
    total_imgs_dict[split_name] += len(img_ids)
    
    img_counts = defaultdict(lambda: defaultdict(int))
    
    for ann in data['annotations']:
        img_id = ann['image_id']
        cat_id = ann['category_id']
        img_counts[img_id][cat_id] += 1
        
    for img_id, counts in img_counts.items():
        for cat_id, count in counts.items():
            if cat_id in class_names:
                stats_dict[split_name][class_names[cat_id]].append(count)

adhered_datasets = []
all_datasets = []

for d in os.listdir(base_dir):
    dir_path = os.path.join(base_dir, d)
    if os.path.isdir(dir_path) and not d.startswith('.'):
        all_datasets.append(d)
        is_adh = is_adhered(d)
        if is_adh:
            adhered_datasets.append(d)

        train_path = os.path.join(dir_path, 'train_new_annotations.json')
        if not os.path.exists(train_path):
            train_path = os.path.join(dir_path, 'train_annotations.json')
        process_dataset(train_path, 'Train', stats_all, total_images_all)
        if is_adh:
            process_dataset(train_path, 'Train', stats_adhered, total_images_adhered)
        
        valid_path = os.path.join(dir_path, 'valid_annotations.json')
        process_dataset(valid_path, 'Val', stats_all, total_images_all)
        if is_adh:
            process_dataset(valid_path, 'Val', stats_adhered, total_images_adhered)
        
        test_path = os.path.join(dir_path, 'test_annotations.json')
        process_dataset(test_path, 'Test', stats_all, total_images_all)
        if is_adh:
            process_dataset(test_path, 'Test', stats_adhered, total_images_adhered)

print(f"Total datasets: {len(all_datasets)}")
print(f"Total adhered datasets: {len(adhered_datasets)}")
print("Adhered datasets:")
for ds in adhered_datasets:
    print(f" - {ds}")

def print_stats(stats_dict, total_imgs_dict, title):
    print(f"\n--- {title} ---")
    print(f"{'Split':<10} {'Class':<15} {'Mean':<10} {'Median':<10} {'Max':<10} {'Images w/ BBoxes':<20} {'Total Images':<15}")
    for split in ['Train', 'Val', 'Test']:
        for c_id, c_name in class_names.items():
            counts = stats_dict[split].get(c_name, [])
            if len(counts) == 0:
                continue
            mean_val = np.mean(counts)
            median_val = np.median(counts)
            max_val = np.max(counts)
            images_with_bboxes = len(counts)
            
            print(f"{split:<10} {c_name:<15} {mean_val:<10.2f} {median_val:<10.1f} {max_val:<10} {images_with_bboxes:<20,d} {total_imgs_dict[split]:<15,d}")

print_stats(stats_all, total_images_all, "All Datasets")
print_stats(stats_adhered, total_images_adhered, "Adhered Datasets")
