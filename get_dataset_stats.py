import json
import os
import glob
from collections import defaultdict
import numpy as np
import pandas as pd

base_dir = '/mnt/direct-attached/PHASE2'

splits = {
    'Train': 'train',
    'Val': 'valid',
    'Test': 'test'
}

# Stats we want to collect per split, per class
# Mean, Median, Max bounding boxes per image
# Images w/ BBoxes
# Total Images

stats = defaultdict(lambda: defaultdict(list))
total_images = defaultdict(int)

# To keep track of images with > 300 boxes
over_300_boxes = defaultdict(list)

# Classes mapping as per CLAUDE.md: 0:cell, 1:bead, 2:cell-adhered, 3:soma
class_names = {
    0: 'cell',
    1: 'bead',
    2: 'cell-adhered',
    3: 'soma'
}

def process_dataset(json_path, split_name):
    if not os.path.exists(json_path):
        return
    print(f"Processing {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    img_ids = [img['id'] for img in data['images']]
    total_images[split_name] += len(img_ids)
    
    # Map image id to dict of class_id -> count
    img_counts = defaultdict(lambda: defaultdict(int))
    
    # We might have categories defined in the JSON. Let's make sure we map category_id to correct class name.
    # But for robustness, we'll just use the IDs 0, 1, 2, 3 if they match.
    for ann in data['annotations']:
        img_id = ann['image_id']
        cat_id = ann['category_id']
        img_counts[img_id][cat_id] += 1
        
    for img_id, counts in img_counts.items():
        total_boxes = sum(counts.values())
        if total_boxes > 300:
            over_300_boxes[split_name].append((json_path, img_id, total_boxes))
            
        for cat_id, count in counts.items():
            if cat_id in class_names:
                stats[split_name][class_names[cat_id]].append(count)

# Iterate over subdirectories
for d in os.listdir(base_dir):
    dir_path = os.path.join(base_dir, d)
    if os.path.isdir(dir_path) and not d.startswith('.'):
        train_path = os.path.join(dir_path, 'train_new_annotations.json')
        if not os.path.exists(train_path):
            train_path = os.path.join(dir_path, 'train_annotations.json')
        process_dataset(train_path, 'Train')
        
        valid_path = os.path.join(dir_path, 'valid_annotations.json')
        process_dataset(valid_path, 'Val')
        
        test_path = os.path.join(dir_path, 'test_annotations.json')
        process_dataset(test_path, 'Test')

print("\n--- Statistics ---")
print(f"{'Split':<10} {'Class':<15} {'Mean':<10} {'Median':<10} {'Max':<10} {'Images w/ BBoxes':<20} {'Total Images':<15}")
for split in ['Train', 'Val', 'Test']:
    for c_id, c_name in class_names.items():
        counts = stats[split].get(c_name, [])
        if len(counts) == 0:
            continue
        mean_val = np.mean(counts)
        median_val = np.median(counts)
        max_val = np.max(counts)
        images_with_bboxes = len(counts)
        
        # Format the output
        print(f"{split:<10} {c_name:<15} {mean_val:<10.2f} {median_val:<10.1f} {max_val:<10} {images_with_bboxes:<20,d} {total_images[split]:<15,d}")

print("\n--- Over 300 boxes ---")
for split in ['Train', 'Val', 'Test']:
    print(f"{split}: {len(over_300_boxes[split])} images over 300 boxes")
    if len(over_300_boxes[split]) > 0:
        max_boxes = max([x[2] for x in over_300_boxes[split]])
        print(f"  Max boxes in a single image: {max_boxes}")
