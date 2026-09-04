import json
import os
import math

base_dir = '/mnt/direct-attached/PHASE2'

# We'll check all the adhered datasets since the user wants to know about configurations they've tried.
suspension_keywords = ["suspension", "jurkat", "k562", "nk92", "pbmc", "mousepbmc", "tall104", "raji", "jerat", "tal104"]

def is_adhered(dataset_name):
    name_lower = dataset_name.lower()
    for kw in suspension_keywords:
        if kw in name_lower:
            return False
    return True

adhered_datasets = []
for d in os.listdir(base_dir):
    if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith('.'):
        if is_adhered(d):
            adhered_datasets.append(d)

fractions = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]

print(f"{'Dataset':<60} {'Total Base':<15} " + " ".join([f"Frac={f:<6}" for f in fractions]))
print("-" * 140)

def get_base_images_count(dataset_name):
    json_path = os.path.join(base_dir, dataset_name, 'train_new_annotations.json')
    if not os.path.exists(json_path):
        json_path = os.path.join(base_dir, dataset_name, 'train_annotations.json')
        
    if not os.path.exists(json_path):
        return None
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    base_names = set()
    for img in data['images']:
        file_name = img['file_name']
        base_name = file_name.split("_crp_")[0] if "_crp_" in file_name else file_name.rsplit(".", 1)[0]
        base_names.add(base_name)
        
    return len(base_names)

for ds in sorted(adhered_datasets):
    total_base = get_base_images_count(ds)
    if total_base is None:
        continue
        
    counts = []
    for frac in fractions:
        keep = max(1, math.floor(total_base * frac))
        counts.append(f"{keep:<11}")
        
    print(f"{ds:<60} {total_base:<15} " + "".join(counts))
