import hydra
import os
import json
from pycocotools.coco import COCO
from omegaconf import DictConfig

def print_table(headers, rows, title):
    print(f"\n{title}")
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))
            
    # Print header
    header_str = " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    print("-" * len(header_str))
    print(header_str)
    print("-" * len(header_str))
    
    # Print rows
    for row in rows:
        print(" | ".join(f"{str(v):<{w}}" for v, w in zip(row, widths)))
    print("-" * len(header_str))

def analyze_split(name, json_path):
    if not os.path.exists(json_path):
        print(f"\n[Skipping] File not found for {name}: {json_path}")
        return

    print(f"\nAnalyzing {name.upper()} split from: {json_path}")
    try:
        # Suppress pycocotools print output
        import sys
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        coco = COCO(json_path)
        sys.stdout = original_stdout
    except Exception as e:
        print(f"Error loading COCO json: {e}")
        return

    # Categories
    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)
    cat_names = {c['id']: c['name'] for c in cats}
    
    # Initialize counts
    label_counts = {c['id']: 0 for c in cats}
    # COCO definition: small < 32x32, medium < 96x96, large >= 96x96
    size_counts = {'small': 0, 'medium': 0, 'large': 0}
    
    anns = coco.loadAnns(coco.getAnnIds())
    total_anns = len(anns)
    
    if total_anns == 0:
        print("No annotations found.")
        return

    for ann in anns:
        # Label count
        cid = ann['category_id']
        if cid in label_counts:
            label_counts[cid] += 1
            
        # Size count
        area = ann['area']
        if area < 32**2:
            size_counts['small'] += 1
        elif area < 96**2:
            size_counts['medium'] += 1
        else:
            size_counts['large'] += 1
            
    # Prepare Label Table
    label_rows = []
    for cid in sorted(label_counts.keys()):
        count = label_counts[cid]
        pct = (count / total_anns) * 100
        label_rows.append([cid, cat_names.get(cid, 'Unknown'), count, f"{pct:.2f}%"])
    
    print_table(
        headers=['ID', 'Class Name', 'Count', 'Percentage'], 
        rows=label_rows, 
        title=f"--- Label Distribution ({name}) ---"
    )
    
    # Prepare Size Table
    size_rows = []
    for size in ['small', 'medium', 'large']:
        count = size_counts[size]
        pct = (count / total_anns) * 100
        size_rows.append([size.capitalize(), count, f"{pct:.2f}%"])
        
    print_table(
        headers=['Size', 'Count', 'Percentage'], 
        rows=size_rows, 
        title=f"--- Size Distribution ({name}) ---"
    )

@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
    data_path = cfg.data.path
    print(f"Base Data Path: {data_path}")
    
    # We look for the standard naming convention first as requested
    # But also fallback to config names if needed
    
    splits = [
        ('train', 'train_annotations.json'),
        ('valid', 'valid_annotations.json'),
        ('test',  'test_annotations.json'),
        # Fallback/Alternatives based on config if needed
        # ('train_config', f"{cfg.data.train_name}_annotations.json"),
        # ('val_config', f"{cfg.data.val_name}_annotations.json")
    ]
    
    for split_name, filename in splits:
        full_path = os.path.join(data_path, filename)
        analyze_split(split_name, full_path)

if __name__ == "__main__":
    main()
