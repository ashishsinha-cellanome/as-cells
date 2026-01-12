import hydra
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pycocotools.coco import COCO
from omegaconf import DictConfig

def print_table(headers, rows, title):
    print(f"\n{title}")
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))
            
    header_str = " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    print("-" * len(header_str))
    print(header_str)
    print("-" * len(header_str))
    
    for row in rows:
        print(" | ".join(f"{str(v):<{w}}" for v, w in zip(row, widths)))
    print("-" * len(header_str))

def analyze_split(name, json_path):
    if not os.path.exists(json_path):
        print(f"\n[Skipping] File not found for {name}: {json_path}")
        return None, None

    print(f"\nAnalyzing {name.upper()} split from: {json_path}")
    try:
        import sys
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        coco = COCO(json_path)
        sys.stdout = original_stdout
    except Exception as e:
        print(f"Error loading COCO json: {e}")
        return None, None

    # Categories
    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)
    cat_names = {c['id']: c['name'] for c in cats}
    
    # Initialize counts
    label_counts = {c['id']: 0 for c in cats}
    size_counts = {'small': 0, 'medium': 0, 'large': 0}
    
    anns = coco.loadAnns(coco.getAnnIds())
    total_anns = len(anns)
    
    if total_anns == 0:
        print("No annotations found.")
        return [], []

    for ann in anns:
        cid = ann['category_id']
        if cid in label_counts:
            label_counts[cid] += 1
            
        area = ann['area']
        if area < 32**2:
            size_counts['small'] += 1
        elif area < 96**2:
            size_counts['medium'] += 1
        else:
            size_counts['large'] += 1
            
    # Prepare Data for Plotting (return list of dicts)
    label_data = []
    for cid in sorted(label_counts.keys()):
        count = label_counts[cid]
        cname = cat_names.get(cid, 'Unknown')
        label_data.append({'Class Name': cname, 'Count': count, 'Split': name.capitalize()})
    
    size_data = []
    for size in ['small', 'medium', 'large']:
        count = size_counts[size]
        size_data.append({'Size': size.capitalize(), 'Count': count, 'Split': name.capitalize()})
        
    return label_data, size_data

def plot_combined_distributions(all_label_data, all_size_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("colorblind")
    
    # --- 1. Combined Label Distribution ---
    if all_label_data:
        df_label = pd.DataFrame(all_label_data)
        plt.figure(figsize=(12, 6))
        
        # Plot with hue for splits
        ax = sns.barplot(x='Class Name', y='Count', hue='Split', data=df_label, palette=palette)
        
        # Add labels (optional, can get crowded)
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', padding=3, fontsize=9)

        plt.title('Label Distribution Across Splits')
        plt.xlabel('Class Name')
        plt.ylabel('Count')
        plt.legend(title='Dataset Split')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'combined_label_dist.png'), dpi=300)
        plt.close()

    # --- 2. Combined Size Distribution ---
    if all_size_data:
        df_size = pd.DataFrame(all_size_data)
        plt.figure(figsize=(10, 6))
        
        ax = sns.barplot(x='Size', y='Count', hue='Split', data=df_size, palette=palette, 
                         order=['Small', 'Medium', 'Large'])
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', padding=3, fontsize=9)
            
        plt.title('Object Size Distribution Across Splits')
        plt.xlabel('Object Size')
        plt.ylabel('Count')
        plt.legend(title='Dataset Split')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'combined_size_dist.png'), dpi=300)
        plt.close()
        
    print(f"\nCombined plots saved to: {output_dir}")

@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
    data_path = cfg.data.path
    print(f"Base Data Path: {data_path}")
    
    output_dir = "analysis_plots"
    splits = [
        ('train', 'train_annotations.json'),
        ('valid', 'valid_annotations.json'),
        ('test',  'test_annotations.json'),
    ]
    
    all_label_data = []
    all_size_data = []
    
    for split_name, filename in splits:
        full_path = os.path.join(data_path, filename)
        l_data, s_data = analyze_split(split_name, full_path)
        
        if l_data: all_label_data.extend(l_data)
        if s_data: all_size_data.extend(s_data)
        
    # Generate Combined Plots
    plot_combined_distributions(all_label_data, all_size_data, output_dir)

if __name__ == "__main__":
    main()
