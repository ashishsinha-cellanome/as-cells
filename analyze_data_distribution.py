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

def plot_distributions(label_data, size_data, split_name, output_dir):
    """
    Generate and save bar plots for label and size distributions.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # Use a color-blind friendly palette
    palette = sns.color_palette("colorblind")
    
    # --- 1. Label Distribution Plot ---
    if label_data:
        df_label = pd.DataFrame(label_data)
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Class Name', y='Count', data=df_label, palette=palette)
        
        # Add labels on top of bars
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2., height + 0.5,
                    f'{int(height)}', ha="center", fontsize=10)
            
        plt.title(f'Label Distribution - {split_name.capitalize()}')
        plt.xlabel('Class Name')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{split_name}_label_dist.png'), dpi=300)
        plt.close()

    # --- 2. Size Distribution Plot ---
    if size_data:
        df_size = pd.DataFrame(size_data)
        plt.figure(figsize=(8, 6))
        ax = sns.barplot(x='Size', y='Count', data=df_size, palette=palette, order=['Small', 'Medium', 'Large'])
        
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2., height + 0.5,
                    f'{int(height)}', ha="center", fontsize=10)
            
        plt.title(f'Object Size Distribution - {split_name.capitalize()}')
        plt.xlabel('Object Size')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{split_name}_size_dist.png'), dpi=300)
        plt.close()
    
    print(f"Plots saved to: {output_dir}")

def analyze_split(name, json_path, output_dir):
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
            
    # Prepare Label Table & Data for Plotting
    label_rows = []
    plot_label_data = []
    
    for cid in sorted(label_counts.keys()):
        count = label_counts[cid]
        pct = (count / total_anns) * 100
        cname = cat_names.get(cid, 'Unknown')
        label_rows.append([cid, cname, count, f"{pct:.2f}%"])
        plot_label_data.append({'Class Name': cname, 'Count': count})
    
    print_table(
        headers=['ID', 'Class Name', 'Count', 'Percentage'], 
        rows=label_rows, 
        title=f"--- Label Distribution ({name}) ---"
    )
    
    # Prepare Size Table & Data for Plotting
    size_rows = []
    plot_size_data = []
    
    for size in ['small', 'medium', 'large']:
        count = size_counts[size]
        pct = (count / total_anns) * 100
        capital_size = size.capitalize()
        size_rows.append([capital_size, count, f"{pct:.2f}%"])
        plot_size_data.append({'Size': capital_size, 'Count': count})
        
    print_table(
        headers=['Size', 'Count', 'Percentage'], 
        rows=size_rows, 
        title=f"--- Size Distribution ({name}) ---"
    )
    
    # Generate Plots
    plot_distributions(plot_label_data, plot_size_data, name, output_dir)

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
    
    for split_name, filename in splits:
        full_path = os.path.join(data_path, filename)
        analyze_split(split_name, full_path, output_dir)

if __name__ == "__main__":
    main()