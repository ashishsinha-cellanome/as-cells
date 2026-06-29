import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

KNOWN_CLASSES = {0: "cell", 1: "bead", 2: "cell-adhered", 3: "soma"}

def analyze_datasets(phase2_dir, output_dir):
    phase2_dir = Path(phase2_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = [d for d in phase2_dir.iterdir() if d.is_dir()]
    
    all_data = []
    
    for dataset in sorted(datasets):
        dataset_name = dataset.name
        
        splits = ["train", "val", "test", "valid", "valid_no300", "test_no300"]
        
        for split in splits:
            json_file = dataset / f"{split}_annotations.json"
            if not json_file.exists():
                continue
                
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            num_images = len(data.get('images', []))
            
            class_counts = {}
            for ann in data.get('annotations', []):
                cat_id = ann['category_id']
                cat_name = KNOWN_CLASSES.get(cat_id, f"NEW_CLASS_ID_{cat_id}")
                class_counts[cat_name] = class_counts.get(cat_name, 0) + 1
                
            if num_images > 0 or class_counts:
                # Store the base info
                row = {
                    "Dataset": dataset_name,
                    "Split": split,
                    "Images": num_images,
                }
                # Add counts for known classes
                for k_name in KNOWN_CLASSES.values():
                    row[k_name] = class_counts.get(k_name, 0)
                
                # Add counts for unknown/new classes
                new_classes = {k: v for k, v in class_counts.items() if k not in KNOWN_CLASSES.values()}
                row["New_Classes_Count"] = sum(new_classes.values())
                row["New_Classes_List"] = ", ".join([f"{k}({v})" for k, v in new_classes.items()]) if new_classes else "None"
                
                all_data.append(row)
                
    if not all_data:
        print("No data found to analyze.")
        return
        
    df = pd.DataFrame(all_data)
    
    # Save detailed split-wise data table
    csv_path = output_dir / "phase2_split_class_counts.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved detailed split-wise class counts to {csv_path}")
    
    # Aggregating over splits to plot total dataset composition
    agg_df = df.groupby('Dataset').sum(numeric_only=True).reset_index()
    # We want to plot the distribution of classes per dataset
    class_columns = list(KNOWN_CLASSES.values()) + ["New_Classes_Count"]
    
    plot_df = agg_df.set_index('Dataset')[class_columns]
    
    # Visualization: Heatmap (Combined)
    plt.figure(figsize=(14, 12))
    sns.heatmap(plot_df, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Label Counts per Class and Dataset (All Splits Combined)")
    plt.ylabel("Dataset")
    plt.xlabel("Class")
    plt.tight_layout()
    plt.savefig(output_dir / "class_counts_heatmap_combined.png")
    plt.close()
    
    # Visualization: Heatmap per split
    for split in df['Split'].unique():
        split_df = df[df['Split'] == split].copy()
        if split_df.empty:
            continue
        plot_df_split = split_df.set_index('Dataset')[class_columns]
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(plot_df_split, annot=True, fmt="d", cmap="YlGnBu")
        plt.title(f"Label Counts per Class and Dataset ({split.capitalize()} Split)")
        plt.ylabel("Dataset")
        plt.xlabel("Class")
        plt.tight_layout()
        plt.savefig(output_dir / f"class_counts_heatmap_{split}.png")
        plt.close()
    
    # Visualization: Stacked Bar Chart
    plot_df['Total'] = plot_df.sum(axis=1)
    plot_df_sorted = plot_df.sort_values(by='Total', ascending=False).drop(columns=['Total'])
    
    plot_df_sorted.plot(kind='bar', stacked=True, figsize=(18, 10), colormap='viridis')
    plt.title("Label Counts per Dataset (Stacked by Class)")
    plt.xlabel("Dataset")
    plt.ylabel("Number of Labels")
    plt.legend(title="Class")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "class_counts_stacked_bar.png")
    plt.close()

if __name__ == "__main__":
    analyze_datasets("/mnt/direct-attached/PHASE2", "/mnt/direct-attached/PHASE2_ANALYSIS")
