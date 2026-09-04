import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import os
import pandas as pd
import numpy as np
import textwrap
import matplotlib.pyplot as plt
import seaborn as sns

MODELS_OF_INTEREST = ['yolo', 'yolov26', 'rf_detr_seg']
METRICS = ['mAP@50', 'mAP@50-95']

def load_and_clean_data(csv_paths):
    dfs = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        dfs.append(df)
    
    if not dfs:
        raise ValueError("No valid CSV files provided or found.")
        
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['Model'].isin(MODELS_OF_INTEREST)].copy()
    
    # Handle missing classes denoted by -1.0
    for metric in METRICS:
        if metric in df.columns:
            df[metric] = df[metric].apply(lambda x: x if x >= 0 else 0)
        
    return df

def plot_grouped_bar_charts(df, output_dir):
    sns.set_theme(style="whitegrid")
    classes = df['Class'].unique()
    
    for cls in classes:
        df_cls = df[df['Class'] == cls]
        
        for metric in METRICS:
            plt.figure(figsize=(20, 12))
            ax = sns.barplot(data=df_cls, x='Dataset', y=metric, hue='Model', palette='viridis')  # noqa: F841
            plt.xticks(rotation=45, ha="right")
            
            title_cls = "Overall" if cls == "all" else f'Class: "{cls}"'
            plt.title(f'{title_cls} {metric} Comparison Across Datasets', fontsize=16)
            plt.tight_layout()
            
            safe_metric = metric.replace('@', '_')
            filename = f"bar_overall_{safe_metric}.png" if cls == "all" else f"bar_class_{cls}_{safe_metric}.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=300)
            plt.close()

def plot_spider_charts(df, output_dir):
    classes = df['Class'].unique()
    datasets = df['Dataset'].unique()
    
    # Text wrap dataset names for the spider plot labels
    wrapped_labels = [textwrap.fill(ds, width=15) for ds in datasets]
    
    # Calculate angle for each dataset
    angles = np.linspace(0, 2 * np.pi, len(datasets), endpoint=False).tolist()
    angles += angles[:1] # Close the circle
    
    for cls in classes:
        df_cls = df[df['Class'] == cls]
        
        for metric in METRICS:
            fig, ax = plt.subplots(figsize=(20, 20), subplot_kw=dict(polar=True))  # noqa: F841
            
            for model in MODELS_OF_INTEREST:
                model_data = []
                for ds in datasets:
                    val = df_cls[(df_cls['Model'] == model) & (df_cls['Dataset'] == ds)][metric].values
                    model_data.append(val[0] if len(val) > 0 else 0)
                    
                model_data += model_data[:1] # Close the circle
                
                ax.plot(angles, model_data, linewidth=2, linestyle='solid', label=model)
                ax.fill(angles, model_data, alpha=0.1)
                
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_thetagrids(np.degrees(angles[:-1]), wrapped_labels, fontsize=10)
            
            # Add y-ticks
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
            ax.set_ylim(0, 1)
            
            title_cls = "Overall" if cls == "all" else f'Class: "{cls}"'
            plt.title(f'{title_cls} {metric} Spider Chart', size=20, y=1.1)
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            
            safe_metric = metric.replace('@', '_')
            filename = f"spider_overall_{safe_metric}.png" if cls == "all" else f"spider_class_{cls}_{safe_metric}.png"
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, filename), dpi=300)
            plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize metrics from CSV files.")
    parser.add_argument("csv_files", nargs='+', help="One or more CSV files containing metrics (e.g. all_metrics*.csv)")
    parser.add_argument("--output-dir", default="/mnt/direct-attached/PHASE2_EVAL_RESULTS/plots", help="Directory to save the plots")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    df = load_and_clean_data(args.csv_files)
    print(f"Loaded {len(df)} rows from {len(args.csv_files)} files.")
    plot_grouped_bar_charts(df, args.output_dir)
    plot_spider_charts(df, args.output_dir)
    print(f"All charts generated in {args.output_dir}")