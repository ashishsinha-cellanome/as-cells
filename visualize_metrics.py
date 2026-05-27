import os
import pandas as pd
import numpy as np  # noqa: F401
import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH = "/mnt/direct-attached/PHASE2_EVAL_RESULTS/all_models_summary.csv"
OUTPUT_DIR = "/mnt/direct-attached/PHASE2_EVAL_RESULTS/plots"
MODELS_OF_INTEREST = ['yolo', 'yolov26', 'rf_detr_seg']
METRICS = ['mAP@50', 'mAP@50-95']

def load_and_clean_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df['Model'].isin(MODELS_OF_INTEREST)].copy()
    
    # Handle missing classes denoted by -1.0
    for metric in METRICS:
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

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_and_clean_data(CSV_PATH)
    print(f"Loaded {len(df)} rows.")
    plot_grouped_bar_charts(df, OUTPUT_DIR)
    print(f"Bar charts generated in {OUTPUT_DIR}")