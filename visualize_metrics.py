import os
import pandas as pd
import numpy as np  # noqa: F401
import matplotlib.pyplot as plt  # noqa: F401
import seaborn as sns  # noqa: F401

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

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_and_clean_data(CSV_PATH)
    print(f"Loaded {len(df)} rows.")
    print(f"Classes found: {df['Class'].unique()}")