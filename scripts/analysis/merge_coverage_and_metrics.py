import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import argparse
import sys
import os

def clean_dataset_name(name):
    # Strip common suffixes that might differ between the two CSVs
    name = str(name).replace('_4_class', '').replace('_multichannel_overlay', '')
    return name

def main():
    parser = argparse.ArgumentParser(description="Merge metrics_tidy.csv with coverage_distance_mehdi.csv")
    parser.add_argument("--metrics_csv", default="metrics_tidy.csv")
    parser.add_argument("--coverage_csv", default="coverage_distance_mehdi.csv")
    parser.add_argument("--in-domain", default="a549", help="Substring identifying the training dataset")
    parser.add_argument("--out_csv", default="merged_coverage_metrics.csv")
    args = parser.parse_args()

    if not os.path.exists(args.metrics_csv):
        sys.exit(f"ERROR: {args.metrics_csv} not found.")
    if not os.path.exists(args.coverage_csv):
        sys.exit(f"ERROR: {args.coverage_csv} not found.")

    # Load metrics
    df_metrics = pd.read_csv(args.metrics_csv)
    # We only want 'test_ds' splits and 'all' class for top-level generalization
    df_metrics = df_metrics[(df_metrics['split_type'] == 'test_ds') & (df_metrics['class'] == 'all')].copy()

    # Load coverage distance matrix
    # The first column is unnamed, use it as index
    df_cov = pd.read_csv(args.coverage_csv, index_col=0)
    
    # Identify the exact row name in the coverage matrix for the in-domain dataset
    train_ds_cov_name = None
    for idx in df_cov.index:
        if args.in_domain.lower() in str(idx).lower():
            train_ds_cov_name = idx
            break
            
    if not train_ds_cov_name:
        sys.exit(f"ERROR: Could not find any dataset matching '{args.in_domain}' in {args.coverage_csv}")
    
    print(f"Matched in-domain train dataset in coverage matrix: {train_ds_cov_name}")

    merged_data = []

    for _, row in df_metrics.iterrows():
        test_ds_raw = row['dataset']
        # Metric types are separate rows (BBOX vs SEGM). We will treat them as separate entries or separate metrics.
        # To make it compatible with plot_coverage_generalization.py, we'll pivot or just keep them in rows.
        # Actually, let's keep it simple: just output BBOX for now, or rename columns.
        if row['metric_type'] != 'BBOX':
            continue
            
        test_ds_clean = clean_dataset_name(test_ds_raw)
        
        # Try to find the matching column in coverage matrix
        # Note: The matrix might not have exactly matching names if there are other variations
        matched_col = None
        for col in df_cov.columns:
            if clean_dataset_name(col) == test_ds_clean or col in test_ds_raw:
                matched_col = col
                break
                
        if not matched_col:
            print(f"Warning: Could not find coverage data for test dataset: {test_ds_raw} (cleaned: {test_ds_clean})")
            continue
            
        # Distance from train to test
        distance = df_cov.loc[train_ds_cov_name, matched_col]
        coverage = 1.0 - distance
        
        merged_data.append({
            'Config Name': f"RF-DETR Trained on {args.in_domain}",
            'Motif Type': 'Unknown', # If we had motif info we'd add it here
            'Train Datasets': train_ds_cov_name,
            'Test Datasets': test_ds_raw,
            'Coverage (Train -> Test)': coverage,
            'mAP@0.5': row['mAP50'],
            'mAP@0.5:0.95': row['mAP50_95'],
            'F1 avg': row['F1']
        })

    df_merged = pd.DataFrame(merged_data)
    df_merged.to_csv(args.out_csv, index=False)
    print(f"Successfully merged {len(df_merged)} test datasets with coverage scores.")
    print(f"Saved to {args.out_csv}")
    print(f"\nYou can now run:\nuv run python3 plot_coverage_generalization.py --results_csv {args.out_csv}")

if __name__ == "__main__":
    main()
