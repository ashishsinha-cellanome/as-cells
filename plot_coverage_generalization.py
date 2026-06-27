import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def plot_metrics(df, metric_col, output_path):
    plt.figure(figsize=(10, 6))
    
    # Handle single or multiple training datasets mapping to sizes
    if 'train_datasets_count' not in df.columns:
        df['train_datasets_count'] = df['Train Datasets'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 1)
    
    sns.scatterplot(
        data=df,
        x='Coverage (Train -> Test)',
        y=metric_col,
        hue='Motif Type',
        size='train_datasets_count',
        sizes=(50, 300),
        alpha=0.7
    )
    
    plt.title(f'{metric_col} vs Coverage')
    plt.xlabel('Coverage (0.0 to 1.0)')
    plt.ylabel(metric_col)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Plot Model Generalization vs Coverage')
    parser.add_argument('--results_csv', type=str, required=True, help='CSV file with evaluation metrics and coverage')
    parser.add_argument('--output_dir', type=str, default='coverage_plots', help='Directory to save plots')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Assuming CSV has columns: 'Config Name', 'Motif Type', 'Train Datasets', 'Test Datasets', 'Coverage (Train -> Test)', 
    # 'mAP@0.5', 'mAP@0.5:0.95', 'F1 avg', 'mAP@0.5_class0', etc.
    try:
        df = pd.read_csv(args.results_csv)
    except FileNotFoundError:
        print(f"ERROR: Results CSV not found: {args.results_csv}")
        print("Please run your evaluations and compile the results into this CSV format first.")
        return

    metrics_to_plot = ['mAP@0.5', 'mAP@0.5:0.95', 'F1 avg']
    
    # Add class-specific metrics if they exist in the CSV
    class_metrics = [c for c in df.columns if ('mAP' in c or 'F1' in c) and ('class' in c)]
    metrics_to_plot.extend(class_metrics)

    for metric in metrics_to_plot:
        if metric in df.columns:
            output_file = os.path.join(args.output_dir, f"{metric.replace('@', '_').replace(':', '_')}_vs_coverage.png")
            plot_metrics(df, metric, output_file)
        else:
            print(f"Warning: Metric '{metric}' not found in CSV. Skipping.")

if __name__ == '__main__':
    main()