import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd

df = pd.read_csv("generalization_tracking.csv")
df = df[~df['dataset'].astype(str).str.startswith('#')]
df = df[df['split_type'] == 'test_ds']

baseline = df[df['experiment'] == 'Baseline'][['dataset', 'mAP50_95']].set_index('dataset')

results = []
for exp in df['experiment'].unique():
    if exp == 'Baseline': continue
    
    exp_df = df[df['experiment'] == exp][['dataset', 'mAP50_95']].set_index('dataset')
    
    # Calculate relative performance
    rel_perf = (exp_df['mAP50_95'] / baseline['mAP50_95']) * 100
    
    # Calculate mean and standard deviation
    results.append({
        'experiment': exp,
        'mean_relative_perf': rel_perf.mean(),
        'std_relative_perf': rel_perf.std()
    })

results_df = pd.DataFrame(results).sort_values('mean_relative_perf', ascending=False)
print(results_df.to_string(index=False, float_format="%.2f"))
