import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pickle
import numpy as np
import torch
import networkx as nx

def compute_coverage_distance(X, Y, k=10):
    if X.shape[0] < k + 1:
        return 1.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    X_t = torch.tensor(X, device=device)
    Y_t = torch.tensor(Y, device=device)
    dists_X = torch.cdist(X_t, X_t)
    radii_X = torch.kthvalue(dists_X, k+1, dim=1).values
    dists_Y_to_X = torch.cdist(X_t, Y_t)
    min_dists_Y = torch.min(dists_Y_to_X, dim=1).values
    covered = (min_dists_Y <= radii_X).float()
    return 1.0 - torch.mean(covered).item()

def parse_dataset_name(ds_name):
    lower_name = ds_name.lower()
    suspension_keywords = ["suspension", "jurkat", "k562", "nk92", "pbmc", "mousepbmc", "tall104", "raji", "jerat", "tal104"]
    for kw in suspension_keywords:
        if kw in lower_name: return None
    return ds_name

import sys
import os
import argparse

parser = argparse.ArgumentParser(description='Run Tree Topology Analysis')
parser.add_argument('--input_pkl', type=str, required=True, help='Path to the raw embeddings pickle file')
parser.add_argument('--output_prefix', type=str, required=True, help='Prefix for output files (e.g. dinov2_base)')
args = parser.parse_args()

try:
    with open(args.input_pkl, 'rb') as f:
        all_raw_embs = pickle.load(f)
except FileNotFoundError:
    print(f"ERROR: Embeddings file not found: {args.input_pkl}")
    sys.exit(1)

raw_embs_class2 = all_raw_embs.get(2, {})
raw_embs_class3 = all_raw_embs.get(3, {})
raw_embs = {**raw_embs_class2, **raw_embs_class3}

dataset_embs = {}
np.random.seed(42)
for ds, e in raw_embs.items():
    if parse_dataset_name(ds):
        if e.shape[0] > 3000:
            idx = np.random.choice(e.shape[0], 3000, replace=False)
            dataset_embs[ds] = e[idx]
        else:
            dataset_embs[ds] = e

names = sorted(list(dataset_embs.keys()))
n = len(names)

dist_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i == j: continue
        dist_matrix[i, j] = compute_coverage_distance(dataset_embs[names[i]], dataset_embs[names[j]], k=10)

from coverage_arborescence_viz import visualize_arborescence, coverage_report
import sys
import contextlib

# Save the original stdout
original_stdout = sys.stdout

thresholds = [0.4, 0.5, 0.6]

print("\n--- Coverage Arborescence Analysis ---")

for thresh in thresholds:
    print(f"\nProcessing Threshold: {thresh}")
    
    report_path = f"coverage_arborescence_report_{args.output_prefix}_t{thresh}.txt"
    with open(report_path, 'w') as f:
        with contextlib.redirect_stdout(f):
            coverage_report(dist_matrix, names, threshold=thresh)
    print(f"Saved report to {report_path}")
    
    # Generate the visualization
    plot_path = f"coverage_arborescence_plot_{args.output_prefix}_t{thresh}.png"
    fig, G = visualize_arborescence(dist_matrix, names, threshold=thresh, save_path=plot_path)

