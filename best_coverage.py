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

with open("/Users/ashish.sinha/Documents/project/cellanome/as-cells/custom_cell_line_embeddings_analysis/extracted_raw_embeddings.pkl", 'rb') as f:
    all_raw_embs = pickle.load(f)

raw_embs = all_raw_embs.get(2, {})
dataset_embs = {}
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

# For each dataset, print its best covering set (min distance in row i, excluding i)
# i is the target (X), j is the source (Y). dist_matrix[i, j] = 1 - Coverage(X, Y)
# We want the Y that best covers X. So for row i, find min over j.

for i in range(n):
    best_j = np.argmin(dist_matrix[i, :])
    if best_j == i: 
        # ignore diag, find next
        dists = dist_matrix[i, :].copy()
        dists[i] = np.inf
        best_j = np.argmin(dists)
    
    print(f"{names[i]} is best covered by {names[best_j]} (dist={dist_matrix[i, best_j]:.3f})")

print("\n--- Reverse: What does this dataset best cover? ---")
for j in range(n):
    dists = dist_matrix[:, j].copy()
    dists[j] = np.inf
    best_i = np.argmin(dists)
    print(f"{names[j]} best covers {names[best_i]} (dist={dist_matrix[best_i, j]:.3f})")

