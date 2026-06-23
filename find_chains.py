import numpy as np
import os
import pickle
import torch
from collections import defaultdict
import networkx as nx

def compute_coverage_distance(X, Y, k=10):
    if X.shape[0] < k + 1:
        return 1.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # If no GPU, maybe MPS?
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        
    X_t = torch.tensor(X, device=device)
    Y_t = torch.tensor(Y, device=device)
    
    # Process in batches if out of memory, but 5000 is small
    dists_X = torch.cdist(X_t, X_t)
    radii_X = torch.kthvalue(dists_X, k+1, dim=1).values
    
    dists_Y_to_X = torch.cdist(X_t, Y_t)
    min_dists_Y = torch.min(dists_Y_to_X, dim=1).values
    
    covered = (min_dists_Y <= radii_X).float()
    coverage = torch.mean(covered).item()
    return 1.0 - coverage

def parse_dataset_name(ds_name):
    lower_name = ds_name.lower()
    suspension_keywords = ["suspension", "jurkat", "k562", "nk92", "pbmc", "mousepbmc", "tall104", "raji", "jerat", "tal104"]
    for kw in suspension_keywords:
        if kw in lower_name: return None
    return ds_name

def main():
    base_dir = "/Users/ashish.sinha/Documents/project/cellanome/as-cells/custom_cell_line_embeddings_analysis"
    raw_pkl = os.path.join(base_dir, "extracted_raw_embeddings.pkl")
    
    with open(raw_pkl, 'rb') as f:
        all_raw_embs = pickle.load(f)
        
    class_id = 2 # cell-adhered
    raw_embs = all_raw_embs.get(class_id, {})
    
    dataset_embs = {}
    for ds, e in raw_embs.items():
        if parse_dataset_name(ds):
            # Subsample
            if e.shape[0] > 3000:
                idx = np.random.choice(e.shape[0], 3000, replace=False)
                dataset_embs[ds] = e[idx]
            else:
                dataset_embs[ds] = e
                
    names = sorted(list(dataset_embs.keys()))
    n = len(names)
    
    print(f"Loaded {n} datasets.")
    
    # Try different K values
    for k in [10]: # We can just use K=10 for now
        print(f"\nComputing matrix for K={k}...")
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j: continue
                dist_matrix[i, j] = compute_coverage_distance(dataset_embs[names[i]], dataset_embs[names[j]], k=k)
        
        # Try different thresholds to find interesting multi-level chains
        thresholds = [0.1, 0.15, 0.2, 0.3]
        for threshold in thresholds:
            G = nx.DiGraph()
            for name in names:
                G.add_node(name)
                
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if dist_matrix[i, j] < threshold:
                        G.add_edge(names[i], names[j], weight=dist_matrix[i, j])
                        
            paths = []
            for node in G.nodes():
                for target in G.nodes():
                    if node != target:
                        for p in nx.all_simple_paths(G, source=node, target=target):
                            if len(p) >= 3:
                                paths.append(p)
                                
            if not paths:
                continue
                
            paths = sorted(paths, key=len, reverse=True)
            print(f"\n--- Threshold: distance < {threshold} (Coverage > {1-threshold:.2f}) ---")
            
            seen = set()
            count = 0
            for p in paths:
                frozen_p = tuple(p)
                is_subpath = False
                for s in seen:
                    if ' '.join(frozen_p) in ' '.join(s):
                        is_subpath = True
                        break
                if not is_subpath:
                    distances = [f"{dist_matrix[names.index(p[i]), names.index(p[i+1])]:.3f}" for i in range(len(p)-1)]
                    path_str = ""
                    for i in range(len(p)-1):
                        path_str += f"{p[i]} --({distances[i]})--> "
                    path_str += p[-1]
                    print(f"Chain {count+1} (Length {len(p)}): {path_str}")
                    seen.add(frozen_p)
                    count += 1
                    if count >= 10: break

if __name__ == "__main__":
    main()
