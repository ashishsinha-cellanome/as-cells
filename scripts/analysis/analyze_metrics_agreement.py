import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import os
import pickle
import torch
from scipy.linalg import sqrtm
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import pairwise_distances

def compute_coverage_distance(X, Y, k=10):
    if X.shape[0] < k + 1: return 1.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t, Y_t = torch.tensor(X, device=device), torch.tensor(Y, device=device)
    dists_X = torch.cdist(X_t, X_t)
    radii_X = torch.kthvalue(dists_X, k+1, dim=1).values
    dists_Y_to_X = torch.cdist(X_t, Y_t)
    min_dists_Y = torch.min(dists_Y_to_X, dim=1).values
    covered = (min_dists_Y <= radii_X).float()
    return 1.0 - torch.mean(covered).item()

def compute_mmd_rbf(X, Y, gamma=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t, Y_t = torch.tensor(X, device=device), torch.tensor(Y, device=device)
    XX = torch.cdist(X_t, X_t) ** 2
    YY = torch.cdist(Y_t, Y_t) ** 2
    XY = torch.cdist(X_t, Y_t) ** 2
    K_XX = torch.exp(-gamma * XX).mean()
    K_YY = torch.exp(-gamma * YY).mean()
    K_XY = torch.exp(-gamma * XY).mean()
    return (K_XX + K_YY - 2 * K_XY).item()

def compute_frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)

def compute_l2_var_norm(mu1, var1, mu2, var2):
    return np.linalg.norm(mu1 - mu2) + np.linalg.norm(np.sqrt(var1) - np.sqrt(var2))

def parse_dataset_name(ds_name):
    lower_name = ds_name.lower()
    if any(kw in lower_name for kw in ["suspension", "jurkat", "k562", "nk92", "pbmc", "mousepbmc", "tall104", "raji", "jerat", "tal104"]):
        return None
    # Strip common suffixes for cleaner grouping if needed, but let's keep it as is
    return ds_name

def extract_clusters(dist_matrix, names, num_clusters=4):
    # Make symmetric if not
    sym_dist = (dist_matrix + dist_matrix.T) / 2.0
    np.fill_diagonal(sym_dist, 0)
    Z = linkage(sym_dist[np.triu_indices(len(names), 1)], method='average')
    labels = fcluster(Z, num_clusters, criterion='maxclust')
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters: clusters[label] = []
        clusters[label].append(names[i])
    return list(clusters.values())

def run_analysis(pkl_path, model_name):
    print(f"\n======== {model_name} ========")
    with open(pkl_path, 'rb') as f:
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
    
    stats = {}
    for name, embs in dataset_embs.items():
        stats[name] = {
            'mu': np.mean(embs, axis=0),
            'var': np.var(embs, axis=0),
            'cov': np.cov(embs, rowvar=False)
        }
        
    metrics = {
        'Coverage Distance (k=10)': np.zeros((n, n)),
        'MMD-RBF': np.zeros((n, n)),
        'FID': np.zeros((n, n)),
        'L2-Var-Norm': np.zeros((n, n))
    }
    
    for i in range(n):
        for j in range(n):
            if i == j: continue
            embs_i, embs_j = dataset_embs[names[i]], dataset_embs[names[j]]
            s_i, s_j = stats[names[i]], stats[names[j]]
            
            metrics['Coverage Distance (k=10)'][i, j] = compute_coverage_distance(embs_i, embs_j, k=10)
            metrics['MMD-RBF'][i, j] = compute_mmd_rbf(embs_i, embs_j, gamma=1.0)
            metrics['FID'][i, j] = compute_frechet_distance(s_i['mu'], s_i['cov'], s_j['mu'], s_j['cov'])
            metrics['L2-Var-Norm'][i, j] = compute_l2_var_norm(s_i['mu'], s_i['var'], s_j['mu'], s_j['var'])
            
    for m_name, dist_matrix in metrics.items():
        print(f"\n--- Clusters for {m_name} ---")
        clusters = extract_clusters(dist_matrix, names, num_clusters=4)
        for idx, cluster in enumerate(clusters):
            # Print cleanly
            cleaned = [c.split('_10x_')[0] for c in cluster]
            print(f"Cluster {idx+1}: {cleaned}")

run_analysis("/Users/ashish.sinha/Documents/project/cellanome/as-cells/custom_cell_line_embeddings_analysis/extracted_raw_embeddings.pkl", "RF-DETR")
run_analysis("/Users/ashish.sinha/Documents/project/cellanome/as-cells/custom_dinov2_cell_line_embeddings_analysis/extracted_raw_embeddings.pkl", "DINOv2")
