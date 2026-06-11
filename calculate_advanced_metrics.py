import numpy as np
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.covariance import LedoitWolf

def dist_wasserstein_diag(mu1, mu2, var1, var2):
    return np.sum((mu1 - mu2)**2 + (np.sqrt(var1) - np.sqrt(var2))**2)

def dist_sym_kl_diag(mu1, mu2, var1, var2, eps=1e-8):
    v1_safe, v2_safe = var1 + eps, var2 + eps
    kl1 = np.sum((v1_safe + (mu1 - mu2)**2) / v2_safe)
    kl2 = np.sum((v2_safe + (mu2 - mu1)**2) / v1_safe)
    return 0.5 * (kl1 + kl2 - 2 * len(mu1))

def dist_bhattacharyya_diag(mu1, mu2, var1, var2, eps=1e-8):
    v_mean = (var1 + var2) / 2.0 + eps
    term1 = 0.125 * np.sum((mu1 - mu2)**2 / v_mean)
    v1_safe, v2_safe = var1 + eps, var2 + eps
    term2 = 0.5 * np.sum(np.log(v_mean / np.sqrt(v1_safe * v2_safe)))
    return term1 + term2

def dist_hellinger_diag(mu1, mu2, var1, var2, eps=1e-8):
    db = dist_bhattacharyya_diag(mu1, mu2, var1, var2, eps)
    return min(1.0, np.sqrt(1.0 - np.exp(-db)))

def dist_bhattacharyya_full(mu1, mu2, cov1, cov2):
    cov_sum = (cov1 + cov2) / 2.0
    inv_cov_sum = np.linalg.pinv(cov_sum)
    diff = mu1 - mu2
    term1 = 0.125 * np.dot(diff.T, np.dot(inv_cov_sum, diff))
    _, logdet_sum = np.linalg.slogdet(cov_sum)
    _, logdet1 = np.linalg.slogdet(cov1)
    _, logdet2 = np.linalg.slogdet(cov2)
    term2 = 0.5 * (logdet_sum - 0.5 * (logdet1 + logdet2))
    return term1 + term2

def dist_mahalanobis_pooled(mu1, mu2, cov1, cov2, n1, n2):
    cov_pooled = ((n1 - 1) * cov1 + (n2 - 1) * cov2) / max(1, n1 + n2 - 2)
    inv_cov_pooled = np.linalg.pinv(cov_pooled)
    diff = mu1 - mu2
    return np.sqrt(max(0, np.dot(diff.T, np.dot(inv_cov_pooled, diff))))

def dist_swd(X, Y, num_projections=50):
    dim = X.shape[1]
    projections = np.random.randn(num_projections, dim)
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)
    min_len = min(X.shape[0], Y.shape[0])
    X_sub = X[np.random.choice(X.shape[0], min_len, replace=False)] if X.shape[0] > min_len else X
    Y_sub = Y[np.random.choice(Y.shape[0], min_len, replace=False)] if Y.shape[0] > min_len else Y
    X_proj = np.sort(np.dot(X_sub, projections.T), axis=0)
    Y_proj = np.sort(np.dot(Y_sub, projections.T), axis=0)
    return np.mean((X_proj - Y_proj)**2)

def generate_clustermap(matrix, names, output_dir, metric_name, model_name):
    os.makedirs(output_dir, exist_ok=True)
    fig_w = max(12, min(36, len(names) * 1.5))
    fig_h = max(10, min(30, len(names) * 1.2))
    
    out_path = os.path.join(output_dir, f"clustermap_{metric_name}.png")
    # if os.path.exists(out_path):
    #     print(f"Skipping existing {out_path}")
    #     return
        
    if matrix.shape[0] > 1:
        Z = linkage(matrix, method='average')
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = sns.clustermap(matrix, row_linkage=Z, col_linkage=Z, 
                             xticklabels=names, yticklabels=names, 
                             cmap='viridis', figsize=(fig_w, fig_h), 
                             annot=True, fmt='.3g', annot_kws={'size': 10})
            g.fig.suptitle(f"{model_name}: Clustermap {metric_name} Distance", fontsize=28, y=1.02)
            plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=16)
            plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=16)
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {out_path}")

def parse_dataset_name(ds_name):
    lower_name = ds_name.lower()
    suspension_keywords = ["suspension", "jurkat", "k562", "nk92", "pbmc", "mousepbmc", "tall104", "raji", "jerat", "tal104"]
    for kw in suspension_keywords:
        if kw in lower_name: return None
    clean_name = lower_name
    for ps in ["-adhered", "-uncaged", "-caged", "10x", "1x", "20x"]:
        clean_name = clean_name.replace(ps, "")
    return clean_name.strip("-_")

def main():
    base_dir = "/mnt/direct-attached/PHASE2_EVAL_RESULTS"
    models = {
        "RF-DETR": "custom_cell_line_embeddings_analysis",
        "DINOv2": "custom_dinov2_cell_line_embeddings_analysis"
    }
    
    CLASS_MAP = {0: "cell", 2: "cell-adhered", 3: "soma"}
    
    for model_name, folder in models.items():
        raw_pkl = os.path.join(base_dir, folder, "extracted_raw_embeddings.pkl")
        if not os.path.exists(raw_pkl):
            print(f"Warning: Missing files for {model_name}, skipping.")
            continue
            
        import gc
        
        with open(raw_pkl, 'rb') as f:
            all_raw_embs = pickle.load(f)
            
        for class_id, class_name in CLASS_MAP.items():
            if class_id not in all_raw_embs:
                continue
                
            raw_embs = all_raw_embs[class_id]
            if not raw_embs:
                continue
            
            # 1. Dataset Level
            dataset_embs = {}
            for ds, e in raw_embs.items():
                cl = parse_dataset_name(ds)
                if not cl: continue # Skip suspension datasets
                dataset_embs[ds] = e
                
            # 2. Cell Line Level
            cell_line_embs = {}
            for ds, e in raw_embs.items():
                cl = parse_dataset_name(ds)
                if not cl: continue
                if cl not in cell_line_embs:
                    cell_line_embs[cl] = []
                cell_line_embs[cl].append(e)
                
            for cl in cell_line_embs:
                stacked = np.vstack(cell_line_embs[cl])
                if stacked.shape[0] > 5000:
                    sub_idx = np.random.choice(stacked.shape[0], 5000, replace=False)
                    stacked = stacked[sub_idx]
                cell_line_embs[cl] = stacked
            
            levels = [("dataset_level", dataset_embs), ("cell_line_level", cell_line_embs)]
            
            for level_name, embs_dict in levels:
                print(f"Processing {model_name} | {class_name} at {level_name}...")
                names = sorted(list(embs_dict.keys()))
                num_entities = len(names)
                
                if num_entities < 2:
                    continue
                
                full_covs = {}
                stacked_embs = {}
                for name in names:
                    stacked = embs_dict[name]
                    if stacked.shape[0] > 5000:
                        sub_idx = np.random.choice(stacked.shape[0], 5000, replace=False)
                        stacked = stacked[sub_idx]
                    stacked_embs[name] = stacked
                    
                    if stacked.shape[0] > 1:
                        cov = LedoitWolf().fit(stacked).covariance_
                    else:
                        cov = np.eye(stacked.shape[1]) * 1e-6
                    full_covs[name] = cov
                    
                metrics = {
                    'wasserstein_diag': np.zeros((num_entities, num_entities)),
                    'sym_kl_diag': np.zeros((num_entities, num_entities)),
                    'bhattacharyya_diag': np.zeros((num_entities, num_entities)),
                    'hellinger_diag': np.zeros((num_entities, num_entities)),
                    'bhattacharyya_full': np.zeros((num_entities, num_entities)),
                    'mahalanobis_pooled': np.zeros((num_entities, num_entities)),
                    'sliced_wasserstein': np.zeros((num_entities, num_entities))
                }
                
                print(f"Computing advanced metrics for {model_name} | {class_name} ({level_name})...")
                for i in range(num_entities):
                    for j in range(i, num_entities):
                        if i == j: continue
                        name_A, name_B = names[i], names[j]
                        
                        mu1, var1 = np.mean(stacked_embs[name_A], axis=0), np.var(stacked_embs[name_A], axis=0)
                        mu2, var2 = np.mean(stacked_embs[name_B], axis=0), np.var(stacked_embs[name_B], axis=0)
                        cov1, cov2 = full_covs[name_A], full_covs[name_B]
                        n1, n2 = stacked_embs[name_A].shape[0], stacked_embs[name_B].shape[0]
                        
                        metrics['wasserstein_diag'][i, j] = metrics['wasserstein_diag'][j, i] = dist_wasserstein_diag(mu1, mu2, var1, var2)
                        metrics['sym_kl_diag'][i, j] = metrics['sym_kl_diag'][j, i] = dist_sym_kl_diag(mu1, mu2, var1, var2)
                        metrics['bhattacharyya_diag'][i, j] = metrics['bhattacharyya_diag'][j, i] = dist_bhattacharyya_diag(mu1, mu2, var1, var2)
                        metrics['hellinger_diag'][i, j] = metrics['hellinger_diag'][j, i] = dist_hellinger_diag(mu1, mu2, var1, var2)
                        
                        metrics['bhattacharyya_full'][i, j] = metrics['bhattacharyya_full'][j, i] = dist_bhattacharyya_full(mu1, mu2, cov1, cov2)
                        metrics['mahalanobis_pooled'][i, j] = metrics['mahalanobis_pooled'][j, i] = dist_mahalanobis_pooled(mu1, mu2, cov1, cov2, n1, n2)
                        metrics['sliced_wasserstein'][i, j] = metrics['sliced_wasserstein'][j, i] = dist_swd(stacked_embs[name_A], stacked_embs[name_B])
                        
                out_dir = os.path.join(base_dir, folder, f"class_{class_name}", f"advanced_metrics_v2_{level_name}")
                for metric_name, matrix in metrics.items():
                    generate_clustermap(matrix, names, out_dir, metric_name, model_name)
                    
            # Free memory of the un-subsampled grouped arrays
            del dataset_embs
            del cell_line_embs
            gc.collect()
            
        del all_raw_embs
        gc.collect()

if __name__ == "__main__":
    main()