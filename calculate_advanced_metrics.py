import numpy as np
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
import sys

def dist_wasserstein_diag(mu1, mu2, var1, var2):
    """Diagonal 2-Wasserstein Distance."""
    return np.sum((mu1 - mu2)**2 + (np.sqrt(var1) - np.sqrt(var2))**2)

def dist_sym_kl_diag(mu1, mu2, var1, var2, eps=1e-8):
    """Symmetric KL Divergence (Jeffreys Divergence) for diagonal covariance."""
    v1_safe = var1 + eps
    v2_safe = var2 + eps
    kl1 = np.sum((v1_safe + (mu1 - mu2)**2) / v2_safe)
    kl2 = np.sum((v2_safe + (mu2 - mu1)**2) / v1_safe)
    return 0.5 * (kl1 + kl2 - 2 * len(mu1))

def dist_bhattacharyya_diag(mu1, mu2, var1, var2, eps=1e-8):
    """Diagonal Bhattacharyya Distance."""
    v_mean = (var1 + var2) / 2.0 + eps
    term1 = 0.125 * np.sum((mu1 - mu2)**2 / v_mean)
    v1_safe = var1 + eps
    v2_safe = var2 + eps
    term2 = 0.5 * np.sum(np.log(v_mean / np.sqrt(v1_safe * v2_safe)))
    return term1 + term2

def dist_hellinger_diag(mu1, mu2, var1, var2, eps=1e-8):
    """Hellinger Distance bounded between 0 and 1."""
    db = dist_bhattacharyya_diag(mu1, mu2, var1, var2, eps)
    # cap at 1.0 to avoid floating point issues
    return min(1.0, np.sqrt(1.0 - np.exp(-db)))

def generate_clustermap(matrix, names, output_dir, metric_name, model_name):
    os.makedirs(output_dir, exist_ok=True)
    fig_w = max(12, min(36, len(names) * 1.5))
    fig_h = max(10, min(30, len(names) * 1.2))
    
    out_path = os.path.join(output_dir, f"clustermap_{metric_name}.png")
    if os.path.exists(out_path):
        print(f"Skipping existing {out_path}")
        return
    
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
            out_path = os.path.join(output_dir, f"clustermap_{metric_name}.png")
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {out_path}")

def main():
    base_dir = "/mnt/direct-attached/PHASE2_EVAL_RESULTS"
    models = {
        "RF-DETR": "custom_cell_line_embeddings_analysis",
        "DINOv2": "custom_dinov2_cell_line_embeddings_analysis"
    }
    
    for model_name, folder in models.items():
        stats_pkl = os.path.join(base_dir, folder, "class_cell", "dataset_level", "dataset_statistics.pkl")
        if not os.path.exists(stats_pkl):
            print(f"Warning: {stats_pkl} not found, skipping {model_name}.")
            continue
            
        with open(stats_pkl, 'rb') as f:
            stats = pickle.load(f)
            
        names = sorted(list(stats.keys()))
        num_entities = len(names)
        
        metrics = {
            'wasserstein_diag': np.zeros((num_entities, num_entities)),
            'sym_kl_diag': np.zeros((num_entities, num_entities)),
            'bhattacharyya_diag': np.zeros((num_entities, num_entities)),
            'hellinger_diag': np.zeros((num_entities, num_entities))
        }
        
        print(f"Computing advanced metrics for {model_name}...")
        for i in range(num_entities):
            for j in range(i, num_entities):
                if i == j:
                    continue
                name_A, name_B = names[i], names[j]
                mu1, var1 = stats[name_A]['mu'], stats[name_A]['var']
                mu2, var2 = stats[name_B]['mu'], stats[name_B]['var']
                
                metrics['wasserstein_diag'][i, j] = metrics['wasserstein_diag'][j, i] = dist_wasserstein_diag(mu1, mu2, var1, var2)
                metrics['sym_kl_diag'][i, j] = metrics['sym_kl_diag'][j, i] = dist_sym_kl_diag(mu1, mu2, var1, var2)
                metrics['bhattacharyya_diag'][i, j] = metrics['bhattacharyya_diag'][j, i] = dist_bhattacharyya_diag(mu1, mu2, var1, var2)
                metrics['hellinger_diag'][i, j] = metrics['hellinger_diag'][j, i] = dist_hellinger_diag(mu1, mu2, var1, var2)
                
        out_dir = os.path.join(base_dir, folder, "class_cell", "advanced_metrics")
        for metric_name, matrix in metrics.items():
            generate_clustermap(matrix, names, out_dir, metric_name, model_name)

if __name__ == "__main__":
    main()
