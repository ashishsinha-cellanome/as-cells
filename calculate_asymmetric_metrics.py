import numpy as np
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from sklearn.covariance import LedoitWolf
import torch
import gc

def dist_kl_asym_diag(mu_P, mu_Q, var_P, var_Q, eps=1e-8):
    """
    Asymmetric KL Divergence KL(P || Q) for diagonal covariance.
    P is the test/target distribution, Q is the train/source distribution.
    """
    v_P = var_P + eps
    v_Q = var_Q + eps
    
    term1 = np.sum(np.log(v_Q / v_P))
    term2 = np.sum((v_P + (mu_P - mu_Q)**2) / v_Q)
    return 0.5 * (term1 + term2 - len(mu_P))

def compute_coverage_distance(X, Y, k=5):
    """
    Coverage(X, Y) defined as the fraction of X samples whose k-NN ball 
    (computed in X) contains at least one Y sample.
    Returns 1 - Coverage(X, Y) so it acts as a distance (0 = perfect coverage).
    """
    if X.shape[0] < k + 1:
        return 1.0 # Not enough points for k-NN
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_t = torch.tensor(X, device=device)
    Y_t = torch.tensor(Y, device=device)
    
    # Distance to k-th nearest neighbor in X
    # k+1 because the nearest neighbor is the point itself (dist = 0)
    dists_X = torch.cdist(X_t, X_t)
    radii_X = torch.kthvalue(dists_X, k+1, dim=1).values
    
    # Distance to the nearest neighbor in Y
    dists_Y_to_X = torch.cdist(X_t, Y_t)
    min_dists_Y = torch.min(dists_Y_to_X, dim=1).values
    
    # Check if 1-NN in Y is within the k-NN radius in X
    covered = (min_dists_Y <= radii_X).float()
    coverage = torch.mean(covered).item()
    
    return 1.0 - coverage

def generate_clustermap(matrix, names, output_dir, metric_name, model_name):
    os.makedirs(output_dir, exist_ok=True)
    fig_w = max(12, min(36, len(names) * 1.5))
    fig_h = max(10, min(30, len(names) * 1.2))
    
    out_path = os.path.join(output_dir, f"clustermap_{metric_name}.png")
    
    if matrix.shape[0] > 1:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # For asymmetric matrices, calculate row and col linkage separately
            row_linkage = linkage(matrix, method='average')
            col_linkage = linkage(matrix.T, method='average')
            
            g = sns.clustermap(matrix, row_linkage=row_linkage, col_linkage=col_linkage, 
                             xticklabels=names, yticklabels=names, 
                             cmap='viridis', figsize=(fig_w, fig_h), 
                             annot=True, fmt='.3g', annot_kws={'size': 10})
            
            # Note about asymmetry in title
            title = f"{model_name}: Clustermap {metric_name}\nRows=Test(P/X), Cols=Train(Q/Y)"
            g.fig.suptitle(title, fontsize=24, y=1.02)
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

def process_level(level_name, embs_dict, out_dir, model_name, class_name):
    names = sorted(list(embs_dict.keys()))
    num_entities = len(names)
    
    if num_entities < 2:
        return

    stacked_embs = {}
    for name in names:
        stacked = embs_dict[name]
        if stacked.shape[0] > 5000:
            sub_idx = np.random.choice(stacked.shape[0], 5000, replace=False)
            stacked = stacked[sub_idx]
        stacked_embs[name] = stacked
        
    metrics = {
        'kl_divergence_asym': np.zeros((num_entities, num_entities)),
        'coverage_distance': np.zeros((num_entities, num_entities)),
    }
    
    print(f"Computing asymmetric metrics for {model_name} | {class_name} | {level_name}...")
    for i in range(num_entities):
        for j in range(num_entities):
            if i == j: continue
            
            # i is row (Test / P / X), j is col (Train / Q / Y)
            name_P, name_Q = names[i], names[j]
            embs_P, embs_Q = stacked_embs[name_P], stacked_embs[name_Q]
            
            mu_P, var_P = np.mean(embs_P, axis=0), np.var(embs_P, axis=0)
            mu_Q, var_Q = np.mean(embs_Q, axis=0), np.var(embs_Q, axis=0)
            
            # KL(P || Q)
            metrics['kl_divergence_asym'][i, j] = dist_kl_asym_diag(mu_P, mu_Q, var_P, var_Q)
            
            # 1 - Coverage(P, Q) where P is target, Q is source
            metrics['coverage_distance'][i, j] = compute_coverage_distance(embs_P, embs_Q, k=5)
            
    for metric_name, matrix in metrics.items():
        generate_clustermap(matrix, names, out_dir, metric_name, model_name)

def main():
    base_dir = "/mnt/direct-attached/PHASE2_EVAL_RESULTS"
    models = {
        "RF-DETR": "custom_cell_line_embeddings_analysis",
        "DINOv2": "custom_dinov2_cell_line_embeddings_analysis"
    }
    
    # Class map from original scripts
    CLASS_MAP = {0: "cell", 2: "cell-adhered", 3: "soma"}
    
    for model_name, folder in models.items():
        raw_pkl = os.path.join(base_dir, folder, "extracted_raw_embeddings.pkl")
        if not os.path.exists(raw_pkl):
            print(f"Warning: Missing files for {model_name}, skipping.")
            continue
            
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
                cell_line_embs[cl] = np.vstack(cell_line_embs[cl])
                
            out_dir_ds = os.path.join(base_dir, folder, f"class_{class_name}", "asymmetric_metrics_dataset_level")
            process_level("dataset_level", dataset_embs, out_dir_ds, model_name, class_name)
            
            out_dir_cl = os.path.join(base_dir, folder, f"class_{class_name}", "asymmetric_metrics_cell_line_level")
            process_level("cell_line_level", cell_line_embs, out_dir_cl, model_name, class_name)
            
            del dataset_embs
            del cell_line_embs
            gc.collect()
            
        del all_raw_embs
        gc.collect()

if __name__ == "__main__":
    main()
