import numpy as np
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.covariance import LedoitWolf
from sklearn.manifold import TSNE
import umap.umap_ as umap
import torch
import gc
from scipy.linalg import sqrtm
from sklearn.metrics import pairwise_distances
import re

def dist_l2_var_norm(mu_A, mu_B, var_A, var_B, eps=1e-8):
    num = np.linalg.norm(mu_A - mu_B)
    den = np.sqrt(np.mean(var_A)) + np.sqrt(np.mean(var_B)) + eps
    return num / den

def dist_fid(mu_A, mu_B, cov_A, cov_B):
    diff = mu_A - mu_B
    covmean = sqrtm(cov_A.dot(cov_B))
    if not np.isfinite(covmean).all():
        offset = np.eye(cov_A.shape[0]) * 1e-6
        covmean = sqrtm((cov_A + offset).dot(cov_B + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(cov_A) + np.trace(cov_B) - 2 * tr_covmean

def dist_mmd_rbf(X, Y):
    if X.shape[0] > 5000:
        indices = np.random.choice(X.shape[0], 5000, replace=False)
        X = X[indices]
    if Y.shape[0] > 5000:
        indices = np.random.choice(Y.shape[0], 5000, replace=False)
        Y = Y[indices]
    pooled = np.vstack([X, Y])
    sub_pooled = pooled
    if pooled.shape[0] > 1000:
        sub_pooled = pooled[np.random.choice(pooled.shape[0], 1000, replace=False)]
    dists = pairwise_distances(sub_pooled, sub_pooled, metric='euclidean')
    sigma_m = np.median(dists[dists > 0])
    if sigma_m == 0:
        sigma_m = 1.0
    sigmas = [0.5 * sigma_m, sigma_m, 2.0 * sigma_m]
    
    dist_XX = pairwise_distances(X, X, metric='sqeuclidean')
    dist_YY = pairwise_distances(Y, Y, metric='sqeuclidean')
    dist_XY = pairwise_distances(X, Y, metric='sqeuclidean')
    
    mmd_val = 0
    for s in sigmas:
        K_XX = np.exp(-dist_XX / (2 * s**2))
        K_YY = np.exp(-dist_YY / (2 * s**2))
        K_XY = np.exp(-dist_XY / (2 * s**2))
        mmd_val += np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    return mmd_val / 3.0

def dist_kl_asym_diag(mu_P, mu_Q, var_P, var_Q, eps=1e-8):
    v_P = var_P + eps
    v_Q = var_Q + eps
    term1 = np.sum(np.log(v_Q / v_P))
    term2 = np.sum((v_P + (mu_P - mu_Q)**2) / v_Q)
    return 0.5 * (term1 + term2 - len(mu_P))

def compute_coverage(X, Y, k=5):
    if X.shape[0] < k + 1:
        return 0.0
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t = torch.tensor(X, device=device)
    Y_t = torch.tensor(Y, device=device)
    
    dists_X = torch.cdist(X_t, X_t)
    radii_X = torch.kthvalue(dists_X, k+1, dim=1).values
    
    dists_Y_to_X = torch.cdist(X_t, Y_t)
    min_dists_Y = torch.min(dists_Y_to_X, dim=1).values
    
    covered = (min_dists_Y <= radii_X).float()
    coverage = torch.mean(covered).item()
    
    return coverage

def generate_heatmap(matrix, names, output_dir, metric_name, model_name, is_asymmetric=False):
    os.makedirs(output_dir, exist_ok=True)
    fig_w = max(12, min(36, len(names) * 1.5))
    fig_h = max(10, min(30, len(names) * 1.2))
    
    out_path = os.path.join(output_dir, f"heatmap_{metric_name}.png")
    
    if matrix.shape[0] > 1:
        plt.figure(figsize=(fig_w, fig_h), dpi=300)
        sns.heatmap(matrix, xticklabels=names, yticklabels=names, 
                    cmap='viridis', annot=True, fmt='.3g', annot_kws={'size': 10})
        
        title = f"{model_name}: Heatmap {metric_name}"
        if is_asymmetric:
            title += "\nRows=Test(P/X), Cols=Train(Q/Y)"
            
        plt.title(title, fontsize=24, pad=20)
        plt.xticks(rotation=90, fontsize=16)
        plt.yticks(rotation=0, fontsize=16)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {out_path}")

def generate_clustermap(matrix, names, output_dir, metric_name, model_name, is_asymmetric=False):
    os.makedirs(output_dir, exist_ok=True)
    fig_w = max(12, min(36, len(names) * 1.5))
    fig_h = max(10, min(30, len(names) * 1.2))
    
    out_path = os.path.join(output_dir, f"clustermap_{metric_name}.png")
    
    if matrix.shape[0] > 1:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if is_asymmetric:
                row_linkage = linkage(matrix, method='average')
                col_linkage = linkage(matrix.T, method='average')
                g = sns.clustermap(matrix, row_linkage=row_linkage, col_linkage=col_linkage, 
                                 xticklabels=names, yticklabels=names, 
                                 cmap='viridis', figsize=(fig_w, fig_h), 
                                 annot=True, fmt='.3g', annot_kws={'size': 10})
                title = f"{model_name}: {metric_name}\nRows=Test(P/X), Cols=Train(Q/Y)"
                
                plt.figure(figsize=(fig_w, fig_h), dpi=300)
                dendrogram(row_linkage, labels=names, leaf_rotation=90, leaf_font_size=16)
                plt.title(f"{model_name}: Dendrogram {metric_name} (Test/Row)", fontsize=28, pad=20)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"dendrogram_{metric_name}_test.png"), dpi=300, bbox_inches='tight')
                plt.close()
                
                plt.figure(figsize=(fig_w, fig_h), dpi=300)
                dendrogram(col_linkage, labels=names, leaf_rotation=90, leaf_font_size=16)
                plt.title(f"{model_name}: Dendrogram {metric_name} (Train/Col)", fontsize=28, pad=20)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"dendrogram_{metric_name}_train.png"), dpi=300, bbox_inches='tight')
                plt.close()
            else:
                Z = linkage(matrix, method='average')
                g = sns.clustermap(matrix, row_linkage=Z, col_linkage=Z, 
                                 xticklabels=names, yticklabels=names, 
                                 cmap='viridis', figsize=(fig_w, fig_h), 
                                 annot=True, fmt='.3g', annot_kws={'size': 10})
                title = f"{model_name}: {metric_name}"
                
                plt.figure(figsize=(fig_w, fig_h), dpi=300)
                dendrogram(Z, labels=names, leaf_rotation=90, leaf_font_size=16)
                plt.title(f"{model_name}: Dendrogram {metric_name}", fontsize=28, pad=20)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"dendrogram_{metric_name}.png"), dpi=300, bbox_inches='tight')
                plt.close()
                
            g.fig.suptitle(title, fontsize=24, y=1.02)
            plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=16)
            plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=16)
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {out_path}")

def generate_scatter_plots(features, labels, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    n_samples = features.shape[0]
    if n_samples < 2: return
    
    unique_labels = list(np.unique(labels))
    num_labels = len(unique_labels)
    
    base_colors = sns.color_palette("colorblind")
    palette = [base_colors[i % len(base_colors)] for i in range(num_labels)]
    
    all_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', 'X', 'd']
    markers_dict = {label: all_markers[i % len(all_markers)] for i, label in enumerate(unique_labels)}
    
    # t-SNE
    for p in [30, 50]:
        if n_samples - 1 < p: continue
        print(f"Running t-SNE for {prefix} (perp={p})...")
        tsne_2d = TSNE(n_components=2, perplexity=p, random_state=42).fit_transform(features)
        plt.figure(figsize=(14, 10), dpi=300)
        sns.scatterplot(x=tsne_2d[:, 0], y=tsne_2d[:, 1], hue=labels, style=labels, palette=palette, markers=markers_dict, s=10, alpha=0.8, edgecolor="none")
        plt.title(f"t-SNE (perp={p})", fontsize=20)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"scatter_tsne_perp{p}.png"), bbox_inches='tight')
        plt.close()
    
    # UMAP
    print(f"Running UMAP for {prefix}...")
    n_neighbors = min(15, n_samples - 1)
    umap_2d = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42).fit_transform(features)
    plt.figure(figsize=(12, 10), dpi=300)
    sns.scatterplot(x=umap_2d[:, 0], y=umap_2d[:, 1], hue=labels, palette=palette, s=15, alpha=0.8, edgecolor="none")
    plt.title(f"UMAP (n_neighbors={n_neighbors})", fontsize=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"scatter_umap_nn{n_neighbors}.png"), bbox_inches='tight')
    plt.close()

def parse_dataset_name(ds_name):
    lower_name = ds_name.lower()
    suspension_keywords = ["suspension", "jurkat", "k562", "nk92", "pbmc", "mousepbmc", "tall104", "raji", "jerat", "tal104"]
    for kw in suspension_keywords:
        if kw in lower_name: return None
    clean_name = lower_name
    
    # Remove leading date
    clean_name = re.sub(r'^\d{6,8}_', '', clean_name)
    # Remove trailing _4_class
    clean_name = clean_name.replace('_4_class', '')
    
    # Strip caging, magnification, and adherence
    to_remove = ["-adhered", "-adherent", "_10x", "10x_", "_1x", "1x_", "_20x", "20x_", "_at_4x", "at_4x", "_caged", "-caged", "caged", "_uncaged", "-uncaged", "uncaged"]
    for word in to_remove:
        clean_name = clean_name.replace(word, "")
        
    return clean_name.strip("-_")

def process_level(level_name, embs_dict, out_dir, model_name, class_name, coverage_subsample=10000, scatter_subsample=5000):
    names = sorted(list(embs_dict.keys()))
    num_entities = len(names)
    
    if num_entities < 2: return

    stats = {}
    stacked_embs = {}
    
    all_features = []
    all_labels = []
    
    for name in names:
        full_embs = embs_dict[name]
        
        mu = np.mean(full_embs, axis=0)
        var = np.var(full_embs, axis=0)
        if full_embs.shape[0] > 1:
            cov = LedoitWolf().fit(full_embs).covariance_
        else:
            cov = np.eye(full_embs.shape[1]) * 1e-6
        stats[name] = {'mu': mu, 'var': var, 'cov': cov}
        
        if full_embs.shape[0] > coverage_subsample:
            sub_idx = np.random.choice(full_embs.shape[0], coverage_subsample, replace=False)
            stacked_embs[name] = full_embs[sub_idx]
        else:
            stacked_embs[name] = full_embs
            
        if full_embs.shape[0] > scatter_subsample:
            sc_idx = np.random.choice(full_embs.shape[0], scatter_subsample, replace=False)
            all_features.append(full_embs[sc_idx])
        else:
            all_features.append(full_embs)
            
        all_labels.extend([name] * min(full_embs.shape[0], scatter_subsample))
        
    metrics_sym = {
        'l2_var_norm': np.zeros((num_entities, num_entities)),
        'fid': np.zeros((num_entities, num_entities)),
        'mmd_rbf': np.zeros((num_entities, num_entities))
    }
    
    metrics_asym = {
        'kl_divergence_asym': np.zeros((num_entities, num_entities)),
        'coverage_k5': np.zeros((num_entities, num_entities)),
        'coverage_distance_k5': np.zeros((num_entities, num_entities)),
        'coverage_k10': np.zeros((num_entities, num_entities)),
        'coverage_distance_k10': np.zeros((num_entities, num_entities)),
        'coverage_k15': np.zeros((num_entities, num_entities)),
        'coverage_distance_k15': np.zeros((num_entities, num_entities)),
        'coverage_k30': np.zeros((num_entities, num_entities)),
        'coverage_distance_k30': np.zeros((num_entities, num_entities))
    }
    
    print(f"Computing metrics for {model_name} | {class_name} | {level_name}...")
    for i in range(num_entities):
        for j in range(num_entities):
            if i == j: continue
            
            name_P, name_Q = names[i], names[j]
            embs_P, embs_Q = stacked_embs[name_P], stacked_embs[name_Q]
            
            mu_P, var_P = stats[name_P]['mu'], stats[name_P]['var']
            mu_Q, var_Q = stats[name_Q]['mu'], stats[name_Q]['var']
            cov_P, cov_Q = stats[name_P]['cov'], stats[name_Q]['cov']
            
            if i < j: # Compute symmetric metrics only once per pair
                metrics_sym['l2_var_norm'][i, j] = metrics_sym['l2_var_norm'][j, i] = dist_l2_var_norm(mu_P, mu_Q, var_P, var_Q)
                metrics_sym['fid'][i, j] = metrics_sym['fid'][j, i] = dist_fid(mu_P, mu_Q, cov_P, cov_Q)
                metrics_sym['mmd_rbf'][i, j] = metrics_sym['mmd_rbf'][j, i] = dist_mmd_rbf(embs_P, embs_Q)
                
            # Asymmetric metrics
            metrics_asym['kl_divergence_asym'][i, j] = dist_kl_asym_diag(mu_P, mu_Q, var_P, var_Q)
            for k_val in [5, 10, 15, 30]:
                cov_val = compute_coverage(embs_P, embs_Q, k=k_val)
                metrics_asym[f'coverage_k{k_val}'][i, j] = cov_val
                metrics_asym[f'coverage_distance_k{k_val}'][i, j] = 1.0 - cov_val
            
    # Save Clustermaps and Heatmaps, and the Raw Matrices
    for metric_name, matrix in metrics_sym.items():
        np.save(os.path.join(out_dir, f"matrix_{metric_name}.npy"), matrix)
        generate_clustermap(matrix, names, out_dir, metric_name, model_name, is_asymmetric=False)
        generate_heatmap(matrix, names, out_dir, metric_name, model_name, is_asymmetric=False)
        
    for metric_name, matrix in metrics_asym.items():
        np.save(os.path.join(out_dir, f"matrix_{metric_name}.npy"), matrix)
        generate_clustermap(matrix, names, out_dir, metric_name, model_name, is_asymmetric=True)
        generate_heatmap(matrix, names, out_dir, metric_name, model_name, is_asymmetric=True)
        
    # Save the names mapping so topology scripts know which row/col corresponds to which dataset/cell-line
    with open(os.path.join(out_dir, "entity_names.txt"), "w") as f:
        f.write("\n".join(names))

    # Save Scatter plots directly in the same folder under a scatter_plots subdirectory
    scatter_dir = os.path.join(out_dir, "scatter_plots")
    all_features = np.vstack(all_features)
    generate_scatter_plots(all_features, all_labels, scatter_dir, f"{model_name}_{class_name}_{level_name}")

def main():
    base_dir = "/mnt/direct-attached/PHASE2_EVAL_RESULTS"
    models = {
        "RF-DETR": "custom_cell_line_embeddings_analysis_train",
        "DINOv2": "custom_dinov2_cell_line_embeddings_analysis_train"
    }
    
    for model_name, folder in models.items():
        raw_pkl = os.path.join(base_dir, folder, "extracted_raw_embeddings.pkl")
        if not os.path.exists(raw_pkl):
            print(f"Warning: Missing files for {model_name}, skipping.")
            continue
            
        with open(raw_pkl, 'rb') as f:
            all_raw_embs = pickle.load(f)
            
        dataset_embs = {}
        all_ds_names = set()
        for cls_id in [2, 3]:
            if cls_id in all_raw_embs:
                all_ds_names.update(all_raw_embs[cls_id].keys())
        
        for ds in sorted(list(all_ds_names)):
            if "neuron" in ds.lower():
                if 3 in all_raw_embs and ds in all_raw_embs[3] and all_raw_embs[3][ds].shape[0] > 0:
                    dataset_embs[ds] = all_raw_embs[3][ds]
            else:
                if 2 in all_raw_embs and ds in all_raw_embs[2] and all_raw_embs[2][ds].shape[0] > 0:
                    dataset_embs[ds] = all_raw_embs[2][ds]
                    
        # Filter suspension
        dataset_embs_filtered = {}
        for ds, e in dataset_embs.items():
            if parse_dataset_name(ds) is not None:
                dataset_embs_filtered[ds] = e
                
        if not dataset_embs_filtered:
            print(f"No valid datasets found for {model_name}.")
            continue
            
        class_name = "cell-adhered" # Using fixed name for downstream compatibility
        
        out_dir_ds = os.path.join(base_dir, folder, f"class_{class_name}", "dataset_level")
        process_level("dataset_level", dataset_embs_filtered, out_dir_ds, model_name, class_name)
        
        cell_line_embs = {}
        for ds, e in dataset_embs_filtered.items():
            cl = parse_dataset_name(ds)
            if not cl: continue
            if cl not in cell_line_embs:
                cell_line_embs[cl] = []
            cell_line_embs[cl].append(e)
            
        for cl in cell_line_embs:
            cell_line_embs[cl] = np.vstack(cell_line_embs[cl])
            
        out_dir_cl = os.path.join(base_dir, folder, f"class_{class_name}", "cell_line_level")
        process_level("cell_line_level", cell_line_embs, out_dir_cl, model_name, class_name)
        
        del all_raw_embs
        gc.collect()

if __name__ == "__main__":
    main()
