import numpy as np
import os
import pickle
import torch
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform

def compute_coverage_distance(X, Y, k=5):
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
    coverage = torch.mean(covered).item()
    return 1.0 - coverage

def parse_dataset_name(ds_name):
    lower_name = ds_name.lower()
    suspension_keywords = ["suspension", "jurkat", "k562", "nk92", "pbmc", "mousepbmc", "tall104", "raji", "jerat", "tal104"]
    for kw in suspension_keywords:
        if kw in lower_name: return None
    return ds_name

def analyze_hierarchical_topology(raw_embs_pkl_path, k_values, out_dir="topology_hierarchical_output"):
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading {raw_embs_pkl_path}...")
    with open(raw_embs_pkl_path, 'rb') as f:
        all_raw_embs = pickle.load(f)
        
    class_id = 2 # cell-adhered
    raw_embs = all_raw_embs.get(class_id, {})
    
    dataset_embs = {}
    for ds, e in raw_embs.items():
        parsed_name = parse_dataset_name(ds)
        if parsed_name:
            if e.shape[0] > 3000:
                idx = np.random.choice(e.shape[0], 3000, replace=False)
                dataset_embs[parsed_name] = e[idx]
            else:
                dataset_embs[parsed_name] = e
                
    names = sorted(list(dataset_embs.keys()))
    n = len(names)
    
    report_lines = ["# Hierarchical Clustering Topology Report\n"]
    report_lines.append("This report constructs a tree based on standard Agglomerative Hierarchical Clustering (what seaborn.clustermap and scipy dendrograms use).")
    report_lines.append("Because linkage requires symmetric distances, we use the average coverage between the two directions: `(Coverage(X->Y) + Coverage(Y->X)) / 2`.")
    report_lines.append("Each internal node shows the average mutual coverage between its two merged sub-clusters.\n")

    for k in k_values:
        print(f"\n--- Analyzing K={k} ---")
        report_lines.append(f"## Hierarchical Tree for K={k}\n")
        
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j: continue
                dist_matrix[i, j] = compute_coverage_distance(dataset_embs[names[i]], dataset_embs[names[j]], k=k)
                
        # Make symmetric for linkage
        sym_dist = (dist_matrix + dist_matrix.T) / 2.0
        # Ensure exact 0s on diagonal
        np.fill_diagonal(sym_dist, 0)
        
        # Condensed distance matrix
        condensed_dist = squareform(sym_dist)
        
        # Perform hierarchical clustering (average linkage)
        Z = linkage(condensed_dist, method='average')
        
        root_node, node_list = to_tree(Z, rd=True)
        
        report_lines.append("```text")
        
        def print_tree(node, prefix=""):
            if node.is_leaf():
                return
            
            left = node.get_left()
            right = node.get_right()
            
            children = [left, right]
            # sort children by size or something so tree looks nice
            children.sort(key=lambda c: getattr(c, 'count', 0), reverse=True)
            
            for i, child in enumerate(children):
                is_last = (i == len(children) - 1)
                connector = "└── " if is_last else "├── "
                
                if child.is_leaf():
                    name = names[child.id]
                    report_lines.append(f"{prefix}{connector}[LEAF] `{name}`")
                else:
                    child_cov = 100 * (1 - child.dist)
                    report_lines.append(f"{prefix}{connector}[CLUSTER] (Merged at {child_cov:.1f}% mutual coverage)")
                    extension = "    " if is_last else "│   "
                    print_tree(child, prefix + extension)
        
        root_cov = 100 * (1 - root_node.dist)
        report_lines.append(f"[GLOBAL ROOT] (Merged at {root_cov:.1f}% mutual coverage)")
        print_tree(root_node)
        report_lines.append("```\n")
        
    report_path = os.path.join(out_dir, "hierarchical_topology_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Saved report to {report_path}")

def main():
    print("=== RF-DETR Hierarchical Analysis ===")
    rf_pkl = "/Users/ashish.sinha/Documents/project/cellanome/as-cells/custom_cell_line_embeddings_analysis/extracted_raw_embeddings.pkl"
    if os.path.exists(rf_pkl):
        analyze_hierarchical_topology(rf_pkl, k_values=[5, 10, 15, 30], out_dir="topology_hierarchical_output/rfdetr")
    
    print("\n=== DINOv2 Hierarchical Analysis ===")
    dino_pkl = "/Users/ashish.sinha/Documents/project/cellanome/as-cells/custom_dinov2_cell_line_embeddings_analysis/extracted_raw_embeddings.pkl"
    if os.path.exists(dino_pkl):
        analyze_hierarchical_topology(dino_pkl, k_values=[5, 10, 15, 30], out_dir="topology_hierarchical_output/dinov2")

if __name__ == "__main__":
    main()