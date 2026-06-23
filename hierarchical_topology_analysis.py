import numpy as np
import os
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform

def analyze_hierarchical_topology(base_metrics_dir, k_values, out_dir="topology_hierarchical_output"):
    os.makedirs(out_dir, exist_ok=True)
    
    names_path = os.path.join(base_metrics_dir, "entity_names.txt")
    if not os.path.exists(names_path):
        print(f"Warning: {names_path} not found.")
        return
        
    with open(names_path, "r") as f:
        names = [line.strip() for line in f if line.strip()]
        
    n = len(names)
    
    report_lines = ["# Hierarchical Clustering Topology Report\n"]
    report_lines.append("This report constructs a tree based on standard Agglomerative Hierarchical Clustering (what seaborn.clustermap and scipy dendrograms use).")
    report_lines.append("Because linkage requires symmetric distances, we use the average coverage between the two directions: `(Coverage(X->Y) + Coverage(Y->X)) / 2`.")
    report_lines.append("Each internal node shows the average mutual coverage between its two merged sub-clusters.\n")

    for k in k_values:
        matrix_path = os.path.join(base_metrics_dir, f"matrix_coverage_distance_k{k}.npy")
        if not os.path.exists(matrix_path):
            print(f"Warning: {matrix_path} not found, skipping K={k}")
            continue
            
        dist_matrix = np.load(matrix_path)

        print(f"\n--- Analyzing K={k} ---")
        report_lines.append(f"## Hierarchical Tree for K={k}\n")
        
        # Make symmetric for linkage
        sym_dist = (dist_matrix + dist_matrix.T) / 2.0
        # Ensure exact 0s on diagonal
        np.fill_diagonal(sym_dist, 0)
        
        # Condensed distance matrix
        condensed_dist = squareform(sym_dist)
        
        # Perform hierarchical clustering (average linkage)
        Z = linkage(condensed_dist, method='average')
        
        root_node, node_list = to_tree(Z, rd=True)
        
        def build_dataset_tree(node):
            if node.is_leaf():
                return node.id, []
            
            left = node.get_left()
            right = node.get_right()
            
            left_rep, left_children = build_dataset_tree(left)
            right_rep, right_children = build_dataset_tree(right)
            
            # Coverage(Parent -> Child). dist_matrix[i, j] is 1 - Coverage(j over i)
            cov_left_over_right = 1.0 - dist_matrix[right_rep, left_rep]
            cov_right_over_left = 1.0 - dist_matrix[left_rep, right_rep]
            
            # The node that better covers the other becomes the parent of this combined cluster
            if cov_left_over_right >= cov_right_over_left:
                parent = left_rep
                # The right representative becomes a child of the left representative
                new_children = left_children + [(right_rep, right_children, cov_left_over_right)]
            else:
                parent = right_rep
                new_children = right_children + [(left_rep, left_children, cov_right_over_left)]
                
            return parent, new_children
            
        global_rep, tree_structure = build_dataset_tree(root_node)
        
        report_lines.append("```text")
        
        def print_dataset_tree(rep_id, children, prefix=""):
            # Sort children by coverage (highest first)
            children.sort(key=lambda x: x[2], reverse=True)
            
            for i, (child_rep, child_children, coverage) in enumerate(children):
                is_last = (i == len(children) - 1)
                connector = "└── " if is_last else "├── "
                
                role = "[NODE]" if child_children else "[LEAF]"
                name = names[child_rep]
                
                warning = " ⚠️ WEAK LINK" if coverage < 0.5 else ""
                report_lines.append(f"{prefix}{connector}{role} `{name}` (Covered by parent at {100*coverage:.1f}%){warning}")
                
                extension = "    " if is_last else "│   "
                print_dataset_tree(child_rep, child_children, prefix + extension)
                
        report_lines.append(f"[GLOBAL ROOT] `{names[global_rep]}`")
        print_dataset_tree(global_rep, tree_structure)
        report_lines.append("```\n")
        
    report_path = os.path.join(out_dir, "hierarchical_topology_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Saved report to {report_path}")

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="dinov2_base", help="Model name (rf-detr, dinov2_base, dinov2_large, dinov2_giant)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train, test)")
    args = parser.parse_args()

    base_dir = "/mnt/direct-attached/PHASE2_EVAL_RESULTS"
    MODEL_FOLDERS = {
        "rf-detr":        "custom_cell_line_embeddings_analysis_{split}",
        "dinov2_base":    "custom_dinov2_base_cell_line_embeddings_analysis_{split}",
        "dinov2_large":   "custom_dinov2_large_cell_line_embeddings_analysis_{split}",
        "dinov2_giant":   "custom_dinov2_giant_cell_line_embeddings_analysis_{split}",
    }
    
    if args.model not in MODEL_FOLDERS:
        print(f"Error: Unknown model '{args.model}'. Choose from: {list(MODEL_FOLDERS.keys())}")
        return

    folder = MODEL_FOLDERS[args.model].format(split=args.split)
    levels = ["dataset_level", "cell_line_level"]
    
    print(f"\n=== {args.model} Hierarchical Analysis ({args.split}) ===")
    for level in levels:
        metrics_dir = os.path.join(base_dir, folder, "class_cell-adhered", level)
        if os.path.exists(metrics_dir):
            print(f"Processing {level}...")
            out_dir = os.path.join(f"topology_hierarchical_output_{args.split}", args.model.lower(), level)
            analyze_hierarchical_topology(metrics_dir, k_values=[5, 10, 15, 30], out_dir=out_dir)
        else:
            print(f"Metrics dir not found: {metrics_dir}")

if __name__ == "__main__":
    main()