import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt

def analyze_topology(base_metrics_dir, k_values, threshold=0.20, out_dir="topology_output"):
    os.makedirs(out_dir, exist_ok=True)
    
    names_path = os.path.join(base_metrics_dir, "entity_names.txt")
    if not os.path.exists(names_path):
        print(f"Warning: {names_path} not found.")
        return
        
    with open(names_path, "r") as f:
        names = [line.strip() for line in f if line.strip()]
        
    n = len(names)
    print(f"Loaded {n} entities from {base_metrics_dir}")
    
    report_lines = [f"# Topology Analysis Report"]
    report_lines.append(f"Threshold for Coverage: Distance < {threshold} (> {100*(1-threshold):.0f}% Coverage)\n")
    
    for k in k_values:
        matrix_path = os.path.join(base_metrics_dir, f"matrix_coverage_distance_k{k}.npy")
        if not os.path.exists(matrix_path):
            print(f"Warning: {matrix_path} not found, skipping K={k}")
            continue
            
        dist_matrix = np.load(matrix_path)
        
        print(f"\n--- Analyzing K={k} ---")
        report_lines.append(f"## Analysis for K={k}")
        
        G = nx.DiGraph()
        for name in names:
            G.add_node(name)
            
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if dist_matrix[i, j] < threshold:
                    # Edge X -> Y means Y covers X (Y is superset of X)
                    G.add_edge(names[i], names[j], weight=dist_matrix[i, j])
                    
        roots = []
        leaves = []
        internal = []
        isolates = []
        
        for node in G.nodes():
            in_deg = G.in_degree(node)
            out_deg = G.out_degree(node)
            
            if in_deg == 0 and out_deg == 0:
                isolates.append(node)
            elif out_deg == 0 and in_deg > 0:
                roots.append(node) # Ultimate supersets
            elif in_deg == 0 and out_deg > 0:
                leaves.append(node) # Ultimate subsets
            else:
                internal.append(node)
                
        report_lines.append("### Topological Roles")
        report_lines.append(f"**Roots (Ultimate Supersets - Best for Generalizing Downward):**")
        for r in roots:
            report_lines.append(f"- `{r}` (Covers {G.in_degree(r)} subsets)")
            
        report_lines.append(f"\n**Leaves (Ultimate Subsets - Narrow Domains):**")
        for l in leaves:
            report_lines.append(f"- `{l}` (Covered by {G.out_degree(l)} supersets)")
            
        report_lines.append(f"\n**Internal Nodes (Clusters/Mid-level):**")
        for m in internal:
            report_lines.append(f"- `{m}`")
            
        report_lines.append(f"\n**Isolates (Independent Domains):**")
        for iso in isolates:
            report_lines.append(f"- `{iso}`")
            
        report_lines.append("\n### Hierarchical Tree & Coverage (% Covered by Parent)")
        for r in roots:
            report_lines.append(f"- **[ROOT]** `{r}`")
            
            def print_tree(node, prefix="", visited=None):
                if visited is None:
                    visited = set()
                visited.add(node)
                children = list(G.predecessors(node))
                children.sort(key=lambda c: dist_matrix[names.index(c), names.index(node)])
                
                for i, child in enumerate(children):
                    is_last = (i == len(children) - 1)
                    dist = dist_matrix[names.index(child), names.index(node)]
                    coverage = 100 * (1 - dist)
                    
                    connector = "└── " if is_last else "├── "
                    role = "[LEAF]" if child in leaves else "[NODE]"
                    
                    report_lines.append(f"{prefix}{connector}{role} `{child}` (Coverage: {coverage:.1f}%)")
                    
                    if child not in visited:
                        extension = "    " if is_last else "│   "
                        print_tree(child, prefix + extension, visited)
                        
            report_lines.append("```text")
            report_lines.append(f"[ROOT] {r}")
            print_tree(r)
            report_lines.append("```")

        report_lines.append("\n---\n")
        
        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue')
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        edges = G.edges()
        nx.draw_networkx_edges(G, pos, edgelist=edges, arrowstyle='->', arrowsize=20, edge_color='gray')
        
        plt.title(f"Topology K={k} (Edge X->Y means Y covers X)")
        plt.axis('off')
        
        plot_path = os.path.join(out_dir, f"topology_graph_k{k}.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"Saved graph to {plot_path}")
        
    report_path = os.path.join(out_dir, "topology_report.md")
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
    
    print(f"\n=== {args.model} Analysis ({args.split}) ===")
    for level in levels:
        metrics_dir = os.path.join(base_dir, folder, "class_cell-adhered", level)
        if os.path.exists(metrics_dir):
            print(f"Processing {level}...")
            out_dir = os.path.join(f"topology_output_{args.split}", args.model.lower(), level)
            analyze_topology(metrics_dir, k_values=[5, 10, 15, 30], out_dir=out_dir)
        else:
            print(f"Metrics dir not found: {metrics_dir}")

if __name__ == "__main__":
    main()