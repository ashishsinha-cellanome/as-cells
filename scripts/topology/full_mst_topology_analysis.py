import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt

def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')
    if root is None:
        root = next(iter(nx.topological_sort(G)))
        
    def _hierarchy_pos(G, root, left, right, vert_gap, vert_loc, pos=None):
        if pos is None:
            pos = {root: ((left + right) / 2, vert_loc)}
        else:
            pos[root] = ((left + right) / 2, vert_loc)
            
        children = list(G.successors(root))
        if len(children) != 0:
            def get_leaves(node):
                c = list(G.successors(node))
                if not c:
                    return 1
                return sum(get_leaves(child) for child in c)
                
            leaves_counts = [get_leaves(child) for child in children]
            total_leaves = sum(leaves_counts)
            
            dx = (right - left) / total_leaves
            current_left = left
            for child, count in zip(children, leaves_counts):
                current_right = current_left + dx * count
                pos = _hierarchy_pos(G, child, current_left, current_right,
                                     vert_gap, vert_loc - vert_gap, pos=pos)
                current_left = current_right
        return pos
    return _hierarchy_pos(G, root, xcenter - width/2, xcenter + width/2, vert_gap, vert_loc)

def format_node_label(name):
    clean = name.replace("_10x", "").replace("_4_class", "").replace("-adhered", "").replace("-adherent", "")
    clean = clean.replace("_caged", "\nCAGED").replace("_uncaged", "\nUNCAGED")
    return clean

def analyze_mst_topology(base_metrics_dir, k_values, out_dir="topology_mst_output"):
    os.makedirs(out_dir, exist_ok=True)
    
    names_path = os.path.join(base_metrics_dir, "entity_names.txt")
    if not os.path.exists(names_path):
        print(f"Warning: {names_path} not found.")
        return
        
    with open(names_path, "r") as f:
        names = [line.strip() for line in f if line.strip()]
        
    n = len(names)
    print(f"Loaded {n} entities from {base_metrics_dir}")
    
    report_lines = [f"# Optimal Minimum Spanning Tree Topology Report"]
    report_lines.append("This report constructs a fully connected Directed Tree (Minimum Spanning Arborescence) for all datasets.")
    report_lines.append("Every node is assigned to its *optimal* parent (the dataset that covers it best), ensuring zero isolates.")
    report_lines.append("The global root of the tree is the dataset that naturally provides the broadest overarching coverage for the entire domain.\n")
    
    for k in k_values:
        matrix_path = os.path.join(base_metrics_dir, f"matrix_coverage_distance_k{k}.npy")
        if not os.path.exists(matrix_path):
            print(f"Warning: {matrix_path} not found, skipping K={k}")
            continue
            
        dist_matrix = np.load(matrix_path)
        
        print(f"\n--- Analyzing K={k} ---")
        report_lines.append(f"## Hierarchical Tree for K={k}\n")
        
        G = nx.DiGraph()
        for name in names:
            G.add_node(name)
            
        for i in range(n): # Target (Subset)
            for j in range(n): # Source (Superset)
                if i == j: continue
                # Edge J -> I means J covers I. Weight is distance (1 - coverage).
                G.add_edge(names[j], names[i], weight=dist_matrix[i, j])
                
        # Calculate Minimum Spanning Arborescence
        try:
            T = nx.minimum_spanning_arborescence(G)
        except nx.NetworkXException as e:
            print(f"Could not compute minimum spanning arborescence for K={k}: {e}")
            continue
        
        # Find the root (node with in_degree == 0)
        root = [n for n, d in T.in_degree() if d == 0][0]
        
        def print_mst_tree(node, prefix=""):
            children = list(T.successors(node))
            # Sort children by coverage (descending), meaning lowest weight first
            children.sort(key=lambda c: T[node][c]['weight'])
            
            for i, child in enumerate(children):
                is_last = (i == len(children) - 1)
                dist = T[node][child]['weight']
                coverage = 100 * (1 - dist)
                
                # Tag links that are statistically weak
                warning = ""
                if coverage < 50:
                    warning = " ⚠️ WEAK LINK"
                    
                connector = "└── " if is_last else "├── "
                role = "[NODE]" if list(T.successors(child)) else "[LEAF]"
                
                report_lines.append(f"{prefix}{connector}{role} `{child}` (Coverage: {coverage:.1f}%){warning}")
                
                extension = "    " if is_last else "│   "
                print_mst_tree(child, prefix + extension)
                
        report_lines.append("```text")
        report_lines.append(f"[ROOT] {root}")
        print_mst_tree(root)
        report_lines.append("```\n")
        
        # Draw the graph using custom hierarchical layout
        plt.figure(figsize=(24, 16))
        pos = hierarchy_pos(T, root=root)
            
        nx.draw_networkx_nodes(T, pos, node_size=3500, node_color='lightgreen', edgecolors='black')
        
        # Format node labels for visual clarity (prevents overlap)
        node_labels = {n: format_node_label(n) for n in T.nodes()}
        
        # Format edge labels to show coverage %
        edge_labels = {}
        for u, v, d in T.edges(data=True):
            edge_labels[(u, v)] = f"{100*(1-d['weight']):.0f}%"
            
        nx.draw_networkx_edges(T, pos, edgelist=T.edges(), arrowstyle='->', arrowsize=25, edge_color='gray', width=2)
        nx.draw_networkx_edge_labels(T, pos, edge_labels=edge_labels, font_color='red', font_size=11, label_pos=0.5)
        nx.draw_networkx_labels(T, pos, labels=node_labels, font_size=9, font_weight='bold')
        
        plt.title(f"Fully Connected Dataset Topology K={k}\n(Arrow points from Superset -> Subset. Percentage is coverage provided by the Superset)", fontsize=20, pad=20)
        plt.axis('off')
        
        plot_path = os.path.join(out_dir, f"mst_topology_graph_k{k}.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved graph to {plot_path}")
        
    report_path = os.path.join(out_dir, "mst_topology_report.md")
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
    
    print(f"\n=== {args.model} MST Analysis ({args.split}) ===")
    for level in levels:
        metrics_dir = os.path.join(base_dir, folder, "class_cell-adhered", level)
        if os.path.exists(metrics_dir):
            print(f"Processing {level}...")
            out_dir = os.path.join(f"topology_mst_output_{args.split}", args.model.lower(), level)
            analyze_mst_topology(metrics_dir, k_values=[5, 10, 15, 30], out_dir=out_dir)
        else:
            print(f"Metrics dir not found: {metrics_dir}")

if __name__ == "__main__":
    main()