import numpy as np
import os
import pickle
import torch
import networkx as nx
import matplotlib.pyplot as plt

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

def analyze_mst_topology(raw_embs_pkl_path, k_values, out_dir="topology_mst_output"):
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading {raw_embs_pkl_path}...")
    with open(raw_embs_pkl_path, 'rb') as f:
        all_raw_embs = pickle.load(f)
        
    class_id = 2 # cell-adhered
    raw_embs = all_raw_embs.get(class_id, {})
    if not raw_embs:
        print("No embeddings found for class 2.")
        return
        
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
    print(f"Loaded {n} valid cell-adhered datasets.")
    
    report_lines = [f"# Optimal Minimum Spanning Tree Topology Report"]
    report_lines.append("This report constructs a fully connected Directed Tree (Minimum Spanning Arborescence) for all datasets.")
    report_lines.append("Every node is assigned to its *optimal* parent (the dataset that covers it best), ensuring zero isolates.")
    report_lines.append("The global root of the tree is the dataset that naturally provides the broadest overarching coverage for the entire domain.\n")
    
    for k in k_values:
        print(f"\n--- Analyzing K={k} ---")
        report_lines.append(f"## Hierarchical Tree for K={k}\n")
        
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j: continue
                dist_matrix[i, j] = compute_coverage_distance(dataset_embs[names[i]], dataset_embs[names[j]], k=k)
                
        G = nx.DiGraph()
        for name in names:
            G.add_node(name)
            
        for i in range(n): # Target (Subset)
            for j in range(n): # Source (Superset)
                if i == j: continue
                # Edge J -> I means J covers I. Weight is distance (1 - coverage).
                G.add_edge(names[j], names[i], weight=dist_matrix[i, j])
                
        # Calculate Minimum Spanning Arborescence
        T = nx.minimum_spanning_arborescence(G)
        
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

def main():
    print("=== RF-DETR Analysis (MST) ===")
    rf_pkl = "/Users/ashish.sinha/Documents/project/cellanome/as-cells/custom_cell_line_embeddings_analysis/extracted_raw_embeddings.pkl"
    if os.path.exists(rf_pkl):
        analyze_mst_topology(rf_pkl, k_values=[5, 10, 15, 30], out_dir="topology_mst_output/rfdetr")
    
    print("\n=== DINOv2 Analysis (MST) ===")
    dino_pkl = "/Users/ashish.sinha/Documents/project/cellanome/as-cells/custom_dinov2_cell_line_embeddings_analysis/extracted_raw_embeddings.pkl"
    if os.path.exists(dino_pkl):
        analyze_mst_topology(dino_pkl, k_values=[5, 10, 15, 30], out_dir="topology_mst_output/dinov2")

if __name__ == "__main__":
    main()
