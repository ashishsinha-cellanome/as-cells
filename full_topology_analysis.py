import numpy as np
import os
import pickle
import torch
import networkx as nx
from collections import defaultdict
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

def analyze_topology(raw_embs_pkl_path, k_values, threshold=0.20, out_dir="topology_output"):
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
    
    report_lines = [f"# Topology Analysis Report"]
    report_lines.append(f"Threshold for Coverage: Distance < {threshold} (> {100*(1-threshold):.0f}% Coverage)\n")
    
    for k in k_values:
        print(f"\n--- Analyzing K={k} ---")
        report_lines.append(f"## Analysis for K={k}")
        
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j: continue
                dist_matrix[i, j] = compute_coverage_distance(dataset_embs[names[i]], dataset_embs[names[j]], k=k)
                
        G = nx.DiGraph()
        for name in names:
            G.add_node(name)
            
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if dist_matrix[i, j] < threshold:
                    # Edge X -> Y means Y covers X (Y is superset of X)
                    G.add_edge(names[i], names[j], weight=dist_matrix[i, j])
                    
        # Classify nodes based on the graph topology
        # Roots (Parents): Out-degree > 0, In-degree == 0 (They cover others, but nobody covers them)
        # However, because our edge direction is X -> Y (meaning Y covers X), 
        # Y is the superset. 
        # So a "Root" (the ultimate superset) has IN-DEGREE > 0 (many things point to it), OUT-DEGREE == 0 (it points to nothing else as a superset)
        
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
                # children are nodes that point to this node (Subset -> Superset)
                children = list(G.predecessors(node))
                # Sort children by distance (highest coverage first)
                children.sort(key=lambda c: dist_matrix[names.index(c), names.index(node)])
                
                for i, child in enumerate(children):
                    is_last = (i == len(children) - 1)
                    dist = dist_matrix[names.index(child), names.index(node)]
                    coverage = 100 * (1 - dist)
                    
                    connector = "└── " if is_last else "├── "
                    role = "[LEAF]" if child in leaves else "[NODE]"
                    
                    # Markdown block code preserves spacing perfectly
                    report_lines.append(f"{prefix}{connector}{role} `{child}` (Coverage: {coverage:.1f}%)")
                    
                    if child not in visited:
                        extension = "    " if is_last else "│   "
                        print_tree(child, prefix + extension, visited)
                        
            # Wrap the tree in a code block so markdown renders the spaces/lines correctly
            report_lines.append("```text")
            report_lines.append(f"[ROOT] {r}")
            print_tree(r)
            report_lines.append("```")

        report_lines.append("\n---\n")
        
        # Draw the graph
        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue')
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        # Edge arrows (X -> Y)
        edges = G.edges()
        nx.draw_networkx_edges(G, pos, edgelist=edges, arrowstyle='->', arrowsize=20, edge_color='gray')
        
        plt.title(f"Dataset Topology K={k} (Edge X->Y means Y covers X)")
        plt.axis('off')
        
        plot_path = os.path.join(out_dir, f"topology_graph_k{k}.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"Saved graph to {plot_path}")
        
    report_path = os.path.join(out_dir, "topology_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Saved report to {report_path}")

def main():
    # Run for RF-DETR
    print("=== RF-DETR Analysis ===")
    rf_pkl = "/Users/ashish.sinha/Documents/project/cellanome/as-cells/custom_cell_line_embeddings_analysis/extracted_raw_embeddings.pkl"
    if os.path.exists(rf_pkl):
        analyze_topology(rf_pkl, k_values=[5, 10, 15, 30], out_dir="topology_output/rfdetr")
    
    # Run for DINOv2
    print("\n=== DINOv2 Analysis ===")
    dino_pkl = "/Users/ashish.sinha/Documents/project/cellanome/as-cells/custom_dinov2_cell_line_embeddings_analysis/extracted_raw_embeddings.pkl"
    if os.path.exists(dino_pkl):
        analyze_topology(dino_pkl, k_values=[5, 10, 15, 30], out_dir="topology_output/dinov2")
    else:
        print(f"Warning: DINOv2 embeddings not found at {dino_pkl}")

if __name__ == "__main__":
    main()
