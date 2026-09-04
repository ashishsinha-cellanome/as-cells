import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
import argparse

def hierarchy_pos(G, root=None, width=1.5, vert_gap=0.4, vert_loc=0, xcenter=0.5):
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

def _compute_mst_component(G_full, component_nodes):
    subgraph = G_full.subgraph(component_nodes).copy()
    try:
        T = nx.minimum_spanning_arborescence(subgraph)
    except nx.NetworkXException:
        return None
    return T

def _draw_tree(T, pos, ax, node_size=2500, font_size=8, arrow_size=14, edge_width=1.2, label_font=9, threshold=None, drop_weak=False):
    nx.draw_networkx_nodes(T, pos, ax=ax, node_size=node_size, node_color='lightgreen', edgecolors='black')
    node_labels = {n: format_node_label(n) for n in T.nodes()}
    nx.draw_networkx_labels(T, pos, ax=ax, labels=node_labels, font_size=font_size, font_weight='bold')

    edge_labels = {}
    strong_edges = []
    weak_edges = []
    
    for u, v, d in T.edges(data=True):
        if 'weight' in d:
            cov = 1 - d['weight']
            edge_labels[(u, v)] = f"{100*cov:.0f}%"
            if threshold is not None and isinstance(threshold, float) and cov < threshold:
                weak_edges.append((u, v))
            else:
                strong_edges.append((u, v))

    if strong_edges:
        nx.draw_networkx_edges(
            T, pos, ax=ax, edgelist=strong_edges,
            arrows=True, arrowstyle='-|>', arrowsize=arrow_size,
            edge_color='gray', width=edge_width,
            min_source_margin=20, min_target_margin=20,
        )
    if weak_edges and not drop_weak:
        nx.draw_networkx_edges(
            T, pos, ax=ax, edgelist=weak_edges,
            arrows=True, arrowstyle='-|>', arrowsize=arrow_size,
            edge_color='red', width=edge_width * 0.7, style='dashed',
            min_source_margin=20, min_target_margin=20,
        )

    if drop_weak:
        edge_labels = {k: v for k, v in edge_labels.items() if k in strong_edges}
    nx.draw_networkx_edge_labels(T, pos, ax=ax, edge_labels=edge_labels, font_color='red', font_size=label_font, label_pos=0.4)

def _plot_single_tree(T, comp, k, t, idx, out_dir, suffix):
    fig, ax = plt.subplots(1, 1, figsize=(24, 14))

    root = [n for n, d in T.in_degree() if d == 0]
    if not root:
        root = list(T.nodes())[0]
    else:
        root = root[0]

    try:
        pos = hierarchy_pos(T, root=root, width=1.5, vert_gap=0.4)
        if len(pos) != len(T.nodes()):
            raise TypeError("hierarchy_pos did not position all nodes")
    except TypeError:
        pos = nx.spring_layout(T, seed=42)

    drop_weak = (suffix == "hard")
    _draw_tree(T, pos, ax=ax, threshold=t, drop_weak=drop_weak)

    label_str = f"Coverage >= {t}" if isinstance(t, float) else "Fully Connected MST"
    if drop_weak:
        label_str += " (weak links hidden)"
        
    ax.set_title(
        f"K={k}, {label_str} — Component {idx+1} ({len(comp)} datasets)",
        fontsize=14
    )
    ax.axis('off')
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"tree_k{k}_t{t}_c{idx+1}_{suffix}.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {plot_path}")


def _plot_trees(component_trees, isolates, k, t, out_dir, suffix):
    if not component_trees:
        return

    for idx, (comp, T) in enumerate(component_trees):
        _plot_single_tree(T, comp, k, t, idx, out_dir, suffix)

    n_trees = len(component_trees)
    fig_height = max(12, n_trees * 7)
    fig, axes = plt.subplots(n_trees, 1, figsize=(28, fig_height))
    if n_trees == 1:
        axes = [axes]

    for idx, (comp, T) in enumerate(component_trees):
        ax = axes[idx]

        root = [n for n, d in T.in_degree() if d == 0]
        if not root:
            root = list(T.nodes())[0]
        else:
            root = root[0]

        try:
            pos = hierarchy_pos(T, root=root, width=1.5, vert_gap=0.4)
            if len(pos) != len(T.nodes()):
                raise TypeError("hierarchy_pos did not position all nodes")
        except TypeError:
            pos = nx.spring_layout(T, seed=42)

        drop_weak = (suffix == "hard")
        _draw_tree(T, pos, ax=ax, node_size=1800, font_size=6, arrow_size=10, edge_width=0.8, label_font=7, threshold=t, drop_weak=drop_weak)
        ax.set_title(f"Component {idx+1} ({len(comp)} datasets)", fontsize=13)
        ax.axis('off')

    label_str = f"Coverage >= {t}" if isinstance(t, float) else "Fully Connected MST"
    if suffix == "hard":
        label_str += " (weak links hidden)"
        
    fig.suptitle(
        f"K={k}, {label_str} — {n_trees} components\nArrow: Superset \u2192 Subset",
        fontsize=15, y=1.02
    )
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"tree_k{k}_t{t}_combined_{suffix}.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {plot_path}")

def plot_dissolution(evolution_rows, k, out_dir):
    rows = [r for r in evolution_rows if r[0] == k]
    if not rows:
        return

    thresholds = [r[1] for r in rows]
    n_components = [r[2] for r in rows]
    n_isolates = [r[4] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, n_components, 'o-', linewidth=2, label='Components', markersize=8)
    ax.plot(thresholds, n_isolates, 's--', linewidth=2, label='Isolates', markersize=8)
    ax.set_xlabel("Coverage Threshold")
    ax.set_ylabel("Count")
    ax.set_title(f"Component Dissolution (K={k})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(out_dir, f"tree_k{k}_component_dissolution.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {plot_path}")

def _process_threshold(G_full, dist_matrix, names, k, t, out_dir, report_lines):
    n = len(names)
    G_t = nx.DiGraph()
    for name in names:
        G_t.add_node(name)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            coverage = 1.0 - dist_matrix[i, j]
            if coverage >= t:
                G_t.add_edge(names[j], names[i], weight=dist_matrix[i, j])

    undirected = G_t.to_undirected()
    components = list(nx.connected_components(undirected))
    components = sorted(components, key=len, reverse=True)

    isolates = [c for c in components if len(c) == 1]
    non_trivial = [c for c in components if len(c) > 1]

    component_trees_full = []
    for comp in non_trivial:
        T = _compute_mst_component(G_full, comp)
        if T is not None:
            component_trees_full.append((comp, T))

    # The hard variant will use the exact same trees, 
    # but the weak edges will be dropped during drawing via `drop_weak=True`.
    component_trees_hard = component_trees_full

    print(f"    Components: {len(components)}, Non-trivial trees: {len(component_trees_full)}, Isolates: {len(isolates)}")

    report_lines.append(f"## K={k}, Coverage >= {t}")
    report_lines.append(f"**Components:** {len(components)}")
    report_lines.append(f"**Non-trivial trees:** {len(component_trees_full)}")
    report_lines.append(f"**Isolates:** {len(isolates)}\n")
    report_lines.append(f"*Arrow direction: Superset \u2192 Subset. Edge label = Coverage %.*\n")

    for idx, (comp, T) in enumerate(component_trees_full):
        report_lines.append(f"### Component {idx+1} ({len(comp)} datasets)")
        root = [n for n, d in T.in_degree() if d == 0]
        if not root:
            continue
        root = root[0]
        report_lines.append("```text")

        def write_tree(node, prefix=""):
            children = list(T.successors(node))
            children.sort(key=lambda c: T[node][c]['weight'])
            for i, child in enumerate(children):
                is_last = (i == len(children) - 1)
                dist = T[node][child]['weight']
                coverage = 100 * (1 - dist)
                connector = "└── " if is_last else "├── "
                role = "[NODE]" if list(T.successors(child)) else "[LEAF]"
                report_lines.append(f"{prefix}{connector}{role} `{child}` ({coverage:.1f}%)")
                extension = "    " if is_last else "│   "
                write_tree(child, prefix + extension)

        report_lines.append(f"[ROOT] {root}")
        write_tree(root)
        report_lines.append("```\n")

    if isolates:
        report_lines.append(f"**Isolates:** {', '.join(sorted([list(i)[0] for i in isolates]))}\n")
    report_lines.append("---\n")

    _plot_trees(component_trees_hard, isolates, k, t, out_dir, suffix="hard")
    _plot_trees(component_trees_full, isolates, k, t, out_dir, suffix="full_weak_links")

    return len(components), len(non_trivial), len(isolates)


def analyze_trees(base_metrics_dir, k_values, thresholds, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    names_path = os.path.join(base_metrics_dir, "entity_names.txt")
    if not os.path.exists(names_path):
        print(f"Warning: {names_path} not found.")
        return

    with open(names_path, "r") as f:
        names = [line.strip() for line in f if line.strip()]

    n = len(names)
    print(f"Loaded {n} entities from {base_metrics_dir}")

    report_lines = [f"# Tree Topology Analysis Report"]
    report_lines.append(f"Model: {os.path.basename(os.path.dirname(os.path.dirname(base_metrics_dir)))}")
    report_lines.append(f"Coverage thresholds: {thresholds}")
    report_lines.append(f"\n*Arrow direction: Superset \u2192 Subset. Edge label = Coverage %.*\n")

    evolution_rows = []

    for k in k_values:
        matrix_path = os.path.join(base_metrics_dir, f"matrix_coverage_distance_k{k}.npy")
        if not os.path.exists(matrix_path):
            print(f"Warning: {matrix_path} not found, skipping K={k}")
            continue

        dist_matrix = np.load(matrix_path)
        print(f"\n--- Analyzing K={k} ---")
        
        G_full = nx.DiGraph()
        for name in names:
            G_full.add_node(name)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                G_full.add_edge(names[j], names[i], weight=dist_matrix[i, j])

        try:
            T_full = nx.minimum_spanning_arborescence(G_full)
            _plot_trees([(set(names), T_full)], [], k, "fully_connected", out_dir, suffix="mst")
        except nx.NetworkXException as e:
            print(f"  Could not compute full MST for K={k}: {e}")

        for t in thresholds:
            print(f"  Threshold coverage >= {t}")
            nc, nnt, ni = _process_threshold(G_full, dist_matrix, names, k, t, out_dir, report_lines)
            evolution_rows.append((k, t, nc, nnt, ni))

        plot_dissolution(evolution_rows, k, out_dir)

    report_path = os.path.join(out_dir, "tree_topology_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Saved report to {report_path}")

    csv_path = os.path.join(out_dir, "component_evolution.csv")
    with open(csv_path, "w") as f:
        f.write("k,threshold,n_components,n_non_trivial,n_isolates\n")
        for k, t, nc, nnt, ni in evolution_rows:
            f.write(f"{k},{t},{nc},{nnt},{ni}\n")

def analyze_from_csv(csv_path, thresholds, out_dir):
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path, index_col=0)
    names = list(df.index)
    dist_matrix = df.values
    n = len(names)

    report_lines = [f"# Tree Topology Analysis Report (From CSV)"]
    report_lines.append(f"Source: {csv_path}")
    report_lines.append(f"Coverage thresholds: {thresholds}")
    report_lines.append(f"\n*Arrow direction: Superset \u2192 Subset. Edge label = Coverage %.*\n")

    evolution_rows = []
    k = "CSV"
    
    G_full = nx.DiGraph()
    for name in names:
        G_full.add_node(name)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            G_full.add_edge(names[j], names[i], weight=dist_matrix[i, j])

    try:
        T_full = nx.minimum_spanning_arborescence(G_full)
        _plot_trees([(set(names), T_full)], [], k, "fully_connected", out_dir, suffix="mst")
    except nx.NetworkXException as e:
        print(f"  Could not compute full MST for CSV: {e}")

    for t in thresholds:
        print(f"  Threshold coverage >= {t}")
        nc, nnt, ni = _process_threshold(G_full, dist_matrix, names, k, t, out_dir, report_lines)
        evolution_rows.append((k, t, nc, nnt, ni))

    plot_dissolution(evolution_rows, k, out_dir)

    report_path = os.path.join(out_dir, "tree_topology_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Saved report to {report_path}")

    csv_path_out = os.path.join(out_dir, "component_evolution.csv")
    with open(csv_path_out, "w") as f:
        f.write("k,threshold,n_components,n_non_trivial,n_isolates\n")
        for k, t, nc, nnt, ni in evolution_rows:
            f.write(f"{k},{t},{nc},{nnt},{ni}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="dinov2_base")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 15, 30])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--csv", type=str, help="Path to a precomputed CSV file to analyze directly")
    parser.add_argument("--out-dir", type=str, default="tree_topology_output_csv", help="Output directory for CSV mode")
    args = parser.parse_args()

    if args.csv:
        print(f"\n=== CSV Tree Analysis ===")
        print(f"Processing {args.csv}...")
        analyze_from_csv(args.csv, args.thresholds, args.out_dir)
        return

    base_dir = "/mnt/direct-attached/PHASE2_EVAL_RESULTS"
    MODEL_FOLDERS = {
        "rf-detr":        "custom_cell_line_embeddings_analysis_{split}",
        "dinov2_base":    "custom_dinov2_cell_line_embeddings_analysis_{split}",
        "dinov2_large":   "custom_dinov2_large_cell_line_embeddings_analysis_{split}",
        "dinov2_giant":   "custom_dinov2_giant_cell_line_embeddings_analysis_{split}",
    }

    if args.model not in MODEL_FOLDERS:
        print(f"Error: Unknown model '{args.model}'. Choose from: {list(MODEL_FOLDERS.keys())}")
        return

    folder = MODEL_FOLDERS[args.model].format(split=args.split)
    levels = ["dataset_level", "cell_line_level"]

    print(f"\n=== {args.model} Tree Analysis ({args.split}) ===")
    for level in levels:
        metrics_dir = os.path.join(base_dir, folder, "class_cell-adhered", level)
        if os.path.exists(metrics_dir):
            print(f"Processing {level}...")
            out_dir = os.path.join(f"tree_topology_output_{args.split}", args.model.lower(), level)
            analyze_trees(metrics_dir, args.k, args.thresholds, out_dir)
        else:
            print(f"Metrics dir not found: {metrics_dir}")

if __name__ == "__main__":
    main()
