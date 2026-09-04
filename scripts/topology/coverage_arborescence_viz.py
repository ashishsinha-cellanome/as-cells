import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import textwrap

def build_arborescence(C):
    """
    Takes distance matrix C, computes coverage as 1-C. 
    Finds the root by maximizing mean out-coverage (mean of 1-C across row). 
    Builds a Prim-style greedy directed tree (arborescence) ensuring non-increasing coverage with depth. 
    Returns root, parent dict, edge_cov dict, out_cov array.
    """
    coverage = 1.0 - C
    n = coverage.shape[0]
    out_cov = np.mean(coverage, axis=1)
    
    root = int(np.argmax(out_cov))
    
    # Prim-style greedy directed tree construction
    visited = {root}
    unvisited = set(range(n)) - {root}
    
    parent = {} # child -> parent
    edge_cov = {} # child -> coverage
    
    while unvisited:
        best_cov = -np.inf
        best_u = None
        best_v = None
        
        for u in visited:
            for v in unvisited:
                cov = coverage[u, v]
                if cov > best_cov:
                    best_cov = cov
                    best_u = u
                    best_v = v
                    
        if best_v is None:
            # Fallback for disconnected components (unlikely in distance matrix)
            best_v = list(unvisited)[0]
            best_u = root
            best_cov = coverage[best_u, best_v]
            
        visited.add(best_v)
        unvisited.remove(best_v)
        parent[best_v] = best_u
        edge_cov[best_v] = best_cov
        
    return root, parent, edge_cov, out_cov


def assign_levels(root, parent):
    """
    Assigns a BFS level to each node starting with root=0.
    """
    levels = {root: 0}
    
    children = {}
    for c, p in parent.items():
        if p not in children:
            children[p] = []
        children[p].append(c)
        
    queue = [root]
    while queue:
        curr = queue.pop(0)
        curr_level = levels[curr]
        for child in children.get(curr, []):
            levels[child] = curr_level + 1
            queue.append(child)
            
    return levels


def hierarchical_layout(parent, levels):
    """
    Computes DFS post-order x-coordinates and depth-based y-coordinates to prevent sub-tree overlap.
    """
    children = {}
    for c, p in parent.items():
        if p not in children:
            children[p] = []
        children[p].append(c)
        
    nodes = set(levels.keys())
    children_nodes = set(parent.keys())
    roots = list(nodes - children_nodes)
    if not roots:
        return {}
    root = roots[0]

    pos = {}
    x_counter = 0
    
    def dfs(node):
        nonlocal x_counter
        # If leaf
        if node not in children or len(children[node]) == 0:
            pos[node] = (x_counter, -levels[node] * 1.5)
            x_counter += 4.0
            return pos[node][0]
        else:
            child_xs = []
            for child in children.get(node, []):
                child_xs.append(dfs(child))
            
            # center parent over children
            x = sum(child_xs) / len(child_xs)
            pos[node] = (x, -levels[node] * 1.5)
            return x

    dfs(root)
    return pos


def visualize_arborescence(C, node_labels, threshold, figsize=(28, 16), title="Coverage Arborescence", show_edge_labels=True, save_path=None):
    """
    Plots the tree using Matplotlib and NetworkX. Handles node styling, edge coloring 
    (solid green for cov >= threshold, dashed red for < threshold), level rulers on the left, 
    root callout, and a legend. Returns fig, G.
    """
    root, parent, edge_cov, out_cov = build_arborescence(C)
    levels = assign_levels(root, parent)
    pos = hierarchical_layout(parent, levels)
    
    G = nx.DiGraph()
    for i, label in enumerate(node_labels):
        G.add_node(i, label=label, out_cov=out_cov[i])
        
    for child, p in parent.items():
        G.add_edge(p, child, weight=edge_cov[child])
        
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw edges
    edges = G.edges()
    green_edges = [(u, v) for u, v in edges if G[u][v]['weight'] >= threshold]
    red_edges = [(u, v) for u, v in edges if G[u][v]['weight'] < threshold]
    
    nx.draw_networkx_edges(G, pos, edgelist=green_edges, ax=ax, edge_color='green', style='solid', width=2.0, arrows=True, arrowsize=20)
    nx.draw_networkx_edges(G, pos, edgelist=red_edges, ax=ax, edge_color='red', style='dashed', width=2.0, arrows=True, arrowsize=20)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=6500, edgecolors='black', linewidths=1.5)
    
    # Node labels - include out_cov and name (wrapped)
    labels = {i: f"{textwrap.fill(node_labels[i], width=25)}\n(Cov: {out_cov[i]:.2f})" for i in range(len(node_labels))}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=11, font_weight='bold')
    
    # Edge labels
    if show_edge_labels:
        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_color='black', font_size=12, label_pos=0.5, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
        
    # Level rulers
    max_level = max(levels.values()) if levels else 0
    xs = [x for x, y in pos.values()]
    min_x = min(xs) - 2 if xs else 0
    
    for level in range(max_level + 1):
        ax.axhline(-level * 1.5, color='gray', linestyle=':', alpha=0.5)
        ax.text(min_x, -level * 1.5, f'Level {level}', verticalalignment='center', color='gray', fontsize=14, fontweight='bold')
        
    # Root callout
    if root in pos:
        root_x, root_y = pos[root]
        ax.annotate('ROOT\nBroadest Coverage', xy=(root_x, root_y + 0.1), xytext=(root_x, root_y + 0.6),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=12, fontweight='bold', ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", alpha=0.8))
                
    # Legend
    red_line = plt.Line2D([0], [0], color='red', linestyle='dashed', lw=2, label=f'Coverage < {threshold}')
    green_line = plt.Line2D([0], [0], color='green', linestyle='solid', lw=2, label=f'Coverage >= {threshold}')
    ax.legend(handles=[green_line, red_line], loc='upper right')
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    ax.set_ylim(-max_level * 1.5 - 1.0, 1.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
        
    return fig, G


def coverage_report(C, node_labels, threshold):
    """
    Prints a text summary of the generated arborescence including root and ranked node table + edge list.
    """
    root, parent, edge_cov, out_cov = build_arborescence(C)
    levels = assign_levels(root, parent)
    
    print("=" * 60)
    print("COVERAGE ARBORESCENCE REPORT")
    print("=" * 60)
    print(f"\nRoot Node (Broadest Dataset): {node_labels[root]} (Mean Out-Coverage: {out_cov[root]:.3f})")
    
    print("\nRanked Nodes (by Mean Out-Coverage):")
    print(f"{'Rank':<6}{'Node':<30}{'Mean Out-Coverage':<20}")
    print("-" * 56)
    
    ranked_nodes = sorted(range(len(out_cov)), key=lambda x: out_cov[x], reverse=True)
    for rank, i in enumerate(ranked_nodes):
        print(f"{rank+1:<6}{node_labels[i]:<30}{out_cov[i]:.3f}")
        
    print("\nTree Edges:")
    print(f"{'Source':<20} -> {'Target':<20} | {'Coverage':<10} | {'Status':<10}")
    print("-" * 65)
    
    # Sort edges by level of target, then by coverage
    edges_list = []
    for target, src in parent.items():
        cov = edge_cov[target]
        lvl = levels[target]
        edges_list.append((src, target, cov, lvl))
        
    edges_list.sort(key=lambda x: (x[3], -x[2]))
    
    for src, target, cov, lvl in edges_list:
        status = "STRONG" if cov >= threshold else "WEAK"
        print(f"{node_labels[src]:<20} -> {node_labels[target]:<20} | {cov:.3f}{' ':>6}| {status}")
        
    print("=" * 60)
