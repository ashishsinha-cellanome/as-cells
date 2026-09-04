import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'topology')))
import pickle
import numpy as np
import yaml
import argparse
from coverage_arborescence_viz import build_arborescence
import torch

def compute_coverage_distance(X, Y, k=10):
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
    return 1.0 - torch.mean(covered).item()

def parse_dataset_name(ds_name):
    lower_name = ds_name.lower()
    suspension_keywords = ["suspension", "jurkat", "k562", "nk92", "pbmc", "mousepbmc", "tall104", "raji", "jerat", "tal104"]
    for kw in suspension_keywords:
        if kw in lower_name: return None
    return ds_name

def main():
    parser = argparse.ArgumentParser(description='Generate YAML splits from tree topology')
    parser.add_argument('--input_pkl', type=str, required=True, help='Path to raw embeddings')
    parser.add_argument('--output_dir', type=str, default='configs/data/coverage_splits', help='Directory to save YAMLs')
    parser.add_argument('--readme_path', type=str, default='docs/COVERAGE_SPLITS_README.md', help='Path for tracking README')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.readme_path), exist_ok=True)

    with open(args.input_pkl, 'rb') as f:
        all_raw_embs = pickle.load(f)
    
    raw_embs_class2 = all_raw_embs.get(2, {})
    raw_embs_class3 = all_raw_embs.get(3, {})
    raw_embs = {**raw_embs_class2, **raw_embs_class3}

    np.random.seed(42)
    dataset_embs = {}
    for ds, e in raw_embs.items():
        if parse_dataset_name(ds):
            if e.shape[0] > 3000:
                idx = np.random.choice(e.shape[0], 3000, replace=False)
                dataset_embs[ds] = e[idx]
            else:
                dataset_embs[ds] = e

    names = sorted(list(dataset_embs.keys()))
    n = len(names)

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j: continue
            dist_matrix[i, j] = compute_coverage_distance(dataset_embs[names[i]], dataset_embs[names[j]], k=10)

    # Build Tree
    root, parent, edge_cov, out_cov = build_arborescence(dist_matrix)
    
    children_map = {}
    for c, p in parent.items():
        if p not in children_map:
            children_map[p] = []
        children_map[p].append(c)

    # Generate Motifs
    configs_data = []
    
    def get_short_name(full_name):
        return full_name.split('_')[1] if len(full_name.split('_')) > 1 else full_name[:10]
        
    motif_counter = 1
    
    for p_idx, children_idx in children_map.items():
        p_name = names[p_idx]
        
        # 1. Downward (Train on Parent, Test on all Children)
        test_names = [names[c] for c in children_idx]
        cov_val = np.mean([1.0 - dist_matrix[c, p_idx] for c in children_idx])  # average coverage of parent over children
        
        cfg = {
            'name': f"motif_{motif_counter:02d}_{get_short_name(p_name)}_downward",
            'type': 'Downward',
            'train': [p_name],
            'test': test_names,
            'coverage': cov_val
        }
        configs_data.append(cfg)
        motif_counter += 1
        
        # 2. Upward Single (Train on 1 Child, Test on Parent)
        for c_idx in children_idx:
            c_name = names[c_idx]
            cov_val = 1.0 - dist_matrix[p_idx, c_idx] # coverage of child over parent
            cfg = {
                'name': f"motif_{motif_counter:02d}_{get_short_name(c_name)}_to_{get_short_name(p_name)}_upward",
                'type': 'Upward Single',
                'train': [c_name],
                'test': [p_name],
                'coverage': cov_val
            }
            configs_data.append(cfg)
            motif_counter += 1
            
        # 3. Upward Multiple (Train on all Children, Test on Parent)
        if len(children_idx) > 1:
            # Approximate coverage by taking max coverage provided by any child
            cov_val = max([1.0 - dist_matrix[p_idx, c] for c in children_idx]) 
            cfg = {
                'name': f"motif_{motif_counter:02d}_multichildren_to_{get_short_name(p_name)}_upward",
                'type': 'Upward Multiple',
                'train': test_names,
                'test': [p_name],
                'coverage': cov_val
            }
            configs_data.append(cfg)
            motif_counter += 1
            
        # 4. Sibling Cross-Branch (Train on Child A, Test on Child B)
        if len(children_idx) > 1:
            for i in range(len(children_idx)):
                for j in range(i+1, len(children_idx)):
                    c1_idx = children_idx[i]
                    c2_idx = children_idx[j]
                    
                    cov_val = 1.0 - dist_matrix[c2_idx, c1_idx]
                    cfg = {
                        'name': f"motif_{motif_counter:02d}_{get_short_name(names[c1_idx])}_to_{get_short_name(names[c2_idx])}_sibling",
                        'type': 'Sibling',
                        'train': [names[c1_idx]],
                        'test': [names[c2_idx]],
                        'coverage': cov_val
                    }
                    configs_data.append(cfg)
                    motif_counter += 1
                    
                    cov_val_rev = 1.0 - dist_matrix[c1_idx, c2_idx]
                    cfg = {
                        'name': f"motif_{motif_counter:02d}_{get_short_name(names[c2_idx])}_to_{get_short_name(names[c1_idx])}_sibling",
                        'type': 'Sibling',
                        'train': [names[c2_idx]],
                        'test': [names[c1_idx]],
                        'coverage': cov_val_rev
                    }
                    configs_data.append(cfg)
                    motif_counter += 1

    # Write YAMLs
    readme_lines = [
        "# Coverage Arborescence Data Splits\n",
        "This file tracks the generated Hydra data configs based on local motifs extracted from the Tree Topology.\n",
        "| Config YAML | Motif Type | Train Datasets | Test Datasets | Coverage (Train -> Test) |\n",
        "|-------------|------------|----------------|---------------|--------------------------|\n"
    ]
    
    for cfg in configs_data:
        yaml_path = os.path.join(args.output_dir, f"{cfg['name']}.yaml")
        yaml_content = {
            'defaults': ['default@data'],
            'data': {
                'train_datasets': cfg['train'],
                'test_datasets': cfg['test'],
                'split_motif': cfg['type'],
                'coverage': float(round(cfg['coverage'], 4))
            }
        }
        
        with open(yaml_path, 'w') as f:
            f.write("# @package _global_\n")
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
            
        readme_lines.append(f"| `{cfg['name']}.yaml` | {cfg['type']} | {len(cfg['train'])} dataset(s) | {len(cfg['test'])} dataset(s) | {cfg['coverage']:.3f} |\n")

    with open(args.readme_path, 'w') as f:
        f.writelines(readme_lines)
        
    print(f"Generated {len(configs_data)} split YAMLs in {args.output_dir}")
    print(f"Tracking README updated at {args.readme_path}")

if __name__ == '__main__':
    main()