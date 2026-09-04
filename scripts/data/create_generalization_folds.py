import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import yaml
import numpy as np
from pathlib import Path
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import pairwise_distances

import dataset_selection_engine as dse

OUTPUT_DIR = "/mnt/direct-attached/as-cells/configs/data"

def categorize_datasets(names):
    suspension = []
    adhered = []
    for name in names:
        if any(x in name.lower() for x in ["suspension", "pbmc", "jurkat", "k562", "raji", "tall104"]):
            suspension.append(name)
        else:
            adhered.append(name)
    return suspension, adhered

def get_medoids(names, matrix, k=3):
    dist_matrix = pairwise_distances(matrix, metric='cosine')
    kmedoids = KMedoids(n_clusters=k, metric='precomputed', random_state=42).fit(dist_matrix)
    medoid_indices = kmedoids.medoid_indices_
    return [names[i] for i in medoid_indices]

def write_hydra_config(filename, train_sets, test_sets):
    config = {
        "defaults": [{"full": "_self_"}],
        "name": filename.replace(".yaml", ""),
        "train": {
            "datasets": [f"/mnt/direct-attached/PHASE2/{ds}" for ds in train_sets]
        },
        "val": {
            "datasets": [f"/mnt/direct-attached/PHASE2/{ds}" for ds in test_sets]
        },
        "test": {
            "datasets": [f"/mnt/direct-attached/PHASE2/{ds}" for ds in test_sets]
        }
    }
    path = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Created {filename}")

if __name__ == "__main__":
    print("Extracting DINOv2 embeddings...")
    model, device = dse.build_dinov2()
    
    base_dir = Path(dse.DATA_DIR)
    all_embeddings = {}
    for ds in base_dir.iterdir():
        if not ds.is_dir():
            continue
        img_dir = ds / "images" / "test"
        mask_dir = ds / "masks" / "test"
        if not img_dir.exists() or not mask_dir.exists():
            continue
        
        imgs = list(img_dir.iterdir())
        if not imgs:
            continue
        
        first_img = imgs[0]
        mask_path = mask_dir / f"{first_img.stem}.pkl"
        if not mask_path.exists():
            continue
        
        emb, _ = dse.extract_largest_object_and_features(first_img, mask_path, ds.name, model, device)
        if emb is not None:
            all_embeddings[ds.name] = emb

    names = list(all_embeddings.keys())
    
    susp_names, adh_names = categorize_datasets(names)
    
    # 1. Suspension Medoids
    susp_matrix = np.array([all_embeddings[n] for n in susp_names])
    susp_medoids = get_medoids(susp_names, susp_matrix, k=3)
    unseen_susp = [n for n in susp_names if n not in susp_medoids]
    
    write_hydra_config("exp_susp_to_sus.yaml", susp_medoids, unseen_susp)
    write_hydra_config("exp_susp_to_adh.yaml", susp_medoids, adh_names)
    
    # 2. Global Medoids (exclude moc22)
    candidate_global = [n for n in names if "moc22" not in n.lower()]
    global_matrix = np.array([all_embeddings[n] for n in candidate_global])
    global_medoids = get_medoids(candidate_global, global_matrix, k=4)
    global_test = [n for n in names if n not in global_medoids]
    
    write_hydra_config("exp_global_to_rest.yaml", global_medoids, global_test)
