import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap
import gc

def generate_scatter_plots(features, labels, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    n_samples = features.shape[0]
    if n_samples < 2: return
    
    unique_labels = list(np.unique(labels))
    num_labels = len(unique_labels)
    
    base_colors = sns.color_palette("colorblind")
    palette = [base_colors[i % len(base_colors)] for i in range(num_labels)]
    
    all_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', 'X', 'd']
    markers_dict = {label: all_markers[i % len(all_markers)] for i, label in enumerate(unique_labels)}
    
    # t-SNE
    for p in [30, 50]:
        if n_samples - 1 < p: continue
        print(f"Running t-SNE for {prefix} (perp={p})...")
        tsne_2d = TSNE(n_components=2, perplexity=p, random_state=42).fit_transform(features)
        plt.figure(figsize=(14, 10), dpi=300)
        sns.scatterplot(x=tsne_2d[:, 0], y=tsne_2d[:, 1], hue=labels, style=labels, palette=palette, markers=markers_dict, s=10, alpha=0.8, edgecolor="none")
        plt.title(f"t-SNE (perp={p})", fontsize=20)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"scatter_tsne_perp{p}.png"), bbox_inches='tight')
        plt.close()
    
    # UMAP
    for n_neighbors in [5, 15, 30, 50]:
        if n_samples <= n_neighbors: continue
        print(f"Running UMAP for {prefix} (n_neighbors={n_neighbors})...")
        umap_2d = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42).fit_transform(features)
        plt.figure(figsize=(14, 10), dpi=300)
        sns.scatterplot(x=umap_2d[:, 0], y=umap_2d[:, 1], hue=labels, style=labels, palette=palette, markers=markers_dict, s=10, alpha=0.8, edgecolor="none")
        plt.title(f"UMAP (n_neighbors={n_neighbors})", fontsize=20)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"scatter_umap_nn{n_neighbors}.png"), bbox_inches='tight')
        plt.close()

def parse_dataset_name(ds_name):
    lower_name = ds_name.lower()
    suspension_keywords = ["suspension", "jurkat", "k562", "nk92", "pbmc", "mousepbmc", "tall104", "raji", "jerat", "tal104"]
    for kw in suspension_keywords:
        if kw in lower_name: return None
    clean_name = lower_name
    for ps in ["-adhered", "-uncaged", "-caged", "10x", "1x", "20x"]:
        clean_name = clean_name.replace(ps, "")
    return clean_name.strip("-_")

def process_level(level_name, embs_dict, out_dir, model_name, class_name):
    names = sorted(list(embs_dict.keys()))
    if len(names) < 2: return

    all_features = []
    all_labels = []
    
    for name in names:
        stacked = embs_dict[name]
        if stacked.shape[0] > 5000:
            sub_idx = np.random.choice(stacked.shape[0], 5000, replace=False)
            stacked = stacked[sub_idx]
            
        all_features.append(stacked)
        all_labels.extend([name] * stacked.shape[0])
        
    scatter_dir = os.path.join(out_dir, "scatter_plots")
    all_features = np.vstack(all_features)
    generate_scatter_plots(all_features, all_labels, scatter_dir, f"{model_name}_{class_name}_{level_name}")

def main():
    base_dir = "/mnt/direct-attached/PHASE2_EVAL_RESULTS"
    models = {
        "RF-DETR": "custom_cell_line_embeddings_analysis_train",
        "DINOv2": "custom_dinov2_cell_line_embeddings_analysis_train"
    }
    CLASS_MAP = {0: "cell", 2: "cell-adhered", 3: "soma"}
    
    for model_name, folder in models.items():
        raw_pkl = os.path.join(base_dir, folder, "extracted_raw_embeddings.pkl")
        if not os.path.exists(raw_pkl):
            print(f"Warning: Missing files for {model_name}, skipping.")
            continue
            
        print(f"Loading raw embeddings for {model_name}...")
        with open(raw_pkl, 'rb') as f:
            all_raw_embs = pickle.load(f)
            
        for class_id, class_name in CLASS_MAP.items():
            if class_id not in all_raw_embs: continue
            raw_embs = all_raw_embs[class_id]
            if not raw_embs: continue
                
            print(f"Processing {model_name} | {class_name}...")
            
            # Dataset Level Grouping
            dataset_embs = {}
            for ds, e in raw_embs.items():
                cl = parse_dataset_name(ds)
                if not cl: continue 
                dataset_embs[ds] = e
                
            # Cell Line Level Grouping
            cell_line_embs = {}
            for ds, e in raw_embs.items():
                cl = parse_dataset_name(ds)
                if not cl: continue
                if cl not in cell_line_embs:
                    cell_line_embs[cl] = []
                cell_line_embs[cl].append(e)
                
            for cl in cell_line_embs:
                cell_line_embs[cl] = np.vstack(cell_line_embs[cl])
                
            out_dir_ds = os.path.join(base_dir, folder, f"class_{class_name}", "dataset_level")
            process_level("dataset_level", dataset_embs, out_dir_ds, model_name, class_name)
            
            out_dir_cl = os.path.join(base_dir, folder, f"class_{class_name}", "cell_line_level")
            process_level("cell_line_level", cell_line_embs, out_dir_cl, model_name, class_name)
            
            del dataset_embs
            del cell_line_embs
            gc.collect()
            
        del all_raw_embs
        gc.collect()

if __name__ == "__main__":
    main()
