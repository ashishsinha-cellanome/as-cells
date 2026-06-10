import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap
import os

OUTPUT_DIR = "/mnt/direct-attached/PHASE2_EVAL_RESULTS/custom_cell_line_embeddings_analysis"
pkl_path = os.path.join(OUTPUT_DIR, "extracted_raw_embeddings.pkl")

print(f"Loading cached embeddings from {pkl_path}")
with open(pkl_path, 'rb') as f:
    embs = pickle.load(f)

# we care about class 0 (cell)
cell_embs = embs[0]
names = sorted(list(cell_embs.keys()))

def parse_dataset_name(ds_name):
    lower_name = ds_name.lower()
    suspension_keywords = ["suspension", "jurkat", "k562", "nk92", "pbmc", "mousepbmc", "tall104", "raji", "jerat", "tal104"]
    for kw in suspension_keywords:
        if kw in lower_name:
            return None
    clean_name = lower_name
    prefixes_suffixes_to_remove = ["-adhered", "-uncaged", "-caged", "10x", "1x", "20x"]
    for ps in prefixes_suffixes_to_remove:
        clean_name = clean_name.replace(ps, "")
    clean_name = clean_name.strip("-_")
    return clean_name

# Collect features
all_features = []
ds_labels = []
cl_labels = []

for ds, e in cell_embs.items():
    cl = parse_dataset_name(ds)
    if not cl:
        continue
    n = e.shape[0]
    if n > 5000:
        indices = np.random.choice(n, 5000, replace=False)
        sub_embs = e[indices]
    else:
        sub_embs = e
        
    all_features.append(sub_embs)
    ds_labels.extend([ds] * sub_embs.shape[0])
    cl_labels.extend([cl] * sub_embs.shape[0])

if all_features:
    all_features = np.vstack(all_features)
    print(f"Total features to plot: {all_features.shape[0]}")
    
    inst_dir = os.path.join(OUTPUT_DIR, "class_cell", "test_instances")
    os.makedirs(inst_dir, exist_ok=True)
    
    # Just run t-SNE with perp 30 and UMAP as a quick test
    print("Running t-SNE...")
    tsne_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(all_features)
    
    print("Running UMAP...")
    n_samples = all_features.shape[0]
    n_neighbors = min(15, n_samples - 1)
    umap_2d = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42).fit_transform(all_features)
    
    for labels, label_type in [(ds_labels, 'dataset'), (cl_labels, 'cell_line')]:
        unique_labels = len(np.unique(labels))
        palette = sns.color_palette("husl", unique_labels)
        
        # Plot t-SNE
        plt.figure(figsize=(12, 10), dpi=300)
        sns.scatterplot(x=tsne_2d[:, 0], y=tsne_2d[:, 1], hue=labels, palette=palette, s=10)
        plt.title(f"t-SNE colored by {label_type} (perp=30)", fontsize=20)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(os.path.join(inst_dir, f"tsne_{label_type}_perp30.png"), bbox_inches='tight')
        plt.close()
        
        # Plot UMAP
        plt.figure(figsize=(12, 10), dpi=300)
        sns.scatterplot(x=umap_2d[:, 0], y=umap_2d[:, 1], hue=labels, palette=palette, s=10)
        plt.title(f"UMAP colored by {label_type}", fontsize=20)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(os.path.join(inst_dir, f"umap_{label_type}.png"), bbox_inches='tight')
        plt.close()

    print(f"Plots saved to {inst_dir}")
