import os
import cv2
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from pathlib import Path
from tqdm import tqdm
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from adjustText import adjust_text

DATA_DIR = "/mnt/direct-attached/PHASE2"
OUTPUT_DIR = "/mnt/direct-attached/PHASE2_EVAL_RESULTS/selection_engine"

CLASS_MAP = {
    0: "cell",
    1: "bead",
    2: "soma",
    3: "cell-adhered"
}

def build_dinov2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, device

def extract_largest_object_and_features(img_path, mask_pkl_path, dataset_name, dinov2_model, device):
    img = cv2.imread(str(img_path))
    if img is None:
        return {}
    
    with open(mask_pkl_path, 'rb') as f:
        mask_data = pickle.load(f)
        
    if not mask_data.get('annotations'):
        return {}
        
    results = {}
    
    # Group annotations by category
    anns_by_cat = {}
    for ann in mask_data['annotations']:
        cat = int(ann['category_id'])
        if cat not in anns_by_cat:
            anns_by_cat[cat] = []
        anns_by_cat[cat].append(ann)
        
    for cat, anns in anns_by_cat.items():
        if cat not in CLASS_MAP:
            continue
            
        class_name = CLASS_MAP[cat]
        
        # Find largest annotation by bbox area for this class
        largest_ann = max(anns, key=lambda a: (a['bbox'][2]-a['bbox'][0])*(a['bbox'][3]-a['bbox'][1]))
        
        x1, y1, x2, y2 = [int(v) for v in largest_ann['bbox']]
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
            
        # Padding to square
        size = max(w, h)
        
        cx, cy = x1 + w//2, y1 + h//2
        sq_x1 = max(0, cx - size//2)
        sq_y1 = max(0, cy - size//2)
        sq_x2 = sq_x1 + size
        sq_y2 = sq_y1 + size
        
        # Ensure crop boundaries do not go out of bounds of the actual image
        sq_x1_img = max(0, sq_x1)
        sq_y1_img = max(0, sq_y1)
        sq_x2_img = min(img.shape[1], sq_x2)
        sq_y2_img = min(img.shape[0], sq_y2)
        
        actual_crop = img[sq_y1_img:sq_y2_img, sq_x1_img:sq_x2_img]
        if actual_crop.size == 0:
            continue
            
        # Place it inside the squared padding
        crop = np.zeros((size, size, 3), dtype=np.uint8)
        
        # The actual crop size could be smaller than 'size' if it hit the image boundary
        offset_x = sq_x1_img - sq_x1
        offset_y = sq_y1_img - sq_y1
        
        crop[offset_y:offset_y+actual_crop.shape[0], offset_x:offset_x+actual_crop.shape[1]] = actual_crop
        
        resized_crop = cv2.resize(crop, (224, 224))
        
        # Pass through DINOv2
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(resized_crop).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = dinov2_model.forward_features(input_tensor)
            patch_tokens = features['x_norm_patchtokens'] # (1, 256, 768)
            cls_token = features['x_norm_clstoken'] # (1, 768)
            
        # PCA overlay
        patches = patch_tokens[0].cpu().numpy()
        pca = PCA(n_components=1)
        pca_features = pca.fit_transform(patches) # (256, 1)
        
        # Reshape to 16x16
        grid = pca_features.reshape(16, 16)
        
        # Normalize grid to (-1, 1) mathematically for the colorbar representation
        grid_min, grid_max = grid.min(), grid.max()
        if grid_max - grid_min < 1e-8:
            grid_norm_01 = np.zeros_like(grid)
        else:
            grid_norm_01 = (grid - grid_min) / (grid_max - grid_min)
            
        grid_255 = (grid_norm_01 * 255).astype(np.uint8)
        
        # Colormap and resize
        heatmap = cv2.applyColorMap(grid_255, cv2.COLORMAP_VIRIDIS)
        heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
        
        # Overlay
        overlay = cv2.addWeighted(resized_crop, 0.5, heatmap, 0.5, 0)
        
        # Save with colorbar using matplotlib
        plt.figure(figsize=(5, 4), dpi=300)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        plt.imshow(overlay_rgb)
        plt.axis('off')
        
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0.0, vmax=1.0))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)
        
        out_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_{img_path.stem}_{class_name}_emb.png")
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        results[cat] = cls_token[0].cpu().numpy()
        
    return results

def process_all_datasets(model, device):
    # Group embeddings by class id
    all_embeddings = {0: {}, 1: {}, 2: {}, 3: {}}
    base_dir = Path(DATA_DIR)
    
    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist.")
        return

    for ds in tqdm(list(base_dir.iterdir())):
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
        
        class_embs = extract_largest_object_and_features(first_img, mask_path, ds.name, model, device)
        for cat, emb in class_embs.items():
            all_embeddings[cat][ds.name] = emb
            
    # Process each class
    for cat, embs in all_embeddings.items():
        if not embs:
            continue
            
        class_name = CLASS_MAP[cat]
        names = list(embs.keys())
        matrix = np.array([embs[n] for n in names])
        
        # Distance heatmaps
        for dist_metric in ['cosine', 'euclidean']:
            dist_matrix = pairwise_distances(matrix, metric=dist_metric)
            
            # Adjust figure size based on number of datasets
            fig_w = max(12, min(28, len(names) * 1.0))
            fig_h = max(10, min(24, len(names) * 0.8))
            
            plt.figure(figsize=(fig_w, fig_h), dpi=300)
            sns.heatmap(dist_matrix, xticklabels=names, yticklabels=names, cmap='viridis')
            plt.title(f"Pairwise {dist_metric.capitalize()} Distance of DINOv2 Signatures (Class: {class_name})", fontsize=28, pad=20)
            plt.xticks(rotation=90, fontsize=16)
            plt.yticks(rotation=0, fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"pairwise_distance_heatmap_{dist_metric}_class_{class_name}.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # KMeans Clustering
        n_clusters = min(4, len(names))
        if n_clusters < 2:
            continue
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(matrix)
        
        print(f"\nClustering Results for Class: {class_name}")
        for i in range(n_clusters):
            cluster_names = [names[j] for j in range(len(names)) if kmeans.labels_[j] == i]
            print(f"Cluster {i}: {cluster_names}")
            
        # Cluster Visualization using t-SNE
        n_samples = matrix.shape[0]
        perplexities = [5, 15, 30]
        for perplexity in perplexities:
            if n_samples - 1 < perplexity:
                continue
            
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            tsne_2d = tsne.fit_transform(matrix)
            
            plt.figure(figsize=(16, 16), dpi=300)
            sns.scatterplot(x=tsne_2d[:, 0], y=tsne_2d[:, 1], hue=kmeans.labels_, palette="tab10", s=300)
            
            # Annotate points with dataset names using adjust_text
            texts = []
            for i, name in enumerate(names):
                texts.append(plt.text(tsne_2d[i, 0], tsne_2d[i, 1], name, fontsize=14, alpha=0.9))
                
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
                
            plt.title(f"t-SNE (perp={perplexity}) of DINOv2 Signatures Colored by Cluster (Class: {class_name})", fontsize=26, pad=20)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16, borderaxespad=0.)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"cluster_visualization_tsne_p{perplexity}_class_{class_name}.png"), dpi=300, bbox_inches='tight')
            plt.close()

        # Cluster Visualization using UMAP
        if n_samples > 2:
            n_neighbors = min(15, n_samples - 1)
            umap_reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
            umap_2d = umap_reducer.fit_transform(matrix)
            
            plt.figure(figsize=(16, 16), dpi=300)
            sns.scatterplot(x=umap_2d[:, 0], y=umap_2d[:, 1], hue=kmeans.labels_, palette="tab10", s=300)
            
            texts = []
            for i, name in enumerate(names):
                texts.append(plt.text(umap_2d[i, 0], umap_2d[i, 1], name, fontsize=14, alpha=0.9))
                
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
                
            plt.title(f"UMAP of DINOv2 Signatures Colored by Cluster (Class: {class_name})", fontsize=26, pad=20)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16, borderaxespad=0.)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"cluster_visualization_umap_class_{class_name}.png"), dpi=300, bbox_inches='tight')
            plt.close()

    print("\nProcessing All Classes Combined...")
    all_dataset_names = set()
    for cat in all_embeddings:
        all_dataset_names.update(all_embeddings[cat].keys())
    
    all_dataset_names = list(all_dataset_names)
    combined_matrix = []
    for name in all_dataset_names:
        combined_emb = np.concatenate([all_embeddings[cat].get(name, np.zeros(768)) for cat in sorted(CLASS_MAP.keys())])
        combined_matrix.append(combined_emb)
    combined_matrix = np.array(combined_matrix)

    if len(all_dataset_names) > 0:
        for dist_metric in ['cosine', 'euclidean']:
            dist_matrix = pairwise_distances(combined_matrix, metric=dist_metric)
            
            fig_w = max(12, min(28, len(all_dataset_names) * 1.0))
            fig_h = max(10, min(24, len(all_dataset_names) * 0.8))
            
            plt.figure(figsize=(fig_w, fig_h), dpi=300)
            sns.heatmap(dist_matrix, xticklabels=all_dataset_names, yticklabels=all_dataset_names, cmap='viridis')
            plt.title(f"Pairwise {dist_metric.capitalize()} Distance of DINOv2 Signatures (All Classes)", fontsize=28, pad=20)
            plt.xticks(rotation=90, fontsize=16)
            plt.yticks(rotation=0, fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"pairwise_distance_heatmap_{dist_metric}_all_classes.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        n_clusters = min(4, len(all_dataset_names))
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(combined_matrix)
            
            print("\nClustering Results for All Classes Combined:")
            for i in range(n_clusters):
                cluster_names = [all_dataset_names[j] for j in range(len(all_dataset_names)) if kmeans.labels_[j] == i]
                print(f"Cluster {i}: {cluster_names}")
                
            n_samples = combined_matrix.shape[0]
            perplexities = [5, 15, 30]
            for perplexity in perplexities:
                if n_samples - 1 < perplexity:
                    continue
                
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                tsne_2d = tsne.fit_transform(combined_matrix)
                
                plt.figure(figsize=(16, 16), dpi=300)
                sns.scatterplot(x=tsne_2d[:, 0], y=tsne_2d[:, 1], hue=kmeans.labels_, palette="tab10", s=300)
                
                texts = []
                for i, name in enumerate(all_dataset_names):
                    texts.append(plt.text(tsne_2d[i, 0], tsne_2d[i, 1], name, fontsize=14, alpha=0.9))
                    
                adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
                    
                plt.title(f"t-SNE (perp={perplexity}) of DINOv2 Signatures Colored by Cluster (All Classes)", fontsize=26, pad=20)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16, borderaxespad=0.)
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, f"cluster_visualization_tsne_p{perplexity}_all_classes.png"), dpi=300, bbox_inches='tight')
                plt.close()

            # Cluster Visualization using UMAP
            if n_samples > 2:
                n_neighbors = min(15, n_samples - 1)
                umap_reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
                umap_2d = umap_reducer.fit_transform(combined_matrix)
                
                plt.figure(figsize=(16, 16), dpi=300)
                sns.scatterplot(x=umap_2d[:, 0], y=umap_2d[:, 1], hue=kmeans.labels_, palette="tab10", s=300)
                
                texts = []
                for i, name in enumerate(all_dataset_names):
                    texts.append(plt.text(umap_2d[i, 0], umap_2d[i, 1], name, fontsize=14, alpha=0.9))
                    
                adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
                    
                plt.title("UMAP of DINOv2 Signatures Colored by Cluster (All Classes)", fontsize=26, pad=20)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16, borderaxespad=0.)
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, "cluster_visualization_umap_all_classes.png"), dpi=300, bbox_inches='tight')
                plt.close()

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading model...")
    model, device = build_dinov2()
    print("Model loaded. Processing all datasets...")
    process_all_datasets(model, device)
    print(f"Results saved to {OUTPUT_DIR}")