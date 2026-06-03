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
from pycocotools.coco import COCO
import pycocotools.mask as mask_util
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from adjustText import adjust_text
from scipy.cluster.hierarchy import linkage, dendrogram
from omegaconf import OmegaConf

DATA_DIR = "/mnt/direct-attached/PHASE2"
OUTPUT_DIR = "/mnt/direct-attached/PHASE2_EVAL_RESULTS/selection_engine_rfdetr_robust"

# Load the label map dynamically from the rfdetr_seg model config
_model_cfg = OmegaConf.load("configs/model/rfdetr_seg.yaml")
CLASS_MAP = OmegaConf.to_container(_model_cfg.label_map, resolve=True)

# Add model checkpoint path
CHECKPOINT_PATH = "/mnt/personal/cellanome/checkpoints/ALL_CKPTS/RFDETR-Seg-ckpts/output/checkpoint_best_ema.pth"

def build_rfdetr_backbone(checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import rfdetr
    print(f"Loading RFDETRSegLarge from {checkpoint}")
    if not os.path.exists(checkpoint):
        print(f"ERROR: Checkpoint file does not exist at {checkpoint}")
        print("Falling back to randomly initialized RF-DETR backbone.")
        model = rfdetr.RFDETRSegLarge(pretrain_weights=None, group_detr=1, num_classes=4)
    else:
        try:
            model = rfdetr.RFDETRSegLarge(pretrain_weights=checkpoint, group_detr=1, num_classes=4)
            print("Successfully loaded the fine-tuned RF-DETR-Seg checkpoint.")
        except Exception as e:
            import traceback
            print(f"Failed to load checkpoint from {checkpoint}.")
            print(f"Exception details:\n{traceback.format_exc()}")
            print("Falling back to randomly initialized RF-DETR backbone.")
            model = rfdetr.RFDETRSegLarge(pretrain_weights=None, group_detr=1, num_classes=4)
        
    model.model.model.to(device)
    model.model.model.eval()
    
    encoder = model.model.model.backbone[0].encoder
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder, device

def extract_features_for_image(img_path, mask_pkl_path, dataset_name, encoder, device):
    img = cv2.imread(str(img_path))
    if img is None:
        return {}
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    with open(mask_pkl_path, 'rb') as f:
        mask_data = pickle.load(f)
        
    if not mask_data.get('annotations'):
        return {}
        
    results = {cat: [] for cat in CLASS_MAP.keys()}
    
    # Group annotations by category
    anns_by_cat = {}
    for ann in mask_data['annotations']:
        cat = int(ann['category_id'])
        if cat not in anns_by_cat:
            anns_by_cat[cat] = []
        anns_by_cat[cat].append(ann)
        
    for cat, cat_anns in anns_by_cat.items():
        if cat not in CLASS_MAP:
            continue
            
        class_name = CLASS_MAP[cat]
        
        # Sort by bounding box area descending and take top 5
        sorted_anns = sorted(cat_anns, key=lambda a: (a['bbox'][2]-a['bbox'][0])*(a['bbox'][3]-a['bbox'][1]), reverse=True)
        top_anns = sorted_anns[:5]
        
        for rank, ann in enumerate(top_anns):
            x, y, x2, y2 = [int(v) for v in ann['bbox']]
            w, h = x2 - x, y2 - y
            if w <= 0 or h <= 0:
                continue
                
            # Create a masked version of the image using the segmentation mask
            mask = mask_util.decode(ann['segmentation'])
            mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            
            img_h, img_w = img.shape[:2]
            
            x1_clip, y1_clip = max(0, x), max(0, y)
            x2_clip, y2_clip = min(img_w, x + w), min(img_h, y + h)
            
            mask_y1 = y1_clip - y
            mask_y2 = mask_y1 + (y2_clip - y1_clip)
            mask_x1 = x1_clip - x
            mask_x2 = mask_x1 + (x2_clip - x1_clip)
            
            mask_3d_cropped = mask_3d[mask_y1:mask_y2, mask_x1:mask_x2]
            
            masked_img = np.zeros_like(img)
            masked_img[y1_clip:y2_clip, x1_clip:x2_clip] = img[y1_clip:y2_clip, x1_clip:x2_clip] * mask_3d_cropped
                
            size = max(w, h)
            cx, cy = x + w//2, y + h//2
            sq_x1, sq_y1 = cx - size//2, cy - size//2
            sq_x2, sq_y2 = sq_x1 + size, sq_y1 + size
            
            sq_x1_img, sq_y1_img = max(0, sq_x1), max(0, sq_y1)
            sq_x2_img, sq_y2_img = min(img.shape[1], sq_x2), min(img.shape[0], sq_y2)
            
            actual_crop = masked_img[sq_y1_img:sq_y2_img, sq_x1_img:sq_x2_img]
            if actual_crop.size == 0:
                continue
                
            crop = np.zeros((size, size, 3), dtype=np.uint8)
            offset_x, offset_y = sq_x1_img - sq_x1, sq_y1_img - sq_y1
            crop[offset_y:offset_y+actual_crop.shape[0], offset_x:offset_x+actual_crop.shape[1]] = actual_crop
            
            # Resize for RFDETR backbone (requires input divisible by 24 for standard dinov2 backbone defaults in rfdetr, e.g. 240x240)
            resized_crop = cv2.resize(crop, (240, 240), interpolation=cv2.INTER_AREA)
            
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(resized_crop).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features = encoder(input_tensor)
                last_feat = features[-1]
                cls_token = last_feat.mean(dim=(2, 3)) # (1, C) Global Avg Pooling
                patch_tokens = last_feat.flatten(2).transpose(1, 2) # (1, H*W, C)
                
            # --- Best Practice PCA Overlay ---
            patches = patch_tokens[0].cpu().numpy()
            
            # Downsample the perfect binary mask to the 20x20 patch grid (240 / 12 = 20)
            binary_crop_mask = (resized_crop > 0).any(axis=2).astype(np.uint8)
            patch_mask = cv2.resize(binary_crop_mask, (20, 20), interpolation=cv2.INTER_NEAREST)
            patch_mask_flat = patch_mask.flatten()
            
            foreground_indices = np.where(patch_mask_flat > 0)[0]
            
            if len(foreground_indices) > 0:
                foreground_patches = patches[foreground_indices]
                
                pca = PCA(n_components=1)
                fg_pca_features = pca.fit_transform(foreground_patches)
                
                # Min-Max Scale foreground features [0, 1]
                fg_min, fg_max = fg_pca_features.min(), fg_pca_features.max()
                if fg_max - fg_min < 1e-8:
                    fg_pca_norm = np.zeros_like(fg_pca_features)
                else:
                    fg_pca_norm = (fg_pca_features - fg_min) / (fg_max - fg_min)
                    
                # Reconstruct full 20x20 grid
                grid_norm_01 = np.zeros((400, 1))
                grid_norm_01[foreground_indices] = fg_pca_norm
                
                grid_255 = (grid_norm_01.reshape(20, 20) * 255).astype(np.uint8)
                
                # Apply Viridis
                heatmap = cv2.applyColorMap(grid_255, cv2.COLORMAP_VIRIDIS)
                heatmap = cv2.resize(heatmap, (240, 240), interpolation=cv2.INTER_CUBIC)
                
                # Mask out the background exactly
                heatmap = cv2.bitwise_and(heatmap, heatmap, mask=binary_crop_mask)
                
                # Overlay
                overlay = cv2.addWeighted(resized_crop, 0.5, heatmap, 0.5, 0)
                
                plt.figure(figsize=(5, 4), dpi=300)
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                plt.imshow(overlay_rgb)
                plt.axis('off')
                
                sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0.0, vmax=1.0))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=10)
                
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                out_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_{Path(img_path).stem}_{class_name}_rank{rank+1}_emb.png")
                plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
                plt.close()
                
            results[cat].append(cls_token[0].cpu().numpy())
            
    return results

def process_all_datasets(encoder, device):
    all_embeddings = {cat: {} for cat in CLASS_MAP.keys()}
    base_dir = Path(DATA_DIR)
    
    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist.")
        return all_embeddings

    for ds in tqdm(list(base_dir.iterdir())):
        if not ds.is_dir():
            continue
        img_dir = ds / "images" / "test"
        mask_dir = ds / "masks" / "test"
        
        if not img_dir.exists() or not mask_dir.exists():
            continue
        
        imgs = sorted(list(img_dir.iterdir()))
        valid_imgs = []
        for img in imgs:
            mask_path = mask_dir / f"{img.stem}.pkl"
            if mask_path.exists():
                valid_imgs.append((img, mask_path))
            if len(valid_imgs) == 10:
                break
                
        if not valid_imgs:
            continue
            
        ds_cat_embeddings = {cat: [] for cat in CLASS_MAP.keys()}
        
        for img_path, mask_path in valid_imgs:
            class_embs = extract_features_for_image(img_path, mask_path, ds.name, encoder, device)
            for cat, embs in class_embs.items():
                ds_cat_embeddings[cat].extend(embs)
                
        # Aggregate (Mean) Top 50 features per class for this dataset
        for cat, embs in ds_cat_embeddings.items():
            if embs:
                all_embeddings[cat][ds.name] = np.mean(embs, axis=0)
            
    return all_embeddings

def generate_visualizations(all_embeddings):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each class individually
    for cat, embs in all_embeddings.items():
        if not embs:
            continue
            
        class_name = CLASS_MAP[cat]
        names = list(embs.keys())
        matrix = np.array([embs[n] for n in names])
        
        # 1. Pairwise Heatmaps
        for metric in ['cosine', 'euclidean']:
            dist_matrix = pairwise_distances(matrix, metric=metric)
            fig_w = max(12, min(28, len(names) * 1.0))
            fig_h = max(10, min(24, len(names) * 0.8))
            plt.figure(figsize=(fig_w, fig_h), dpi=300)
            sns.heatmap(dist_matrix, xticklabels=names, yticklabels=names, cmap='viridis')
            plt.title(f"Pairwise {metric.capitalize()} Distance (Class: {class_name})", fontsize=24, pad=20)
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"heatmap_{metric}_{class_name}.png"))
            plt.close()
            
            if metric == 'euclidean' and matrix.shape[0] > 1:
                try:
                    # Dendrogram
                    Z = linkage(matrix, method='ward', metric='euclidean')
                    plt.figure(figsize=(fig_w, fig_h), dpi=300)
                    dendrogram(Z, labels=names, leaf_rotation=90, leaf_font_size=16)
                    plt.title(f"Hierarchical Clustering Dendrogram of RF-DETR Signatures (Class: {class_name})", fontsize=28, pad=20)
                    plt.tight_layout()
                    plt.savefig(os.path.join(OUTPUT_DIR, f"dendrogram_euclidean_class_{class_name}.png"), dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # Clustermap
                    g = sns.clustermap(dist_matrix, row_linkage=Z, col_linkage=Z, xticklabels=names, yticklabels=names, cmap='viridis', figsize=(fig_w, fig_h))
                    g.fig.suptitle(f"Clustermap Euclidean Distance of RF-DETR Signatures (Class: {class_name})", fontsize=28, y=1.02)
                    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=16)
                    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=16)
                    plt.savefig(os.path.join(OUTPUT_DIR, f"clustermap_euclidean_class_{class_name}.png"), dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"Failed to generate hierarchical clustering for {class_name}: {e}")
            
        # 2. KMeans for colors
        n_clusters = min(4, len(names))
        if n_clusters < 2:
            continue
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(matrix)
        
        # 3. t-SNE
        for perplexity in [5, 15, 30]:
            if matrix.shape[0] - 1 < perplexity:
                continue
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            tsne_2d = tsne.fit_transform(matrix)
            
            plt.figure(figsize=(16, 16), dpi=300)
            sns.scatterplot(x=tsne_2d[:, 0], y=tsne_2d[:, 1], hue=kmeans.labels_, palette="tab10", s=300)
            
            texts = [plt.text(tsne_2d[i, 0], tsne_2d[i, 1], n, fontsize=14) for i, n in enumerate(names)]
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray'))
            
            plt.title(f"t-SNE (perp={perplexity}) colored by Cluster ({class_name})", fontsize=24)
            plt.savefig(os.path.join(OUTPUT_DIR, f"tsne_p{perplexity}_{class_name}.png"))
            plt.close()
            
        # 4. UMAP
        if matrix.shape[0] > 2:
            umap_reducer = umap.UMAP(n_components=2, n_neighbors=min(15, matrix.shape[0]-1), random_state=42)
            umap_2d = umap_reducer.fit_transform(matrix)
            
            plt.figure(figsize=(16, 16), dpi=300)
            sns.scatterplot(x=umap_2d[:, 0], y=umap_2d[:, 1], hue=kmeans.labels_, palette="tab10", s=300)
            
            texts = [plt.text(umap_2d[i, 0], umap_2d[i, 1], n, fontsize=14) for i, n in enumerate(names)]
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray'))
            
            plt.title(f"UMAP colored by Cluster ({class_name})", fontsize=24)
            plt.savefig(os.path.join(OUTPUT_DIR, f"umap_{class_name}.png"))
            plt.close()

    print("\nProcessing All Classes Combined...")
    all_dataset_names = set()
    for cat in all_embeddings:
        all_dataset_names.update(all_embeddings[cat].keys())
    
    all_dataset_names = list(all_dataset_names)
    combined_matrix = []
    for name in all_dataset_names:
        # Aggregate over each class label
        available_embs = [all_embeddings[cat][name] for cat in sorted(CLASS_MAP.keys()) if name in all_embeddings[cat]]
        if available_embs:
            combined_emb = np.mean(available_embs, axis=0)
        else:
            dim = next(iter(next(iter(all_embeddings.values())).values())).shape[0]
            combined_emb = np.zeros(dim)
        combined_matrix.append(combined_emb)
    combined_matrix = np.array(combined_matrix)

    if len(all_dataset_names) > 0:
        for dist_metric in ['cosine', 'euclidean']:
            dist_matrix = pairwise_distances(combined_matrix, metric=dist_metric)
            
            fig_w = max(12, min(28, len(all_dataset_names) * 1.0))
            fig_h = max(10, min(24, len(all_dataset_names) * 0.8))
            
            plt.figure(figsize=(fig_w, fig_h), dpi=300)
            sns.heatmap(dist_matrix, xticklabels=all_dataset_names, yticklabels=all_dataset_names, cmap='viridis')
            plt.title(f"Pairwise {dist_metric.capitalize()} Distance of RF-DETR Signatures (All Classes)", fontsize=28, pad=20)
            plt.xticks(rotation=90, fontsize=16)
            plt.yticks(rotation=0, fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"heatmap_{dist_metric}_all_classes.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            if dist_metric == 'euclidean' and combined_matrix.shape[0] > 1:
                try:
                    # Dendrogram
                    Z = linkage(combined_matrix, method='ward', metric='euclidean')
                    plt.figure(figsize=(fig_w, fig_h), dpi=300)
                    dendrogram(Z, labels=all_dataset_names, leaf_rotation=90, leaf_font_size=16)
                    plt.title("Hierarchical Clustering Dendrogram of RF-DETR Signatures (All Classes)", fontsize=28, pad=20)
                    plt.tight_layout()
                    plt.savefig(os.path.join(OUTPUT_DIR, "dendrogram_euclidean_all_classes.png"), dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # Clustermap
                    g = sns.clustermap(dist_matrix, row_linkage=Z, col_linkage=Z, xticklabels=all_dataset_names, yticklabels=all_dataset_names, cmap='viridis', figsize=(fig_w, fig_h))
                    g.fig.suptitle("Clustermap Euclidean Distance of RF-DETR Signatures (All Classes)", fontsize=28, y=1.02)
                    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=16)
                    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=16)
                    plt.savefig(os.path.join(OUTPUT_DIR, "clustermap_euclidean_all_classes.png"), dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"Failed to generate hierarchical clustering for All Classes: {e}")
        
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
                    
                plt.title(f"t-SNE (perp={perplexity}) of RF-DETR Signatures Colored by Cluster (All Classes)", fontsize=26, pad=20)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16, borderaxespad=0.)
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, f"tsne_p{perplexity}_all_classes.png"), dpi=300, bbox_inches='tight')
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
                    
                plt.title("UMAP of RF-DETR Signatures Colored by Cluster (All Classes)", fontsize=26, pad=20)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16, borderaxespad=0.)
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, "umap_all_classes.png"), dpi=300, bbox_inches='tight')
                plt.close()

if __name__ == "__main__":
    print("Loading RF-DETR-Seg Backbone...")
    encoder, device = build_rfdetr_backbone(CHECKPOINT_PATH)
    print("Extracting Embeddings...")
    embeddings = process_all_datasets(encoder, device)
    print("Generating Visualizations...")
    generate_visualizations(embeddings)
    print(f"Done. Results saved to {OUTPUT_DIR}")
