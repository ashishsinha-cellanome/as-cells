import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap.umap_ as umap
from pathlib import Path
from tqdm import tqdm
import pickle
import pycocotools.mask as mask_util
from sklearn.metrics import pairwise_distances
from sklearn.covariance import LedoitWolf
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.linalg import sqrtm
from omegaconf import OmegaConf

DATA_DIR = "/mnt/direct-attached/PHASE2"
OUTPUT_DIR = "/mnt/direct-attached/PHASE2_EVAL_RESULTS/custom_cell_line_embeddings_analysis"
CHECKPOINT_PATH = "/mnt/personal/cellanome/checkpoints/ALL_CKPTS/RFDETR-Seg-ckpts/output/checkpoint_best_ema.pth"

# Load the label map dynamically from the rfdetr_seg model config
_model_cfg = OmegaConf.load("configs/model/rfdetr_seg.yaml")
CLASS_MAP = OmegaConf.to_container(_model_cfg.label_map, resolve=True)
# We only care about 0 (cell), 2 (cell-adhered), 3 (soma). Class 1 (beads) is skipped everywhere.

TARGET_CLASSES = {0, 2, 3}

def parse_dataset_name(ds_name):
    # Exclude suspension datasets and beads
    lower_name = ds_name.lower()
    suspension_keywords = ["suspension", "jurkat", "k562", "nk92", "pbmc", "mousepbmc", "tall104", "raji", "jerat", "tal104"]
    for kw in suspension_keywords:
        if kw in lower_name:
            return None
    
    # Extract clean cell_line string
    clean_name = lower_name
    prefixes_suffixes_to_remove = ["-adhered", "-uncaged", "-caged", "10x", "1x", "20x"]
    for ps in prefixes_suffixes_to_remove:
        clean_name = clean_name.replace(ps, "")
    
    # Maybe also clean up underscores/hyphens at edges
    clean_name = clean_name.strip("-_")
    return clean_name

def build_rfdetr_backbone(checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import rfdetr
    print(f"Loading RFDETRSegLarge from {checkpoint}")
    if not os.path.exists(checkpoint):
        print("Falling back to randomly initialized RF-DETR backbone.")
        model = rfdetr.RFDETRSegLarge(pretrain_weights=None, group_detr=1, num_classes=4)
    else:
        try:
            model = rfdetr.RFDETRSegLarge(pretrain_weights=checkpoint, group_detr=1, num_classes=4)
        except Exception:
            model = rfdetr.RFDETRSegLarge(pretrain_weights=None, group_detr=1, num_classes=4)
        
    model.model.model.to(device)
    model.model.model.eval()
    
    encoder = model.model.model.backbone[0].encoder
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder, device


def save_crop_visualizations(ds_name, class_name, crops, output_dir):
    if not crops:
        return
    crops_np = np.array(crops, dtype=np.float32) / 255.0
    mean_img = np.mean(crops_np, axis=0)
    var_img = np.var(crops_np, axis=0)
    if np.max(var_img) > 0:
        var_img_norm = var_img / np.max(var_img)
    else:
        var_img_norm = var_img
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(mean_img)
    plt.title(f"{ds_name}\nMean Crop (n={len(crops)})")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    var_heatmap = np.mean(var_img_norm, axis=-1)
    plt.imshow(var_heatmap, cmap='hot')
    plt.title("Variance Heatmap")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{ds_name}_{class_name}_mean_var.png"), bbox_inches='tight')
    plt.close()
    
    n_samples = min(16, len(crops))
    if n_samples > 0:
        grid_size = int(np.ceil(np.sqrt(n_samples)))
        plt.figure(figsize=(grid_size * 2, grid_size * 2))
        for i in range(n_samples):
            plt.subplot(grid_size, grid_size, i + 1)
            plt.imshow(crops[i])
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{ds_name}_{class_name}_samples.png"), bbox_inches='tight')
        plt.close()

def extract_crops_cpu(img_path, mask_pkl_path, input_size):
    img = cv2.imread(str(img_path))
    if img is None:
        return {}
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    with open(mask_pkl_path, 'rb') as f:
        mask_data = pickle.load(f)
        
    if not mask_data.get('annotations'):
        return {}
        
    results = {cat: [] for cat in TARGET_CLASSES}
    crops_by_cat = {cat: [] for cat in TARGET_CLASSES}
    
    for ann in mask_data['annotations']:
        cat = int(ann['category_id'])
        if cat not in TARGET_CLASSES:
            continue
            
        x, y, x2, y2 = [int(float(v)) for v in ann['bbox']]
        w, h = x2 - x, y2 - y
        if w <= 0 or h <= 0:
            continue
            
        img_h, img_w = img.shape[:2]
        
        # Create a masked version of the image using the segmentation mask
        if isinstance(ann['segmentation'], list):
            rle = mask_util.frPyObjects(ann['segmentation'], img_h, img_w)
            mask = mask_util.decode(rle)
        else:
            mask = mask_util.decode(ann['segmentation'])
        if len(mask.shape) == 3: # sometimes decode returns (H, W, 1)
            mask = mask.squeeze(2)
            
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        
        x1_clip, y1_clip = max(0, x), max(0, y)
        x2_clip, y2_clip = min(img_w, x + w), min(img_h, y + h)
        
        if mask.shape[:2] == (img_h, img_w):
            mask_3d_cropped = mask_3d[y1_clip:y2_clip, x1_clip:x2_clip]
        else:
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
        
        # Resize for RFDETR backbone (240x240)
        resized_crop = cv2.resize(crop, (input_size, input_size), interpolation=cv2.INTER_AREA)
        crops_by_cat[cat].append(resized_crop)
        
    return crops_by_cat


def process_all_datasets(encoder, device):
    all_embeddings = {cat: {} for cat in TARGET_CLASSES}
    base_dir = Path(DATA_DIR)
    if not base_dir.exists():
        return all_embeddings
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_size = 240
    
    for ds in tqdm(list(base_dir.iterdir())):
        if not ds.is_dir(): continue
        cell_line = parse_dataset_name(ds.name)
        if not cell_line: continue
        
        img_dir = ds / "images" / "test"
        mask_dir = ds / "masks" / "test"
        if not img_dir.exists() or not mask_dir.exists():
            img_dir = ds / "images" / "train"
            mask_dir = ds / "masks" / "train"
        if not img_dir.exists() or not mask_dir.exists(): continue
        
        imgs = sorted(list(img_dir.iterdir()))
        viz_saved = {cat: False for cat in TARGET_CLASSES}
        viz_dir = os.path.join(OUTPUT_DIR, "crop_visualizations")
        
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            for img in imgs:
                mask_path = mask_dir / f"{img.stem}.pkl"
                if mask_path.exists():
                    futures.append(executor.submit(extract_crops_cpu, img, mask_path, input_size))
            
            for future in as_completed(futures):
                crops_by_cat = future.result()
                if not crops_by_cat: continue
                
                for cat, crops in crops_by_cat.items():
                    if not crops: continue
                    
                    if not viz_saved[cat]:
                        save_crop_visualizations(ds.name, str(cat), crops, viz_dir)
                        viz_saved[cat] = True
                        
                    tensors = [transform(c) for c in crops]
                    cat_embs = []
                    for i in range(0, len(tensors), 32):
                        batch = torch.stack(tensors[i:i + 32]).to(device)
                        with torch.no_grad():
                            features = encoder(batch)
                            cls_tokens = features[-1].mean(dim=(2, 3))
                        cat_embs.extend(cls_tokens.cpu().numpy())
                        
                    if cat_embs:
                        if ds.name not in all_embeddings[cat]:
                            all_embeddings[cat][ds.name] = []
                        all_embeddings[cat][ds.name].extend(cat_embs)
                        
    return all_embeddings


def dist_euclidean(mu_A, mu_B):
    return np.linalg.norm(mu_A - mu_B)

def dist_cosine(mu_A, mu_B):
    norm_A = np.linalg.norm(mu_A)
    norm_B = np.linalg.norm(mu_B)
    if norm_A == 0 or norm_B == 0:
        return 1.0
    return 1 - np.dot(mu_A, mu_B) / (norm_A * norm_B)

def dist_l2_var_norm(mu_A, mu_B, var_A, var_B, eps=1e-8):
    num = np.linalg.norm(mu_A - mu_B)
    den = np.sqrt(np.mean(var_A)) + np.sqrt(np.mean(var_B)) + eps
    return num / den

def dist_cosine_var_weighted(mu_A, mu_B, var_A, var_B, eps=1e-8):
    w = 1.0 / np.sqrt(var_A + var_B + eps)
    mu_A_prime = mu_A * w
    mu_B_prime = mu_B * w
    norm_A = np.linalg.norm(mu_A_prime)
    norm_B = np.linalg.norm(mu_B_prime)
    if norm_A == 0 or norm_B == 0:
        return 1.0
    return 1 - np.dot(mu_A_prime, mu_B_prime) / (norm_A * norm_B)

def dist_var_norm_dim(mu_A, mu_B, var_A, var_B, eps=1e-8):
    return np.sqrt(np.sum((mu_A - mu_B)**2 / (var_A + var_B + eps)))

def dist_fid(mu_A, mu_B, cov_A, cov_B):
    diff = mu_A - mu_B
    covmean = sqrtm(cov_A.dot(cov_B))
    if not np.isfinite(covmean).all():
        offset = np.eye(cov_A.shape[0]) * 1e-6
        covmean = sqrtm((cov_A + offset).dot(cov_B + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(cov_A) + np.trace(cov_B) - 2 * tr_covmean

def rbf_kernel(X, Y, sigma):
    dists = pairwise_distances(X, Y, metric='sqeuclidean')
    return np.exp(-dists / (2 * sigma**2))

def dist_mmd_rbf(X, Y):
    # Subsample if > 5000 instances to avoid O(N^2) blowup
    if X.shape[0] > 5000:
        indices = np.random.choice(X.shape[0], 5000, replace=False)
        X = X[indices]
    if Y.shape[0] > 5000:
        indices = np.random.choice(Y.shape[0], 5000, replace=False)
        Y = Y[indices]
    
    # Median heuristic for sigma
    pooled = np.vstack([X, Y])
    # calculate pairwise dists on a smaller subset if pooled is still large to find median
    sub_pooled = pooled
    if pooled.shape[0] > 1000:
        sub_pooled = pooled[np.random.choice(pooled.shape[0], 1000, replace=False)]
    
    dists = pairwise_distances(sub_pooled, sub_pooled, metric='euclidean')
    sigma_m = np.median(dists[dists > 0])
    if sigma_m == 0:
        sigma_m = 1.0
        
    sigmas = [0.5 * sigma_m, sigma_m, 2.0 * sigma_m]
    
    # Compute pairwise distance matrices ONCE (was: recomputed 9 times in loop)
    dist_XX = pairwise_distances(X, X, metric='sqeuclidean')
    dist_YY = pairwise_distances(Y, Y, metric='sqeuclidean')
    dist_XY = pairwise_distances(X, Y, metric='sqeuclidean')
    
    mmd_val = 0
    for s in sigmas:
        K_XX = np.exp(-dist_XX / (2 * s**2))
        K_YY = np.exp(-dist_YY / (2 * s**2))
        K_XY = np.exp(-dist_XY / (2 * s**2))
        mmd_val += np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    return mmd_val / 3.0

def generate_distance_plots(matrix, names, output_dir, metric_name, class_name, level_name):
    os.makedirs(output_dir, exist_ok=True)
    
    # Heatmap
    fig_w = max(12, min(36, len(names) * 1.5))
    fig_h = max(10, min(30, len(names) * 1.2))
    plt.figure(figsize=(fig_w, fig_h), dpi=300)
    sns.heatmap(matrix, xticklabels=names, yticklabels=names, cmap='viridis', annot=True, fmt='.3g', annot_kws={'size': 10})
    plt.title(f"Pairwise {metric_name} Distance (Class: {class_name}, {level_name})", fontsize=24, pad=20)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"heatmap_{metric_name}.png"))
    plt.close()
    
    if matrix.shape[0] > 1:
        try:
            # Dendrogram
            Z = linkage(matrix, method='average')
            plt.figure(figsize=(fig_w, fig_h), dpi=300)
            dendrogram(Z, labels=names, leaf_rotation=90, leaf_font_size=16)
            plt.title(f"Hierarchical Clustering Dendrogram ({metric_name})", fontsize=28, pad=20)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"dendrogram_{metric_name}.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Clustermap
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                g = sns.clustermap(matrix, row_linkage=Z, col_linkage=Z, xticklabels=names, yticklabels=names, cmap='viridis', figsize=(fig_w, fig_h), annot=True, fmt='.3g', annot_kws={'size': 10})
                g.fig.suptitle(f"Clustermap {metric_name} Distance", fontsize=28, y=1.02)
                plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=16)
                plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=16)
                plt.savefig(os.path.join(output_dir, f"clustermap_{metric_name}.png"), dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"Failed to generate clustering for {metric_name}: {e}")

def generate_scatter_plots(features, ds_labels, cl_labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    n_samples = features.shape[0]
    if n_samples < 2:
        return
        
    # t-SNE
    perplexities = [30, 50, 70, 90]
    
    for labels, label_type in [(ds_labels, 'dataset'), (cl_labels, 'cell_line')]:
        for p in perplexities:
            if n_samples - 1 < p:
                continue
            tsne = TSNE(n_components=2, perplexity=p, random_state=42)
            tsne_2d = tsne.fit_transform(features)
            
            plt.figure(figsize=(12, 10), dpi=300)
            unique_labels = len(np.unique(labels))
            palette = sns.color_palette("husl", unique_labels)
            sns.scatterplot(x=tsne_2d[:, 0], y=tsne_2d[:, 1], hue=labels, palette=palette, s=10)
            plt.title(f"t-SNE colored by {label_type} (perp={p})", fontsize=20)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, borderaxespad=0.)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"tsne_{label_type}_perp{p}.png"), bbox_inches='tight')
            plt.close()
    
    # UMAP
    for labels, label_type in [(ds_labels, 'dataset'), (cl_labels, 'cell_line')]:
        n_neighbors = min(15, n_samples - 1)
        umap_reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
        umap_2d = umap_reducer.fit_transform(features)
        
        plt.figure(figsize=(12, 10), dpi=300)
        unique_labels = len(np.unique(labels))
        palette = sns.color_palette("husl", unique_labels)
        sns.scatterplot(x=umap_2d[:, 0], y=umap_2d[:, 1], hue=labels, palette=palette, s=10)
        plt.title(f"UMAP colored by {label_type}", fontsize=20)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"umap_{label_type}.png"), bbox_inches='tight')
        plt.close()

def compute_and_plot_metrics(class_name, embeddings_dict, level_name, out_dir):
    names = sorted(list(embeddings_dict.keys()))
    num_entities = len(names)
    
    if num_entities < 2:
        return
        
    metrics = {
        'euclidean': np.zeros((num_entities, num_entities)),
        'cosine': np.zeros((num_entities, num_entities)),
        'l2_var_norm': np.zeros((num_entities, num_entities)),
        'cosine_var_weighted': np.zeros((num_entities, num_entities)),
        'var_norm_dim': np.zeros((num_entities, num_entities)),
        'fid': np.zeros((num_entities, num_entities)),
        'mmd_rbf': np.zeros((num_entities, num_entities))
    }
    
    stats = {}
    print(f"Computing per-dataset statistics (mean, var, cov)...")
    csv_rows = []
    for idx, name in enumerate(names):
        embs = embeddings_dict[name]
        print(f"  [{idx+1}/{num_entities}] {name}: {embs.shape[0]} instances")
        mu = np.mean(embs, axis=0)
        var = np.var(embs, axis=0)
        
        mu_norm = np.linalg.norm(mu)
        var_norm = np.linalg.norm(var)
        csv_rows.append(f"{name},{embs.shape[0]},{class_name},{level_name},{mu_norm},{var_norm}")
        
        # Subsample for LedoitWolf to avoid O(N*D^2) blowup with large datasets
        if embs.shape[0] > 5000:
            sub_idx = np.random.choice(embs.shape[0], 5000, replace=False)
            cov = LedoitWolf().fit(embs[sub_idx]).covariance_
        elif embs.shape[0] > 1:
            cov = LedoitWolf().fit(embs).covariance_
        else:
            cov = np.zeros((embs.shape[1], embs.shape[1]))
        stats[name] = {'mu': mu, 'var': var, 'cov': cov, 'embs': embs}
        
    csv_path = os.path.join(out_dir, "dataset_statistics.csv")
    with open(csv_path, 'w') as f:
        header = "dataset_name,num_instances,class,level,mean_l2_norm,var_l2_norm"
        f.write(header + "\n")
        f.write("\n".join(csv_rows) + "\n")
        
    pkl_path = os.path.join(out_dir, "dataset_statistics.pkl")
    stats_to_save = {name: {'mu': stats[name]['mu'], 'var': stats[name]['var']} for name in names}
    with open(pkl_path, 'wb') as f:
        pickle.dump(stats_to_save, f)
        
    print(f"Statistics saved to {csv_path} and {pkl_path}")
        
    total_pairs = num_entities * (num_entities - 1) // 2
    print(f"Computing metrics for {class_name} ({level_name}) - {num_entities} entities, {total_pairs} pairs")
    pbar = tqdm(total=total_pairs, desc=f"{class_name} {level_name}", position=0, leave=True, file=sys.stdout)
    pair_idx = 0
    
    for i in range(num_entities):
        for j in range(i, num_entities):
            if i == j:
                continue
                
            name_A, name_B = names[i], names[j]
            s_A, s_B = stats[name_A], stats[name_B]
            
            metrics['euclidean'][i, j] = metrics['euclidean'][j, i] = dist_euclidean(s_A['mu'], s_B['mu'])
            metrics['cosine'][i, j] = metrics['cosine'][j, i] = dist_cosine(s_A['mu'], s_B['mu'])
            metrics['l2_var_norm'][i, j] = metrics['l2_var_norm'][j, i] = dist_l2_var_norm(s_A['mu'], s_B['mu'], s_A['var'], s_B['var'])
            metrics['cosine_var_weighted'][i, j] = metrics['cosine_var_weighted'][j, i] = dist_cosine_var_weighted(s_A['mu'], s_B['mu'], s_A['var'], s_B['var'])
            metrics['var_norm_dim'][i, j] = metrics['var_norm_dim'][j, i] = dist_var_norm_dim(s_A['mu'], s_B['mu'], s_A['var'], s_B['var'])
            metrics['fid'][i, j] = metrics['fid'][j, i] = dist_fid(s_A['mu'], s_B['mu'], s_A['cov'], s_B['cov'])
            metrics['mmd_rbf'][i, j] = metrics['mmd_rbf'][j, i] = dist_mmd_rbf(s_A['embs'], s_B['embs'])
            pair_idx += 1
            pbar.update(1)
    pbar.close()
    for metric_name, matrix in metrics.items():
        generate_distance_plots(matrix, names, out_dir, metric_name, class_name, level_name)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Build and extract (or load)
    out_pkl = os.path.join(OUTPUT_DIR, "extracted_raw_embeddings.pkl")
    if os.path.exists(out_pkl):
        print(f"Loading cached embeddings from {out_pkl}")
        with open(out_pkl, 'rb') as f:
            all_embeddings = pickle.load(f)
    else:
        print("Loading RF-DETR-Seg Backbone...")
        encoder, device = build_rfdetr_backbone(CHECKPOINT_PATH)
        print("Extracting Embeddings...")
        all_embeddings = process_all_datasets(encoder, device)
        
        # Convert list of embeddings to numpy arrays
        for cat in all_embeddings:
            for ds in list(all_embeddings[cat].keys()):
                if len(all_embeddings[cat][ds]) > 0:
                    all_embeddings[cat][ds] = np.vstack(all_embeddings[cat][ds])
                else:
                    del all_embeddings[cat][ds]
                    
        with open(out_pkl, 'wb') as f:
            pickle.dump(all_embeddings, f)
            
    # 2. Process per class
    for cat in TARGET_CLASSES:
        if cat not in CLASS_MAP or cat not in all_embeddings:
            continue
            
        class_name = CLASS_MAP[cat]
        class_embs = all_embeddings[cat]
        
        if not class_embs:
            continue
            
        print(f"Processing Class: {class_name}")
        class_dir = os.path.join(OUTPUT_DIR, f"class_{class_name}")
        os.makedirs(class_dir, exist_ok=True)
        
        # Collect cell lines
        dataset_embs = {}
        cell_line_embs = {}
        
        for ds, embs in class_embs.items():
            if embs.shape[0] == 0:
                continue
            dataset_embs[ds] = embs
            cl = parse_dataset_name(ds)
            if cl not in cell_line_embs:
                cell_line_embs[cl] = []
            cell_line_embs[cl].append(embs)
            
        for cl in cell_line_embs:
            stacked = np.vstack(cell_line_embs[cl])
            if stacked.shape[0] > 5000:
                sub_idx = np.random.choice(stacked.shape[0], 5000, replace=False)
                stacked = stacked[sub_idx]
            cell_line_embs[cl] = stacked
            
        # Distance plots
        compute_and_plot_metrics(class_name, dataset_embs, "dataset_level", os.path.join(class_dir, "dataset_level"))
        compute_and_plot_metrics(class_name, cell_line_embs, "cell_line_level", os.path.join(class_dir, "cell_line_level"))
        
        # Scatter plots
        inst_dir = os.path.join(class_dir, "instances")
        all_features = []
        ds_labels = []
        cl_labels = []
        
        for ds, embs in dataset_embs.items():
            n = embs.shape[0]
            if n > 5000:
                indices = np.random.choice(n, 5000, replace=False)
                sub_embs = embs[indices]
            else:
                sub_embs = embs
                
            all_features.append(sub_embs)
            ds_labels.extend([ds] * sub_embs.shape[0])
            cl = parse_dataset_name(ds)
            cl_labels.extend([cl] * sub_embs.shape[0])
            
        if all_features:
            all_features = np.vstack(all_features)
            os.makedirs(inst_dir, exist_ok=True)
            scatter_data_path = os.path.join(inst_dir, "scatter_plot_data.pkl")
            with open(scatter_data_path, 'wb') as f:
                pickle.dump({'features': all_features, 'ds_labels': ds_labels, 'cl_labels': cl_labels}, f)
            print(f"Saved scatter plot features to {scatter_data_path}")
            generate_scatter_plots(all_features, ds_labels, cl_labels, inst_dir)

if __name__ == "__main__":
    main()
