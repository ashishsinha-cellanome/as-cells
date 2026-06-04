import os
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

def extract_crops_and_features(img_path, mask_pkl_path, encoder, device):
    img = cv2.imread(str(img_path))
    if img is None:
        return {}
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    with open(mask_pkl_path, 'rb') as f:
        mask_data = pickle.load(f)
        
    if not mask_data.get('annotations'):
        return {}
        
    results = {cat: [] for cat in TARGET_CLASSES}
    
    for ann in mask_data['annotations']:
        cat = int(ann['category_id'])
        if cat not in TARGET_CLASSES:
            continue
            
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
        
        # Resize for RFDETR backbone (240x240)
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
            
        results[cat].append(cls_token[0].cpu().numpy())
            
    return results

def process_all_datasets(encoder, device):
    all_embeddings = {cat: {} for cat in TARGET_CLASSES}
    base_dir = Path(DATA_DIR)
    
    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist.")
        return all_embeddings

    for ds in tqdm(list(base_dir.iterdir())):
        if not ds.is_dir():
            continue
            
        cell_line = parse_dataset_name(ds.name)
        if not cell_line:
            continue
            
        img_dir = ds / "images" / "test"
        mask_dir = ds / "masks" / "test"
        
        if not img_dir.exists() or not mask_dir.exists():
            img_dir = ds / "images" / "train"
            mask_dir = ds / "masks" / "train"
            
        if not img_dir.exists() or not mask_dir.exists():
            continue
        
        imgs = sorted(list(img_dir.iterdir()))
        
        for img in imgs:
            mask_path = mask_dir / f"{img.stem}.pkl"
            if mask_path.exists():
                class_embs = extract_crops_and_features(img, mask_path, encoder, device)
                for cat, embs in class_embs.items():
                    if embs:
                        if ds.name not in all_embeddings[cat]:
                            all_embeddings[cat][ds.name] = []
                        all_embeddings[cat][ds.name].extend(embs)
                        
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
    covmean, _ = sqrtm(cov_A.dot(cov_B), disp=False)
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
    
    mmd_val = 0
    for s in sigmas:
        K_XX = rbf_kernel(X, X, s)
        K_YY = rbf_kernel(Y, Y, s)
        K_XY = rbf_kernel(X, Y, s)
        mmd_val += np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    return mmd_val / 3.0

def generate_distance_plots(matrix, names, output_dir, metric_name, class_name, level_name):
    os.makedirs(output_dir, exist_ok=True)
    
    # Heatmap
    fig_w = max(12, min(28, len(names) * 1.0))
    fig_h = max(10, min(24, len(names) * 0.8))
    plt.figure(figsize=(fig_w, fig_h), dpi=300)
    sns.heatmap(matrix, xticklabels=names, yticklabels=names, cmap='viridis')
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
                g = sns.clustermap(matrix, row_linkage=Z, col_linkage=Z, xticklabels=names, yticklabels=names, cmap='viridis', figsize=(fig_w, fig_h))
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
        fig, axes = plt.subplots(2, 2, figsize=(20, 20), dpi=300)
        axes = axes.flatten()
        
        for i, p in enumerate(perplexities):
            if n_samples - 1 < p:
                continue
            tsne = TSNE(n_components=2, perplexity=p, random_state=42)
            tsne_2d = tsne.fit_transform(features)
            
            sns.scatterplot(x=tsne_2d[:, 0], y=tsne_2d[:, 1], hue=labels, ax=axes[i], palette="tab10", s=50)
            axes[i].set_title(f"t-SNE perp={p}")
            axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, borderaxespad=0.)
            
        plt.suptitle(f"t-SNE colored by {label_type}", fontsize=24)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"tsne_{label_type}.png"), bbox_inches='tight')
        plt.close()
    
    # UMAP
    for labels, label_type in [(ds_labels, 'dataset'), (cl_labels, 'cell_line')]:
        n_neighbors = min(15, n_samples - 1)
        umap_reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
        umap_2d = umap_reducer.fit_transform(features)
        
        plt.figure(figsize=(12, 10), dpi=300)
        sns.scatterplot(x=umap_2d[:, 0], y=umap_2d[:, 1], hue=labels, palette="tab10", s=50)
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
    for name in names:
        embs = embeddings_dict[name]
        mu = np.mean(embs, axis=0)
        var = np.var(embs, axis=0)
        cov = LedoitWolf().fit(embs).covariance_ if embs.shape[0] > 1 else np.zeros((embs.shape[1], embs.shape[1]))
        stats[name] = {'mu': mu, 'var': var, 'cov': cov, 'embs': embs}
        
    print(f"Computing metrics for {class_name} ({level_name}) - {num_entities} entities")
    
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
            cell_line_embs[cl] = np.vstack(cell_line_embs[cl])
            
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
            if n > 500:
                indices = np.random.choice(n, 500, replace=False)
                sub_embs = embs[indices]
            else:
                sub_embs = embs
                
            all_features.append(sub_embs)
            ds_labels.extend([ds] * sub_embs.shape[0])
            cl = parse_dataset_name(ds)
            cl_labels.extend([cl] * sub_embs.shape[0])
            
        if all_features:
            all_features = np.vstack(all_features)
            generate_scatter_plots(all_features, ds_labels, cl_labels, inst_dir)

if __name__ == "__main__":
    main()
