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
from pycocotools.coco import COCO
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from adjustText import adjust_text

DATA_DIR = "/mnt/direct-attached/PHASE2"
OUTPUT_DIR = "/mnt/direct-attached/PHASE2_EVAL_RESULTS/rfdetr_seg_features"

CLASS_MAP = {
    0: "cell",
    1: "bead",
    2: "cell-adhered",
    3: "soma"
}

# Add model checkpoint path (can be overridden via CLI args later)
CHECKPOINT_PATH = "/mnt/personal/cellanome/checkpoints/RFDETR-Seg-ckpts/output/checkpoint_best_ema.pth"

def build_rfdetr_backbone(checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading RFDETRSegLarge from {checkpoint}")
    import rfdetr
    try:
        model = rfdetr.RFDETRSegLarge(pretrain_weights=checkpoint, group_detr=1, num_classes=4)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        raise e
        
    model.to(device)
    model.eval()
    
    encoder = model.model.model.backbone[0].encoder
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder, device

def extract_features_for_image(img_path, coco, img_info, encoder, device):
    img = cv2.imread(str(img_path))
    if img is None:
        return {}
    
    ann_ids = coco.getAnnIds(imgIds=img_info['id'])
    anns = coco.loadAnns(ann_ids)
    
    results = {}
    
    # Process largest object per class
    anns_by_cat = {}
    for ann in anns:
        cat = int(ann['category_id'])
        if cat not in anns_by_cat:
            anns_by_cat[cat] = []
        anns_by_cat[cat].append(ann)
        
    for cat, cat_anns in anns_by_cat.items():
        if cat not in CLASS_MAP:
            continue
            
        largest_ann = max(cat_anns, key=lambda a: a['bbox'][2] * a['bbox'][3])
        x, y, w, h = [int(v) for v in largest_ann['bbox']]
        
        if w <= 0 or h <= 0:
            continue
            
        size = max(w, h)
        cx, cy = x + w//2, y + h//2
        sq_x1, sq_y1 = max(0, cx - size//2), max(0, cy - size//2)
        sq_x2, sq_y2 = min(img.shape[1], sq_x1 + size), min(img.shape[0], sq_y1 + size)
        
        actual_crop = img[sq_y1:sq_y2, sq_x1:sq_x2]
        if actual_crop.size == 0:
            continue
            
        crop = np.zeros((size, size, 3), dtype=np.uint8)
        offset_x, offset_y = sq_x1 - (cx - size//2), sq_y1 - (cy - size//2)
        crop[offset_y:offset_y+actual_crop.shape[0], offset_x:offset_x+actual_crop.shape[1]] = actual_crop
        
        # Resize for RFDETR backbone (requires input divisible by 24, e.g. 240x240)
        resized_crop = cv2.resize(crop, (240, 240))
        
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(resized_crop).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = encoder(input_tensor)
            last_feat = features[-1]
            cls_token = last_feat.mean(dim=(2, 3)) # (1, C) Global Avg Pooling
            
        results[cat] = cls_token[0].cpu().numpy()
        
    return results

def process_all_datasets(encoder, device):
    all_embeddings = {cat: {} for cat in CLASS_MAP.keys()}
    base_dir = Path(DATA_DIR)
    
    for ds in tqdm(list(base_dir.iterdir())):
        if not ds.is_dir():
            continue
            
        anno_file = ds / "test_annotations.json"
        img_dir = ds / "images" / "test"
        
        if not anno_file.exists() or not img_dir.exists():
            continue
            
        coco = COCO(str(anno_file))
        image_ids = coco.getImgIds()
        if not image_ids:
            continue
            
        # Select first image for signature
        img_info = coco.loadImgs(image_ids[0])[0]
        img_path = img_dir / img_info['file_name']
        
        class_embs = extract_features_for_image(img_path, coco, img_info, encoder, device)
        for cat, emb in class_embs.items():
            all_embeddings[cat][ds.name] = emb
            
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
            plt.figure(figsize=(14, 12), dpi=300)
            sns.heatmap(dist_matrix, xticklabels=names, yticklabels=names, cmap='viridis')
            plt.title(f"Pairwise {metric.capitalize()} Distance (Class: {class_name})", fontsize=24, pad=20)
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"heatmap_{metric}_{class_name}.png"))
            plt.close()
            
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

if __name__ == "__main__":
    print("Loading RF-DETR-Seg Backbone...")
    encoder, device = build_rfdetr_backbone(CHECKPOINT_PATH)
    print("Extracting Embeddings...")
    embeddings = process_all_datasets(encoder, device)
    print("Generating Visualizations...")
    generate_visualizations(embeddings)
    print(f"Done. Results saved to {OUTPUT_DIR}")
