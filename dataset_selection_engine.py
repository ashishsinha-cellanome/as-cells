import os
import cv2
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from pathlib import Path
from tqdm import tqdm
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

DATA_DIR = "/mnt/direct-attached/PHASE2"
OUTPUT_DIR = "/mnt/direct-attached/PHASE2_EVAL_RESULTS/selection_engine"

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
        return None, None
    
    with open(mask_pkl_path, 'rb') as f:
        mask_data = pickle.load(f)
        
    if not mask_data.get('annotations'):
        return None, None
        
    # Find largest annotation by bbox area
    largest_ann = max(mask_data['annotations'], key=lambda a: (a['bbox'][2]-a['bbox'][0])*(a['bbox'][3]-a['bbox'][1]))
    
    x1, y1, x2, y2 = [int(v) for v in largest_ann['bbox']]
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None, None
        
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
    
    # Normalize grid to 0-255
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    grid = (grid * 255).astype(np.uint8)
    
    # Colormap and resize
    heatmap = cv2.applyColorMap(grid, cv2.COLORMAP_VIRIDIS)
    heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
    
    # Overlay
    overlay = cv2.addWeighted(resized_crop, 0.5, heatmap, 0.5, 0)
    
    out_path = os.path.join(OUTPUT_DIR, f"{dataset_name}-{img_path.stem}-emb.png")
    cv2.imwrite(out_path, overlay)
    
    return cls_token[0].cpu().numpy(), out_path

def process_all_datasets(model, device):
    all_embeddings = {}
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
        
        emb, _ = extract_largest_object_and_features(first_img, mask_path, ds.name, model, device)
        if emb is not None:
            all_embeddings[ds.name] = emb
            
    if not all_embeddings:
        print("No embeddings extracted.")
        return
        
    names = list(all_embeddings.keys())
    matrix = np.array([all_embeddings[n] for n in names])
    
    # Distance heatmap
    dist_matrix = pairwise_distances(matrix, metric='cosine')
    plt.figure(figsize=(12, 10))
    sns.heatmap(dist_matrix, xticklabels=names, yticklabels=names, cmap='viridis')
    plt.title("Pairwise Cosine Distance of Morphological Signatures")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pairwise_distance_heatmap.png"), dpi=300)
    plt.close()
    
    # KMeans Clustering
    n_clusters = min(4, len(names))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(matrix)
    
    print("\nClustering Results:")
    for i in range(n_clusters):
        cluster_names = [names[j] for j in range(len(names)) if kmeans.labels_[j] == i]
        print(f"Cluster {i}: {cluster_names}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading model...")
    model, device = build_dinov2()
    print("Model loaded. Processing all datasets...")
    process_all_datasets(model, device)
    print(f"Results saved to {OUTPUT_DIR}")