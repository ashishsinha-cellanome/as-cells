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
        offset_x, offset_y = sq_x1 - max(0, cx - size//2), sq_y1 - max(0, cy - size//2)
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
