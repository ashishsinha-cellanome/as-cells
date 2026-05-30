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
    2: "soma",
    3: "cell-adhered"
}

# Add model checkpoint path (can be overridden via CLI args later)
CHECKPOINT_PATH = "/mnt/personal/cellanome/checkpoints/RFDETR-Seg-ckpts/output/checkpoint_best_ema.pth"