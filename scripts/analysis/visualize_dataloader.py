#!/usr/bin/env python3

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import os
import sys
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from data.motif_data_module import MotifDataModule
from utils.distributed_utils import setup_cluster_env

setup_cluster_env()
torch.set_float32_matmul_precision("medium")
OmegaConf.register_new_resolver("oc.eval", eval, replace=True)

def plot_batch(batch_tensors, targets, label_map, out_path, title_prefix):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    batch_size = len(batch_tensors)
    num_to_plot = min(8, batch_size)
    cols = min(2, num_to_plot)
    rows = (num_to_plot + cols - 1) // cols
    
    fig, axs = plt.subplots(rows, cols, figsize=(8 * cols, 8 * rows))
    if num_to_plot == 1:
        axs = np.array([axs])
    axs = axs.flatten()
    
    colors = ['#FF3838', '#2C99A8', '#FF701F', '#28B463', '#FF9D97', '#9D00FF', '#0000FF']
    def hex_to_rgb(hex_color):
        h = hex_color.lstrip('#')
        return np.array(tuple(int(h[i:i+2], 16) for i in (0, 2, 4))) / 255.0
    
    for i in range(num_to_plot):
        img = batch_tensors[i]
        target = targets[i]
        
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        img_np = img.permute(1, 2, 0).cpu().numpy()
        h, w = img_np.shape[:2]
        
        axs[i].imshow(img_np)
        axs[i].set_title(f"{title_prefix}\nImg {target.get('image_id', torch.tensor(i)).item()}: {w}x{h}", fontsize=12)
        
        if "masks" in target and len(target["masks"]) > 0:
            masks = target["masks"].cpu().numpy()
            labels = target["labels"].cpu().numpy()
            
            mask_overlay = np.zeros((*img_np.shape[:2], 4))
            for m_idx, (m, label) in enumerate(zip(masks, labels)):
                mh, mw = m.shape
                mh, mw = min(mh, h), min(mw, w)
                
                color_hex = colors[label % len(colors)]
                color_rgb = hex_to_rgb(color_hex)
                
                for c in range(3):
                    mask_overlay[:mh, :mw, c] = np.where(m[:mh, :mw], color_rgb[c], mask_overlay[:mh, :mw, c])
                mask_overlay[:mh, :mw, 3] = np.where(m[:mh, :mw], 0.5, mask_overlay[:mh, :mw, 3])
                
            axs[i].imshow(mask_overlay)
            
        if "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"].cpu().numpy()
            labels = target["labels"].cpu().numpy()
            
            for box, label in zip(boxes, labels):
                class_name = label_map.get(label, str(label))
                color_hex = colors[label % len(colors)]
                color_rgb = hex_to_rgb(color_hex)
                
                if box.max() <= 1.0:
                    bcx, bcy, bw, bh = box
                    bx = (bcx - bw / 2) * w
                    by = (bcy - bh / 2) * h
                    bw = bw * w
                    bh = bh * h
                else:
                    bx, by, bxmax, bymax = box
                    bw = bxmax - bx
                    bh = bymax - by
                
                bx = max(1.5, bx)
                by = max(1.5, by)
                bw = min(w - bx - 1.5, bw)
                bh = min(h - by - 1.5, bh)
                
                rect = patches.Rectangle((bx, by), bw, bh, linewidth=2.5, edgecolor=color_rgb, facecolor='none')
                axs[i].add_patch(rect)
                
                axs[i].text(bx + 2, by + 12, class_name, color='white', fontsize=11, weight='bold',
                            bbox=dict(facecolor=color_rgb, alpha=0.7, edgecolor='none', pad=2))
                
        axs[i].axis('off')
        axs[i].set_xlim(0, w)
        axs[i].set_ylim(h, 0)
        
    for j in range(num_to_plot, len(axs)):
        axs[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved visualization to {out_path}")
    plt.close()

@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(config: DictConfig):
    OmegaConf.set_struct(config, False)
    
    motif = getattr(config.data, 'split_motif', 'Unknown')
    train_dsets = config.data.get('train_datasets', [])
    test_dsets = config.data.get('test_datasets', [])
    print(f"Data config selected: Motif {motif} | Train Datasets: {train_dsets}")
    
    label_map = {int(k): v for k, v in config.model.label_map.items()}
    
    base_path = config.data.path
    base_args = type('Args', (), {})()
    
    data_module = MotifDataModule(base_path=base_path, config=config, base_args=base_args)
    data_module.setup("fit")
    data_module.setup("test")
    
    print("\n---------------------------------------------------------")
    print("Fetching a batch from the TRAIN dataloader...")
    train_loader = data_module.train_dataloader()
    batch, targets = next(iter(train_loader))
    batch_tensors = batch.tensors if hasattr(batch, 'tensors') else batch
    
    ds_name_str = train_dsets[0][:30] if train_dsets else "unknown_train"
    
    motif_id = "unknown"
    import sys
    for arg in sys.argv:
        if "motif_" in arg:
            try:
                motif_id = arg.split("motif_")[1].split("_")[0]
                break
            except:
                pass

    os.makedirs("visualizations/dataloader", exist_ok=True)
    out_path = f"visualizations/dataloader/viz_motif_{motif_id}_train_{ds_name_str}.png"
    plot_batch(batch_tensors, targets, label_map, out_path, "TRAIN (With Augmentations)")
    
    print("\n---------------------------------------------------------")
    print("Fetching a batch from the VAL dataloader (Merged Train Datasets)...")
    val_loaders = data_module.val_dataloader()
    if val_loaders:
        batch, targets = next(iter(val_loaders[0]))
        batch_tensors = batch.tensors if hasattr(batch, 'tensors') else batch
        out_path = f"visualizations/dataloader/viz_motif_{motif_id}_val_{ds_name_str}.png"
        plot_batch(batch_tensors, targets, label_map, out_path, "VAL (Merged, No Augmentations)")
        
    print("\n---------------------------------------------------------")
    print("Fetching a batch from the TEST dataloader...")
    test_loaders = data_module.test_dataloader()
    if test_loaders:
        batch, targets = next(iter(test_loaders[0]))
        batch_tensors = batch.tensors if hasattr(batch, 'tensors') else batch
        test_ds_name_str = test_dsets[0][:30] if test_dsets else "unknown_test"
        out_path = f"visualizations/dataloader/viz_motif_{motif_id}_test_{test_ds_name_str}.png"
        plot_batch(batch_tensors, targets, label_map, out_path, "TEST (Zero-shot, No Augmentations)")
    print("---------------------------------------------------------")

if __name__ == '__main__':
    import sys
    if not any(arg.startswith("data=") for arg in sys.argv):
        sys.argv.append("data=coverage_splits/motif_31_preadipocytes-adhered_to_imr90_sibling")
    if not any(arg.startswith("model=") for arg in sys.argv):
        sys.argv.append("model=rfdetr_seg")
    if not any(arg.startswith("model.rfdetr.size=") for arg in sys.argv):
        sys.argv.append("model.rfdetr.size=large")
    if not any(arg.startswith("data.batch_size=") for arg in sys.argv):
        sys.argv.append("data.batch_size=4")
    if not any(arg.startswith("data.eval_batch_size=") or arg.startswith("+data.eval_batch_size=") for arg in sys.argv):
        sys.argv.append("+data.eval_batch_size=4")
    if not any(arg.startswith("data.path=") for arg in sys.argv):
        sys.argv.append("data.path=/mnt/direct-attached/PHASE2")
    
    # To truly see the geometry-altering augmentations (cropping, zooming), we need
    # to explicitly ensure multi_scale and expanded_scales are turned on for the train loader!
    if not any(arg.startswith("model.rfdetr.multi_scale=") for arg in sys.argv):
        sys.argv.append("model.rfdetr.multi_scale=True")
    if not any(arg.startswith("model.rfdetr.expanded_scales=") for arg in sys.argv):
        sys.argv.append("model.rfdetr.expanded_scales=True")
        
    main()
