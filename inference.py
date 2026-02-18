#!/usr/bin/env python3
import os
import sys
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import shutil
import cv2
import logging
import time
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Optional, Union

from models.rt_detr_lightning_module import RTDETRLightningModule
from data.coco_data_module import COCODataModule
from transformers import RTDetrImageProcessor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.pairing_utils import pair_gts_dets_bbox
from torch.utils.data import ConcatDataset
from utils.dataset_utils import create_dataset_classes

# Import sliding window utility from the Inference package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Inference'))
from model_utils import get_crop_corners


def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return x


def convert_to_xywh(boxes):
    """Convert xyxy boxes to xywh."""
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def draw_boxes_on_image(
    pil_img: Image.Image,
    boxes: Union[np.ndarray, list],
    labels: Union[np.ndarray, list],
    label_map: Dict[int, str],
    color: tuple = (0, 255, 0),
    scores: Optional[Union[np.ndarray, list]] = None,
    prefix: str = "",
):
    draw = ImageDraw.Draw(pil_img)
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        cls_id = int(labels[idx])
        cls_name = label_map.get(cls_id, str(cls_id))
        text = f"{prefix}{cls_name}"
        if scores is not None:
            text += f" {float(scores[idx]):.2f}"
        draw.text((x1, max(0, y1 - 12)), text, fill=color)


# ---------------------------------------------------------------------------
# MODE: dataset  — uses COCODataModule test loader, no cropping
# ---------------------------------------------------------------------------
def run_dataset_mode(config: DictConfig, model, processor):
    data_module = COCODataModule(
        dataset_path=hydra.utils.to_absolute_path(config.data.path),
        processor=processor,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        model_input_size=config.data.model_input_size,
        min_random_scale=config.data.min_random_scale,
        max_random_scale=config.data.max_random_scale,
        p_noise=config.data.p_noise,
        org_images_in_model_input_size=config.data.org_images_in_model_input_size,
        config=config,
    )
    
    data_module.setup(stage="test")
    breakpoint()
    test_loader = data_module.test_dataloader()
    coco_gt = data_module.test_dataset.dataset_coco.coco

    label_map = {int(k): v for k, v in config.model.label_map.items()}
    class_ids = list(label_map.keys())
    detection_threshold = config.model.detection_threshold
    max_viz = config.inference.max_viz

    output_dir = config.inference.output_dir
    viz_dir = os.path.join(output_dir, "visualizations")
    metrics_dir = os.path.join(output_dir, "metrics")
    if os.path.exists(viz_dir):
        shutil.rmtree(viz_dir)
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    predictions_coco = []
    per_image_analysis = []
    viz_count = 0
    gt_remap = _build_gt_label_remap(coco_gt, config)

    print(f"[dataset mode] Using detection threshold: {detection_threshold}")
    print(f"[dataset mode] Starting inference on {len(data_module.test_dataset)} images ...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Dataset inference"):
            pixel_values = batch["pixel_values"].to(model.device)
            labels_batch = batch["labels"]

            outputs = model.model(pixel_values=pixel_values, labels=None)
            batch_image_sizes = [to_cpu(x["orig_size"]).numpy().tolist() for x in labels_batch]
            post_processed = processor.post_process_object_detection(
                outputs, threshold=detection_threshold, target_sizes=batch_image_sizes,
            )

            for i, pred in enumerate(post_processed):
                image_id = int(labels_batch[i]["image_id"])
                boxes = to_cpu(pred["boxes"])
                scores = to_cpu(pred["scores"])
                pred_labels = to_cpu(pred["labels"])

                # Visualization
                if max_viz == -1 or viz_count < max_viz:
                    mean = torch.tensor(processor.image_mean, device=pixel_values.device).view(1, 3, 1, 1)
                    std = torch.tensor(processor.image_std, device=pixel_values.device).view(1, 3, 1, 1)
                    img_tensor = torch.clamp((pixel_values[i : i + 1] * std) + mean, 0, 1)
                    img_np = (img_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                    pil_img = Image.fromarray(img_np)

                    ann_ids = coco_gt.getAnnIds(imgIds=image_id)
                    anns = coco_gt.loadAnns(ann_ids)
                    gt_boxes_viz = []
                    gt_labels_viz = []
                    for ann in anns:
                        x, y, w, h = ann["bbox"]
                        gt_boxes_viz.append([x, y, x + w, y + h])
                        gt_labels_viz.append(ann["category_id"])

                    draw_boxes_on_image(pil_img, gt_boxes_viz, gt_labels_viz, label_map, color=(0, 255, 0), prefix="GT: ")
                    draw_boxes_on_image(
                        pil_img, boxes.numpy(), pred_labels.numpy(), label_map,
                        color=(255, 0, 0), scores=scores.numpy(), prefix="Pred: ",
                    )
                    pil_img.save(os.path.join(viz_dir, f"img_{image_id}_viz.jpg"))
                    viz_count += 1

                # COCO format accumulation
                xywh_boxes = convert_to_xywh(boxes)
                for b, s, l in zip(xywh_boxes.tolist(), scores.tolist(), pred_labels.tolist()):
                    predictions_coco.append(
                        {"image_id": image_id, "category_id": int(l), "bbox": b, "score": s}
                    )

                # Per-image TP/FP/FN
                ann_ids = coco_gt.getAnnIds(imgIds=image_id)
                anns = coco_gt.loadAnns(ann_ids)
                gt_boxes_all = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns]) if anns else np.zeros((0, 4))
                gt_labels_all = np.array([a["category_id"] for a in anns]) if anns else np.array([])
                pred_boxes_np = boxes.numpy()
                pred_labels_np = pred_labels.numpy()

                image_stats = {"image_id": image_id}
                for cls_id in class_ids:
                    cls_name = label_map[cls_id]
                    tp, fp, fn = _compute_tp_fp_fn(gt_boxes_all, gt_labels_all, pred_boxes_np, pred_labels_np, cls_id, gt_remap=gt_remap)
                    image_stats[f"{cls_name}_TP"] = tp
                    image_stats[f"{cls_name}_FP"] = fp
                    image_stats[f"{cls_name}_FN"] = fn
                per_image_analysis.append(image_stats)

    _save_results(predictions_coco, per_image_analysis, coco_gt, label_map, class_ids, metrics_dir, viz_dir)


# ---------------------------------------------------------------------------
# MODE: folder  — reads images from a folder, sliding-window inference
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# MODE: folder   — sliding window on images in folder (flat or Cellanome dataset)
# ---------------------------------------------------------------------------
def run_folder_mode(config: DictConfig, model, processor):
    folder_path = hydra.utils.to_absolute_path(config.inference.folder_path)
    output_dir = hydra.utils.to_absolute_path(config.inference.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    viz_dir = os.path.join(output_dir, "visualizations")
    metrics_dir = os.path.join(output_dir, "metrics")
    if os.path.exists(viz_dir):
        shutil.rmtree(viz_dir)
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Check for Cellanome dataset structure
    csv_path = os.path.join(folder_path, 'annotation_images_mapping.csv')
    is_cellanome_dataset = os.path.exists(csv_path)

    # Prepare sliding window parameters
    sliding_window_cfg = config.inference.sliding_window
    overlap_x = sliding_window_cfg.overlap_x
    overlap_y = sliding_window_cfg.overlap_y
    nms_threshold = sliding_window_cfg.nms_threshold
    
    # Model input size from config (default to 640 if missing)
    input_size = (640, 640)
    if hasattr(config.model, 'val_image_size'):
         sz = config.model.val_image_size
         input_size = (sz, sz)

    # Sliding window crops generator
    corners_generator = lambda img_h, img_w: get_crop_corners(
        image_width=img_w,
        image_height=img_h,
        input_size=input_size,
        overlap_in_x=overlap_x,
        overlap_in_y=overlap_y
    )

    # Setup COCO GT object strictly for evaluation at the end
    coco_gt = COCO()
    coco_gt.dataset = {"images": [], "annotations": [], "categories": [], "info": []}
    coco_gt.createIndex()
    
    # Categories
    label_map = dict(config.model.label_map) # id -> name
    # Ensure keys are ints
    label_map = {int(k): v for k, v in label_map.items()}
    class_ids = list(label_map.keys())
    categories = [{"id": k, "name": v} for k, v in label_map.items()]
    coco_gt.dataset["categories"] = categories

    dataset_iterator = []
    
    if is_cellanome_dataset:
        print(f"[folder mode] Found annotation_images_mapping.csv. Using Cellanome dataset loader...")
        # Invert label map for create_dataset_classes (name -> id)
        class_name_to_id = {v: k for k, v in label_map.items()}
        
        # Use a large max_side to avoid resizing during loading (we want full res for sliding window)
        train_ds, test_ds = create_dataset_classes(
            dataset_path=folder_path,
            class_names_to_class_ids_map=class_name_to_id,
            max_larger_side=10000, 
            max_smaller_side=10000
        )
        try:
             # Combine both splits to run on everything in the folder
             dataset = ConcatDataset([train_ds, test_ds])
        except Exception:
             dataset = train_ds if len(train_ds) > 0 else test_ds
        
        print(f"[folder mode] Loaded {len(dataset)} images from Cellanome dataset.")
        
        # Helper to yield from dataset
        def cellanome_generator():
            for i in range(len(dataset)):
                sample = dataset[i]
                # sample is a dict: {'name': str, 'image': np.ndarray, 'annotations': DataFrame, ...}
                img_name = sample['name']
                img_rgb = sample['image'] # Already RGB (or grayscale converted)
                if len(img_rgb.shape) == 2:
                    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
                
                # Parse annotations to standard format: list of [x1, y1, x2, y2] (xyxy) and labels
                gt_boxes = []
                gt_labels = []
                
                if 'annotations' in sample and not sample['annotations'].empty:
                    df = sample['annotations']
                    for _, row in df.iterrows():
                        x1, y1, x2, y2 = row['xtl'], row['ytl'], row['xbr'], row['ybr']
                        label = row['label']
                        gt_boxes.append([x1, y1, x2, y2])
                        gt_labels.append(int(label))
                
                yield i + 1, img_name, img_rgb, gt_boxes, gt_labels

        dataset_iterator = cellanome_generator()

    else:
        # Standard flat folder mode
        print(f"[folder mode] Using flat folder structure (no csv found).")
        valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
        image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)])
        
        # Load COCO GT if available
        gt_lookup = {}
        coco_json_path = config.inference.gt_json
        gt_coco_obj = None
        if coco_json_path and os.path.exists(os.path.join(folder_path, coco_json_path)):
             gt_coco_obj = COCO(os.path.join(folder_path, coco_json_path))
             # Ensure 'info' key exists
             if 'info' not in gt_coco_obj.dataset:
                 gt_coco_obj.dataset['info'] = []
             # Map image filename to annotations
             for img_id in gt_coco_obj.getImgIds():
                 img_info = gt_coco_obj.loadImgs(img_id)[0]
                 fname = img_info['file_name']
                 ann_ids = gt_coco_obj.getAnnIds(imgIds=img_id)
                 anns = gt_coco_obj.loadAnns(ann_ids)
                 gt_lookup[fname] = anns
        
        def flat_folder_generator():
            for i, img_name in enumerate(image_files):
                img_path = os.path.join(folder_path, img_name)
                # Load image
                img_bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img_bgr is None:
                    continue

                if len(img_bgr.shape) == 2:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
                elif img_bgr.shape[2] == 1:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
                else:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                gt_boxes = []
                gt_labels = []
                
                if img_name in gt_lookup:
                    anns = gt_lookup[img_name]
                    for ann in anns:
                        bbox = ann['bbox'] # xywh
                        # Convert to xyxy
                        gt_boxes.append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
                        gt_labels.append(ann['category_id'])
                        
                yield i + 1, img_name, img_rgb, gt_boxes, gt_labels

        dataset_iterator = flat_folder_generator()

    # Shared processing loop
    results_coco = [] # For COCOeval
    per_image_metrics = []
    
    # Global accumulator for creating the final COCO GT object for evaluation
    coco_gt_images = []
    coco_gt_annotations = []
    ann_id_counter = 1
    
    gt_remap = _build_gt_label_remap(coco_gt, config) 

    print(f"[folder mode] Starting inference on images...")
    
    for image_id, img_name, img_rgb, gt_boxes_list, gt_labels_list in tqdm(dataset_iterator, desc="Processing"):
        
        h, w = img_rgb.shape[:2]
        
        # Add to COCO GT for final eval
        coco_gt_images.append({
            "id": image_id, 
            "width": w, 
            "height": h, 
            "file_name": img_name
        })
        
        # Add GT annotations
        gt_boxes_xyxy = []
        gt_labels_np = []
        
        for box, lbl in zip(gt_boxes_list, gt_labels_list):
            # box is xyxy from generator
            bx1, by1, bx2, by2 = box
            gt_boxes_xyxy.append([bx1, by1, bx2, by2])
            gt_labels_np.append(lbl)
            
            coco_gt_annotations.append({
                "id": ann_id_counter,
                "image_id": image_id,
                "category_id": int(lbl),
                "bbox": [bx1, by1, bx2 - bx1, by2 - by1], # xywh for COCO
                "area": (bx2 - bx1) * (by2 - by1),
                "iscrowd": 0
            })
            ann_id_counter += 1
            
        gt_boxes_xyxy = np.array(gt_boxes_xyxy) if gt_boxes_xyxy else np.zeros((0, 4))
        gt_labels_np = np.array(gt_labels_np) if gt_labels_np else np.zeros((0,))
        
        # SLIDING WINDOW INFERENCE
        crops = corners_generator(h, w)
        all_pred_boxes = []
        all_pred_scores = []
        all_pred_labels = []
        
        for crop_id, (cx1, cy1, cx2, cy2) in enumerate(crops):
            crop_img = img_rgb[cy1:cy2, cx1:cx2]
            crop_h, crop_w = crop_img.shape[:2]

            inputs = processor(images=crop_img, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.model(pixel_values=inputs['pixel_values'], labels=None)
            
            # Post-process
            target_sizes = torch.tensor([[crop_h, crop_w]], device=model.device)
            results = processor.post_process_object_detection(
                outputs, 
                target_sizes=target_sizes, 
                threshold=0.0 # Get all, filter later
            )[0]
            
            boxes = results["boxes"].cpu().numpy()
            scores = results["scores"].cpu().numpy()
            labels = results["labels"].cpu().numpy()
            
            # Filter by score
            mask = scores >= config.model.detection_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]
            
            # Shift boxes to global coordinates
            if len(boxes) > 0:
                boxes[:, [0, 2]] += cx1
                boxes[:, [1, 3]] += cy1
                
                all_pred_boxes.append(boxes)
                all_pred_scores.append(scores)
                all_pred_labels.append(labels)
                
        # Merge crops with NMS
        if all_pred_boxes:
            final_boxes, final_scores, final_labels = _cross_crop_nms(
                all_pred_boxes, 
                all_pred_scores, 
                all_pred_labels, 
                nms_threshold
            )
        else:
            final_boxes, final_scores, final_labels = np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
            
        # VISUALIZATION
        if config.inference.max_viz == -1 or len(per_image_metrics) < config.inference.max_viz:
            viz_img = img_rgb.copy()
            pil_viz = Image.fromarray(viz_img)

            # Draw GT (Green)
            if len(gt_boxes_xyxy) > 0:
                draw_boxes_on_image(pil_viz, gt_boxes_xyxy, gt_labels_np, label_map, color=(0, 255, 0), prefix="GT: ")
            
            # Draw Pred (Red)
            draw_boxes_on_image(
                pil_viz, final_boxes, final_labels, label_map,
                color=(255, 0, 0), scores=final_scores, prefix="Pred: ",
            )
            stem = os.path.splitext(os.path.basename(img_name))[0]
            pil_viz.save(os.path.join(viz_dir, f"{stem}_viz.jpg"))

        # METRICS (Per Image)
        # Compute TP/FP/FN
        image_stats = {"image_id": image_id, "file_name": img_name}
        for cls_id in class_ids:
            cls_name = label_map[cls_id]
            tp, fp, fn = _compute_tp_fp_fn(gt_boxes_xyxy, gt_labels_np, final_boxes, final_labels, cls_id, gt_remap=gt_remap)
            image_stats[f"{cls_name}_TP"] = tp
            image_stats[f"{cls_name}_FP"] = fp
            image_stats[f"{cls_name}_FN"] = fn
        per_image_metrics.append(image_stats)
        
        # Add to COCO results
        for box, sc, lbl in zip(final_boxes, final_scores, final_labels):
            results_coco.append({
                "image_id": image_id,
                "category_id": int(lbl),
                "bbox": [float(box[0]), float(box[1]), float(box[2]-box[0]), float(box[3]-box[1])],
                "score": float(sc)
            })

    # Save Per-Image CSV
    pd.DataFrame(per_image_metrics).to_csv(os.path.join(output_dir, "per_image_stats.csv"), index=False)
    
    # Run COCO Eval
    if results_coco and coco_gt_annotations:
        print("[folder mode] Running COCO evaluation...")
        coco_gt.dataset["images"] = coco_gt_images
        coco_gt.dataset["annotations"] = coco_gt_annotations
        coco_gt.createIndex()
        
        coco_dt = coco_gt.loadRes(results_coco)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Save metrics
        stats = {
            "mAP_50_95": coco_eval.stats[0],
            "mAP_50": coco_eval.stats[1],
            "mAP_75": coco_eval.stats[2],
        }
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(stats, f, indent=2)
            
    print(f"[folder mode] Done. Output saved to {output_dir}")


# ---------------------------------------------------------------------------
# MODE: single  — one image, sliding-window inference
# ---------------------------------------------------------------------------
def run_single_mode(config: DictConfig, model, processor):
    image_path = hydra.utils.to_absolute_path(config.inference.image_path)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    label_map = {int(k): v for k, v in config.model.label_map.items()}
    class_ids = list(label_map.keys())
    detection_threshold = config.model.detection_threshold
    model_input_size = config.data.model_input_size
    sw_cfg = config.inference.sliding_window

    output_dir = config.inference.output_dir
    viz_dir = os.path.join(output_dir, "visualizations")
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Try to find COCO GT in the image's parent folder
    parent_folder = os.path.dirname(image_path)
    gt_json_name = config.inference.gt_json
    gt_json_path = os.path.join(parent_folder, gt_json_name)
    has_gt = os.path.isfile(gt_json_path)
    coco_gt = COCO(gt_json_path) if has_gt else None
    if has_gt:
        print(f"[single mode] Loaded GT from {gt_json_path}")

    img_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    if len(img_bgr.shape) == 2:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    elif img_bgr.shape[2] == 1:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    H, W = img_rgb.shape[:2]
    crop_size = (model_input_size, model_input_size)
    crop_corners = get_crop_corners(
        image_width=W, image_height=H,
        overlap_in_x=sw_cfg.overlap_x, overlap_in_y=sw_cfg.overlap_y,
        input_size=crop_size,
    )
    print(f"[single mode] Image {os.path.basename(image_path)} ({W}x{H}): {len(crop_corners)} crops")

    all_boxes, all_scores, all_labels = [], [], []
    with torch.no_grad():
        for corners in tqdm(crop_corners, desc="Sliding window"):
            x1c, y1c, x2c, y2c = corners
            crop_img = img_rgb[y1c:y2c, x1c:x2c]
            crop_h, crop_w = crop_img.shape[:2]

            inputs = processor(images=crop_img, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(model.device)
            outputs = model.model(pixel_values=pixel_values, labels=None)
            post = processor.post_process_object_detection(
                outputs, threshold=detection_threshold, target_sizes=[(crop_h, crop_w)]
            )
            pred = post[0]
            boxes_crop = to_cpu(pred["boxes"]).numpy()
            scores_crop = to_cpu(pred["scores"]).numpy()
            labels_crop = to_cpu(pred["labels"]).numpy()

            if len(scores_crop) == 0:
                continue

            boundary_mask = (
                (boxes_crop[:, 0] < 4)
                | (boxes_crop[:, 1] < 4)
                | (boxes_crop[:, 2] > crop_w - 4)
                | (boxes_crop[:, 3] > crop_h - 4)
            )
            scores_crop[boundary_mask] = detection_threshold

            boxes_crop[:, [0, 2]] += x1c
            boxes_crop[:, [1, 3]] += y1c

            all_boxes.append(boxes_crop)
            all_scores.append(scores_crop)
            all_labels.append(labels_crop)

    if len(all_boxes) > 0:
        merged_boxes, merged_scores, merged_labels = _cross_crop_nms(
            all_boxes, all_scores, all_labels, sw_cfg.nms_threshold
        )
    else:
        merged_boxes = np.zeros((0, 4))
        merged_scores = np.zeros((0,))
        merged_labels = np.zeros((0,), dtype=int)

    print(f"[single mode] {len(merged_boxes)} detections after NMS")

    # Save detections JSON
    detections_out = {
        "boxes": merged_boxes.tolist(),
        "scores": merged_scores.tolist(),
        "labels": [int(l) for l in merged_labels],
    }
    det_path = os.path.join(metrics_dir, "detections.json")
    with open(det_path, "w") as f:
        json.dump(detections_out, f, indent=2)

    # Visualization
    pil_img = Image.fromarray(img_rgb)

    # If GT is available, compute metrics for this image
    gt_remap = _build_gt_label_remap(coco_gt, config)
    if coco_gt is not None:
        file_name = os.path.basename(image_path)
        # Find image_id from GT by matching filename
        image_id = None
        for img_info in coco_gt.imgs.values():
            if img_info["file_name"] == file_name:
                image_id = img_info["id"]
                break

        if image_id is not None:
            ann_ids = coco_gt.getAnnIds(imgIds=image_id)
            anns = coco_gt.loadAnns(ann_ids)
            gt_boxes_viz = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns]
            gt_labels_viz = [a["category_id"] for a in anns]
            draw_boxes_on_image(pil_img, gt_boxes_viz, gt_labels_viz, label_map, color=(0, 255, 0), prefix="GT: ")

            gt_boxes_all = np.array(gt_boxes_viz) if gt_boxes_viz else np.zeros((0, 4))
            gt_labels_all = np.array(gt_labels_viz) if gt_labels_viz else np.array([])

            image_stats = {"image_id": image_id, "file_name": file_name}
            for cls_id in class_ids:
                cls_name = label_map[cls_id]
                tp, fp, fn = _compute_tp_fp_fn(gt_boxes_all, gt_labels_all, merged_boxes, merged_labels, cls_id, gt_remap=gt_remap)
                image_stats[f"{cls_name}_TP"] = tp
                image_stats[f"{cls_name}_FP"] = fp
                image_stats[f"{cls_name}_FN"] = fn
            
            df = pd.DataFrame([image_stats])
            df.to_csv(os.path.join(metrics_dir, "per_image_analysis.csv"), index=False)

            # Build COCO predictions and run eval
            predictions_coco = []
            for b, s, l in zip(merged_boxes, merged_scores, merged_labels):
                x1, y1, x2, y2 = b
                predictions_coco.append({
                    "image_id": image_id,
                    "category_id": int(l),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(s),
                })
            if predictions_coco:
                coco_dt = coco_gt.loadRes(predictions_coco)
                coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
                coco_eval.params.imgIds = [image_id]
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
            print(f"[single mode] Metrics for {file_name}: {image_stats}")
        else:
            logging.warning(f"Image {file_name} not found in GT JSON, skipping evaluation")

    draw_boxes_on_image(
        pil_img, merged_boxes, merged_labels, label_map,
        color=(255, 0, 0), scores=merged_scores, prefix="Pred: ",
    )
    stem = os.path.splitext(os.path.basename(image_path))[0]
    viz_path = os.path.join(viz_dir, f"{stem}_viz.jpg")
    pil_img.save(viz_path)
    print(f"[single mode] Visualization saved to {viz_path}")
    print(f"[single mode] Detections saved to {det_path}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _build_gt_label_remap(coco_gt, config) -> Dict[int, int]:
    """Build a dict mapping GT category IDs to model class IDs, applying class_remapping.

    Only applies remapping when config.remap_labels is True.
    For example, if GT has category 'cell-adhered' (id=4) and config says
    class_remapping: {'cell-adhered': 'cell'}, and model label_map says {0: 'cell'},
    then the remap will be {4: 0} — GT annotations with cat_id=4 will be treated as
    class 0 during metric computation.
    """
    if coco_gt is None:
        return {}

    # Only apply remapping if explicitly enabled
    if not getattr(config, 'remap_labels', False):
        return {}

    # Check if class_remapping is configured
    remapping_rules = {}
    if hasattr(config, 'data') and config.data and 'class_remapping' in config.data:
        class_remap = config.data.class_remapping
        if class_remap is not None:
            remapping_rules = dict(class_remap)

    if not remapping_rules:
        return {}

    model_label_map = {int(k): v for k, v in config.model.label_map.items()}
    name_to_model_id = {v: k for k, v in model_label_map.items()}

    remap = {}
    for cat_id, cat_info in coco_gt.cats.items():
        src_name = cat_info['name']
        effective_name = remapping_rules.get(src_name, src_name)
        if effective_name in name_to_model_id:
            remap[cat_id] = name_to_model_id[effective_name]

    if remap:
        print(f"[annotation filter] GT label remap: {remap}")
    return remap


def _remap_gt_labels(gt_labels: np.ndarray, gt_remap: Dict[int, int]) -> np.ndarray:
    """Apply the GT label remap to an array of category IDs."""
    if not gt_remap or len(gt_labels) == 0:
        return gt_labels
    remapped = gt_labels.copy()
    for src_id, tgt_id in gt_remap.items():
        remapped[gt_labels == src_id] = tgt_id
    return remapped


def _compute_tp_fp_fn(gt_boxes, gt_labels, pred_boxes, pred_labels, cls_id, gt_remap=None):
    # Remap GT labels if annotation filter is configured
    if gt_remap:
        gt_labels = _remap_gt_labels(gt_labels, gt_remap)

    if len(gt_boxes) > 0:
        gt_mask = gt_labels == cls_id
        cls_gt = gt_boxes[gt_mask]
    else:
        cls_gt = np.zeros((0, 4))

    if len(pred_boxes) > 0:
        pred_mask = pred_labels == cls_id
        cls_pred = pred_boxes[pred_mask]
    else:
        cls_pred = np.zeros((0, 4))

    if len(cls_gt) == 0 and len(cls_pred) == 0:
        return 0, 0, 0
    elif len(cls_gt) == 0:
        return 0, len(cls_pred), 0
    elif len(cls_pred) == 0:
        return 0, 0, len(cls_gt)
    else:
        paired, unpaired_gt, unpaired_pred = pair_gts_dets_bbox(cls_gt, cls_pred, min_iou=0.5)
        return len(paired), len(unpaired_pred), len(unpaired_gt)


def _cross_crop_nms(
    all_boxes: List[np.ndarray],
    all_scores: List[np.ndarray],
    all_labels: List[np.ndarray],
    nms_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class NMS across all crop detections using torchvision."""
    from torchvision.ops import nms as tv_nms

    boxes_cat = np.concatenate(all_boxes, axis=0)
    scores_cat = np.concatenate(all_scores, axis=0)
    labels_cat = np.concatenate(all_labels, axis=0)

    if len(boxes_cat) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)

    keep_indices = []
    for cls_id in np.unique(labels_cat):
        cls_mask = labels_cat == cls_id
        cls_boxes = torch.tensor(boxes_cat[cls_mask], dtype=torch.float32)
        cls_scores = torch.tensor(scores_cat[cls_mask], dtype=torch.float32)
        keep = tv_nms(cls_boxes, cls_scores, nms_threshold)
        original_indices = np.where(cls_mask)[0]
        keep_indices.extend(original_indices[keep.numpy()].tolist())

    keep_indices = sorted(keep_indices)
    return boxes_cat[keep_indices], scores_cat[keep_indices], labels_cat[keep_indices]


def _save_results(predictions_coco, per_image_analysis, coco_gt, label_map, class_ids, metrics_dir, viz_dir):
    """Save COCO metrics, per-image CSV, and aggregate JSON (shared by dataset and folder modes)."""
    metrics_summary = {}

    # Ensure coco_gt has 'info' key to avoid KeyError
    if coco_gt is not None and 'info' not in coco_gt.dataset:
        coco_gt.dataset['info'] = []

    if len(predictions_coco) > 0 and coco_gt is not None:
        print("\nComputing COCO Metrics...")
        coco_dt = coco_gt.loadRes(predictions_coco)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        metrics_summary["mAP"] = coco_eval.stats[0]
        metrics_summary["mAP_50"] = coco_eval.stats[1]
        metrics_summary["mAP_75"] = coco_eval.stats[2]
        metrics_summary["mAR_1"] = coco_eval.stats[6]
        metrics_summary["mAR_10"] = coco_eval.stats[7]
        metrics_summary["mAR_100"] = coco_eval.stats[8]

        print("\nPer-class mAP:")
        for class_id in coco_gt.getCatIds():
            cat_name = label_map.get(class_id, f"class_{class_id}")
            if class_id in coco_eval.params.catIds:
                k_idx = coco_eval.params.catIds.index(class_id)
                p = coco_eval.eval["precision"][:, :, k_idx, 0, 2]
                p = p[p > -1]
                ap_val = float(np.mean(p)) if len(p) > 0 else 0.0
            else:
                ap_val = 0.0
            metrics_summary[f"mAP_{cat_name}"] = ap_val
            print(f"  {cat_name}: {ap_val:.4f}")
    elif coco_gt is None:
        print("\nNo GT file — skipping COCO evaluation.")
    else:
        print("\nNo predictions generated!")

    with open(os.path.join(metrics_dir, "coco_metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)

    # Per-image CSV
    df = pd.DataFrame(per_image_analysis)
    df.to_csv(os.path.join(metrics_dir, "per_image_analysis.csv"), index=False)

    # Aggregate stats
    agg_stats = {}
    for cls_id in class_ids:
        cls_name = label_map[cls_id]
        tp_col, fp_col, fn_col = f"{cls_name}_TP", f"{cls_name}_FP", f"{cls_name}_FN"
        if tp_col in df.columns:
            tp = int(df[tp_col].sum())
            fp = int(df[fp_col].sum())
            fn = int(df[fn_col].sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            agg_stats[cls_name] = {
                "Total_TP": tp, "Total_FP": fp, "Total_FN": fn,
                "Precision": float(precision), "Recall": float(recall), "F1": float(f1),
            }

    with open(os.path.join(metrics_dir, "aggregate_metrics.json"), "w") as f:
        json.dump(agg_stats, f, indent=4)

    print(f"\nResults saved to {metrics_dir}")
    print(f"  - coco_metrics.json")
    print(f"  - per_image_analysis.csv")
    print(f"  - aggregate_metrics.json")
    print(f"Visualizations saved to {viz_dir}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(config: DictConfig):
    OmegaConf.set_struct(config, False)

    # Load checkpoint
    ckpt_path = config.initialization.load_from_checkpoint
    if not ckpt_path:
        raise ValueError("Provide a checkpoint path via 'initialization.load_from_checkpoint=/path/to/ckpt'")

    ckpt_path = hydra.utils.to_absolute_path(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    print(f"Loading checkpoint from: {ckpt_path}")

    # First load checkpoint to get the saved config
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    # Extract config from checkpoint if available
    if 'hyper_parameters' in checkpoint:
        ckpt_config = checkpoint['hyper_parameters'].get('config')
        if ckpt_config is not None:
            print("Using config from checkpoint (override any CLI args)")
            # Merge checkpoint config with current config (checkpoint takes priority)
            config = OmegaConf.merge(config, ckpt_config)

    # Build model architecture based on the (possibly merged) config
    from models.custom_rt_detr_with_dinov2_backbone import (
        RTDetrV2ForObjectDetectionWithCustomBackbone,
        RTDetrV2ConfigWithCustomBackBone
    )
    from transformers import RTDetrV2ForObjectDetection

    model_checkpoint_path = config.model.rtdetr.pretrained_name_or_path
    print(f"Loading base model from: {model_checkpoint_path}")

    # Determine model class based on config
    if hasattr(config.model, 'backbone') and hasattr(config.model.backbone, 'type'):
        model_cls = RTDetrV2ForObjectDetectionWithCustomBackbone
    else:
        model_cls = RTDetrV2ForObjectDetection

    # First, determine the number of classes from the checkpoint
    # by inspecting the shape of classification head weights
    state_dict = checkpoint.get('state_dict', checkpoint)
    num_classes = None
    for key in state_dict.keys():
        if 'class_embed.0.weight' in key or 'enc_score_head.weight' in key:
            num_classes = state_dict[key].shape[0]
            print(f"Detected {num_classes} classes from checkpoint")
            break

    # Load config and modify num_labels before building model
    model_config = RTDetrV2ConfigWithCustomBackBone.from_pretrained(model_checkpoint_path)
    if num_classes is not None:
        print(f"Setting model num_labels to {num_classes}")
        model_config.num_labels = num_classes

    # Build model with correct number of classes
    model = model_cls.from_pretrained(model_checkpoint_path, config=model_config)
    model.eval()

    # Setup processor
    processor = RTDetrImageProcessor.from_pretrained(config.model.rtdetr.pretrained_name_or_path)
    processor.do_normalize = True
    processor.resample = 3
    processor.size = {"height": config.data.model_input_size, "width": config.data.model_input_size}

    # Create LightningModule and load checkpoint weights
    lightning_module = RTDETRLightningModule(
        model=model,
        image_processor=processor,
        config=config
    )

    # Load state dict from checkpoint
    if 'state_dict' in checkpoint:
        lightning_module.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        lightning_module.load_state_dict(checkpoint, strict=False)

    lightning_module.eval()
    lightning_module.cuda()

    mode = config.inference.mode
    print(f"\n{'='*60}")
    print(f"Inference mode: {mode}")
    print(f"{'='*60}\n")

    if mode == "dataset":
        run_dataset_mode(config, lightning_module, processor)
    elif mode == "folder":
        run_folder_mode(config, lightning_module, processor)
    elif mode == "single":
        run_single_mode(config, lightning_module, processor)
    else:
        raise ValueError(f"Unknown inference mode: '{mode}'. Use 'dataset', 'folder', or 'single'.")


if __name__ == "__main__":
    main()
