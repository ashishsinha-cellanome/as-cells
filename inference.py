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
    test_loader = data_module.test_dataloader()
    coco_gt = data_module.test_dataset.coco

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
def run_folder_mode(config: DictConfig, model, processor):
    folder_path = hydra.utils.to_absolute_path(config.inference.folder_path)
    gt_json_name = config.inference.gt_json
    gt_json_path = os.path.join(folder_path, gt_json_name)

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    has_gt = os.path.isfile(gt_json_path)
    coco_gt = COCO(gt_json_path) if has_gt else None
    if has_gt:
        print(f"[folder mode] Loaded GT from {gt_json_path}")
    else:
        print(f"[folder mode] No GT file found at {gt_json_path}, skipping evaluation")

    label_map = {int(k): v for k, v in config.model.label_map.items()}
    class_ids = list(label_map.keys())
    detection_threshold = config.model.detection_threshold
    model_input_size = config.data.model_input_size
    sw_cfg = config.inference.sliding_window
    max_viz = config.inference.max_viz

    output_dir = config.inference.output_dir
    viz_dir = os.path.join(output_dir, "visualizations")
    metrics_dir = os.path.join(output_dir, "metrics")
    if os.path.exists(viz_dir):
        shutil.rmtree(viz_dir)
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Collect image files
    if coco_gt is not None:
        img_infos = list(coco_gt.imgs.values())
    else:
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        img_infos = []
        for fname in sorted(os.listdir(folder_path)):
            if os.path.splitext(fname)[1].lower() in exts:
                img_infos.append({"id": len(img_infos) + 1, "file_name": fname})

    predictions_coco = []
    per_image_analysis = []
    viz_count = 0
    gt_remap = _build_gt_label_remap(coco_gt, config)

    print(f"[folder mode] Processing {len(img_infos)} images with sliding window (crop={model_input_size}x{model_input_size}) ...")

    with torch.no_grad():
        for img_info in tqdm(img_infos, desc="Folder inference"):
            image_id = img_info["id"]
            file_name = img_info["file_name"]
            img_path = os.path.join(folder_path, file_name)
            if not os.path.isfile(img_path):
                logging.warning(f"Image file not found: {img_path}, skipping")
                continue

            img_bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img_bgr is None:
                logging.warning(f"Could not read image: {img_path}, skipping")
                continue

            # Convert to 8-bit grayscale → RGB (images are already 8-bit per user)
            if len(img_bgr.shape) == 2:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
            elif img_bgr.shape[2] == 1:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            H, W = img_rgb.shape[:2]

            # Sliding-window crop corners
            crop_size = (model_input_size, model_input_size)
            crop_corners = get_crop_corners(
                image_width=W,
                image_height=H,
                overlap_in_x=sw_cfg.overlap_x,
                overlap_in_y=sw_cfg.overlap_y,
                input_size=crop_size,
            )
            logging.info(f"Image {file_name} ({W}x{H}): {len(crop_corners)} crops")

            # Run model on each crop, accumulate detections in image coords
            all_boxes, all_scores, all_labels = [], [], []
            confidence_threshold = detection_threshold

            for corners in crop_corners:
                x1c, y1c, x2c, y2c = corners
                crop_img = img_rgb[y1c:y2c, x1c:x2c]
                crop_h, crop_w = crop_img.shape[:2]

                inputs = processor(images=crop_img, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(model.device)
                outputs = model.model(pixel_values=pixel_values, labels=None)
                post = processor.post_process_object_detection(
                    outputs, threshold=confidence_threshold, target_sizes=[(crop_h, crop_w)]
                )
                pred = post[0]
                boxes_crop = to_cpu(pred["boxes"]).numpy()
                scores_crop = to_cpu(pred["scores"]).numpy()
                labels_crop = to_cpu(pred["labels"]).numpy()

                if len(scores_crop) == 0:
                    continue

                # Reduce scores for boundary detections (same logic as detect_by_cropping)
                boundary_mask = (
                    (boxes_crop[:, 0] < 4)
                    | (boxes_crop[:, 1] < 4)
                    | (boxes_crop[:, 2] > crop_w - 4)
                    | (boxes_crop[:, 3] > crop_h - 4)
                )
                scores_crop[boundary_mask] = confidence_threshold

                # Shift to image coords
                boxes_crop[:, [0, 2]] += x1c
                boxes_crop[:, [1, 3]] += y1c

                all_boxes.append(boxes_crop)
                all_scores.append(scores_crop)
                all_labels.append(labels_crop)

            # Merge with cross-crop NMS
            if len(all_boxes) > 0:
                merged_boxes, merged_scores, merged_labels = _cross_crop_nms(
                    all_boxes, all_scores, all_labels, sw_cfg.nms_threshold
                )
            else:
                merged_boxes = np.zeros((0, 4))
                merged_scores = np.zeros((0,))
                merged_labels = np.zeros((0,), dtype=int)

            # COCO predictions
            for b, s, l in zip(merged_boxes, merged_scores, merged_labels):
                x1, y1, x2, y2 = b
                predictions_coco.append({
                    "image_id": image_id,
                    "category_id": int(l),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(s),
                })

            # Per-image analysis (if GT available)
            image_stats = {"image_id": image_id, "file_name": file_name}
            if coco_gt is not None:
                ann_ids = coco_gt.getAnnIds(imgIds=image_id)
                anns = coco_gt.loadAnns(ann_ids)
                gt_boxes_all = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns]) if anns else np.zeros((0, 4))
                gt_labels_all = np.array([a["category_id"] for a in anns]) if anns else np.array([])

                for cls_id in class_ids:
                    cls_name = label_map[cls_id]
                    tp, fp, fn = _compute_tp_fp_fn(gt_boxes_all, gt_labels_all, merged_boxes, merged_labels, cls_id, gt_remap=gt_remap)
                    image_stats[f"{cls_name}_TP"] = tp
                    image_stats[f"{cls_name}_FP"] = fp
                    image_stats[f"{cls_name}_FN"] = fn
            per_image_analysis.append(image_stats)

            # Visualization
            if max_viz == -1 or viz_count < max_viz:
                pil_img = Image.fromarray(img_rgb)
                if coco_gt is not None:
                    ann_ids = coco_gt.getAnnIds(imgIds=image_id)
                    anns = coco_gt.loadAnns(ann_ids)
                    gt_boxes_viz = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns]
                    gt_labels_viz = [a["category_id"] for a in anns]
                    draw_boxes_on_image(pil_img, gt_boxes_viz, gt_labels_viz, label_map, color=(0, 255, 0), prefix="GT: ")
                draw_boxes_on_image(
                    pil_img, merged_boxes, merged_labels, label_map,
                    color=(255, 0, 0), scores=merged_scores, prefix="Pred: ",
                )
                stem = os.path.splitext(file_name)[0]
                pil_img.save(os.path.join(viz_dir, f"{stem}_viz.jpg"))
                viz_count += 1

    _save_results(predictions_coco, per_image_analysis, coco_gt, label_map, class_ids, metrics_dir, viz_dir)


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

    For example, if GT has category 'cell-adhered' (id=4) and config says
    class_remapping: {'cell-adhered': 'cell'}, and model label_map says {0: 'cell'},
    then the remap will be {4: 0} — GT annotations with cat_id=4 will be treated as
    class 0 during metric computation.
    """
    if coco_gt is None:
        return {}

    model_label_map = {int(k): v for k, v in config.model.label_map.items()}
    name_to_model_id = {v: k for k, v in model_label_map.items()}

    remapping_rules = {}
    if hasattr(config, 'data') and config.data and 'class_remapping' in config.data:
        remapping_rules = dict(config.data.class_remapping)

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

    print(f"Loading model from: {ckpt_path}")
    model = RTDETRLightningModule.load_from_checkpoint(ckpt_path, config=config)
    model.eval()
    model.cuda()

    # Setup processor
    processor = RTDetrImageProcessor.from_pretrained(config.model.rtdetr.pretrained_name_or_path)
    processor.do_normalize = True
    processor.resample = 3
    processor.size = {"height": config.data.model_input_size, "width": config.data.model_input_size}
    model.image_processor = processor

    mode = config.inference.mode
    print(f"\n{'='*60}")
    print(f"Inference mode: {mode}")
    print(f"{'='*60}\n")

    if mode == "dataset":
        run_dataset_mode(config, model, processor)
    elif mode == "folder":
        run_folder_mode(config, model, processor)
    elif mode == "single":
        run_single_mode(config, model, processor)
    else:
        raise ValueError(f"Unknown inference mode: '{mode}'. Use 'dataset', 'folder', or 'single'.")


if __name__ == "__main__":
    main()