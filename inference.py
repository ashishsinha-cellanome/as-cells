import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import shutil
from PIL import Image, ImageDraw, ImageFont

from models.rt_detr_lightning_module import RTDETRLightningModule
from data.coco_data_module import COCODataModule
from transformers import RTDetrImageProcessor
from pycocotools.cocoeval import COCOeval
from utils.pairing_utils import pair_gts_dets_bbox

# Helper to move data to CPU
def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return x

# Helper to convert xyxy to xywh
def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(config: DictConfig):
    # Unlock config
    OmegaConf.set_struct(config, False)
    
    # 1. Load Checkpoint
    ckpt_path = config.initialization.load_from_checkpoint
    if not ckpt_path:
        # Fallback to finding the best checkpoint in the run directory if not specified
        # But usually specific inference implies a specific model.
        raise ValueError("Please provide a checkpoint path via 'initialization.load_from_checkpoint=/path/to/ckpt'")
        
    ckpt_path = hydra.utils.to_absolute_path(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    print(f"Loading model from: {ckpt_path}")
    
    # Load model
    model = RTDETRLightningModule.load_from_checkpoint(ckpt_path, config=config)
    model.eval()
    model.cuda()
    
    # 2. Setup Data
    processor = RTDetrImageProcessor.from_pretrained(config.model.rtdetr.pretrained_name_or_path)
    processor.do_normalize = True
    processor.resample = 3
    processor.size = {"height": config.data.model_input_size, "width": config.data.model_input_size}
    model.image_processor = processor # Ensure processor is attached

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
        config=config
    )
    data_module.setup(stage='test')
    test_loader = data_module.test_dataloader()
    
    # Access COCO GT
    coco_gt = data_module.test_dataset.coco
    
    # 3. Inference Loop
    print("Starting Inference...")
    predictions = []
    
    # Prepare Visualization Directory
    viz_dir = os.path.join("predictions", "inference_viz")
    if os.path.exists(viz_dir):
        shutil.rmtree(viz_dir)
    os.makedirs(viz_dir, exist_ok=True)
    
    max_viz_samples = 20
    viz_count = 0
    
    # For TP/FP/FN analysis
    per_image_analysis = []
    label_map = config.model.label_map
    # Ensure keys are integers
    label_map = {int(k): v for k, v in label_map.items()}
    class_ids_of_interest = list(label_map.keys())

    # Get threshold from config or default
    detection_threshold = config.model.detection_threshold
    print(f"Using detection threshold: {detection_threshold}")

    with torch.no_grad():
        for batch in tqdm(test_loader):
            pixel_values = batch["pixel_values"].to(model.device)
            labels = batch["labels"] # List of dicts on CPU usually
            
            # Forward pass
            outputs = model.model(pixel_values=pixel_values, labels=None)
            
            # Post-process
            batch_image_sizes = [to_cpu(x["orig_size"]).numpy().tolist() for x in labels]
            post_processed_outputs = processor.post_process_object_detection(
                outputs,
                threshold=detection_threshold,
                target_sizes=batch_image_sizes
            )
            
            # Process batch
            for i, pred in enumerate(post_processed_outputs):
                image_id = int(labels[i]["image_id"])
                
                # Convert predictions for COCO Eval
                boxes = to_cpu(pred["boxes"])
                scores = to_cpu(pred["scores"])
                pred_labels = to_cpu(pred["labels"])
                
                # --- Visualization (First N samples) ---
                if viz_count < max_viz_samples:
                    # Un-normalize image
                    mean = torch.tensor(processor.image_mean, device=pixel_values.device).view(1, 3, 1, 1)
                    std = torch.tensor(processor.image_std, device=pixel_values.device).view(1, 3, 1, 1)
                    img_tensor = torch.clamp((pixel_values[i:i+1] * std) + mean, 0, 1)
                    img_np = (img_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                    pil_img = Image.fromarray(img_np)
                    
                    # Get GT boxes for this image
                    ann_ids = coco_gt.getAnnIds(imgIds=image_id)
                    anns = coco_gt.loadAnns(ann_ids)
                    gt_boxes_viz = []
                    gt_labels_viz = []
                    for ann in anns:
                        x, y, w, h = ann['bbox']
                        gt_boxes_viz.append([x, y, x+w, y+h])
                        gt_labels_viz.append(ann['category_id'])
                        
                    # Draw GT (Green)
                    model.draw_boxes(pil_img, gt_boxes_viz, gt_labels_viz, color_override=(0, 255, 0), label_prefix="GT: ")
                    # Draw Pred (Red)
                    model.draw_boxes(pil_img, boxes, pred_labels, scores, color_override=(255, 0, 0), label_prefix="Pred: ")
                    
                    pil_img.save(os.path.join(viz_dir, f"img_{image_id}_viz.jpg"))
                    viz_count += 1

                # --- COCO Format Accumulation ---
                xywh_boxes = convert_to_xywh(boxes)
                for b, s, l in zip(xywh_boxes.tolist(), scores.tolist(), pred_labels.tolist()):
                    predictions.append({
                        "image_id": image_id,
                        "category_id": int(l),
                        "bbox": b,
                        "score": s
                    })
                
                # --- TP/FP/FN Analysis per Image ---
                # Get GT for this image
                ann_ids = coco_gt.getAnnIds(imgIds=image_id)
                anns = coco_gt.loadAnns(ann_ids)
                
                # Prepare GT per class
                gt_boxes_all = []
                gt_labels_all = []
                for ann in anns:
                    x, y, w, h = ann['bbox']
                    # Convert xywh to xyxy for pairing
                    gt_boxes_all.append([x, y, x+w, y+h])
                    gt_labels_all.append(ann['category_id'])
                gt_boxes_all = np.array(gt_boxes_all)
                gt_labels_all = np.array(gt_labels_all)
                
                # Prepare Preds per class
                pred_boxes_all = boxes.numpy()
                pred_labels_all = pred_labels.numpy()
                
                image_stats = {"image_id": image_id}
                
                for cls_id in class_ids_of_interest:
                    cls_name = label_map[cls_id]
                    
                    # Filter GT
                    if len(gt_boxes_all) > 0:
                        gt_mask = gt_labels_all == cls_id
                        cls_gt_boxes = gt_boxes_all[gt_mask]
                    else:
                        cls_gt_boxes = np.array([])
                        
                    # Filter Pred
                    if len(pred_boxes_all) > 0:
                        pred_mask = pred_labels_all == cls_id
                        cls_pred_boxes = pred_boxes_all[pred_mask]
                    else:
                        cls_pred_boxes = np.array([])
                    
                    # Handle empty cases
                    if len(cls_gt_boxes) == 0 and len(cls_pred_boxes) == 0:
                        tp, fp, fn = 0, 0, 0
                    elif len(cls_gt_boxes) == 0:
                        tp, fp, fn = 0, len(cls_pred_boxes), 0
                    elif len(cls_pred_boxes) == 0:
                        tp, fp, fn = 0, 0, len(cls_gt_boxes)
                    else:
                        # Pair
                        paired, unpaired_gt, unpaired_pred = pair_gts_dets_bbox(
                            cls_gt_boxes, cls_pred_boxes, min_iou=0.5
                        )
                        tp = len(paired)
                        fp = len(unpaired_pred)
                        fn = len(unpaired_gt)
                        
                    image_stats[f"{cls_name}_TP"] = tp
                    image_stats[f"{cls_name}_FP"] = fp
                    image_stats[f"{cls_name}_FN"] = fn
                
                per_image_analysis.append(image_stats)

    # 4. Compute Standard COCO Metrics
    print("\nComputing COCO Metrics...")
    metrics_summary = {}
    if len(predictions) > 0:
        coco_dt = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Save overall metrics
        metrics_summary["mAP"] = coco_eval.stats[0]
        metrics_summary["mAP_50"] = coco_eval.stats[1]
        metrics_summary["mAP_75"] = coco_eval.stats[2]
        metrics_summary["mAR_1"] = coco_eval.stats[6]
        metrics_summary["mAR_10"] = coco_eval.stats[7]
        metrics_summary["mAR_100"] = coco_eval.stats[8]
        
        # Save per-class metrics
        print("\nPer-class mAP:")
        for i, class_id in enumerate(coco_gt.getCatIds()):
            # COCOeval stores stats in `eval['precision']` which is [TxRxKxAxM]
            # T=10 (IoU thresholds), R=101 (recall thresholds), K=classes, A=4 (areas), M=3 (max dets)
            # We want average over T, R, A=all, M=100
            
            # Precision shape: (10, 101, num_classes, 4, 3)
            # Take mean over IoU (dim 0), Max Dets=100 (dim -1 index 2), Area=all (dim -2 index 0)
            # Then mean over Recall (dim 1)
            
            # Actually, using coco_eval.params.catIds to ensure alignment
            cat_name = label_map.get(class_id, f"class_{class_id}")
            
            # Get precision array for this class
            # Index in eval['precision'] corresponds to index in params.catIds
            k_idx = coco_eval.params.catIds.index(class_id)
            
            # mAP (IoU=0.50:0.95)
            # precision[T, R, K, A, M]
            p = coco_eval.eval['precision'][:, :, k_idx, 0, 2]
            # Remove -1 (invalid)
            p = p[p > -1]
            ap_val = np.mean(p) if len(p) > 0 else 0.0
            
            metrics_summary[f"mAP_{cat_name}"] = ap_val
            print(f"  {cat_name}: {ap_val:.4f}")

    else:
        print("No predictions generated!")

    # 5. Save Results
    output_dir = "predictions/metrics"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save COCO Metrics
    with open(os.path.join(output_dir, "coco_metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)
        
    # Save TP/FP/FN Analysis
    df = pd.DataFrame(per_image_analysis)
    
    # Calculate Aggregate Stats per class
    agg_stats = {}
    for cls_id in class_ids_of_interest:
        cls_name = label_map[cls_id]
        tp = df[f"{cls_name}_TP"].sum()
        fp = df[f"{cls_name}_FP"].sum()
        fn = df[f"{cls_name}_FN"].sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        agg_stats[cls_name] = {
            "Total_TP": int(tp),
            "Total_FP": int(fp),
            "Total_FN": int(fn),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1": float(f1)
        }
    
    # Save Aggregate Stats
    with open(os.path.join(output_dir, "aggregate_analysis.json"), "w") as f:
        json.dump(agg_stats, f, indent=4)
        
    # Save Per-Image Stats to CSV
    csv_path = os.path.join(output_dir, "per_image_analysis.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\nAnalysis Saved to {output_dir}")
    print(f"  - COCO Metrics: coco_metrics.json")
    print(f"  - Aggregate TP/FP/FN: aggregate_analysis.json")
    print(f"  - Per-image CSV: per_image_analysis.csv")
    print(f"Visualizations saved to: {viz_dir}")

if __name__ == "__main__":
    main()