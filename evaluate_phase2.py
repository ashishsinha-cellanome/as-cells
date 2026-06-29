import os
import sys
import builtins

# Force unbuffered output
def print(*args, **kwargs):
    kwargs['flush'] = True
    builtins.print(*args, **kwargs)

import json
import torch
import hydra
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.pairing_utils import pair_gts_dets_bbox

def load_model(model_key, model_cfg):
    model_type = model_cfg.type
    checkpoint = model_cfg.weights
    
    # Check if checkpoint exists before attempting to load
    if model_type != "yolo" or (model_type == "yolo" and model_key != "yolov5"):
        if checkpoint and not os.path.exists(checkpoint):
            print(f"[WARN] Checkpoint file not found: {checkpoint}. Skipping model {model_key}.")
            return None

    if model_type == "yolo":
        if "yolov5" in model_key.lower():
            import sys
            import copy
            
            original_sys_path = copy.copy(sys.path)
            
            while '' in sys.path: sys.path.remove('')
            while os.getcwd() in sys.path: sys.path.remove(os.getcwd())
            if os.path.abspath(os.getcwd()) in sys.path: sys.path.remove(os.path.abspath(os.getcwd()))
            
            # Remove any loaded 'utils' modules from as-cells to prevent conflicts
            for k in list(sys.modules.keys()):
                if k == 'utils' or k.startswith('utils.'):
                    del sys.modules[k]
            
            repo_path = os.path.join(os.getcwd(), "models", "yolov5")
            sys.path.insert(0, repo_path)
            
            # Cache all current modules
            saved_modules = {k: v for k, v in sys.modules.items()}
            
            try:
                model = torch.hub.load(repo_path, 'custom', path=checkpoint, source='local')
            except Exception as e:
                print(f"Failed to load YOLOv5 via torch.hub: {e}")
                raise e
            finally:
                sys.path = original_sys_path
                # Restore modules exactly as they were
                for k in list(sys.modules.keys()):
                    if k not in saved_modules and k.startswith(("models", "utils", "ultralytics")):
                        del sys.modules[k]
                for k, v in saved_modules.items():
                    sys.modules[k] = v
            return model
        else:
            from ultralytics import YOLO
            try:
                model = YOLO(checkpoint)
            except Exception as e:
                print(f"Failed to load YOLO model from {checkpoint}: {e}")
                raise e
            return model
        
    elif model_type in ["rf_detr", "rf_detr_lightning"]:
        if model_type == "rf_detr":
            try:
                import rfdetr
                if "base" in model_key.lower():
                    model = rfdetr.RFDETRBase(pretrain_weights=checkpoint, group_detr=1, num_classes=4)
                elif "medium" in model_key.lower() and "seg" not in model_key.lower():
                    model = rfdetr.RFDETRMedium(pretrain_weights=checkpoint, group_detr=1, num_classes=4, patch_size=12)
                elif "seg" in model_key.lower() and "medium" in model_key.lower():
                    model = rfdetr.RFDETRSegMedium(pretrain_weights=checkpoint, group_detr=1, num_classes=4)
                else:
                    model = rfdetr.RFDETRSegLarge(pretrain_weights=checkpoint, group_detr=1, num_classes=4)
            except ImportError:
                raise NotImplementedError("RF-DETR loading stub - 'rfdetr' module not found in environment.")
            return model
        else:
            from models.rf_detr_lightning_module import RFDETRLightningModule
            from omegaconf import OmegaConf
            try:
                ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
                # Some checkpoints might be raw state_dicts, some might be PL checkpionts
                if "hyper_parameters" in ckpt:
                    config = ckpt["hyper_parameters"].get("config")
                else:
                    # Load default config
                    config = OmegaConf.load("configs/config.yaml")
                
                model = RFDETRLightningModule(config=config)
                if "state_dict" in ckpt:
                    model.load_state_dict(ckpt["state_dict"], strict=False)
                elif "ema_state_dict" in ckpt:
                    model.model.load_state_dict(ckpt["ema_state_dict"], strict=False)
                else:
                    model.load_state_dict(ckpt, strict=False)
            except Exception as e:
                raise NotImplementedError(f"RF-DETR Lightning loading failed: {e}")
            return model
        
    elif model_type == "rt_detr":
        try:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            config = ckpt['hyper_parameters']['config']
            from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
            from models.custom_rt_detr_with_dinov2_backbone import RTDetrV2ForObjectDetectionWithCustomBackbone, RTDetrV2ConfigWithCustomBackBone
            from models.rt_detr_lightning_module import RTDETRLightningModule
            
            model_checkpoint_path = config.model.rtdetr.pretrained_name_or_path
            
            # Reconstruct model based on config
            if "v2" in model_key or "v2" in config.model.rtdetr.model_name:
                model_config = RTDetrV2ConfigWithCustomBackBone.from_pretrained(model_checkpoint_path)
                model_config.num_labels = 4
                base_model = RTDetrV2ForObjectDetectionWithCustomBackbone.from_pretrained(model_checkpoint_path, config=model_config, ignore_mismatched_sizes=True)
            else:
                base_model = RTDetrForObjectDetection.from_pretrained(model_checkpoint_path, num_labels=4, ignore_mismatched_sizes=True)
                
            model = RTDETRLightningModule(model=base_model, image_processor=RTDetrImageProcessor.from_pretrained(model_checkpoint_path), config=config)
            # Find state_dict key 
            sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt.get("ema_state_dict", ckpt)
            
            mapped_sd = {}
            for k, v in sd.items():
                new_k = k
                # ONLY map encoder/decoder transformers v5.4 renaming changes
                if 'model.denoising_class_embed' in new_k:
                    new_k = new_k.replace('model.denoising_class_embed', 'model.model.denoising_class_embed')
                if 'model.model.encoder.encoder.' in new_k:
                    new_k = new_k.replace('model.model.encoder.encoder.', 'model.model.encoder.aifi.')
                if 'self_attn.out_proj' in new_k:
                    new_k = new_k.replace('self_attn.out_proj', 'self_attn.o_proj')
                if '.fc1' in new_k and 'mlp' not in new_k:
                    new_k = new_k.replace('.fc1', '.mlp.fc1')
                if '.fc2' in new_k and 'mlp' not in new_k:
                    new_k = new_k.replace('.fc2', '.mlp.fc2')
                    
                # Do NOT overwrite existing good keys with duplicate bad keys
                if new_k not in mapped_sd:
                    mapped_sd[new_k] = v
                
            model.load_state_dict(mapped_sd, strict=False)
        except Exception as e:
            raise NotImplementedError(f"RT-DETR inference stub: {e}")
        return model
        
    elif model_type in ["deim", "mask2former"]:
        try:
            import wandb
            api = wandb.Api()
            runs = []
            for proj in ["cell-detection", "rt-detr-cell-detection", "cellanome"]:
                try:
                    runs.extend(list(api.runs(f"sinashish/{proj}")))
                except Exception:
                    pass
                    
            target_config = None
            for run in runs:
                cfg = run.config
                ckpt_dir = ""
                if "checkpointing" in cfg and isinstance(cfg["checkpointing"], dict):
                    ckpt_dir = cfg["checkpointing"].get("save_dir", "")
                elif "data" in cfg and isinstance(cfg["data"], dict):
                    ckpt_dir = cfg["data"].get("save_dir", "")
                
                if ckpt_dir and ckpt_dir.split("/")[-1] in checkpoint:
                    target_config = OmegaConf.create(cfg)
                    break
                    
            if target_config is None:
                raise ValueError(f"Could not find WandB config for checkpoint {checkpoint}")
                
            if model_type == "deim":
                from models.deim_v2_lightning_module import DeimV2LightningModule
                model = DeimV2LightningModule.load_from_checkpoint(checkpoint, strict=False, map_location="cpu", config=target_config)
            else:
                from models.mask2former_lightning_module import Mask2FormerLightningModule
                model = Mask2FormerLightningModule.load_from_checkpoint(checkpoint, strict=False, map_location="cpu", config=target_config)
        except Exception as e:
            raise NotImplementedError(f"{model_type} inference stub failed: {e}")
        return model
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def run_inference(model, model_type, image_path, model_cfg=None):
    """
    Run inference for various models and return pycocotools format:
    [{"image_id": id, "category_id": id, "bbox": [x, y, w, h], "score": conf}, ...]
    """
    import cv2
    import torch
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return []
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    preds = []
    
    imgsz = model_cfg.get("imgsz", 672) if model_cfg else 672
    conf_thresh = model_cfg.get("conf", 0.45) if model_cfg else 0.45
    
    if model_type == "rf_detr":
        img_resized = cv2.resize(img_rgb, (imgsz, imgsz))
        
        # RFDETR prediction
        detections = model.predict(img_resized, threshold=conf_thresh)
        boxes = detections.xyxy
        
        if len(boxes) > 0:
            # Scale back to original image
            boxes[:, 0] = boxes[:, 0] * w / imgsz
            boxes[:, 1] = boxes[:, 1] * h / imgsz
            boxes[:, 2] = boxes[:, 2] * w / imgsz
            boxes[:, 3] = boxes[:, 3] * h / imgsz
            
            for bbox, score, cls_id in zip(boxes, detections.confidence, detections.class_id):
                x1, y1, x2, y2 = bbox
                preds.append({
                    "category_id": int(cls_id),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score)
                })
                
    elif model_type == "yolo":
        if hasattr(model, 'predict'):
            res = model.predict(img_rgb, verbose=False, conf=conf_thresh, imgsz=imgsz)[0]
            boxes = res.boxes
            if len(boxes) > 0:
                pred_boxes = boxes.xyxy.cpu().numpy()
                pred_scores = boxes.conf.cpu().numpy()
                pred_labels = boxes.cls.cpu().numpy().astype(int)
                for i in range(len(pred_boxes)):
                    x1, y1, x2, y2 = pred_boxes[i]
                    preds.append({
                        "category_id": int(pred_labels[i]),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(pred_scores[i])
                    })
        else: # AutoShape (YOLOv5)
            model.conf = conf_thresh
            res = model(img_rgb, size=imgsz)
            pred_df = res.pandas().xyxy[0]
            for i in range(len(pred_df)):
                x1, y1, x2, y2 = pred_df.iloc[i]['xmin'], pred_df.iloc[i]['ymin'], pred_df.iloc[i]['xmax'], pred_df.iloc[i]['ymax']
                score = pred_df.iloc[i]['confidence']
                label = pred_df.iloc[i]['class']
                preds.append({
                    "category_id": int(label),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score)
                })
                
    elif model_type in ["rt_detr", "deim", "mask2former", "rf_detr_lightning"]:
        import torch
        from PIL import Image
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model.to(device)
        model.eval()
        
        if model_type == "rf_detr_lightning":
            import torchvision.transforms.functional as F
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous()
            img_tensor = F.convert_image_dtype(img_tensor, dtype=torch.float32).to(device)
            # Resize image to match imgsz
            img_tensor = F.resize(img_tensor, [imgsz, imgsz])
            if next(model.parameters()).dtype == torch.float16:
                img_tensor = img_tensor.half()
            elif next(model.parameters()).dtype == torch.bfloat16:
                img_tensor = img_tensor.bfloat16()
            
            with torch.no_grad():
                out = model.model.model(img_tensor.unsqueeze(0))
            
            orig_size = torch.tensor([[h, w]], device=device)
            post = model.model.postprocess(out, orig_size)[0]
            boxes = post['boxes'].detach().cpu().numpy()
            scores = post['scores'].detach().cpu().numpy()
            labels = post['labels'].detach().cpu().numpy()
            
            for i in range(len(scores)):
                if scores[i] >= conf_thresh:
                    x1, y1, x2, y2 = boxes[i]
                    preds.append({
                        "category_id": int(labels[i]),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(scores[i])
                    })
        elif model_type == "rt_detr":
            img_pil = Image.fromarray(img_rgb)
            inputs = model.image_processor(images=img_pil, return_tensors="pt")
            pixel_val = inputs["pixel_values"].to(device)
            if next(model.parameters()).dtype == torch.float16:
                pixel_val = pixel_val.half()
            elif next(model.parameters()).dtype == torch.bfloat16:
                pixel_val = pixel_val.bfloat16()
                
            with torch.no_grad():
                out = model.model(pixel_values=pixel_val, labels=None)
                
            target_sizes = torch.tensor([[h, w]])
            post_out = model.image_processor.post_process_object_detection(
                out, threshold=conf_thresh, target_sizes=target_sizes
            )[0]
            
            boxes = post_out['boxes'].detach().cpu().numpy()
            scores = post_out['scores'].detach().cpu().numpy()
            labels = post_out['labels'].detach().cpu().numpy()
            
            for i in range(len(scores)):
                x1, y1, x2, y2 = boxes[i]
                preds.append({
                    "category_id": int(labels[i]),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(scores[i])
                })
                    
        elif model_type == "deim":
            import torchvision.transforms.functional as F
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous()
            img_tensor = F.convert_image_dtype(img_tensor, dtype=torch.float32).to(device)
            if next(model.parameters()).dtype == torch.float16:
                img_tensor = img_tensor.half()
            elif next(model.parameters()).dtype == torch.bfloat16:
                img_tensor = img_tensor.bfloat16()
                
            with torch.no_grad():
                out = model.model(img_tensor.unsqueeze(0))
                
            orig_size = torch.tensor([[h, w]], device=device)
            post = model.postprocessor(out, orig_size)[0]
            
            boxes = post['boxes'].detach().cpu().numpy()
            scores = post['scores'].detach().cpu().numpy()
            labels = post['labels'].detach().cpu().numpy()
            
            for i in range(len(scores)):
                if scores[i] >= conf_thresh:
                    x1, y1, x2, y2 = boxes[i]
                    cat_id = int(model.model_to_coco.get(int(labels[i]), int(labels[i]))) if hasattr(model, 'model_to_coco') and model.model_to_coco else int(labels[i])
                    preds.append({
                        "category_id": cat_id,
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(scores[i])
                    })
                    
        elif model_type == "mask2former":
            img_pil = Image.fromarray(img_rgb)
            inputs = model.image_processor(images=img_pil, return_tensors="pt")
            pixel_val = inputs["pixel_values"].to(device)
            if next(model.parameters()).dtype == torch.float16:
                pixel_val = pixel_val.half()
            elif next(model.parameters()).dtype == torch.bfloat16:
                pixel_val = pixel_val.bfloat16()
                
            with torch.no_grad():
                out = model.model(pixel_values=pixel_val)
                
            target_sizes = [img_pil.size[::-1]]
            post_out = model.image_processor.post_process_instance_segmentation(
                out, threshold=conf_thresh, target_sizes=target_sizes
            )[0]
            
            # TODO: Convert Mask2Former masks to boxes if needed, or if it outputs boxes:
            # Mask2Former segmentation outputs dict with 'segmentation' and 'segments_info'
            segments_info = post_out.get("segments_info", [])
            segmentation_mask = post_out.get("segmentation")
            
            if segmentation_mask is not None:
                segmentation_mask = segmentation_mask.detach().cpu().numpy()
                for seg in segments_info:
                    seg_id = seg["id"]
                    cat_id = seg["label_id"]
                    score = seg.get("score", 1.0)
                    if score >= conf_thresh:
                        import cv2
                        mask = (segmentation_mask == seg_id).astype(np.uint8)
                        y_indices, x_indices = np.where(mask > 0)
                        if len(y_indices) > 0:
                            x1, y1 = np.min(x_indices), np.min(y_indices)
                            x2, y2 = np.max(x_indices), np.max(y_indices)
                            
                            cat_id_mapped = int(model.model_to_coco.get(int(cat_id), int(cat_id))) if hasattr(model, 'model_to_coco') and model.model_to_coco else int(cat_id)
                            preds.append({
                                "category_id": cat_id_mapped,
                                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                "score": float(score)
                            })
                
    return preds

@hydra.main(version_base=None, config_path="configs", config_name="evaluate_phase2")
def main(cfg: DictConfig):
    # Depending on config structure, data_root might be under inference or at root
    data_root_path = cfg.get("data_root", None)
    output_root_path = cfg.get("output_dir", None)
    max_viz = 10
    if hasattr(cfg, "inference"):
        if not data_root_path:
            data_root_path = cfg.inference.get("data_root", None)
        if not output_root_path:
            output_root_path = cfg.inference.get("output_dir", None)
        max_viz = cfg.inference.get("max_viz", 10)
        
    if not data_root_path:
        print("[ERROR] 'data_root' not found in config.")
        return
        
    data_root = Path(data_root_path)
    if not data_root.exists() or not data_root.is_dir():
        print(f"[ERROR] Data root {data_root} does not exist.")
        return
        
    output_root = Path(output_root_path) if output_root_path else Path("outputs")
        
    if not hasattr(cfg, 'models') or not cfg.models:
        print("[ERROR] 'models' key not found in config.")
        return
        
    import cv2
    for model_key, model_cfg in cfg.models.items():
        print(f"\n======================================")
        print(f"[INFO] Initializing model {model_key} of type {model_cfg.type}...")
        print(f"======================================")
        try:
            model = load_model(model_key, model_cfg)
            if model is None:
                continue
            if model_cfg.type == "rf_detr":
                pass # model.optimize_for_inference() # Removed as it raises AttributeError
        except Exception as e:
            print(f"[ERROR] Skipping {model_key} due to load failure: {e}")
            continue
            
        model_type = model_cfg.type
        
        for dataset_dir in data_root.iterdir():
            if not dataset_dir.is_dir():
                continue
                
            anno_file = dataset_dir / "test_annotations.json"
            if not anno_file.exists():
                continue
                
            out_dir = output_root / dataset_dir.name / model_key
            out_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n[INFO] Evaluating dataset: {dataset_dir.name} with model {model_key}")
            
            coco_gt = COCO(str(anno_file))
            all_predictions = []
            image_ids = coco_gt.getImgIds()
            
            viz_count = 0
            
            for img_id in image_ids:
                img_info = coco_gt.loadImgs(img_id)[0]
                
                # Locate image file
                img_path = dataset_dir / img_info['file_name']
                if not img_path.exists():
                    img_path = dataset_dir / "images" / img_info['file_name']
                if not img_path.exists():
                    # For COCO datasets created with train/val/test splits
                    img_path = dataset_dir / "images" / "test" / img_info['file_name']
                    
                if img_path.exists():
                    preds = run_inference(model, model_type, str(img_path), model_cfg)
                    for p in preds:
                        p["image_id"] = img_id
                    all_predictions.extend(preds)
                    
                    if len(preds) > 0 and viz_count < max_viz:
                        img_viz = cv2.imread(str(img_path))
                        if img_viz is not None:
                            for p in preds:
                                bbox = p["bbox"]
                                x1, y1, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                                cv2.rectangle(img_viz, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
                                cv2.putText(img_viz, f"Cls {p['category_id']}: {p['score']:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            viz_path = out_dir / f"viz_{img_info['file_name']}"
                            cv2.imwrite(str(viz_path), img_viz)
                            viz_count += 1
                else:
                    print(f"[WARN] Image {img_info['file_name']} not found in {dataset_dir}")
                    
            if not all_predictions:
                print(f"[WARN] No predictions generated for {model_key} on {dataset_dir.name}. Skipping COCO eval.")
                continue
                
            res_file = out_dir / f"{model_key}_results.json"
            with open(res_file, "w") as f:
                json.dump(all_predictions, f)
                
            try:
                coco_dt = coco_gt.loadRes(str(res_file))
                
                metrics_dict = {"overall": {}, "class_wise": {}}
                
                # 1. Standard COCO Evaluation
                print(f"\n[INFO] Overall Metrics for {model_key} on {dataset_dir.name}:")
                coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                
                if coco_eval.stats is not None and len(coco_eval.stats) > 1:
                    metrics_dict["overall"]["AP"] = float(coco_eval.stats[0])
                    metrics_dict["overall"]["AP50"] = float(coco_eval.stats[1])
                
                # 2. Class-wise metrics
                print(f"\n[INFO] Class-wise Metrics for {model_key} on {dataset_dir.name}:")
                class_metrics = {cat_id: {"TP": 0, "FP": 0, "FN": 0} for cat_id in coco_gt.getCatIds()}
                
                for img_id in image_ids:
                    ann_ids = coco_gt.getAnnIds(imgIds=img_id)
                    anns = coco_gt.loadAnns(ann_ids)
                    
                    det_ids = coco_dt.getAnnIds(imgIds=img_id)
                    dets = coco_dt.loadAnns(det_ids)
                    
                    for cat_id in class_metrics.keys():
                        cat_anns = [a for a in anns if a['category_id'] == cat_id]
                        cat_dets = [d for d in dets if d['category_id'] == cat_id]
                        
                        gt_boxes = np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0]+a['bbox'][2], a['bbox'][1]+a['bbox'][3]] for a in cat_anns]) if len(cat_anns) > 0 else np.zeros((0, 4))
                        det_boxes = np.array([[d['bbox'][0], d['bbox'][1], d['bbox'][0]+d['bbox'][2], d['bbox'][1]+d['bbox'][3]] for d in cat_dets]) if len(cat_dets) > 0 else np.zeros((0, 4))
                        
                        paired_idx, unpaired_gts, unpaired_dets = pair_gts_dets_bbox(gt_boxes, det_boxes, 0.5)
                        
                        class_metrics[cat_id]["TP"] += len(paired_idx)
                        class_metrics[cat_id]["FP"] += len(unpaired_dets)
                        class_metrics[cat_id]["FN"] += len(unpaired_gts)
                        
                print(f"{'Class':<14} {'Images':<7} {'Labels':<10} {'P':<7} {'R':<8} {'mAP@.5':<10} {'mAP@.5:.95':<10}")
                
                total_images = len(image_ids)
                total_labels = len(coco_gt.getAnnIds())
                
                ap_all = coco_eval.stats[0] if coco_eval.stats is not None and len(coco_eval.stats) > 0 else -1
                ap50_all = coco_eval.stats[1] if coco_eval.stats is not None and len(coco_eval.stats) > 0 else -1
                
                total_tp = sum(m["TP"] for m in class_metrics.values())
                total_fp = sum(m["FP"] for m in class_metrics.values())
                total_fn = sum(m["FN"] for m in class_metrics.values())
                
                p_all = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                r_all = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                
                def fmt(val):
                    return f"{val:.3f}" if val >= 0 else "0.000"
                
                print(f"   {'all':<13} {total_images:<7} {total_labels:<10} {fmt(p_all):<7} {fmt(r_all):<8} {fmt(ap50_all):<10} {fmt(ap_all):<10}")
                
                KNOWN_CLASSES = {0: "cell", 1: "bead", 2: "cell-adhered", 3: "soma"}
                all_cat_ids = sorted(list(set(list(KNOWN_CLASSES.keys()) + coco_gt.getCatIds())))
                
                for cat_id in all_cat_ids:
                    cat_name = KNOWN_CLASSES.get(cat_id, f"class_{cat_id}")
                    
                    if cat_id in coco_gt.getCatIds():
                        cat_labels = len(coco_gt.getAnnIds(catIds=[cat_id]))
                    else:
                        cat_labels = 0
                        
                    ap = -1.0
                    ap50 = -1.0
                    if cat_id in coco_eval.params.catIds:
                        import sys
                        from io import StringIO
                        old_stdout = sys.stdout
                        sys.stdout = StringIO()
                        
                        cat_eval = COCOeval(coco_gt, coco_dt, 'bbox')
                        cat_eval.params.catIds = [cat_id]
                        cat_eval.evaluate()
                        cat_eval.accumulate()
                        cat_eval.summarize()
                        
                        sys.stdout = old_stdout
                        
                        ap = cat_eval.stats[0] if cat_eval.stats is not None and len(cat_eval.stats) > 0 else -1.0
                        ap50 = cat_eval.stats[1] if cat_eval.stats is not None and len(cat_eval.stats) > 0 else -1.0
                    
                    tp = class_metrics[cat_id]["TP"] if cat_id in class_metrics else 0
                    fp = class_metrics[cat_id]["FP"] if cat_id in class_metrics else 0
                    fn = class_metrics[cat_id]["FN"] if cat_id in class_metrics else 0
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                    
                    metrics_dict["class_wise"][cat_name] = {
                        "AP": float(ap), "AP50": float(ap50),
                        "TP": int(tp), "FP": int(fp), "FN": int(fn),
                        "Precision": float(precision), "Recall": float(recall), "F1": float(f1)
                    }
                    
                    print(f"   {cat_name:<13} {total_images:<7} {cat_labels:<10} {fmt(precision):<7} {fmt(recall):<8} {fmt(ap50):<10} {fmt(ap):<10}")
                        
                metrics_file = out_dir / "coco_metrics.json"
                with open(metrics_file, "w") as f:
                    json.dump(metrics_dict, f, indent=4)
                        
            except Exception as e:
                print(f"[ERROR] Evaluation failed for {model_key} on {dataset_dir.name}: {e}")

if __name__ == "__main__":
    main()
