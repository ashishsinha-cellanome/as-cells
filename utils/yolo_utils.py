import os
import yaml
from pathlib import Path
from pycocotools.coco import COCO
from tqdm import tqdm

def convert_coco_to_yolo(split_name, image_dir, json_path, cache_dir, label_map, fraction=1.0):
    cache_dir = Path(cache_dir)
    split_img_dir = cache_dir / "images" / split_name
    split_lbl_dir = cache_dir / "labels" / split_name

    split_img_dir.mkdir(parents=True, exist_ok=True)
    split_lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Converting {split_name} COCO to YOLO format...")
    coco = COCO(json_path)

    name_to_id = {v: int(k) for k, v in label_map.items()}

    img_ids = coco.getImgIds()
    if fraction < 1.0:
        import random
        random.seed(42)
        img_ids = random.sample(img_ids, int(len(img_ids) * fraction))

    for img_id in tqdm(img_ids, desc=f"Converting {split_name}"):
        img_info = coco.loadImgs(img_id)[0]
        img_name = img_info["file_name"]
        src_img_path = os.path.join(image_dir, img_name)
        dst_img_path = split_img_dir / img_name

        if not dst_img_path.exists() and os.path.exists(src_img_path):
            os.symlink(src_img_path, dst_img_path)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        yolo_labels = []
        img_w, img_h = img_info["width"], img_info["height"]

        for ann in anns:
            cat_info = coco.loadCats(ann["category_id"])[0]
            if cat_info["name"] not in name_to_id:
                continue

            yolo_class_id = name_to_id[cat_info["name"]]
            x, y, w, h = ann["bbox"]
            if w == 0 or h == 0:
                continue

            x_c, y_c = (x + w / 2) / img_w, (y + h / 2) / img_h
            w_norm, h_norm = w / img_w, h / img_h
            yolo_labels.append(f"{yolo_class_id} {x_c:.6f} {y_c:.6f} {w_norm:.6f} {h_norm:.6f}")

        with open(split_lbl_dir / f"{Path(img_name).stem}.txt", "w") as f:
            f.write("\n".join(yolo_labels))

def create_data_yaml(cache_dir, train_name, val_name, test_name, label_map):
    cache_dir = Path(cache_dir)
    yaml_path = cache_dir / "data.yaml"
    
    names = [label_map[k] for k in sorted(label_map.keys())]
    
    data = {
        "path": str(cache_dir),
        "train": f"images/{train_name}",
        "val": f"images/{val_name}",
        "test": f"images/{test_name}",
        "names": {i: name for i, name in enumerate(names)}
    }
    
    with open(yaml_path, "w") as f:
        yaml.dump(data, f)
        
    return str(yaml_path)

def visualize_predictions(model, config, epoch, split="val"):
    from utils.distributed_utils import get_rank
    if get_rank() != 0:
        return
        
    viz_every = config.checkpointing.get("visualize_every_n_epochs", 5)
    if epoch is not None and (epoch + 1) % viz_every != 0 and (epoch + 1) != config.model.yolov26.epochs:
        return

    max_samples = config.checkpointing.get("visualize_samples", 100)
    if max_samples == -1:
        max_samples = None

    save_dir = os.path.join(
        config.checkpointing.save_dir,
        config.checkpointing.get("visualization_dir", "predictions"),
        f"epoch_{(epoch + 1):03d}" if epoch is not None else "test",
        split,
    )
    os.makedirs(save_dir, exist_ok=True)
    print(f"[VIZ] Saving {split.upper()} visualizations to: {save_dir}")

    import torchvision.transforms.functional as F
    from torchvision.utils import draw_bounding_boxes
    from PIL import Image
    import torch

    dp = config.data.path
    val_name = config.data.val_name if split == "val" else config.data.test_name
    coco_gt_path = os.path.join(dp, f"{val_name}_annotations.json")
    img_dir = os.path.join(dp, "images", val_name)
    
    if not os.path.exists(coco_gt_path):
        return

    coco_gt = COCO(coco_gt_path)
    img_ids = sorted(coco_gt.getImgIds())
    
    saved_count = 0
    viz_threshold = float(config.model.get("draw_threshold", 0.4))
    
    label_map = config.model.label_map
    
    for img_id in img_ids:
        if max_samples is not None and saved_count >= max_samples:
            break
            
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            continue
            
        try:
            image_pil = Image.open(img_path).convert("RGB")
            image_tensor = F.pil_to_tensor(image_pil)
        except Exception:
            continue
            
        # Get GT
        gt_boxes = []
        gt_labels = []
        try:
            gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id]))
            for ann in gt_anns:
                x, y, w, h = ann["bbox"]
                if w > 0 and h > 0:
                    gt_boxes.append([x, y, x + w, y + h])
                    cat_info = coco_gt.loadCats(ann["category_id"])[0]
                    cat_name = cat_info["name"]
                    gt_labels.append(f"GT: {cat_name}")
        except Exception:
            pass

        # Get Preds
        results = model.predict(image_pil, verbose=False, conf=viz_threshold)[0]
        pred_boxes = []
        pred_labels = []
        if len(results.boxes):
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy()
            
            for box, conf, cls_id in zip(boxes, confs, clss):
                pred_boxes.append(box.tolist())
                class_name = label_map.get(int(cls_id), str(cls_id))
                pred_labels.append(f"Pred: {class_name} {conf:.2f}")

        colors = []
        all_boxes = []
        all_labels = []
        
        if len(gt_boxes) > 0:
            all_boxes.extend(gt_boxes)
            all_labels.extend(gt_labels)
            colors.extend(["green"] * len(gt_boxes))
            
        if len(pred_boxes) > 0:
            all_boxes.extend(pred_boxes)
            all_labels.extend(pred_labels)
            colors.extend(["red"] * len(pred_boxes))
            
        if len(all_boxes) > 0:
            boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
            drawn_image = draw_bounding_boxes(
                image_tensor,
                boxes=boxes_tensor,
                labels=all_labels,
                colors=colors,
                width=2,
                font_size=12
            )
        else:
            drawn_image = image_tensor
            
        out_pil = F.to_pil_image(drawn_image)
        out_path = os.path.join(save_dir, img_info["file_name"])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out_pil.save(out_path)
        saved_count += 1
