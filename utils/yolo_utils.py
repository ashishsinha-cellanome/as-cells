import os
import yaml
from pathlib import Path
from pycocotools.coco import COCO
from tqdm import tqdm

def convert_coco_to_yolo(split_name, image_dir, json_path, cache_dir, label_map):
    cache_dir = Path(cache_dir)
    split_img_dir = cache_dir / "images" / split_name
    split_lbl_dir = cache_dir / "labels" / split_name

    split_img_dir.mkdir(parents=True, exist_ok=True)
    split_lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Converting {split_name} COCO to YOLO format...")
    coco = COCO(json_path)

    name_to_id = {v: int(k) for k, v in label_map.items()}

    for img_id in tqdm(coco.getImgIds(), desc=f"Converting {split_name}"):
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
