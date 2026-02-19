import json
import shutil
from pathlib import Path


def _safe_symlink(src: Path, dst: Path):
    """Create or refresh a symlink, falling back to copy if symlink fails."""
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    try:
        dst.symlink_to(src, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _load_coco(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_coco(path: Path, payload):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def prepare_rfdetr_roboflow_layout(
    dataset_path: str,
    cache_root: str,
    train_name: str,
    val_name: str,
    test_name: str,
):
    """
    Create a Roboflow-style dataset view for RF-DETR:
      root/train/_annotations.coco.json + root/train/images/*
      root/valid/_annotations.coco.json + root/valid/images/*
      root/test/_annotations.coco.json  + root/test/images/*
    """
    dataset_root = Path(dataset_path)
    out_root = Path(cache_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train": train_name,
        "valid": val_name,
        "test": test_name,
    }

    for target_split, source_split in split_map.items():
        src_img_dir = dataset_root / "images" / source_split
        src_ann = dataset_root / f"{source_split}_annotations.json"

        split_dir = out_root / target_split
        split_dir.mkdir(parents=True, exist_ok=True)
        split_images_dir = split_dir / "images"
        _safe_symlink(src_img_dir, split_images_dir)

        coco = _load_coco(src_ann)
        for image in coco.get("images", []):
            file_name = image.get("file_name", "")
            if not file_name.startswith("images/"):
                image["file_name"] = f"images/{file_name}"

        _write_coco(split_dir / "_annotations.coco.json", coco)

    return str(out_root)


def _build_yolo_class_maps(coco_payload, label_map=None):
    """
    Returns:
      coco_to_yolo: dict[coco_cat_id] -> contiguous yolo_class_id
      yolo_to_coco: dict[yolo_class_id] -> coco_cat_id
      yolo_names:   list[str] names ordered by yolo_class_id
    """
    categories = coco_payload.get("categories", [])
    if not categories:
        return {}, {}, []

    if label_map:
        desired = {int(k): v for k, v in label_map.items()}
        ordered_names = [desired[idx] for idx in sorted(desired.keys())]
        name_to_idx = {name: idx for idx, name in enumerate(ordered_names)}

        coco_to_yolo = {}
        yolo_to_coco = {}
        for cat in categories:
            cat_name = cat["name"]
            if cat_name in name_to_idx:
                yolo_idx = name_to_idx[cat_name]
                coco_cat_id = int(cat["id"])
                coco_to_yolo[coco_cat_id] = yolo_idx
                yolo_to_coco[yolo_idx] = coco_cat_id

        yolo_names = ordered_names
        return coco_to_yolo, yolo_to_coco, yolo_names

    sorted_categories = sorted(categories, key=lambda x: int(x["id"]))
    coco_to_yolo = {int(cat["id"]): idx for idx, cat in enumerate(sorted_categories)}
    yolo_to_coco = {idx: int(cat["id"]) for idx, cat in enumerate(sorted_categories)}
    yolo_names = [cat["name"] for cat in sorted_categories]
    return coco_to_yolo, yolo_to_coco, yolo_names


def _convert_coco_annotations_to_yolo(coco_payload, labels_dir: Path, coco_to_yolo):
    labels_dir.mkdir(parents=True, exist_ok=True)
    img_by_id = {int(img["id"]): img for img in coco_payload.get("images", [])}
    anns_by_img = {}
    for ann in coco_payload.get("annotations", []):
        anns_by_img.setdefault(int(ann["image_id"]), []).append(ann)

    path_to_image_id = {}
    for image_id, img in img_by_id.items():
        file_name = img["file_name"]
        width = float(img["width"])
        height = float(img["height"])

        label_file = labels_dir / (Path(file_name).stem + ".txt")
        path_to_image_id[file_name] = image_id

        lines = []
        for ann in anns_by_img.get(image_id, []):
            coco_cat = int(ann["category_id"])
            if coco_cat not in coco_to_yolo:
                continue
            cls = coco_to_yolo[coco_cat]
            x, y, w, h = ann["bbox"]
            cx = (float(x) + float(w) / 2.0) / width
            cy = (float(y) + float(h) / 2.0) / height
            nw = float(w) / width
            nh = float(h) / height
            lines.append(f"{cls} {cx:.8f} {cy:.8f} {nw:.8f} {nh:.8f}")

        with open(label_file, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

    return path_to_image_id


def prepare_yolov5_layout(
    dataset_path: str,
    cache_root: str,
    train_name: str,
    val_name: str,
    test_name: str,
    label_map=None,
):
    """
    Create a YOLOv5 dataset cache:
      root/images/{train,val,test}
      root/labels/{train,val,test}
      root/dataset.yaml
    """
    dataset_root = Path(dataset_path)
    out_root = Path(cache_root).expanduser().resolve()
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "labels").mkdir(parents=True, exist_ok=True)

    source_splits = {
        "train": train_name,
        "val": val_name,
        "test": test_name,
    }

    # Build class mapping from training categories for deterministic ids.
    train_coco = _load_coco(dataset_root / f"{train_name}_annotations.json")
    coco_to_yolo, yolo_to_coco, yolo_names = _build_yolo_class_maps(train_coco, label_map=label_map)

    split_path_to_image_id = {}
    for target_split, source_split in source_splits.items():
        src_img_dir = dataset_root / "images" / source_split
        src_ann_path = dataset_root / f"{source_split}_annotations.json"
        coco_payload = _load_coco(src_ann_path)

        dst_img_dir = out_root / "images" / target_split
        _safe_symlink(src_img_dir, dst_img_dir)
        dst_label_dir = out_root / "labels" / target_split

        split_path_to_image_id[target_split] = _convert_coco_annotations_to_yolo(
            coco_payload=coco_payload,
            labels_dir=dst_label_dir,
            coco_to_yolo=coco_to_yolo,
        )

    dataset_yaml = {
        "path": str(out_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(yolo_names),
        "names": yolo_names,
    }
    yaml_path = out_root / "dataset.yaml"
    try:
        import yaml

        with open(yaml_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(dataset_yaml, fh, sort_keys=False)
    except Exception:
        with open(yaml_path, "w", encoding="utf-8") as fh:
            for key, value in dataset_yaml.items():
                fh.write(f"{key}: {value}\n")

    return {
        "root": str(out_root),
        "yaml": str(yaml_path),
        "yolo_to_coco": yolo_to_coco,
        "split_path_to_image_id": split_path_to_image_id,
        "names": yolo_names,
    }
