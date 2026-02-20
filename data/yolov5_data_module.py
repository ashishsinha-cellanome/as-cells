from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CocoDetection

from omegaconf import OmegaConf
from utils.dataset_utils import get_transform


def _letterbox(image: np.ndarray, img_size: int, color=(114, 114, 114)):
    """Resize + pad while keeping aspect ratio."""
    shape = image.shape[:2]  # h, w
    ratio = min(img_size / shape[0], img_size / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))  # w, h
    dw = img_size - new_unpad[0]
    dh = img_size - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, ratio, (dw, dh)


class YoloCocoDataset(Dataset):
    """
    COCO-backed dataset that emits YOLOv5-style training targets directly,
    avoiding per-image .txt label generation.
    """

    def __init__(
        self,
        dataset_path: str,
        split_name: str,
        img_size: int,
        transforms,
        coco_cat_to_model_id: Dict[int, int],
    ):
        self.dataset_path = Path(dataset_path)
        self.split_name = split_name
        self.img_size = int(img_size)
        self.transforms = transforms
        self.coco_cat_to_model_id = coco_cat_to_model_id

        self.dataset = CocoDetection(
            root=str(self.dataset_path / "images" / split_name),
            annFile=str(self.dataset_path / f"{split_name}_annotations.json"),
            transforms=None,
        )

    @property
    def coco(self):
        return self.dataset.coco

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_pil, anns = self.dataset[idx]
        image = np.array(image_pil.convert("RGB"))
        h0, w0 = image.shape[:2]
        image_id = int(self.dataset.ids[idx])
        file_name = self.dataset.coco.imgs[image_id]["file_name"]

        boxes = []
        labels = []
        for ann in anns:
            cat_id = int(ann["category_id"])
            if cat_id not in self.coco_cat_to_model_id:
                continue
            boxes.append(ann["bbox"])  # COCO xywh absolute
            labels.append(self.coco_cat_to_model_id[cat_id])

        if self.transforms is not None and len(boxes) > 0:
            transformed = self.transforms(image=image, bboxes=boxes, category=labels)
            image = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["category"]

        image_lb, ratio, (dw, dh) = _letterbox(image, self.img_size)
        targets = []
        for box, cls in zip(boxes, labels):
            x, y, w, h = [float(v) for v in box]
            x1 = x * ratio + dw
            y1 = y * ratio + dh
            x2 = (x + w) * ratio + dw
            y2 = (y + h) * ratio + dh

            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            if bw <= 0.0 or bh <= 0.0:
                continue

            cx = (x1 + x2) / 2.0 / self.img_size
            cy = (y1 + y2) / 2.0 / self.img_size
            nw = bw / self.img_size
            nh = bh / self.img_size
            targets.append([float(cls), cx, cy, nw, nh])

        image_tensor = torch.from_numpy(image_lb).permute(2, 0, 1).contiguous()
        labels_tensor = (
            torch.tensor(targets, dtype=torch.float32)
            if targets
            else torch.zeros((0, 5), dtype=torch.float32)
        )
        shape_meta = ((h0, w0), (ratio, (dw, dh)))
        return image_tensor, labels_tensor, file_name, shape_meta, image_id


def yolo_collate_fn(batch):
    images, labels, paths, shapes, image_ids = zip(*batch)
    images = torch.stack(images, 0)

    targets_out = []
    for batch_idx, sample_labels in enumerate(labels):
        if sample_labels.numel() == 0:
            continue
        sample_targets = torch.zeros((sample_labels.shape[0], 6), dtype=torch.float32)
        sample_targets[:, 0] = batch_idx
        sample_targets[:, 1:] = sample_labels
        targets_out.append(sample_targets)

    targets = (
        torch.cat(targets_out, 0)
        if targets_out
        else torch.zeros((0, 6), dtype=torch.float32)
    )
    return images, targets, list(paths), list(shapes), list(image_ids)


class YOLOv5DataModule(pl.LightningDataModule):
    """COCO-native YOLOv5 datamodule with no label-file generation."""

    def __init__(self, dataset_root: str, config):
        super().__init__()
        self.dataset_root = str(Path(dataset_root).expanduser().resolve())
        self.config = config

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.coco_cat_to_model_id: Dict[int, int] = {}
        self.model_to_coco_map: Dict[int, int] = {}

    def _build_class_maps(self, train_coco):
        target_label_map = {int(k): v for k, v in self.config.model.label_map.items()}
        name_to_model_id = {name: model_id for model_id, name in target_label_map.items()}
        remap_rules = {}
        if getattr(self.config, "remap_labels", False):
            remap_rules = dict(getattr(self.config.data, "class_remapping", {}))

        cat_to_model = {}
        model_to_coco = {}
        for cat_id, cat_info in train_coco.cats.items():
            src_name = cat_info["name"]
            effective_name = remap_rules.get(src_name, src_name)
            if effective_name not in name_to_model_id:
                continue
            model_id = name_to_model_id[effective_name]
            cat_to_model[int(cat_id)] = int(model_id)
            if model_id not in model_to_coco:
                model_to_coco[int(model_id)] = int(cat_id)

        self.coco_cat_to_model_id = cat_to_model
        self.model_to_coco_map = model_to_coco

    def _build_transforms(self, train: bool):
        data_cfg = self.config.data
        transforms_config = None
        if hasattr(data_cfg, "transforms"):
            split_key = "train" if train else "test"
            if hasattr(data_cfg.transforms, split_key):
                transforms_config = OmegaConf.to_container(getattr(data_cfg.transforms, split_key), resolve=True)

        return get_transform(
            model_input_width=int(self.config.model.input_size),
            model_input_height=int(self.config.model.input_size),
            min_random_scale=float(data_cfg.min_random_scale),
            max_random_scale=float(data_cfg.max_random_scale),
            p_noise=float(data_cfg.p_noise),
            org_images_in_model_input_size=bool(data_cfg.org_images_in_model_input_size),
            train=train,
            transforms_config=transforms_config,
        )

    def setup(self, stage=None):
        train_coco = CocoDetection(
            root=str(Path(self.dataset_root) / "images" / self.config.train_name),
            annFile=str(Path(self.dataset_root) / f"{self.config.train_name}_annotations.json"),
            transforms=None,
        )
        self._build_class_maps(train_coco.coco)

        if stage in ("fit", None):
            self.train_dataset = YoloCocoDataset(
                dataset_path=self.dataset_root,
                split_name=self.config.train_name,
                img_size=int(self.config.model.input_size),
                transforms=self._build_transforms(train=True),
                coco_cat_to_model_id=self.coco_cat_to_model_id,
            )
            self.val_dataset = YoloCocoDataset(
                dataset_path=self.dataset_root,
                split_name=self.config.val_name,
                img_size=int(self.config.model.input_size),
                transforms=self._build_transforms(train=False),
                coco_cat_to_model_id=self.coco_cat_to_model_id,
            )

        if stage in ("test", None):
            self.test_dataset = YoloCocoDataset(
                dataset_path=self.dataset_root,
                split_name=self.config.test_name,
                img_size=int(self.config.model.input_size),
                transforms=self._build_transforms(train=False),
                coco_cat_to_model_id=self.coco_cat_to_model_id,
            )

    @property
    def val_coco_gt(self):
        if self.val_dataset is None:
            return None
        return self.val_dataset.coco

    @property
    def test_coco_gt(self):
        if self.test_dataset is None:
            return None
        return self.test_dataset.coco

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.config.data.batch_size),
            shuffle=True,
            num_workers=int(self.config.data.num_workers),
            collate_fn=yolo_collate_fn,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=int(getattr(self.config.data, "eval_batch_size", 1)),
            shuffle=False,
            num_workers=int(self.config.data.num_workers),
            collate_fn=yolo_collate_fn,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=int(getattr(self.config.data, "eval_batch_size", 1)),
            shuffle=False,
            num_workers=int(self.config.data.num_workers),
            collate_fn=yolo_collate_fn,
            pin_memory=True,
            drop_last=False,
        )
