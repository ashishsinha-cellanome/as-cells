import os
import torch
import numpy as np
import albumentations as A
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision.datasets import CocoDetection

from utils.dataset_utils import get_transform


class DeimDataset(CocoDetection):
    def __init__(self, root, annFile, transforms=None, remap_dict=None):
        super().__init__(root, annFile, transforms=None)
        self.albu_transforms = transforms
        self.remap_dict = remap_dict

    def __getitem__(self, idx):
        image, annotations = super().__getitem__(idx)
        image_array = np.array(image.convert("RGB"))

        orig_h, orig_w = image_array.shape[:2]

        if len(annotations) > 0:
            image_id = annotations[0]["image_id"]
        else:
            image_id = self.ids[idx]

        valid_annotations = []
        if self.remap_dict is not None:
            for record in annotations:
                cat_id = record["category_id"]
                if cat_id in self.remap_dict:
                    new_record = record.copy()
                    new_record["category_id"] = self.remap_dict[cat_id]
                    valid_annotations.append(new_record)
            annotations = valid_annotations

        boxes = []
        labels = []
        for record in annotations:
            # Albumentations expects [x_min, y_min, w, h] for coco format
            if record["bbox"][2] > 0 and record["bbox"][3] > 0:
                boxes.append(record["bbox"])
                labels.append(record["category_id"])

        if self.albu_transforms is not None and len(boxes) > 0:
            transformed = self.albu_transforms(
                image=image_array, bboxes=boxes, category=labels
            )
            img = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["category"]
        elif self.albu_transforms is not None and len(boxes) == 0:
            transformed = self.albu_transforms(
                image=image_array, bboxes=[], category=[]
            )
            img = transformed["image"]
            boxes = []
            labels = []
        else:
            img = image_array

        # Convert image to [C, H, W] tensor and normalize [0, 1]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Apply ImageNet normalization manually (as required by DEIM/DINO)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std

        h, w = img.shape[-2:]

        out_boxes = []
        out_labels = []
        out_areas = []

        for i, bbox in enumerate(boxes):
            if int(bbox[2]) * int(bbox[3]) == 0:
                continue
            # Convert coco [x, y, w, h] to cxcywh normalized
            x_min, y_min, bw, bh = bbox
            cx = x_min + bw / 2.0
            cy = y_min + bh / 2.0

            # Normalize by new image size
            cx /= w
            cy /= h
            nw = bw / w
            nh = bh / h

            out_boxes.append([cx, cy, nw, nh])
            out_labels.append(labels[i])
            out_areas.append(bw * bh)

        target_dict = {
            "boxes": torch.as_tensor(out_boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(out_labels, dtype=torch.int64),
            "area": torch.as_tensor(out_areas, dtype=torch.float32),
            "image_id": torch.tensor([image_id]),
            "orig_size": torch.tensor([orig_h, orig_w]),
        }

        return img, target_dict


class DeimV2DataModule(pl.LightningDataModule):
    def __init__(self, dataset_path: str, config):
        super().__init__()
        self.dataset_path = dataset_path
        self.config = config

        # Provide defaults
        self.batch_size = getattr(self.config.data, "batch_size", 16)
        self.num_workers = getattr(self.config.data, "num_workers", 4)
        self.model_input_size = getattr(self.config.data, "model_input_size", 640)
        self.min_random_scale = getattr(self.config.data, "min_random_scale", 0.7)
        self.max_random_scale = getattr(self.config.data, "max_random_scale", 1.0)
        self.p_noise = getattr(self.config.data, "p_noise", 0.25)
        self.org_images_in_model_input_size = getattr(
            self.config.data, "org_images_in_model_input_size", True
        )

    def _get_remap_dict(self, coco_dataset: CocoDetection):
        if hasattr(self.config, "remap_labels") and not self.config.remap_labels:
            return None
        if not hasattr(self.config, "model") or "label_map" not in self.config.model:
            return None

        target_label_map = self.config.model.label_map
        name_to_target_id = {v: int(k) for k, v in target_label_map.items()}

        remapping_rules = {}
        if (
            hasattr(self.config, "data")
            and self.config.data
            and "class_remapping" in self.config.data
        ):
            remapping_rules = self.config.data.class_remapping
        elif "class_remapping" in self.config:
            remapping_rules = self.config.class_remapping

        remap_dict = {}
        for cat_id, cat_info in coco_dataset.coco.cats.items():
            src_name = cat_info["name"]
            effective_name = remapping_rules.get(src_name, src_name)
            if effective_name in name_to_target_id:
                target_id = name_to_target_id[effective_name]
                remap_dict[cat_id] = target_id

        if remap_dict:
            print(
                f"[INFO] Built Label Remap Dict: {remap_dict} (Source IDs -> Target IDs)"
            )
            return remap_dict
        return None

    def setup(self, stage=None):
        if stage in ("fit", None):
            train_images_path = os.path.join(
                self.dataset_path, "images", self.config.train_name
            )
            train_annot_path = os.path.join(
                self.dataset_path, f"{self.config.train_name}_annotations.json"
            )
            if getattr(self.config, "debug", False):
                train_images_path = os.path.join(
                    self.dataset_path, "images", self.config.val_name
                )
                train_annot_path = os.path.join(
                    self.dataset_path, f"{self.config.val_name}_annotations.json"
                )

            train_coco = CocoDetection(root=train_images_path, annFile=train_annot_path)
            train_remap = self._get_remap_dict(train_coco)

            # Simple fallback transform if none provided
            train_transforms = get_transform(
                model_input_width=self.model_input_size,
                model_input_height=self.model_input_size,
                min_random_scale=self.min_random_scale,
                max_random_scale=self.max_random_scale,
                p_noise=self.p_noise,
                org_images_in_model_input_size=self.org_images_in_model_input_size,
                train=True,
                transforms_config=None,
            )

            self.train_dataset = DeimDataset(
                root=train_images_path,
                annFile=train_annot_path,
                transforms=train_transforms,
                remap_dict=train_remap,
            )

            val_images_path = os.path.join(
                self.dataset_path, "images", self.config.val_name
            )
            val_annot_path = os.path.join(
                self.dataset_path, f"{self.config.val_name}_annotations.json"
            )
            val_coco = CocoDetection(root=val_images_path, annFile=val_annot_path)
            val_remap = self._get_remap_dict(val_coco)

            val_transforms = get_transform(
                model_input_width=self.model_input_size,
                model_input_height=self.model_input_size,
                min_random_scale=self.min_random_scale,
                max_random_scale=self.max_random_scale,
                p_noise=self.p_noise,
                org_images_in_model_input_size=self.org_images_in_model_input_size,
                train=False,
                transforms_config=None,
            )

            self.val_dataset = DeimDataset(
                root=val_images_path,
                annFile=val_annot_path,
                transforms=val_transforms,
                remap_dict=val_remap,
            )

        if stage in ("test", None):
            test_name = getattr(self.config, "test_name", self.config.val_name)
            test_images_path = os.path.join(self.dataset_path, "images", test_name)
            test_annot_path = os.path.join(
                self.dataset_path, f"{test_name}_annotations.json"
            )
            if getattr(self.config, "debug", False):
                test_images_path = os.path.join(
                    self.dataset_path, "images", self.config.val_name
                )
                test_annot_path = os.path.join(
                    self.dataset_path, f"{self.config.val_name}_annotations.json"
                )

            test_coco = CocoDetection(root=test_images_path, annFile=test_annot_path)
            test_remap = self._get_remap_dict(test_coco)

            test_transforms = get_transform(
                model_input_width=self.model_input_size,
                model_input_height=self.model_input_size,
                min_random_scale=self.min_random_scale,
                max_random_scale=self.max_random_scale,
                p_noise=self.p_noise,
                org_images_in_model_input_size=self.org_images_in_model_input_size,
                train=False,
                transforms_config=None,
            )

            self.test_dataset = DeimDataset(
                root=test_images_path,
                annFile=test_annot_path,
                transforms=test_transforms,
                remap_dict=test_remap,
            )

    def collate_fn(self, batch):
        images = torch.stack([x[0] for x in batch])
        targets = [x[1] for x in batch]
        return images, targets

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=getattr(self.config.data, "eval_batch_size", 1),
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        if not hasattr(self, "test_dataset") or self.test_dataset is None:
            self.setup(stage="test")
        return DataLoader(
            self.test_dataset,
            batch_size=getattr(self.config.data, "eval_batch_size", 1),
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
