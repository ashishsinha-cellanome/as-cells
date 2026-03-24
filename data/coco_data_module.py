import os
from typing import Optional, Dict
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision.datasets import CocoDetection
import albumentations as A
from omegaconf import OmegaConf

from utils.dataset_utils import get_transform


class CocoDataset(torch.utils.data.Dataset):
    """Dataset wrapper for COCO format data with image processor."""

    def __init__(
        self,
        dataset_coco: CocoDetection,
        processor,
        transforms: Optional[A.core.composition.Compose] = None,
        remap_dict: Optional[Dict[int, int]] = None,
    ):
        self.dataset_coco = dataset_coco
        self.processor = processor
        self.transforms = transforms
        self.remap_dict = remap_dict

    def __len__(self):
        return len(self.dataset_coco)

    def __getitem__(self, idx):
        import numpy as np

        image, annotations = self.dataset_coco[idx]

        # Convert image to RGB numpy array
        image_array = np.array(image.convert("RGB"))

        # Extract annotations
        if len(annotations) > 0:
            image_id = annotations[0]["image_id"]
        else:
            image_id = idx + 1

        # Remap and Filter annotations if remap_dict is provided
        valid_annotations = []
        if self.remap_dict is not None:
            for record in annotations:
                cat_id = record["category_id"]
                if cat_id in self.remap_dict:
                    # Create a copy to avoid modifying the original dataset cache if any
                    new_record = record.copy()
                    new_record["category_id"] = self.remap_dict[cat_id]
                    valid_annotations.append(new_record)
            annotations = valid_annotations

        boxes = np.array([record["bbox"] for record in annotations])
        labels = [record["category_id"] for record in annotations]

        # Apply augmentations
        if self.transforms and len(boxes) > 0:
            transformed = self.transforms(
                image=image_array, bboxes=boxes, category=labels
            )
            img = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["category"]
        else:
            img = image_array

        # Reformat annotations
        formatted_annotations = []
        for i, bbox in enumerate(boxes):
            if int(bbox[2]) * int(bbox[3]) == 0:
                # Skip zero area invalid annotations post augmentation
                continue
            record = {
                "image_id": image_id,
                "category_id": int(labels[i]),
                "bbox": np.array([int(v) for v in bbox]),
                "iscrowd": 0,
                "area": bbox[2] * bbox[3],
            }
            formatted_annotations.append(record)

        formatted_annotation = {
            "image_id": image_id,
            "annotations": formatted_annotations,
        }

        # Process with image processor
        if self.processor is not None:
            results = self.processor(
                images=img, annotations=formatted_annotation, return_tensors="pt"
            )
            # Squeeze batch dimension
            results = {
                k: v.squeeze() if isinstance(v, torch.Tensor) else v[0]
                for k, v in results.items()
            }
            return results

        return img, formatted_annotation


class COCODataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for COCO format datasets."""

    def __init__(
        self,
        dataset_path: str,
        processor,
        batch_size: int = 16,
        num_workers: int = 4,
        model_input_size: int = 672,
        min_random_scale: float = 0.7,
        max_random_scale: float = 1.0,
        p_noise: float = 0.25,
        org_images_in_model_input_size: bool = True,
        config=None,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_input_size = model_input_size
        self.min_random_scale = min_random_scale
        self.max_random_scale = max_random_scale
        self.p_noise = p_noise
        self.org_images_in_model_input_size = org_images_in_model_input_size
        self.config = config

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _get_remap_dict(self, coco_dataset: CocoDetection) -> Optional[Dict[int, int]]:
        """Builds a dictionary to map dataset category IDs to model class IDs."""
        # 0. Check if remapping is enabled
        if hasattr(self.config, "remap_labels") and not self.config.remap_labels:
            return None

        # 1. Get target label map (Model IDs -> Class Names)
        if not hasattr(self.config, "model") or "label_map" not in self.config.model:
            return None

        target_label_map = self.config.model.label_map
        # Invert to (Class Name -> Model ID)
        name_to_target_id = {v: int(k) for k, v in target_label_map.items()}

        # 2. Get remapping rules (Source Name -> Target Name)
        remapping_rules = {}
        # Check various config locations
        if (
            hasattr(self.config, "data")
            and self.config.data
            and "class_remapping" in self.config.data
        ):
            remapping_rules = self.config.data.class_remapping
        elif "class_remapping" in self.config:
            remapping_rules = self.config.class_remapping

        if not remapping_rules:
            # If no remapping rules, just check if names match directly
            pass

        # 3. Build the map (Dataset Category ID -> Model ID)
        remap_dict = {}
        # coco_dataset.coco.cats is Dict[int, Dict]
        for cat_id, cat_info in coco_dataset.coco.cats.items():
            src_name = cat_info["name"]

            # Apply remapping if exists, otherwise keep original name
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

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage."""
        stage_str = str(stage).lower() if stage is not None else None
        is_fit_stage = (
            stage is None
            or stage == "fit"
            or (stage_str is not None and "fit" in stage_str)
        )
        is_test_stage = (
            stage is None
            or stage == "test"
            or (stage_str is not None and "test" in stage_str)
        )

        if is_fit_stage:
            if self.train_dataset is None or self.val_dataset is None:
                # Training dataset
                train_images_path = os.path.join(
                    self.dataset_path, "images", self.config.train_name
                )
                train_annot_path = os.path.join(
                    self.dataset_path, f"{self.config.train_name}_annotations.json"
                )

                if self.config.debug:
                    train_images_path = os.path.join(
                        self.dataset_path, "images", self.config.val_name
                    )
                    train_annot_path = os.path.join(
                        self.dataset_path, f"{self.config.val_name}_annotations.json"
                    )

                train_coco = CocoDetection(
                    root=train_images_path, annFile=train_annot_path, transforms=None
                )
                train_remap = self._get_remap_dict(train_coco)

                # Check for transforms config
                transforms_config_train = None
                if (
                    hasattr(self.config, "data")
                    and hasattr(self.config.data, "transforms")
                    and hasattr(self.config.data.transforms, "train")
                ):
                    transforms_config_train = OmegaConf.to_container(
                        self.config.data.transforms.train, resolve=True
                    )

                train_transforms = get_transform(
                    model_input_width=self.model_input_size,
                    model_input_height=self.model_input_size,
                    min_random_scale=self.min_random_scale,
                    max_random_scale=self.max_random_scale,
                    p_noise=self.p_noise,
                    org_images_in_model_input_size=self.org_images_in_model_input_size,
                    train=True,
                    transforms_config=transforms_config_train,
                )

                self.train_dataset = CocoDataset(
                    dataset_coco=train_coco,
                    processor=self.processor,
                    transforms=train_transforms,
                    remap_dict=train_remap,
                )

                # Validation dataset
                val_images_path = os.path.join(
                    self.dataset_path, "images", self.config.val_name
                )
                val_annot_path = os.path.join(
                    self.dataset_path, f"{self.config.val_name}_annotations.json"
                )

                val_coco = CocoDetection(
                    root=val_images_path, annFile=val_annot_path, transforms=None
                )

                val_remap = self._get_remap_dict(val_coco)

                transforms_config_test = None
                if (
                    hasattr(self.config, "data")
                    and hasattr(self.config.data, "transforms")
                    and hasattr(self.config.data.transforms, "test")
                ):
                    transforms_config_test = OmegaConf.to_container(
                        self.config.data.transforms.test, resolve=True
                    )

                val_transforms = get_transform(
                    model_input_width=self.model_input_size,
                    model_input_height=self.model_input_size,
                    min_random_scale=self.min_random_scale,
                    max_random_scale=self.max_random_scale,
                    p_noise=self.p_noise,
                    org_images_in_model_input_size=self.org_images_in_model_input_size,
                    train=False,
                    transforms_config=transforms_config_test,
                )

                self.val_dataset = CocoDataset(
                    dataset_coco=val_coco,
                    processor=self.processor,
                    transforms=val_transforms,
                    remap_dict=val_remap,
                )
                print(
                    "Training set includes %d annotated images."
                    % len(self.train_dataset)
                )
                print(
                    "Validation set includes %d annotated images."
                    % len(self.val_dataset)
                )

        if is_test_stage:
            if self.test_dataset is None:
                # Test dataset
                test_images_path = os.path.join(
                    self.dataset_path, "images", self.config.test_name
                )
                test_annot_path = os.path.join(
                    self.dataset_path, f"{self.config.test_name}_annotations.json"
                )
                if self.config.debug:
                    test_images_path = os.path.join(
                        self.dataset_path, "images", self.config.val_name
                    )
                    test_annot_path = os.path.join(
                        self.dataset_path, f"{self.config.val_name}_annotations.json"
                    )

                test_coco = CocoDetection(
                    root=test_images_path, annFile=test_annot_path, transforms=None
                )

                test_remap = self._get_remap_dict(test_coco)

                transforms_config_test = None
                if (
                    hasattr(self.config, "data")
                    and hasattr(self.config.data, "transforms")
                    and hasattr(self.config.data.transforms, "test")
                ):
                    transforms_config_test = OmegaConf.to_container(
                        self.config.data.transforms.test, resolve=True
                    )

                test_transforms = get_transform(
                    model_input_width=self.model_input_size,
                    model_input_height=self.model_input_size,
                    min_random_scale=self.min_random_scale,
                    max_random_scale=self.max_random_scale,
                    p_noise=self.p_noise,
                    org_images_in_model_input_size=self.org_images_in_model_input_size,
                    train=False,
                    transforms_config=transforms_config_test,
                )

                self.test_dataset = CocoDataset(
                    dataset_coco=test_coco,
                    processor=self.processor,
                    transforms=test_transforms,
                    remap_dict=test_remap,
                )
                print("Test set includes %d annotated images." % len(self.test_dataset))

    @staticmethod
    def collate_fn(batch):
        """Collate function for RT-DETR."""
        data = {}
        data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
        data["labels"] = [x["labels"] for x in batch]
        return data

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,  # Use batch size 1 for validation to match notebook
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        # Defensive fallback in case trainer did not trigger setup("test") as expected.
        if self.test_dataset is None:
            self.setup(stage="test")
        if self.test_dataset is None:
            raise RuntimeError(
                "test_dataset is not initialized. setup(stage='test') did not create it."
            )
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
