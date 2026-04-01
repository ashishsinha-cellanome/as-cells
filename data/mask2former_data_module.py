import json
import os
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset


def _resolve_image_path(image_root: Path, file_name: str) -> Path:
    candidate = image_root / file_name
    if candidate.exists():
        return candidate
    fallback = image_root / Path(file_name).name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Unable to resolve image '{file_name}' under '{image_root}'."
    )


def _decode_sidecar_annotation_mask(
    ann: dict[str, Any], image_height: int, image_width: int
) -> np.ndarray:
    rle = ann["segmentation"]
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded[..., 0]
    decoded = decoded.astype(np.uint8)

    if decoded.shape == (image_height, image_width):
        return decoded

    bbox = ann.get("bbox")
    if bbox is None or len(bbox) != 4:
        raise ValueError(
            "Sidecar mask bbox is required when segmentation is not full resolution."
        )

    x0, y0, x1, y1 = [int(v) for v in bbox]
    full_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    box_h = max(0, y1 - y0)
    box_w = max(0, x1 - x0)
    if box_h == 0 or box_w == 0:
        return full_mask

    if decoded.shape != (box_h, box_w):
        if decoded.shape[::-1] == (box_h, box_w):
            decoded = decoded.T
        else:
            raise ValueError(
                f"Decoded mask shape {decoded.shape} does not match bbox {(box_h, box_w)}."
            )

    full_mask[y0:y1, x0:x1] = decoded
    return full_mask


def _encode_binary_mask(mask: np.ndarray) -> dict[str, Any]:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": list(encoded["size"]), "counts": encoded["counts"]}


def build_segmentation_coco_gt(
    annotation_path: str | Path, mask_root: str | Path
) -> COCO:
    annotation_path = Path(annotation_path)
    mask_root = Path(mask_root)
    payload = json.loads(annotation_path.read_text())

    images = [dict(img) for img in payload.get("images", [])]
    categories = [dict(cat) for cat in payload.get("categories", [])]
    annotations: list[dict[str, Any]] = []
    next_ann_id = 1

    for image in images:
        image_id = int(image["id"])
        image_height = int(image["height"])
        image_width = int(image["width"])
        mask_path = mask_root / f"{Path(str(image['file_name'])).stem}.pkl"
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Missing sidecar mask for image_id={image_id}: {mask_path}"
            )

        with mask_path.open("rb") as fh:
            mask_payload = pickle.load(fh)

        for ann in mask_payload.get("annotations", []):
            full_mask = _decode_sidecar_annotation_mask(ann, image_height, image_width)
            if int(full_mask.sum()) == 0:
                continue

            encoded = _encode_binary_mask(full_mask)
            bbox = mask_utils.toBbox(encoded).tolist()
            area = float(mask_utils.area(encoded))

            annotations.append(
                {
                    "id": next_ann_id,
                    "image_id": image_id,
                    "category_id": int(ann["category_id"]),
                    "segmentation": encoded,
                    "bbox": [float(x) for x in bbox],
                    "area": area,
                    "iscrowd": 0,
                }
            )
            next_ann_id += 1

    coco_gt = COCO()
    coco_gt.dataset = {
        "images": images,
        "categories": categories,
        "annotations": annotations,
        "info": payload.get("info", {}),
    }
    coco_gt.createIndex()
    return coco_gt


class Mask2FormerDataset(Dataset):
    def __init__(
        self,
        annotation_path: str | Path,
        image_root: str | Path,
        mask_root: str | Path,
        image_processor,
    ):
        self.annotation_path = Path(annotation_path)
        self.image_root = Path(image_root)
        self.mask_root = Path(mask_root)
        self.image_processor = image_processor

        self.payload = json.loads(self.annotation_path.read_text())
        self.images = list(self.payload.get("images", []))
        self.categories = list(self.payload.get("categories", []))
        self.image_ids = [int(image["id"]) for image in self.images]
        self.image_id_to_info = {int(image["id"]): image for image in self.images}

    def __len__(self) -> int:
        return len(self.images)

    def _load_sidecar_annotations(self, image_info: dict[str, Any]) -> list[dict[str, Any]]:
        mask_path = self.mask_root / f"{Path(str(image_info['file_name'])).stem}.pkl"
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Missing sidecar mask for image_id={image_info['id']}: {mask_path}"
            )

        with mask_path.open("rb") as fh:
            payload = pickle.load(fh)

        annotations = payload.get("annotations")
        if not isinstance(annotations, list):
            raise ValueError(f"Unexpected sidecar payload structure in {mask_path}.")
        return annotations

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_info = self.images[idx]
        image_id = int(image_info["id"])
        image_height = int(image_info["height"])
        image_width = int(image_info["width"])
        image_path = _resolve_image_path(self.image_root, str(image_info["file_name"]))

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        sidecar_annotations = self._load_sidecar_annotations(image_info)

        instance_map = np.full((image_height, image_width), fill_value=-1, dtype=np.int64)
        instance_id_to_semantic_id: dict[int, int] = {}
        instance_id = 1

        for ann in sidecar_annotations:
            full_mask = _decode_sidecar_annotation_mask(ann, image_height, image_width)
            if int(full_mask.sum()) == 0:
                continue
            instance_map[full_mask > 0] = instance_id
            instance_id_to_semantic_id[instance_id] = int(ann["category_id"])
            instance_id += 1

        processed = self.image_processor(
            images=image_np,
            segmentation_maps=instance_map,
            instance_id_to_semantic_id=instance_id_to_semantic_id,
            return_tensors="pt",
        )

        return {
            "pixel_values": processed["pixel_values"].squeeze(0),
            "pixel_mask": processed["pixel_mask"].squeeze(0),
            "mask_labels": processed["mask_labels"][0],
            "class_labels": processed["class_labels"][0],
            "image_id": torch.tensor(image_id, dtype=torch.long),
            "orig_size": torch.tensor([image_height, image_width], dtype=torch.long),
            "file_name": str(image_info["file_name"]),
        }


class Mask2FormerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        image_processor,
        config,
    ):
        super().__init__()
        self.dataset_path = str(Path(dataset_path).expanduser().resolve())
        self.image_processor = image_processor
        self.config = config

        self.train_dataset: Optional[Mask2FormerDataset] = None
        self.val_dataset: Optional[Mask2FormerDataset] = None
        self.test_dataset: Optional[Mask2FormerDataset] = None

        self._val_coco_gt: Optional[COCO] = None
        self._test_coco_gt: Optional[COCO] = None
        self._val_segm_coco_gt: Optional[COCO] = None
        self._test_segm_coco_gt: Optional[COCO] = None

    def _annotation_path(self, split_name: str) -> Path:
        return Path(self.dataset_path) / f"{split_name}_annotations.json"

    def _image_root(self, split_name: str) -> Path:
        return Path(self.dataset_path) / "images" / split_name

    def _mask_root(self, split_name: str) -> Path:
        return Path(self.dataset_path) / "masks" / split_name

    def _build_dataset(self, split_name: str) -> Mask2FormerDataset:
        return Mask2FormerDataset(
            annotation_path=self._annotation_path(split_name),
            image_root=self._image_root(split_name),
            mask_root=self._mask_root(split_name),
            image_processor=self.image_processor,
        )

    def _load_coco(self, split_name: str) -> COCO:
        coco = COCO(str(self._annotation_path(split_name)))
        if "info" not in coco.dataset:
            coco.dataset["info"] = {}
        return coco

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            if self.train_dataset is None:
                self.train_dataset = self._build_dataset(self.config.train_name)
            if self.val_dataset is None:
                self.val_dataset = self._build_dataset(self.config.val_name)
            if self._val_coco_gt is None:
                self._val_coco_gt = self._load_coco(self.config.val_name)
            if self._val_segm_coco_gt is None:
                self._val_segm_coco_gt = build_segmentation_coco_gt(
                    self._annotation_path(self.config.val_name),
                    self._mask_root(self.config.val_name),
                )

        if stage in (None, "test"):
            if self.test_dataset is None:
                split_name = self.config.val_name if self.config.debug else self.config.test_name
                self.test_dataset = self._build_dataset(split_name)
            if self._test_coco_gt is None:
                split_name = self.config.val_name if self.config.debug else self.config.test_name
                self._test_coco_gt = self._load_coco(split_name)
            if self._test_segm_coco_gt is None:
                split_name = self.config.val_name if self.config.debug else self.config.test_name
                self._test_segm_coco_gt = build_segmentation_coco_gt(
                    self._annotation_path(split_name),
                    self._mask_root(split_name),
                )

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
            "pixel_mask": torch.stack([x["pixel_mask"] for x in batch]),
            "mask_labels": [x["mask_labels"] for x in batch],
            "class_labels": [x["class_labels"] for x in batch],
            "image_ids": torch.stack([x["image_id"] for x in batch]),
            "orig_sizes": torch.stack([x["orig_size"] for x in batch]),
            "file_names": [x["file_name"] for x in batch],
        }

    @property
    def val_coco_gt(self) -> Optional[COCO]:
        return self._val_coco_gt

    @property
    def test_coco_gt(self) -> Optional[COCO]:
        return self._test_coco_gt

    @property
    def val_segm_coco_gt(self) -> Optional[COCO]:
        return self._val_segm_coco_gt

    @property
    def test_segm_coco_gt(self) -> Optional[COCO]:
        return self._test_segm_coco_gt

    @property
    def val_image_root(self) -> str:
        return str(self._image_root(self.config.val_name))

    @property
    def test_image_root(self) -> str:
        split_name = self.config.val_name if self.config.debug else self.config.test_name
        return str(self._image_root(split_name))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.config.data.batch_size),
            shuffle=True,
            num_workers=int(self.config.data.num_workers),
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        eval_batch_size = int(getattr(self.config.data, "eval_batch_size", 1))
        return DataLoader(
            self.val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=int(self.config.data.num_workers),
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        eval_batch_size = int(getattr(self.config.data, "eval_batch_size", 1))
        return DataLoader(
            self.test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=int(self.config.data.num_workers),
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
