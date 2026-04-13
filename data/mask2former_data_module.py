import json
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


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


def _process_single_image(image: dict, mask_root: Path) -> list[dict[str, Any]]:
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

    annotations = []
    for ann in mask_payload.get("annotations", []):
        full_mask = _decode_sidecar_annotation_mask(ann, image_height, image_width)
        if int(full_mask.sum()) == 0:
            continue

        encoded = _encode_binary_mask(full_mask)
        bbox = mask_utils.toBbox(encoded).tolist()
        area = float(mask_utils.area(encoded))

        annotations.append(
            {
                "image_id": image_id,
                "category_id": int(ann["category_id"]),
                "segmentation": encoded,
                "bbox": [float(x) for x in bbox],
                "area": area,
                "iscrowd": 0,
            }
        )
    return annotations


def build_segmentation_coco_gt(
    annotation_path: str | Path,
    mask_root: str | Path,
    cache_dir: Optional[str | Path] = None,
) -> COCO:
    annotation_path = Path(annotation_path)
    mask_root = Path(mask_root)

    if cache_dir is None:
        cache_dir = annotation_path.parent
    else:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    final_cache_path = cache_dir / f"{annotation_path.stem}_segm_coco_gt.pkl"

    if final_cache_path.exists():
        t0 = time.time()
        print(
            f"\n[Data] Loading cached segmentation COCO GT from {final_cache_path.name}..."
        )
        with final_cache_path.open("rb") as fh:
            coco_gt = pickle.load(fh)
        print(f"[Data] Cached COCO GT loaded in {time.time() - t0:.1f}s\n")
        return coco_gt

    t0 = time.time()
    file_size_gb = annotation_path.stat().st_size / 1e9
    print(f"\n[Data] Loading {annotation_path.name} ({file_size_gb:.2f} GB)...")
    payload = json.loads(annotation_path.read_text())
    print(f"[Data] Loaded JSON in {time.time() - t0:.1f}s")

    images = [dict(img) for img in payload.get("images", [])]
    categories = [dict(cat) for cat in payload.get("categories", [])]

    print(
        f"[Data] Processing masks for {len(images)} images from {mask_root.name} in chunks..."
    )

    # Process in chunks to prevent memory bloat
    chunk_size = 5000
    chunk_files = []

    for i in range(0, len(images), chunk_size):
        chunk_images = images[i : i + chunk_size]
        chunk_idx = i // chunk_size
        chunk_cache_path = (
            cache_dir / f"{annotation_path.stem}_segm_chunk_{chunk_idx:04d}.pkl"
        )

        if chunk_cache_path.exists():
            print(f"[Data] Found cached chunk {chunk_idx} ({len(chunk_images)} images)")
            chunk_files.append(chunk_cache_path)
            continue

        print(f"[Data] Processing chunk {chunk_idx} ({len(chunk_images)} images)...")
        chunk_annotations: list[dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(_process_single_image, img, mask_root): img
                for img in chunk_images
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Loading chunk {chunk_idx}",
            ):
                chunk_annotations.extend(future.result())

        # Save chunk to disk to free memory
        with chunk_cache_path.open("wb") as fh:
            pickle.dump(chunk_annotations, fh)
        chunk_files.append(chunk_cache_path)

        # Explicitly free memory
        del chunk_annotations
        del futures

    print("[Data] Merging chunks and building COCO object index...")
    t0 = time.time()

    # Load chunks sequentially and assign global IDs
    all_annotations = []
    global_ann_id = 1

    for chunk_file in chunk_files:
        with chunk_file.open("rb") as fh:
            chunk_annotations = pickle.load(fh)

        for ann in chunk_annotations:
            ann["id"] = global_ann_id
            global_ann_id += 1
            all_annotations.append(ann)

        # Free memory of chunk
        del chunk_annotations

    coco_gt = COCO()
    coco_gt.dataset = {
        "images": images,
        "categories": categories,
        "annotations": all_annotations,
        "info": payload.get("info", {}),
    }
    coco_gt.createIndex()

    # Save the final compiled COCO object
    with final_cache_path.open("wb") as fh:
        pickle.dump(coco_gt, fh)

    # Clean up temporary chunk files
    for chunk_file in chunk_files:
        if chunk_file.exists():
            chunk_file.unlink()

    print(f"[Data] Index built and cached in {time.time() - t0:.1f}s\n")
    return coco_gt


class Mask2FormerDataset(Dataset):
    def __init__(
        self,
        annotation_path: str | Path,
        image_root: str | Path,
        mask_root: str | Path,
        image_processor,
        transforms=None,
    ):
        self.annotation_path = Path(annotation_path)
        self.image_root = Path(image_root)
        self.mask_root = Path(mask_root)
        self.image_processor = image_processor
        self.transforms = transforms

        import time

        t0 = time.time()

        # Check for lightweight cache first
        cache_path = self.annotation_path.with_name(
            f"{self.annotation_path.stem}_lightweight.pkl"
        )

        if cache_path.exists():
            print(
                f"[Dataset] Loading lightweight cached annotations from {cache_path.name}..."
            )
            with cache_path.open("rb") as fh:
                cached_data = pickle.load(fh)
            self.images = cached_data["images"]
            self.categories = cached_data["categories"]
        else:
            file_size_gb = self.annotation_path.stat().st_size / 1e9
            print(
                f"[Dataset] Loading annotations from {self.annotation_path.name} ({file_size_gb:.2f} GB)..."
            )
            self.payload = json.loads(self.annotation_path.read_text())

            self.images = list(self.payload.get("images", []))
            self.categories = list(self.payload.get("categories", []))

        self.image_ids = [int(image["id"]) for image in self.images]
        self.image_id_to_info = {int(image["id"]): image for image in self.images}
        print(
            f"[Dataset] Init complete in {time.time() - t0:.1f}s ({len(self.images)} images)"
        )

    def __len__(self) -> int:
        return len(self.images)

    def _load_sidecar_annotations(
        self, image_info: dict[str, Any]
    ) -> list[dict[str, Any]]:
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

        masks = []
        semantic_ids = []
        for ann in sidecar_annotations:
            full_mask = _decode_sidecar_annotation_mask(ann, image_height, image_width)
            if int(full_mask.sum()) == 0:
                continue
            masks.append(full_mask)
            semantic_ids.append(int(ann["category_id"]))

        # Apply spatial/color augmentations
        if self.transforms is not None and len(masks) > 0:
            transformed = self.transforms(image=image_np, masks=masks)
            image_np = transformed["image"]
            masks = transformed["masks"]
        elif self.transforms is not None and len(masks) == 0:
            transformed = self.transforms(image=image_np)
            image_np = transformed["image"]

        # Recompute image dimensions post-transform
        image_height, image_width = image_np.shape[:2]
        instance_map = np.full(
            (image_height, image_width), fill_value=-1, dtype=np.int64
        )
        instance_id_to_semantic_id: dict[int, int] = {}
        instance_id = 1

        for mask, sem_id in zip(masks, semantic_ids):
            if int(mask.sum()) == 0:
                continue
            instance_map[mask > 0] = instance_id
            instance_id_to_semantic_id[instance_id] = sem_id
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

    def _build_transforms(self, split_name: str):
        import albumentations as A
        from omegaconf import OmegaConf
        from utils.dataset_utils import build_transforms_from_config

        is_train = split_name == getattr(self.config, "train_name", "train")

        transforms_path = Path("configs/data/mask2former_transforms.yaml")
        transform_cfg = None

        if transforms_path.exists():
            cfg = OmegaConf.load(transforms_path)
            if hasattr(cfg, "transforms"):
                transform_cfg = (
                    cfg.transforms.train if is_train else cfg.transforms.test
                )
                if transform_cfg:
                    # Resolve model input size since OmegaConf.load doesn't have global context
                    raw_yaml = OmegaConf.to_yaml(transform_cfg)
                    raw_yaml = raw_yaml.replace(
                        "${model.input_size}", str(self.config.model.input_size)
                    )
                    transform_cfg = OmegaConf.create(raw_yaml)

        if not transform_cfg:
            if not hasattr(self.config, "data") or not hasattr(
                self.config.data, "transforms"
            ):
                return None
            transform_cfg = (
                self.config.data.transforms.train
                if is_train
                else self.config.data.transforms.test
            )

        if not transform_cfg:
            return None

        trsfms_list = OmegaConf.to_container(transform_cfg, resolve=True)
        albumentation_list = build_transforms_from_config(trsfms_list)

        # For Mask2Former we only need masks and images to be transformed (no bboxes required)
        return A.Compose(albumentation_list)

    def _build_dataset(self, split_name: str) -> Mask2FormerDataset:
        return Mask2FormerDataset(
            annotation_path=self._annotation_path(split_name),
            image_root=self._image_root(split_name),
            mask_root=self._mask_root(split_name),
            image_processor=self.image_processor,
            transforms=self._build_transforms(split_name),
        )

    def _load_coco(self, split_name: str) -> COCO:
        import time

        ann_path = self._annotation_path(split_name)
        t0 = time.time()
        file_size_gb = ann_path.stat().st_size / 1e9
        print(f"\n[COCO API] Loading {ann_path.name} ({file_size_gb:.2f} GB)...")
        coco = COCO(str(ann_path))
        print(f"[COCO API] Loaded in {time.time() - t0:.1f}s")
        if "info" not in coco.dataset:
            coco.dataset["info"] = {}
        return coco

    def prepare_data(self):
        """
        Runs once per node to safely cache the dataset indexes and preprocess segmentation ground truth
        without race conditions from multiple GPUs.
        """
        # Cache lightweight JSONs for training, val, and test splits
        splits = [
            self.config.train_name,
            self.config.val_name,
            self.config.test_name
            if not getattr(self.config, "debug", False)
            else self.config.val_name,
        ]

        for split_name in set(splits):
            ann_path = self._annotation_path(split_name)
            cache_path = ann_path.with_name(f"{ann_path.stem}_lightweight.pkl")

            if not cache_path.exists() and ann_path.exists():
                print(f"[prepare_data] Building lightweight cache for {split_name}...")
                with ann_path.open("r") as f:
                    payload = json.load(f)

                # We only need images and categories for Mask2FormerDataset initialization
                lightweight_data = {
                    "images": list(payload.get("images", [])),
                    "categories": list(payload.get("categories", [])),
                }

                with cache_path.open("wb") as f:
                    pickle.dump(lightweight_data, f)
                print(f"[prepare_data] Saved lightweight cache to {cache_path.name}")

        # Precompute the massive segmentation COCO objects for Val & Test splits
        eval_splits = [
            self.config.val_name,
            self.config.test_name
            if not getattr(self.config, "debug", False)
            else self.config.val_name,
        ]

        for split_name in set(eval_splits):
            ann_path = self._annotation_path(split_name)
            mask_root = self._mask_root(split_name)

            # Since build_segmentation_coco_gt uses caching internally now,
            # calling it here guarantees the cache will be generated once.
            if ann_path.exists() and mask_root.exists():
                print(
                    f"[prepare_data] Ensuring segmentation GT cache for {split_name}..."
                )
                build_segmentation_coco_gt(ann_path, mask_root)

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            if self.train_dataset is None:
                self.train_dataset = self._build_dataset(self.config.train_name)
            if self.val_dataset is None:
                self.val_dataset = self._build_dataset(self.config.val_name)
            # COCO objects are now lazy-loaded via properties

        if stage in (None, "test"):
            if self.test_dataset is None:
                split_name = (
                    self.config.val_name
                    if getattr(self.config, "debug", False)
                    else self.config.test_name
                )
                self.test_dataset = self._build_dataset(split_name)
            # COCO objects are now lazy-loaded via properties

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
        if self._val_coco_gt is None:
            self._val_coco_gt = self._load_coco(self.config.val_name)
        return self._val_coco_gt

    @property
    def test_coco_gt(self) -> Optional[COCO]:
        if self._test_coco_gt is None:
            split_name = (
                self.config.val_name
                if getattr(self.config, "debug", False)
                else self.config.test_name
            )
            self._test_coco_gt = self._load_coco(split_name)
        return self._test_coco_gt

    @property
    def val_segm_coco_gt(self) -> Optional[COCO]:
        if self._val_segm_coco_gt is None:
            self._val_segm_coco_gt = build_segmentation_coco_gt(
                self._annotation_path(self.config.val_name),
                self._mask_root(self.config.val_name),
            )
        return self._val_segm_coco_gt

    @property
    def test_segm_coco_gt(self) -> Optional[COCO]:
        if self._test_segm_coco_gt is None:
            split_name = (
                self.config.val_name
                if getattr(self.config, "debug", False)
                else self.config.test_name
            )
            self._test_segm_coco_gt = build_segmentation_coco_gt(
                self._annotation_path(split_name),
                self._mask_root(split_name),
            )
        return self._test_segm_coco_gt

    @property
    def val_image_root(self) -> str:
        return str(self._image_root(self.config.val_name))

    @property
    def test_image_root(self) -> str:
        split_name = (
            self.config.val_name if self.config.debug else self.config.test_name
        )
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
