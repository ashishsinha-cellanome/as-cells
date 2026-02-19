import sys
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf


class YOLOv5DataModule(pl.LightningDataModule):
    """LightningDataModule that wraps official YOLOv5 dataloaders."""

    def __init__(
        self,
        yolo_repo_path: str,
        dataset_root: str,
        config,
        stride: int,
        split_path_to_image_id: dict,
    ):
        super().__init__()
        self.yolo_repo_path = Path(yolo_repo_path).expanduser().resolve()
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.config = config
        self.stride = int(stride)
        self.split_path_to_image_id = split_path_to_image_id

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def _import_create_dataloader(self):
        if str(self.yolo_repo_path) not in sys.path:
            sys.path.insert(0, str(self.yolo_repo_path))
        from utils.dataloaders import create_dataloader  # type: ignore

        return create_dataloader

    def setup(self, stage=None):
        create_dataloader = self._import_create_dataloader()
        data_cfg = self.config.data
        model_cfg = self.config.model.yolov5
        hyp = OmegaConf.to_container(model_cfg.hyp, resolve=True)

        imgsz = int(self.config.model.input_size)
        batch_size = int(data_cfg.batch_size)
        workers = int(data_cfg.num_workers)
        single_cls = False
        rank = -1

        train_path = str(self.dataset_root / "images" / "train")
        val_path = str(self.dataset_root / "images" / "val")
        test_path = str(self.dataset_root / "images" / "test")

        if stage in ("fit", None):
            self.train_loader = create_dataloader(
                path=train_path,
                imgsz=imgsz,
                batch_size=batch_size,
                stride=self.stride,
                single_cls=single_cls,
                hyp=hyp,
                augment=True,
                cache=bool(model_cfg.cache_images),
                pad=0.0,
                rect=False,
                rank=rank,
                workers=workers,
                image_weights=False,
                quad=False,
                prefix="train: ",
                shuffle=True,
                seed=int(self.config.seed),
            )[0]

            self.val_loader = create_dataloader(
                path=val_path,
                imgsz=imgsz,
                batch_size=int(getattr(data_cfg, "eval_batch_size", 1)),
                stride=self.stride,
                single_cls=single_cls,
                hyp=hyp,
                augment=False,
                cache=bool(model_cfg.cache_images),
                pad=0.5,
                rect=True,
                rank=rank,
                workers=workers,
                image_weights=False,
                quad=False,
                prefix="val: ",
                shuffle=False,
                seed=int(self.config.seed),
            )[0]

        if stage in ("test", None):
            self.test_loader = create_dataloader(
                path=test_path,
                imgsz=imgsz,
                batch_size=int(getattr(data_cfg, "eval_batch_size", 1)),
                stride=self.stride,
                single_cls=single_cls,
                hyp=hyp,
                augment=False,
                cache=bool(model_cfg.cache_images),
                pad=0.5,
                rect=True,
                rank=rank,
                workers=workers,
                image_weights=False,
                quad=False,
                prefix="test: ",
                shuffle=False,
                seed=int(self.config.seed),
            )[0]

    def image_id_from_path(self, split_name: str, path: str):
        split_map = self.split_path_to_image_id.get(split_name, {})
        file_name = Path(path).name
        return split_map.get(file_name)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
