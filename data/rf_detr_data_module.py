import pytorch_lightning as pl
from pathlib import Path
from omegaconf import OmegaConf

from torch.utils.data import DataLoader

from rfdetr.datasets.coco import (
    CocoDetection,
    make_coco_transforms,
    make_coco_transforms_square_div_64,
)
from rfdetr.util.misc import collate_fn


class RFDETRDataModule(pl.LightningDataModule):
    """LightningDataModule backed by RF-DETR's native dataset pipeline."""

    def __init__(self, dataset_path: str, config, base_args=None):
        super().__init__()
        self.dataset_path = str(Path(dataset_path).expanduser().resolve())
        self.config = config
        self.base_args = base_args

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._args = None

    def _build_args(self):
        model_cfg = self.config.model.rfdetr
        data_cfg = self.config.data
        trainer_cfg = self.config.trainer

        if self.base_args is None:
            raise ValueError(
                "RFDETRDataModule requires `base_args` built from the active RF-DETR model context."
            )
        args = self.base_args

        args.dataset_file = "roboflow"
        args.dataset_dir = self.dataset_path
        args.resolution = int(self.config.model.input_size)
        args.batch_size = int(data_cfg.batch_size)
        args.num_workers = int(data_cfg.num_workers)
        args.epochs = int(trainer_cfg.max_epochs)
        args.num_classes = len(self.config.model.label_map)
        args.amp = str(trainer_cfg.precision).startswith("16")
        args.lr = float(self.config.optimizer.optimizer.lr)
        args.weight_decay = float(self.config.optimizer.optimizer.weight_decay)
        args.grad_accum_steps = int(trainer_cfg.accumulate_grad_batches)
        args.clip_max_norm = float(trainer_cfg.max_grad_norm)
        args.lr_scheduler = model_cfg.lr_scheduler
        args.warmup_epochs = float(model_cfg.warmup_epochs)
        args.lr_min_factor = float(model_cfg.lr_min_factor)
        args.multi_scale = bool(model_cfg.multi_scale)
        args.expanded_scales = bool(model_cfg.expanded_scales)
        args.do_random_resize_via_padding = bool(model_cfg.do_random_resize_via_padding)
        args.use_ema = bool(model_cfg.use_ema)
        args.ema_decay = float(model_cfg.ema_decay)
        args.ema_tau = float(model_cfg.ema_tau)
        args.square_resize_div_64 = bool(model_cfg.square_resize_div_64)
        args.segmentation_head = False
        args.run_test = True
        args.device = "cuda"
        if model_cfg.get("num_queries") is not None:
            args.num_queries = int(model_cfg.num_queries)
        if model_cfg.get("num_select") is not None:
            args.num_select = int(model_cfg.num_select)
        return args

    def _build_transforms(self, image_set: str, aug_config=None):
        args = self._args
        transform_fn = (
            make_coco_transforms_square_div_64
            if bool(args.square_resize_div_64)
            else make_coco_transforms
        )
        # Non-square path in upstream library has no explicit "test" branch.
        transform_image_set = (
            image_set
            if bool(args.square_resize_div_64)
            else ("val" if image_set == "test" else image_set)
        )
        return transform_fn(
            transform_image_set,
            args.resolution,
            multi_scale=args.multi_scale,
            expanded_scales=args.expanded_scales,
            skip_random_resize=not args.do_random_resize_via_padding,
            patch_size=args.patch_size,
            num_windows=args.num_windows,
            aug_config=aug_config,
        )

    def _load_aug_config(self):
        """Load RF-DETR augmentation config from rfdetr_transforms.yaml."""
        transforms_path = Path(__file__).parent.parent / "configs" / "data" / "rfdetr_transforms.yaml"
        if not transforms_path.exists():
            return None

        cfg = OmegaConf.load(transforms_path)
        if hasattr(cfg, "rfdetr_transforms"):
            return OmegaConf.to_container(cfg.rfdetr_transforms, resolve=True)
        return None

    def _make_dataset(self, split_name: str):
        image_root = Path(self.dataset_path) / "images" / split_name
        ann_path = Path(self.dataset_path) / f"{split_name}_annotations.json"

        # Load augmentation config for training split only
        aug_config = None
        if split_name == self.config.train_name:
            aug_config = self._load_aug_config()

        return CocoDetection(
            img_folder=image_root,
            ann_file=ann_path,
            transforms=self._build_transforms(
                "test"
                if split_name == self.config.test_name
                else ("val" if split_name == self.config.val_name else "train"),
                aug_config=aug_config,
            ),
            include_masks=False,
        )

    def setup(self, stage=None):
        self._args = self._build_args()
        if stage in ("fit", None):
            self.train_dataset = self._make_dataset(self.config.train_name)
            self.val_dataset = self._make_dataset(self.config.val_name)
        if stage in ("test", None):
            self.test_dataset = self._make_dataset(self.config.test_name)

    @property
    def args(self):
        return self._args

    @property
    def val_coco_gt(self):
        if self.val_dataset is None:
            return None
        return getattr(self.val_dataset, "coco", None)

    @property
    def test_coco_gt(self):
        if self.test_dataset is None:
            return None
        return getattr(self.test_dataset, "coco", None)

    @property
    def val_image_root(self):
        return str(Path(self.dataset_path) / "images" / self.config.val_name)

    @property
    def test_image_root(self):
        return str(Path(self.dataset_path) / "images" / self.config.test_name)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self._args.batch_size,
            shuffle=True,
            num_workers=self._args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        eval_batch_size = int(getattr(self.config.data, "eval_batch_size", 1))
        return DataLoader(
            self.val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=self._args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        eval_batch_size = int(getattr(self.config.data, "eval_batch_size", 1))
        return DataLoader(
            self.test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=self._args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False,
        )
