import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
from rfdetr.datasets.coco import CocoDetection, make_coco_transforms, make_coco_transforms_square_div_64
from rfdetr.util.misc import collate_fn
from omegaconf import OmegaConf

class MotifDataModule(pl.LightningDataModule):
    def __init__(self, base_path: str, config, base_args=None):
        super().__init__()
        self.base_path = Path(base_path).expanduser().resolve()
        self.config = config
        self.base_args = base_args

        self.train_datasets_objs = []
        self.val_datasets_objs = []
        self.test_datasets_objs = []
        
        self.train_dataset_names = config.data.train_datasets
        self.test_dataset_names = config.data.test_datasets
        
        self.val_name = getattr(config.data, "val_name", "val")
        self.test_name = getattr(config.data, "test_name", "test")
        self.train_name = getattr(config.data, "train_name", "train")

        self._args = None

    def _build_args(self):
        model_cfg = self.config.model.rfdetr
        data_cfg = self.config.data
        trainer_cfg = self.config.trainer

        args = self.base_args if self.base_args else type('Args', (), {})()

        args.resolution = int(self.config.model.input_size)
        args.batch_size = int(data_cfg.batch_size)
        args.num_workers = int(data_cfg.num_workers)
        args.multi_scale = bool(model_cfg.get("multi_scale", False))
        args.expanded_scales = bool(model_cfg.get("expanded_scales", False))
        args.do_random_resize_via_padding = bool(model_cfg.get("do_random_resize_via_padding", False))
        args.square_resize_div_64 = bool(model_cfg.get("square_resize_div_64", False))
        args.patch_size = int(model_cfg.get("patch_size", 16))
        args.num_windows = int(model_cfg.get("num_windows", 4))
        return args

    def _build_transforms(self, image_set: str, aug_config=None):
        args = self._args
        transform_fn = make_coco_transforms_square_div_64 if args.square_resize_div_64 else make_coco_transforms
        transform_image_set = image_set if args.square_resize_div_64 else ("val" if image_set == "test" else image_set)
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
        transforms_path = Path(__file__).parent.parent / "configs" / "data" / "rfdetr_transforms.yaml"
        if not transforms_path.exists():
            return None
        cfg = OmegaConf.load(transforms_path)
        if hasattr(cfg, "rfdetr_transforms"):
            return OmegaConf.to_container(cfg.rfdetr_transforms, resolve=True)
        return None

    def _make_dataset(self, ds_name: str, split_name: str):
        image_root = self.base_path / ds_name / "images" / split_name
        ann_path = self.base_path / ds_name / f"{split_name}_annotations.json"

        aug_config = self._load_aug_config() if split_name == self.train_name else None
        
        # Non-square path in upstream library has no explicit "test" branch, maps test->val
        image_set = "test" if split_name == self.test_name else ("val" if split_name == self.val_name else "train")

        return CocoDetection(
            img_folder=str(image_root),
            ann_file=str(ann_path),
            transforms=self._build_transforms(image_set, aug_config=aug_config),
            include_masks=True,
            remap_category_ids=False
        )

    def setup(self, stage=None):
        self._args = self._build_args()
        
        if stage in ("fit", None):
            self.train_datasets_objs = [self._make_dataset(ds, self.train_name) for ds in self.train_dataset_names]
            self.val_datasets_objs = [self._make_dataset(ds, self.val_name) for ds in self.test_dataset_names]
            self.concat_train = ConcatDataset(self.train_datasets_objs)
            
        if stage in ("test", None):
            self.test_datasets_objs = [self._make_dataset(ds, self.test_name) for ds in self.test_dataset_names]

    def train_dataloader(self):
        return DataLoader(
            self.concat_train,
            batch_size=self._args.batch_size,
            shuffle=True,
            num_workers=self._args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        eval_batch_size = int(getattr(self.config.data, "eval_batch_size", 1))
        limit_val_batches = getattr(self.config.trainer, "limit_val_batches", 1.0)
        
        # If debugging, we might just return the first dataloader or slice them to avoid taking forever.
        dataloaders = []
        for ds in self.val_datasets_objs:
            dl = DataLoader(
                ds,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=self._args.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=False,
            )
            dataloaders.append(dl)
        return dataloaders

    def test_dataloader(self):
        eval_batch_size = int(getattr(self.config.data, "eval_batch_size", 1))
        dataloaders = []
        for ds in self.test_datasets_objs:
            dl = DataLoader(
                ds,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=self._args.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=False,
            )
            dataloaders.append(dl)
        return dataloaders
