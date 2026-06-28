import torch
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
        self.train_name = getattr(config.data, "train_name", "train_new")

        self._args = None

    def _build_args(self):
        model_cfg = self.config.model.rfdetr
        data_cfg = self.config.data

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
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
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
        # Determine paths
        # If 'test' or 'val', look for 'images/test', else 'images/train'
        img_split = "test" if "test" in split_name else "train"
        image_root = self.base_path / ds_name / "images" / img_split
            
        ann_path = self.base_path / ds_name / f"{split_name}_annotations.json"

        # Only apply augmentations on train_new
        aug_config = self._load_aug_config() if split_name == self.train_name else None
        
        # Non-square path in upstream library has no explicit "test" branch, maps test->val
        image_set = "test" if split_name == self.test_name else ("val" if "val" in split_name or "valid" in split_name else "train")

        dataset = CocoDetection(
            img_folder=str(image_root),
            ann_file=str(ann_path),
            transforms=self._build_transforms(image_set, aug_config=aug_config),
            include_masks=True,
            remap_category_ids=False
        )
        
        # Fix category names dynamically
        is_neuron = 'neuron' in ds_name.lower()
        for cat in dataset.coco.dataset.get('categories', []):
            if not is_neuron and cat['name'] == 'soma':
                cat['name'] = 'cell-adhered'
            if is_neuron and cat['name'] == 'cell-adhered':
                cat['name'] = 'soma'
            if cat['name'] == 'Cell': cat['name'] = 'cell'
            if cat['name'] == 'cytoplasm': cat['name'] = 'cell-adhered'
            if cat['name'] == 'Bead' or cat['name'] == 'beads': cat['name'] = 'bead'
        dataset.coco.cats = {cat['id']: cat for cat in dataset.coco.dataset.get('categories', [])}
            
        return dataset

    def setup(self, stage=None):
        self._args = self._build_args()
        
        if stage in ("fit", None):
            self.train_datasets_objs = [self._make_dataset(ds, self.train_name) for ds in self.train_dataset_names]
            
            # Validation on training datasets (using 'valid' split generated offline)
            self.val_train_datasets_objs = [self._make_dataset(ds, self.val_name) for ds in self.train_dataset_names]
            
            # Validation on test datasets for zero-shot monitoring
            self.val_test_datasets_objs = [self._make_dataset(ds, self.val_name) for ds in self.test_dataset_names]
            
            self.concat_train = ConcatDataset(self.train_datasets_objs)
            
        if stage in ("test", None):
            self.test_datasets_objs = [self._make_dataset(ds, self.test_name) for ds in self.test_dataset_names]

    @property
    def args(self):
        return self._args

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
        
        dataloaders = []
        all_val_datasets = self.val_train_datasets_objs + self.val_test_datasets_objs
        for ds in all_val_datasets:
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
