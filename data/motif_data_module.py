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
            
        return dataset

    def setup(self, stage=None):
        self._args = self._build_args()
        
        if stage in ("fit", None):
            self.train_datasets_objs = [self._make_dataset(ds, self.train_name) for ds in self.train_dataset_names]
            
            # Merge all valid_annotations.json into one combined file
            import json
            import os
            import hashlib
            
            hash_str = hashlib.md5(str(self.train_dataset_names).encode()).hexdigest()[:8]
            merged_json_path = self.base_path / f"merged_val_annotations_{hash_str}.json"
            merged_data = {"images": [], "annotations": [], "categories": []}
            
            # Use predefined categories based on config
            if hasattr(self.config.model, 'label_map'):
                for k, v in self.config.model.label_map.items():
                    merged_data["categories"].append({"id": int(k), "name": v, "supercategory": "biology"})
                
            for ds_idx, ds_name in enumerate(self.train_dataset_names):
                val_json = self.base_path / ds_name / f"{self.val_name}_annotations.json"
                if not val_json.exists():
                    continue
                    
                with open(val_json, 'r') as f:
                    ds_data = json.load(f)
                        
                id_offset = ds_idx * 1000000
                
                img_split = "test" if "test" in self.val_name else "train"
                
                for img in ds_data.get('images', []):
                    img = img.copy()
                    img['id'] += id_offset
                    img['file_name'] = f"{ds_name}/images/{img_split}/{os.path.basename(img['file_name'])}"
                    merged_data['images'].append(img)
                    
                for ann in ds_data.get('annotations', []):
                    ann = ann.copy()
                    ann['id'] += id_offset
                    ann['image_id'] += id_offset
                    merged_data['annotations'].append(ann)
                        
            from utils.distributed_utils import get_rank
            import time
            if get_rank() == 0:
                tmp_json_path = str(merged_json_path) + f".tmp_{os.getpid()}"
                if not os.path.exists(merged_json_path):
                    with open(tmp_json_path, 'w') as f:
                        json.dump(merged_data, f)
                    try:
                        os.rename(tmp_json_path, str(merged_json_path))
                    except FileExistsError:
                        pass
                    except OSError:
                        pass
                    if os.path.exists(tmp_json_path):
                        try:
                            os.remove(tmp_json_path)
                        except OSError:
                            pass
            else:
                # Wait for rank 0 to finish writing atomically
                while not os.path.exists(merged_json_path):
                    time.sleep(0.5)
                
            # Create a single validation dataset obj
            from rfdetr.datasets.coco import CocoDetection
            aug_config = None
            self.val_train_datasets_objs = [CocoDetection(
                img_folder=str(self.base_path),
                ann_file=str(merged_json_path),
                transforms=self._build_transforms("val", aug_config=aug_config),
                include_masks=True,
                remap_category_ids=False
            )]
            
            # Validation on test datasets for zero-shot monitoring
            # self.val_test_datasets_objs = [self._make_dataset(ds, self.val_name) for ds in self.test_dataset_names]
            
            self.concat_train = ConcatDataset(self.train_datasets_objs)
            
        if stage in ("test", None):
            self.train_test_datasets_objs = [self._make_dataset(ds, self.test_name) for ds in self.train_dataset_names]
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
        eval_num_workers = min(self._args.num_workers, 8)
        
        dataloaders = []
        all_val_datasets = self.val_train_datasets_objs
        for ds in all_val_datasets:
            sampler = None
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=False)
                
            dl = DataLoader(
                ds,
                batch_size=eval_batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=eval_num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=False,
            )
            dataloaders.append(dl)
        return dataloaders

    def test_dataloader(self):
        eval_batch_size = int(getattr(self.config.data, "eval_batch_size", 1))
        eval_num_workers = min(self._args.num_workers, 8)
        
        dataloaders = []
        all_test_datasets = getattr(self, "train_test_datasets_objs", []) + getattr(self, "test_datasets_objs", [])
        for ds in all_test_datasets:
            sampler = None
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=False)
                
            dl = DataLoader(
                ds,
                batch_size=eval_batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=eval_num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=False,
            )
            dataloaders.append(dl)
        return dataloaders
