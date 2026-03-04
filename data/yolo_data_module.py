import os
import sys
import importlib
import torch
import pytorch_lightning as pl
from pathlib import Path
from pycocotools.coco import COCO
from tqdm import tqdm


def _import_from_yolo_repo(repo_path: str, module_name: str):
    """Import a module from the YOLOv5 repository using normal module resolution."""
    repo = Path(repo_path).expanduser().resolve()
    if not repo.exists():
        raise FileNotFoundError(f"YOLOv5 repo not found at: {repo}.")

    repo_str = str(repo)
    original_path = sys.path.copy()
    original_modules = {}

    try:
        sys.path = [p for p in sys.path if p not in ("", ".", str(Path.cwd()))]
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        for key in list(sys.modules.keys()):
            if key.startswith(("models", "utils", "detect", "export")):
                original_modules[key] = sys.modules.pop(key)

        module = importlib.import_module(module_name)
        return module
    except ImportError as e:
        raise ImportError(f"Could not import '{module_name}' from YOLOv5.\n{e}")
    finally:
        sys.path = original_path

class YOLOv5DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_config = config.data
        self.yolo_config = config.model.yolov5
        self.yolo_repo_path = os.path.abspath(self.yolo_config.repo_path)
        
        self.cache_dir = Path(self.yolo_config.dataset_cache_dir)
        self.yaml_path = self.cache_dir / "data.yaml"
        self._setup_called = False

    def _convert_coco_to_yolo(self, split_name, image_dir, json_path):
        split_img_dir = self.cache_dir / "images" / split_name
        split_lbl_dir = self.cache_dir / "labels" / split_name
        
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[INFO] Converting {split_name} COCO to YOLO format...")
        coco = COCO(json_path)
        
        # Mapping: COCO name -> YOLO 0-indexed integer
        name_to_id = {v: int(k) for k, v in self.config.model.label_map.items()}
        
        for img_id in tqdm(coco.getImgIds(), desc=f"Converting {split_name} COCO to YOLO format"):
            img_info = coco.loadImgs(img_id)[0]
            img_name = img_info['file_name']
            src_img_path = os.path.join(image_dir, img_name)
            dst_img_path = split_img_dir / img_name
            
            if not dst_img_path.exists() and os.path.exists(src_img_path):
                os.symlink(src_img_path, dst_img_path)
                
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            
            yolo_labels = []
            img_w, img_h = img_info['width'], img_info['height']
            
            for ann in anns:
                cat_info = coco.loadCats(ann['category_id'])[0]
                if cat_info['name'] not in name_to_id: 
                    continue
                    
                yolo_class_id = name_to_id[cat_info['name']]
                x, y, w, h = ann['bbox']
                if w == 0 or h == 0: 
                    continue
                
                x_c, y_c = (x + w / 2) / img_w, (y + h / 2) / img_h
                w_norm, h_norm = w / img_w, h / img_h
                yolo_labels.append(f"{yolo_class_id} {x_c:.6f} {y_c:.6f} {w_norm:.6f} {h_norm:.6f}")
                
            with open(split_lbl_dir / f"{Path(img_name).stem}.txt", "w") as f:
                f.write("\n".join(yolo_labels))

    def setup(self, stage=None):
        if self._setup_called: return
        
        if self.trainer.is_global_zero and not self.yaml_path.exists():
            dp = self.data_config.path
            self._convert_coco_to_yolo(self.data_config.train_name, os.path.join(dp, 'images', self.data_config.train_name), os.path.join(dp, f'{self.data_config.train_name}_annotations.json'))
            self._convert_coco_to_yolo(self.data_config.val_name, os.path.join(dp, 'images', self.data_config.val_name), os.path.join(dp, f'{self.data_config.val_name}_annotations.json'))
            self._convert_coco_to_yolo(self.data_config.test_name, os.path.join(dp, 'images', self.data_config.test_name), os.path.join(dp, f'{self.data_config.test_name}_annotations.json'))
            
            names = [self.config.model.label_map[k] for k in sorted(self.config.model.label_map.keys())]
            yaml_content = f"train: {self.cache_dir / 'images' / self.data_config.train_name}\nval: {self.cache_dir / 'images' / self.data_config.val_name}\ntest: {self.cache_dir / 'images' / self.data_config.test_name}\nnc: {len(names)}\nnames: {names}\n"
            with open(self.yaml_path, "w") as f: f.write(yaml_content)
        
        if self.trainer is not None and self.trainer.world_size > 1:
            self.trainer.strategy.barrier()
        # if torch.distributed.is_available() and torch.distributed.is_initialized():
        #     torch.distributed.barrier()

        # Safely import YOLO dataloader
        yolo_dataloaders = _import_from_yolo_repo(self.yolo_repo_path, "utils.dataloaders")
        create_dataloader = yolo_dataloaders.create_dataloader
        
        hyp = dict(self.yolo_config.hyp)
        imgsz = self.config.model.input_size
        bs, nw = self.data_config.batch_size, self.data_config.num_workers
        
        self.train_loader, _ = create_dataloader(
            str(self.cache_dir / 'images' / self.data_config.train_name),
            imgsz,
            bs,
            32, # stride
            single_cls=False, 
            hyp=hyp, 
            augment=True, 
            cache=self.yolo_config.cache_images, 
            rect=False, 
            rank=-1, 
            workers=nw, 
            image_weights=False, 
            quad=False, 
            prefix="[Train] ", 
            shuffle=True, 
            seed=self.config.seed)
        self.val_loader, _ = create_dataloader(
            str(self.cache_dir / 'images' / self.data_config.val_name),
            imgsz,
            1,
            32, # stride
            single_cls=False, 
            hyp=hyp, 
            augment=False, 
            cache=self.yolo_config.cache_images, 
            rect=True, 
            rank=-1, 
            workers=nw, 
            pad=0.0, # 0.5 
            prefix="[Val] ")
        self.test_loader, _ = create_dataloader(
            str(self.cache_dir / 'images' / self.data_config.test_name),
            imgsz,
            1,
            32, # stride
            single_cls=False, 
            hyp=hyp, 
            augment=False, 
            cache=self.yolo_config.cache_images, 
            rect=True, 
            rank=-1, 
            workers=nw, 
            pad=0., # 0.5
            prefix="[Test] ")
        self._setup_called = True

    def train_dataloader(self): return self.train_loader
    def val_dataloader(self): return self.val_loader
    def test_dataloader(self): return self.test_loader