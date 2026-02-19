import pytorch_lightning as pl
from torch.utils.data import DataLoader

from rfdetr.datasets import build_dataset
from rfdetr.main import populate_args
from rfdetr.util.misc import collate_fn


class RFDETRDataModule(pl.LightningDataModule):
    """LightningDataModule backed by RF-DETR's native dataset pipeline."""

    def __init__(self, dataset_dir: str, config, base_args=None):
        super().__init__()
        self.dataset_dir = dataset_dir
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
            args = populate_args()
        else:
            args = self.base_args

        args.dataset_file = "roboflow"
        args.dataset_dir = self.dataset_dir
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
        return args

    def setup(self, stage=None):
        self._args = self._build_args()
        if stage in ("fit", None):
            self.train_dataset = build_dataset("train", self._args, self._args.resolution)
            self.val_dataset = build_dataset("val", self._args, self._args.resolution)
        if stage in ("test", None):
            self.test_dataset = build_dataset("test", self._args, self._args.resolution)

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
