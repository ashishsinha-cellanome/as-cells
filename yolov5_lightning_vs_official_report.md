## Addendum: `vulcan` vs `coco128` (Config + Result Differences)

This section compares expected configuration and result differences between:
1. `train_yolov5.py` with `data=vulcan` and `model=yolov5`
2. `models/yolov5/train.py` with `--data data/coco128.yaml`

### Dataset and Label Space Mismatch
- **Dataset roots and formats differ.**  
  `data=vulcan` points to `/project/.../DATA/TRAINING_DATA` with COCO JSON annotations and a custom 4‑class label map. See `configs/data/vulcan.yaml` and `data/yolov5_data_module.py`.  
  `coco128.yaml` uses the COCO128 subset (128 images) and YOLO txt labels under `../datasets/coco128`. See `models/yolov5/data/coco128.yaml`.
- **Class count and taxonomy are not comparable.**  
  Vulcan uses 4 classes (`cell`, `bead`, `cell-adhered`, `soma`) from `configs/model/yolov5.yaml`. COCO128 uses 80 classes in `models/yolov5/data/coco128.yaml`. This changes the detection head shape, loss balance, and mAP scale.

### Config and Hyperparameter Defaults That Diverge
- **Training schedule defaults differ.**  
  Lightning defaults to `max_epochs: 50` in `configs/trainer/default.yaml`.  
  Official `train.py` defaults to `--epochs 100` in `models/yolov5/train.py`.
- **Hyperparameter source differs.**  
  Lightning pulls hyp from `configs/model/yolov5.yaml` (`model.yolov5.hyp`).  
  Official `train.py` defaults to `data/hyps/hyp.scratch-low.yaml`. See `models/yolov5/data/hyps/hyp.scratch-low.yaml`.
- **Optimizer/scheduler behavior differs.**  
  Lightning uses `configure_optimizers` in `models/yolov5_lightning_module.py` and scheduler settings from `configs/scheduler/cosine_warmup.yaml`.  
  Official uses its own warmup, accumulation, weight‑decay scaling, and `LambdaLR` schedule inside `models/yolov5/train.py`.
- **Inference thresholds differ by default.**  
  Lightning uses `detection_threshold: 0.001`, `iou_threshold: 0.2`, `max_detections: 100` from `configs/model/yolov5.yaml`.  
  Official val defaults are `conf_thres=0.001`, `iou_thres=0.6`, `max_det=300` in `models/yolov5/val.py`.

### Data Pipeline Differences
- **Augmentations are fundamentally different.**  
  Lightning uses Albumentations pipelines defined in `utils/dataset_utils.py` and configured by `configs/data/transforms.yaml` (random scale, pad/crop, perspective, brightness, hue, noise).  
  Official YOLOv5 uses mosaic/mixup, random perspective, HSV, flips, etc. in `models/yolov5/utils/dataloaders.py` and `models/yolov5/utils/augmentations.py`.
- **Letterbox and label normalization differ.**  
  Lightning’s `_letterbox` is implemented in `data/yolov5_data_module.py`, and labels are produced from COCO xywh then normalized after letterbox.  
  Official uses `letterbox` in `models/yolov5/utils/augmentations.py`, and label flow is normalized YOLO txt → pixel xyxy → augmentation → normalized xywh in `models/yolov5/utils/dataloaders.py`.

### Evaluation Differences (mAP Not Directly Comparable)
- **Metric backend differs.**  
  Lightning uses `pycocotools` COCOeval in `utils/coco_eval_utils.py`, with `max_detections` from config.  
  Official uses its own validation flow in `models/yolov5/val.py` with different default NMS and `max_det`.
- **Even identical predictions can score differently** due to different NMS thresholds, max detections, and COCOeval params.

### Why Results Differ (Short Summary)
Your `train_yolov5.py` + `data=vulcan` run and the official `train.py` + `coco128.yaml` run are not comparable because they differ in dataset, label space, augmentation pipeline, hyperparameter source, scheduler/warmup logic, and evaluation thresholds/metrics. Even if the same model weights were used, the reported mAP would change due to different validation settings and COCOeval parameters.
