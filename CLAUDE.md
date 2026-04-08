# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The project focuses on improving the detection of cells and other small and tiny objects in brightfield microscopy images.
Uses pytorch lightning, pytorch for training models, hydra for configuration management, wandb for experiment tracking, and uv for package management. The main models implemented are RT-DETR v1/v2, RF-DETR, and YOLOv5 (both Lightning wrapper and official Ultralytics training). The dataset is in COCO format with custom splits (including no300 variants that exclude images with >300 bboxes). Evaluation uses COCO metrics via pycocotools, with support for sliced inference via SAHI. SLURM is used for cluster job scheduling.

## Common Commands

### Package Management (UV)
```bash
uv sync              # Install dependencies from uv.lock
uv add <package>     # Add new dependency
uv run python <script>.py  # Run script with virtualenv
```

### Training
```bash
# RT-DETR v2 (DINOv2 backbone)
uv run train_rt_detr_v2.py model=rtdetr_v2 data=vulcan

# RF-DETR
uv run train_rf_detr.py model=rfdetr data=vulcan model.rfdetr.size=base

# YOLOv5 (Lightning wrapper, requires YOLOv5 repo at models/yolov5)
uv run train_yolov5.py model=yolov5 data=vulcan model.yolov5.yolo_size=m

# YOLOv5 (official Ultralytics training in models/yolov5/)
uv run models/yolov5/train.py --weights yolov5m.pt --img 640 --data .cache/datasets/yolov5_train_valid_test/data.yaml

# Multi-run sweeps
uv run train_rt_detr_v2.py --multirun model=rtdetr_v1,rtdetr_v2 optimizer.optimizer.lr=5e-4,5e-5

# Debug mode
uv run train_rt_detr_v2.py debug=true data.limit_train_batches=1 data.limit_val_batches=1

# Resume from checkpoint
uv run train_rt_detr_v2.py initialization.load_from_checkpoint=/path/to/ckpt.pt
```

### Data Split Generation (no300 variants)
```bash
# Create filtered datasets excluding images with >300 bboxes
# - valid_no300: val split with high-bbox images removed
# - test_no300: test split with high-bbox images removed
# - train_plus_valgt300: train + promoted val images (>300 bboxes)
uv run create_no300_data_splits.py data=vulcan
```

### SLURM Job Submission
```bash
# Submit RF-DETR jobs (array 0-15, sweeps lr/size/scheduler/data)
sbatch run_rfdetr.sh

# Submit RT-DETR jobs (array 0-7, sweeps model/backbone/scheduler)
sbatch run_rtdetrv1.sh

# Submit YOLOv5 jobs (Lightning, array 0-7)
sbatch submit_yolo_jobs.sh

# Submit YOLOv5 jobs (Ultralytics official, array 0)
sbatch run_yolo.sh
```

### Inference
```bash
uv run inference.py initialization.load_from_checkpoint=checkpoint.pt data.path=/path/to/data
```

### Evaluation
```bash
uv run evaluate_all_models.py  # Evaluate all models in config.py
```

### Checkpoint Conversion (Lightning .ckpt → Ultralytics .pt)
```bash
# Convert Lightning checkpoint for use with official YOLOv5 val.py
uv run convert_yolov5_pl_ckpt_to_pt.py --ckpt /path/to/yolov5-epoch-**.ckpt
```

### Linting
```bash
uv run ruff check .      # Run linter
uv run ruff check . --fix  # Auto-fix
```

## Architecture Overview

### Configuration System (Hydra)

```
configs/
├── config.yaml           # Main: selects defaults for each group
├── model/                # Model configs (rtdetr_v1/v2, rfdetr, yolov5)
│   └── backbone/         # Backbones (dinov2, resnet50/18/34/101)
├── data/                 # Data configs
│   ├── vulcan.yaml       # Standard vulcan paths
│   ├── vulcan_no300_eval.yaml           # val/test with >300 bbox images removed
│   └── vulcan_no300_eval_train_plus_valgt300.yaml  # + promoted images in train
├── optimizer/            # AdamW, SGD with param groups
├── scheduler/            # cosine_warmup, step, onecycle, multistep
├── trainer/              # Accelerator, devices, precision
├── checkpointing/        # Save dirs, monitor metrics
└── logging/              # WandB settings
```

**Key patterns:**
- `@package _global_` in data configs for path overrides
- `${oc.eval:...}` for eval expressions in configs
- `${hydra:runtime.cwd}` for absolute paths
- Hydra overrides: `key=value` syntax, dot notation for nested

### Data Pipeline

```
Raw Images (12/14-bit) → 8-bit JPEG → COCO Annotations → COCODataModule/RFDETRDataModule/YOLOv5DataModule
```

**Data configs** define:
- `path`: Base dataset directory (`/project/aip-robsc/asinha/cellanome/DATA/TRAINING_DATA`)
- `val_name`, `test_name`, `train_name`: Split names
- `limit_*_batches`: For debugging

**no300 data variants** (created by `create_no300_data_splits.py`):
- Threshold: excludes images with bbox_count > 300
- Valid: 52 images excluded (all have 483-541 bboxes)
- Test: 4 images excluded (302-303 bboxes)
- Excluded val images promoted to training set (`train_plus_valgt300`)

**Magnification-dependent sizes:**
- 10x: 2000x1600 pixels
- 4x: 4512x4512 pixels

### Model Architectures

**Training entry points:**
- `train_rt_detr_v2.py` - RT-DETR v1/v2 with ResNet or DINOv2 backbones
- `train_rf_detr.py` - RF-DETR (nano/small/medium/base/large)
- `train_yol5.py` - YOLOv5 Lightning wrapper (m/l variants)
- `models/yolov5/train.py` - Official Ultralytics YOLOv5 training

**Lightning modules:**
- `models/rt_detr_lightning_module.py`
- `models/rf_detr_lightning_module.py`
- `models/yolov5_lightning_module.py`

**Supported model keys:**
- `rtdetr_v1`, `rtdetr_v2`, `rfdetr`, `yolov5`
- Backbones: `dinov2`, `resnet50`, `resnet18`, `resnet34`, `resnet101`

### Class Mappings

All models use same label map (defined in model configs):

| Model | Classes (ID: name) |
|-------|-------------------|
| All | 0:cell, 1:bead, 2:cell-adhered, 3:soma |

**Label remapping** via `data.class_remapping`:
- `cell-adhered` → `cell`
- `soma` → `cell`

### Key Paths

```python
DATA_DIR = "/project/aip-robsc/asinha/cellanome/DATA/TRAINING_DATA"
CKPT_DIR = "/project/aip-robsc/asinha/cellanome/DATA/checkpoints/"
LOGS_DIR = "/project/aip-robsc/asinha/cellanome/logs/"
DINOv2_CKPT = "/project/aip-robsc/asinha/cellanome/DATA/checkpoints/backbones/dinov2"
```

### EMA (Exponential Moving Average)

All models support EMA:
- RT-DETR: `decay=0.999`, `tau=2000`
- RF-DETR: `decay=0.993`, `tau=100`
- YOLOv5: `decay=0.9999`, `tau=2000`

Checkpoint naming:
- Regular: `rtdetr-epoch{epoch:02d}-val_map{...:.4f}`
- EMA: `rtdetr-ema-{epoch:02d}-val_map_ema{...:.4f}`

### SLURM Environment

Training scripts auto-detect SLURM:
- Sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE` from env vars
- Derives `MASTER_PORT` from `SLURM_JOB_ID`
- Uses `scontrol` for `MASTER_ADDR`

**Job arrays** sweep hyperparameters:
- Learning rates: `5e-4`, `5e-5`, `1e-5` (RF-DETR), `0.01`, `0.001` (YOLOv5)
- Model sizes: `nano`, `small`, `medium`, `base`, `large`
- Schedulers: `cosine_warmup`, `step`, `onecycle`
- Data configs: `vulcan`, `vulcan_no300_eval_train_plus_valgt300`

### Evaluation

Uses COCO metrics via `pycocotools`:
- mAP, mAP_50, mAP_75, mAP_small/medium/large
- Precision-recall curves in `utils/precision_recall_eval.py`
- Custom sliced inference via SAHI (config: `eval_inference.sahi`)

**YOLOv5 evaluation comparison:**
- Lightning val: `pycocotools` COCOeval, `iou_threshold=0.2`, `max_detections=100`
- Official YOLOv5 val: custom val.py, `iou_thres=0.6`, `max_det=300`
- Results not directly comparable due to different NMS/metric settings

**Performance degradation investigation (YOLOv5m):**
- Issue: YOLOv5m shows degraded val performance when trained with Lightning vs Ultralytics
- Root causes being investigated:
  - Different augmentation pipelines (Albumentations vs YOLO mosaic/mixup)
  - Different label normalization flows
  - Different optimizer/scheduler implementations
  - Different evaluation thresholds
- See `yolov5_lightning_vs_official_report.md` for detailed analysis

## Project Conventions

### Dataset Structure
```
<DATA_DIR>/
└── <date>_<cell-type>_<magnification>_<condition>/
    ├── annotations/
    ├── test_images/
    └── test_annotations/
```

### Environment Variables
```bash
NCCL_P2P_DISABLE=1      # Disabled for stability
NCCL_IB_DISABLE=1       # Disabled for stability
RANK, LOCAL_RANK, WORLD_SIZE  # Distributed training
MASTER_ADDR, MASTER_PORT      # Distributed coordination
```

### Key Parameters
- `MIN_MASK_AREA=16` - Minimum object mask area
- `RESIZED_BB_IMAGE_SIZE=3840` - Training input size
- `MAX_IMAGE_SIDE=4512` - Maximum dimension
- `detection_threshold=0.05` - Detection threshold
- `draw_threshold=0.4` - Visualization threshold

### Integration Points
- **W&B**: Experiment tracking, model artifacts
- **AWS S3**: Data storage
- **Darwin V7**: Annotation platform
- **Multi-backend inference**: TensorRT, OpenVINO, ONNX (`utils/benchmark_utils.py`)
