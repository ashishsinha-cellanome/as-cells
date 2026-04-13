# RF-DETR Segmentation Integration Design

## 1. Overview
The goal is to integrate the RF-DETR segmentation (`rfdetr-seg`) models into the existing training and evaluation pipeline. Instead of retrofitting the bbox script, we will create standalone scripts and modules specialized for segmentation.

## 2. Architecture & Components

### 2.1 Training Script (`train_rf_detr_seg.py`)
- We will duplicate `train_rf_detr.py` to `train_rf_detr_seg.py`.
- **Model Instantiation**: We will import `RFDETRSegSmall`, `RFDETRSegMedium`, and `RFDETRSegLarge` from the `rfdetr` package (or `rfdetr.models`).
- **Allowed Sizes**: We will strictly enforce that only `small`, `medium`, and `large` sizes are allowed. If the user passes `base` or any other size, we will raise a `ValueError`.
- **Inference Setup**: The script will correctly handle EMA inference based on the user's configuration, explicitly passing `config` to `_select_eval_weights_source`.

### 2.2 Lightning Module (`models/rf_detr_seg_lightning_module.py`)
- We will duplicate `models/rf_detr_lightning_module.py` to `models/rf_detr_seg_lightning_module.py`.
- **Metrics Computation**: 
  - Instead of standard bounding box mAP, we will compute segmentation mAP.
  - In `on_validation_epoch_end` and `on_test_epoch_end`, when calling `compute_coco_metrics`, we will set `iou_type="segm"` and `metric_prefix="segm"`.
  - The metric dictionary returned will be populated with keys like `segm_map`, `segm_map_50`, etc.
- **Monitoring**: The PyTorch Lightning `ModelCheckpoint` and `EarlyStopping` callbacks will be configured to monitor the segmentation metrics (e.g., `val/segm_map` or `val/segm_map_ema` if EMA is enabled).
- **Inference logic**: Both regular model and EMA model will be evaluated during validation and testing, exactly like the bounding box module.

### 2.3 Configuration (`configs/model/rfdetr_seg.yaml`)
- A new configuration file will be created based on `rfdetr.yaml`.
- The `size` parameter will be constrained conceptually to `small`, `medium`, and `large`.
- The checkpointing monitor metric will default to `val/segm_map` or `val/segm_map_ema`.
- EMA settings (`use_ema: true`, `ema_decay: 0.993`, `ema_tau: 100`) will be retained.

### 2.4 Utility Updates (`utils/test_only_checkpoint_restore.py`)
- We will update the `_select_eval_weights_source` function.
- It will explicitly look at `config.inference.use_ema` (defaulting to False).
- If `config.inference.use_ema=True`, the function will return `"ema"` to force loading the EMA state dict, even if the provided checkpoint path does not contain the word "ema".

## 3. Data Flow
1. Data loaded via `RFDETRDataModule`. (We will continue to use the same datamodule since COCO format supports both bbox and segmentation, and RF-DETR's data module handles it).
2. The `RFDETRSegLightningModule` processes the samples.
3. The loss dict from the model includes mask-related losses alongside detection losses.
4. `compute_coco_metrics` uses `pycocotools.cocoeval` with `iouType='segm'`.

## 4. Error Handling
- Invalid sizes (e.g., `base`) will trigger a descriptive `ValueError` during model instantiation.
- If `test_only=True` but no EMA weights exist inside a checkpoint when `use_ema=True`, an explicit error/warning will be raised.

## 5. Testing
- Testing mode (`test_only=True`) will use the provided checkpoint and evaluate either the regular or EMA weights based on `config.inference.use_ema` and the checkpoint path name.