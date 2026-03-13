# Object Detection and Instance Segmentation models for Cell Detection

This repository provides Jupyter Notebooks and python scripts for pre-processing the training data, training and running inference for the following models:
1) Object Detection: YOLOv5
2) Instance Segmentation: Mask R-CNN

## Training (PyTorch Lightning + Hydra)

Use model configs from `configs/model/`:

```bash
# RT-DETR (existing)
uv run train_rt_detr_v2.py model=rtdetr_v1

# RF-DETR (new)
uv run train_rf_detr.py model=rf_detr

# YOLOv5 (new, requires official YOLOv5 source at model.yolov5.repo_path)
uv run train_yolov5.py model=yolov5
```

## Inference
Use model configs from `configs/model/`:

```bash
# RT-DETR (existing)
uv run train_rt_detr_v2.py model=rtdetr_v1 test_only=True initialization.load_from_checkpoint=/path/to/checkpoint.ckpt

# RF-DETR (new)
uv run train_rf_detr.py model=rf_detr test_only=True initialization.load_from_checkpoint=/path/to/checkpoint.ckpt

# YOLOv5 (new, requires official YOLOv5 source at model.yolov5.repo_path)
uv run train_yolov5.py model=yolov5 test_only=True initialization.load_from_checkpoint=/path/to/checkpoint.ckpt
```
