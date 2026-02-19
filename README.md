# Object Detection and Instance Segmentation models for Cell Detection

This repository provides Jupyter Notebooks and python scripts for pre-processing the training data, training and running inference for the following models:
1) Object Detection: YOLOv5
2) Instance Segmentation: Mask R-CNN

## Training (PyTorch Lightning + Hydra)

Use model configs from `configs/model/`:

```bash
# RT-DETR (existing)
uv run train_rt_detr_v2.py model=rtdetr_base

# RF-DETR (new)
uv run train_rf_detr.py model=rf_detr_base

# YOLOv5 (new, requires official YOLOv5 source at model.yolov5.repo_path)
uv run train_yolov5.py model=yolov5_base
```
