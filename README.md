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

## Advanced Embedding Distance Metrics

We calculate distances between cell line embedding distributions using several specialized metrics:

1. **Diagonal 2-Wasserstein Distance**: Measures shifts in the mean and variance. Useful as a general additive energy penalty.
2. **Symmetric KL Divergence (Jeffreys Divergence)**: Strictly penalizes cases where one distribution's variance collapses compared to the other. Ensures training bounds cover test edge cases.
3. **Diagonal Bhattacharyya Distance**: Measures sample overlap robustness to outliers using only diagonals.
4. **Hellinger Distance**: Derived from Bhattacharyya, but bounded between 0 and 1. Highly interpretable percentage of distribution mismatch.
5. **Full Bhattacharyya Distance**: Computes distribution overlap using the full covariance matrices (via Ledoit-Wolf shrinkage). Highly robust to outliers, considering both multidimensional spread and directional correlations.
6. **Mahalanobis Distance (Pooled Covariance)**: Computes the distance between means, scaled inversely by the combined (pooled) variance structure of the two datasets. Ideal for seeing how "far" datasets are relative to their natural biological variation.
7. **Sliced Wasserstein Distance (SWD)**: A non-parametric distance computed directly from empirical samples. It slices the high-dimensional space into 1D projections and calculates the Wasserstein distance, capturing complex non-Gaussian structural differences.
