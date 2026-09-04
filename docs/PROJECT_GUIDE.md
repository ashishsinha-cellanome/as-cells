# Cellanome Project Guide

## 1. Project Description
This repository contains a comprehensive pipeline for evaluating and deploying transformer-based object detection and instance segmentation models for automated cellular analysis (Cellanome). The primary goal is to accurately detect and segment cells, beads, cell-adhered instances, and soma across high-throughput microscopy imagery. 

The project operates across three phases:
- **Phase 1: Model Benchmarking** - Evaluating DETR-style detectors against a YOLOv5 baseline to find the best architecture for accuracy and zero-shot generalization. **YOLOv5** (for detection speed) and **RF-DETR-Seg** (for combined detection/segmentation and zero-shot capabilities) emerged as the top models.
- **Phase 2: Coverage-Based Dataset Selection** - Addressing zero-shot performance drops on novel cell lines by identifying a minimal, maximally-covering training set using a metric called "Coverage Distance".
- **Phase 3: LoRA-Based Domain Adaptation** - Using Parameter-Efficient Fine-Tuning (PEFT) via Low-Rank Adaptation (LoRA) to adapt the model to new, extreme morphological outlier cell lines with minimal data (as low as 1-5% of annotations), avoiding catastrophic forgetting.

## 2. Project Structure
To maintain a clean repository, the codebase is split into core execution scripts (at the root) and peripheral support scripts (under `scripts/`).

### Core Execution Scripts (Root Directory)
- **Training & Evaluation:** `train_*.py`, `evaluate_*.py`, `inference.py`, `benchmark_*.py`
- **Experiment Tracking:** `track_generalization.py`, `track_lora_fractions.py`
- **Shell Entrypoints:** `launch_*.sh`, `run_*.sh`, `slurm_train_*.sh`, `submit_*.sh`

### Peripheral Scripts (`scripts/`)
- **`scripts/data/`**: Dataset preparation, COCO splitting, and dataset statistics (e.g., `prepare_phase2_datasets.py`, `get_dataset_stats.py`, `generate_coverage_splits.py`).
- **`scripts/analysis/`**: Metrics computation, EDA, and plotting (e.g., `visualize_domain_metrics.py`, `plot_stats.py`, `aggregate_metrics.py`).
- **`scripts/topology/`**: Scripts for Phase 2 embedding extraction, coverage distances, and tree topology mapping (e.g., `custom_dinov2_embedding_pipeline.py`, `tree_topology_analysis.py`, `coverage_arborescence_viz.py`).
- **`scripts/utils/`**: Helper scripts for checkpoints, setup verification, etc. (e.g., `convert_ckpt.py`, `compile_dashboard.py`).

## 3. How to Run the Models and Configs

### Phase 1: Model Benchmarking
The two recommended models are **YOLOv5** and **RF-DETR-Seg**.

**Training:**
```bash
# YOLOv5
uv run train_yolov5.py model=yolov5

# RF-DETR-Seg
uv run train_rf_detr.py model=rfdetr_seg
```

**Evaluation:**
```bash
uv run evaluate_all_models.py
# Or use targeted evaluation for phase 2
uv run evaluate_phase2.py
```

### Phase 2: Coverage-Based Dataset Selection
To find the minimal dataset configuration, use the scripts inside `scripts/topology/` and `scripts/analysis/`:
1. **Extract embeddings:** `uv run scripts/topology/custom_dinov2_embedding_pipeline.py`
2. **Generate distance matrices and coverage:** `uv run scripts/analysis/compute_all_metrics.py`
3. **Build the coverage tree and rank nodes:** `uv run scripts/topology/best_coverage.py`

### Phase 3: LoRA-Based Domain Adaptation
When a novel cell line is encountered, adapt RF-DETR-Seg using LoRA without training the entire model.

1. **Launch LoRA fraction experiments:**
```bash
bash launch_lora_fractions.sh
# or combinations
bash launch_lora_combos.sh
```

2. **Track generalization and LoRA metrics:**
```bash
uv run track_lora_fractions.py --add-exp <path_to_report> --exp-name "Target: U87"
uv run track_generalization.py --add-exp <path_to_report> --exp-name "8 Nodes"
```

## 4. Understanding and Modifying Configs
Configurations are managed by Hydra and located in the `configs/` folder.
- **`configs/model/`**: Architecture-specific settings (e.g., `rfdetr_seg.yaml`, `yolov5.yaml`). Change hyperparameters, number of queries, backbones, etc.
- **`configs/data/`**: Datasets and splits (e.g., `coverage_splits/lora_finetune_mix.yaml`). Here you define which datasets to use for training/testing.
- **`configs/trainer/`**: PyTorch Lightning training parameters (epochs, precision, strategy).
- **`configs/optimizer/` & `configs/scheduler/`**: Learning rates and scheduling strategies.

**Example: Overriding Configs via CLI**
```bash
uv run train_rfdetr_phase2.py \
    model=rfdetr_seg \
    data=coverage_splits/lora_finetune_mix \
    data.target_data_frac=0.05
```

## 5. Actively Used Scripts
Based on recent development and commits, the following scripts are heavily utilized:
- `track_generalization.py` and `track_lora_fractions.py` (Tracking LoRA performance)
- `launch_lora_fractions.sh` and `launch_lora_combos.sh` (Initiating PEFT runs)
- Plotting utilities (`scripts/analysis/visualize_domain_metrics.py` etc.)
- Dataset utilities (`scripts/data/generate_coverage_splits.py`)
