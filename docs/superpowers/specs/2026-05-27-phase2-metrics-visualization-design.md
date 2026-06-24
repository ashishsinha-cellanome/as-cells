# Design Spec: PHASE2 Metrics Visualization

## Objective
Generate high-resolution visualization charts (Grouped Bar Charts and Spider Charts) to compare the performance of three object detection models (`yolo`, `yolov26`, `rf_detr_seg`) across various datasets in PHASE2.

## 1. Data Source and Preprocessing
- **Source File**: `/mnt/direct-attached/PHASE2_EVAL_RESULTS/all_models_summary.csv`
- **Models to Evaluate**: `yolo` (YOLOv5), `yolov26`, `rf_detr_seg`
- **Metrics**: `mAP@50`, `mAP@50-95`
- **Data Cleaning**: 
  - Filter the DataFrame to include only the specified models.
  - Handle missing classes in specific datasets: Values of `-1.0` in the CSV will be converted to `0` or `NaN` to prevent skewing the visualizations.

## 2. Output Configuration
- **Output Directory**: `/mnt/direct-attached/PHASE2_EVAL_RESULTS/plots/`
- **Image Specifications**: High resolution (DPI 300), large figure sizes (e.g., 20x12 inches) to ensure text legibility given the large number of datasets (21 datasets).

## 3. Visualization Details

### A. Grouped Bar Charts
- **Overall Performance (`Class == 'all'`)**: 
  - Plot 1: `mAP@50` for all datasets grouped by model.
  - Plot 2: `mAP@50-95` for all datasets grouped by model.
- **Per-Class Performance**:
  - For each specific class (`cell`, `bead`, `soma`, `cell-adhered`), generate and save separate plots.
  - Plot: `mAP@50` grouped by model.
  - Plot: `mAP@50-95` grouped by model.
- **Formatting**: X-axis labels (datasets) will be rotated for readability.

### B. Spider (Radar) Charts
- **Overall Performance (`Class == 'all'`)**: 
  - Plot 1: `mAP@50` across all datasets (each dataset is an axis).
  - Plot 2: `mAP@50-95` across all datasets.
- **Per-Class Performance**:
  - For each specific class, generate and save separate spider plots.
  - Plot: `mAP@50` across all datasets.
  - Plot: `mAP@50-95` across all datasets.
- **Formatting**: Dataset names will be text-wrapped to prevent overlap around the perimeter. Legend will clearly indicate the models.

## 4. Implementation Approach
- A single Python script `visualize_metrics.py` located in the project root.
- Uses `pandas` for data manipulation, `matplotlib` and `seaborn` for generating the static charts, and `numpy` for angle calculations in spider charts.
