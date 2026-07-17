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

## Advanced Embedding Distance Metrics for Cell Line Analysis

To quantify the differences between cell line embedding distributions (e.g., 256-D for RF-DETR, 768-D for DINOv2), we use a comprehensive suite of statistical distance metrics. These metrics help in selecting representative training datasets and ensuring generalizability.

### 1. Diagonal 2-Wasserstein Distance
**Formula:**
$$ W_{2, diag}^2 = \sum_{i=1}^{D} \left( (\mu_{1,i} - \mu_{2,i})^2 + (\sigma_{1,i} - \sigma_{2,i})^2 \right) $$
**Interpretation:** This simplified Fréchet Inception Distance (FID) assumes a diagonal covariance matrix. It balances shifts in the mean with differences in the spread (variance). It acts as an additive "energy penalty" representing the minimum effort to transform one diagonal Gaussian into another.

### 2. Symmetric KL Divergence (Jeffreys Divergence)
**Formula:**
$$ D_{symKL} = \frac{1}{2} \sum_{i=1}^{D} \left( \frac{\sigma_{1,i}^2 + (\mu_{1,i} - \mu_{2,i})^2}{\sigma_{2,i}^2} + \frac{\sigma_{2,i}^2 + (\mu_{2,i} - \mu_{1,i})^2}{\sigma_{1,i}^2} - 2 \right) $$
**Interpretation:** Evaluates how much information is lost when approximating one distribution with the other. It aggressively penalizes cases where one cell line has near-zero variance in a dimension where the other varies widely. It acts as a strict check to ensure a training set "envelopes" the variance of the test sets.

### 3. Diagonal Bhattacharyya Distance
**Formula:**
$$ D_{B, diag} = \frac{1}{8} \sum_{i=1}^{D} \frac{(\mu_{1,i} - \mu_{2,i})^2}{\bar{\sigma}_i^2} + \frac{1}{2} \sum_{i=1}^{D} \ln \left( \frac{\bar{\sigma}_i^2}{\sigma_{1,i} \sigma_{2,i}} \right) $$
*(where $\bar{\sigma}_i^2 = \frac{\sigma_{1,i}^2 + \sigma_{2,i}^2}{2}$)*
**Interpretation:** Measures the overlap of two statistical samples. It separates the distance into a Mahalanobis-like term (mean shift normalized by joint spread) and a variance-ratio term. It provides a smooth measure of distribution overlap that is robust to outliers.

### 4. Hellinger Distance
**Formula:**
$$ H = \sqrt{1 - e^{-D_{B, diag}}} $$
**Interpretation:** Directly derived from the Bhattacharyya distance, but strictly bounded between **0 and 1**. It is highly interpretable as a percentage of distribution mismatch. $H \approx 0$ indicates perfect overlap, while $H \approx 1$ implies the cell lines occupy completely different morphological spaces.

### 5. Full Bhattacharyya Distance
**Formula:**
$$ D_{B, full} = \frac{1}{8} (\mu_1 - \mu_2)^T \Sigma^{-1} (\mu_1 - \mu_2) + \frac{1}{2} \ln \left( \frac{\det(\Sigma)}{\sqrt{\det(\Sigma_1)\det(\Sigma_2)}} \right) $$
*(where $\Sigma = \frac{\Sigma_1 + \Sigma_2}{2}$)*
**Interpretation:** Unlike the diagonal version, this incorporates the full covariance structure of the embeddings. We utilize Ledoit-Wolf shrinkage to stabilize the covariance matrices in high-dimensional space, preventing numerical instability while capturing directional correlations between dimensions.

### 6. Mahalanobis Distance (Pooled Covariance)
**Formula:**
$$ D_M = \sqrt{(\mu_1 - \mu_2)^T \Sigma_{pooled}^{-1} (\mu_1 - \mu_2)} $$
*(where $\Sigma_{pooled} = \frac{(n_1-1)\Sigma_1 + (n_2-1)\Sigma_2}{n_1+n_2-2}$)*
**Interpretation:** Measures the distance between the distribution means, scaled inversely by the combined (pooled) variance structure of the two datasets. It is ideal for evaluating how "far" apart datasets are relative to their natural biological and morphological variation.

### 7. Sliced Wasserstein Distance (SWD)
**Formula:**
$$ SWD = \frac{1}{L} \sum_{l=1}^L \int_0^1 \left| F^{-1}_{\theta_l \cdot X}(p) - F^{-1}_{\theta_l \cdot Y}(p) \right|^2 dp $$
**Interpretation:** A non-parametric distance computed directly from the empirical samples (raw embeddings) rather than assuming a Gaussian distribution. It slices the high-dimensional space into numerous 1D random projections ($\theta_l$), sorts the projected values, and averages the 1D Wasserstein distances. This captures complex, non-Gaussian structural differences in the embedding distributions perfectly.

### 8. Asymmetric KL Divergence
**Formula:**
$$ KL(P || Q) = \frac{1}{2} \sum_{i=1}^{D} \left( \ln\left(\frac{\sigma_{Q,i}^2}{\sigma_{P,i}^2}\right) + \frac{\sigma_{P,i}^2 + (\mu_{P,i} - \mu_{Q,i})^2}{\sigma_{Q,i}^2} - 1 \right) $$
**Interpretation:** Measures the penalty of approximating the target test distribution $P$ using the source training distribution $Q$. It evaluates subset/superset relationships: if training set $Q$ "covers" test set $P$, the penalty is low. If $P$ contains morphologies outside the span of $Q$, the penalty explodes.

### 9. Coverage (Naeem et al., ICML 2020)
**Formula:**
$$ Coverage(X, Y) = \frac{1}{|X|} \sum_{x \in X} \mathbb{1} \left[ \min_{y \in Y} d(x, y) \le d(x, NNI_k(x, X)) \right] $$
**Interpretation:** Evaluates the fraction of target test samples $X$ whose $k$-NN ball (computed within $X$) contains at least one source train sample $Y$. We compute the distance as **$1 - Coverage(X, Y)$**. This effectively measures whether the training manifold spatially encompasses the test manifold.

## Analysis Pipeline and Scripts

This section describes the order of execution for the various analysis scripts to compute embeddings, generate distance matrices (heatmaps/clustermaps), and perform tree topology analysis.

### Step 1: Embedding Calculation
**Script:** `custom_dinov2_embedding_pipeline.py` (or `rfdetr_seg_embeddings_analysis.py` / `custom_rfdetr_cell_line_embeddings_analysis.py`)
- **What it does:** Loads datasets, passes them through the selected backbone (e.g., DINOv2 or RF-DETR), and extracts high-dimensional morphological embeddings. It saves the raw extracted embeddings (e.g., `extracted_raw_embeddings.pkl`) for downstream processing.
- **How to run:**
  ```bash
  uv run custom_dinov2_embedding_pipeline.py
  ```

### Step 2: Distance Matrix & Heatmap/Clustermap Computation
**Script:** `compute_all_metrics.py` (and variations like `calculate_advanced_metrics.py` / `calculate_asymmetric_metrics.py`)
- **What it does:** Loads the raw embeddings from Step 1 and calculates all the distance metrics described above (Wasserstein, Sym-KL, Asymmetric KL, Coverage, etc.). It generates pairwise distance matrices and visualizes them using Seaborn heatmaps and hierarchical dendrogram clustermaps.
- **How to run:**
  ```bash
  uv run compute_all_metrics.py
  ```

### Step 3: Tree Topology Analysis (Coverage Arborescence)
**Script:** `best_coverage.py` (which utilizes `coverage_arborescence_viz.py`)
- **What it does:** Uses the Coverage Distance metric (computed from the raw embeddings) to build a directed spanning arborescence (a rooted tree). This structure identifies the "broadest" dataset (the root) and creates a level-ordered hierarchy showing which datasets cover other datasets best. Edges are styled based on a coverage threshold to show strong vs. weak morphological coverage.
- **How to run:**
  ```bash
  uv run best_coverage.py
  ```
- **Outputs:**
  - `coverage_arborescence_report.txt`: A text summary of the ranked nodes and tree edges.
  - `coverage_arborescence_plot.png`: A Matplotlib/NetworkX hierarchical tree visualization.

### Step 4: Cross-Dataset Generalization (In-Domain vs Zero-Shot) Visualization
**Script:** `visualize_domain_metrics.py` (an evolution of `visualize_html_metrics.py`)
- **What it does:** Parses RF-DETR HTML evaluation reports and isolates `test_ds` records to prevent metric duplication. It categorizes datasets into an **In-domain** subset (which represents the training distribution) and **Zero-shot** (the remaining out-of-domain datasets). It extracts these into a tidy CSV and generates 6 generalization plots (ranked mAP bars, dumbbell plots, precision-recall iso-F1 contours, and heatmaps) to visualize model robustness on out-of-distribution data.
- **How to run:**
  ```bash
  uv run python3 visualize_domain_metrics.py path/to/your_report.html --in-domain a549
  ```
  *(You can pass one or more substrings to `--in-domain` to specify the dataset(s) your model was actually trained on. For example, `--in-domain a549 hela`. It defaults to `a549` if not specified).*
