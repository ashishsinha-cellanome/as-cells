# Custom Embedding Analysis: Distance Metrics

This document outlines the distance metrics used to compute morphological differences between cell lines based on their DINOv2 / RF-DETR embeddings. The metrics vary in how they utilize the mean and variance (covariance) of the embeddings to quantify differences, ranging from simple Euclidean distances to advanced information-theoretic and non-parametric coverage metrics.

## 1. Euclidean Distance
**Formula**: 
$$ d(A, B) = \|\mu_A - \mu_B\|_2 $$

**Intuition**: The straight-line distance between the mean embedding vectors of two cell lines in the high-dimensional feature space.

**Role of Variance**: None. This is a baseline metric that only considers the mean differences and ignores the spread or uncertainty of the embeddings.

**Interpretation**: Larger values indicate that the cell lines are farther apart in absolute feature space. Useful for a quick baseline, but may be sensitive to scale and ignores morphological heterogeneity.

## 2. Cosine Distance
**Formula**: 
$$ d(A, B) = 1 - \frac{\mu_A \cdot \mu_B}{\|\mu_A\|\|\mu_B\|} $$

**Intuition**: Measures the angular difference between the mean embedding vectors.

**Role of Variance**: None. This baseline metric is scale-invariant but ignores variance.

**Interpretation**: Values range from 0 (identical direction) to 2 (opposite direction). Good for comparing feature signatures independent of their overall magnitude, but again fails to account for morphological spread.

## 3. L2 Variance-Normalized (L2 Var-Norm)
**Formula**: 
$$ d(A, B) = \frac{\|\mu_A - \mu_B\|_2}{\sqrt{\text{mean}(\sigma_A^2)} + \sqrt{\text{mean}(\sigma_B^2)} + \epsilon} $$

**Intuition**: The Euclidean distance scaled by the overall standard deviation of both cell lines. The denominator acts as a global scalar normalization factor (Root Mean Square pooled standard deviation).

**Role of Variance**: Uses scalar variance. Normalizes the distance based on how spread out the representations are on average across all dimensions.

**Interpretation**: If both cell lines have highly variable morphology (high variance), their mean distance is penalized (reduced). This prevents noisy or highly variable cell lines from artificially inflating the distance.

## 4. Cosine Variance-Weighted (Cosine Var-Weighted)
**Formula**: 
First, compute weights for each dimension $d$:
$$ w_d = \frac{1}{\sqrt{\sigma_{A,d}^2 + \sigma_{B,d}^2 + \epsilon}} $$
Scale the means: $\mu'_A = \mu_A \odot w$, $\mu'_B = \mu_B \odot w$
$$ d(A, B) = 1 - \frac{\mu'_A \cdot \mu'_B}{\|\mu'_A\|\|\mu'_B\|} $$

**Intuition**: An angular distance where the contribution of each feature dimension is down-weighted if that dimension is highly variable (noisy) across the cell line populations.

**Role of Variance**: Per-dimension variance weighting. Dimensions with low variance (stable features) are amplified, while dimensions with high variance are suppressed before computing the angle.

**Interpretation**: Focuses the distance computation on the most stable, reliable morphological features, making the cosine distance robust to dimensions that fluctuate wildly within a cell line.

## 5. Variance-Normalized Dimensional Euclidean (Var-Norm Dim)
**Formula**: 
$$ d(A, B) = \sqrt{\sum_d \frac{(\mu_{A,d} - \mu_{B,d})^2}{\sigma_{A,d}^2 + \sigma_{B,d}^2 + \epsilon}} $$

**Intuition**: Similar to the Mahalanobis distance assuming a diagonal covariance matrix. It computes the Euclidean distance where each dimension is scaled by its specific pooled variance.

**Role of Variance**: Per-dimension variance normalization.

**Interpretation**: A feature dimension that is naturally highly variable will contribute less to the total distance. It effectively standardizes differences dimension-by-dimension, providing a more robust distance measure than unweighted Euclidean.

## 6. Fréchet Inception Distance (FID)
**Formula**: 
$$ d(A, B) = \|\mu_A - \mu_B\|_2^2 + \text{Tr}\left(\Sigma_A + \Sigma_B - 2(\Sigma_A \Sigma_B)^{1/2}\right) $$
*(Note: Computed using Ledoit-Wolf shrinkage to estimate the covariance matrices $\Sigma_A$ and $\Sigma_B$)*

**Intuition**: Measures the distance between two multivariate Gaussian distributions fitted to the cell line embeddings. It considers both the shift in the mean and the distortion in the covariance shape.

**Role of Variance**: Captures the full covariance structure (all variances and pairwise covariances between dimensions).

**Interpretation**: A highly comprehensive metric that accounts for how features correlate with each other. A lower score means the cell lines share both mean morphological features and the structural relationships between those features.

## 7. Maximum Mean Discrepancy with RBF Kernel (MMD-RBF)
**Formula**: 
$$ \text{MMD}^2 = E[k(X, X)] + E[k(Y, Y)] - 2E[k(X, Y)] $$
Where $k(x, y)$ is a Radial Basis Function (RBF) kernel evaluated at 3 scales of the median heuristic (0.5x, 1x, 2x).

**Intuition**: Compares the full empirical distributions of the two cell lines in a reproducing kernel Hilbert space. Instead of assuming a Gaussian shape (like FID), it directly compares all moments of the distributions.

**Role of Variance**: Non-parametric; it implicitly captures all moments of the distributions, including variance, skewness, and multi-modality.

**Interpretation**: If two cell line populations have complex, non-Gaussian morphological distributions (e.g., distinct sub-populations of cells), MMD will capture this discrepancy. It is the most flexible metric for distribution matching.

## 8. Symmetric KL Divergence (Sym-KL)
**Formula**: 
$$ D_{KL-Sym}(A, B) = \frac{1}{2} \left( D_{KL}(A || B) + D_{KL}(B || A) \right) $$
*(Computed under a diagonal Gaussian assumption).*

**Intuition**: Measures the total information lost if we use the morphological distribution of cell line A to encode cell line B, and vice-versa. It provides a symmetric view of how divergent the two cell populations are from each other.

**Role of Variance**: Explicitly uses the per-dimension variances. The distance is heavily penalized if the two distributions have very different variances, or if their means are far apart relative to their variances.

**Interpretation**: A high symmetric KL divergence means the two cell lines are statistically very distinct in their morphology. Useful for determining overall topological clustering and separability between classes.

## 9. Asymmetric KL Divergence
**Formula**: 
$$ D_{KL}(Test || Train) = \sum_d \left( \log\left(\frac{\sigma_{Train,d}}{\sigma_{Test,d}}\right) + \frac{\sigma_{Test,d}^2 + (\mu_{Test,d} - \mu_{Train,d})^2}{2\sigma_{Train,d}^2} - \frac{1}{2} \right) $$

**Intuition**: Measures how well a "Train" cell line's distribution covers or models a "Test" cell line's distribution. It asks: "If I only know the morphology of the Train set, how surprising is the morphology of the Test set?"

**Role of Variance**: Highly sensitive to variance mismatches. If the Test set has morphology extending outside the Train set's variance (i.e., Test has higher variance), the penalty is massive. If the Test set is a narrow subset entirely contained within the Train set's broad variance, the penalty is low.

**Interpretation**: This is a directional metric perfectly suited for selecting datasets for fine-tuning. A low $D_{KL}(Test || Train)$ indicates that training on the "Train" dataset provides excellent morphological coverage for zero-shot generalization to the "Test" dataset. 

## 10. Coverage (KNN-based)
**Formula**: 
Computes the fraction of samples in the "Test" dataset whose $k$-nearest neighbors in the combined Train/Test pool include samples from the "Train" dataset.

**Intuition**: A non-parametric, geometric measure of overlap. It directly measures what percentage of the Test cell line's manifold is densely intertwined with the Train cell line's manifold in high-dimensional feature space.

**Role of Variance**: Non-parametric. Does not explicitly model variance, but rather relies on the actual point-cloud spread, density, and local neighborhood of the embeddings.

**Interpretation**: High coverage (e.g., > 0.8) means the Train dataset's morphological variations encompass almost all the variations seen in the Test dataset. This is a direct proxy for expected zero-shot model performance.

## 12. Analysis Pipeline and Scripts

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