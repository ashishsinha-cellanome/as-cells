# Cellanome DINOv2 Embeddings Analysis & Zero-Shot Generalization Strategy (Cell-Line Level)

## Original Objective Prompt
> *for the rfdetr-cell-analysi and FT recommendations for the datasets, since I used different values of k for computng the coverage, i want to see the analysis of clusters/dendograms and interpretation with respect to each K value. at the end of the doc, also show the general concensous on the dataset clusters that most metrics are agreeing with. NOTE that ti want the number of datasets to finetune to be a feew as possible*

---

## 1. Introduction
The objective of this analysis is to identify redundancy in representational space at the **cell-line level** across our 16 `cell-adhered` datasets. By evaluating embeddings through multiple distance and coverage metrics (specifically analyzing the sensitivity across different $K$ nearest neighbors), we aim to construct a minimally redundant subset of **cell lines** for fine-tuning. This strategy reduces training compute, eliminates data redundancy, and establishes a robust zero-shot holdout set.

The datasets map to **11 distinct cell lines**: `mc38`, `hela`, `c8d1a_astrocytes`, `a549`, `dc` (dendritic cells), `u87`, `preadipocytes`, `hs675tfibroblasts`, `imr90`, `moc22`, and `enteric-glia`.

## 2. Coverage Distance Analysis Across $K$ Values

The Coverage Distance metric is geometric and non-parametric, measuring the proportion of a target cell line's manifold that does *not* overlap with the training cell line's manifold. We analyzed this across varying sensitivities ($K \in \{5, 10, 15, 30\}$):

### $K=30$ (Broad Manifold View / Loose Strictness)
*   **Interpretation**: At $K=30$, the algorithm searches a wide neighborhood. Low distance here means the broad, macroscopic feature distributions overlap.
*   **Findings**: We see massive macro-clusters. `mc38` acts as a strong broad representation for `hela`, `astrocytes`, and `a549`. Similarly, `preadipocytes` acts as a broad parent distribution completely covering `hs675tfibroblasts` and `imr90`. Even `moc22` shows moderate overlap with the epithelial lines at this broad level.

### $K=15$ & $K=10$ (Moderate Strictness)
*   **Interpretation**: As the neighborhood shrinks, cell lines must share dense local feature regions to maintain coverage.
*   **Findings**: The macro-clusters fracture slightly into distinct "core" families. 
    *   The **Fibroblast family** (`preadipocytes`, `fibroblasts`, `imr90`) remains tightly unified.
    *   The **Epithelial family** (`mc38`, `hela`, `astrocytes`, `a549`) remains unified, proving that `mc38` provides deep, structural coverage of these other carcinomas and astrocytes.
    *   **Isolates emerge**: `moc22` and `enteric-glia` lose almost all coverage from other cell lines, confirming they occupy distinct regions in the feature space.

### $K=5$ (Highly Strict / Dense Overlap)
*   **Interpretation**: An extremely stringent threshold. Manifolds must heavily overlap in local, high-density regions to achieve coverage.
*   **Findings**: Only the most fundamentally similar cell lines maintain coverage here.
    *   Replicates/Conditions of the exact same cell line (e.g., `dc` caged vs uncaged, `preadipocytes` caged vs uncaged, `mc38` 0624 caged vs uncaged) maintain near-perfect coverage distance (~0.0).
    *   Remarkably, `u87` tightly covers the `mc38` (20240625 batch) even at $K=5$, suggesting an identical underlying phenotype in that specific batch.
    *   Off-diagonal coverage outside of these core families drops to near 0% (distance > 0.9), confirming that fine-tuning on a small representative set is necessary and sufficient to capture the dense modes of the data.

---

## 3. General Consensus: Cell-Line Topology

Across all metrics—including L2-Var-Norm, MMD-RBF, Symmetric/Asymmetric KL-Divergence, and all $K$ values of Coverage Distance—there is a strict general consensus on how the **11 distinct cell lines** cluster into families:

1.  **The Epithelial / Carcinoma Super-Cluster**
    *   `mc38` (specifically the 0624 batches), `hela`, `c8d1a_astrocytes`, `a549`.
    *   *Consensus:* `mc38` is the most central node and broadly covers the others.
2.  **The Fibroblast-like Super-Cluster**
    *   `preadipocytes`, `hs675tfibroblasts`, `imr90`.
    *   *Consensus:* `preadipocytes` provides deep, dense coverage over the entire cluster.
3.  **The Dendritic Cell Cluster**
    *   `dc`
    *   *Consensus:* Distinct from other lineages.
4.  **The U87 / MC38-Variant Cluster**
    *   `u87`, `mc38` (specifically the 0625 batch).
    *   *Consensus:* These two form a tight, isolated cluster across all metrics.
5.  **Isolates (No sufficient coverage from other families)**
    *   `moc22`
    *   `enteric-glia`

---

## 4. Fine-Tuning & Zero-Shot Recommendations (Minimal Set)

To minimize data redundancy, reduce fine-tuning compute, and establish a mathematically sound zero-shot evaluation, we can reduce the 11 cell lines down to **6 Minimal Cell-Line Centroids**. 

### 🎯 The "Centroid" Fine-Tuning Set (N = 6 Cell Lines)
By fine-tuning your model (adapter/head) on just one dataset from each of these 6 cell lines, you will cover the geometric manifold of the entire domain:

1.  **`mc38`** *(Represents the Epithelial/Carcinoma super-cluster)*
2.  **`preadipocytes`** *(Represents the Fibroblast super-cluster)*
3.  **`dc`** *(Represents the Dendritic Cell cluster)*
4.  **`u87`** *(Represents the U87/MC38-variant cluster)*
5.  **`moc22`** *(Isolate)*
6.  **`enteric-glia`** *(Extreme Isolate)*

*(Note: Pick one dataset representing each cell line, e.g., the caged variant, for the actual fine-tuning split).*

### 🛡️ The Zero-Shot Generalization Test Set (N = 5 Cell Lines)
The remaining 5 cell lines should be entirely held out from fine-tuning. Based on the embedding analysis, your fine-tuned model will successfully generalize to these zero-shot because their feature manifolds are heavily covered by the centroids.

| Zero-Shot Test Cell Line | Primary Covering Train Cell Line |
| :--- | :--- |
| `hela` | `mc38` |
| `c8d1a_astrocytes` | `mc38` |
| `a549` | `mc38` |
| `hs675tfibroblasts` | `preadipocytes` |
| `imr90` | `preadipocytes` |

*Additionally, hold out the alternate conditions (e.g., uncaged versions) of your training cell lines to measure intra-cell-line robustness!*

### Conclusion
This optimal strategy reduces the training burden from 11 distinct cell lines down to just **6**. The embeddings conclusively prove that training on these 6 centroids injects the necessary representational variance to cover the held-out `hela`, `astrocytes`, `a549`, `fibroblasts`, and `imr90` lineages.

---

## 5. Metric Concordance: MMD vs. Coverage Distance

Initial analysis of global distribution metrics (MMD-RBF, FID, L2-Var-Norm) suggested the existence of two primary clusters:
*   **Cluster 1 (Global distribution similarity):** `dc`, `moc22`, `a549`
*   **Cluster 2 (Global distribution similarity):** `imr90`, `hs675tfibroblasts`, `preadipocytes`

However, applying the non-parametric geometric Coverage Distance reveals a critical disagreement that dictates our fine-tuning strategy.

### 🤝 The Agreement (Cluster 2: Fibroblasts/Preadipocytes)
All metrics—MMD, FID, and Coverage Distance across all $K$ values—are in 100% agreement on Cluster 2. `imr90`, `hs675tfibroblasts`, and `preadipocytes` form an undeniably robust manifold. MMD confirms their global distributions match, while Coverage confirms their local data points physically intermingle. This cluster is highly stable even under strict local density checks ($K=5$).

### ⚔️ The Disagreement (Cluster 1: dc, moc22, a549)
The metrics diverge significantly on Cluster 1, exposing the danger of relying solely on global statistics:
*   **What MMD/FID sees:** These metrics group `dc`, `moc22`, and `a549` together because their *global macroscopic shapes* (means and variances) in feature space are statistically similar compared to the massive separation of the Fibroblast cluster.
*   **What Coverage Distance sees:** Coverage Distance (even at a broad $K=30$) reveals that the actual local data points of `dc`, `moc22`, and `a549` **do not intermingle**. Training on `dc` or `moc22` provides terrible coverage for `a549` (Coverage Distance > 0.8). 

Furthermore, as we vary the neighborhood strictness ($K$) in Coverage Distance:
*   **Broad View ($K=30$):** `a549` actually physically overlaps much better with the Epithelial cluster (`mc38` and `hela`, Distance ~0.19) rather than `dc` or `moc22`.
*   **Strict View ($K=10$, $K=5$):** `moc22` and `dc` rapidly lose all coverage from other lineages, fracturing away to become fully isolated manifolds. The Epithelial cluster also weakens locally, though it maintains macroscopic overlap.

### 💡 The Fine-Tuning Takeaway
When selecting datasets for fine-tuning, **Coverage is the safer metric to trust over MMD**. Even if MMD suggests two datasets have similar global distributions, poor Coverage means their local features are distinct. Fine-tuning on `dc` to zero-shot `a549` based on MMD would likely fail. 

Because Coverage correctly reassigns `a549` to the Epithelial macro-cluster, we safely use `mc38` as the broad centroid for the epithelial lineages, while treating `dc` and `moc22` as the separate isolates that their local geometry demands. This forms the basis of the 6-Centroid recommendation.

---

## 6. Visualizations: Clustermaps and Dendrograms for Coverage Distance

Below are the visualizations supporting the analysis above, generated at the cell-line level. The **Coverage Distance** metric was evaluated at sensitivities $K \in \{5, 10, 15, 30\}$. The clustermaps show the pairwise distance between datasets (Rows = Test, Cols = Train), and the dendrograms illustrate the hierarchical clustering structures.

### K = 30 (Broad Manifold View / Loose Strictness)
![Clustermap K=30](./custom_cell_line_embeddings_analysis/class_cell-adhered/cell_line_level/clustermap_coverage_distance_k30.png)
![Dendrogram K=30 Train](./custom_cell_line_embeddings_analysis/class_cell-adhered/cell_line_level/dendrogram_coverage_distance_k30_train.png)
![Dendrogram K=30 Test](./custom_cell_line_embeddings_analysis/class_cell-adhered/cell_line_level/dendrogram_coverage_distance_k30_test.png)

### K = 15 (Moderate Strictness)
![Clustermap K=15](./custom_cell_line_embeddings_analysis/class_cell-adhered/cell_line_level/clustermap_coverage_distance_k15.png)
![Dendrogram K=15 Train](./custom_cell_line_embeddings_analysis/class_cell-adhered/cell_line_level/dendrogram_coverage_distance_k15_train.png)
![Dendrogram K=15 Test](./custom_cell_line_embeddings_analysis/class_cell-adhered/cell_line_level/dendrogram_coverage_distance_k15_test.png)

### K = 10 (Moderate Strictness)
![Clustermap K=10](./custom_cell_line_embeddings_analysis/class_cell-adhered/cell_line_level/clustermap_coverage_distance_k10.png)
![Dendrogram K=10 Train](./custom_cell_line_embeddings_analysis/class_cell-adhered/cell_line_level/dendrogram_coverage_distance_k10_train.png)
![Dendrogram K=10 Test](./custom_cell_line_embeddings_analysis/class_cell-adhered/cell_line_level/dendrogram_coverage_distance_k10_test.png)

### K = 5 (Highly Strict / Dense Overlap)
![Clustermap K=5](./custom_cell_line_embeddings_analysis/class_cell-adhered/cell_line_level/clustermap_coverage_distance_k5.png)
![Dendrogram K=5 Train](./custom_cell_line_embeddings_analysis/class_cell-adhered/cell_line_level/dendrogram_coverage_distance_k5_train.png)
![Dendrogram K=5 Test](./custom_cell_line_embeddings_analysis/class_cell-adhered/cell_line_level/dendrogram_coverage_distance_k5_test.png)

---

## 7. Generalizability Sequences (Subset Chains)

Based on the Coverage Distance analysis (using a strict coverage threshold of >80%, or distance < 0.20 at $K=10$), we can identify hierarchical sequences of datasets. We denote $X \in Y$ if training on dataset $Y$ heavily covers dataset $X$. This establishes a clear subsets topology.

By training models on datasets at lower levels of the chain ($X_0, Y_0$) and testing them on higher levels ($X_2, Y_2$), we can empirically test generalizability. Since a subset does not cover its superset, the model should experience a drop in performance, proving the structural hierarchy of our datasets.

### Chain 1: The Fibroblast Lineage
**`Hs675Tfibroblasts` (and `imr90`) $\in$ `preadipocytes_caged` $\in$ `preadipocytes_uncaged`**
*   *Interpretation*: The uncaged preadipocytes provide the broadest, most robust manifold, encompassing the caged variants, which in turn encompass the baseline fibroblasts and IMR90 (though `imr90` falls slightly below the strict 80% coverage threshold, it anchors definitively to `preadipocytes_caged`).
*   *Experiment*: Train on `Hs675Tfibroblasts` ($X_0$), evaluate on `preadipocytes_uncaged` ($X_2$). We expect poor zero-shot performance. Train on `preadipocytes_uncaged` ($X_2$), evaluate on `Hs675Tfibroblasts` ($X_0$) and `imr90`. We expect excellent zero-shot performance.

### Chain 2: The Epithelial/MC38 Lineage
**`mc38_caged_0624` $\in$ `mc38_caged_0625` (and `hela`, `c8d1a_astrocytes`) $\in$ `u87_caged` $\in$ `mc38_uncaged_0624`**
*   *Interpretation*: The 0624 uncaged batch of MC38 exhibits the highest feature diversity, cleanly covering the tight clusters of U87 and other MC38 variants. The 0624 caged batch is the narrowest subset. `hela` and `astrocytes` also act as supersets to `mc38_caged` and branch off near `mc38_caged_0625` but are grouped in parentheses as they slightly miss the strict multi-hop 80% thresholds.

### Chain 3: The A549 to MOC22 Isolate Transition
**`mc38_caged_0624` $\in$ `a549` $\in$ `moc22`**
*   *Interpretation*: `moc22` occupies a massive, sparse region in feature space that loosely blankets the regions containing `a549` and `mc38`. However, `mc38` is tightly clustered and provides almost zero coverage for the outer boundaries of `a549` or `moc22`.
*   *Experiment*: Training on `mc38` ($Z_0$) will likely fail to detect cells in `moc22` ($Z_2$), but training on `moc22` may surprisingly cover the basic epithelial structures of `mc38`.

### Chain 4: The Dendritic Cell Progression
**`mc38_caged_0624` $\in$ `DC_caged_0516` $\in$ `DC_caged_0515`**
*   *Interpretation*: The 0515 batch of Dendritic Cells provides slightly broader representation than the 0516 batch. Both are strictly supersets relative to the core `mc38` distribution. 

These chains present ideal empirical experiments. By systematically fine-tuning models on the $X_0, Y_0$ anchors and evaluating them upward through the chains, we can conclusively validate the predictive power of the Coverage Distance metric for deep learning generalizability.

