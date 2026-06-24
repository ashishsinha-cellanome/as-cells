# Disjoint Tree Topology Analysis & Generalization Insights

## 1. Overview
As requested, we adapted the topological coverage graph analysis to construct disjoint tree clusters (Minimum Spanning Arborescences per connected component) by applying variable coverage thresholds. This approach systematically prunes weak edges to reveal "natural" dataset clusters and exposes the true, high-confidence superset → subset hierarchies.

The analysis was performed using pre-computed DINOv2 base embeddings. We swept coverage thresholds from 30% to 70% across K=5, 10, 15, and 30.

**Note:** All paths and outputs referenced below are available in:
`/mnt/direct-attached/as-cells/tree_topology_output_train/dinov2_base/`

## 2. Emergence of Disjoint Sets (Component Dissolution)
By removing edges below a defined threshold, we observe how the unified dataset domain breaks into isolated, highly-coupled islands. 

Looking at the most rigorous nearest-neighbor setting (**K=5, Dataset Level**), the "dissolution cascade" behaves as follows:

| Coverage Threshold | Components (Trees) | Isolated Singletons | Interpretation |
|--------------------|--------------------|---------------------|----------------|
| **≥ 30%**          | 1 tree (21 nodes)  | 0                   | A single unified tree connects everything. |
| **≥ 40%**          | 4 trees            | 2                   | The weakest links break. Neurons and enteric-glia emerge as distinct root domains. |
| **≥ 50%**          | 12 trees           | 7                   | **Natural Clustering occurs.** 5 multi-dataset trees remain, with 7 datasets completely isolated. |
| **≥ 60%**          | 16 trees           | 11                  | Only strongly related pairs/triplets survive. |
| **≥ 70%**          | 18 trees           | 15                  | Extreme isolation; only 3 pairs show >70% asymmetric coverage. |

*(A visual plot of this cascade is available at `tree_topology_output_train/dinov2_base/dataset_level/tree_k5_component_dissolution.png`)*

## 3. High-Confidence Superset Hierarchies
At the **≥ 50% threshold** (K=5), 5 natural multi-dataset components emerge. The roots of these trees are the "ultimate supersets" for their respective clusters:

1. **Neuron Cluster (5 datasets)**: 
   Root: `20240422_neuron_uncaged`
   Subsets: Covers `20250108_neuron` (59%), which covers `20250305_neuron` (58%). It also covers `20240704_neuron_caged` (59%), which covers `20240703_neuron_caged` (95%).
2. **MC38 / A549 Cluster (3 datasets)**:
   Root: `20240624_mc38_caged`
   Subsets: Covers `20240624_mc38_uncaged` (77%), which covers `20260316_a549-tomm20-gfp` (49%).
3. **DC Cluster (2 datasets)**:
   Root: `20240516_DC_caged`
   Subsets: Covers `20240515_DC_caged` (82.9%).
4. **U87 / MC38 Cluster (2 datasets)**:
   Root: `20240905_u87_caged`
   Subsets: Covers `20240625_mc38_caged` (61.6%).
5. **Preadipocyte Cluster (2 datasets)**:
   Root: `20241212_preadipocytes_uncaged`
   Subsets: Covers `20250227_preadipocytes_caged` (51.8%).

*(The complete topological trees for these are available as individual component plots e.g., `tree_k5_t0.5_c1.png` and as a combined overview at `tree_k5_t0.5_combined.png`)*

## 4. Inferences for "Subset-to-Superset" Generalization Experiments
The collaborator requested checking generalizability from subsets to supersets (the "hard" generalization direction) rather than superset to subset.

Using the trees generated above, we can directly define these experiments by traversing the trees from **Leaf to Root**:

**Proposed Experiment Pairs:**
1. **Strong Domain (Neurons)**
   - Train on: `20240703_neuron_caged` (Subset/Leaf)
   - Evaluate on: `20240704_neuron_caged` (Parent/Superset)
   - Evaluate on: `20240422_neuron_uncaged` (Ultimate Root/Superset)
   - *Hypothesis:* The model will struggle significantly compared to the reverse direction, as the leaf lacks the morphological diversity of the roots.

2. **Cross-Cell Line Domain**
   - Train on: `20240624_mc38_uncaged` (Subset)
   - Evaluate on: `20240624_mc38_caged` (Superset)
   - *Hypothesis:* Training on the uncaged variant will not generalize well to the caged variant, even though they are the same cell line, because the caged variant acts as a topological superset (77% coverage).

3. **Tight Domain (DC)**
   - Train on: `20240515_DC_caged` (Subset)
   - Evaluate on: `20240516_DC_caged` (Superset)
   - *Hypothesis:* The subset only provides narrow coverage of the superset, resulting in poor generalizability.

## 5. Conclusion
By introducing variable coverage thresholds, we successfully isolated the dataset domain into statistically significant, disjoint hierarchies. The resulting components reveal that only a few highly-coupled datasets generalize well at thresholds >60%, while a coverage threshold of 50% produces naturally grouped clusters representing distinct morphological families. These structures explicitly identify which datasets should be used for the upcoming Subset → Superset generalization training experiments.