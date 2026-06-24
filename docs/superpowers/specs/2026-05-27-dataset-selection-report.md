# Dataset Selection Engine Report

## 1. Introduction
To optimally select datasets for Parameter-Efficient Fine-Tuning (PEFT) and zero-shot generalization experiments, we implemented an embedding-based dataset selection engine. The engine quantitatively groups datasets by their morphological similarities, allowing us to select maximally representative training folds without human bias.

## 2. Methodology: How the Embeddings and Matrices Were Created
We isolated morphological features of the objects in the datasets using a frozen **DINOv2** Vision Transformer. 

To ensure the embeddings reflect true morphology and not background context or processing artifacts, the following robust extraction pipeline was used:
1. **Class-Specific Parsing**: The engine processes the datasets on a **per-class basis** (e.g., separating `cell`, `bead`, `soma`, `cell-adhered`). The analysis is conducted specifically using these ground-truth class labels.
2. **Top Instance Isolation**: For a given image and class, the script finds the largest annotated bounding box to avoid noise from partially occluded or tiny, low-resolution objects.
3. **Aspect-Ratio Preserving Crop**: The bounding box is padded with zeros (black pixels) to form a perfect square *before* being resized to the DINOv2 input size (224x224). This prevents the cell from being unnaturally stretched or squashed.
4. **DINOv2 Forward Pass**: The masked, padded crop is passed through DINOv2 to extract the global `[CLS]` token, representing the canonical morphological signature of that class instance.
5. **Combined Class Features**: In addition to per-class embeddings, we concatenate the embeddings of all possible classes (filling in zeros for physically absent classes) to form a holistic, dataset-level morphological profile representing the entire available environment of each cell line.

## 3. How to Interpret the Pairwise Distance Matrix
Using the canonical morphological signatures, we computed a Pairwise Distance Matrix using **Cosine Distance**. 
- A **distance close to 0** indicates that the objects (e.g., cells) in Dataset A and Dataset B look virtually identical to the DINOv2 model.
- A **higher distance** indicates significant morphological divergence (e.g., small, perfectly round suspension cells versus large, sprawled adhered cells).

*Clarification on "Morphological Features" Title:* The title refers to the raw, high-dimensional visual textures, shapes, and patterns encoded by the DINOv2 foundational model extracted purely from the instances labeled by their ground-truth class labels. Unlike traditional ML features (like manually computed area or perimeter), these embeddings capture complex, non-linear visual semantics directly from the pixel data.

## 4. Visualizations
The script generated several high-resolution (300 DPI) visualizations, which are saved in `/mnt/direct-attached/PHASE2_EVAL_RESULTS/selection_engine/`:

- **Embedding Overlays** (`<dataset-name>_<image_name>_<class_name>_emb.png`): These images overlay a PCA reduction of the DINOv2 spatial patch tokens onto the original cell crop using the `viridis` colormap. The PCA values are mathematically normalized to the `[-1, 1]` range. They visually highlight which specific physical structures (like the nucleus or membrane) the model is focusing on. A colorbar (colorscale) spanning `[-1, 1]` is included to map the feature values.
- **Pairwise Distance Heatmaps** (`pairwise_distance_heatmap_<metric>_class_<class_name>.png` & `_all_classes.png`): High-resolution heatmaps showing the exact mathematical distances between every cell line. Both Cosine and Euclidean (L2) distance metrics are computed. The plot dimensions and font sizes dynamically scale to the number of datasets compared to ensure text remains highly legible, with increased font sizes for axes labels and titles.
- **Cluster Visualizations** (`cluster_visualization_tsne_p<perp>_class_<class_name>.png`, `cluster_visualization_umap_class_<class_name>.png`, etc.): 2D scatter plots mapping the datasets into physical space, color-coded by their K-Means cluster assignments. We evaluate both **t-SNE** (with perplexity values 5, 15, 30) and **UMAP** dimensionality reduction techniques to explore the topological structures at different scales. To ensure legibility, we used the `adjustText` algorithm so that dataset labels never overlap each other or the centroids. The figures are specifically formatted as large squares (`16x16`) with tight layout parameters ensuring that all points, titles, and legends remain fully contained inside the frame without getting cut off. Both per-class and 'all classes combined' plots are provided.

## 5. Conclusion
By clustering the datasets using these morphological distance matrices, we successfully grouped the datasets into distinct clusters. Selecting the "medoid" (center) of each cluster guarantees that the resulting training fold covers the maximum possible morphological variance, setting up a rigorous zero-shot generalization test.