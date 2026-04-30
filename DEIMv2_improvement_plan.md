# DEIMv2 Improvement Plan: Class Imbalance & Large Object Detection

This document summarizes all the data, architectural, and learnable dynamic changes discussed to improve the DEIMv2 models, specifically targeting the poor performance on the `cell-adhered` class (~40%) and the bottleneck on large object detection.

## 1. Tackling Class Imbalance (`cell-adhered` class)

| Idea / Trick | What (Implementation) | Why (Reasoning) | Intuition / Prior Research |
| :--- | :--- | :--- | :--- |
| **Weighted Random Sampling (RFS)** | Use `WeightedRandomSampler` in the PyTorch DataLoader to oversample images containing `cell-adhered`. | Standard training rarely sees minority classes, treating them as noise. Oversampling guarantees the network gets enough gradient updates for these rare classes. | If you only see a rare object once every 100 images, the network forgets its features. Seeing it more frequently forces the model to retain its representation. |
| **Class-Weighted Focal Loss** | Multiply the Focal Loss output by a class-specific weight vector (e.g., `[1.0, 1.0, 3.0, 1.0]`). | Focal loss handles foreground/background imbalance but treats all classes equally. We need to heavily penalize errors on minority classes. | "Cost-Sensitive Learning" dictates that mistakes on rare classes should cost the model more, forcing the optimizer to adjust weights to recognize them. |
| **Increase Contrastive Denoising (CDN)** | Set `num_denoising: 300` and `label_noise_ratio: 0.8` in `deimv2.yaml`. | CDN injects noisy ground-truth boxes to train the network to reconstruct them. Higher noise forces the model to learn more discriminative features between similar classes. | CDN (DN-DETR) stabilizes early training by bypassing the unstable Hungarian matching. More noise acts as a strong regularizer for fine-grained classification. |
| **Adjust Matcher Costs** | Increase `cost_class: 4` in the Hungarian Matcher config. | The matcher assigns predictions to ground truths. If `cost_class` is low, it might assign a `cell-adhered` GT to a standard `cell` prediction just because the bounding box is tight. | Penalizing incorrect class matching forces the network to prioritize exact semantic class alignment over purely spatial overlaps. |
| **Large Scale Jittering (LSJ) & Mosaic** | Add aggressive zoom-out transforms (down to 50%) and pad the rest. Stitich images YOLO-style. | Objects scaled down physically become smaller, changing the context and exposing the network to more varied backgrounds. | Mosaic inherently acts as a regularizer, forcing the model to detect objects at multiple scales and in foreign contexts. |

---

## 2. Improving Large Object Detection

| Idea / Trick | What (Implementation) | Why (Reasoning) | Intuition / Prior Research |
| :--- | :--- | :--- | :--- |
| **Increase Regression Bounds** | Change `reg_max: 48` and `reg_scale: 6.0` in `deimv2.yaml`. | DEIMv2 uses discrete bins for bounding box regression. If an object exceeds the mathematical bounds of `reg_max`, the model *cannot* predict it. | Expanding the bins allows the Fine-Grained Localization (FGL) distribution to stretch further, covering massive bounding boxes. |
| **Increase Attention Points** | Change `num_points: [6, 12, 12]` in `deimv2.yaml`. | Multi-Scale Deformable Attention samples a fixed number of points. More points at lower resolutions (Stride 32) cover a wider spatial area. | The maximum reach of the attention offsets scales linearly with the number of points. More points = wider receptive field. |
| **Increase Base Anchor Size** | Change `grid_size=0.15` (from `0.05`) in `_generate_anchors()` in `deim_decoder.py`. | At `grid_size=0.05`, the maximum initial anchor is only 20% of the image. The model struggles to scale a 20% box up to an 80% object. | Giving the regression head a larger starting anchor (e.g., 60% of the image) provides a massive head start for large objects. |
| **Add Stride 64 Feature Level** | Extract a 4th FPN level (Stride 64) from the backbone. | Attention at Stride 32 is limited by that resolution. Stride 64 means each attention point covers 4x the area. | Large objects are most efficiently detected at very low resolutions where their entire structure fits in a small spatial window. |
| **Shift Matcher Cost to GIoU** | Swap costs to `cost_bbox: 2`, `cost_giou: 5` in config. | Absolute pixel errors (L1 loss) scale linearly with object size. The matcher unfairly penalizes large objects if L1 cost is too high. | GIoU is scale-invariant. Favoring GIoU ensures a large box with a 10-pixel error is treated similarly to a small box with a 2-pixel error. |

---

## 3. Making Dynamics Learnable (Architecture Hacks)

| Idea / Trick | What (Implementation) | Why (Reasoning) | Intuition / Prior Research |
| :--- | :--- | :--- | :--- |
| **Learnable Attention Reach** | Change `offset_scale` to a learnable parameter per attention head (`nn.Parameter(...)`). | Instead of a fixed stretch factor, the network dynamically learns how far each head should attend. | **Deformable DETR / DINO:** Networks naturally evolve "local" and "global" heads. Making the scale learnable gives the model explicit freedom to specialize its attention. |
| **Learnable Anchor Sizes** | Create a learnable `grid_size` parameter for each FPN level (`self.learned_grid_sizes`). | Replaces human-hardcoded anchor sizes (0.05) with dynamic bounds driven by the dataset's loss landscape. | **DAB-DETR:** Formulating queries as fully learnable $(x, y, w, h)$ boxes proves that letting the network learn its initial priors drastically improves convergence and scale variance. |
| **Learnable `reg_scale`** | Change `requires_grad=True` for the `reg_scale` parameter. | Allows the model to dynamically adjust the curvature and maximum reach of the bounding box regression distribution. | **Generalized Focal Loss (GFL):** Learning the variance/scale of the bounding box distribution dynamically improves performance, especially for extreme object sizes. |
