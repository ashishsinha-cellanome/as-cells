# Minimal-Compute Adaptation Strategies for Zero-Shot Generalization

When adapting a base model (trained on the 6-Centroid core set) to an entirely unseen, out-of-distribution dataset (e.g., an extreme isolate cell line), full-scale fine-tuning can be computationally expensive and risks "Catastrophic Forgetting" of the original domains. 

Below are 5 advanced strategies for minimal-compute adaptation. For each strategy, a corresponding Python script has been generated (based on the original PyTorch Lightning/Hydra structure).

---

## 1. Decoder/Head-Only Fine-Tuning (Few-Shot)
**Script**: `finetune_decoder_only.py`
*   **Concept**: The visual features extracted by the DINOv2 backbone and the RF-DETR Transformer Encoder are highly generalized. We can freeze them completely. We only unfreeze the Transformer Decoder queries and the segmentation/bounding box heads.
*   **Why it works**: Drastically reduces the number of trainable parameters and GPU memory requirements. It forces the model to learn *how to query* the new domain's features without forgetting the fundamental visual representations of cells.
*   **Compute Cost**: Very Low.

## 2. Low-Rank Adaptation (LoRA) for RF-DETR
**Script**: `finetune_lora.py`
*   **Concept**: We freeze the entire base model and inject tiny, trainable low-rank matrices into the attention layers (Query, Key, Value projections) of the Transformer. 
*   **Why it works**: We only train a few megabytes of parameters. We can train a separate LoRA adapter for every distinct cell line and dynamically load them during inference. This is the modern PEFT (Parameter-Efficient Fine-Tuning) standard.
*   **Compute Cost**: Extremely Low.

## 3. Exponential Moving Average (EMA) Ensembling (Weight Averaging)
**Script**: `finetune_ema_ensemble.py`
*   **Concept**: Fine-tune the base model normally on the new dataset, but instead of just deploying the new weights, we mathematically average the new weights with the original base weights ($W_{final} = 0.5 \times W_{base} + 0.5 \times W_{new}$).
*   **Why it works**: This acts as a powerful regularizer against catastrophic forgetting. The model retains its generalized zero-shot capabilities from the base training while pulling in the specific local manifold data of the new cell line.
*   **Compute Cost**: Low to Medium (requires a normal fine-tuning pass, but no extra compute for the ensembling itself).

## 4. Test-Time Adaptation via Pseudo-Labeling
**Script**: `finetune_pseudo_label.py`
*   **Concept**: Pass the new, unlabeled dataset through the base model using heavy Test-Time Augmentation (TTA). Use high-confidence predictions to automatically generate a "pseudo-annotated" COCO JSON, and then briefly fine-tune the model on its own predictions.
*   **Why it works**: Adapts the model's batch normalization and feature maps to the new lighting, background, and noise of the target domain without requiring any human annotation.
*   **Compute Cost**: Low (Inference pass + 1-2 epochs of training).

## 5. Domain-Specific Query Prompts
**Script**: `finetune_domain_prompts.py`
*   **Concept**: Modify the architecture to accept a "Domain Prompt" (e.g., a one-hot vector for the cell line family) which is added directly to the object queries before they enter the Decoder. 
*   **Why it works**: Instead of training different models for different domains, a single model learns to dynamically shift its query search patterns based on the text/domain prompt provided during inference.
*   **Compute Cost**: Medium (Requires modifying the architecture and re-training).
