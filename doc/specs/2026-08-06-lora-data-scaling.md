# LoRA Data Scaling Specification

**Goal:** Test how scaling target domain data (1%, 10%, 20%) affects mAP using a fixed LoRA rank, while retaining a minimum number of samples from the "anchor" datasets to preserve baseline mAP.

## Scope Limits
**Out of Scope:**
- Testing variable LoRA ranks (rank is fixed at 32).
- Modifying other model architectures.
- Changes to data augmentations or image resolutions.

**Prioritized Target Datasets for LoRA (Fine-tuned strictly ONE by ONE):**
1. `neuron-adhered-uncaged` (Both `20250108_neuron-adhered_10x_uncaged_4_class` and `20250305_neuron-adhered_10x_uncaged_4_class`)
2. `Hs675`
3. `Astrocytes` (`c8d1a`)
4. `u87`

## Architecture

We will add a dynamic subsampling feature to `Phase2MotifDataModule.setup()` and use a new `lora_frac` parameter in the Hydra configuration.

## Components

1.  **Config Parameters**:
    *   `data.lora_frac`: Determines the fraction of images to retain for target datasets.
    *   `data.target_datasets`: Explicit list of one or more datasets we are currently fine-tuning on.
    *   `data.anchor_datasets`: The baseline datasets. Defaults to the original `train_datasets` defined in the base coverage split.
    *   `data.anchor_samples_per_dataset`: Defaults to 4. **Rationale for 4:** Each original full image is sliced into ~32 crops. Retaining exactly 4 original images yields ~128 crops per anchor dataset. This provides a minimal "replay buffer" large enough to prevent catastrophic forgetting of the baseline domains, but small enough that it won't overwhelm the gradients of the new target dataset we are trying to learn.

2.  **Dynamic Subsampling in `Phase2MotifDataModule`**:
    *   In `setup(stage="fit")`, we partition the `train_datasets_objs`. Datasets matching `data.target_datasets` are targets; the rest (defaulting to the original `train_datasets`) are anchors.
    *   For anchor datasets, we sample exactly `anchor_samples_per_dataset` (default 4).
    *   For target datasets, we will sample `total_images * lora_frac` images.
    *   Subsampling is performed by modifying the dataset's internal `coco.imgs` and `coco.imgToAnns` dictionaries, ensuring reproducibility with a fixed random seed.

3.  **LoRA Configuration (PEFT)**:
    *   Rank `r` will be fixed at `32`.
    *   **No Backbone Finetuning**: We will use the `peft` library, but we will completely override any native `rfdetr` backbone LoRA logic. We will wrap the **entire** model with `get_peft_model(self.model, lora_config)`.
    *   To keep parameters absolutely minimal and entirely skip the backbone, PEFT will strictly target these precise module substrings: `["segmentation_head", "enc_out_class_embed", "enc_out_bbox_embed", "query_feat", "refpoint_embed", "class_embed", "bbox_embed"]`. The backbone modules (e.g., `backbone.0.encoder...`) will be fully frozen.

## Data flow

1.  Hydra loads config with `data.lora_frac`, `data.target_datasets`, and optionally overridden anchor settings.
2.  `Phase2MotifDataModule.setup("fit")` identifies targets (from `data.target_datasets`) and treats all other configured `train_datasets` as anchors.
3.  Anchor dataset subsampling mechanism:
    *   Iterate all images for a given anchor dataset.
    *   Use a fixed seed (e.g. 42) to randomly sample exactly `anchor_samples_per_dataset` (e.g. 4) images.
    *   Mutate the internal `coco.imgs` and `coco.imgToAnns` dictionaries to retain only those sampled keys.
4.  The target dataset is subsampled to `max(1, math.floor(total * lora_frac))` using the exact same random-key-sampling mechanism.
5.  `PreBuiltRFDETRModelModule` applies LoRA with `r=32` targeting the query selection and segmentation head (skipping the backbone entirely).
6.  During training, standard Lightning checkpoints are saved. When the "best" model is identified and saved by the callback, a lightweight PEFT adapter is also extracted and saved to a dedicated `adapters/` directory.

## Error handling and edge cases

*   **Fraction < 1 image**: If `max(1, math.floor(total * lora_frac))` is < 1, ensure at least 1 image is sampled.
*   **Missing `target_datasets`**: If not provided, but `lora_frac` is set, log a warning and apply `lora_frac` to all datasets.

## Testing approach

*   Run a quick test with `--debug` or `limit_train_batches=1` to ensure subsampling doesn't crash dataloaders.
*   Verify log outputs indicate the correct number of samples retained per dataset.

## Hyperparameters
Based on common practices for LoRA on vision/transformer models (e.g., maintaining high effective learning rates since few parameters are trained):
*   **Learning Rate Sweep**: Before full scaling experiments, we will run a quick sweep on the 10% data split for the first dataset to determine the optimal LR for these specific target modules (testing e.g., `1e-4`, `5e-4`, `1e-3`).
*   **Parameter Count Validation**: The explicitly targeted linear/embedding layers within the decoder and segmentation heads yield `864,640` trainable parameters against a frozen base of `36,429,183` params (~`2.37%` trainable capacity).
*   **Batch Size**: Maintained at the typical hardware maximum (e.g. `16` or `32`), as LoRA's small memory footprint easily permits it.
*   **Checkpointing Strategy**: 
    1. **Full Checkpoint**: We will rely on the default PyTorch Lightning `ModelCheckpoint`. This saves the **full model state_dict** (frozen base weights + `lora_A`/`lora_B` adapter weights). This integrates seamlessly with the existing `test_only` restore flow during the immediate evaluation phase.
    2. **Adapter-Only Checkpoint**: To enable dynamic adapter switching at inference (the primary benefit of LoRA), we will also save just the lightweight adapter weights. We will implement this by hooking into Lightning's `on_save_checkpoint` (or creating a custom ModelCheckpoint callback wrapper) to call PEFT's `save_pretrained()`. The adapter directory will be named explicitly using the target dataset and LoRA config (e.g., `adapters/<target_dataset>_r32_frac<lora_frac>_best`).
*   These will be configured via standard Hydra overrides when invoking the script.
## Documentation impact
- Feature / user-facing docs introduced: none
- Materially amended existing docs: none
- Derived / memory docs invalidated: none
