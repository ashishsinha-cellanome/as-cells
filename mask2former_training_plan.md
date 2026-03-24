# Mask2Former Training Integration Plan

## Context

The user wants to train Mask2Former on their brightfield microscopy dataset (cells, beads, soma, cell-adhered). The dataset is stored on vulcan server at `~/cellanome/DATA/TRAINING_DATA` with images, masks (pkl files), and COCO-style labels for train/valid/test splits.

**Current State:**
- Mask2Former inference code already exists (`models/mask2former_model.py`, `Inference/mask2former_model.py`)
- Uses DINOv2 backbone with Mask2Former for instance segmentation
- No training script exists for Mask2Former
- Existing models (RF-DETR, RT-DETR, YOLOv5) have complete training pipelines with:
  - LightningDataModule for data loading
  - LightningModule for training
  - Hydra config files for hyperparameters
  - WandB logging integration
  - SLURM job submission scripts

## Problem Statement

Mask2Former training needs to be integrated into the existing training framework following the same patterns as RF-DETR/RT-DETR/YOLOv5, with support for:
- Instance segmentation (masks + boxes)
- DINOv2 backbone (frozen or fine-tuned)
- COCO metrics evaluation (mAP, mask mAP)
- Hydra configuration management
- WandB experiment tracking
- SLURM cluster job submission

## Key Findings from Codebase Exploration

### Existing Architecture

1. **Data Pipeline** (`data/`):
   - `RFDETRDataModule` - Uses RF-DETR's native COCO transforms
   - `COCODataModule` - Generic COCO with Albumentations transforms
   - `YOLOv5DataModule` - Converts COCO to YOLO format

2. **Model Architecture** (`models/`):
   - `mask2former_model.py` - Inference-only, uses HuggingFace transformers
   - `rf_detr_model.py` / `rt_detr_model.py` - Detection models
   - Lightning modules wrap models for training

3. **Training Scripts**:
   - `train_rf_detr.py` - Complete training entry point
   - `train_rt_detr_v2.py` - Similar structure

4. **Configs** (`configs/`):
   - `model/rfdetr.yaml`, `model/rtdetr_v2.yaml` - Model-specific configs
   - Hydra defaults in `config.yaml`

### Mask2Former Specifics

From existing inference code:
- Uses `Mask2FormerForUniversalSegmentation` from transformers
- DINOv2 backbone (facebook/dinov2-base)
- Input processor: `Mask2FormerImageProcessor`
- Output: instance masks + labels + scores
- Post-processing: `largest_blob_by_area()` cleanup for fragmented masks

## Recommended Approach

### 1. Data Module (`data/mask2former_data_module.py`)

**Reuse Pattern:** Follow `COCODataModule` structure with modifications for instance segmentation:
- Load COCO annotations with mask RLE decoding
- Apply Mask2Former-specific transforms (resize, normalize with mean/std)
- Return: `pixel_values`, `masks`, `labels`, `boxes`, `image_id`

**Key difference from detection:**
- Must include ground truth masks (not just boxes)
- Masks need to be processed to binary format per instance

### 2. Lightning Module (`models/mask2former_lightning_module.py`)

**Confirmed from HuggingFace Docs:**

**Loss Auto-Computation:** YES - `Mask2FormerForUniversalSegmentation` auto-computes loss when `labels` is provided:
```python
# Training
outputs = model(pixel_values=pixel_values, labels=labels)
loss = outputs.loss  # Auto-computed: cross_entropy + dice + bbox + giou
```

**Labels format** (per image, passed via `segmentation_maps` in image processor):
```python
# During training, prepare masks as:
{
    "masks": (num_instances, H, W) binary masks,  # Required
    "labels": (num_instances,) class IDs,          # Required
    "boxes": (num_instances, 4) xywh format,       # Optional (for box loss)
}
# Pass through processor: processor(images, segmentation_maps=masks, return_tensors="pt")
```

**Post-processing** (confirmed API):
```python
processed = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,           # Probability threshold
    mask_threshold=0.5,      # Binary mask threshold
    target_sizes=batch_image_sizes,
    return_binary_maps=True  # Get (num_instances, H, W) tensor
)
# Returns: List[Dict] per image
#   - "segmentation": (num_instances, H, W) or RLE if return_coco_annotation=True
#   - "segments_info": List[Dict] with id, label_id, score
```

**COCO Evaluation:**
- Use `pycocotools` for both bbox mAP and segm mAP
- Support EMA validation (optional)
- Support sliced inference via SAHI (reuse `utils/sahi_eval.py`)
- For COCO RLE format: use `return_coco_annotation=True` in post_process

### 3. Model Wrapper (`models/mask2former_model.py` - training version)

**Reuse Pattern:** Similar to `rf_detr_model.py`:
- Wrap `Mask2FormerForUniversalSegmentation`
- Support DINOv2 backbone loading
- Expose `forward(pixel_values, labels)` for training
- Expose `predict(pixel_values)` for inference

### 4. Training Script (`train_mask2former.py`)

**Reuse Pattern:** Mirror `train_rf_detr.py`:
- Hydra decorator `@hydra.main`
- Setup logger, callbacks, datamodule, model
- Trainer fit + test flow
- WandB integration
- SLURM environment detection

**Key sections:**
```python
- _setup_logger() -> WandBLogger
- _setup_callbacks() -> ModelCheckpoint, EarlyStopping, EMACallback
- main() -> Trainer fit/test
```

### 5. Config Files

**New files to create:**
1. `configs/model/mask2former.yaml` - Model architecture, queries, loss weights, EMA
2. `configs/model/backbone/dinov2_mask2former.yaml` - DINOv2 backbone config (`@package model.backbone`)
3. `configs/model/backbone/resnet50_mask2former.yaml` - ResNet50 backbone config (`@package model.backbone`)
4. `configs/peft/lora.yaml` - Reusable LoRA hyperparams (`@package peft`)

**Config structure:**

`configs/peft/lora.yaml`:
```yaml
# @package peft
enabled: false
type: lora
rank: 8
alpha: 16
dropout: 0.1
# target_modules NOT here - model-specific, defined in backbone config

# --- LoRA Target Module Recommendations ---
# Architecture       | Best for OD      | Modules to target
# -------------------|------------------|------------------------------------------
# ViT/DINOv2         | q_proj, v_proj   | ["q_proj", "v_proj"]
#                    | All attention    | ["q_proj", "k_proj", "v_proj", "out_proj"]
#                    | Minimal (fast)     | ["v_proj"] or ["q_proj"]
# ResNet             | Bottleneck convs | ["conv1", "conv2", "conv3"]
#                    | Last stage only  | ["layer4.0.conv1", "layer4.0.conv2", ...]
# Swin Transformer   | Attention        | ["attention.self.query", "attention.self.value"]
# DETR variants      | Cross-attention  | ["cross_attn.q_proj", "cross_attn.v_proj"]
# Mask2Former        | Pixel decoder    | ["pixel_level_module.decoder.0", ...]
#
# Notes:
# - q_proj, v_proj: Good balance of performance vs params (recommended default)
# - Adding k_proj: Small gains, more params
# - Adding out_proj: Diminishing returns for detection
# - For microscopy (small objects): consider including more layers
```

`configs/model/backbone/dinov2_mask2former.yaml`:
```yaml
# @package model.backbone
type: "dinov2"
pretrained_name_or_path: "facebook/dinov2-base"
name: "dinov2-base"

# Training mode: frozen | full_finetune | lora | finetune_from_layer
training_mode: "frozen"
finetune_from_layer: 10  # finetune transformer layers 10+, freeze 0-9

# PEFT target modules (model-specific)
peft_target_modules: ["q_proj", "v_proj", "key_proj", "value_proj"]
```

`configs/model/backbone/resnet50_mask2former.yaml`:
```yaml
# @package model.backbone
type: "resnet"
name: "resnet50"
pretrained_name_or_path: "microsoft/resnet-50"

# Training mode: frozen | full_finetune | lora | finetune_up_to_stage | finetune_stages
training_mode: "frozen"
finetune_up_to_stage: 4      # finetune stages 0-4 (stem + layer1-4)
finetune_stages: [3, 4]      # OR target specific stages only

# PEFT target modules (model-specific)
peft_target_modules: ["conv1", "conv2", "conv3"]  # bottleneck convs
```

`configs/model/mask2former.yaml`:
```yaml
name: mask2former_${model.backbone.name}
input_size: 672

mask2former:
  pretrain_weights: facebook/mask2former-swin-large-coco-instance
  num_queries: 100
  transformer_dims: 256

  # Training behavior (inherited from backbone.training_mode)
  use_ema: true
  ema_decay: 0.999
  ema_tau: 2000

# Merge PEFT config (optional override)
peft: ${peft}

label_map:
  0: "cell"
  1: "bead"
  2: "cell-adhered"
  3: "soma"

# Loss weights
loss_weight_dict:
  cross_entropy: 1.0
  dice: 1.0
  bbox: 5.0
  giou: 2.0
```

**Usage examples:**
```bash
# Frozen DINOv2
uv run train_mask2former.py model=mask2former model.backbone=dinov2_mask2former

# LoRA DINOv2 (inherits rank/alpha/dropout from peft/lora.yaml)
uv run train_mask2former.py model=mask2former model.backbone=dinov2_mask2former peft.enabled=true peft.rank=16

# ResNet50 finetune up to stage 3
uv run train_mask2former.py model=mask2former model.backbone=resnet50_mask2former model.backbone.training_mode=finetune_up_to_stage model.backbone.finetune_up_to_stage=3
```

### 6. SLURM Script (`run_mask2former.sh`)

**Reuse Pattern:** Mirror `run_rfdetr.sh`:
- Array job for hyperparameter sweeps
- Learning rate sweep
- Backbone size sweep (nano, small, base, large)
- Freeze vs fine-tune backbone

### 7. Utilities to Reuse

- `utils/ema.py` - EMA callback (already exists)
- `utils/sahi_eval.py` - Sliced inference evaluation
- `utils/distributed_utils.py` - DDP setup
- `utils/coco_eval_utils.py` - COCO metric computation
- `utils/train_utils.py` - BackupToNASCallback

## Architectural Decisions

### Decision 1: Backbone Strategy (User Specified)
**Backbone Options:**
- **DINOv2** (small/base/large/giant) with three training modes:
  - Frozen: All backbone params frozen (`requires_grad=False`)
  - Full finetune: All backbone params trainable
  - LoRA/PEFT: Low-rank adaptation on attention layers (via `peft` library)
- **ResNet50** with four training modes:
  - Frozen: All backbone params frozen
  - Full finetune: All backbone params trainable
  - Layer-wise finetune: Specific stages unfrozen (e.g., stage3, stage4 only)
  - Standard C4/C5 stride configuration (matching RT-DETR implementation)

**Config Pattern:** Mirror `rt_detr_v1.yaml` / `rt_detr_v2.yaml`:
```yaml
backbone:
  type: dinov2-base  # or resnet50
  training_mode: frozen  # frozen | full_finetune | lora | upto_stage_N

  # For ResNet50 - stage-wise finetuning (HF transformers structure)
  finetune_up_to_stage: 4  # finetune stages 0-4, freeze earlier
  # or
  finetune_stages: [3, 4]  # target specific stages only

  # For DINOv2/ViT - layer-wise finetuning
  finetune_from_layer: 10  # finetune transformer layers 10+, freeze earlier

# LoRA/PEFT config (separate reusable group: configs/models/peft.yaml)
# Imported by model configs that need PEFT
peft:
  enabled: false
  type: lora  # lora | adapter | prefix
  rank: 8
  alpha: 16
  dropout: 0.1

# Model-specific PEFT overrides (in model/backbone config):
# peft.target_modules: ["q_proj", "v_proj"]  # DINOv2 attention
# peft.target_modules: ["attention"]  # ResNet50 bottleneck
```

### Decision 2: Mask Processing
**Recommendation:** Use existing `largest_blob_by_area()` post-processing
- Already implemented in inference code
- Handles fragmented mask issue common in microscopy

### Decision 3: Input Size
**Recommendation:** Support multiple sizes via config
- 560x560 (RF-DETR default)
- 672x672 (DINOv2 native)
- Configurable via `model.input_size`

### Decision 4: Instance Mask Format
**Recommendation:** Store masks as RLE in COCO format, decode during training
- Matches existing dataset structure
- `pycocotools.mask.decode()` for conversion

### Decision 5: Priority Metric
**User Specified:** bbox mAP is the priority metric
- Primary checkpoint monitor: `val/map_bbox_50` or `val/map_bbox`
- Secondary metrics: mask mAP, AR
- COCO eval will compute both, but checkpoint selection prioritizes bbox

## Files to Create/Modify

### New Files:
1. `data/mask2former_data_module.py` - Data loading with masks
2. `models/mask2former_lightning_module.py` - Training wrapper
3. `train_mask2former.py` - Training entry point
4. `configs/model/mask2former.yaml` - Model config
5. `run_mask2former.sh` - SLURM submission script

### Modified Files:
1. `config.py` - Add training config entry for mask2former
2. `configs/config.yaml` - Add mask2former to defaults (optional)

## Implementation Steps

1. **Data Module** - Create `COCOInstanceSegDataset` with mask support
2. **Lightning Module** - Wrap Mask2Former model, implement train/val steps
3. **Training Script** - Mirror RF-DETR structure
4. **Config** - Define hyperparameters
5. **SLURM Script** - Job submission
6. **Test** - Run debug training, verify metrics

## Verification Plan

1. **Debug Mode:** `uv run train_mask2former.py debug=true`
2. **Single GPU:** Verify loss decreases, mAP improves
3. **Multi-GPU:** Test DDP with `trainer.devices=2`
4. **Resume:** Test checkpoint loading
5. **Eval:** Compare val mAP with inference script

## Suggestions/Improvements

### Suggestion 1: Unified Data Interface
Consider creating a base class for instance segmentation data modules that can be reused for future models (Mask R-CNN, etc.)

### Suggestion 2: Mask Visualization
Add mask overlay visualization to WandB logging (similar to bbox viz in RF-DETR)

### Suggestion 3: Gradual Unfreezing
Implement callback for gradual backbone unfreezing after N epochs

### Suggestion 4: Memory Efficiency
Use gradient checkpointing for large backbones (DINOv2-large)

---

## Questions for User

1. **Data format confirmation:** You mentioned masks are in pkl files. Are these:
   - RLE-encoded masks (COCO format)?
   - Binary masks (HxW per instance)?
   - Semantic masks (one mask per class)?

2. **Backbone preference:** Start with frozen DINOv2 or fine-tune from beginning?

3. **GPU resources:** How many GPUs per job? This affects batch size and gradient accumulation settings.

4. **Priority:** Which metric matters most - bbox mAP or mask mAP?

---

## Instructions for Next Session

### Before Starting Implementation

**Read these files first** (in order):
1. `models/mask2former_model.py` - Existing inference code (understand model loading, post-processing)
2. `data/rf_detr_data_module.py` - Data pipeline pattern
3. `models/rf_detr_lightning_module.py` - Training step, validation, loss logging pattern
4. `train_rf_detr.py` - Main training script structure
5. `configs/model/rfdetr.yaml` - Config structure
6. `configs/model/backbone/dinov2.yaml` - Backbone config pattern
7. `configs/model/backbone/resnet50.yaml` - ResNet backbone pattern

### Key Implementation Checklist

**Phase 1: Data Module**
- [ ] Create `data/mask2former_data_module.py`
- [ ] Implement COCO mask loading (RLE → binary mask conversion)
- [ ] Use `Mask2FormerImageProcessor` for preprocessing
- [ ] Test: `python -c "from data.mask2former_data_module import ..."`

**Phase 2: Lightning Module**
- [ ] Create `models/mask2former_lightning_module.py`
- [ ] Implement `training_step` with auto-loss from transformers
- [ ] Implement `validation_step` with post_process_instance_segmentation
- [ ] Implement `on_validation_epoch_end` with COCO metrics
- [ ] Add EMA support (reuse `utils/ema.py`)
- [ ] Add SAHI sliced eval support (reuse `utils/sahi_eval.py`)

**Phase 3: Training Script**
- [ ] Create `train_mask2former.py` mirroring `train_rf_detr.py`
- [ ] Hydra decorator, logger, callbacks, trainer setup
- [ ] Test debug mode: `uv run train_mask2former.py debug=true`

**Phase 4: Configs**
- [ ] Create `configs/peft/lora.yaml`
- [ ] Create `configs/model/mask2former.yaml`
- [ ] Create `configs/model/backbone/dinov2_mask2former.yaml`
- [ ] Create `configs/model/backbone/resnet50_mask2former.yaml`

**Phase 5: SLURM Script**
- [ ] Create `run_mask2former.sh` mirroring `run_rfdetr.sh`

### Verification Commands

```bash
# 1. Debug mode (single batch)
uv run train_mask2former.py debug=true data.limit_train_batches=1 data.limit_val_batches=1

# 2. Single GPU smoke test
uv run train_mask2former.py trainer.max_epochs=1 trainer.devices=1

# 3. Check checkpoint saved
ls -la checkpoints/

# 4. Resume from checkpoint
uv run train_mask2former.py initialization.load_from_checkpoint=checkpoints/xxx.ckpt

# 5. Test-only evaluation
uv run train_mask2former.py test_only=true initialization.load_from_checkpoint=xxx.ckpt
```

### Potential Blockers & Solutions

| Blocker | Solution |
|---------|----------|
| OOM on 4512x4512 images | Use gradient checkpointing (`trainer.gradient_checkpointing=True`) |
| Slow data loading | Increase `data.num_workers`, use `prefetch_factor` |
| Loss NaN | Reduce LR, check mask format, enable gradient clipping |
| COCO eval mismatch | Verify mask RLE encoding matches ground truth format |
| DDP hanging | Check `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1` env vars |

### Ask User If Unclear

1. **Mask format in pkl files** - How to decode? Show sample structure.
2. **LoRA target modules** - Which layers for DINOv2 vs ResNet50?
3. **Checkpoint format** - Save full model or just state_dict?
4. **WandB project name** - What project to use?

### Success Criteria

- Training starts without errors
- Loss decreases over first 10 epochs
- Validation mAP improves
- Checkpoints save correctly
- Can resume from checkpoint
- Multi-GPU (4 GPUs) scaling works
