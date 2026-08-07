# LoRA Data Scaling Implementation Plan

> **REQUIRED SUB-SKILL:** Use the subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Test how scaling target domain data (1%, 10%, 20%) affects mAP using a fixed LoRA rank (r=32), while retaining exactly 4 images from the "anchor" datasets to preserve baseline mAP without diluting gradients.

**Architecture:** We bypass the backbone and inject PEFT adapters strictly into the decoder and segmentation heads (query selection, bounding box, class embed, and mask features) utilizing ~2.37% of model parameters. Dynamic dataset subsampling is injected into `Phase2MotifDataModule.setup("fit")` using configurable fractions. 

**Tech Stack:** PyTorch, Lightning, PEFT, Hydra
**Spec:** `.worktrees/lora-scaling-experiments/doc/specs/2026-08-06-lora-data-scaling.md`

---

## Files

**Create:**
- `configs/data/coverage_splits/lora_finetune_mix.yaml`
- `tests/test_lora_subsampling.py` (Local debugging script)

**Modify:**
- `train_rfdetr_phase2.py`

## Wave 1 — Configuration & Dynamic Subsampling

### Task 1: Create LoRA Finetune Config

**TDD scenario:** New feature — full TDD cycle

**Files:**
- Create: `configs/data/coverage_splits/lora_finetune_mix.yaml`

- [ ] **Step 1: Write the minimal config implementation**

  ```yaml
  # @package _global_
  defaults:
    - default@data
  data:
    lora_frac: 0.1
    anchor_samples_per_dataset: 4
    target_datasets: []
    anchor_datasets:
      - 20240516_DC-adhered_10x_caged_4_class
      - 20240422_neuron-adhered_10x_uncaged_4_class
      - 20240703_neuron-adhered_10x_caged_4_class
      - 20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class
      - 231212_imr90_multichannel_overlay_4_class
      - 20250227_preadipocytes-adhered_10x_caged_4_class
      - 20240624_mc38_10x_uncaged_4_class
      - 20240509_Hs675Tfibroblasts_10x_caged_4_class
    train_datasets: ${data.anchor_datasets}
    test_datasets:
      - 20240422_neuron-adhered_10x_uncaged_4_class
      - 20240509_Hs675Tfibroblasts_10x_caged_4_class
      - 20240509_hela-adhered_10x_caged_4_class
      - 20240515_DC-adhered_10x_caged_4_class
      - 20240516_DC-adhered_10x_caged_4_class
      - 20240624_mc38_10x_caged_4_class
      - 20240624_mc38_10x_uncaged_4_class
      - 20240625_mc38_10x_caged_4_class
      - 20240703_neuron-adhered_10x_caged_4_class
      - 20240704_neuron-adhered_10x_caged_4_class
      - 20240905_u87-adhered_10x_caged_4_class
      - 20240924_enteric-glia-adhered_10x_uncaged_4_class
      - 20241212_preadipocytes-adhered_10x_uncaged_4_class
      - 20250108_neuron-adhered_10x_uncaged_4_class
      - 20250227_preadipocytes-adhered_10x_caged_4_class
      - 20250305_neuron-adhered_10x_uncaged_4_class
      - 20250820_c8d1a_astrocytes-adherent_10x_caged_4_class
      - 20250917_moc22-adhered_10x_caged_4_class
      - 20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class
      - 231212_imr90_multichannel_overlay_4_class
      - 240213_imr90_multichannel_overlay_4_class
  ```

- [ ] **Step 2: Commit**

  ```bash
  git add configs/data/coverage_splits/lora_finetune_mix.yaml
  git commit -m "add lora_finetune_mix config for dataset subsampling"
  ```

### Task 2: Implement Data Subsampling in `train_rfdetr_phase2.py`

**TDD scenario:** Modifying tested code — use judgment

**Files:**
- Modify: `train_rfdetr_phase2.py`

- [ ] **Step 1: Modify `Phase2MotifDataModule.setup()` to apply `lora_frac`**

  In `train_rfdetr_phase2.py`, modify `Phase2MotifDataModule.setup` to add the subsampling logic:

  ```python
      def setup(self, stage=None):
          # Build datasets using parent setup logic
          super().setup(stage)
          
          # --- NEW: Subsample crops to prevent catastrophic forgetting ---
          lora_frac = getattr(self.config.data, "lora_frac", None)
          if stage in ("fit", None) and lora_frac is not None:
              import random
              import math
              from utils.distributed_utils import rank_zero_print
              
              # Subsampling must be fully deterministic
              random.seed(self.config.get("seed", 42))
              
              target_datasets = list(getattr(self.config.data, "target_datasets", []))
              anchor_datasets = list(getattr(self.config.data, "anchor_datasets", []))
              anchor_samples = int(getattr(self.config.data, "anchor_samples_per_dataset", 4))
              
              if not target_datasets:
                  rank_zero_print("[WARNING] No target_datasets specified! Applying lora_frac to ALL datasets.")
              
              for ds_obj in self.train_datasets_objs:
                  dataset_name = getattr(ds_obj, "name", "") 
                  if not dataset_name: # fallback if name isn't attached
                      dataset_name = ds_obj.dataset_name if hasattr(ds_obj, "dataset_name") else ""
  
                  img_ids = list(ds_obj.coco.imgs.keys())
                  total_imgs = len(img_ids)
                  
                  if not target_datasets or any(t in dataset_name for t in target_datasets):
                      # Target dataset: apply fraction
                      keep_count = max(1, math.floor(total_imgs * lora_frac))
                      sampled_ids = set(random.sample(img_ids, keep_count))
                      rank_zero_print(f"[LoRA Subsample] Target Domain {dataset_name}: keeping {keep_count}/{total_imgs} images (frac={lora_frac})")
                  else:
                      # Anchor dataset: apply replay buffer constraint
                      keep_count = min(total_imgs, anchor_samples)
                      sampled_ids = set(random.sample(img_ids, keep_count))
                      rank_zero_print(f"[LoRA Subsample] Anchor Domain {dataset_name}: keeping {keep_count}/{total_imgs} images (replay buffer)")
                      
                  # Filter the internal coco dictionary inplace
                  ds_obj.coco.imgs = {k: v for k, v in ds_obj.coco.imgs.items() if k in sampled_ids}
                  if hasattr(ds_obj, 'ids'):
                      ds_obj.ids = list(sampled_ids)
                  
                  # Filter annotations map
                  ds_obj.coco.imgToAnns = {k: ds_obj.coco.imgToAnns.get(k, []) for k in sampled_ids}
          # ---------------------------------------------------------------
  ```

- [ ] **Step 2: Commit**

  ```bash
  git add train_rfdetr_phase2.py
  git commit -m "implement dynamic data subsampling for lora fine-tuning"
  ```

## Wave 2 — PEFT LoRA Module Injection

### Task 3: Modify `PreBuiltRFDETRModelModule._apply_lora`

**TDD scenario:** Modifying tested code — use judgment

**Files:**
- Modify: `train_rfdetr_phase2.py`

- [ ] **Step 1: Replace native LoRA with global PEFT regex targeting decoder/head**

  In `train_rfdetr_phase2.py`, inside `PreBuiltRFDETRModelModule._apply_lora`, replace the hardcoded `target_modules` list and the backbone-specific injection. Note the explicit regex targeting `pwconv1`, `query_features_proj`, `spatial_features_proj`, `refpoint_embed`, `query_feat`, and `layers.\d+` for MLPs.

  ```python
      def _apply_lora(self) -> None:
          """Customizable LoRA injection overriding the upstream hardcoded method."""
          from peft import LoraConfig, get_peft_model
          lc = self._lora_cfg
          
          # Exact regex matching leaf nodes for segmentation head and decoder object queries.
          target_modules = lc.get("target_modules", r".*(pwconv1|spatial_features_proj|query_features_proj|refpoint_embed|query_feat|(class_embed|bbox_embed).*layers\.\d+)$")
          exclude_modules = lc.get("exclude_modules", r".*(dwconv|norm|bn|act|relu|gelu|backbone).*")
          
          lora_config = LoraConfig(
              r=lc.get("r", 32),
              lora_alpha=lc.get("alpha", 32),
              use_dora=lc.get("use_dora", False),
              target_modules=target_modules,
              exclude_modules=exclude_modules,
              lora_dropout=lc.get("dropout", 0.05),
              bias="none"
          )
          
          # We wrap the ENTIRE model so PEFT can find the decoder/segmentation modules
          self.model = get_peft_model(self.model, lora_config)
          self.model.print_trainable_parameters()
  ```

- [ ] **Step 2: Commit**

  ```bash
  git add train_rfdetr_phase2.py
  git commit -m "update PEFT application to target decoder and segmentation head"
  ```

### Task 4: Extract and Save Standalone Adapter

**TDD scenario:** Modifying tested code — use judgment

**Files:**
- Modify: `train_rfdetr_phase2.py`

- [ ] **Step 1: Hook into Trainer for saving adapters**

  In `train_rfdetr_phase2.py`, directly under the definition of `module.config = config` (around line ~359), add an `on_save_checkpoint` hook to the module so that whenever lightning saves the full model, PEFT also saves the lightweight adapter.

  ```python
      # 3. Create Module
      module = PreBuiltRFDETRModelModule(
          model_config=model_config, 
          train_config=train_config, 
          inner_model=inner_model, 
          lora_cfg=lora_cfg
      )
      module.config = config
      
      # Hook into the module to save the PEFT adapter alongside the main checkpoint
      def save_adapter_hook(checkpoint_dir: str):
          if hasattr(module.model, "save_pretrained"):
              import os
              target = getattr(config.data, "target_datasets", ["unknown_target"])[0]
              frac = getattr(config.data, "lora_frac", "unknown")
              adapter_dir = os.path.join("adapters", f"{target}_r{lora_cfg.get('r', 32)}_frac{frac}")
              os.makedirs(adapter_dir, exist_ok=True)
              module.model.save_pretrained(adapter_dir)
              rank_zero_print(f"Saved lightweight PEFT adapter to {adapter_dir}")
              
      # Lightning modules have an `on_save_checkpoint` method we can override
      original_on_save = module.on_save_checkpoint
      def custom_on_save_checkpoint(checkpoint):
          original_on_save(checkpoint)
          # Use rank zero only
          if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
              save_adapter_hook("")
      module.on_save_checkpoint = custom_on_save_checkpoint
  ```

- [ ] **Step 2: Commit**

  ```bash
  git add train_rfdetr_phase2.py
  git commit -m "add peft save_pretrained hook to lightning checkpointing"
  ```
