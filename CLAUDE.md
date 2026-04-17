# CLAUDE.md (Cellanome `as-cells` Project)

**Goal:** Detect cells and tiny objects (beads) in brightfield microscopy images.

## 🚨 AI ASSISTANT CONSTRAINTS (CRITICAL)
1. **NEVER delete the branch `ashish` or force-merge it to `main`.**
2. **ALWAYS run tests/training scripts on the `vulcan` server via SSH (`ask1gpu`, `sbatch`, `salloc`).** NEVER run them on the local macOS machine.
3. **DO NOT assume dataset properties (e.g., image resolutions)**. Write probing scripts. All images are **672x672** for training/validation (Full-scale 4512x4512 is for future inference only).
4. **ALWAYS use superpowers:** Invoke the `using-superpowers` skill before taking any action. If a task matches a skill description, you MUST use the `skill` tool to load it.
5. **ALWAYS use `subagent-driven-development`:** When executing multi-step plans, use the Subagent process (dispatching fresh subagents per task with two-stage reviews). Do NOT execute tasks inline yourself.
6. **ALWAYS use `systematic-debugging`:** When encountering bugs or test failures, complete the 4-phase root cause investigation before proposing any fixes.
7. **ALWAYS use `brainstorming`:** Before building features, load the brainstorming skill to present designs and get approval.

## Architecture & Models
- **Tech Stack:** PyTorch, Lightning, Hydra (configs), wandb, `uv` (package management).
- **Models:** RT-DETR v1/v2, RF-DETR, YOLOv5, **Mask2Former**.
- **Backbones:** DINOv2, ResNets.
- **Mask2Former FPNs:** `adapter` (Custom ViT-Adapter extracting true spatial strides 4/8/16/32), `sfp`, `fused`, `tiny`.
  - *Note:* FPN weights need explicit Kaiming init. Pretrained decoder uses 0.1x LR to prevent catastrophic forgetting.
- **Classes:** 0:cell, 1:bead, 2:cell-adhered (maps to cell), 3:soma (maps to cell).

## Data & Configs
- **Data paths:** `/project/aip-robsc/asinha/cellanome/DATA/TRAINING_DATA`
- **Splits:** COCO format. `vulcan_no300_eval` excludes images with >300 bboxes from val/test.
- **Configs (Hydra):** Under `configs/`. Uses `@package _global_` and `${oc.eval:...}`.

## Commands Reference
```bash
# UV & Linting
uv sync; uv add <pkg>; uv run ruff check . --fix

# Training (Mask2Former, RT-DETR, etc.)
uv run train_mask2former.py model=mask2former model/backbone=dinov2_mask2former model.backbone.fpn_type=adapter data=vulcan
uv run train_rt_detr_v2.py model=rtdetr_v2 data=vulcan

# SLURM Execution
sbatch run_rfdetr.sh
# For interactive vulcan GPU: salloc --account=aip-robsc --nodes=1 --gpus-per-node=1 ...

# Eval & Inference
uv run evaluate_all_models.py
uv run inference.py initialization.load_from_checkpoint=ckpt.pt data.path=/path/to/data
```