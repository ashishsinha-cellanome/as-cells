# CLAUDE.md (Cellanome `as-cells` Project)

**Goal:** Detect cells and tiny objects (beads) in brightfield microscopy images.

## 🚨 AI ASSISTANT CONSTRAINTS (CRITICAL)

0. STOP being a kiss-ass, and stop praising or apologizing. Just answer to the point.
1. **NEVER delete the branch `ashish` or force-merge it to `main`.**
2. **ALWAYS check the hostname before running tests/training scripts to decide whether to run  `vulcan` or `denvr: odin*` server. if running on vulcan (SLURM) viaa SSH (`ask1gpu`, `sbatch`, `salloc`) or when running on odin machines, also use ssh with simply `uv run <script.py> <options>`.** NEVER run them on the local macOS machine.
3. **DO NOT assume dataset properties (e.g., image resolutions)**. Write probing scripts. All images are **672x672** for training/validation (Full-scale 4512x4512 is for future inference only).
4. **ALWAYS use superpowers:** Invoke the `using-superpowers` skill before taking any action. If a task matches a skill description, you MUST use the `skill` tool to load it.
5. **ALWAYS use `subagent-driven-development`:** When executing multi-step plans, use the Subagent process (dispatching fresh subagents per task with two-stage reviews). Do NOT execute tasks inline yourself.
6. **ALWAYS use `systematic-debugging`:** When encountering bugs or test failures, complete the 4-phase root cause investigation before proposing any fixes.
7. **ALWAYS use `brainstorming`:** Before building features, load the brainstorming skill to present designs and get approval.

## Architecture & Models

- **Tech Stack:** PyTorch, Lightning, Hydra (configs), wandb, `uv` (package management).
- **Classes:** 0:cell, 1:bead, 2:cell-adhered (maps to cell), 3:soma (maps to cell).

## Data & Configs

- **Data paths:** `/mnt/direct-attached/PHASE2` (Currently used for motif experiments, generalization, embedding computation, coverage analysis, and zero-shot robustness instead of `TRAINING_DATA`).
- **Old Data paths:** `/project/aip-robsc/asinha/cellanome/DATA/TRAINING_DATA` (Do not use for motifs)
- **Splits:** COCO format. `vulcan_no300_eval` excludes images with >300 bboxes from val/test.
- **Configs (Hydra):** Under `configs/`. Uses `@package _global_` and `${oc.eval:...}`.

## Commands Reference

```bash
# UV & Linting
uv sync; uv add <pkg>; uv run ruff check . --fix

# Training 

1. ALways use `uv` for running python files such as:
uv run <python script> <script args>

# SLURM Execution
ALWAYS check which machine the code is running on (slurm cluster or a normal GPU or local MAC) to figure appropriate configs, and run commands.

For interactive vulcan GPU: salloc --account=aip-robsc --nodes=1 --gpus-per-node=1 ...

# Eval & Inference
uv run inference.py initialization.load_from_checkpoint=ckpt.pt data.path=/path/to/data
```
