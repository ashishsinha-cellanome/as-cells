#!/bin/bash
#SBATCH --job-name=mask2former
#SBATCH --account=aip-robsc
#SBATCH --nodes=1
#SBATCH --mail-user=ashish.sinha@amii.ca
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
##SBATCH --exclusive
#SBATCH --mem-per-gpu=32G
#SBATCH --array=0-3
#SBATCH --time=2-10:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# --- Training Logic (Only runs inside SLURM) ---
set -e

# module load python/3.10 cuda/12.2

# Ensure uv is in PATH (common install location)
export PATH="$HOME/.cargo/bin:$PATH"

echo "Job started at: $(date)"
echo "Account used: $SLURM_JOB_ACCOUNT"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# LRS=("5e-4" "5e-5" "1e-5")
LRS=("3e-4" "5e-5")
# MODEL_SIZE=("medium" "base" "small")  # ResNet stages: 0 (train all), 2 (partial freeze), 4 (freeze all)
# MODEL_SIZE=("medium" "base")
# SCHEDULERS=("onecycle" "step" "cosine_warmup")
SCHEDULERS=("onecycle" "multistep")
DATA_CONFIG=("vulcan_no300_eval_train_plus_valgt300")
CONFIGS=()
for lr in "${LRS[@]}"; do
  for sched in "${SCHEDULERS[@]}"; do
    for data_path in "${DATA_CONFIG[@]}"; do
      CONFIGS+=("model.input_size=512 optimizer.optimizer.lr=${lr} scheduler=${sched} data=${data_path}")
    done
  done
done

# Extract the specific config string for this task's ID
MY_CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "========================================================"
echo "Starting SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Assigned Configuration: $MY_CONFIG"
echo "========================================================"

# Run the training command
srun uv run train_mask2former.py \
  model=mask2former \
  trainer.max_epochs=50 \
  data.batch_size=32 \
  model.mask2former.num_queries=300 \
  ${MY_CONFIG}

# srun uv run train_rt_detr_v2.py data=vulcan model=rtdetr_v1 model/backbone=resnet50  model.backbone.train_backbone=True model.backbone.freeze_at_stage=2 scheduler=onecycle optimizer.optimizer.lr=5e-5 trainer.max_epochs=100
# srun uv run train_rt_detr_v2.py data=vulcan model=rtdetr_v1 model/backbone=resnet50  model.backbone.train_backbone=True model.backbone.freeze_at_stage=2 scheduler=step optimizer.optimizer.lr=5e-5 trainer.max_epochs=100
# srun uv run train_rt_detr_v2.py data=vulcan model=rtdetr_v1 model/backbone=resnet50  model.backbone.train_backbone=True model.backbone.freeze_at_stage=2 scheduler=cosine_warmup optimizer.optimizer.lr=5e-5 trainer.max_epochs=100

# srun uv run train_rt_detr_v2.py data=vulcan model=rtdetr_v1 model/backbone=resnet50  model.backbone.train_backbone=True model.backbone.freeze_at_stage=2 optimizer.optimizer.use_param_groups=True scheduler=onecycle optimizer.optimizer.lr=3e-4 \
# initialization.load_from_checkpoint="/project/aip-robsc/asinha/cellanome/DATA/checkpoints/rtdetr_v1_resnet50_2026-02-24_4071863_08-02/ckpts/rtdetr-ema-06-val_map0.5387.ckpt" trainer.max_epochs=100

# srun uv run train_rt_detr_v2.py data=vulcan model=rtdetr_v1 model/backbone=resnet50  model.backbone.train_backbone=True model.backbone.freeze_at_stage=2 optimizer.optimizer.use_param_groups=False scheduler=onecycle optimizer.optimizer.lr=3e-4 \
# initialization.load_from_checkpoint="/project/aip-robsc/asinha/cellanome/DATA/checkpoints/rtdetr_v1_resnet50_2026-02-24_4071863_08-02/ckpts/rtdetr-ema-06-val_map0.5387.ckpt" trainer.max_epochs=100
