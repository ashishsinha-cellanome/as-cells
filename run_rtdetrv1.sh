#!/bin/bash
#SBATCH --job-name=rtdetrv1
#SBATCH --account=aip-robsc
#SBATCH --nodes=1
#SBATCH --mail-user=ashish.sinha@amii.ca
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
##SBATCH --exclusive
#SBATCH --mem-per-gpu=32G
#SBATCH --array=0-26
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

ACCOUNT="aip-robsc"
# --- Account Handling ---
# If not running under SLURM, submit this script to sbatch
if [ -z "$SLURM_JOB_ID" ]; then
    echo "Submitting job with account: $ACCOUNT"
    sbatch --account=$ACCOUNT "$0"
    exit
fi

# --- Training Logic (Only runs inside SLURM) ---
set -e

# module load python/3.10 cuda/12.2

# Ensure uv is in PATH (common install location)
export PATH="$HOME/.cargo/bin:$PATH"

echo "Job started at: $(date)"
echo "Account used: $SLURM_JOB_ACCOUNT"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

LRS=("5e-4" "5e-5" "1e-5")
FREEZE_STAGES=("2" "4" "0")  # ResNet stages: 0 (train all), 2 (partial freeze), 4 (freeze all)
SCHEDULERS=("onecycle" "step" "cosine_warmup")
CONFIGS=()
for lr in "${LRS[@]}"; do
    for stage in "${FREEZE_STAGES[@]}"; do
        for sched in "${SCHEDULERS[@]}"; do
            CONFIGS+=("optimizer.optimizer.lr=${lr} model.backbone.freeze_at_stage=${stage} scheduler=${sched}")
        done
    done
done

# 3. SELF-SUBMISSION LOGIC
# If $SLURM_ARRAY_TASK_ID is empty, we are running this from the login node
# if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
#     TOTAL_JOBS=${#CONFIGS[@]}
#     MAX_INDEX=$((TOTAL_JOBS - 1))
    
#     echo "Calculated $TOTAL_JOBS total combinations."
#     echo "Submitting SLURM array job with range 0-$MAX_INDEX..."
    
#     # Submit THIS script to SLURM, passing the dynamic array range
#     sbatch --array=0-$MAX_INDEX "$0"
    
#     # Exit the "submitter" script so it doesn't run the training command
#     exit 0
# fi

# ==============================================================================
# 4. WORKER LOGIC
# If we reach here, we are on a compute node running as a specific array task!
# ==============================================================================

# Extract the specific config string for this task's ID
MY_CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "========================================================"
echo "Starting SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Assigned Configuration: $MY_CONFIG"
echo "========================================================"

# Run the training command
srun uv run train_rt_detr_v2.py \
    data=vulcan \
    model=rtdetr_v1 \
    model/backbone=resnet50 \
    model.backbone.train_backbone=True \
    trainer.max_epochs=100 \
    ${MY_CONFIG}

# srun uv run train_rt_detr_v2.py data=vulcan model=rtdetr_v1 model/backbone=resnet50  model.backbone.train_backbone=True model.backbone.freeze_at_stage=2 scheduler=onecycle optimizer.optimizer.lr=5e-5 trainer.max_epochs=100
# srun uv run train_rt_detr_v2.py data=vulcan model=rtdetr_v1 model/backbone=resnet50  model.backbone.train_backbone=True model.backbone.freeze_at_stage=2 scheduler=step optimizer.optimizer.lr=5e-5 trainer.max_epochs=100
# srun uv run train_rt_detr_v2.py data=vulcan model=rtdetr_v1 model/backbone=resnet50  model.backbone.train_backbone=True model.backbone.freeze_at_stage=2 scheduler=cosine_warmup optimizer.optimizer.lr=5e-5 trainer.max_epochs=100


# srun uv run train_rt_detr_v2.py data=vulcan model=rtdetr_v1 model/backbone=resnet50  model.backbone.train_backbone=True model.backbone.freeze_at_stage=2 optimizer.optimizer.use_param_groups=True scheduler=onecycle optimizer.optimizer.lr=3e-4 \
# initialization.load_from_checkpoint="/project/aip-robsc/asinha/cellanome/DATA/checkpoints/rtdetr_v1_resnet50_2026-02-24_4071863_08-02/ckpts/rtdetr-ema-06-val_map0.5387.ckpt" trainer.max_epochs=100

# srun uv run train_rt_detr_v2.py data=vulcan model=rtdetr_v1 model/backbone=resnet50  model.backbone.train_backbone=True model.backbone.freeze_at_stage=2 optimizer.optimizer.use_param_groups=False scheduler=onecycle optimizer.optimizer.lr=3e-4 \
# initialization.load_from_checkpoint="/project/aip-robsc/asinha/cellanome/DATA/checkpoints/rtdetr_v1_resnet50_2026-02-24_4071863_08-02/ckpts/rtdetr-ema-06-val_map0.5387.ckpt" trainer.max_epochs=100
