#!/bin/bash
#SBATCH --job-name=rtdetrv2
#SBATCH --account=aip-robsc
#SBATCH --nodes=1
#SBATCH --mail-user="ashishsinha108@gmail.com"
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
##SBATCH --exclusive
#SBATCH --mem-per-gpu=24G
#SBATCH --array=0-95
#SBATCH --time=3-23:59:59
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

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

LRS=("5e-4"  "3e-5")
FREEZE_STAGES=("4" "0" "2")  # ResNet stages: 0 (train all), 2 (partial freeze), 4 (freeze all)
# SCHEDULERS=("onecycle" "step" "cosine_warmup")
SCHEDULERS=("onecycle" "multistep")
BACKBONES=("resnet50" "dinov2")
DATA_CONFIG=("vulcan" "vulcan_no300_eval_train_plus_valgt300")
NUM_QUERIES=("300"  "600")
CONFIGS=()

# loop over above configs
for data in "${DATA_CONFIG[@]}"; do
    for stage in "${FREEZE_STAGES[@]}"; do
        for sched in "${SCHEDULERS[@]}"; do
            for lr in "${LRS[@]}"; do
                for backbone in "${BACKBONES[@]}"; do
                    for num_query in "${NUM_QUERIES[@]}"; do
                        CONFIGS+=("optimizer.optimizer.lr=${lr} model.backbone.freeze_at_stage=${stage} scheduler=${sched} model/backbone=${backbone} model.rtdetr.num_queries=${num_query} data=${data}")
                    done
                done
            done
        done
    done
done
# for lr in "${LRS[@]}"; do
#     for stage in "${FREEZE_STAGES[@]}"; do
#         for sched in "${SCHEDULERS[@]}"; do
#         	for model in "${MODELS[@]}"; do
#     			for data_path in "${DATA_CONFIG[@]}"; do
#                     CONFIGS+=("optimizer.optimizer.lr=${lr} model.backbone.freeze_at_stage=${stage} scheduler=${sched} model=${model} data=${data_path}")
# 				done
# 			done
#         done
#     done
# done

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
    model=rtdetr_v2 \
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
