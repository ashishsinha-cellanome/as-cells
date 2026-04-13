#!/bin/bash
#SBATCH --job-name=rfdetr_seg
#SBATCH --account=aip-robsc
#SBATCH --nodes=1
#SBATCH --mail-user=ashish.sinha@amii.ca
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=32G
#SBATCH --array=0-11
#SBATCH --time=3-10:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -e

export PATH="$HOME/.cargo/bin:$PATH"

echo "Job started at: $(date)"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

LRS=("5e-4" "5e-5")
SCHEDULERS=("onecycle" "step")
MODEL_SIZE=("small" "medium" "large")
DATA_CONFIG=("vulcan_no300_eval_train_plus_valgt300")

CONFIGS=()
for lr in "${LRS[@]}"; do
    for sched in "${SCHEDULERS[@]}"; do
        for model_size in "${MODEL_SIZE[@]}"; do
            for data_path in "${DATA_CONFIG[@]}"; do
                CONFIGS+=("optimizer.optimizer.lr=${lr} model.rfdetr.size=${model_size} scheduler=${sched} data=${data_path}")
            done
        done
    done
done

MY_CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "========================================================"
echo "Starting SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Assigned Configuration: $MY_CONFIG"
echo "========================================================"

srun uv run train_rf_detr_seg.py \
    model=rfdetr_seg \
    trainer.max_epochs=100 \
    ${MY_CONFIG}
