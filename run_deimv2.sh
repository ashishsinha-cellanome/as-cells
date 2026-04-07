#!/bin/bash
#SBATCH --job-name=deimv2
#SBATCH --account=aip-robsc
#SBATCH --nodes=1
#SBATCH --mail-user=ashish.sinha@amii.ca
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=32G
#SBATCH --array=0-3

set -e

# Ensure uv is in PATH
export PATH="$HOME/.cargo/bin:$HOME/.local/share/pipx/venvs/uv/bin:$PATH"

echo "Job started at: $(date)"
echo "Account used: $SLURM_JOB_ACCOUNT"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

LRS=("1e-4" "5e-5")
MODEL_SIZE=("m" "x")
SCHEDULERS=("onecycle")
DATA_CONFIG=("vulcan")
CONFIGS=()
for lr in "${LRS[@]}"; do
    for sched in "${SCHEDULERS[@]}"; do
        for model_size in "${MODEL_SIZE[@]}"; do
            for data_path in "${DATA_CONFIG[@]}"; do
                CONFIGS+=("optimizer.optimizer.lr=${lr} model.deimv2.size=${model_size} scheduler=${sched} data=${data_path}")
            done
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
srun uv run train_deim_v2.py \
    model=deimv2 \
    trainer.max_epochs=100 \
    ${MY_CONFIG}