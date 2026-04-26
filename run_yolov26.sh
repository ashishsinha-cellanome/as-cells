#!/bin/bash
#SBATCH --job-name=yolo26
#SBATCH --account=aip-robsc
#SBATCH --nodes=1
#SBATCH --mail-user=ashish.sinha@amii.ca
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time=1-12:00:00
#SBATCH --array=0
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

ACCOUNT="aip-robsc"

# If not running under SLURM, submit this script to sbatch automatically
if [ -z "$SLURM_JOB_ID" ]; then
    echo "Submitting job array with account: $ACCOUNT"
    mkdir -p logs   
    sbatch --account=$ACCOUNT "$0"
    exit
fi

set -e

# Ensure uv is in PATH
export PATH="$HOME/.cargo/bin:$PATH"

echo "Job started at: $(date)"
echo "Account used: $SLURM_JOB_ACCOUNT"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

CONFIGS=()
DATA_CONFIG=("vulcan" "vulcan_no300_eval_train_plus_valgt300")

for data_path in "${DATA_CONFIG[@]}"; do
    CONFIGS+=("model=yolov26 data=${data_path}")
done

MY_CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "======================================================================"
echo "🚀 Array Task ID: $SLURM_ARRAY_TASK_ID / ${#CONFIGS[@]}"
echo "⚙️  Config: $MY_CONFIG"
echo "======================================================================"

srun uv run train_yolov26.py $MY_CONFIG

echo "✅ Finished Array Task $SLURM_ARRAY_TASK_ID"