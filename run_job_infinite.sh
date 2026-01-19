#!/bin/bash
#SBATCH --job-name=large_training_job
#SBATCH --account=aip-robsc
#SBATCH --nodes=8
#SBATCH --mail-user=ashish.sinha@amii.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# --- Account Handling ---
# If not running under SLURM, submit this script to sbatch
if [ -z "$SLURM_JOB_ID" ]; then
    echo "Submitting job with account: $ACCOUNT"
    sbatch --account=$ACCOUNT "$0"
    exit
fi

# --- Training Logic (Only runs inside SLURM) ---
set -e

module load python/3.10 cuda/12.2

# Ensure uv is in PATH (common install location)
export PATH="$HOME/.cargo/bin:$PATH"

echo "Job started at: $(date)"
echo "Account used: $SLURM_JOB_ACCOUNT"

srun uv run train_rt_detr.py --config configs/rt_detr_dinov2_config.yaml
