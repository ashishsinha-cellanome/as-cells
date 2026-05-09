#!/bin/bash
#SBATCH --job-name=rfdetr_orig
#SBATCH --account=aip-robsc
#SBATCH --nodes=1
#SBATCH --mail-user=ashish.sinha@amii.ca
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
##SBATCH --exclusive
#SBATCH --mem-per-gpu=32G
#SBATCH --time=3-10:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# --- Training Logic (Only runs inside SLURM) ---
set -e

# Ensure uv is in PATH (common install location)
export PATH="$HOME/.cargo/bin:$PATH"

echo "Job started at: $(date)"
echo "Account used: $SLURM_JOB_ACCOUNT"

# Ensure logs directory exists
mkdir -p logs

echo "========================================================"
echo "Starting SLURM Job: $SLURM_JOB_ID"
echo "========================================================"

# Run the training command using 4 GPUs
# Since this uses PyTorch Lightning's DDP internally via strategy="ddp_find_unused_parameters_true",
# we let Lightning handle the distributed launching rather than srun wrapper.
uv run train_original_rfdetr_seg.py \
    --dataset_dir="/project/aip-robsc/asinha/cellanome/DATA/TRAINING_DATA" \
    --epochs=50 \
    --batch_size=2 \
    --grad_accum_steps=8 \
    --devices=4
