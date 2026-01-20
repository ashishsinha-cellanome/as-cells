#!/bin/bash
#SBATCH --job-name=large_training_job
#SBATCH --account=aip-robsc
#SBATCH --nodes=8
#SBATCH --mail-user=ashish.sinha@amii.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH --mem-per-gpu=32G
#SBATCH --time=7-00:00:00
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

# srun -o logs/%x-%j-%t.out -e logs/%x-%j-%t.err uv run train_rt_detr.py -m model=rtdetr_v1 model.ema.enabled=True,False model.backbone.train_backbone=True 
srun  uv run train_rt_detr_v2.py -m data=vulcan   trainer.max_epochs=2 debug=True  model=rtdetr_v1,rtdetr_v2 model/backbone=resnet50  model.ema.enabled=True,False model.backbone.train_backbone=True,False model.backbone.freeze_at_stage=0,1 optimizer.optimizer.use_param_groups=true,false
