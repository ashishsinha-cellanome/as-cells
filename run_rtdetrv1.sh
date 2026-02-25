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

srun uv run train_rt_detr_v2.py data=vulcan model=rtdetr_v1 model/backbone=resnet50  model.backbone.train_backbone=True model.backbone.freeze_at_stage=2 optimizer.optimizer.use_param_groups=True scheduler=onecycle optimizer.optimizer.lr=3e-4 \
initialization.load_from_checkpoint="/project/aip-robsc/asinha/cellanome/DATA/checkpoints/rtdetr_v1_resnet50_2026-02-24_4071863_08-02/ckpts/rtdetr-ema-06-val_map0.5387.ckpt" trainer.max_epochs=100

srun uv run train_rt_detr_v2.py data=vulcan model=rtdetr_v1 model/backbone=resnet50  model.backbone.train_backbone=True model.backbone.freeze_at_stage=2 optimizer.optimizer.use_param_groups=False scheduler=onecycle optimizer.optimizer.lr=3e-4 \
initialization.load_from_checkpoint="/project/aip-robsc/asinha/cellanome/DATA/checkpoints/rtdetr_v1_resnet50_2026-02-24_4071863_08-02/ckpts/rtdetr-ema-06-val_map0.5387.ckpt" trainer.max_epochs=100
