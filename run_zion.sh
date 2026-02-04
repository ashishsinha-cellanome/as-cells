#!/bin/bash
#SBATCH --job-name=rtdetr
#SBATCH --nodes=1
#SBATCH --mail-user=ashish.sinha@amii.ca
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --gpus=3
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=2
##SBATCH --exclusive
#SBATCH --mem-per-gpu=24G
#SBATCH --time=17-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# --- Training Logic (Only runs inside SLURM) ---
set -e

# module load python/3.10 cuda/12.2

# Ensure uv is in PATH (common install location)
export PATH="$HOME/.cargo/bin:$PATH"

echo "Job started at: $(date)"ß

# srun -o logs/%x-%j-%t.out -e logs/%x-%j-%t.err uv run train_rt_detr.py -m model=rtdetr_v1 model.ema.enabled=True,False model.backbone.train_backbone=True 
srun  uv run train_rt_detr_v2.py -m data=full   trainer.max_epochs=50  model=rtdetr_v1,rtdetr_v2 model/backbone=resnet50  model.ema.enabled=True,False model.backbone.train_backbone=True,False model.backbone.freeze_at_stage=0,1 optimizer.optimizer.use_param_groups=true,false
