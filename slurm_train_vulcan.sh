#!/bin/bash
#SBATCH --job-name=rfdetr_phase2
#SBATCH --account=aip-robsc
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Create log directory to capture outputs
mkdir -p slurm_logs

# Execute using srun for multi-node PyTorch Lightning training
srun uv run train_rfdetr_phase2.py \
    model=rfdetr_seg \
    model.rfdetr.size=large \
    data=coverage_splits/upperbound \
    data.path=/project/aip-robsc/asinha/cellanome/DATA/PHASE2 \
    data.batch_size=16 \
    data.num_workers=4 \
    model.rfdetr.lr_scheduler=cosine
