#!/bin/bash
#SBATCH --job-name=m2f-swin-test
#SBATCH --account=aip-robsc
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e

export PATH="$HOME/.local/share/pipx/venvs/uv/bin:$HOME/.cargo/bin:$PATH"

echo "Starting Swin-Large Mask2Former Training on 4 GPUs"
echo "Job started at: $(date)"

srun uv run train_mask2former.py   model=mask2former   model/backbone=swin_large_mask2former   data=vulcan   scheduler=onecycle   data.batch_size=8   trainer.max_epochs=10   model.mask2former.num_queries=100
