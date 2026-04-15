#!/bin/bash
#SBATCH --job-name=m2f_swinL
#SBATCH --account=aip-robsc
#SBATCH --nodes=4
#SBATCH --mail-user=ashish.sinha@amii.ca
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-gpu=32G
#SBATCH --array=0-3
#SBATCH --time=2-10:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -e

export PATH="$HOME/.cargo/bin:$PATH"

echo "Job started at: $(date)"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

LRS=("3e-4" "5e-5")
SCHEDULERS=("onecycle" "multistep")
DATA_CONFIG=("vulcan_no300_eval_train_plus_valgt300")

CONFIGS=()
for lr in "${LRS[@]}"; do
  for sched in "${SCHEDULERS[@]}"; do
    for data_path in "${DATA_CONFIG[@]}"; do
      CONFIGS+=("model.input_size=640 optimizer.optimizer.lr=${lr} scheduler=${sched} data=${data_path}")
    done
  done
done

MY_CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "========================================================"
echo "Starting SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Assigned Configuration: $MY_CONFIG"
echo "========================================================"

srun uv run train_mask2former.py \
  model=mask2former \
  model/backbone=swin_large_mask2former \
  trainer.max_epochs=50 \
  data.batch_size=16 \
  model.mask2former.num_queries=300 \
  ${MY_CONFIG}
