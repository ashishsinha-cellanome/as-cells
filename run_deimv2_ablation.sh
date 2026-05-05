#!/bin/bash
#SBATCH --job-name=deimv2_ablate
#SBATCH --account=aip-robsc
#SBATCH --nodes=4
#SBATCH --mail-user=ashish.sinha@amii.ca
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-gpu=32G
#SBATCH --array=0-71
#SBATCH --time=1-23:59:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -e

export PATH="$HOME/.cargo/bin:$HOME/.local/share/pipx/venvs/uv/bin:$PATH"

echo "Job started at: $(date)"
echo "Account used: $SLURM_JOB_ACCOUNT"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# 1. We will fix the model size to 'm' for ablation to save compute
MODEL_SIZE="m"

# 2. Define the ablation spaces
NUM_DENOISING_OPTIONS=(0 100 300)
REG_SCALE_OPTIONS=(4.0 8.0 12.0)
GRID_SIZE_OPTIONS=(0.05 0.15) # Default vs Large Anchors

# Generate all 18 combinations (3 x 3 x 2)
CONFIGS=()
for dn in "${NUM_DENOISING_OPTIONS[@]}"; do
  for rs in "${REG_SCALE_OPTIONS[@]}"; do
    for gs in "${GRID_SIZE_OPTIONS[@]}"; do
      # Note: We pass grid_size as a custom arg that we can intercept in the hydra config or python script later,
      # but for now, we rely on the standard Hydra overrides for dn and rs.
      # To properly ablate grid_size without code changes, you'd need to expose it in deimv2.yaml.
      # For now, we will ablate the matcher giou cost instead of grid_size to keep it purely config-driven.
      # Let's adjust the loops below to be purely config-driven:
      :
    done
  done
done

# Let's rebuild a clean config-driven ablation matrix
NUM_DENOISING_OPTS=(0 100 300)
REG_SCALE_OPTS=(4.0 8.0 12.0)
MATCHER_GIOU_OPTS=(2 5) # Default vs High GIoU focus
GRID_SIZE_OPTS=(0.05 0.15) # Default vs Large Anchors
NUM_POINTS_OPTS=("[3,6,3]" "[6,12,12]") # Default vs Wide Receptive Field

CONFIGS=()
for dn in "${NUM_DENOISING_OPTS[@]}"; do
  for rs in "${REG_SCALE_OPTS[@]}"; do
    for giou in "${MATCHER_GIOU_OPTS[@]}"; do
      for gs in "${GRID_SIZE_OPTS[@]}"; do
        for np in "${NUM_POINTS_OPTS[@]}"; do
          # Note: when changing reg_scale, we usually scale reg_max accordingly, but we'll leave reg_max fixed at 48 for safety to allow large bounds
          CONFIGS+=("model.deimv2.decoder.num_denoising=${dn} model.deimv2.decoder.reg_scale=${rs} model.deimv2.criterion.matcher.weight_dict.cost_giou=${giou} model.deimv2.decoder.grid_size=${gs} model.deimv2.decoder.num_points=${np} model.deimv2.decoder.reg_max=48 model.deimv2.criterion.reg_max=48 model.deimv2.size=${MODEL_SIZE} data=vulcan_no300_eval")
        done
      done
    done
  done
done

# We have 3 * 3 * 2 * 2 * 2 = 72 combinations. Our SLURM array is 0-71.
MY_CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "========================================================"
echo "Starting SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Assigned Configuration: $MY_CONFIG"
echo "========================================================"

srun uv run train_deim_v2.py \
  model=deimv2 \
  trainer.max_epochs=50 \
  ${MY_CONFIG}
