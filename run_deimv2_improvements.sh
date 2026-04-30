#!/bin/bash
#SBATCH --job-name=deimv2_improved
#SBATCH --account=aip-robsc
#SBATCH --nodes=1
#SBATCH --mail-user=ashish.sinha@amii.ca
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=32G
#SBATCH --array=0-7
#SBATCH --time=1-23:59:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -e

# Ensure uv is in PATH
export PATH="$HOME/.cargo/bin:$HOME/.local/share/pipx/venvs/uv/bin:$PATH"

echo "Job started at: $(date)"
echo "Account used: $SLURM_JOB_ACCOUNT"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# We will test two model sizes
MODEL_SIZES=("m" "x")

# Define the improvement configs via Hydra overrides
# 1. Baseline
CONF_BASELINE=""

# 2. Imbalance-focused: CDN tweaks and high class matching cost
CONF_IMBALANCE="model.deimv2.decoder.num_denoising=300 model.deimv2.decoder.label_noise_ratio=0.8 model.deimv2.criterion.matcher.weight_dict.cost_class=4"

# 3. Large-object focused: Expanded regression bounds, increased attention points, and GIoU focused matcher
CONF_LARGE_OBJ="model.deimv2.decoder.reg_max=48 model.deimv2.decoder.reg_scale=6.0 model.deimv2.criterion.reg_max=48 model.deimv2.decoder.num_points=[6,12,12] model.deimv2.criterion.matcher.weight_dict.cost_bbox=2 model.deimv2.criterion.matcher.weight_dict.cost_giou=5"

# 4. Combined: Both Imbalance and Large-object improvements
CONF_COMBINED="$CONF_IMBALANCE $CONF_LARGE_OBJ"

EXPERIMENTS=("$CONF_BASELINE" "$CONF_IMBALANCE" "$CONF_LARGE_OBJ" "$CONF_COMBINED")

# Generate the combinations
CONFIGS=()
for size in "${MODEL_SIZES[@]}"; do
  for exp in "${EXPERIMENTS[@]}"; do
    CONFIGS+=("model.deimv2.size=${size} data=vulcan_no300_eval $exp")
  done
done

# Extract the specific config string for this array task ID
MY_CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "========================================================"
echo "Starting SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Assigned Configuration: $MY_CONFIG"
echo "========================================================"

# Run the training command with Hydra overrides
# (You might need to adjust trainer.max_epochs based on your usual run times)
srun uv run train_deim_v2.py \
  model=deimv2 \
  trainer.max_epochs=50 \
  ${MY_CONFIG}
