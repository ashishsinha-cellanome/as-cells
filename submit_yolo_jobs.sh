#!/bin/bash
#SBATCH --job-name=yolo
#SBATCH --account=aip-robsc
#SBATCH --nodes=1
#SBATCH --mail-user=ashish.sinha@amii.ca
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-59  # 🟢 Launches 60 jobs (0 through 59)
#SBATCH --output=logs/%x-%A_%a.out  # %A is the array job ID, %a is the specific task ID
#SBATCH --error=logs/%x-%A_%a.err

ACCOUNT="aip-robsc"

# --- Self-Submission Handling ---
# If not running under SLURM, submit this script to sbatch automatically
if [ -z "$SLURM_JOB_ID" ]; then
    echo "Submitting job array with account: $ACCOUNT"
    # Ensure logs directory exists before submitting, or SLURM will fail silently
    mkdir -p logs
    sbatch --account=$ACCOUNT "$0"
    exit
fi

# --- Training Logic (Only runs inside SLURM) ---
set -e

# Ensure uv is in PATH (common install location)
export PATH="$HOME/.cargo/bin:$PATH"

echo "Job started at: $(date)"
echo "Account used: $SLURM_JOB_ACCOUNT"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# 1. Define the parameter arrays
YOLO_SIZES=("s" "m" "n" "l" "x")
LRS=("0.01" "0.001" "0.0001")
SCHEDULERS=("cosine_warmup" "step" "reduce_lr_on_plateau" "onecycle")

# 2. Build a flat list of all combinations in memory
CONFIGS=()
for size in "${YOLO_SIZES[@]}"; do
    for lr in "${LRS[@]}"; do
        for sched in "${SCHEDULERS[@]}"; do
            # Combine the dynamic arguments into a single string
            CONFIGS+=("model.yolov5.yolo_size=${size} model.yolov5.optimizer.lr=${lr} scheduler=${sched}")
        done
    done
done

# 3. Use SLURM_ARRAY_TASK_ID to pick this specific job's configuration
MY_CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "======================================================================"
echo "🚀 Array Task ID: $SLURM_ARRAY_TASK_ID / 59"
echo "⚙️  Config: $MY_CONFIG"
echo "======================================================================"

# 4. Run the training script using the selected config
# Notice we pass $MY_CONFIG unquoted so Bash expands it into separate arguments
srun uv run train_yolov5.py \
    data=vulcan \
    model=yolov5 \
    $MY_CONFIG

echo "✅ Finished Array Task $SLURM_ARRAY_TASK_ID"