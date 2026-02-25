#!/bin/bash

# run_rfdetr_vulcan.sh
# Specific script for training RF-DETR on Vulcan

ACCOUNT="aip-robsc"

# Define resources
NODES=1
GPUS=4
CPUS_PER_TASK=4
MEM_PER_GPU=32G
TIME="7:59:00"

echo "🚀 Requesting allocation and starting RF-DETR training..."
echo "Account: $ACCOUNT | Nodes: $NODES | GPUs: $GPUS"

# Ensure logs directory exists
mkdir -p logs

# Run srun within salloc or directly (srun will wait for allocation if used with salloc resources)
# For interactive simulation, we use salloc to get the node then srun to launch.
# However, to run in background, we'll use salloc and then run srun inside.

salloc --account=$ACCOUNT \
       --nodes=$NODES \
       --gpus-per-node=$GPUS \
       --ntasks-per-node=$GPUS \
       --cpus-per-task=$CPUS_PER_TASK \
       --mem-per-gpu=$MEM_PER_GPU \
       --time=$TIME \
       bash -c "
         export PATH=\"\$HOME/.cargo/bin:\$PATH\"
         echo \"Allocation granted. Starting training...\"
         srun uv run train_rf_detr.py \
              data=vulcan \
              model=rfdetr \
              trainer.devices=4 \
              trainer.strategy=ddp \
              logging.wandb.project='cell-detection' \
              run_name='rfdetr_vulcan_$(date +%Y%m%d_%H%M%S)' \
              > logs/rfdetr-\$SLURM_JOB_ID.out 2> logs/rfdetr-\$SLURM_JOB_ID.err
       " &

echo "✅ Training job submitted to background. Monitor logs in logs/rfdetr-[JOB_ID].out"
