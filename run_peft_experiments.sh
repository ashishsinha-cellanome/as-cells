#!/bin/bash

echo "Submitting PEFT Generalization Experiments to SLURM on Vulcan..."

CONFIGS=("exp_susp_to_sus" "exp_susp_to_adh" "exp_global_to_rest")
MODELS=("yolov26" "rfdetr_seg")

for config in "${CONFIGS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "Submitting $model on $config..."
        # Replace with your actual SLURM submission command. Example using sbatch:
        # sbatch <<EOT
        # #!/bin/bash
        # #SBATCH --account=aip-robsc
        # #SBATCH --nodes=1
        # #SBATCH --gpus-per-node=1
        # uv run train_${model%_*}.py data=$config model=$model model.peft=True
        # EOT
        
        # For dry-run printing:
        echo "Command: uv run train_${model%_*}.py data=$config model=$model model.peft=True"
    done
done

echo "Done."
