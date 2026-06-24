#!/bin/bash

echo "Submitting PEFT Generalization Experiments to SLURM on Vulcan..."

CONFIGS=("exp_susp_to_sus" "exp_susp_to_adh" "exp_global_to_rest")
MODELS=("yolov26" "rfdetr_seg")

for config in "${CONFIGS[@]}"; do
    for model in "${MODELS[@]}"; do
        if [ "$model" = "rfdetr_seg" ]; then
            script_name="train_rf_detr_seg.py"
        elif [ "$model" = "yolov26" ]; then
            script_name="train_yolov26.py"
        else
            script_name="train_${model}.py"
        fi
        echo "Submitting $model on $config..."
        # Replace with your actual SLURM submission command. Example using sbatch:
        # sbatch <<EOT
        # #!/bin/bash
        # #SBATCH --account=aip-robsc
        # #SBATCH --nodes=1
        # #SBATCH --gpus-per-node=1
        # uv run $script_name data=$config model=$model model.peft=True
        # EOT
        
        # For dry-run printing:
        echo "Command: uv run $script_name data=$config model=$model model.peft=True"
    done
done

echo "Done."
