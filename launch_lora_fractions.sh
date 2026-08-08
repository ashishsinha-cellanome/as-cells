#!/bin/bash

# Target data fractions to sweep
fractions=(0.05 0.10 0.25 0.50)

for frac in "${fractions[@]}"; do
    echo "====================================================================="
    echo "🚀 Starting LoRA Fine-Tuning for target_data_frac = $frac"
    echo "====================================================================="
    
    uv run train_rfdetr_phase2.py \
        data=coverage_splits/lora_finetune_mix \
        model=rfdetr_seg \
        model.rfdetr.finetune_mode=lora \
        data.target_data_frac=$frac \
        data.target_crops_per_base=32 \
        data.anchor_crops_per_base=16 \
        data.target_datasets='[20250108_neuron-adhered_10x_uncaged_4_class]' \
        optimizer.optimizer.lr=1e-3 \
        model.rfdetr.lr_scheduler=cosine
        
    echo "✅ Finished run for target_data_frac = $frac"
    echo ""
done

echo "🎉 All fraction scaling experiments completed!"
