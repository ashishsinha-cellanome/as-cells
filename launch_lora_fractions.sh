#!/bin/bash

# Target data fractions to sweep
fractions=(0.01 0.1)

for frac in "${fractions[@]}"; do
    echo "====================================================================="
    echo "🚀 Starting LoRA Fine-Tuning for target_data_frac = $frac"
    echo "====================================================================="
    
    # Adjust learning rate based on data fraction
    if (( $(echo "$frac <= 0.1" | bc -l) )); then
        lr="5e-5"
    else
        lr="5e-4"
    fi

    echo "Using LR = $lr for fraction = $frac"
    
    uv run train_rfdetr_phase2.py \
        data=coverage_splits/lora_finetune_mix \
        model=rfdetr_seg \
        +model.rfdetr.lora.r=64 \
        +model.rfdetr.lora.alpha=128 \
        model.rfdetr.finetune_mode=lora \
        data.target_data_frac=$frac \
        trainer.check_val_every_n_epoch=10 \
        data.target_crops_per_base=32 \
        data.anchor_crops_per_base=32 \
        data.target_datasets='[20250108_neuron-adhered_10x_uncaged_4_class]' \
        optimizer.optimizer.lr=1e-3 \
        model.rfdetr.lr_scheduler=cosine
        
    echo "✅ Finished run for target_data_frac = $frac"
    echo ""
done

echo "🎉 All fraction scaling experiments completed!"
