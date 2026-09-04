#!/bin/bash

ulimit -n 65535

run_training() {
    local datasets=$1
    local r=$2
    local frac=$3
    
    echo "====================================================================="
    echo "🚀 Starting LoRA Fine-Tuning | Rank: $r | Frac: $frac"
    echo "Datasets: $datasets"
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
        data.batch_size=16 \
        +data.eval_batch_size=64 \
        +model.rfdetr.lora.r=$r \
        model.rfdetr.finetune_mode=lora \
        data.target_data_frac=$frac \
        trainer.check_val_every_n_epoch=20 \
        data.target_crops_per_base=32 \
        data.anchor_crops_per_base=32 \
        data.target_datasets="[$datasets]" \
        optimizer.optimizer.lr=$lr \
        model.rfdetr.lr_scheduler=cosine
}

# 3 node combo: u87 + 2025-neuron-adhered
COMBO1="20240905_u87-adhered_10x_caged_4_class,20250108_neuron-adhered_10x_uncaged_4_class,20250305_neuron-adhered_10x_uncaged_4_class"

# 2 node combo: 2025-neuron-adhered
COMBO2="20250108_neuron-adhered_10x_uncaged_4_class,20250305_neuron-adhered_10x_uncaged_4_class"

# ----------------------------------------
# COMBO 1 (3 node combo)
# ----------------------------------------

# Rank 32: 1, 5, 10, 25, 50%
for frac in 0.01 0.05 0.1 0.25 0.5; do
    run_training $COMBO1 32 $frac
done

# Rank 64: 10, 50%
for frac in 0.1 0.5; do
    run_training $COMBO1 64 $frac
done

# ----------------------------------------
# COMBO 2 (2 node combo)
# ----------------------------------------

# Rank 32 & Rank 64: 1, 5, 10, 25, 50%
for r in 32 64; do
    for frac in 0.01 0.05 0.1 0.25 0.5; do
        run_training $COMBO2 $r $frac
    done
done

echo "🎉 All fraction scaling multi-combo experiments completed!"