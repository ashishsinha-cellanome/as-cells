#!/bin/bash
echo "Note: This script should be run in an ask1gpu session on Vulcan."
echo "Testing FPN types: adapter, sfp, fused, tiny"

FPNS=("adapter" "sfp" "fused" "tiny")

for fpn in "${FPNS[@]}"; do
    echo "=========================================================="
    echo "TESTING: DINOv2 with FPN Type -> $fpn"
    echo "=========================================================="
    uv run train_mask2former.py \
        model=mask2former \
        model/backbone=dinov2_mask2former \
        model.backbone.fpn_type=$fpn \
        data=vulcan_no300_eval \
        trainer.max_epochs=10 \
        data.limit_train_batches=100 \
        data.limit_val_batches=100 \
        optimizer.optimizer.lr=1e-4 \
        data.batch_size=8
    echo "Completed test for $fpn."
    echo ""
done
