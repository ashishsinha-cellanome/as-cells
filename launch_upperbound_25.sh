uv run train_rfdetr_phase2.py \
    data=coverage_splits/lora_finetune_mix \
    model=rfdetr_seg \
    data.batch_size=16 \
    +data.eval_batch_size=64 \
    model.rfdetr.finetune_mode=full \
    data.anchor_datasets='[]' \
    "data.target_datasets='\${data.test_datasets}'" \
    data.target_data_frac=0.25 \
    data.target_crops_per_base=32 \
    model.rfdetr.lr_scheduler=cosine \
    initialization.base_checkpoint=null
