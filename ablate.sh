uv run train_rt_detr_v2.py --multirun \
    model.rtdetr.model_name="rtdetr_v2_r18d","rtdetr_v2_r34vd","rtdetr_v2_r50vd","rtdetr_v2_101vd"\
    model.dinov2.output_indices_for_fpn=[1,3,5],[4,8,12],[8,10,12]\
    trainer.max_epochs=40\
    model.dinov2.fpn_type="simple" #,'tiny','fused'\
    # model.dinov2.scale_factor=1,2\
    # data.batch_size=32\
    # optimizer.optimizer.lr=1e-4,3e-4,5e-4\
    # optimizer.scheduler.warmup_steps=500,1000,3000\
    

    # model.rtdetr.decoder_n_points=[1,2,4]\

    # model.rtdetr.label_noise_ratio=0.5,0.25\
    # model.rtdetr.box_noise_scale=0.5,1\
    # model.rtdetr.learn_initial_query=False,True

    # model.rtdetr.auxiliary_loss=true,false\
    # model.rtdetr.use_focal_loss=true,false\
    # model.rtdetr.decoder_n_levels=4\
    # model.rtdetr.focal_loss_alpha=0.25,0.5,0.75,1\