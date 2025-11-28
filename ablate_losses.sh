uv run train_rt_detr_v2.py --multirun \
    model.dinov2.output_indices_for_fpn=[1,3,5],[4,8,12],[3,7,11],[8,10,12]\
    model.dinov2.fpn_type='tiny','fused'\
    data.batch_size=32\
    

    # model.rtdetr.decoder_n_points=[1,2,4]\

    # model.rtdetr.label_noise_ratio=0.5,0.25\
    # model.rtdetr.box_noise_scale=0.5,1\
    # model.rtdetr.learn_initial_query=False,True
    
    

    # model.rtdetr.auxiliary_loss=true,false\
    # model.rtdetr.use_focal_loss=true,false\
    # model.rtdetr.decoder_n_levels=4\
    # model.rtdetr.focal_loss_alpha=0.25,0.5,0.75,1\