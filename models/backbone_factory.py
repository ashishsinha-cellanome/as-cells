import hydra
from omegaconf import OmegaConf
from models.dinov2_backbone_with_fpn import Dinov2BackBoneWithFPN

def get_backbone_unique_id(backbone_cfg, rtdetr_model_name):
    """
    Pure function: Generates the unique hash string from config.
    Does NOT load model or weights.
    """
    model_type = backbone_cfg.type
    unique_str = f"_{model_type}"
    
    # RT-DETR model name
    unique_str += f"_{rtdetr_model_name}"

    # Hash key parameters
    cfg_dict = OmegaConf.to_container(backbone_cfg, resolve=True)
    keys_to_hash = ['fpn_type', 'scale_factor', 'output_indices_for_fpn', 'upscale_method']
    
    for k in keys_to_hash:
        if k in cfg_dict:
            val = str(cfg_dict[k]).replace('[','').replace(']','').replace(', ','_').replace("'", "")
            unique_str += f"_{k}_{val}"
            
    if model_type == "resnet":
        unique_str = f"_{model_type}_{backbone_cfg.name}_freeze_stage_{backbone_cfg.freeze_at_stage}"
        
    return unique_str

def build_backbone(backbone_cfg, rtdetr_model_name):
    """
    Factory function to initialize different backbones based on config.
    Returns: (model, config_object, unique_cache_string)
    """
    unique_str = get_backbone_unique_id(backbone_cfg, rtdetr_model_name)
    model_type = backbone_cfg.type
    
    if model_type == "dinov2":
        # Resolve interpolation before passing to class
        resolved_channels = OmegaConf.to_container(backbone_cfg.intermediate_channel_sizes, resolve=True)
        
        model = Dinov2BackBoneWithFPN.from_pretrained(
            backbone_cfg.pretrained_name_or_path,
            output_indices_for_fpn=OmegaConf.to_container(backbone_cfg.output_indices_for_fpn, resolve=True),
            first_layer_dims=OmegaConf.to_container(backbone_cfg.first_layer_dims, resolve=True),
            fpn_type=backbone_cfg.fpn_type,
            scale_factor=backbone_cfg.scale_factor,
            upscale_method=backbone_cfg.upscale_method,
            intermediate_channel_sizes=resolved_channels, 
        )
        return model, model.config, unique_str

    elif model_type == "resnet":
        unique_str = f"_{backbone_cfg.model_name}_{backbone_cfg.name}_freeze_stage_{backbone_cfg.freeze_at_stage}"
        return None, None, unique_str
        # raise NotImplementedError("ResNet backbone not yet implemented")
         
    else:
        raise ValueError(f"Unknown backbone type: {model_type}")
    

def freeze_backbone_layers(model, freeze_at_stage):
    """
    Freezes ResNet backbone layers in an RT-DETR model.
    freeze_at_stage:
      0: Unfrozen
      1: Freeze Stem (Conv1 + BN)
      2: Freeze Stem + Layer 1
      3: Freeze Stem + Layer 1 + Layer 2
      4: Freeze Stem + Layer 1 + Layer 2 + Layer 3
      5: Freeze Entire Backbone
    """
    # RT-DETRv2 standard backbone structure in Transformers:
    # model.model.backbone.model -> (conv1, bn1, layer1, layer2, layer3, layer4)
    
    # Locate the actual backbone module
    if hasattr(model.model.backbone, 'model'):
        # If it's wrapped (e.g. TimmBackbone)
        backbone = model.model.backbone.model
    else:
        # If it's native Transformers ResNet
        backbone = model.model.backbone

    print(f"[INFO] Freezing backbone layers up to stage {freeze_at_stage}...")

    # Helper to freeze a module
    def freeze_module(module):
        # TODO: try training where the batchnorm stats are being updated
        # TODO: comment later
        
        module.eval()  # Set to eval mode
        for param in module.parameters():
            param.requires_grad = False

    # 1. Stem (Conv1, BN, ReLU, MaxPool) - Always freeze if stage >= 1
    if freeze_at_stage >= 1:
        if hasattr(backbone, 'conv1'): freeze_module(backbone.conv1)
        if hasattr(backbone, 'bn1'):   freeze_module(backbone.bn1)
        if hasattr(backbone, 'embedder'): freeze_module(backbone.embedder) # Alternate name

    # 2. ResNet Stages
    # Note: layers are usually named 'layer1', 'layer2', etc. or 'encoder.stages.0'
    stages_to_freeze = []
    
    # Identify the stages container
    if hasattr(backbone, 'layer1'):
        # Standard TorchVision/Timm naming
        if freeze_at_stage >= 2: stages_to_freeze.append(backbone.layer1)
        if freeze_at_stage >= 3: stages_to_freeze.append(backbone.layer2)
        if freeze_at_stage >= 4: stages_to_freeze.append(backbone.layer3)
        if freeze_at_stage >= 5: stages_to_freeze.append(backbone.layer4)
    elif hasattr(backbone, 'encoder') and hasattr(backbone.encoder, 'stages'):
        # Transformers ResNetBackbone naming
        stages = backbone.encoder.stages
        if freeze_at_stage >= 2: stages_to_freeze.append(stages[0])
        if freeze_at_stage >= 3: stages_to_freeze.append(stages[1])
        if freeze_at_stage >= 4: stages_to_freeze.append(stages[2])
        if freeze_at_stage >= 5: stages_to_freeze.append(stages[3])

    for stage in stages_to_freeze:
        freeze_module(stage)
        
    print(f"✓ Frozen {len(stages_to_freeze)} stages + stem.")