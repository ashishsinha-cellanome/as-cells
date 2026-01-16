import os
from typing import Tuple, Union, List, Dict, Final, Optional

from transformers import RTDetrForObjectDetection, RTDetrConfig
from safetensors.torch import load_file as safe_load

from models.dinov2_backbone_with_fpn import Dinov2BackBoneWithFPNConfig, Dinov2BackBoneWithFPN

class RTDetrConfigWithCustomBackBone(RTDetrConfig):
   
    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        if isinstance(self.backbone_config, Dinov2BackBoneWithFPNConfig):
            # save the backbone config before replacing the backbone config with a default one
            self.backbone_config.save_pretrained(os.path.join(save_directory, "backbone_config"))
            
            # set it the default one used by the pre-configured model, first back it up to restore after saving
            backbone_config = self.backbone_config
            if self._name_or_path:
                # set it the default one used by the pre-configured model
                try:
                    self.backbone_config = RTDetrConfig.from_pretrained(self._name_or_path).backbone_config
                except Exception:
                    self.backbone_config = RTDetrConfig().backbone_config
            else:
                self.backbone_config = RTDetrConfig().backbone_config    
        
            # save main RT-DETR config
            super().save_pretrained(save_directory, **kwargs)
            # restore the backbone config
            self.backbone_config = backbone_config
        else:
            super().save_pretrained(save_directory, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # try loading the backbone config if present
        backbone_config_path = os.path.join(pretrained_model_name_or_path, "backbone_config")
        if os.path.exists(backbone_config_path):
            backbone_config = Dinov2BackBoneWithFPNConfig.from_pretrained(backbone_config_path)
        else:
            backbone_config = config.backbone_config

        config.backbone_config = backbone_config
        config._name_or_path = pretrained_model_name_or_path
        return config

class RTDetrV1Model(RTDetrForObjectDetection):
    """
    Wrapper for RT-DETRv1 model to ensure compatibility with the training pipeline.
    Supports swapping the backbone (e.g. for DINOv2).
    """
    config_class = RTDetrConfigWithCustomBackBone
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, config=None, **kwargs):
        if config is None:
            config = RTDetrConfigWithCustomBackBone.from_pretrained(pretrained_model_name_or_path, **kwargs)
            
        model = cls(config, *model_args)

        # load FPN weights from safetensors file if they exist and we are using custom backbone
        safe_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        if os.path.isfile(safe_path) and isinstance(config.backbone_config, Dinov2BackBoneWithFPNConfig):
            state_dict = safe_load(safe_path)
            model.load_state_dict(state_dict, strict=False)

        return model

    def __init__(self, config):
        if isinstance(config.backbone_config, Dinov2BackBoneWithFPNConfig):
            print(f"[INFO]: The passed config.backbone_config is a custom one of type Dinov2BackBoneWithFPNConfig.")
            
            backbone_config = config.backbone_config
            num_intermediate_channels = backbone_config.intermediate_channel_sizes
            
            # Use default backbone config for super().__init__ to satisfy shape checks/encoder building
            if config._name_or_path:
                try:
                    config.backbone_config = RTDetrConfig.from_pretrained(config._name_or_path).backbone_config
                except Exception:
                    config.backbone_config = RTDetrConfig().backbone_config
            else:
                config.backbone_config = RTDetrConfig().backbone_config
                
            # initialize the model
            super().__init__(config)

            # Restore custom config and inject backbone
            config.backbone_config = backbone_config
            self.model.backbone = Dinov2BackBoneWithFPN(backbone_config)
        else:
            super().__init__(config)
            
        self.post_init()
