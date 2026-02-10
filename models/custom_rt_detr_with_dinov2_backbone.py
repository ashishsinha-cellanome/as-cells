import os

from transformers import RTDetrV2ForObjectDetection, RTDetrV2Config
from safetensors.torch import load_file as safe_load

from models.dinov2_backbone_with_fpn import Dinov2BackBoneWithFPNConfig, Dinov2BackBoneWithFPN

class RTDetrV2ConfigWithCustomBackBone(RTDetrV2Config):
   
    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        if isinstance(self.backbone_config, Dinov2BackBoneWithFPNConfig):
            # save the backbone config before replacing the backbone config with a default one
            # and saving the rest of the model
            self.backbone_config.save_pretrained(os.path.join(save_directory, "backbone_config"))
            # set it the default one used by the pre-configured model, first back it up to restore after saving
            backbone_config = self.backbone_config
            if self._name_or_path:
                # set it the default one used by the pre-configured model
                self.backbone_config = RTDetrV2Config.from_pretrained(self._name_or_path).backbone_config
            else:
                # the default backbone of RTDetrV2Config() may not be compatible with the rest
                # of the model, but this not important as we will replace it when reading the config (from_pretrained below)
                self.backbone_config = RTDetrV2Config().backbone_config    
        
            # save main RT-DETR config
            super().save_pretrained(save_directory, **kwargs)
            # restore the backbone config
            self.backbone_config = backbone_config
        else:
            super().save_pretrained(save_directory, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # load RT-DETR config, the config here has the default backbone_config 
        # (because we have saved it like that here in the checkpoint)
        # so there is no issue here
        config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # try loading the backbone config if present
        backbone_config_path = os.path.join(pretrained_model_name_or_path, "backbone_config")
        if os.path.exists(backbone_config_path):
            backbone_config = Dinov2BackBoneWithFPNConfig.from_pretrained(backbone_config_path)
        else:
            # use the one read
            backbone_config = config.backbone_config

        config.backbone_config = backbone_config
        # save the pretrained_model_name_or_path in the config as it will be used to built the model 
        config._name_or_path = pretrained_model_name_or_path
        # return combined config
        return config

class RTDetrV2ForObjectDetectionWithCustomBackbone(RTDetrV2ForObjectDetection):
    
    config_class = RTDetrV2ConfigWithCustomBackBone
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, config=None, **kwargs):
        # if config isn't passed, load it manually
        if config is None:
            config = RTDetrV2ConfigWithCustomBackBone.from_pretrained(pretrained_model_name_or_path, **kwargs)
            
        # initialize the model
        model = cls(config, *model_args)

        # load FPN weights from safetensors file if they exist
        safe_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        if os.path.isfile(safe_path):
            state_dict = safe_load(safe_path)
            model.load_state_dict(state_dict, strict=False)

        return model

    def __init__(self, config):
        # breakpoint()
        if isinstance(config.backbone_config, Dinov2BackBoneWithFPNConfig):
            print("[INFO]: The passed config.backbone_config is a custom one of type Dinov2BackBoneWithFPNConfig. "
                  "The model and the backbone are instantiated separately as the backbone is not supported by the original model.")
            # save the backbone config before initializing the model
            backbone_config = config.backbone_config
            num_intermediate_channels = backbone_config.intermediate_channel_sizes
            # set it the default one used by the pre-configured model
            if config._name_or_path:
                # set it the default one used by the pre-configured model
                print(f"[INFO]: The RT-DETRv2 model is instantiated with the backbone_config taken from {config._name_or_path}. "
                      f"This backbone will be replaced with the custom one once the RT-DETR model is instantiated.")
                config.backbone_config = RTDetrV2Config.from_pretrained(config._name_or_path).backbone_config
            else:
                # this part may lead to incorrectly built RT-DETRv2 model; this is because the encoder projection convolution layers are
                # automatically built for the output channel dimensions of the backbone, and the default backbone of RTDetrV2Config() 
                # may have different dimensions than the intended model
                print("[INFO]: The RT-DETRv2 model is instantiated with the default backbone config! "
                      "NOTE: This backbone affects how the rest of the model is built and may lead to incorrect model architecture.")
                config.backbone_config = RTDetrV2Config().backbone_config
                if (len(config.backbone_config.hidden_sizes[1:]) != len(num_intermediate_channels) or 
                   len([s for s in config.backbone_config.hidden_sizes[1:] if s not in num_intermediate_channels]) > 0):
                   print(f"[ERROR]: Incorrect model architecture! The number of intermediate channels {config.backbone_config.hidden_sizes[1:]} "
                         f" from the used default backbone_config is not consistent with the custom backbone_config {num_intermediate_channels}!") 
                
            # initialize the model
            super().__init__(config)

            config.backbone_config = backbone_config
            # now configure the backbone, the backbone here may not be consistent with the rest of the model if the ERROR above is shown
            # note that Dinov2BackBoneWithFPN(backbone_config) automatically load the DINOv2 pre-trained weights from the checkpoint
            # included in backbone_config and freeze them
            self.model.backbone = Dinov2BackBoneWithFPN(backbone_config)
        else:
            # initialize the model
            super().__init__(config)
            
        
        self.post_init()