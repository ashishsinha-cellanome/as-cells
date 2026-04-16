import os
from typing import Tuple, Union, List, Dict, Final, Optional

from transformers import PreTrainedModel, PretrainedConfig
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation
from safetensors.torch import save_file as safe_save
from safetensors.torch import load_file as safe_load

from models.dinov2_backbone_with_fpn import (
    Dinov2BackBoneWithFPNConfig,
    Dinov2BackBoneWithFPN,
)
from models.dinov2_adapter import Dinov2AdapterConfig, Dinov2Adapter


class Mask2FormerConfigWithCustomBackBone(Mask2FormerConfig):
    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        if isinstance(
            self.backbone_config, (Dinov2BackBoneWithFPNConfig, Dinov2AdapterConfig)
        ):
            # save the backbone config before replacing the backbone config with a default one
            # and saving the rest of the model
            self.backbone_config.save_pretrained(
                os.path.join(save_directory, "backbone_config")
            )
            # set it the default one used by the pre-configured model, first back it up to restore after saving
            backbone_config = self.backbone_config
            if self._name_or_path:
                # set it the default one used by the pre-configured model
                self.backbone_config = Mask2FormerConfig.from_pretrained(
                    self._name_or_path
                ).backbone_config
            else:
                # the default backbone of Mask2FormerConfig() may not be compatible with the rest
                # of the model, but this not important as we will replace it when reading the config (from_pretrained below)
                self.backbone_config = Mask2FormerConfig().backbone_config

            # save main Mask2Former config
            super().save_pretrained(save_directory, **kwargs)
            # restore the backbone config
            self.backbone_config = backbone_config
        else:
            super().save_pretrained(save_directory, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # load Mask2Former config, the config here has the default backbone_config
        # (because we have saved it like that here in the checkpoint)
        # so there is no issue here
        config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # try loading the backbone config if present
        backbone_config_path = os.path.join(
            pretrained_model_name_or_path, "backbone_config"
        )
        if os.path.exists(backbone_config_path):
            backbone_config = Dinov2BackBoneWithFPNConfig.from_pretrained(
                backbone_config_path
            )
        else:
            # use the one read
            backbone_config = config.backbone_config

        config.backbone_config = backbone_config
        # save the pretrained_model_name_or_path in the config as it will be used to built the model
        config._name_or_path = pretrained_model_name_or_path
        # return combined config
        return config


class Mask2FormerSegmentationWithCustomBackbone(Mask2FormerForUniversalSegmentation):
    config_class = Mask2FormerConfigWithCustomBackBone

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, *model_args, config=None, **kwargs
    ):
        # if config isn't passed, load it manually
        if config is None:
            config = Mask2FormerConfigWithCustomBackBone.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )

        # initialize the model
        model = cls(config, *model_args)

        # load FPN weights from safetensors file if they exist
        safe_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        if os.path.isfile(safe_path):
            state_dict = safe_load(safe_path)
            model.load_state_dict(state_dict, strict=False)

        return model

    def __init__(self, config):
        if isinstance(
            config.backbone_config, (Dinov2BackBoneWithFPNConfig, Dinov2AdapterConfig)
        ):
            print(
                f"[INFO]: The passed config.backbone_config is a custom one of type {type(config.backbone_config).__name__}. "
                f"The model and the backbone are instantiated separately as the backbone is not supported by the original model."
            )
            # save the backbone config before initializing the model
            backbone_config = config.backbone_config
            # set it the default one used by the pre-configured model
            if config._name_or_path:
                # set it the default one used by the pre-configured model
                print(
                    f"[INFO]: The Mask2Former model is instantiated with the backbone_config taken from {config._name_or_path}. "
                    f"This backbone will be replaced with the custom one once the Mask2Former model is instantiated."
                )
                config.backbone_config = Mask2FormerConfig.from_pretrained(
                    config._name_or_path
                ).backbone_config
            else:
                # this part may lead to incorrectly built Mask2Former model; this is because the feature map resolutions of the
                # backbone
                print(
                    f"[INFO]: The Mask2Former model is instantiated with the default backbone config!"
                    f"This backbone will be replaced with the custom one once the Mask2Former model is instantiated."
                )
                config.backbone_config = Mask2FormerConfig().backbone_config

            # initialize the model
            super().__init__(config)

            config.backbone_config = backbone_config
            # now configure the backbone, the backbone here may not be consistent with the rest of the model if the ERROR above is shown
            # note that Dinov2BackBoneWithFPN(backbone_config) automatically load the DINOv2 pre-trained weights from the checkpoint
            # included in backbone_config and freeze them
            if isinstance(backbone_config, Dinov2AdapterConfig):
                self.model.pixel_level_module.encoder = Dinov2Adapter(backbone_config)
            else:
                self.model.pixel_level_module.encoder = Dinov2BackBoneWithFPN(
                    backbone_config
                )

            # In the original implementation, the Swin transformer returns feature maps with dimensions
            # - 192 (1/4 resolution)
            # - 384 (1/8 resolution)
            # - 768 (1/16 resolution)
            # - 1536 (1/32 resolution)
            # and then applu 1x1 prejections to change these dimentions to 256. In the DINOv2 embeddings, we can return the
            # same feature map dimensions, or return all 256 and change the following prejection layers in the model
            # for now, return as the Swin transformer backbone does

            # get rid of the projection layers in the decoder
            # self.model.pixel_level_module.decoder.input_projections = nn.ModuleList(
            #     [nn.GroupNorm(32, 256, eps=1e-05, affine=True) for _ in range(3)]
            # )
            # self.model.pixel_level_module.decoder.adapter_1 = nn.GroupNorm(32, 256, eps=1e-05, affine=True)
        else:
            # initialize the model
            super().__init__(config)

        self.post_init()
