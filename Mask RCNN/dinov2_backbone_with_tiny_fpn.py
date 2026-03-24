import os
from typing import Tuple, List

import torch
import torch.nn as nn

from transformers import PreTrainedModel, Dinov2Model, Dinov2Config
from safetensors.torch import save_file as safe_save
from safetensors.torch import load_file as safe_load


class RescaleFeatures(nn.Module):
    def __init__(self, size, mode="bilinear"):
        super().__init__()
        self.size = size
        self.mode = mode

    def forward(self, x):
        return nn.functional.interpolate(
            x,
            size=self.size,
            # scale_factor=2,
            mode=self.mode,
            align_corners=False,
        )


class TinyFPN(nn.Module):
    def __init__(self, input_dim, out_dims, first_layer_dims):
        super().__init__()

        self.lateral_convs = nn.ModuleList(
            [
                nn.Sequential(
                    RescaleFeatures(first_layer_dims),
                    nn.Conv2d(input_dim, out_dim, kernel_size=1),
                )
                for out_dim in out_dims
            ]
        )

        self.feature_pyramid = nn.ModuleList([])
        for i, out_dim in enumerate(out_dims):
            self.feature_pyramid.append(
                nn.Sequential(
                    nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2**i, padding=1),
                    nn.GroupNorm(32, out_dim),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, features):
        # features = list of feature maps from DINOv2
        outs = []
        for x, conv, resize in zip(features, self.lateral_convs, self.feature_pyramid):
            outs.append(resize(conv(x)))
        return outs  # list of [P3, P4, P5] features


class Dinov2BackBoneWithFPNConfig(Dinov2Config):
    model_type = "dinov2_backbone_with_fpn"

    def __init__(
        self,
        dinov2_pretrained_backbone_name_or_path: str = "",
        first_layer_dims: Tuple[int, int] = (
            80,
            80,
        ),  # to be consistent with the feature map dims of RT-DETRv2 default backbone
        output_indices_for_fpn: List[int] = [8, 10, 12],
        intermediate_channel_sizes: List[int] = [
            128,
            256,
            512,
        ],  # to be consistent with the feature map dims of RT-DETRv2 default backbone
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dinov2_pretrained_backbone_name_or_path: str = (
            dinov2_pretrained_backbone_name_or_path
        )
        self.first_layer_dims: Tuple[int, int] = first_layer_dims
        self.num_fpn_layers: int = len(output_indices_for_fpn)
        self.output_indices_for_fpn: List[int] = output_indices_for_fpn
        self.intermediate_channel_sizes: List[int] = intermediate_channel_sizes

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # call superclass method to load config dict
        config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # set name/path for future reference (used by model to load DINOv2 weights)
        config._name_or_path = pretrained_model_name_or_path

        # also set dinov2_pretrained_backbone_name_or_path if not explicitly set
        # this happens when a DINOv2 checkpoint is passed in pretrained_model_name_or_path
        if not config.dinov2_pretrained_backbone_name_or_path:
            config.dinov2_pretrained_backbone_name_or_path = (
                pretrained_model_name_or_path
            )

        return config


class Dinov2BackBoneWithFPN(PreTrainedModel):
    config_class = Dinov2BackBoneWithFPNConfig

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, *model_args, config=None, **kwargs
    ):
        # if config isn't passed, load it manually
        if config is None:
            config = Dinov2BackBoneWithFPNConfig.from_pretrained(
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

    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)

        # save config to config.json
        self.config.save_pretrained(save_directory)

        # Save as safetensors (this is the new standard)
        safe_path = os.path.join(save_directory, "model.safetensors")
        safe_save(self.state_dict(), safe_path)

    def __init__(self, config):
        super().__init__(config)

        # needed to be used as a backbone for RT-DETR (it is used to build encoder projection conv layers)
        self.intermediate_channel_sizes = config.intermediate_channel_sizes
        self.output_indices_for_fpn = config.output_indices_for_fpn
        if config.dinov2_pretrained_backbone_name_or_path:
            # load pre-trained DINOv2 weights if a given path is specified in the config
            self.backbone = Dinov2Model.from_pretrained(
                config.dinov2_pretrained_backbone_name_or_path
            )
            freeze_dinov2_weights: bool = True
            print(
                f"[INFO]: DINOv2 parameters loaded from pretrained path: {config.dinov2_pretrained_backbone_name_or_path}"
            )
        else:
            # otherwise, randomly initialize the weights for DINOv2
            print(
                "[WARN]: No path was provided in the config to load DINOv2 parameters. This backbone has to be trained!"
            )
            self.backbone = Dinov2Model(config)
            freeze_dinov2_weights: bool = False

        # freeze backbone parameters
        if freeze_dinov2_weights:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

        # the DINOv2 hidden later outputs are a list of tensors of size (B, C, H, W) with C=768 or 1024
        # apply TinyFPN to create multi-scale features
        self.fpn = TinyFPN(
            input_dim=config.hidden_size,
            out_dims=config.intermediate_channel_sizes,
            first_layer_dims=config.first_layer_dims,
        )

        self.post_init()

    def forward(
        self, pixel_values, pixel_mask=None
    ):  # pixel_mask incldued for compatibility with RT-DETR backbone
        # step 1: extract multiple feature maps
        backbone_outputs = self.backbone(pixel_values, output_hidden_states=True)
        feature_maps = [
            backbone_outputs.hidden_states[i] for i in self.output_indices_for_fpn
        ]  # 3 layers (8, 10, 12 for the base)

        processed_feats = []
        for features in feature_maps:
            # reshape if needed (ViT outputs may be (B, N, C))
            B, N, C = features.shape
            # subtract the class token
            H = W = int((N - 1) ** 0.5)
            processed_feats.append(
                features[:, 1:, :].transpose(1, 2).reshape(B, C, H, W)
            )

        # step 2: Apply FPN
        multi_scale_feats = self.fpn(processed_feats)
        # print([feat.shape for feat in multi_scale_feats])
        out = []
        for feature_map in multi_scale_feats:
            if pixel_mask is not None:
                mask = nn.functional.interpolate(
                    pixel_mask[None].float(), size=feature_map.shape[-2:]
                ).to(torch.bool)[0]
                out.append((feature_map, mask))
            else:
                out.append((feature_map,))

        return out
