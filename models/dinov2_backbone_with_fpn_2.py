import os
from typing import Tuple, Union, List, Dict, Final, Optional

import torch
import torch.nn as nn

from transformers import PreTrainedModel, PretrainedConfig, Dinov2Model, Dinov2Config
from transformers.modeling_outputs import BackboneOutput
from safetensors.torch import save_file as safe_save
from safetensors.torch import load_file as safe_load

class FusedFPN(nn.Module):
    def __init__(
        self, 
        input_dim: int,                    # DINOv2 embedding size
        output_dims: List[int],            # the list of feature sizes for each feature map
        resolutions: List[Tuple[int, int]] = None
    ):
        # resolutions is a list of the same size as output_dims indicating the requires resolution (in (height, width) for each feature map
        # when None is passed, the last feature map will be L/2 x L/2, where L is the DINOv2 feature resolution (image size / patch size) 
        # and each layer down will be doubled, e.g., for 4 layers, the sizes will be 4L x 4L, 2L x 2L, L x L and L/2 x L/2 
        super().__init__()

        self.output_dims = output_dims
        self.resolutions = resolutions
        if resolutions is not None:
            assert len(output_dims) == len(resolutions), "The number of output resolutions and dimentions should be the same"
        # convolutions to change the embeddings dimensions
        self.lateral_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(32, output_dim),
                )
                for output_dim in output_dims
            ]
        )

        # fusion convolusions, top-down
        # this is how the feature map fusion is done
        # the last/top feature map (the lowest resolution, richest semantics) is obtained from the 
        # last DINOv2 hidden layer and resizing either by a factor of 1/2, or to an exact size if provided in the resolutions
        # (this is needed for Mask2Former); no fusion with other layers is done here for this top feature map layer - just another convolution
        # then for any i < last, feature map i after going through the lateral convolution (to correct the number of channels) and the 
        # fused features at layer i + 1 are both extrapolated to the expected size for map i, concatenated and then fused
        self.fusion_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(output_dims[i] + output_dims[i + 1], output_dims[i], kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(32, output_dims[i]),
                    nn.GELU(),
                )
                for i in range(len(output_dims) - 1) # exclude the last feature map that will not be fused with any other features
            ] + [ # last feature map, no fusion
                nn.Sequential(
                    nn.Conv2d(output_dims[-1], output_dims[-1], kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(32, output_dims[-1]),
                    nn.GELU(),
                )
            ]
        )
        
    def forward(self, features):
        assert len(features) == len(self.output_dims), "The number of input features should be the same as the number of output features"
        # features = list of feature maps from DINOv2, each feature map is B x 1024 x L x L (L = 48 for input size 672)
        fused_features = [None] * len(features)
        # last_feature_map is B x output_dims[-1] x L x L
        last_feature_map = self.lateral_convs[-1](features[-1])
        if self.resolutions is not None:
            # resize to the exact size needed for this feature map, otherwise, resize to B x output_dims[-1] x L/2 x L/2
            last_feature_map_resized  = nn.functional.interpolate(last_feature_map, size=self.resolutions[-1], mode="bilinear", align_corners=False)
        else:
            last_feature_map_resized = nn.functional.interpolate(last_feature_map, scale_factor=0.5, mode="bilinear", align_corners=False)

        fused_features[-1] = self.fusion_convs[-1](last_feature_map_resized) # no concatenation here

        # i goes over [0, 1, 2, ..., len(self.output_dims) -2] in reverse order
        for i in range(len(self.output_dims) - 2, -1, -1):
            # layer_i_features is of size
            # B x output_dims[i - 1] x L x L
            layer_i_features = self.lateral_convs[i](features[i])
            if i == len(self.output_dims) - 2:
                # for the feature map just before the last one, use
                # the original resolution (without going through downsampling) of the last feature map
                # we are doing this to prevent any lost information in downsizing and upsizing of the last feature map
                # but I don't think it matters
                next_layer_features = last_feature_map
            else:
                next_layer_features = fused_features[i + 1]
            if self.resolutions is not None:
                # resize to the exact size needed for this feature map
                layer_i_features_resized = nn.functional.interpolate(
                    layer_i_features, 
                    size=self.resolutions[i], 
                    mode="bilinear",
                    align_corners=False
                )
                next_layer_features_resized = nn.functional.interpolate(
                    next_layer_features, 
                    size=self.resolutions[i],
                    mode="bilinear", 
                    align_corners=False
                )
            else:
                # when no size specified, the last feature map will be L/2 x L/2 and each layer down will be doubles, 
                # e.g., for 4 layers, the sizes will be 4L x 4L, 2L x 2L, L x L and L/2 x L/2 
                scale = 2 ** (len(self.output_dims) - 2 - i)
                out_features_size = (layer_i_features.shape[-2] * scale, layer_i_features.shape[-1] * scale)
                layer_i_features_resized = nn.functional.interpolate(
                    layer_i_features, 
                    size=out_features_size, 
                    mode="bilinear",
                    align_corners=False
                )
                next_layer_features_resized = nn.functional.interpolate(
                    next_layer_features, 
                    size=out_features_size, 
                    mode="bilinear", 
                    align_corners=False
                )

            fused_features[i] = self.fusion_convs[i](torch.cat([layer_i_features_resized, next_layer_features_resized], dim=1))
       
        return fused_features


class Dinov2BackBoneWithFPNConfig(Dinov2Config):
    model_type = "dinov2_backbone_with_fpn"
    def __init__(
        self, 
        dinov2_pretrained_backbone_name_or_path: str = '',
        output_indices_for_fpn: List[int]= [8, 10, 12], 
        intermediate_channel_sizes: List[int] = [128, 256, 512], # this is feature map dims of RT-DETRv2 default backbone
        intermediate_resolutions: List[Tuple[int, int]] | None = None, # this is the resolution of each feature map in (height, width)
                                                                       # should be the same length as intermediate_channel_sizes or None
                                                                       # for L/2, L, 2L, 4L, ... where L is img_size/patch_size of DINOv2 embs
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dinov2_pretrained_backbone_name_or_path: str = dinov2_pretrained_backbone_name_or_path
        self.num_fpn_layers: int = len(output_indices_for_fpn)
        self.output_indices_for_fpn: List[int] = output_indices_for_fpn
        self.intermediate_channel_sizes: List[int] = intermediate_channel_sizes
        self.intermediate_resolutions: List[Tuple[int, int]] = intermediate_resolutions

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # call superclass method to load config dict
        config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        # set name/path for future reference (used by model to load DINOv2 weights)
        config._name_or_path = pretrained_model_name_or_path

        # also set dinov2_pretrained_backbone_name_or_path if not explicitly set 
        # this happens when a DINOv2 checkpoint is passed in pretrained_model_name_or_path
        if not config.dinov2_pretrained_backbone_name_or_path:
            config.dinov2_pretrained_backbone_name_or_path = pretrained_model_name_or_path

        return config

class Dinov2BackBoneWithFPN(PreTrainedModel):
    
    config_class = Dinov2BackBoneWithFPNConfig

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, config=None, **kwargs):
        # if config isn't passed, load it manually
        if config is None:
            config = Dinov2BackBoneWithFPNConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
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
            self.backbone = Dinov2Model.from_pretrained(config.dinov2_pretrained_backbone_name_or_path)
            freeze_dinov2_weights: bool = True
            print(f"[INFO]: DINOv2 parameters loaded from pretrained path: {config.dinov2_pretrained_backbone_name_or_path}")
        else:
            # otherwise, randomly initialize the weights for DINOv2
            print(f"[WARN]: No path was provided in the config to load DINOv2 parameters. This backbone has to be trained!")
            self.backbone = Dinov2Model(config)
            freeze_dinov2_weights: bool = False

        # freeze backbone parameters
        if freeze_dinov2_weights:
            for param in self.backbone.parameters():
                param.requires_grad_(False)
        
        # the DINOv2 hidden later outputs are a list of tensors of size (B, C, H, W) with C=768 or 1024
        # apply TinyFPN to create multi-scale features
        self.fpn = FusedFPN(
            input_dim=config.hidden_size, 
            output_dims=config.intermediate_channel_sizes, 
            resolutions=config.intermediate_resolutions
        )
 
        self.post_init()
    
    def forward(self, pixel_values):
        # step 1: extract multiple feature maps
        backbone_outputs = self.backbone(pixel_values, output_hidden_states=True)
        feature_maps = [backbone_outputs.hidden_states[i] for i in self.output_indices_for_fpn]  
        
        processed_feats = []
        for features in feature_maps:
            # reshape if needed (ViT outputs may be (B, N, C))
            B, N, C = features.shape
            # subtract the class token
            H = W = int((N - 1) ** 0.5)
            # reshape
            processed_feats.append(features[:, 1:, :].transpose(1, 2).reshape(B, C, H, W))
            
        # step 2: Apply FPN
        multi_scale_feats = self.fpn(processed_feats)
        # print([feat.shape for feat in multi_scale_feats])
        return BackboneOutput(multi_scale_feats)

