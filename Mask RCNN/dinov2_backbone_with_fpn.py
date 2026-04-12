import os
from typing import List

import torch
import torch.nn as nn

from transformers import PreTrainedModel, Dinov2Model, Dinov2Config
from safetensors.torch import save_file as safe_save
from safetensors.torch import load_file as safe_load

class FusedFPN_New(nn.Module):
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
        # the last/top feature map (the lowest resolution, richest semantics) is obtained by downsampling the 
        # last DINOv2 hidden layer by a factor of 2 and potentially resizing to an exact size (needed for Mask2Former); no fusion is done
        # with other layers for this top feature map layer
        # then for any i < last, feature map i and i + 1 will be extrapolated to the expected size for map i, concatenated and then fused
        self.fusion_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(output_dims[i] + output_dims[i + 1], output_dims[i], kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(32, output_dims[i]),
                    nn.GELU(),
                )
                for i in range(len(output_dims) - 1) # exclude the last feature map that we will downsampled by a factor of two
            ] + [ # last feature map, downsampling by a factor of 2 through convolution and no fusion
                nn.Sequential(
                    nn.Conv2d(output_dims[-1], output_dims[-1], kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, output_dims[-1]),
                    nn.GELU(),
                )
            ]
        )
        
    def forward(self, features):
        assert len(features) == len(self.output_dims), "The number of input features should be the same as the number of output features"
        # features = list of feature maps from DINOv2, each feature map is B x 1024 x L x L (L = 48 for input size 672)
        fused_features = [None] * len(features)
        # last_feature_map is B x output_dims[-1] x L/2 x L/2 here
        last_feature_map = self.fusion_convs[-1](self.lateral_convs[-1](features[-1]))
        if self.resolutions is not None:
            # resize to the exact size needed for this feature map, otherwise, leave it at B x output_dims[-1] x L/2 x L/2
            nn.functional.interpolate(last_feature_map, size=self.resolutions[-1], mode="bilinear", align_corners=False)

        fused_features[-1] = last_feature_map

        # i goes over [0, 1, 2, ..., len(self.output_dims) -2] in reverse order
        for i in range(len(self.output_dims) - 2, -1, -1):
            # layer_i_features is of size
            # B x output_dims[i - 1] x L x L
            layer_i_features = self.lateral_convs[i](features[i])
            if i == len(self.output_dims) - 2:
                # for the next of last (top) feature, use the unfused feature map as top level because it has higher resolution
                # and hence more info (this is consistent with the old implementation)
                next_layer_features = features[i + 1]
            else:
                next_layer_features = fused_features[i + 1]
            if self.resolutions is not None:
                # resize to the exact size needed for this feature map
                layer_i_features_up = nn.functional.interpolate(
                    layer_i_features, 
                    size=self.resolutions[i], 
                    mode="bilinear",
                    align_corners=False
                )
                next_layer_features_up = nn.functional.interpolate(
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
                layer_i_features_up = nn.functional.interpolate(
                    layer_i_features, 
                    size=out_features_size, 
                    mode="bilinear",
                    align_corners=False
                )
                next_layer_features_up = nn.functional.interpolate(
                    next_layer_features, 
                    size=out_features_size, 
                    mode="bilinear", 
                    align_corners=False
                )

            fused = self.fusion_convs[i](torch.cat([layer_i_features_up, next_layer_features_up], dim=1))
            fused_features[i] = fused
       
        return fused_features

class FusedFPN(nn.Module):
    def __init__(self, input_dim, out_dims):
        super().__init__()

        # 1x1 lateral convolutions (projections to change the number of channels)
        self.lateral_conv_2 = nn.Conv2d(input_dim, out_dims[0], kernel_size=1)
        self.lateral_conv_3 = nn.Conv2d(input_dim, out_dims[1], kernel_size=1)
        self.lateral_conv_4 = nn.Conv2d(input_dim, out_dims[2], kernel_size=1)

        # fusion (3x3 convs after addition)
        # we down-sample the resolution of only the last layer by a factor of 2
        # we add that at the end
        self.fusion_conv_2 = nn.Sequential(
            nn.Conv2d(
                out_dims[0] + out_dims[1],
                out_dims[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GroupNorm(32, out_dims[0]),
            nn.ReLU(inplace=True),
        )
        self.fusion_conv_3 = nn.Sequential(
            nn.Conv2d(
                out_dims[1] + out_dims[2],
                out_dims[1],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GroupNorm(32, out_dims[1]),
            nn.ReLU(inplace=True),
        )

        # the last later, since this layer will not be fused with any other layer, out_dims[2]
        # is the input channel dimention
        self.fusion_conv_4 = nn.Sequential(
            nn.Conv2d(out_dims[2], out_dims[2], kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, out_dims[2]),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        # features = list of feature maps from DINOv2
        features_2, features_3, features_4 = (
            features  # each feature map is B x 1024 x L x L (L = 48 for input size 672)
        )
        # l_4 is B x 512 x L x L
        l_4 = self.lateral_conv_4(features_4)
        # p_4 is B x 512 x L/2 x L/2
        p_4 = self.fusion_conv_4(l_4)
        # l_3 is B x 256 x L x L
        l_3 = self.lateral_conv_3(features_3)
        # torch.cat([l_3, l_4], dim=1) is B x (256 + 512) x L x L
        # p_3 is B x 256, L X L
        p_3 = self.fusion_conv_3(torch.cat([l_3, l_4], dim=1))
        # l_2 is B x 128 x L x L
        l_2 = self.lateral_conv_2(features_2)
        # l_2_up is B x 128 x 2*L x 2*L
        l_2_up = nn.functional.interpolate(
            l_2, scale_factor=2.0, mode="bilinear", align_corners=False
        )
        # l_3_up is B x 256 x 2*L x 2*L
        l_3_up = nn.functional.interpolate(
            l_3, scale_factor=2.0, mode="bilinear", align_corners=False
        )
        # torch.cat([l_2_up, l_3_up], dim=1) is B x (128 + 256) x 2*L x 2*L
        # p_2 is B x 128, 2*L x 2*L
        p_2 = self.fusion_conv_2(torch.cat([l_2_up, l_3_up], dim=1))

        return [p_2, p_3, p_4]


class Dinov2BackBoneWithFPNConfig(Dinov2Config):
    model_type = "dinov2_backbone_with_fpn"

    def __init__(
        self,
        dinov2_pretrained_backbone_name_or_path: str = "",
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
        self.fpn = FusedFPN(
            input_dim=config.hidden_size,
            out_dims=config.intermediate_channel_sizes,
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
