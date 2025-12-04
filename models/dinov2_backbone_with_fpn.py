import os
from typing import Tuple, Union, List, Dict, Final, Optional

import torch
import torch.nn as nn

from transformers import PreTrainedModel, PretrainedConfig, Dinov2Model, Dinov2Config
from safetensors.torch import save_file as safe_save
from safetensors.torch import load_file as safe_load

from .dinov2_backbone_with_tiny_fpn import TinyFPN
import torchshow as ts

def plot_embeddings(features, title="Feature Maps"):
    # features: list of fqeature maps from DINOv2
    # only plot a max of 8 feature maps
    feat = features[:, 1:, :]  # remove class token
    B, N, C = feat.shape
    grid_size = min(int(C ** 0.5), 4)
    H = W = int((N) ** 0.5)
    # take the first image in the batch
    feat_img = feat.permute(0, 2, 1).view(B, C, H, W)[0]  # C x H x W
    # normalize to [0, 1]
    feat_img = (feat_img - feat_img.min()) / (feat_img.max() - feat_img.min() + 1e-5)
    # feat_img = feat.norm(dim=-1)
    # reshape to grid
    
    feat_img = feat_img[:grid_size * grid_size].reshape(grid_size, grid_size, H, W)
    feat_img = feat_img.permute(0, 2, 1, 3).reshape(grid_size * H, grid_size * W)
    # breakpoint()
    ts.save(feat_img, title=f"{title}.png", cmap='magma')

class FusedFPN(nn.Module):
    def __init__(self, input_dim, out_dims, scale_factor=1):
        super().__init__()
        self.scale_factor = scale_factor
        # 1x1 lateral convolutions (projections to change the number of channels)
        self.lateral_conv_2 = nn.Conv2d(input_dim, out_dims[0], kernel_size=1)
        self.lateral_conv_3 = nn.Conv2d(input_dim, out_dims[1], kernel_size=1)
        self.lateral_conv_4 = nn.Conv2d(input_dim, out_dims[2], kernel_size=1)
        
        # fusion (3x3 convs after addition)
        # we down-sample the resolution of only the last layer by a factor of 2
        # we add that at the end
        self.fusion_conv_2 = nn.Sequential(
            nn.Conv2d(out_dims[0] + out_dims[1], out_dims[0], kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, out_dims[0]),
            nn.ReLU(inplace=True)
        )
        self.fusion_conv_3 = nn.Sequential(
            nn.Conv2d(out_dims[1] + out_dims[2], out_dims[1], kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, out_dims[1]),
            nn.ReLU(inplace=True)
        )
        
        # the last later, since this layer will not be fused with any other layer, out_dims[2]
        # is the input channel dimention
        self.fusion_conv_4 = nn.Sequential(
            nn.Conv2d(out_dims[2], out_dims[2], kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, out_dims[2]),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features):
        # features = list of feature maps from DINOv2
        # breakpoint()
        features_2, features_3, features_4 = features # each feature map is B x 1024 x L x L (L = 48 for input size 672)
        # l_4 is B x 512 x L x L
        l_4 = self.lateral_conv_4(features_4)
        # p_4 is B x 512 x L/2 x L/2
        p_4 = self.fusion_conv_4(l_4)
        # upsample p_4 to L x L
        p_4_up = nn.functional.interpolate(p_4, scale_factor=2.0, mode='bilinear', align_corners=False)
        # l_3 is B x 256 x L x L
        l_3 = self.lateral_conv_3(features_3)
        # torch.cat([l_3, l_4], dim=1) is B x (256 + 512) x L x L
        # p_3 is B x 256, L X L
        p_3 = self.fusion_conv_3(torch.cat([l_3, l_4], dim=1)) 
        # upsample p_3 to 2L x 2L
        p_3_up = nn.functional.interpolate(p_3, scale_factor=2.0, mode='bilinear', align_corners=False)
        # l_2 is B x 128 x L x L
        l_2 = self.lateral_conv_2(features_2)
        # l_2_up is B x 128 x 2*L x 2*L
        # TODO: upsample from 2x -> 4x
        l_2_up = nn.functional.interpolate(l_2, scale_factor=2.0, mode='bilinear', align_corners=False)
        # l_3_up is B x 256 x 2*L x 2*L
        l_3_up = nn.functional.interpolate(l_3, scale_factor=2.0, mode='bilinear', align_corners=False)
        # torch.cat([l_2_up, l_3_up], dim=1) is B x (128 + 256) x 2*L x 2*L
        # p_2 is B x 128, 2*L x 2*L
        p_2 = self.fusion_conv_2(torch.cat([l_2_up, l_3_up], dim=1)) 
        
        p_2_up = nn.functional.interpolate(p_2, scale_factor=2.0, mode='bilinear', align_corners=False)
        if self.scale_factor == 1:
            return [p_2, p_3, p_4]
        elif self.scale_factor == 2:
            return [p_2_up, p_3_up, p_4_up]

class Dinov2BackBoneWithFPNConfig(Dinov2Config):
    model_type = "dinov2_backbone_with_fpn"
    def __init__(
        self, 
        dinov2_pretrained_backbone_name_or_path: str = '',
        first_layer_dims:list[int, int] = [80, 80], # to be consistent with the feature map dims of RT-DETRv2 default backbone
        output_indices_for_fpn: List[int]= [8, 10, 12], 
        intermediate_channel_sizes: List[int] = [128, 256, 512], # to be consistent with the feature map dims of RT-DETRv2 default backbone 
        fpn_type: str = 'fused', # choices fused/tiny/none
        scale_factor: int =1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dinov2_pretrained_backbone_name_or_path: str = dinov2_pretrained_backbone_name_or_path
        self.num_fpn_layers: int = len(output_indices_for_fpn)
        self.output_indices_for_fpn: List[int] = output_indices_for_fpn
        self.intermediate_channel_sizes: List[int] = intermediate_channel_sizes
        self.first_layer_dims: List[int, int] = first_layer_dims
        self.fpn_type: str = fpn_type
        self.scale_factor: int = scale_factor

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
        # breakpoint()
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
        # breakpoint()
        # config.fpn_type = 'FusedFPN'  # default FPN type
        # needed to be used as a backbone for RT-DETR (it is used to build encoder projection conv layers)
        self.intermediate_channel_sizes = config.intermediate_channel_sizes
        self.output_indices_for_fpn = config.output_indices_for_fpn
        self.fpn_type = config.fpn_type
        self.first_layer_dims = config.first_layer_dims
        self.scale_factor = config.scale_factor
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
        # breakpoint()
        if config.fpn_type == 'tiny':
            self.fpn = TinyFPN(
                input_dim=config.hidden_size, 
                out_dims=config.intermediate_channel_sizes, 
                first_layer_dims=config.first_layer_dims,
                scale_factor=config.scale_factor,
            )
        elif config.fpn_type == 'fused':
            self.fpn = FusedFPN(
                input_dim=config.hidden_size, 
                out_dims=config.intermediate_channel_sizes, 
                scale_factor=config.scale_factor,
            )
        else:
            # self.fpn = FusedFPN(
            #     input_dim=config.hidden_size, 
            #     out_dims=config.intermediate_channel_sizes, 
            # )
            raise ValueError(f"Unsupported FPN type: {config.fpn_type}")
        
        self.post_init()
    
    def forward(self, pixel_values, pixel_mask=None): # pixel_mask incldued for compatibility with RT-DETR backbone 
        # step 1: extract multiple feature maps
        # breakpoint()
        backbone_outputs = self.backbone(pixel_values, output_hidden_states=True)
        feature_maps = [backbone_outputs.hidden_states[i] for i in self.output_indices_for_fpn]  # 3 layers (8, 10, 12 for the base)
        # plot_embeddings(feature_maps[0], title="DINOv2_Feature_Map_Layer_" + str(self.output_indices_for_fpn[0]))
        # plot_embeddings(feature_maps[1], title="DINOv2_Feature_Map_Layer_" + str(self.output_indices_for_fpn[1]))
        # plot_embeddings(feature_maps[2], title="DINOv2_Feature_Map_Layer_" + str(self.output_indices_for_fpn[2]))
        # reshape feature maps to (B, C, H, W)
        processed_feats = []
        for features in feature_maps:
            # reshape if needed (ViT outputs may be (B, N, C))
            B, N, C = features.shape
            # subtract the class token
            H = W = int((N - 1) ** 0.5)
            # TODO: replace transpose with permute?
            # breakpoint()
            processed_feats.append(features[:, 1:, :].transpose(1, 2).contiguous().reshape(B, C, H, W))
            
        # step 2: Apply FPN

        multi_scale_feats = self.fpn(processed_feats)
        # print([feat.shape for feat in multi_scale_feats])
        # breakpoint()
        out = []
        for feature_map in multi_scale_feats:
            if pixel_mask is not None:
                mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
                out.append((feature_map, mask))
            else:
                out.append((feature_map,))

        return out