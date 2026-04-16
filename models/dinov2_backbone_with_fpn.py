import os
from typing import Tuple, Union, List, Dict, Final, Optional

import torch
import torch.nn as nn

from transformers import PreTrainedModel, PretrainedConfig, Dinov2Model, Dinov2Config
from transformers.modeling_outputs import BackboneOutput
from safetensors.torch import save_file as safe_save
from safetensors.torch import load_file as safe_load


def _init_fpn_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif (
        isinstance(m, nn.GroupNorm)
        or isinstance(m, nn.BatchNorm2d)
        or isinstance(m, nn.SyncBatchNorm)
    ):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class FusedFPN(nn.Module):
    def __init__(
        self,
        input_dim: int,  # DINOv2 embedding size
        out_dims: List[int],  # the list of feature sizes for each feature map
        resolutions: Optional[List[int]] = None,
    ):
        super().__init__()

        self.out_dims = out_dims
        self.resolutions = resolutions
        if resolutions is not None:
            assert len(out_dims) == len(resolutions), (
                "The number of output resolutions and dimensions should be the same"
            )

        # convolutions to change the embeddings dimensions
        self.lateral_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(32, output_dim),
                )
                for output_dim in out_dims
            ]
        )

        self.fusion_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        out_dims[i] + out_dims[i + 1],
                        out_dims[i],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.GroupNorm(32, out_dims[i]),
                    nn.GELU(),
                )
                for i in range(len(out_dims) - 1)
            ]
            + [
                nn.Sequential(
                    nn.Conv2d(
                        out_dims[-1], out_dims[-1], kernel_size=3, stride=1, padding=1
                    ),
                    nn.GroupNorm(32, out_dims[-1]),
                    nn.GELU(),
                )
            ]
        )

        self.apply(_init_fpn_weights)

    def forward(self, features, target_sizes=None):
        assert len(features) == len(self.out_dims), (
            "The number of input features should be the same as the number of output features"
        )
        fused_features = [None] * len(features)

        last_feature_map = self.lateral_convs[-1](features[-1])
        if target_sizes is not None:
            last_feature_map_resized = nn.functional.interpolate(
                last_feature_map,
                size=target_sizes[-1],
                mode="bilinear",
                align_corners=False,
            )
        elif self.resolutions is not None:
            res = self.resolutions[-1]
            last_feature_map_resized = nn.functional.interpolate(
                last_feature_map, size=(res, res), mode="bilinear", align_corners=False
            )
        else:
            last_feature_map_resized = nn.functional.interpolate(
                last_feature_map, scale_factor=0.5, mode="bilinear", align_corners=False
            )

        fused_features[-1] = self.fusion_convs[-1](
            last_feature_map_resized
        )  # no concatenation here

        for i in range(len(self.out_dims) - 2, -1, -1):
            layer_i_features = self.lateral_convs[i](features[i])
            if i == len(self.out_dims) - 2:
                next_layer_features = last_feature_map
            else:
                next_layer_features = fused_features[i + 1]
            if target_sizes is not None:
                layer_i_features_resized = nn.functional.interpolate(
                    layer_i_features,
                    size=target_sizes[i],
                    mode="bilinear",
                    align_corners=False,
                )
                next_layer_features_resized = nn.functional.interpolate(
                    next_layer_features,
                    size=target_sizes[i],
                    mode="bilinear",
                    align_corners=False,
                )
            elif self.resolutions is not None:
                res = self.resolutions[i]
                layer_i_features_resized = nn.functional.interpolate(
                    layer_i_features,
                    size=(res, res),
                    mode="bilinear",
                    align_corners=False,
                )
                next_layer_features_resized = nn.functional.interpolate(
                    next_layer_features,
                    size=(res, res),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                scale = 2 ** (len(self.out_dims) - 2 - i)
                out_features_size = (
                    layer_i_features.shape[-2] * scale,
                    layer_i_features.shape[-1] * scale,
                )
                layer_i_features_resized = nn.functional.interpolate(
                    layer_i_features,
                    size=out_features_size,
                    mode="bilinear",
                    align_corners=False,
                )
                next_layer_features_resized = nn.functional.interpolate(
                    next_layer_features,
                    size=out_features_size,
                    mode="bilinear",
                    align_corners=False,
                )

            fused_features[i] = self.fusion_convs[i](
                torch.cat(
                    [layer_i_features_resized, next_layer_features_resized], dim=1
                )
            )

        return fused_features


class TinyFPN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dims: List[int],
        resolutions: Optional[List[int]] = None,
    ):
        super().__init__()
        self.resolutions = resolutions

        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(input_dim, out_dim, kernel_size=1) for out_dim in out_dims]
        )

        self.feature_pyramid = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(32, out_dim),
                    nn.ReLU(inplace=True),
                )
                for out_dim in out_dims
            ]
        )

    def forward(self, features, target_sizes=None):
        outs = []
        for i, (x, conv, py) in enumerate(
            zip(features, self.lateral_convs, self.feature_pyramid)
        ):
            feat = conv(x)
            if target_sizes is not None:
                feat = nn.functional.interpolate(
                    feat, size=target_sizes[i], mode="bilinear", align_corners=False
                )
            elif self.resolutions is not None:
                res = self.resolutions[i]
                feat = nn.functional.interpolate(
                    feat, size=(res, res), mode="bilinear", align_corners=False
                )
            else:
                scale = 2 ** (len(features) - 1 - i)
                if scale > 1:
                    feat = nn.functional.interpolate(
                        feat, scale_factor=scale, mode="bilinear", align_corners=False
                    )
            outs.append(py(feat))
        return outs


class SimpleFPN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dims: List[int],
        resolutions: Optional[List[int]] = None,
    ):
        super().__init__()
        self.resolutions = resolutions

        self.lateral_convs = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(input_dim, out_dim, kernel_size=1), nn.ReLU())
                for out_dim in out_dims
            ]
        )

    def forward(self, features, target_sizes=None):
        outs = []
        for i, (feat, conv) in enumerate(zip(features, self.lateral_convs)):
            x = conv(feat)
            if target_sizes is not None:
                x = nn.functional.interpolate(
                    x, size=target_sizes[i], mode="bilinear", align_corners=False
                )
            elif self.resolutions is not None:
                res = self.resolutions[i]
                x = nn.functional.interpolate(
                    x, size=(res, res), mode="bilinear", align_corners=False
                )
            else:
                scale = 2 ** (len(features) - 1 - i)
                if scale > 1:
                    x = nn.functional.interpolate(
                        x, scale_factor=scale, mode="bilinear", align_corners=False
                    )
            outs.append(x)
        return outs


class SFP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dims: List[int],
        resolutions: Optional[List[int]] = None,
    ):
        super().__init__()
        self.resolutions = resolutions
        c = input_dim

        self.fpns = nn.ModuleList()
        for i, out_dim in enumerate(out_dims):
            layers = []
            if i == 0 and len(out_dims) == 4:
                layers.extend(
                    [
                        nn.ConvTranspose2d(c, c, kernel_size=2, stride=2),
                        nn.SyncBatchNorm(c)
                        if torch.cuda.device_count() > 1
                        else nn.BatchNorm2d(c),
                        nn.GELU(),
                        nn.ConvTranspose2d(c, c, kernel_size=2, stride=2),
                    ]
                )
            elif i == len(out_dims) - 3:
                layers.append(nn.ConvTranspose2d(c, c, kernel_size=2, stride=2))
            elif i == len(out_dims) - 2:
                layers.append(nn.Identity())
            elif i == len(out_dims) - 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Identity())  # Fallback

            layers.append(nn.Conv2d(c, out_dim, kernel_size=1))
            self.fpns.append(nn.Sequential(*layers))

    def forward(self, features, target_sizes=None):
        outs = []
        for i, (feat, py) in enumerate(zip(features, self.fpns)):
            x = py(feat)
            if target_sizes is not None:
                x = nn.functional.interpolate(
                    x, size=target_sizes[i], mode="bilinear", align_corners=False
                )
            elif self.resolutions is not None:
                res = self.resolutions[i]
                x = nn.functional.interpolate(
                    x, size=(res, res), mode="bilinear", align_corners=False
                )
            else:
                scale = 2 ** (len(features) - 1 - i)
                if (
                    scale > 1 and len(features) != 4
                ):  # generic scaling if not matching exact SFP expectations
                    x = nn.functional.interpolate(
                        x, scale_factor=scale, mode="bilinear", align_corners=False
                    )
            outs.append(x)
        return outs


class Dinov2BackBoneWithFPNConfig(Dinov2Config):
    model_type = "dinov2_backbone_with_fpn"

    def __init__(
        self,
        dinov2_pretrained_backbone_name_or_path: str = "",
        output_indices_for_fpn: List[int] = [8, 10, 12],
        intermediate_channel_sizes: List[int] = [128, 256, 512],
        intermediate_resolutions: Optional[List[int]] = None,
        fpn_type: str = "fused",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dinov2_pretrained_backbone_name_or_path: str = (
            dinov2_pretrained_backbone_name_or_path
        )
        self.num_fpn_layers: int = len(output_indices_for_fpn)
        self.output_indices_for_fpn: List[int] = output_indices_for_fpn
        self.intermediate_channel_sizes: List[int] = intermediate_channel_sizes
        self.intermediate_resolutions: Optional[List[int]] = intermediate_resolutions
        self.fpn_type = fpn_type

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        config._name_or_path = pretrained_model_name_or_path
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
        if config is None:
            config = Dinov2BackBoneWithFPNConfig.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )
        model = cls(config, *model_args)
        safe_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        if os.path.isfile(safe_path):
            state_dict = safe_load(safe_path)
            model.load_state_dict(state_dict, strict=False)
        return model

    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        safe_path = os.path.join(save_directory, "model.safetensors")
        safe_save(self.state_dict(), safe_path)

    def __init__(self, config):
        super().__init__(config)
        self.intermediate_channel_sizes = config.intermediate_channel_sizes
        self.output_indices_for_fpn = config.output_indices_for_fpn
        if getattr(config, "dinov2_pretrained_backbone_name_or_path", ""):
            self.backbone = Dinov2Model.from_pretrained(
                config.dinov2_pretrained_backbone_name_or_path
            )
            freeze_dinov2_weights: bool = True
            print(
                f"[INFO]: DINOv2 parameters loaded from pretrained path: {config.dinov2_pretrained_backbone_name_or_path}"
            )
        else:
            print(
                "[WARN]: No path was provided in the config to load DINOv2 parameters. This backbone has to be trained!"
            )
            self.backbone = Dinov2Model(config)
            freeze_dinov2_weights: bool = False

        if freeze_dinov2_weights:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

        fpn_args = {
            "input_dim": config.hidden_size,
            "out_dims": config.intermediate_channel_sizes,
            "resolutions": getattr(config, "intermediate_resolutions", None),
        }
        fpn_type = getattr(config, "fpn_type", "fused")
        if fpn_type == "fused":
            self.fpn = FusedFPN(**fpn_args)
        elif fpn_type == "tiny":
            self.fpn = TinyFPN(**fpn_args)
        elif fpn_type == "simple":
            self.fpn = SimpleFPN(**fpn_args)
        elif fpn_type == "sfp":
            self.fpn = SFP(**fpn_args)
        else:
            raise ValueError(f"Unsupported FPN type: {fpn_type}")
        self.post_init()

    def forward(self, pixel_values, pixel_mask=None):
        backbone_outputs = self.backbone(pixel_values, output_hidden_states=True)
        feature_maps = [
            backbone_outputs.hidden_states[i] for i in self.output_indices_for_fpn
        ]

        orig_B, orig_C, orig_H, orig_W = pixel_values.shape
        H = orig_H // 14
        W = orig_W // 14

        processed_feats = []
        for features in feature_maps:
            B, N, C = features.shape
            processed_feats.append(
                features[:, 1:, :].transpose(1, 2).reshape(B, C, H, W)
            )

        target_sizes = [
            (orig_H // 4, orig_W // 4),
            (orig_H // 8, orig_W // 8),
            (orig_H // 16, orig_W // 16),
            (orig_H // 32, orig_W // 32),
        ]

        multi_scale_feats = self.fpn(
            processed_feats, target_sizes=target_sizes[: len(processed_feats)]
        )

        if pixel_mask is not None:
            out = []
            for feature_map in multi_scale_feats:
                mask = nn.functional.interpolate(
                    pixel_mask[None].float(), size=feature_map.shape[-2:]
                ).to(torch.bool)[0]
                out.append((feature_map, mask))
            return out

        return BackboneOutput(feature_maps=tuple(multi_scale_feats))
