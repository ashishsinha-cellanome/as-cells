import torch
import torch.nn as nn


class SpatialPriorModule(nn.Module):
    """CNN stem to extract multi-scale spatial features from the input image."""

    def __init__(self, in_channels=3, embed_dim=768):
        super().__init__()
        # Extract Stride 4 (Initial spatial resolution)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(
                embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(
                embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
        )

        # Stride 8 downsample
        self.down1 = nn.Sequential(
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
        )

        # Stride 16 downsample
        self.down2 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
        )

    def forward(self, x):
        # x is B x 3 x H x W
        f4 = self.stem(x)  # B x (C/2) x (H/4) x (W/4)
        f8 = self.down1(f4)  # B x C x (H/8) x (W/8)
        f16 = self.down2(f8)  # B x C x (H/16) x (W/16)
        return [f4, f8, f16]


class Injector(nn.Module):
    """Injects spatial CNN features into ViT tokens via cross-attention."""

    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        # Projection for multiscale spatial features
        self.c_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, query_tokens, spatial_features):
        # query_tokens: B x N x C
        # spatial_features: B x C x H x W (usually stride 16 flattened)
        B, C, H, W = spatial_features.shape
        spatial_flat = (
            self.c_proj(spatial_features).flatten(2).transpose(1, 2)
        )  # B x (H*W) x C

        # Cross attention
        q = self.query(self.norm1(query_tokens))
        k = self.key(spatial_flat)
        v = self.value(spatial_flat)

        attn = (q @ k.transpose(-2, -1)) * (C**-0.5)
        attn = attn.softmax(dim=-1)

        out = query_tokens + (attn @ v)
        out = out + self.ffn(self.norm2(out))
        return out


class Extractor(nn.Module):
    """Extracts semantic ViT tokens back to the spatial CNN pathway."""

    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, spatial_features, vit_tokens):
        # spatial_features: B x C x H x W
        B, C, H, W = spatial_features.shape
        spatial_flat = spatial_features.flatten(2).transpose(1, 2)  # B x N x C

        q = self.query(self.norm1(spatial_flat))
        normed_vit = self.norm2(vit_tokens)
        k = self.key(normed_vit)
        v = self.value(normed_vit)

        attn = (q @ k.transpose(-2, -1)) * (C**-0.5)
        attn = attn.softmax(dim=-1)

        out_flat = spatial_flat + (attn @ v)
        out_flat = out_flat + self.ffn(self.norm3(out_flat))

        # Reshape back to spatial
        return out_flat.transpose(1, 2).reshape(B, C, H, W)


from transformers import PreTrainedModel, Dinov2Model
from models.dinov2_backbone_with_fpn import Dinov2BackBoneWithFPNConfig


class Dinov2AdapterConfig(Dinov2BackBoneWithFPNConfig):
    model_type = "dinov2_adapter"

    def __init__(self, interaction_indices=[2, 5, 8, 11], **kwargs):
        super().__init__(**kwargs)
        self.interaction_indices = interaction_indices


class Dinov2Adapter(PreTrainedModel):
    config_class = Dinov2AdapterConfig

    def __init__(self, config):
        super().__init__(config)
        self.intermediate_channel_sizes = config.intermediate_channel_sizes
        self.output_indices_for_fpn = config.output_indices_for_fpn
        self.interaction_indices = config.interaction_indices

        # ViT Backbone (Frozen)
        if config.dinov2_pretrained_backbone_name_or_path:
            self.backbone = Dinov2Model.from_pretrained(
                config.dinov2_pretrained_backbone_name_or_path
            )
        else:
            self.backbone = Dinov2Model(config)

        for param in self.backbone.parameters():
            param.requires_grad_(False)

        embed_dim = config.hidden_size

        # Spatial Pathway
        self.spm = SpatialPriorModule(in_channels=3, embed_dim=embed_dim)

        # Interaction layers
        self.injectors = nn.ModuleList(
            [Injector(embed_dim) for _ in self.interaction_indices]
        )
        self.extractors = nn.ModuleList(
            [Extractor(embed_dim) for _ in self.interaction_indices]
        )

        # Final Multiscale Projections to match Mask2Former expected dimensions
        # Usually: 128 (stride 4), 256 (stride 8), 512 (stride 16), 1024 (stride 32)
        out_dims = config.intermediate_channel_sizes

        self.proj_4 = nn.Sequential(
            nn.Conv2d(embed_dim // 2, out_dims[0], kernel_size=1),
            nn.BatchNorm2d(out_dims[0]),
        )
        self.proj_8 = nn.Sequential(
            nn.Conv2d(embed_dim, out_dims[1], kernel_size=1),
            nn.BatchNorm2d(out_dims[1]),
        )
        self.proj_16 = nn.Sequential(
            nn.Conv2d(embed_dim, out_dims[2], kernel_size=1),
            nn.BatchNorm2d(out_dims[2]),
        )
        # Stride 32 is created by pooling Stride 16
        self.proj_32 = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_dims[3] if len(out_dims) > 3 else embed_dim,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(out_dims[3] if len(out_dims) > 3 else embed_dim),
        )

        self.post_init()

    def forward(self, pixel_values, pixel_mask=None):
        B, C, H, W = pixel_values.shape

        # Extract initial spatial features
        spm_features = self.spm(pixel_values)
        f4, f8, f16 = spm_features

        # Initial ViT Embeddings
        embeddings = self.backbone.embeddings(pixel_values)

        # We need to process block by block to interact
        hidden_states = embeddings
        interaction_idx = 0

        for i, layer_module in enumerate(self.backbone.encoder.layer):
            if i in self.interaction_indices:
                # Inject Spatial info (stride 16) into ViT tokens
                hidden_states = self.injectors[interaction_idx](hidden_states, f16)

            hidden_states = layer_module(hidden_states)[0]

            if i in self.interaction_indices:
                # Extract ViT semantics back to spatial features
                # Skip CLS token when extracting to spatial
                f16 = self.extractors[interaction_idx](f16, hidden_states[:, 1:, :])
                interaction_idx += 1

        # Final output formatting
        out_4 = self.proj_4(f4)
        out_8 = self.proj_8(f8)
        out_16 = self.proj_16(f16)
        out_32 = self.proj_32(f16)

        feature_maps = [out_4, out_8, out_16, out_32]

        out = []
        for feature_map in feature_maps:
            if pixel_mask is not None:
                mask = nn.functional.interpolate(
                    pixel_mask[None].float(), size=feature_map.shape[-2:]
                ).to(torch.bool)[0]
                out.append((feature_map, mask))
            else:
                out.append((feature_map,))

        return out
