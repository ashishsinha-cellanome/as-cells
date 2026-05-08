import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class SpatialPriorModule(nn.Module):
    """CNN stem to extract multi-scale spatial features from the input image."""

    def __init__(self, in_channels=3, adapter_dim=384, spm_dim=32):
        super().__init__()
        # Extract Stride 4 (Initial spatial resolution)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, spm_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(spm_dim),
            nn.GELU(),
            nn.Conv2d(spm_dim, spm_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(spm_dim),
            nn.GELU(),
            nn.Conv2d(spm_dim, spm_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(spm_dim),
            nn.GELU(),
        )

        # Stride 8 downsample
        self.down1 = nn.Sequential(
            nn.Conv2d(spm_dim, spm_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(spm_dim * 2),
        )

        # Stride 16 downsample
        self.down2 = nn.Sequential(
            nn.Conv2d(spm_dim * 2, spm_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(spm_dim * 4),
        )

        self.proj_f4 = nn.Conv2d(spm_dim, adapter_dim // 2, kernel_size=1)
        self.proj_f8 = nn.Conv2d(spm_dim * 2, adapter_dim, kernel_size=1)
        self.proj_f16 = nn.Conv2d(spm_dim * 4, adapter_dim, kernel_size=1)

    def forward(self, x):
        # x is B x 3 x H x W
        f4 = self.stem(x)  # B x 64 x (H/4) x (W/4)
        f8 = self.down1(f4)  # B x 128 x (H/8) x (W/8)
        f16 = self.down2(f8)  # B x 256 x (H/16) x (W/16)
        return [self.proj_f4(f4), self.proj_f8(f8), self.proj_f16(f16)]


class Injector(nn.Module):
    """Injects spatial CNN features into ViT tokens via cross-attention."""

    def __init__(
        self,
        token_dim: int,
        spatial_dim: int,
        num_heads: int = 4,
        token_mlp_ratio: float = 0.5,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = spatial_dim // num_heads
        if spatial_dim % num_heads != 0:
            raise ValueError(
                f"spatial_dim={spatial_dim} must be divisible by num_heads={num_heads}"
            )
        token_mlp_hidden = max(spatial_dim, int(token_dim * token_mlp_ratio))

        self.query = nn.Linear(token_dim, spatial_dim)
        self.key = nn.Linear(spatial_dim, spatial_dim)
        self.value = nn.Linear(spatial_dim, spatial_dim)
        self.out_proj = nn.Linear(spatial_dim, token_dim)
        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)

        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_mlp_hidden),
            nn.GELU(),
            nn.Linear(token_mlp_hidden, token_dim),
        )

        # Projection for multiscale spatial features
        self.c_proj = nn.Conv2d(spatial_dim, spatial_dim, kernel_size=1)

    def forward(self, query_tokens, spatial_features):
        # query_tokens: B x N x C
        # spatial_features: B x C x H x W (usually stride 16 flattened)
        B, _, H, W = spatial_features.shape
        spatial_flat = (
            self.c_proj(spatial_features).flatten(2).transpose(1, 2)
        )  # B x (H*W) x spatial_dim

        # Cross attention
        q = self.query(self.norm1(query_tokens))
        k = self.key(spatial_flat)
        v = self.value(spatial_flat)

        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(
            B, -1, self.num_heads * self.head_dim
        )
        attn_out = self.out_proj(attn_out)

        out = query_tokens + attn_out
        out = out + self.ffn(self.norm2(out))
        return out


class Extractor(nn.Module):
    """Extracts semantic ViT tokens back to the spatial CNN pathway."""

    def __init__(
        self,
        token_dim: int,
        spatial_dim: int,
        num_heads: int = 4,
        spatial_mlp_ratio: float = 0.5,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = spatial_dim // num_heads
        if spatial_dim % num_heads != 0:
            raise ValueError(
                f"spatial_dim={spatial_dim} must be divisible by num_heads={num_heads}"
            )
        spatial_mlp_hidden = max(spatial_dim, int(spatial_dim * spatial_mlp_ratio))

        self.query = nn.Linear(spatial_dim, spatial_dim)
        self.key = nn.Linear(token_dim, spatial_dim)
        self.value = nn.Linear(token_dim, spatial_dim)
        self.norm1 = nn.LayerNorm(spatial_dim)
        self.norm2 = nn.LayerNorm(token_dim)
        self.norm3 = nn.LayerNorm(spatial_dim)

        self.ffn = nn.Sequential(
            nn.Linear(spatial_dim, spatial_mlp_hidden),
            nn.GELU(),
            nn.Linear(spatial_mlp_hidden, spatial_dim),
        )

    def forward(self, spatial_features, vit_tokens):
        # spatial_features: B x C x H x W
        B, C, H, W = spatial_features.shape
        spatial_flat = spatial_features.flatten(2).transpose(
            1, 2
        )  # B x N x spatial_dim

        q = self.query(self.norm1(spatial_flat))
        normed_vit = self.norm2(vit_tokens)
        k = self.key(normed_vit)
        v = self.value(normed_vit)

        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, -1, C)

        out_flat = spatial_flat + attn_out
        out_flat = out_flat + self.ffn(self.norm3(out_flat))

        # Reshape back to spatial
        return out_flat.transpose(1, 2).reshape(B, C, H, W)


from transformers import PreTrainedModel, Dinov2Model
from models.dinov2_backbone_with_fpn import Dinov2BackBoneWithFPNConfig


class Dinov2AdapterConfig(Dinov2BackBoneWithFPNConfig):
    model_type = "dinov2_adapter"

    def __init__(
        self,
        interaction_indices=[3, 8, 12],
        gradient_checkpointing: bool = False,
        adapter_dim: int = 384,
        spm_dim: int = 32,
        interaction_num_heads: int = 4,
        token_mlp_ratio: float = 0.5,
        spatial_mlp_ratio: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.interaction_indices = interaction_indices
        self.gradient_checkpointing = gradient_checkpointing
        self.adapter_dim = adapter_dim
        self.spm_dim = spm_dim
        self.interaction_num_heads = interaction_num_heads
        self.token_mlp_ratio = token_mlp_ratio
        self.spatial_mlp_ratio = spatial_mlp_ratio


class Dinov2Adapter(PreTrainedModel):
    config_class = Dinov2AdapterConfig

    def __init__(self, config):
        super().__init__(config)
        self.intermediate_channel_sizes = config.intermediate_channel_sizes
        self.output_indices_for_fpn = config.output_indices_for_fpn
        self.interaction_indices = config.interaction_indices
        self.gradient_checkpointing = bool(
            getattr(config, "gradient_checkpointing", False)
        )
        self.adapter_dim = int(getattr(config, "adapter_dim", config.hidden_size))

        # ViT Backbone (Frozen)
        if config.dinov2_pretrained_backbone_name_or_path:
            self.backbone = Dinov2Model.from_pretrained(
                config.dinov2_pretrained_backbone_name_or_path
            )
        else:
            self.backbone = Dinov2Model(config)

        if self.gradient_checkpointing and hasattr(
            self.backbone, "gradient_checkpointing_enable"
        ):
            self.backbone.gradient_checkpointing_enable()

        for param in self.backbone.parameters():
            param.requires_grad_(False)

        token_dim = config.hidden_size
        adapter_dim = self.adapter_dim

        # Spatial Pathway
        self.spm = SpatialPriorModule(
            in_channels=3,
            adapter_dim=adapter_dim,
            spm_dim=int(getattr(config, "spm_dim", 32)),
        )

        # Interaction layers
        self.injectors = nn.ModuleList(
            [
                Injector(
                    token_dim=token_dim,
                    spatial_dim=adapter_dim,
                    num_heads=int(getattr(config, "interaction_num_heads", 6)),
                    token_mlp_ratio=float(getattr(config, "token_mlp_ratio", 0.5)),
                )
                for _ in self.interaction_indices
            ]
        )
        self.extractors = nn.ModuleList(
            [
                Extractor(
                    token_dim=token_dim,
                    spatial_dim=adapter_dim,
                    num_heads=int(getattr(config, "interaction_num_heads", 6)),
                    spatial_mlp_ratio=float(getattr(config, "spatial_mlp_ratio", 0.5)),
                )
                for _ in self.interaction_indices
            ]
        )

        # Final Multiscale Projections to match Mask2Former expected dimensions
        # Usually: 128 (stride 4), 256 (stride 8), 512 (stride 16), 1024 (stride 32)
        out_dims = config.intermediate_channel_sizes

        self.proj_4 = nn.Sequential(
            nn.Conv2d(adapter_dim // 2, out_dims[0], kernel_size=1),
            nn.BatchNorm2d(out_dims[0]),
        )
        self.proj_8 = nn.Sequential(
            nn.Conv2d(adapter_dim, out_dims[1], kernel_size=1),
            nn.BatchNorm2d(out_dims[1]),
        )
        self.proj_16 = nn.Sequential(
            nn.Conv2d(adapter_dim, out_dims[2], kernel_size=1),
            nn.BatchNorm2d(out_dims[2]),
        )
        # Stride 32 is created by pooling Stride 16
        self.proj_32 = nn.Sequential(
            nn.Conv2d(
                adapter_dim,
                out_dims[3] if len(out_dims) > 3 else adapter_dim,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(out_dims[3] if len(out_dims) > 3 else adapter_dim),
        )

        self.post_init()

    def _run_backbone_layer(
        self, layer_module: nn.Module, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if not (self.training and self.gradient_checkpointing):
            out = layer_module(hidden_states)
            if isinstance(out, tuple):
                return out[0]
            return out

        if not hidden_states.requires_grad:
            hidden_states = hidden_states.requires_grad_(True)

        def custom_forward(inputs: torch.Tensor) -> torch.Tensor:
            out = layer_module(inputs)
            if isinstance(out, tuple):
                return out[0]
            return out

        return checkpoint(custom_forward, hidden_states, use_reentrant=False)

    def forward(self, pixel_values, pixel_mask=None):
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

            hidden_states = self._run_backbone_layer(layer_module, hidden_states)

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

        if pixel_mask is not None:
            out = []
            for feature_map in feature_maps:
                mask = nn.functional.interpolate(
                    pixel_mask[None].float(), size=feature_map.shape[-2:]
                ).to(torch.bool)[0]
                out.append((feature_map, mask))
            return out

        from transformers.modeling_outputs import BackboneOutput

        return BackboneOutput(feature_maps=tuple(feature_maps))
