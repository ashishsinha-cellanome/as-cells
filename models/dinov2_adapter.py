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
        k = self.key(self.norm2(vit_tokens))
        v = self.value(self.norm2(vit_tokens))

        attn = (q @ k.transpose(-2, -1)) * (C**-0.5)
        attn = attn.softmax(dim=-1)

        out_flat = spatial_flat + (attn @ v)
        out_flat = out_flat + self.ffn(self.norm1(out_flat))

        # Reshape back to spatial
        return out_flat.transpose(1, 2).reshape(B, C, H, W)
