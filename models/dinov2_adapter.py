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
