import torch
import torch.nn as nn

class Dinov2WithSFP(nn.Module):
    def __init__(self, original_encoder):
        super().__init__()
        self.original_encoder = original_encoder
        self.channels = original_encoder.channels
        
        c = self.channels[0]
        
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(c, c, kernel_size=2, stride=2),
            nn.BatchNorm2d(c),
            nn.GELU(),
            nn.ConvTranspose2d(c, c, kernel_size=2, stride=2),
        )
        self.fpn2 = nn.ConvTranspose2d(c, c, kernel_size=2, stride=2)
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, pixel_values):
        outputs = self.original_encoder(pixel_values)
        feats = outputs.feature_maps
        
        f1 = self.fpn1(feats[0])
        f2 = self.fpn2(feats[1])
        f3 = self.fpn3(feats[2])
        f4 = self.fpn4(feats[3])
        
        outputs.feature_maps = (f1, f2, f3, f4)
        return outputs

from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, Dinov2Config
config = Mask2FormerConfig.from_pretrained('facebook/mask2former-swin-large-coco-instance')
config.backbone_config = Dinov2Config.from_pretrained('facebook/dinov2-base', out_indices=[6,8,10,12])
model = Mask2FormerForUniversalSegmentation(config)

model.model.pixel_level_module.encoder = Dinov2WithSFP(model.model.pixel_level_module.encoder)

dummy = torch.randn(1, 3, 640, 640)
out = model(dummy)
print("Forward pass successful!")
for i, f in enumerate(model.model.pixel_level_module.encoder(dummy).feature_maps):
    print(f"Level {i}: {f.shape}")
