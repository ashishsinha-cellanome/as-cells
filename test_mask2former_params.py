from models.custom_mask2former_with_dinov2_backbone import (
    Mask2FormerConfigWithCustomBackBone,
    Mask2FormerSegmentationWithCustomBackbone,
)
from models.dinov2_adapter import Dinov2AdapterConfig

try:
    config = Mask2FormerConfigWithCustomBackBone()
    config.backbone_config = Dinov2AdapterConfig()
    model = Mask2FormerSegmentationWithCustomBackbone(config)

    print("--- Trainable parameters ---")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
except Exception:
    import traceback

    traceback.print_exc()
