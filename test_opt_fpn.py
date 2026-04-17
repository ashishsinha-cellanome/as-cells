from models.mask2former_lightning_module import Mask2FormerLightningModule
from models.custom_mask2former_with_dinov2_backbone import (
    Mask2FormerConfigWithCustomBackBone,
    Mask2FormerSegmentationWithCustomBackbone,
)
from models.dinov2_backbone_with_fpn import Dinov2BackBoneWithFPNConfig

config = Mask2FormerConfigWithCustomBackBone()
config.backbone_config = Dinov2BackBoneWithFPNConfig()
model = Mask2FormerSegmentationWithCustomBackbone(config)


class MockConfig:
    class optimizer:
        class optimizer:
            lr = 0.0001
            weight_decay = 0.0001

    class scheduler:
        type = "cosine"
        eta_min = 0.00001

    class model:
        class mask2former:
            threshold = 0.5
            mask_threshold = 0.5
            overlap_mask_area_threshold = 0.5

        max_detections = 100


class MockTrainer:
    estimated_stepping_batches = 100


pl_module = Mask2FormerLightningModule(
    model=model, image_processor=None, config=MockConfig()
)
pl_module.trainer = MockTrainer()

opt_dict = pl_module.configure_optimizers()
opt = opt_dict["optimizer"]
for i, group in enumerate(opt.param_groups):
    print(f"Group {i}: {len(group['params'])} params")
