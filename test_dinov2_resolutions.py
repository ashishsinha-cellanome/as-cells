import torch
import hydra
from omegaconf import DictConfig

# Importing your exact implementation from the models directory
from models.mask2former_model import (
    get_mask2former_instance_segmentation_model_with_dinov2_backbone,
)


@hydra.main(version_base=None, config_path="configs", config_name="mask2former_test")
def test_dinov2_feature_resolutions(cfg: DictConfig):
    print("\n" + "=" * 80)
    print("1. Initializing Mask2Former with Dinov2 Backbone using Hydra Config...")
    print("=" * 80)

    # Load parameters dynamically from Hydra Config
    model_type = cfg.model.mask2former.model_type
    with_registers = cfg.model.mask2former.with_registers

    # Dummy id2label mapping (needed for initialization)
    id2label = {0: "cell", 1: "nucleus"}

    # Instantiate the model utilizing your exact codebase function
    model = get_mask2former_instance_segmentation_model_with_dinov2_backbone(
        id2label=id2label, model_type=model_type, with_registers=with_registers
    )

    # Extract the Mask2Former Pixel Level Module Encoder (which is your Dinov2 backbone)
    encoder = model.model.pixel_level_module.encoder
    encoder.eval()
    # breakpoint()
    # Create a dummy image tensor based on Hydra config sizes
    h = cfg.model.mask2former.input_h
    w = cfg.model.mask2former.input_w
    h, w = 672, 672  # Override to test the specific resolution of interest
    dummy_input = torch.randn(1, 3, h, w)

    print(
        f"\n2. Passing Dummy Input Image of shape: {dummy_input.shape} (H={h}, W={w})"
    )
    print("-" * 80)

    with torch.no_grad():
        # Pass the dummy input through the Hugging Face AutoBackbone
        # (which wraps Dinov2Model for Mask2Former)
        encoder_outputs = encoder(dummy_input)

    features = encoder_outputs.feature_maps

    print(
        "3. Spatial Resolutions of Dinov2 Feature Maps passed to Mask2Former Decoder:\n"
    )

    for idx, f_map in enumerate(features):
        b, c, fh, fw = f_map.shape
        stride_h = h // fh
        stride_w = w // fw
        print(f"Feature Level {idx}:")
        print(f"  -> Shape: {f_map.shape}")
        print(
            f"  -> Calculated Stride: {stride_h}x{stride_w} (Resolution: 1/{stride_h})\n"
        )

    print("-" * 80)
    print("ANALYSIS OF THE ARCHITECTURAL ISSUE:")
    print(
        "Standard Mask2Former expects 4 feature maps with varying resolutions, typically:"
    )
    print("  - Level 0: Stride 4  (1/4 resolution for small objects)")
    print("  - Level 1: Stride 8  (1/8 resolution)")
    print("  - Level 2: Stride 16 (1/16 resolution)")
    print("  - Level 3: Stride 32 (1/32 resolution for large context)\n")
    print("Because Dinov2 is a standard Vision Transformer with a patch size of 14,")
    print(
        "every single feature map has exactly the same spatial resolution (Stride 14)."
    )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    test_dinov2_feature_resolutions()
