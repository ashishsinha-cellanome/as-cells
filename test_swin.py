from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

config = Mask2FormerConfig()
model = Mask2FormerForUniversalSegmentation(config)
encoder = model.model.pixel_level_module.encoder

print("Hasattr backbone?", hasattr(encoder, "backbone"))
for name, _ in encoder.named_parameters():
    print(name)
    break
