def apply_peft(model, model_type):
    """
    Freezes the backbone of the model for Parameter-Efficient Fine-Tuning.
    """
    print(f"Applying PEFT to {model_type} model. Freezing backbone...")
    
    if "yolo" in model_type.lower():
        # For YOLO models (Ultralytics), typically the backbone is layers 0-9 or 0-14.
        # We will freeze all parameters in the 'model.model' except the detection head (typically the last layer, e.g., 'model.model.model[-1]').
        for name, param in model.named_parameters():
            if 'model.model.model' in name:
                # Find the layer index
                try:
                    parts = name.split('.')
                    layer_idx = int(parts[3])
                    # Assuming the last layer is the head (usually layer 24 or similar depending on variant)
                    # A safer generic way for Ultralytics YOLO: freeze everything that isn't the Detect/Segment head
                    if "Detect" not in str(type(model.model.model[layer_idx])) and "Segment" not in str(type(model.model.model[layer_idx])):
                        param.requires_grad = False
                except Exception:
                    pass
    
    elif "rf_detr" in model_type.lower():
        # For RF-DETR, freeze the backbone (DinoV2)
        if hasattr(model, 'backbone'):
            for param in model.backbone.parameters():
                param.requires_grad = False
        # Freeze encoder if present
        if hasattr(model, 'encoder'):
            for param in model.encoder.parameters():
                param.requires_grad = False
                
    # Print trainable parameters to verify
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    
    return model
