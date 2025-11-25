# AI Agent Instructions for Cellanome ML Codebase

This document guides AI coding agents in effectively working with this codebase. Focus on project-specific patterns and critical workflows.

## Project Overview

This is a computer vision project for cell detection and analysis with two main components:

1. **Object Detection & Instance Segmentation**
   - YOLOv5 for object detection
   - Mask R-CNN for instance segmentation
   - Multiple model variants: RT-DETR, Deformable DETR, Point Rend, etc.

2. **Cell Analysis & Classification**
   - Morphological analysis
   - Cell classification using DINOv2 embeddings
   - Support for multiple cell types and conditions (caged/uncaged)

## Key Architecture Components

### Data Pipeline
```
Raw Images (12/14-bit) → 8-bit JPEG → Darwin V7 Annotations → AWS S3 → W&B Versioning
```

### Model Training Infrastructure
- Training configuration in `data_config.yaml`
- Model-specific label formats handled in preprocessing
- Integration with Weights & Biases for experiment tracking
- Multi-backend support (TensorRT, OpenVINO, ONNX)

## Critical Developer Workflows

### Setting Up New Training Run

1. Data Registration:
```bash
python prepare-dataset/01-data-registration.py \
  --pull_darwin_dataset \
  --split_dataset \
  --push_data_to_aws \
  --register_data_to_wandb
```

2. Training Configuration:
- Update `config.py` with model parameters
- Check `DEFAULT_CLASS_NAMES_TO_IDS_MAP` in `json_parser.py`
- Set optical characteristics in `OPTICAL_CHARACTERISTICS`

### Evaluating Models

Use COCO evaluation metrics:
- Refer to `COCO Evaluation/` for precision-recall analysis
- Use `evaluate()` in `detection/engine.py` for standard metrics
- Custom evaluators available in `custom_coco_eval.py`

## Project-Specific Conventions

### Dataset Structure
```
datasets/
  <date>_<cell-type>_<magnification>_<condition>/
    annotations/
    test_images/
    test_annotations/
```

### Key Parameters

- Image sizes are magnification-dependent:
  - 10x: 2000x1600 pixels
  - 4x: 4512x4512 pixels
- Pixel sizes and optical characteristics defined in `json_parser.py`
- Cell diameter thresholds for filtering small objects

## Common Debugging Points

1. Object Size Filtering:
- Check `min_object_diameter` in annotation parsing
- Validate optical characteristics match image resolution

2. Training Issues:
- Verify W&B artifact paths and versioning
- Check model backend compatibility (CUDA/CPU)
- Monitor output shapes in multi-backend detection

## Integration Points

1. **External Services**
- Darwin V7 for annotations
- AWS S3 for data storage
- Weights & Biases for experiment tracking

2. **Model Deployment**
- Flask REST API available in `utils/flask_rest_api/`
- Multiple inference backends supported
- TensorRT optimization options

Remember to maintain consistency with optical characteristics and class mappings when modifying the pipeline.
