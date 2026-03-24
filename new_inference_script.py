import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import shutil
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import time
import logging
import cv2

from models.rt_detr_lightning_module import (
    RTDETRLightningModule,
    to_cpu_device,
    convert_to_xywh,
)  # Re-import necessary helpers
from data.coco_data_module import COCODataModule
from transformers import RTDetrImageProcessor
from pycocotools.cocoeval import COCOeval

# Import VisionModel and related utilities for cropping
from models.AbstractVisionModel import (
    VisionModel,
    run_model,
    post_process_detections,
    DEFAULT_DETECTION_CONFIDENCE,
)
from utils.ema import ModelEma
from utils.json_parser import CellMaskDataset  # For creating a mock dataset for metrics
from utils.precision_recall_eval import AnnotationFilter, evaluate_pr_per_image
from typing import List, Dict, Union, Tuple, Optional, Final, Any

# Importing default crop corners and resize from models/rt_detr_model.py
from models.rt_detr_model import (
    DEFAULT_RESIZE,
    DEFAULT_CROP_CORNERS,
    DINOV2_RESIZE,
    DINOV2_CROP_CORNERS,
)

# Set up a basic logger for this script
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Helper functions (kept from original inference.py, potentially refined) ---
# (to_cpu and convert_to_xywh are already imported from rt_detr_lightning_module)


class LightningModelAdapter(VisionModel):
    """
    Adapts a PyTorch Lightning RTDETRLightningModule to the VisionModel interface
    to enable using run_model and detect_by_cropping utilities.
    """

    def __init__(
        self,
        pl_module: RTDETRLightningModule,
        image_processor: RTDetrImageProcessor,
        config: DictConfig,
        use_ema: bool = False,
        # VisionModel __init__ expects weights_path and model_name,
        # but here the model is already loaded. We pass dummy values.
        weights_path: str = "dummy_path.ckpt",
        model_name: str = "RT-DETR-Lightning-Adapter",
    ):
        # Initialize the base VisionModel with dummy params first,
        # then override internal states with actual PL module info.
        super().__init__(
            weights_path=weights_path,
            model_name=model_name,
            label_map={
                int(k): v for k, v in config.model.label_map.items()
            },  # Ensure label_map is Dict[int, str]
            confidence=config.model.detection_threshold,
            device=pl_module.device,
        )

        self.pl_module = pl_module
        self.image_processor = image_processor
        self.config = config
        self.use_ema = use_ema

        # Ensure model is in eval mode and on correct device
        self.pl_module.eval()
        self.pl_module.to(self.pl_module.device)

        # Load EMA weights if requested
        if self.use_ema:
            if hasattr(self.pl_module, "ema_model") and self.pl_module.ema_model:
                self.model_to_infer = self.pl_module.ema_model.module
                logger.info("Using EMA model for inference.")
            else:
                logger.warning(
                    "EMA model requested but not found in checkpoint or module. Using regular model."
                )
                self.model_to_infer = self.pl_module.model
        else:
            self.model_to_infer = self.pl_module.model

        # Override metadata and other properties from the Lightning module's config
        self._model_name = f"RT-DETR-Lightning ({'EMA' if use_ema else 'Regular'})"
        self._model_input_size = (
            self.config.data.model_input_size,
            self.config.data.model_input_size,
        )
        self._confidence = self.config.model.detection_threshold

        # Populate _metadata, _resize_dict, _crop_corners_dict for run_model compatibility
        # RT-DETR models are typically bbox only, adjust if mask prediction is added
        self._metadata = {
            "predict_masks": False,
            "magnification": "10x",  # Default, can be refined from config if present
            "resolution": self._model_input_size[0],
            "model_type": "Transformer Detector",
        }

        # Select appropriate resize/crop dicts based on model configuration
        # Assuming rtdetr_v2 with dinov2 backbone uses DINOV2_RESIZE/CROP_CORNERS
        # Otherwise, use DEFAULT_RESIZE/CROP_CORNERS for other RT-DETR models
        if "dinov2" in self.config.model.backbone.name.lower():
            self._resize_dict = DINOV2_RESIZE
            self._crop_corners_dict = DINOV2_CROP_CORNERS
        else:
            self._resize_dict = DEFAULT_RESIZE
            self._crop_corners_dict = DEFAULT_CROP_CORNERS

        self._loaded = True  # Mark as loaded since pl_module is already ready

    def load(self) -> None:
        """Model is already loaded during initialization of the adapter."""
        if not self._loaded:
            logger.error(f"{self._model_name} adapter not properly initialized.")
        pass

    @torch.no_grad()
    def detect_batch(
        self,
        input_images_list: List[Union[Image.Image, np.ndarray]],
    ) -> List[Dict[str, list]]:
        """
        Detect objects in a batch of images using the wrapped RTDETRLightningModule.
        This method replaces the abstract detect_batch from VisionModel.
        """
        if not self._loaded:
            logger.error(f"{self._model_name} not loaded for batch detection.")
            return [{"boxes": [], "scores": [], "labels": []}] * len(input_images_list)

        images_to_process = []
        original_sizes = []

        for img_data in input_images_list:
            if isinstance(img_data, Image.Image):
                img_np = np.array(img_data)
            elif isinstance(img_data, np.ndarray):
                img_np = img_data
            else:
                raise TypeError(f"Unsupported image type: {type(img_data)}")

            # Convert to RGB if grayscale, as processor expects 3 channels
            if len(img_np.shape) == 2:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            elif img_np.shape[2] == 4:  # RGBA to RGB
                img_np = img_np[:, :, :3]

            images_to_process.append(Image.fromarray(img_np))
            original_sizes.append(img_np.shape[:2])  # (height, width)

        # Preprocess images using the image_processor
        # The processor usually handles resizing and normalization for the model input
        processed_inputs = self.image_processor(images_to_process, return_tensors="pt")
        pixel_values = processed_inputs.pixel_values.to(self.pl_module.device)

        # Forward pass through the model
        outputs = self.model_to_infer(pixel_values=pixel_values)

        # Post-process raw outputs to get detections in original image coordinates
        # target_sizes needs to be (height, width) tuples or list of lists
        target_sizes_tensor = torch.tensor(original_sizes, device=self.pl_module.device)
        post_processed_outputs = self.image_processor.post_process_object_detection(
            outputs,
            threshold=self._confidence,
            target_sizes=target_sizes_tensor,  # Should be batch_size x 2 (height, width)
        )

        # Convert predictions to a consistent format (e.g., numpy arrays)
        results = []
        for pred in post_processed_outputs:
            boxes = (
                to_cpu_device(pred["boxes"]).numpy()
                if len(pred["boxes"]) > 0
                else np.zeros((0, 4))
            )
            scores = (
                to_cpu_device(pred["scores"]).numpy()
                if len(pred["scores"]) > 0
                else np.zeros((0,))
            )
            labels = (
                to_cpu_device(pred["labels"]).numpy()
                if len(pred["labels"]) > 0
                else np.zeros((0,))
            )

            # Ensure integer labels
            labels = labels.astype(int)

            results.append(
                {
                    "boxes": boxes,
                    "labels": labels,
                    "scores": scores,
                    # "masks": [] # RT-DETR is a detector, no masks by default
                }
            )

        torch.cuda.empty_cache()  # Clear CUDA cache after inference

        return results

    def postprocess(self, **kwargs: Any) -> List[Dict[str, list]]:
        """
        This method is required by VisionModel but the actual post-processing (NMS, scaling)
        is handled by HuggingFace's post_process_object_detection during detect_batch
        and by run_model's post_process_detections.
        This can be a no-op or raise an error if called unexpectedly.
        """
        raise NotImplementedError(
            "LightningModelAdapter handles post-processing within detect_batch or delegates to run_model."
        )

    def get_label_map(self):
        return self._label_map

    def get_reverse_label_map(self):
        return self._reverse_label_map

    def get_class_names(self):
        return list(self._reverse_label_map.keys())

    def get_cropping_info(self):
        return self._resize_dict, self._crop_corners_dict

    def get_model_name(self):
        return self._model_name

    def get_metadata(self):
        return self._metadata


@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(config: DictConfig):
    # Unlock config to make changes
    OmegaConf.set_struct(config, False)

    # --- 1. Configuration Setup ---
    logger.info("Setting up inference configuration...")

    # Add inference-specific configurations if they don't exist
    if not hasattr(config, "inference"):
        config.inference = OmegaConf.create()

    # Create a DictConfig with desired defaults
    inference_defaults = OmegaConf.create(
        {
            "input_type": "dataset",  # Options: "image", "folder", "dataset"
            "input_path": None,  # Path to image file, folder, or dataset root
            "output_dir": "inference_results",
            "use_ema": False,
            "perform_metrics": False,  # Set to True if ground truth is available and metrics are desired
            "gt_json_path": None,  # Required if perform_metrics is True and input_type is "image" or "folder"
            "visualize_preds": True,
            "viz_output_dir": "inference_visualizations",
            "viz_max_samples": 20,  # Max images to visualize
            "cropping_enabled": True,
        }
    )

    # Merge the defaults into config.inference, existing values in config.inference will take precedence
    config.inference = OmegaConf.merge(inference_defaults, config.inference)

    # Ensure output directories exist
    output_base_dir = Path(config.inference.output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    viz_output_dir = output_base_dir / config.inference.viz_output_dir
    if config.inference.visualize_preds:
        viz_output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve checkpoint path
    ckpt_path = config.initialization.load_from_checkpoint
    if not ckpt_path:
        raise ValueError(
            "Please provide a checkpoint path via 'initialization.load_from_checkpoint=/path/to/ckpt'"
        )
    ckpt_path = hydra.utils.to_absolute_path(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
    logger.info(f"Loading model from checkpoint: {ckpt_path}")

    # --- 2. Load Model (RTDETRLightningModule) ---
    # Load the full checkpoint including hparams
    checkpoint = torch.load(ckpt_path, map_location="cpu")  # Load to CPU first

    # Reconstruct the config that the model was trained with
    if "hyper_parameters" in checkpoint and "config" in checkpoint["hyper_parameters"]:
        model_config = OmegaConf.create(checkpoint["hyper_parameters"]["config"])
        # Merge current inference config, overriding model_config if there are conflicts
        merged_config = OmegaConf.merge(model_config, config)
        config = merged_config  # Use the merged config
        logger.info("Loaded model's original config and merged with inference config.")
    else:
        logger.warning(
            "Could not find original training config in checkpoint. Using current config for model setup."
        )

    # Instantiate the Lightning module
    pl_model = RTDETRLightningModule.load_from_checkpoint(ckpt_path, config=config)
    pl_model.eval()
    pl_model.to("cuda" if torch.cuda.is_available() else "cpu")  # Move to device

    # If EMA is requested, manually load EMA state dict if available
    if config.inference.use_ema:
        if "ema_state_dict" in checkpoint:
            logger.info(
                "EMA model requested and 'ema_state_dict' found in checkpoint. Loading EMA weights."
            )
            # Create a ModelEma instance and load the state dict
            # We need to initialize ModelEma with a copy of the pl_model.model
            pl_model.ema_model = ModelEma(
                pl_model.model, decay=0.9999
            )  # Decay rate doesn't matter for inference
            pl_model.ema_model.module.load_state_dict(checkpoint["ema_state_dict"])
            pl_model.ema_model.module.eval()
            pl_model.ema_model.module.to(pl_model.device)
            # The LightningModelAdapter will pick up pl_model.ema_model.module
        else:
            logger.warning(
                "EMA model requested but 'ema_state_dict' not found in checkpoint. Falling back to regular model."
            )
            config.inference.use_ema = False  # Disable EMA flag

    # Create the adapter for compatibility with run_model
    adapter_model = LightningModelAdapter(
        pl_module=pl_model,
        image_processor=pl_model.image_processor,
        config=config,
        use_ema=config.inference.use_ema,
    )
    logger.info(
        f"Initialized model: {adapter_model.get_model_name()} on device: {adapter_model._device}"
    )

    # --- 3. Prepare Input Data ---
    images_to_infer = []  # List of (image_data, image_id, original_image_path, ground_truth)
    # We need a unique ID for each image, even if it's not from COCO
    current_image_unique_id = 0

    if config.inference.input_type == "image":
        image_path = Path(hydra.utils.to_absolute_path(config.inference.input_path))
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found at: {image_path}")
        img = Image.open(image_path).convert("RGB")  # Ensure RGB

        coco_gt = None
        if config.inference.perform_metrics and config.inference.gt_json_path:
            from pycocotools.coco import COCO

            gt_json_path = Path(
                hydra.utils.to_absolute_path(config.inference.gt_json_path)
            )
            if gt_json_path.is_file():
                coco_gt = COCO(str(gt_json_path))
                img_filename = image_path.name
                coco_img_ids = coco_gt.getImgIds()
                for coco_img_id in coco_img_ids:
                    img_info = coco_gt.loadImgs(coco_img_id)[0]
                    if img_info["file_name"] == img_filename:
                        images_to_infer.append(
                            (img, coco_img_id, str(image_path), coco_gt)
                        )
                        logger.info(
                            f"Found GT for image {img_filename} with COCO ID {coco_img_id}"
                        )
                        break
                else:
                    logger.warning(
                        f"No COCO GT found for image {img_filename} in {gt_json_path}. Metrics will be skipped for this image."
                    )
                    images_to_infer.append(
                        (img, current_image_unique_id, str(image_path), None)
                    )
                    current_image_unique_id += 1
            else:
                logger.warning(
                    f"GT JSON not found at {gt_json_path}. Metrics will be skipped."
                )
                config.inference.perform_metrics = False
                images_to_infer.append(
                    (img, current_image_unique_id, str(image_path), None)
                )
                current_image_unique_id += 1
        else:
            images_to_infer.append(
                (img, current_image_unique_id, str(image_path), None)
            )  # Mock image_id
            current_image_unique_id += 1
        dataset_name = image_path.parent.name

    elif config.inference.input_type == "folder":
        folder_path = Path(hydra.utils.to_absolute_path(config.inference.input_path))
        if not folder_path.is_dir():
            raise FileNotFoundError(f"Folder not found at: {folder_path}")

        image_files = sorted(
            list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png"))
        )
        if not image_files:
            raise ValueError(f"No image files found in folder: {folder_path}")

        coco_gt = None
        if config.inference.perform_metrics and config.inference.gt_json_path:
            from pycocotools.coco import COCO

            gt_json_path = Path(
                hydra.utils.to_absolute_path(config.inference.gt_json_path)
            )
            if gt_json_path.is_file():
                coco_gt = COCO(str(gt_json_path))
                logger.info(f"Loaded GT JSON from {gt_json_path} for folder input.")
            else:
                logger.warning(
                    f"GT JSON not found at {gt_json_path}. Metrics will be skipped for all images in folder."
                )
                config.inference.perform_metrics = False

        for img_path in image_files:
            img = Image.open(img_path).convert("RGB")
            current_img_coco_gt = None
            if coco_gt:  # If a GT JSON was loaded for the folder
                img_filename = img_path.name
                coco_img_ids = coco_gt.getImgIds()
                for coco_img_id in coco_img_ids:
                    img_info = coco_gt.loadImgs(coco_img_id)[0]
                    if img_info["file_name"] == img_filename:
                        images_to_infer.append(
                            (img, coco_img_id, str(img_path), coco_gt)
                        )
                        current_img_coco_gt = coco_gt  # Assign the loaded coco_gt
                        break
                else:  # No GT found for this specific image
                    images_to_infer.append(
                        (img, current_image_unique_id, str(img_path), None)
                    )
                    current_image_unique_id += 1
            else:  # No GT JSON provided for folder
                images_to_infer.append(
                    (img, current_image_unique_id, str(img_path), None)
                )  # Mock image_id
                current_image_unique_id += 1
        dataset_name = folder_path.name

    elif config.inference.input_type == "dataset":
        data_module = COCODataModule(
            dataset_path=hydra.utils.to_absolute_path(config.data.path),
            processor=pl_model.image_processor,  # Use the model's processor
            batch_size=config.data.batch_size,  # Can use a smaller batch size for inference
            num_workers=config.data.num_workers,
            model_input_size=config.data.model_input_size,
            # For inference, these can be default or minimal
            min_random_scale=config.data.get("min_random_scale", 0.8),
            max_random_scale=config.data.get("max_random_scale", 1.2),
            p_noise=config.data.get("p_noise", 0.0),
            org_images_in_model_input_size=config.data.get(
                "org_images_in_model_input_size", False
            ),
            config=config,  # Pass the full config
        )
        data_module.setup(stage="test")  # Use test split
        test_dataset_coco = data_module.test_dataset.coco  # Original COCO API object

        # Prepare a simple list of images with their original PIL objects and COCO IDs
        for i in tqdm(
            range(len(data_module.test_dataset)), desc="Loading dataset images"
        ):
            coco_img_info = data_module.test_dataset.coco.loadImgs(
                data_module.test_dataset.ids[i]
            )[0]
            img_path = os.path.join(
                data_module.test_dataset.root, coco_img_info["file_name"]
            )
            img = Image.open(img_path).convert("RGB")
            images_to_infer.append(
                (img, data_module.test_dataset.ids[i], img_path, test_dataset_coco)
            )
        dataset_name = Path(config.data.path).name
        logger.info(f"Loaded {len(images_to_infer)} images from dataset.")

        # Ensure coco_gt is set for metrics if input_type is dataset
        coco_gt = test_dataset_coco if config.inference.perform_metrics else None

    else:
        raise ValueError(f"Unsupported input_type: {config.inference.input_type}")

    # --- 4. Perform Inference and Collect Results ---
    all_predictions_coco_format = []
    all_predictions_for_pr_eval = []
    per_image_metrics_data_rows = []
    inference_times_ms = []

    # Map image_id to its original index in images_to_infer to retrieve correct inference time later
    # This needs to be populated AFTER images_to_infer is finalized.
    image_id_to_original_idx = {item[1]: i for i, item in enumerate(images_to_infer)}

    logger.info("Starting inference...")
    for img_idx, (
        img_pil,
        image_id,
        original_path,
        image_coco_gt_for_this_img,
    ) in enumerate(tqdm(images_to_infer, desc="Inferring on images")):
        start_time = time.time()

        # Run inference using the adapter, which can use cropping if enabled
        # run_model expects img as np.ndarray
        img_np = np.array(img_pil)

        # run_model returns (detections, runtime_s)
        detections_raw, runtime_s = run_model(
            detector=adapter_model,
            input_image=img_np,
            input_resize=adapter_model.get_cropping_info()[
                0
            ],  # Pass the resize dict from adapter
            input_crop_corners=adapter_model.get_cropping_info()[
                1
            ],  # Pass the crop corners from adapter
            normalize_image=False,  # Processor handles normalization
            crop=config.inference.cropping_enabled,
            post_process_class_names=adapter_model.get_class_names(),  # Apply post-processing (e.g., remove redundant smaller cells)
        )
        inference_times_ms.append(runtime_s * 1000)

        # Convert output to list of dicts with numpy arrays for consistency
        # detections_raw should already be in this format from LightningModelAdapter.detect_batch

        # Convert raw detections (numpy arrays) for COCO evaluation
        if detections_raw["boxes"].shape[0] > 0:
            for i in range(len(detections_raw["boxes"])):
                box = detections_raw["boxes"][i]
                score = detections_raw["scores"][i]
                label = detections_raw["labels"][i]

                # Convert to xywh for COCO format
                xmin, ymin, xmax, ymax = box
                bbox_xywh = [xmin, ymin, xmax - xmin, ymax - ymin]

                all_predictions_coco_format.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": bbox_xywh,
                        "score": float(score),
                    }
                )

        # Prepare for per-image PR evaluation (needs list of dicts, each dict is one image's preds)
        # Ensure boxes, labels, scores are lists of numpy arrays if they aren't already
        detections_for_pr = {
            "boxes": detections_raw["boxes"],  # Already numpy arrays from run_model
            "labels": detections_raw["labels"],
            "scores": detections_raw["scores"],
        }
        all_predictions_for_pr_eval.append(detections_for_pr)

        # --- Visualize Predictions ---
        if (
            config.inference.visualize_preds
            and img_idx < config.inference.viz_max_samples
        ):
            img_with_preds = img_pil.copy()  # Start with a fresh copy for drawing

            # Draw Ground Truth if available for this specific image
            if image_coco_gt_for_this_img:
                ann_ids = image_coco_gt_for_this_img.getAnnIds(imgIds=image_id)
                anns = image_coco_gt_for_this_img.loadAnns(ann_ids)
                gt_boxes = []
                gt_labels = []
                for ann in anns:
                    x, y, w, h = ann["bbox"]
                    gt_boxes.append([x, y, x + w, y + h])
                    gt_labels.append(ann["category_id"])

                img_with_preds = adapter_model.draw_boxes(
                    img_with_preds,
                    gt_boxes,
                    gt_labels,
                    scores=None,  # No scores for GT
                    id2label=adapter_model.get_label_map(),
                    color_override=(0, 255, 0),  # Green for GT
                    label_prefix="GT: ",
                )

            # Draw Predictions
            if detections_raw["boxes"].shape[0] > 0:
                img_with_preds = adapter_model.draw_boxes(
                    img_with_preds,
                    detections_raw["boxes"],
                    detections_raw["labels"],
                    detections_raw["scores"],
                    id2label=adapter_model.get_label_map(),
                    color_override=(255, 0, 0),  # Red for predictions
                    label_prefix="Pred: ",
                )

            viz_save_path = viz_output_dir / Path(original_path).name
            img_with_preds.save(viz_save_path)
            logger.info(
                f"Saved visualization for {Path(original_path).name} to {viz_save_path}"
            )

    # --- 5. Compute Metrics (if enabled) ---
    metrics_output_dir = (
        output_base_dir
        / dataset_name
        / adapter_model.get_model_name().replace(" ", "_")
    )
    metrics_output_dir.mkdir(parents=True, exist_ok=True)

    if config.inference.perform_metrics and coco_gt:
        logger.info("Computing metrics...")

        # Mock a CellMaskDataset for evaluate_pr_per_image
        class MockCellMaskDataset:
            def __init__(self, coco_gt_obj, images_data_list):
                self.coco = coco_gt_obj
                # Filter images_data_list to only include items with GT (image_coco_gt_for_this_img is not None)
                self.images_data = [
                    item for item in images_data_list if item[3] is not None
                ]
                self.ids = [
                    item[1] for item in self.images_data
                ]  # COCO image IDs that have GT
                self.image_id_to_original_idx = {
                    item[1]: idx
                    for idx, item in enumerate(images_data_list)
                    if item[3] is not None
                }  # Map COCO ID to its index in the *original* images_to_infer list
                self.class_names_to_ids_map = {
                    cat["name"]: cat["id"] for cat in coco_gt_obj.dataset["categories"]
                }

            def __len__(self):
                return len(self.images_data)

            def __getitem__(self, idx):
                img_pil, img_id, img_path, _ = self.images_data[idx]
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anns = self.coco.loadAnns(ann_ids)

                boxes = []
                labels = []
                for ann in anns:
                    x, y, w, h = ann["bbox"]
                    boxes.append([x, y, x + w, y + h])
                    labels.append(ann["category_id"])

                return {
                    "name": Path(img_path).name,
                    "image": np.array(
                        img_pil
                    ),  # evaluate_pr_per_image expects np array
                    "annotations": pd.DataFrame(
                        {
                            "xtl": [b[0] for b in boxes],
                            "ytl": [b[1] for b in boxes],
                            "xbr": [b[2] for b in boxes],
                            "ybr": [b[3] for b in boxes],
                            "label": labels,
                        }
                    ),
                    "masks": [],  # Assuming bbox only for now for RT-DETR
                }

        # Create the mock dataset only for images that have GT and were processed
        mock_dataset = MockCellMaskDataset(coco_gt, images_to_infer)

        if not mock_dataset.images_data:
            logger.warning(
                "No images with ground truth annotations were processed. Skipping metrics computation."
            )
            config.inference.perform_metrics = False
        else:
            pr_predictions_aligned = []
            # We need to ensure that pr_predictions_aligned corresponds to the images in mock_dataset.images_data
            for img_data_from_mock_ds in mock_dataset.images_data:
                original_img_id = img_data_from_mock_ds[1]
                original_idx_in_images_to_infer = image_id_to_original_idx[
                    original_img_id
                ]
                pr_predictions_aligned.append(
                    all_predictions_for_pr_eval[original_idx_in_images_to_infer]
                )

            class_ids_of_interest = list(adapter_model.get_label_map().keys())

            # Configure AnnotationFilter if needed, similar to evaluate_all_models.py
            annotation_filter = None
            if config.remap_labels and hasattr(config.data, "class_remapping"):
                annotation_filter = AnnotationFilter(
                    classnames_mapping_dict=config.data.class_remapping,
                    class_ids_to_class_names_map=adapter_model.get_label_map(),
                )

            (
                num_true_positives_per_class,
                num_false_positives_per_class,
                num_false_negatives_per_class,
                precision_per_class,
                recall_per_class,
                agg_f1_per_class,  # Renamed to avoid clash with micro_f1_val
                total_tp,
                total_fp,
                total_fn,
                micro_precision,
                micro_recall,
                micro_f1_val,
                per_image_pr_metrics,
            ) = evaluate_pr_per_image(
                predictions=pr_predictions_aligned,
                dataset=mock_dataset,
                class_ids_of_interest=class_ids_of_interest,
                min_iou=0.5,  # Configurable
                use_mask=adapter_model.get_metadata()["predict_masks"],
                annotation_filter=annotation_filter,
            )

            for img_pr_metrics in per_image_pr_metrics:
                row = {
                    "image": img_pr_metrics["name"],
                    "image_id": next(
                        item[1]
                        for item in images_to_infer
                        if Path(item[2]).name == img_pr_metrics["name"]
                    ),  # Get actual image_id
                    "inference_time_ms": inference_times_ms[
                        image_id_to_original_idx[
                            next(
                                item[1]
                                for item in images_to_infer
                                if Path(item[2]).name == img_pr_metrics["name"]
                            )
                        ]
                    ],
                }
                for class_id in class_ids_of_interest:
                    class_name = adapter_model.get_label_map().get(
                        class_id, f"ID_{class_id}"
                    )
                    row[f"{class_name}_TP"] = img_pr_metrics["tp"][class_id]
                    row[f"{class_name}_FP"] = img_pr_metrics["fp"][class_id]
                    row[f"{class_name}_FN"] = img_pr_metrics["fn"][class_id]
                    row[f"{class_name}_Precision"] = img_pr_metrics["precision"][
                        class_id
                    ]
                    row[f"{class_name}_Recall"] = img_pr_metrics["recall"][class_id]
                    row[f"{class_name}_F1"] = img_pr_metrics["f1"][class_id]
                per_image_metrics_data_rows.append(row)

            if per_image_metrics_data_rows:
                per_image_metrics_df = pd.DataFrame(per_image_metrics_data_rows)
                per_image_csv_path = (
                    metrics_output_dir
                    / f"{dataset_name}_{adapter_model.get_model_name().replace(' ', '_')}_per_image_metrics.csv"
                )
                per_image_metrics_df.to_csv(per_image_csv_path, index=False)
                logger.info(f"Per-image metrics saved to: {per_image_csv_path}")

            # --- Save Aggregate Metrics JSON ---
            aggregate_metrics = {
                "model_name": adapter_model.get_model_name(),
                "dataset_name": dataset_name,
                "detection_threshold": adapter_model._confidence,
                "cropping_enabled": config.inference.cropping_enabled,
                "average_inference_time_ms": np.mean(inference_times_ms),
                "total_images_processed": len(images_to_infer),
                "total_images_with_gt_for_metrics": len(mock_dataset),
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "micro_f1": micro_f1_val,
                "per_class_metrics": {
                    adapter_model.get_label_map().get(cid, f"ID_{cid}"): {
                        "TP": num_true_positives_per_class[cid],
                        "FP": num_false_positives_per_class[cid],
                        "FN": num_false_negatives_per_class[cid],
                        "Precision": precision_per_class[cid],
                        "Recall": recall_per_class[cid],
                        "F1": agg_f1_per_class[cid],
                    }
                    for cid in class_ids_of_interest
                },
            }
            aggregate_json_path = (
                metrics_output_dir
                / f"{dataset_name}_{adapter_model.get_model_name().replace(' ', '_')}_aggregate_metrics.json"
            )
            with open(aggregate_json_path, "w") as f:
                json.dump(aggregate_metrics, f, indent=4)
            logger.info(f"Aggregate metrics saved to: {aggregate_json_path}")

    else:
        logger.info(
            "Skipping metrics calculation as 'perform_metrics' is False or no ground truth provided."
        )

    logger.info("Inference complete.")


if __name__ == "__main__":
    main()
