from models.AbstractVisionModel import VisionModel
from utils.model_utils import to_numpy

import numpy as np
import cv2
from PIL import Image
import torch
from transformers import (
    DeformableDetrConfig,
    DeformableDetrForObjectDetection,
    DeformableDetrImageProcessor,
)

import os
import logging
from typing import Tuple, List, Final, Optional, Dict, Union

DEFAULT_DETECTION_CONFIDENCE: Final[float] = 0.4
DEFAULT_MODEL_INPUT_SIZE: Final[Tuple[int, int]] = (1024, 1024)

DEFAULT_RESIZE: Final[Dict[Tuple[int, int], Tuple[int, int]]] = {
    (2000, 1600): (1280, 1024),
    (4512, 4512): (3328, 3328),
}

DEFAULT_CROP_CORNERS: Final[Dict[Tuple[int, int], List[List[int]]]] = {
    (2000, 1600): [[0, 0, 1024, 1024], [256, 0, 1280, 1024]],
    (4512, 4512): [
        [0, 0, 1024, 1024],
        [0, 768, 1024, 1792],
        [0, 1536, 1024, 2560],
        [0, 2304, 1024, 3328],
        [768, 0, 1792, 1024],
        [768, 768, 1792, 1792],
        [768, 1536, 1792, 2560],
        [768, 2304, 1792, 3328],
        [1536, 0, 2560, 1024],
        [1536, 768, 2560, 1792],
        [1536, 1536, 2560, 2560],
        [1536, 2304, 2560, 3328],
        [2304, 0, 3328, 1024],
        [2304, 768, 3328, 1792],
        [2304, 1536, 3328, 2560],
        [2304, 2304, 3328, 3328],
    ],
}


# Deformable-DETR object detection class
class DeformableDetrObjectDetector(VisionModel):
    def __init__(
        self,
        weights_path: str,
        model_name: str = "Deformable DETR",
        label_map: Optional[Dict[int, str]] = None,  # to be read from the weights file
        model_input_size: Optional[
            Tuple[int, int]
        ] = None,  # to be read from the weights file
        confidence: float = DEFAULT_DETECTION_CONFIDENCE,
        device: torch.device = torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
        backbone_name_str: str = "sam",
    ):
        self._backbone = backbone_name_str
        super().__init__(
            weights_path,
            model_name,
            label_map,
            model_input_size,
            confidence,
            device,
        )

    def load(self) -> None:

        if self._model_input_size is None:
            # we need a valid model input size for Mask2Former model
            logging.error(
                f"Missing model input size! It was neither included in the weights file nor passed during instantiation! "
                f"Failed to instantiate {self._model_name} class."
            )
            return

        if self._backbone.lower() == "sam":
            # Deformable DETR model with SAM backbone
            model_config = DeformableDetrConfig(
                use_timm_backbone=True,
                backbone="samvit_base_patch16.sa1b",  # replace the backbone with SAM base model
                use_pretrained_backbone=True,
                id2label=self._label_map,
                label2id=self._reverse_label_map,
            )
            self._model = DeformableDetrForObjectDetection(model_config)
        elif self._backbone.lower() == "dinov2":
            model_config = DeformableDetrConfig(
                use_timm_backbone=True,
                backbone="vit_base_patch14_reg4_dinov2.lvd142m",
                use_pretrained_backbone=True,
                id2label=self._label_map,
                label2id=self._reverse_label_map,
            )
            self._model = DeformableDetrForObjectDetection(model_config)

        # loading the PyTorch model
        try:
            self._model.load_state_dict(self._model_state_dict)
            # this is for freezing the backbone, not really needed for inference time
            for param in self._model.model.backbone.parameters():
                param.requires_grad_(False)
            self._model.to(self._device)
            self._model.eval()

        except Exception as ex:
            logging.error(
                f"Failed to load {self._model_name} model from the weights file: {repr(ex)}."
            )
            return

        self._hg_preprocessor = DeformableDetrImageProcessor.from_pretrained(
            "SenseTime/deformable-detr"
        )
        # modify the resizing part as we will replace the backbone with that of SAM (both sides will be equal for SAM)
        self._hg_preprocessor.size = {
            "longest_edge": max(self._model_input_size),
            "shortest_edge": min(self._model_input_size),
        }

        # added for consistency with the YOLO model
        # metadata will be a dictionary with keys as 'predict_masks', 'resolution', 'release_date', 'model_type',
        # 'model_name', 'model_extra_info', 'names', 'stride', ...
        # example: {'predict_masks': False
        #           'resolution': 640,
        #           'release_date': '20240415',
        #           'model_type': 'YOLO Detector',
        #           'model_name': 'YOLOv5m',
        #           'model_extra_info': 'V5 Medium',
        #           'names': {0: 'cell', 1: 'bead'},
        self._metadata = {
            "predict_masks": False,  # detector model
            "resolution": self._model_input_size[0],  # square input
            "release_date": os.path.basename(self._weights_path).split("_")[
                0
            ],  # the model name starts with the release date
            "model_type": "Transformer Detector",
            "model_name": self._model_name,
            "model_extra_info": (
                "With SAM backbone"
                if self._backbone.lower() == "sam"
                else (
                    "With DINOv2 backbone" if self._backbone.lower() == "dinov2" else ""
                )
            ),
            "names": self._label_map,
            "magnification": "4x"
            if "4x" in os.path.basename(self._weights_path)
            else "10x",  # 4x should be specified in the name
        }

        self._loaded = True

    def detect_batch(
        self,
        input_images_list: List[Union[Image.Image, np.ndarray]],
    ) -> List[Dict[str, list]]:

        if not self._loaded:
            logging.error(
                f"{self._model_name} model has not been initialized. Please initialize the class before detect()."
            )

            out: List[Dict[str, list]] = [
                {"boxes": [], "scores": [], "labels": []}
            ] * len(input_images_list)
            return out

        # convert to 3-channel images if needed, and store the original image dimensions for
        # post processing
        images_list: List[np.ndarray] = []
        org_img_dims: List[Tuple[int, int]] = []

        for img in input_images_list:
            if isinstance(img, Image.Image):
                img_array: np.ndarray = np.array(img)
            else:
                img_array: np.ndarray = img

            img_shape: tuple = img_array.shape
            if len(img_shape) < 3:
                images_list.append(cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB))
            else:
                images_list.append(img_array)
            org_img_dims.append(img_shape[:2])

        processed_imgs_dict = self._hg_preprocessor(images_list, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(
                pixel_values=processed_imgs_dict["pixel_values"].to(self._device)
            )

        results = self.postprocess(outputs, org_img_dims)
        # clear the CUDA cache
        torch.cuda.empty_cache()

        return results

    def postprocess(self, outputs, org_img_dims):
        processed_outputs = self._hg_preprocessor.post_process_object_detection(
            outputs, threshold=self._confidence, target_sizes=org_img_dims
        )

        # processed_outputs is a list of len(input_images_list) dictionary elements, each dictionary containing the detections
        # for the input image in the input list with keys as 'boxes', 'labels' and 'scores', and values as
        # - 'boxes': a (num_detection, 4) torch.float32 tensor of bounding boxes in (xtl, ytl, xbr, ybr) format
        # - 'labels': a (num_detection, 1) torch.int64 tensor of class IDs
        # - 'scores': a (num_detection, 1) torch.float32 tensor of detection confidences

        if len(processed_outputs) == 0:
            # this should not happen and is not expected, return as if the model has not detected anything (for the whole list of images)
            results = [
                {
                    "boxes": [],
                    "labels": [],
                    "scores": [],
                }
            ] * len(outputs)
        else:
            # move to CPU and convert to numpy arrays before returning
            results = [
                {k: list(to_numpy(v)) for k, v in result.items()}
                for result in processed_outputs
            ]

        for result in results:
            # map the class IDs
            if self._detected_class_ids_remap is not None and len(result["labels"]):
                result["labels"] = np.vectorize(
                    lambda x: (
                        self._detected_class_ids_remap[x]
                        if x in self._detected_class_ids_remap
                        else x
                    )
                )(result["labels"]).tolist()

        return results
