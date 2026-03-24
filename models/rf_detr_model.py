from models.AbstractVisionModel import VisionModel

import numpy as np
import cv2
from PIL import Image
import torch
from rfdetr import RFDETRBase

import os
import logging
from typing import Tuple, List, Final, Optional, Dict, Union

DEFAULT_DETECTION_CONFIDENCE: Final[float] = 0.45
DEFAULT_MODEL_INPUT_SIZE: Final[Tuple[int, int]] = (672, 672)
DEFAULT_LABEL_MAP: Final[Dict[int, str]] = {
    0: "cell",
    1: "bead",
    2: "soma",
    3: "cell-adhered",
}
# the aspect ratio of the passed image to the model should be within this threshold (percentagewise)
# of the aspect ratio of the input image size specified above
# if the deviation is more than this threshold, the results may not be reliable
INPUT_IMAGE_ASPECT_RATIO_DIFF_DEV_THRESH: Final[float] = 0.1

"""
DEFAULT_RESIZE: Final[Dict[Tuple[int, int, str], Tuple[int, int]]] = {
    (2000, 1600, "10x"): (1924, 1540),
    (4512, 4512, "10x"): (4342, 4342),
    (4512, 4512, "4x"): (4342, 4342),
}

DEFAULT_CROP_CORNERS: Final[Dict[Tuple[int, int, str], List[List[int]]]] = {
    (2000, 1600, "10x"): get_crop_corners(
        image_width=RESIZE[(2000, 1600, "10x")][0],
        image_height=RESIZE[(2000, 1600, "10x")][1],
        overlap_in_x=180,
        overlap_in_y=308,
        input_size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    ),
    (4512, 4512, "10x"): get_crop_corners(
        image_width=RESIZE[(4512, 4512, "10x")][0],
        image_height=RESIZE[(4512, 4512, "10x")][1],
        overlap_in_x=202,
        overlap_in_y=202,
        input_size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    ),
    (4512, 4512, "4x"): get_crop_corners(
        image_width=RESIZE[(4512, 4512, "4x")][0],
        image_height=RESIZE[(4512, 4512, "4x")][1],
        overlap_in_x=202,
        overlap_in_y=202,
        input_size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    )
}
"""

DEFAULT_RESIZE: Final[Dict[Tuple[int, int, str], Tuple[int, int]]] = {
    (2000, 1600): (2000, 1600),
    (4512, 4512): (4512, 4512),
}

DEFAULT_CROP_CORNERS: Final[Dict[Tuple[int, int, str], List[List[int]]]] = {
    (2000, 1600): [
        [0, 0, 672, 672],
        [0, 464, 672, 1136],
        [0, 928, 672, 1600],
        [444, 0, 1116, 672],
        [444, 464, 1116, 1136],
        [444, 928, 1116, 1600],
        [888, 0, 1560, 672],
        [888, 464, 1560, 1136],
        [888, 928, 1560, 1600],
        [1328, 0, 2000, 672],
        [1328, 464, 2000, 1136],
        [1328, 928, 2000, 1600],
    ],
    (4512, 4512): [
        [0, 0, 672, 672],
        [0, 480, 672, 1152],
        [0, 960, 672, 1632],
        [0, 1440, 672, 2112],
        [0, 1920, 672, 2592],
        [0, 2400, 672, 3072],
        [0, 2880, 672, 3552],
        [0, 3360, 672, 4032],
        [0, 3840, 672, 4512],
        [480, 0, 1152, 672],
        [480, 480, 1152, 1152],
        [480, 960, 1152, 1632],
        [480, 1440, 1152, 2112],
        [480, 1920, 1152, 2592],
        [480, 2400, 1152, 3072],
        [480, 2880, 1152, 3552],
        [480, 3360, 1152, 4032],
        [480, 3840, 1152, 4512],
        [960, 0, 1632, 672],
        [960, 480, 1632, 1152],
        [960, 960, 1632, 1632],
        [960, 1440, 1632, 2112],
        [960, 1920, 1632, 2592],
        [960, 2400, 1632, 3072],
        [960, 2880, 1632, 3552],
        [960, 3360, 1632, 4032],
        [960, 3840, 1632, 4512],
        [1440, 0, 2112, 672],
        [1440, 480, 2112, 1152],
        [1440, 960, 2112, 1632],
        [1440, 1440, 2112, 2112],
        [1440, 1920, 2112, 2592],
        [1440, 2400, 2112, 3072],
        [1440, 2880, 2112, 3552],
        [1440, 3360, 2112, 4032],
        [1440, 3840, 2112, 4512],
        [1920, 0, 2592, 672],
        [1920, 480, 2592, 1152],
        [1920, 960, 2592, 1632],
        [1920, 1440, 2592, 2112],
        [1920, 1920, 2592, 2592],
        [1920, 2400, 2592, 3072],
        [1920, 2880, 2592, 3552],
        [1920, 3360, 2592, 4032],
        [1920, 3840, 2592, 4512],
        [2400, 0, 3072, 672],
        [2400, 480, 3072, 1152],
        [2400, 960, 3072, 1632],
        [2400, 1440, 3072, 2112],
        [2400, 1920, 3072, 2592],
        [2400, 2400, 3072, 3072],
        [2400, 2880, 3072, 3552],
        [2400, 3360, 3072, 4032],
        [2400, 3840, 3072, 4512],
        [2880, 0, 3552, 672],
        [2880, 480, 3552, 1152],
        [2880, 960, 3552, 1632],
        [2880, 1440, 3552, 2112],
        [2880, 1920, 3552, 2592],
        [2880, 2400, 3552, 3072],
        [2880, 2880, 3552, 3552],
        [2880, 3360, 3552, 4032],
        [2880, 3840, 3552, 4512],
        [3360, 0, 4032, 672],
        [3360, 480, 4032, 1152],
        [3360, 960, 4032, 1632],
        [3360, 1440, 4032, 2112],
        [3360, 1920, 4032, 2592],
        [3360, 2400, 4032, 3072],
        [3360, 2880, 4032, 3552],
        [3360, 3360, 4032, 4032],
        [3360, 3840, 4032, 4512],
        [3840, 0, 4512, 672],
        [3840, 480, 4512, 1152],
        [3840, 960, 4512, 1632],
        [3840, 1440, 4512, 2112],
        [3840, 1920, 4512, 2592],
        [3840, 2400, 4512, 3072],
        [3840, 2880, 4512, 3552],
        [3840, 3360, 4512, 4032],
        [3840, 3840, 4512, 4512],
    ],
}


# RT-DETR object detection class
class RfDetrObjectDetector(VisionModel):
    def __init__(
        self,
        weights_path: str,
        model_name: str = "RF-DETR",
        label_map: Optional[Dict[int, str]] = None,  # to be read from the weights file
        model_input_size: Optional[
            Tuple[int, int]
        ] = None,  # to be read from the weights file
        confidence: float = DEFAULT_DETECTION_CONFIDENCE,
        device: torch.device = torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
    ):
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

        # loading the model
        try:
            self._model = RFDETRBase(pretrain_weights=self._weights_path)
            # not needed as they will be taken care of in self._model.predict method
            # self._model.to(self._device)
            # self._model.eval()

        except Exception as ex:
            logging.error(
                f"Failed to load {self._model_name} model from the weights file: {repr(ex)}."
            )
            return

        self._metadata = {
            "predict_masks": False,  # detector model
            "resolution": self._model_input_size,
            "release_date": os.path.basename(self._weights_path).split("_")[
                0
            ],  # the model name starts with the release date
            "model_type": "Transformer Detector",
            "model_name": self._model_name,
            "model_extra_info": "None",
            "names": self._label_map,
            "magnification": "4x"
            if "4x" in os.path.basename(self._weights_path)
            else "10x",
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
        aspect_ratio_diffs: List[float] = []

        for img in input_images_list:
            if isinstance(img, Image.Image):
                img_array: np.ndarray = np.array(img)
            else:
                img_array: np.ndarray = img.copy()

            img_shape: tuple = img_array.shape
            if len(img_shape) < 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

            org_img_dims.append(img_shape[:2])

            # check if the aspect ratio of the input image is almost the same as the aspect ratio of
            # the model input size
            # if not, then the input image will be resized without keeping its aspect ratio when it
            # is passed to the model, and this may lead to inaccurate detection
            aspect_ratio_diff: float = (
                float(img_shape[1] * self._model_input_size[1])
                / float(img_shape[0] * self._model_input_size[0])
                - 1.0
            )

            if np.abs(aspect_ratio_diff) > INPUT_IMAGE_ASPECT_RATIO_DIFF_DEV_THRESH:
                logging.warning(
                    f"The input image has a different aspect ratio: {img_shape[1] / float(img_shape[0])} than the model: "
                    f"{self._model_input_size[0] / float(self._model_input_size[1])}! The results may not be accurate"
                )

            # resize the input image to match the model input size
            if aspect_ratio_diff != 0:
                img_array = cv2.resize(img_array, self._model_input_size)

            aspect_ratio_diffs.append(aspect_ratio_diff)
            images_list.append(img_array)

        results: List[Dict[str, list]] = []
        for img_idx, input_img in enumerate(images_list):
            out: Dict[str, list] = {}
            with torch.no_grad():
                detections = self._model.predict(input_img, threshold=self._confidence)

            if aspect_ratio_diffs[img_idx] != 0:
                # self._model_input_size is width/height, hence self._model_input_size[0] is width
                # we are using square input sizes and this should not really matter
                out["boxes"] = (
                    (
                        detections.xyxy
                        * np.array(
                            [
                                org_img_dims[img_idx][1] / self._model_input_size[0],
                                org_img_dims[img_idx][0] / self._model_input_size[1],
                            ]
                            * 2
                        )
                    )
                    .astype(int)
                    .tolist()
                )
            else:
                out["boxes"] = detections.xyxy.astype(int).tolist()

            out["scores"] = detections.confidence.tolist()
            out["labels"] = detections.class_id.tolist()

            results.append(out)

        results = self.postprocess(results)
        # clear the CUDA cache
        torch.cuda.empty_cache()

        return results

    def postprocess(self, results):
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
