from models.AbstractVisionModel import VisionModel
from utils.model_utils import to_numpy, get_crop_corners 
import torch

import numpy as np
import cv2
from PIL import Image
import onnxruntime
import onnx

import os
import time
import logging
from typing import Tuple, List, Final, Optional, Dict, Union
from collections import OrderedDict


DEFAULT_DETECTION_CONFIDENCE: Final[float] = 0.4

DEFAULT_NMS_THRESHOLD: Final[float] = 0.3
DEFAULT_MODEL_INPUT_SIZE: Final[Tuple[int, int]] = (640, 640)  # width x height
# the aspect ratio of the passed image to the model should be within this threshold (percentagewise)
# of the aspect ratio of the input image size specified above
# if the deviation is more than this threshold, the results may not be reliable
INPUT_IMAGE_ASPECT_RATIO_DIFF_DEV_THRESH: Final[float] = 0.1

DEFAULT_RESIZE_10x: Final[Dict[Tuple[int, int], Tuple[int, int]]] = {
    (2000, 1600): (1000, 800),
    (4512, 4512): (2440, 2440),
}

DEFAULT_CROP_CORNERS_10x: Final[Dict[Tuple[int, int], List[List[int]]]] = {
    (2000, 1600): [
        [0, 0, 640, 640],
        [0, 160, 640, 800],
        [360, 0, 1000, 640],
        [360, 160, 1000, 800],
    ],
    (4512, 4512): get_crop_corners(
        image_width=DEFAULT_RESIZE_10x[(4512, 4512)][0],
        image_height=DEFAULT_RESIZE_10x[(4512, 4512)][1],
        overlap_in_x = 40,
        overlap_in_y = 40,
        input_size = DEFAULT_MODEL_INPUT_SIZE,
    ),
}


DEFAULT_RESIZE_4x: Final[Dict[Tuple[int, int], Tuple[int, int]]] = {(4512, 4512): (4512, 4512),}

DEFAULT_CROP_CORNERS_4x: Final[Dict[Tuple[int, int], List[List[int]]]] = {
    (4512, 4512): get_crop_corners(
        image_width=DEFAULT_RESIZE_4x[(4512, 4512)][0],
        image_height=DEFAULT_RESIZE_4x[(4512, 4512)][1],
        overlap_in_x = 80, # 80
        overlap_in_y = 80, # 80
        input_size = DEFAULT_MODEL_INPUT_SIZE,
    )
}

def load_onnx_model_and_metadata(weights_path: str, use_onnx_runtime: bool, classnames_path: str=None):
    # loading the ONNX model
    logging.info(
        f"Loading YOLOv5 ONNX weights from {weights_path}."
    )
    try:
        if use_onnx_runtime:
            logging.info(
                f"Using ONNX Runtime for running the model."
            )

            net = onnxruntime.InferenceSession(weights_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                
            if 'CUDAExecutionProvider' not in onnxruntime.get_available_providers():
                logging.warning(
                    f"CUDAExecutionProvider is not available in ONNX Runtime! The model will be run on CPU."
                )
        else:
            logging.info(
                f"Using OpenCV dnn for running the model. Setting dnn to use CUDA."
            )
            net = cv2.dnn.readNetFromONNX(weights_path)

            try:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            except Exception:
                logging.warning(
                    "Failed to set dnn to use CUDA! Installed OpenCV is not compiled with CUDA support! "
                    "The model will be run on CPU."
                )

    except Exception as ex:
        logging.error(
            f"Failed to initialize YOLOv5 model with CUDA. Likely the path to model ONNX weights "
            f"{weights_path} is incorrect or ONNX Runtime is not installed: {repr(ex)}."
        )
        return None, None
        
    # read class names and other metadata from ONNX file and create the label map
    try:
        if use_onnx_runtime:
            metadata: Dict[str, str] = net.get_modelmeta().custom_metadata_map
            # convert string values to dicts, or integers or strings
            metadata = {k: eval(v) for k, v in metadata.items()}
            {v: k for k, v in metadata['names'].items()}
            # metadata will be a dictionary with keys as 'resolution', 'release_date', 'model_type', 'model_name', 'model_extra_info', 'names', 'stride'
            # example: {'resolution': 640,
            #           'release_date': '20240415',
            #           'model_type': 'YOLO Detector',
            #           'model_name': 'YOLOv5m',
            #           'model_extra_info': 'V5 Medium',
            #           'names': {0: 'cell', 1: 'bead'},
            #           'stride': 32}
        else:
            # reading the model with OpenCV dnn does not capture the metadata above, try onnx directly
            onnx_model = onnx.load(weights_path)
            metadata: Dict[str, str] = {}
            for field in onnx_model.metadata_props:
                metadata[field.key] = eval(field.value)
            
    except Exception as ex:
        # try reading from a defualt location
        try: 
            # read class names and create the label map
            with open(classnames_path, 'r') as f:  # if fails to read then blow with error
                class_names: List[str] = [cname.strip() for cname in f.readlines()]

            label_map: Dict[int, str] = {i: c for i, c in enumerate(class_names)}
            metadata = {'names': label_map}

        except Exception as ex:
            logging.error(
                "Failed to initialize YOLOv5 model with CUDA. The model does not include label map in its metadata and a valid path to classnames file "
                    f"{classnames_path} is not provided: {repr(ex)}."
                )
            return net, None

    metadata['predict_masks'] = False
    
    return net, metadata


# YOLOv5 detector class
class Yolov5ObjectDetector(VisionModel):
    def __init__(
        self,
        weights_path: str,
        model_name: str = 'YOLOv5', 
        label_map: Optional[Dict[int, str]] = None, # to be read from the weights file
        model_input_size: Optional[Tuple[int, int]] = None, # to be read from the weights file
        confidence: float = DEFAULT_DETECTION_CONFIDENCE,
        # device is not applicable here
        nms_threshold: float = DEFAULT_NMS_THRESHOLD,
    ): 
        self._nms_threshold = nms_threshold
        # this is an ONNX model, which is different than our Hugging Face or Pytorch models which are saved
        # together with the metadata in a .pt, we first read the metadata and then load the model

        _, metadata = load_onnx_model_and_metadata(
            weights_path=weights_path, 
            use_onnx_runtime=True
        )
        
        if label_map is None:
            label_map: Dict[int, str] = metadata['names']

        if model_input_size is None:
            model_input_size: Tuple[int, int] = (metadata['resolution'], metadata['resolution'])
        
        # now initialize the model 
        super().__init__(
            weights_path,
            model_name, 
            label_map,
            model_input_size, 
            confidence,
            None, # no device as this is not a pytorch implementation, we can use string universally (TODO) 
        )

    def load(self):
        # now load the model

        if self._model_input_size is None:
            # we need a valid model input size for Mask2Former model
            logging.error(
                f"Missing model input size! It was neither included in the weights file nor passed during instantiation! "
                f"Failed to instantiate {self._model_name} class."
            )
            return 

        # load the metadata again, existing fields should be consistent
        self._model, self._metadata = load_onnx_model_and_metadata(
            weights_path=self._weights_path, 
            use_onnx_runtime=True
        )
        
        if self._model is None:
            return

        self._loaded = True

    def detect_batch(
        self,
        input_images_list: List[Union[Image.Image, np.ndarray]],
    ) -> List[Dict[str, list]]:

        if not self._loaded:
            logging.error(
                f"{self._model_name} model has not been initialized. Please initialize the class before detect()."
            )
            
            out: List[Dict[str, list]] = [{"boxes": [], "scores": [], "labels": []}] * len(input_images_list)
            return out

        # convert to 3-channel images if needed, and store the original image dimensions for 
        # post processing
        images_list: List[np.ndarray] = []
        org_img_dims: List[Tuple[int, int]] = []
        
        for img in input_images_list:
            if isinstance(img, Image.Image):
                img_array: np.ndarray = np.array(img)
            else:
                img_array: np.ndarray = img.copy()
            
            img_shape: tuple = img_array.shape
            if len(img_shape) < 3:
                # the model expects a 3-D input image
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                
            org_img_dims.append(img_shape[:2])

            # check if the aspect ratio of the input image is almost the same as the aspect ratio of
            # the model input size
            # if not, then the input image will be resized without keeping its aspect ratio when it
            # is passed to the model, and this may lead to inaccurate detection
            aspect_ratio_diff: float = float(img_shape[1] * self._model_input_size[1]) / float(
                    img_shape[0] * self._model_input_size[0]
            ) - 1.0
           
            if np.abs(aspect_ratio_diff) > INPUT_IMAGE_ASPECT_RATIO_DIFF_DEV_THRESH:
                logging.warning(
                    f"The input image has a different aspect ratio: {img_shape[1] / float(img_shape[0])} than the model: "
                    f"{self._model_input_size[0] / float(self._model_input_size[1])}! The results may not be accurate"
                )
            images_list.append(img_array)

        outputs: List[np.ndarray] = []
        for img_idx, input_img in enumerate(images_list):    
            # transpose the input image to C x H x W and add batch dimension
            input_image = np.expand_dims(np.transpose(input_img, (2, 0, 1)), axis=0)
            # construct the input dictionary
            ort_inputs: dict = {self._model.get_inputs()[0].name: input_image.astype(np.float32) / 255.0}
            # run the forward pass to get output
            out: np.array = self._model.run(None, ort_inputs)
        
            # only one output layer, pick the first element in the tuple then remove the batch dimension
            outputs.append(out[0][0])
        
        return self.postprocess(outputs, org_img_dims)

        
    def postprocess(
        self, 
        outputs: List[np.ndarray], 
        org_img_dims: List[Tuple[int, int]]
    ) -> List[Dict[str, list]]:
        """Post process outputs, discarding unreliable detections & performing NMS"""

        results: List[Dict[str, list]] = []
        for predictions, org_img_dim in zip(outputs, org_img_dims):
            # discard unreliable detections before NMS to reduce NMS computations
            predictions = predictions[np.where(predictions[:, 4] >= min(0.5, self._confidence))[0]]
            # class IDs
            labels: np.ndarray = np.argmax(predictions[:, 5:], axis=1)
            # scores
            scores: np.ndarray = np.max(predictions[:, 5:], axis=1)
            # center (x, y) and width and height of each box
            cx: np.array = predictions[:, 0]
            cy: np.array = predictions[:, 1]
            w: np.array = predictions[:, 2]
            h: np.array = predictions[:, 3]
            # convert the boxes from (cx, cy, w, h) format to (xtl, ytl, w, h) before running NMS
            boxes: np.ndarray = np.vstack([cx - w / 2, cy - h / 2, w, h]).T * np.array(
                [  # org_img_dim is in (height, width)
                    org_img_dim[1] / self._model_input_size[0],
                    org_img_dim[0] / self._model_input_size[1],
                ]
                * 2
            )

            # run NMS per class
            index_list_to_return: List[int] = []
            # we are interested in unique class IDs in the output, hence set(classes)
            unique_label_ids = list(set(labels))
            for label_id in unique_label_ids:
                class_indexes = np.where(labels == label_id)[0]
                valid_det_indexes = cv2.dnn.NMSBoxes(
                    boxes[class_indexes],
                    scores[class_indexes],
                    self._confidence,
                    self._nms_threshold,
                )

                index_list_to_return += list(class_indexes[valid_det_indexes])

            # NMS removes repeated detections (may return empty ndarrays)
            boxes = boxes[index_list_to_return, :]
            scores = scores[index_list_to_return]
            labels = labels[index_list_to_return]

            # convert from xtl, ytl, w, h) to (xtl, ytl, xbr, ybr) for each box
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]
            # convert the coordinates to int
            boxes = boxes.astype(int)

            # map the class IDs
            if self._detected_class_ids_remap is not None and len(labels):
                labels = np.vectorize(
                    lambda x: (
                        self._detected_class_ids_remap[x]
                        if x in self._detected_class_ids_remap
                        else x
                    )
                )(labels)

            results.append({"boxes": boxes.tolist(), "labels": labels.tolist(), "scores": scores.tolist()})
        
        return results

