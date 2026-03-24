import os
import sys
import time
import logging
from typing import Tuple, List, Final, Optional, Dict, Union
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import (
    RTDetrImageProcessor,
    RTDetrForObjectDetection,
    RTDetrV2ForObjectDetection,
)

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Mask RCNN"
    )
)
# custom dinov2 related imports
from dinov2_backbone_with_fpn import Dinov2BackBoneWithFPN
from AbstractVisionModel import VisionModel

# BASE_PATH: Final[str] = '/home/cellareye/Cellanome/dl-mehdi/Mask RCNN/checkpoints'
# MODEL_WEIGHTS_PATH: Final[str] = '/home/cellareye/Cellanome/dl-mehdi/Mask RCNN/checkpoints/20250331_sets_1_2_3_6_to_41_rt_detr_16_bs_10_epochs.pt'
# changes made by AS
BASE_PATH: Final[str] = "/global/home/ashish.sinha/cellanome/models/"
MODEL_WEIGHTS_PATH: Final[str] = (
    "/global/home/ashish.sinha/cellanome/models/rt_detr_v2_with_dinov2_fpn_2_7_12/20250603_sets_1_2_3_6_to_41_rt_detrv2_with_dinov2_fpn_2_7_12_16_bs_10_epochs.pt"
)
DEFAULT_DETECTION_CONFIDENCE: Final[float] = 0.4
MODEL_INPUT_SIZE: Final[int] = 640
# TRANSFORM_MEAN: Final[List[float]] = [0.485, 0.456, 0.406]
# TRANSFORM_STD: Final[List[float]] = [0.229, 0.224, 0.225]


# Utility functions
# very efficient batch IoU calculation
# needed for combining detection results from different image crops on
# the overlapping parts; we use this function instead of
# torchvision.ops.box_iou(bboxes1, bboxes2) to remove dependency on torch
def iou_batch(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """Given Nx4 and Mx4 ndarrays of bounding boxes, compute pairwise IoUs"""
    # expand dims to allow computing pairwise IoU via outer products (creates NxM below)
    bboxes1 = np.expand_dims(bboxes1, 1)  # Nx1x4
    bboxes2 = np.expand_dims(bboxes2, 0)  # 1xMx4
    # determine the (x, y) coordinates of the intersection rectangle
    inter_x1s = np.maximum(bboxes1[..., 0], bboxes2[..., 0])  # pairwise max NxM
    inter_y1s = np.maximum(bboxes1[..., 1], bboxes2[..., 1])  # pairwise max NxM
    inter_x2s = np.minimum(bboxes1[..., 2], bboxes2[..., 2])  # pairwise min NxM
    inter_y2s = np.minimum(bboxes1[..., 3], bboxes2[..., 3])  # pairwise min NxM
    inter_ws = np.maximum(
        0.0, inter_x2s - inter_x1s
    )  # pairwise width of intersection rectangle NxM
    inter_hs = np.maximum(
        0.0, inter_y2s - inter_y1s
    )  # pairwise height of intersection rectangle NxM
    inter_areas = inter_ws * inter_hs  # pairwise intersection area NxM
    union_areas = (
        (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        - inter_areas
        + 1e-30
    )  # pairwise union area NxM
    return inter_areas / union_areas  # pairwise intersection divided by union (iou) NxM


def overlap_batch(
    bboxes1: np.ndarray, bboxes2: np.ndarray, ordered: bool = False
) -> np.ndarray:
    """Given Nx4 & Mx4 ndarrays of bounding boxes, compute pairwise overlaps defined as the intersection over
    the smallest box area (ordered set to False); a small box fully enclosed by a large box has an "overlap" of 1.
    if ordered is set to 1, the overlap is the intersection over the area of the box from bboxes2 set. In this case,
    overlap is close to 1 only if the box from bboxes2 lies inside the box from bboxes1"""

    # expand dims to allow computing pairwise overlap via outerproducts (creates NxM below)
    bboxes1 = np.expand_dims(bboxes1, 1)  # Nx1x4
    bboxes2 = np.expand_dims(bboxes2, 0)  # 1xMx4

    # determine the (x, y) coordinates of the intersection rectangle
    inter_x1s = np.maximum(bboxes1[..., 0], bboxes2[..., 0])  # pairwise max NxM
    inter_y1s = np.maximum(bboxes1[..., 1], bboxes2[..., 1])  # pairwise max NxM
    inter_x2s = np.minimum(bboxes1[..., 2], bboxes2[..., 2])  # pairwise min NxM
    inter_y2s = np.minimum(bboxes1[..., 3], bboxes2[..., 3])  # pairwise min NxM
    inter_ws = np.maximum(
        0.0, inter_x2s - inter_x1s
    )  # pairwise width of intersection rectangle NxM
    inter_hs = np.maximum(
        0.0, inter_y2s - inter_y1s
    )  # pairwise height of intersection rectangle NxM
    inter_areas = inter_ws * inter_hs  # pairwise intersection area NxM
    if ordered:
        # use the box area of bboxes2 as the denominator
        smallest_bb_areas = (bboxes2[..., 2] - bboxes2[..., 0]) * (
            bboxes2[..., 3] - bboxes2[..., 1]
        )
    else:
        smallest_bb_areas = (
            np.minimum(
                (bboxes1[..., 2] - bboxes1[..., 0])
                * (bboxes1[..., 3] - bboxes1[..., 1]),
                (bboxes2[..., 2] - bboxes2[..., 0])
                * (bboxes2[..., 3] - bboxes2[..., 1]),
            )
            + 1e-30
        )  # smallest bb area of each paired box NXM
    return (
        inter_areas / smallest_bb_areas
    )  # pairwise intersection divided by smallest box (overlap) NxM


# a function to return the area of bounding box
def box_area(box: np.array) -> float:
    """
    Args:
        box (numpy array of size (4,) or (4, 1) or (1, 4) or a 4-tuple or a 4-elements list): The box.
    Return the area.
    """
    return (box[3] - box[1]) * (box[2] - box[0])


def show_detections(input_image, predictions, label_map):

    # colors for displaying bounding boxes
    COLORS = [
        (0, 0, 255),
        (255, 0, 0),
        (0, 255, 0),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]

    class_ids = list(label_map.keys())
    if isinstance(input_image, np.ndarray):
        image = input_image.copy()
    else:
        # convert to a numpy array
        image = np.array(input_image)

    # convert to 3-channels
    if len(image.shape) < 3:
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2)

    boxes = predictions["boxes"]
    labels = predictions["labels"]

    for i in range(len(boxes)):
        # the bounding box
        (xtl, ytl, xbr, ybr) = boxes[i].astype(int)
        if labels[i] not in class_ids:
            # use black for incorrect label
            color = (0, 0, 0)
            text = "Unknown label %s" % labels[i]
        else:
            color = COLORS[
                (labels[i] + 1) % len(COLORS)
            ]  # add one to start from 1 for consistency with Mask R-CNN
            text = label_map[labels[i]]

        cv2.rectangle(image, (xtl, ytl), (xbr, ybr), color, 1)
        cv2.putText(
            image, text, (xtl, ytl + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )
    return image


def get_crop_corners(
    image_width: int,
    image_height: int,
    overlap_in_x: int = 80,
    overlap_in_y: int = 80,
    input_size: Tuple[int, int] = (1200, 800),
) -> List[List[int]]:
    """
    A function to get the coordinates of the sub-images/crops given the image dimensions, the model input size
    and the desired overlap between the crops in the x and y dimensions.
    Args:
        image_width (int): Input image width.
        image_height (int): Input image height.
        overlap_in_x (int): The desired overlap between the crops in x (this is the minimum, some crops may have more
            than this number of pixels overlap).
        overlap_in_y (int): The desired overlap between the crops in y.
        input_size (2-tuple of integers): The model input size in (width, height).
    Returns:
        List of 4-tuples or 4-elements integer coordinates of the sub-mages/crops covering the original image.
    """
    # the step size for the starting point of each crop in x and y dimension
    crop_start_step_x: int = input_size[0] - overlap_in_x
    crop_start_step_y: int = input_size[1] - overlap_in_y
    crop_corners: List = []

    # overlapping crops
    for x_start in range(0, image_width - overlap_in_x, crop_start_step_x):
        for y_start in range(0, image_height - overlap_in_y, crop_start_step_y):
            # crop coordinates
            xc_tl = x_start
            yc_tl = y_start
            xc_br = x_start + input_size[0]
            yc_br = y_start + input_size[1]
            # make sure we always crop the image with the given size
            # if we get to the boundaries, extend the crop
            # size inside the image to always get the same size crop
            # this is not really needed, but help with capturing more
            # annotations toward the low/right parts of the image
            if xc_br > image_width:
                xc_br = image_width
                xc_tl = xc_br - input_size[0]
            if yc_br > image_height:
                yc_br = image_height
                yc_tl = yc_br - input_size[1]

            crop_coords = [xc_tl, yc_tl, xc_br, yc_br]
            crop_corners.append(crop_coords)

    return crop_corners


# A utility function
def to_numpy(tensor):
    """
    A function to convert a torch input to numpy array.
    Args:
        tensor (torch tensor).
    Returns:
        Converted to numpy array.
    """
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


# RT-DETR object detection class
class RtDetrObjectDetector:
    def __init__(
        self,
        weights_path: Optional[str] = MODEL_WEIGHTS_PATH,
        label_map: Optional[Dict[int, str]] = None,
        confidence: float = DEFAULT_DETECTION_CONFIDENCE,
    ):

        self.model = None
        self._weights_path: str = str(weights_path)
        self._confidence: float = confidence

        # available device
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # the state dictionary of the model to be read from the weights file
        model_state_dict: OrderedDict = None
        # the model's label map if available in the weight file
        loaded_label_map: Dict[int, str] = None

        self._resize_dict: Union[Dict[Tuple[int, int], Tuple[int, int]], None] = None
        self._crop_corners_dict: Union[Dict[Tuple[int, int], List[List[int]]], None] = (
            None
        )
        self._detected_class_ids_remap: Union[Dict[int, int], None] = None
        self._detected_class_names_remap: Union[Dict[str, str], None] = None
        self._class_ids_to_exclude_from_dets: List[int] = []

        # loading the PyTorch weights and the label map
        try:
            logging.info(
                f"Loading Hugging Face RT-DETR model from from {self._weights_path}. Setting to run on {self.device.type}."
            )

            saved_model_param: Union[OrderedDict, list] = torch.load(
                self._weights_path, map_location=self.device
            )
            if (
                isinstance(saved_model_param, dict)
                and "model_state_dict" in saved_model_param
            ):
                # the weights file contains the model state dictionary (the weights) and the label map (both are
                # mandatory) and potentially other model related configs
                # in case a dictionary is provided, the keys and values are as following:
                # - 'model_state_dict': model state dictionary (mandatory)
                # - 'label_map': label map (mandatory)
                # - 'resize_dict': resize dictionary (optional, for resizing and cropping images during inference)
                # - 'crop_corners_dict': crop corners dictionary (optional, for cropping images during inference)
                # ... other optional fields to be added in future
                model_state_dict = saved_model_param["model_state_dict"]
                if "label_map" in saved_model_param:
                    loaded_label_map = saved_model_param["label_map"]
                else:
                    logging.warning(
                        "The weights file should contain the label map but it missing. "
                        "This should never happen..."
                    )
                    loaded_label_map = None

                if (
                    "resize_dict" in saved_model_param
                    and "crop_corners_dict" in saved_model_param
                ):
                    # resize and crop corners are also provided in the weights file
                    logging.info(
                        "The resize and crop_corners dictionary are also provided in the weights file."
                    )
                    self._resize_dict = saved_model_param["resize_dict"]
                    self._crop_corners_dict = saved_model_param["crop_corners_dict"]

                if "detected_class_names_remap" in saved_model_param:
                    self._detected_class_names_remap = saved_model_param[
                        "detected_class_names_remap"
                    ]
                    logging.info(
                        f"The class names detected by the model will be re-mapped following this mapping: "
                        f"{self._detected_class_names_remap}."
                    )

            else:
                # the file only contains the model state dictionary
                model_state_dict = saved_model_param
                loaded_label_map = None

        except Exception as ex:
            logging.error(
                f"Failed to load Mask2Former model. Likely the paths to model .pt weights "
                f"{self._weights_path} is incorrect: {repr(ex)}."
            )

        if label_map is None:
            if loaded_label_map is None:
                logging.error(
                    "The mapping between the class IDs and class names is required for the model and is "
                    "neither provided during class instantiation nor available in the weights file! Returning ..."
                )
                return
            else:
                logging.info(
                    "Mapping between class IDs and class names is provided in the weights file."
                )
                self._label_map: Dict[int, str] = loaded_label_map
        else:
            logging.info(
                "Mapping between class IDs and class names is passed during class instantiation! "
                "It will overwrite the label map passed in the weights file (if provided)."
            )
            self._label_map: Dict[int, str] = label_map

        logging.info(f"Mapping between class IDs and class names: {self._label_map}")
        self._reverse_label_map: Dict[str, int] = {
            value: key for key, value in self._label_map.items()
        }

        if self._detected_class_names_remap is not None:
            # extract the classes that will be mapped to 'bg' and should be excluded from the detections
            class_names_to_exclude_from_dets = [
                k for k, v in self._detected_class_names_remap.items() if v == "bg"
            ]
            # update the passed mapping and removed the ones that are going to be mapped to 'bg' (should be excluded)
            self._detected_class_names_remap = {
                k: v for k, v in self._detected_class_names_remap.items() if v != "bg"
            }
            self._detected_class_ids_remap: Dict[int, int] = {}
            for k, v in self._detected_class_names_remap.items():
                if k in self._reverse_label_map and v in self._reverse_label_map:
                    self._detected_class_ids_remap[self._reverse_label_map[k]] = (
                        self._reverse_label_map[v]
                    )
            self._class_ids_to_exclude_from_dets: List[int] = [
                self._reverse_label_map[k] for k in class_names_to_exclude_from_dets
            ]
            logging.info(
                f"Mapping between class IDs and class names will be updated according to the following "
                f"mapping: {self._detected_class_names_remap}"
            )
            logging.info(
                f"The following class names will be excluded from detections (mapped to 'bg' in the passed mapping): "
                f"{class_names_to_exclude_from_dets}"
            )

        # RT-DETR model loaded from the original checkpoint
        pre_trained_model_checkpoint: str = "PekingU/rtdetr_r50vd_coco_o365"

        self.model = RTDetrForObjectDetection.from_pretrained(
            pre_trained_model_checkpoint,
            id2label=self._label_map,
            label2id=self._reverse_label_map,
            anchor_image_size=None,
            ignore_mismatched_sizes=True,
        )

        # loading the PyTorch model
        try:
            self.model.load_state_dict(model_state_dict)
            self.model.to(self.device)
            self.model.eval()

        except Exception as ex:
            logging.error(f"Failed to load RT-DETR model: {repr(ex)}.")

        self.hg_preprocessor = RTDetrImageProcessor(
            do_convert_annotations=True,
            do_resize=True,
            size={"width": MODEL_INPUT_SIZE, "height": MODEL_INPUT_SIZE},
            reduce_labels=False,
            do_rescale=True,
            do_normalize=True,
        )
        # added for consistency with the YOLO model
        # metadata will be a dictionary with keys as 'resolution', 'release_date', 'model_type',
        # 'model_name', 'model_extra_info', 'names', 'stride'
        # example: {'resolution': 640,
        #           'release_date': '20240415',
        #           'model_type': 'YOLO Detector',
        #           'model_name': 'YOLOv5m',
        #           'model_extra_info': 'V5 Medium',
        #           'names': {0: 'cell', 1: 'bead'},
        self._metadata = {
            "resolution": MODEL_INPUT_SIZE,
            "release_date": "20250321",
            "model_type": "Transformer Detector",
            "model_name": "RT-DETR",
            "model_extra_info": "None",
            "names": self._label_map,
            "magnification": "10x",
            "predict_masks": False,
        }

    def detect(self, img: np.ndarray, log_time: bool = False) -> Dict[str, list]:
        """
        The main function to detect the bounding box and masks for objects in the input image.

        Args:
            img (numpy array): Input image, should have 8 bits per channel bit-depth (np.uint8 numpy array).
            log_time (bool): A flag to log the model run time.

        Returns:
            A dictionary with keys and values as below:
                "boxes": List of 4-tuples or 4-element lists for the detected objects' bounding boxes in
                    xtl, ytl, xbr, ybr format/order.
                "labels": List of integer class IDs for the detected objects.
                "scores": List of float detection scores, after applying the threshold self._confidence.
        """
        return self.detect_batch(input_images_list=[img], log_time=log_time)[0]

    def detect_batch(
        self, input_images_list: List[np.ndarray], log_time: bool = False
    ) -> List[Dict[str, list]]:
        """
        The main function to detect the bounding box and masks for objects in a list of inputs images (batch processing).

        Args:
            input_images_list (list of numpy arrays): Input images, each should have 8 bits per channel bit-depth (np.uint8 numpy array).
            log_time (bool): A flag to log the model run time.

        Returns:
            A list of dictionaries with keys and values as below:
                "boxes": List of 4-tuples or 4-element lists for the detected objects' bounding boxes in
                    xtl, ytl, xbr, ybr format/order.
                "labels": List of integer class IDs for the detected objects.
                "scores": List of float detection scores, after applying the threshold self._confidence.
        """

        if self.model is None:
            logging.error(
                f"{self._model_name} model has not been initialized. Please initialize the class before detect()."
            )

            out: List[Dict[str, list]] = [
                {
                    "boxes": [],
                    "scores": [],
                    "labels": [],
                }
            ] * len(input_images_list)
            return out

        start: float = time.time()
        # convert to 3-channel images if needed, and store the original image dimensions for
        # post processing
        images_list: List[np.array] = []
        org_img_dims: List[Tuple[int, int]] = []

        for img in input_images_list:
            img_shape: tuple = img.shape
            if len(img_shape) < 3:
                images_list.append(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
            else:
                images_list.append(img)
            org_img_dims.append(img_shape[:2])

        processed_imgs_dict = self.hg_preprocessor(images_list, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(
                pixel_values=processed_imgs_dict["pixel_values"].to(self.device)
            )

            processed_outputs = self.hg_preprocessor.post_process_object_detection(
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
            ] * len(images_list)
        else:
            # move to CPU and convert to numpy arrays before returning
            results = [
                {k: list(to_numpy(v)) for k, v in result.items()}
                for result in processed_outputs
            ]

        # Clear the CUDA cache
        torch.cuda.empty_cache()
        elap: float = time.time() - start
        if log_time:
            logging.info(f"RT-DETR instance segmentation took {elap:.4f} seconds")

        return results

    def _update_label_map_if_needed(self):
        if self._detected_class_names_remap is not None:
            # self._detected_class_ids_remap will also be not None
            updated_label_map: Dict[int, str] = {}
            for k, v in self._label_map.items():
                if k in self._class_ids_to_exclude_from_dets:
                    continue
                if k in self._detected_class_ids_remap:
                    if self._detected_class_ids_remap[k] in updated_label_map:
                        continue
                    else:
                        updated_label_map[self._detected_class_ids_remap[k]] = (
                            self._label_map[self._detected_class_ids_remap[k]]
                        )
                elif k not in updated_label_map:
                    updated_label_map[k] = v
            return updated_label_map

        return self._label_map

    def get_label_map(self):
        return self._update_label_map_if_needed()

    def get_reverse_label_map(self):
        label_map: Dict[int, str] = self._update_label_map_if_needed()
        return {v: k for k, v in label_map.items()}

    def get_class_names(self):
        return list(self.get_reverse_label_map().keys())

    def get_cropping_info(self):
        return self._resize_dict, self._crop_corners_dict

    def get_metadata(self):
        return self._metadata

    def detect_by_cropping(
        self,
        img: Union[Image.Image, np.ndarray],
        crop_corners: List[List[int]],
        nms_threshold_for_combining_crop_results: float = 0.15,
        classnames_to_return: Optional[List[str]] = None,
        log_time=False,
    ) -> Dict[str, List]:
        """
        A function to apply the model on a high resolution image. If the
        image is high resolution with many objects, after resizing the image
        to match the models input sizes, the objects may become too small for
        reliable detection. This function evaluate an image by first cropping
        the image into smaller and potentially overlapping sub-images (as
        specified by cropCorners), running the detector on each sub-image and
        then combining the detections over multiple overlapping sub-images
        by applying NMS
        Args:
            img (PIL.Image or numpy array): Input image, should have 8 bits per channel
                bit-depth (np.uint8 in case of a numpy array).
            crop_corners (list of 4-tuples (x1, y1, x2, y2)): Each element of
                this list specifies a cropped sub-image of the input image with
                top-left corner (x1, y1) and bottom-right corner (x2, y2). All
                the coordinates should be with respect to input image sizes.
                The input image is divided into len(crop_corners) sub-images
                before running the model on each.
            nms_threshold_for_combining_crop_results (float): NMS threshold to be used
                for combining detections of cropped sub-images over the overlapping areas.
            classnames_to_return (optional list of string): The class names to return. If
                None is passed (default), all classes detected will be returned.
            log_time (bool): A flag to print the runtime of the function.
        Returns:
            A dictionary with keys and values as below:
                "boxes": List of 4-tuples or 4-element lists for the detected objects' bounding boxes in
                    xtl, ytl, xbr, ybr format/order.
                "labels": List of integer class IDs for the detected objects.
                "scores": List of float detection scores, after applying the threshold self._confidence.
                "masks": List of masks for the detected objects. Each mask is a numpy array of the same size
                    as the bounding box with width and height (xbr - xtl, ybr - ytl).
        """

        start = time.time()

        invalid_out: dict = {"boxes": [], "scores": [], "labels": []}

        if len(crop_corners) == 0:
            logging.error(
                "No crop corners are provided for running YOLOv5 model on sub-images. "
                "Returning no detections ..."
            )
            return invalid_out

        if classnames_to_return is not None:
            class_ids_to_return: List[int] = []
            for class_name in classnames_to_return:
                if class_name not in self._reverse_label_map:
                    logging.warning(
                        f"Classname '{class_name}' is not included in the model label map! Skipping this name ..."
                    )
                else:
                    class_ids_to_return.append(self._reverse_label_map[class_name])

            if len(class_ids_to_return) == 0:
                # nothing to return
                logging.warning(
                    f"No valid classnames was provided in classnames_to_return: {classnames_to_return}!"
                    f"Returning no detections"
                )
                return invalid_out
        else:
            # this list will not be used and is not really needed, define it just in case
            class_ids_to_return: List[int] = list(self._label_map.keys())

        if isinstance(img, Image.Image):
            W, H = img.size
        else:
            H, W = img.shape[:2]

        # check if all the crop sub-images are of the same size,
        # if not, make them equal size for batch processing
        crop_widths: List[int] = [min(c[2], W) - max(c[0], 0) for c in crop_corners]
        crop_heights: List[int] = [min(c[3], H) - max(c[1], 0) for c in crop_corners]

        if min(crop_widths) <= 0 or min(crop_heights) <= 0:
            logging.error(
                "Incorrect corners are provided for running Mask RCNN model on sub-images. "
                "Returning no detections ..."
            )
            return invalid_out

        crop_width: int = max(crop_widths)
        crop_height: int = max(crop_heights)

        # combine the results, filter them based on the score,
        # and update the coordinates of the bounding boxes
        # for applying NMS later
        results: Dict = {
            "scores": [],
            "boxes": [],
            "labels": [],
        }

        # a list to keep track of cropped sub-images with at least one object detection
        crop_ids_with_detection: List[int] = []

        for crop_id, corners in enumerate(crop_corners):
            (x1c, y1c, x2c, y2c) = corners
            # enlarge the crop if necessary to make all the same size
            x2c = x1c + crop_width
            y2c = y1c + crop_height

            # crop the image and run the model
            if isinstance(img, Image.Image):
                # convert to numpy and crop
                cropped_image = np.array(img.crop((x1c, y1c, x2c, y2c)))
            else:
                cropped_image = img[y1c:y2c, x1c:x2c]

            preds = self.detect(cropped_image)

            boxes: np.ndarray = np.array(preds["boxes"])
            labels: np.ndarray = np.array(preds["labels"])
            scores: np.ndarray = np.array(preds["scores"])

            # combine the detection results
            if len(scores) == 0:
                continue

            # find the bounding boxes close to the boundaries of the cropped image
            # these boxes most probably are truncated (because they are close to the boundary)
            # for a properly designed image crops, the overlapping section (in x or y dimensions)
            # between two adjacent crops is larger than the largest object (in each dimension)
            # hence an object can only cross one boundary of an overlapping part and will
            # definitely lie completely in another cropped image
            # modify the score for these detected boxes (assign the minimum score of self._confidence)
            # to give them lower priority during NMS when the results in the overlapping parts
            # of cropped images are combined

            scores[
                (boxes[:, 0] < 4)
                | (boxes[:, 1] < 4)
                | (boxes[:, 2] > crop_width - 4)
                | (boxes[:, 3] > crop_height - 4)
            ] = self._confidence

            crop_ids_with_detection.append(crop_id)
            results["scores"].append(scores)
            results["boxes"].append(boxes + np.array([x1c, y1c, x1c, y1c], dtype=int))
            results["labels"].append(labels)

        # no object detected, return
        if len(crop_ids_with_detection) == 0:
            return invalid_out

        # list to contain the detections
        boxes: list = []
        labels: list = []
        scores: list = []

        # now compare the detection results of one crop with the detections in the
        # rest of the image to identify objects that are uniquely detected in the
        # crop and should be kept
        # this part is needed to pick one object in the overlapping crop areas when
        # detected by the detector in multiple crops
        # note we need to compare the detections in one crop only with overlapping
        # crops; but here we are doing it for all for simplicity of implementation

        # to decide which object to keep when detected in multiple crops, we compute
        # the IoU (for objects of the same class) between the crop under consideration and the rest
        # then we keep objects in the crop under consideration that have
        # IoU <= nms_threshold_for_combining_crop_results with objects detected in the rest
        # we also keep objects with IoU > nms_threshold_for_combining_crop_results, if the detection
        # score for the object in the crop is higher than the objects in the other crops
        for idx, crop_id in enumerate(crop_ids_with_detection):
            crop_labels: np.ndarray = results["labels"][idx]
            crop_scores: np.ndarray = results["scores"][idx]
            crop_boxes: np.ndarray = results["boxes"][idx]

            num_detections_in_rest = 0
            for i in range(len(crop_ids_with_detection)):
                if i != idx:
                    num_detections_in_rest += len(results["labels"][i])

            if num_detections_in_rest == 0:
                # keep all detections in this crop
                for i, label in enumerate(crop_labels):
                    boxes.append(crop_boxes[i])
                    labels.append(label)
                    scores.append(crop_scores[i])
                continue

            # labels for detections in other cropped sub-images
            rest_labels: np.array = np.hstack(
                [
                    results["labels"][i]
                    for i in range(len(crop_ids_with_detection))
                    if i != idx
                ]
            )
            # scores for detections in other cropped sub-images
            rest_scores: np.array = np.hstack(
                [
                    results["scores"][i]
                    for i in range(len(crop_ids_with_detection))
                    if i != idx
                ]
            )
            # boxes for detections in other cropped sub-images
            rest_boxes: np.array = np.vstack(
                [
                    results["boxes"][i]
                    for i in range(len(crop_ids_with_detection))
                    if i != idx
                ]
            )

            for label in set(list(crop_labels) + list(rest_labels)):
                # the indexes of detections of the same label in each crop and rest set
                crop_class_idxs: np.array = np.where(crop_labels == label)
                rest_class_idxs: np.array = np.where(rest_labels == label)

                if len(crop_class_idxs[0]) == 0:
                    continue

                if len(rest_class_idxs[0]) == 0:
                    # keep all detections for this label in this crop
                    idx_to_keep: np.array = np.array(
                        [i for i in range(len(crop_class_idxs[0]))], dtype=int
                    )
                else:
                    # compute the IoU matrix, use torchvision implementation for efficiency
                    iou_matrix: np.array = iou_batch(
                        crop_boxes[crop_class_idxs], rest_boxes[rest_class_idxs]
                    )

                    # keep the bounding boxes from the crop that have
                    # 1) IoU <= nms_threshold_for_combining_crop_results with the boxes in the rest
                    #    of the crops, or
                    # 2) IoU > nms_threshold_for_combining_crop_results and the detection score for
                    #    the box in the crop is higher than the detection scores for the boxes in the rest

                    # IoU <= nms_threshold_for_combining_crop_results
                    idx_to_keep = np.where(
                        np.max(iou_matrix, axis=1)
                        <= nms_threshold_for_combining_crop_results
                    )[0]

                for i in range(len(crop_class_idxs[0])):
                    # detection score of this object
                    crop_det_score = crop_scores[crop_class_idxs[0][i]]
                    if i in idx_to_keep:
                        boxes.append(crop_boxes[crop_class_idxs[0][i]])
                        labels.append(crop_labels[crop_class_idxs[0][i]])
                        scores.append(crop_det_score)
                    else:
                        # there is some boxes in the rest of the crops with IoU more than the threshold
                        # find the maximum detection scores among the objects with IoU more than the threshold
                        # also find the area of the bounding box for that detection (needed to break a tie in case
                        # scores are equal)
                        high_iou_idxs = np.where(
                            iou_matrix[i] > nms_threshold_for_combining_crop_results
                        )[0]
                        scores_to_check = [
                            rest_scores[rest_class_idxs[0][j]] for j in high_iou_idxs
                        ]
                        max_index = np.argmax(scores_to_check)
                        rest_det_score = scores_to_check[max_index]

                        # areas of the matching boxes
                        crop_box_area: float = box_area(
                            crop_boxes[crop_class_idxs[0][i]]
                        )
                        rest_box_area: float = box_area(
                            rest_boxes[rest_class_idxs[0][high_iou_idxs[max_index]]]
                        )

                        # in case of a tie, pick the box with a larger area
                        # a tie can happen specially when the objects are both near a common boundary (e.g., a side of
                        # the image) of the crops (we reduce the scores for both to the threshold score and they become
                        # equal)
                        if crop_det_score > rest_det_score or (
                            crop_det_score == rest_det_score
                            and crop_box_area >= rest_box_area
                        ):
                            # keep this object as it has the highest score among all
                            boxes.append(crop_boxes[crop_class_idxs[0][i]])
                            labels.append(crop_labels[crop_class_idxs[0][i]])
                            scores.append(crop_det_score)
                            if crop_det_score == rest_det_score:
                                # this is added to break the tie if both areas are equal
                                # so we will not add the same box twice when considering
                                # in another crop
                                results["scores"][idx][crop_class_idxs[0][i]] += 1e-5

        # filter the detections
        if classnames_to_return is not None:
            detection_ids: List[int] = []
            for idx, label in enumerate(labels):
                if label not in class_ids_to_return:
                    continue
                detection_ids.append(idx)

            boxes: list = [boxes[i] for i in detection_ids]
            labels: list = [labels[i] for i in detection_ids]
            scores: list = [scores[i] for i in detection_ids]

        elap: float = time.time() - start
        if log_time:
            logging.info(
                "RT-DETR object detection after cropping the image to "
                "{} sub-images took {:.4f} seconds".format(len(crop_corners), elap)
            )

        out: dict = {
            "boxes": [
                [box[0], box[1], box[2], box[3]] for box in boxes
            ],  # convert to list
            "scores": scores,
            "labels": labels,
        }

        return out


detector = RtDetrObjectDetector(weights_path=MODEL_WEIGHTS_PATH)

RESIZE: Final[Dict[Tuple[int, int, str], Tuple[int, int]]] = {
    (2000, 1600, "10x"): (1000, 800),
    (4512, 4512, "10x"): (2440, 2440),
    (4512, 4512, "4x"): (4512, 4512),
}
# A dictionary with keys as the input (original) image size (width, height, magnification)
# tuple and values as the list of coordinates (xtl, ytl, xbr, ybr) of sub-images/crops
# to run YOLOv5 on each
# note that the crop coordinates are with respect to resized image dimensions specified above
CROP_CORNERS: Final[Dict[Tuple[int, int, str], List[List[int]]]] = {
    (2000, 1600, "10x"): [
        [0, 0, 640, 640],
        [0, 160, 640, 800],
        [360, 0, 1000, 640],
        [360, 160, 1000, 800],
    ],
    (4512, 4512, "10x"): [
        [0, 0, 640, 640],
        [0, 600, 640, 1240],
        [0, 1200, 640, 1840],
        [0, 1800, 640, 2440],
        [600, 0, 1240, 640],
        [600, 600, 1240, 1240],
        [600, 1200, 1240, 1840],
        [600, 1800, 1240, 2440],
        [1200, 0, 1840, 640],
        [1200, 600, 1840, 1240],
        [1200, 1200, 1840, 1840],
        [1200, 1800, 1840, 2440],
        [1800, 0, 2440, 640],
        [1800, 600, 2440, 1240],
        [1800, 1200, 2440, 1840],
        [1800, 1800, 2440, 2440],
    ],
    (4512, 4512, "4x"): [
        [0, 0, 640, 640],
        [0, 560, 640, 1200],
        [0, 1120, 640, 1760],
        [0, 1680, 640, 2320],
        [0, 2240, 640, 2880],
        [0, 2800, 640, 3440],
        [0, 3360, 640, 4000],
        [0, 3872, 640, 4512],
        [560, 0, 1200, 640],
        [560, 560, 1200, 1200],
        [560, 1120, 1200, 1760],
        [560, 1680, 1200, 2320],
        [560, 2240, 1200, 2880],
        [560, 2800, 1200, 3440],
        [560, 3360, 1200, 4000],
        [560, 3872, 1200, 4512],
        [1120, 0, 1760, 640],
        [1120, 560, 1760, 1200],
        [1120, 1120, 1760, 1760],
        [1120, 1680, 1760, 2320],
        [1120, 2240, 1760, 2880],
        [1120, 2800, 1760, 3440],
        [1120, 3360, 1760, 4000],
        [1120, 3872, 1760, 4512],
        [1680, 0, 2320, 640],
        [1680, 560, 2320, 1200],
        [1680, 1120, 2320, 1760],
        [1680, 1680, 2320, 2320],
        [1680, 2240, 2320, 2880],
        [1680, 2800, 2320, 3440],
        [1680, 3360, 2320, 4000],
        [1680, 3872, 2320, 4512],
        [2240, 0, 2880, 640],
        [2240, 560, 2880, 1200],
        [2240, 1120, 2880, 1760],
        [2240, 1680, 2880, 2320],
        [2240, 2240, 2880, 2880],
        [2240, 2800, 2880, 3440],
        [2240, 3360, 2880, 4000],
        [2240, 3872, 2880, 4512],
        [2800, 0, 3440, 640],
        [2800, 560, 3440, 1200],
        [2800, 1120, 3440, 1760],
        [2800, 1680, 3440, 2320],
        [2800, 2240, 3440, 2880],
        [2800, 2800, 3440, 3440],
        [2800, 3360, 3440, 4000],
        [2800, 3872, 3440, 4512],
        [3360, 0, 4000, 640],
        [3360, 560, 4000, 1200],
        [3360, 1120, 4000, 1760],
        [3360, 1680, 4000, 2320],
        [3360, 2240, 4000, 2880],
        [3360, 2800, 4000, 3440],
        [3360, 3360, 4000, 4000],
        [3360, 3872, 4000, 4512],
        [3872, 0, 4512, 640],
        [3872, 560, 4512, 1200],
        [3872, 1120, 4512, 1760],
        [3872, 1680, 4512, 2320],
        [3872, 2240, 4512, 2880],
        [3872, 2800, 4512, 3440],
        [3872, 3360, 4512, 4000],
        [3872, 3872, 4512, 4512],
    ],
}


def run_rt_detr(
    input_image: np.ndarray,
    bit_depth: int = 12,
    normalize_image: bool = True,
    is_4x: bool = False,
    detector: RtDetrObjectDetector = detector,
) -> Tuple[np.array, np.array, np.array, float]:

    # make a copy to not modify the input image
    img = input_image.copy()

    if len(img.shape) > 2:
        print(
            "Warning RT-DETR model may suffer loss in precision due to conversion from RGB to grayscale"
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = (255 * img.astype(float) / (2**bit_depth - 1)).astype(np.uint8)

    if normalize_image:
        img = cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    image_height, image_width = img.shape[:2]

    # check if the model includes the resize and crop_corners dictionaries
    resize_dict: Union[Dict[Tuple[int, int], Tuple[int, int]], None] = None
    crop_corners_dict: Union[Dict[Tuple[int, int], List[List[int]]], None] = None
    try:
        resize_dict, crop_corners_dict = detector.get_cropping_info()
    except Exception as ex:
        # the model class is old and not supporting the get_cropping_info()
        logging.warning(
            f"Warning Mask R-CNN class does not implement get_cropping_info(): {repr(ex)}"
        )

    if resize_dict is None or crop_corners_dict is None:
        # use the default values set if not provided in the model
        resize_dict = RESIZE
        crop_corners_dict = CROP_CORNERS

    if is_4x:
        resize_dict_key: Tuple[int, int, str] = (image_width, image_height, "4x")
    else:
        resize_dict_key: Tuple[int, int, str] = (image_width, image_height, "10x")

    if resize_dict_key not in resize_dict:
        logging.error(
            f"The input image size {(image_width, image_height)} is not supported for {'4x' if is_4x else '10x'}! Returning no cells!"
        )
        return (
            np.zeros((0, 4), dtype=int),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=float),
            0,
        )

    # we keep the aspect ratio in RESIZE dictionary, scale_factor is the same for both dimensions

    scale_factor: float = image_width / resize_dict[resize_dict_key][0]
    resized_img: np.ndarray = cv2.resize(
        img, resize_dict[resize_dict_key], interpolation=cv2.INTER_AREA
    )

    crop_corners: List[List[int]] = crop_corners_dict[resize_dict_key]

    st = time.time()

    out = detector.detect_by_cropping(resized_img, crop_corners)
    boxes: np.ndarray = np.zeros((0, 4), dtype=int)
    labels: np.ndarray = np.zeros((0,), dtype=int)
    scores: np.ndarray = np.zeros((0,), dtype=float)

    if len(out["labels"]) > 0:
        boxes, labels, scores = (
            np.array(out["boxes"]),
            np.array(out["labels"]),
            np.array(out["scores"]),
        )

    # scale the detections back to original image resolution
    boxes = (scale_factor * boxes).astype(int)

    et = time.time()

    return boxes, labels, scores, et - st


class RTDeTRObjectDetector(VisionModel):
    def __init__(
        self,
        weights_path: str,
        model_name: str = "RT-DETR",
        label_map: Optional[Dict[int, str]] = None,  # to be read from the weights file
        model_input_size: Optional[
            Tuple[int, int]
        ] = None,  # to be read from the weights file
        confidence: float = DEFAULT_DETECTION_CONFIDENCE,
        device: torch.device = torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
        backbone_name_str: str = None,
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

        if self._backbone is None:
            # RT-DETR model loaded from the original checkpoint
            pre_trained_model_checkpoint: str = "PekingU/rtdetr_r50vd_coco_o365"
            self.model = RTDetrForObjectDetection.from_pretrained(
                pre_trained_model_checkpoint,
                id2label=self._label_map,
                label2id=self._reverse_label_map,
                anchor_image_size=None,
                ignore_mismatched_sizes=True,
            )
            # self.hg_preprocessor = RTDetrImageProcessor.from_pretrained(pre_trained_model_checkpoint)
            # overide the preprocessor settings to match our needs
            self.hg_preprocessor = RTDetrImageProcessor(
                do_convert_annotations=True,
                do_resize=True,
                size={
                    "width": self._model_input_size[0],
                    "height": self._model_input_size[1],
                },
                reduce_labels=False,
                do_rescale=True,
                do_normalize=True,
            )
        elif self._backbone.lower() == "dinov2":
            # monkey patch to load RT-DETR with DINOv2 backbone
            dinov2_backbone = Dinov2BackBoneWithFPN.from_pretrained(
                "facebook/dinov2-base",
                # first_layer_dims = (48, 48),
                output_indices_for_fpn=[4, 8, 12],
            )
            pre_trained_model_checkpoint: str = "PekingU/rtdetr_v2_r18vd"
            self.model = RTDetrV2ForObjectDetection.from_pretrained(
                pre_trained_model_checkpoint,
                id2label=self._label_map,
                label2id=self._reverse_label_map,
                ignore_mismatched_sizes=True,
            )
            self.model.model.backbone = dinov2_backbone
            # self.hg_preprocessor = RTDetrImageProcessor.from_pretrained(pre_trained_model_checkpoint)
            # overide the preprocessor settings to match our needs
            self.hg_preprocessor = RTDetrImageProcessor(
                do_convert_annotations=True,
                do_resize=True,
                size={
                    "width": self._model_input_size[0],
                    "height": self._model_input_size[1],
                },
                reduce_labels=False,
                do_rescale=True,
                do_normalize=True,
            )

        # import pdb; pdb.set_trace()

        # load the model states
        try:
            self.model.load_state_dict(self._model_state_dict)
            self.model.to(self._device)
            self.model.eval()
        except Exception as ex:
            logging.error(f"Failed to load {self._model_name} model: {repr(ex)}.")
        self._metadata = {
            "predict_masks": False,  # detector model
            "resolution": self._model_input_size[0],  # square input
            "release_date": os.path.basename(self._weights_path).split("_")[
                0
            ],  # the model name starts with the release date
            "model_type": "Transformer Detector",
            "model_name": self._model_name,
            "model_extra_info": (
                "Original backbone"
                if self._backbone == None
                else (
                    "With DINOv2 backbone" if self._backbone.lower() == "dinov2" else ""
                )
            ),
            "names": self._label_map,
            "magnification": "4x"
            if "4x" in os.path.basename(self._weights_path)
            else "10x",  # 4x should be specified in the name
        }

    def detect(self, img: np.ndarray, log_time: bool = False) -> Dict[str, list]:
        """
        The main function to detect the bounding box and masks for objects in the input image.

        Args:
            img (numpy array): Input image, should have 8 bits per channel bit-depth (np.uint8 numpy array).
            log_time (bool): A flag to log the model run time.

        Returns:
            A dictionary with keys and values as below:
                "boxes": List of 4-tuples or 4-element lists for the detected objects' bounding boxes in
                    xtl, ytl, xbr, ybr format/order.
                "labels": List of integer class IDs for the detected objects.
                "scores": List of float detection scores, after applying the threshold self._confidence.
        """
        return self.detect_batch(input_images_list=[img], log_time=log_time)[0]

    def detect_batch(
        self, input_images_list: List[np.ndarray], log_time: bool = False
    ) -> List[Dict[str, list]]:
        """
        The main function to detect the bounding box and masks for objects in a list of inputs images (batch processing).

        Args:
            input_images_list (list of numpy arrays): Input images, each should have 8 bits per channel bit-depth (np.uint8 numpy array).
            log_time (bool): A flag to log the model run time.

        Returns:
            A list of dictionaries with keys and values as below:
                "boxes": List of 4-tuples or 4-element lists for the detected objects' bounding boxes in
                    xtl, ytl, xbr, ybr format/order.
                "labels": List of integer class IDs for the detected objects.
                "scores": List of float detection scores, after applying the threshold self._confidence.
        """

        if self.model is None:
            logging.error(
                f"{self._model_name} model has not been initialized. Please initialize the class before detect()."
            )

            out: List[Dict[str, list]] = [
                {
                    "boxes": [],
                    "scores": [],
                    "labels": [],
                }
            ] * len(input_images_list)
            return out

        start: float = time.time()
        # convert to 3-channel images if needed, and store the original image dimensions for
        # post processing
        images_list: List[np.array] = []
        org_img_dims: List[Tuple[int, int]] = []

        for img in input_images_list:
            img_shape: tuple = img.shape
            if len(img_shape) < 3:
                images_list.append(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
            else:
                images_list.append(img)
            org_img_dims.append(img_shape[:2])

        processed_imgs_dict = self.hg_preprocessor(images_list, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(
                pixel_values=processed_imgs_dict["pixel_values"].to(self._device)
            )

        results = self.postprocess(outputs, org_img_dims)
        # Clear the CUDA cache
        torch.cuda.empty_cache()
        elap: float = time.time() - start
        if log_time:
            logging.info(f"RT-DETR  took {elap:.4f} seconds")

        return results

    def _update_label_map_if_needed(self):
        if self._detected_class_names_remap is not None:
            # self._detected_class_ids_remap will also be not None
            updated_label_map: Dict[int, str] = {}
            for k, v in self._label_map.items():
                if k in self._class_ids_to_exclude_from_dets:
                    continue
                if k in self._detected_class_ids_remap:
                    if self._detected_class_ids_remap[k] in updated_label_map:
                        continue
                    else:
                        updated_label_map[self._detected_class_ids_remap[k]] = (
                            self._label_map[self._detected_class_ids_remap[k]]
                        )
                elif k not in updated_label_map:
                    updated_label_map[k] = v
            return updated_label_map

        return self._label_map

    def get_label_map(self):
        return self._update_label_map_if_needed()

    def get_reverse_label_map(self):
        label_map: Dict[int, str] = self._update_label_map_if_needed()
        return {v: k for k, v in label_map.items()}

    def get_class_names(self):
        return list(self.get_reverse_label_map().keys())

    def get_cropping_info(self):
        return self._resize_dict, self._crop_corners_dict

    def get_metadata(self):
        return self._metadata

    def detect_by_cropping(
        self,
        img: Union[Image.Image, np.ndarray],
        crop_corners: List[List[int]],
        nms_threshold_for_combining_crop_results: float = 0.15,
        classnames_to_return: Optional[List[str]] = None,
        log_time=False,
    ) -> Dict[str, List]:
        """
        A function to apply the model on a high resolution image. If the
        image is high resolution with many objects, after resizing the image
        to match the models input sizes, the objects may become too small for
        reliable detection. This function evaluate an image by first cropping
        the image into smaller and potentially overlapping sub-images (as
        specified by cropCorners), running the detector on each sub-image and
        then combining the detections over multiple overlapping sub-images
        by applying NMS
        Args:
            img (PIL.Image or numpy array): Input image, should have 8 bits per channel
                bit-depth (np.uint8 in case of a numpy array).
            crop_corners (list of 4-tuples (x1, y1, x2, y2)): Each element of
                this list specifies a cropped sub-image of the input image with
                top-left corner (x1, y1) and bottom-right corner (x2, y2). All
                the coordinates should be with respect to input image sizes.
                The input image is divided into len(crop_corners) sub-images
                before running the model on each.
            nms_threshold_for_combining_crop_results (float): NMS threshold to be used
                for combining detections of cropped sub-images over the overlapping areas.
            classnames_to_return (optional list of string): The class names to return. If
                None is passed (default), all classes detected will be returned.
            log_time (bool): A flag to print the runtime of the function.
        Returns:
            A dictionary with keys and values as below:
                "boxes": List of 4-tuples or 4-element lists for the detected objects' bounding boxes in
                    xtl, ytl, xbr, ybr format/order.
                "labels": List of integer class IDs for the detected objects.
                "scores": List of float detection scores, after applying the threshold self._confidence.
                "masks": List of masks for the detected objects. Each mask is a numpy array of the same size
                    as the bounding box with width and height (xbr - xtl, ybr - ytl).
        """

        start = time.time()

        invalid_out: dict = {"boxes": [], "scores": [], "labels": []}

        if len(crop_corners) == 0:
            logging.error(
                "No crop corners are provided for running YOLOv5 model on sub-images. "
                "Returning no detections ..."
            )
            return invalid_out

        if classnames_to_return is not None:
            class_ids_to_return: List[int] = []
            for class_name in classnames_to_return:
                if class_name not in self._reverse_label_map:
                    logging.warning(
                        f"Classname '{class_name}' is not included in the model label map! Skipping this name ..."
                    )
                else:
                    class_ids_to_return.append(self._reverse_label_map[class_name])

            if len(class_ids_to_return) == 0:
                # nothing to return
                logging.warning(
                    f"No valid classnames was provided in classnames_to_return: {classnames_to_return}!"
                    f"Returning no detections"
                )
                return invalid_out
        else:
            # this list will not be used and is not really needed, define it just in case
            class_ids_to_return: List[int] = list(self._label_map.keys())

        if isinstance(img, Image.Image):
            W, H = img.size
        else:
            H, W = img.shape[:2]

        # check if all the crop sub-images are of the same size,
        # if not, make them equal size for batch processing
        crop_widths: List[int] = [min(c[2], W) - max(c[0], 0) for c in crop_corners]
        crop_heights: List[int] = [min(c[3], H) - max(c[1], 0) for c in crop_corners]

        if min(crop_widths) <= 0 or min(crop_heights) <= 0:
            logging.error(
                "Incorrect corners are provided for running Mask RCNN model on sub-images. "
                "Returning no detections ..."
            )
            return invalid_out

        crop_width: int = max(crop_widths)
        crop_height: int = max(crop_heights)

        # combine the results, filter them based on the score,
        # and update the coordinates of the bounding boxes
        # for applying NMS later
        results: Dict = {
            "scores": [],
            "boxes": [],
            "labels": [],
        }

        # a list to keep track of cropped sub-images with at least one object detection
        crop_ids_with_detection: List[int] = []

        for crop_id, corners in enumerate(crop_corners):
            (x1c, y1c, x2c, y2c) = corners
            # enlarge the crop if necessary to make all the same size
            x2c = x1c + crop_width
            y2c = y1c + crop_height

            # crop the image and run the model
            if isinstance(img, Image.Image):
                # convert to numpy and crop
                cropped_image = np.array(img.crop((x1c, y1c, x2c, y2c)))
            else:
                cropped_image = img[y1c:y2c, x1c:x2c]

            preds = self.detect(cropped_image)

            boxes: np.ndarray = np.array(preds["boxes"])
            labels: np.ndarray = np.array(preds["labels"])
            scores: np.ndarray = np.array(preds["scores"])

            # combine the detection results
            if len(scores) == 0:
                continue

            # find the bounding boxes close to the boundaries of the cropped image
            # these boxes most probably are truncated (because they are close to the boundary)
            # for a properly designed image crops, the overlapping section (in x or y dimensions)
            # between two adjacent crops is larger than the largest object (in each dimension)
            # hence an object can only cross one boundary of an overlapping part and will
            # definitely lie completely in another cropped image
            # modify the score for these detected boxes (assign the minimum score of self._confidence)
            # to give them lower priority during NMS when the results in the overlapping parts
            # of cropped images are combined

            scores[
                (boxes[:, 0] < 4)
                | (boxes[:, 1] < 4)
                | (boxes[:, 2] > crop_width - 4)
                | (boxes[:, 3] > crop_height - 4)
            ] = self._confidence

            crop_ids_with_detection.append(crop_id)
            results["scores"].append(scores)
            results["boxes"].append(boxes + np.array([x1c, y1c, x1c, y1c], dtype=int))
            results["labels"].append(labels)

        # no object detected, return
        if len(crop_ids_with_detection) == 0:
            return invalid_out

        # list to contain the detections
        boxes: list = []
        labels: list = []
        scores: list = []

        # now compare the detection results of one crop with the detections in the
        # rest of the image to identify objects that are uniquely detected in the
        # crop and should be kept
        # this part is needed to pick one object in the overlapping crop areas when
        # detected by the detector in multiple crops
        # note we need to compare the detections in one crop only with overlapping
        # crops; but here we are doing it for all for simplicity of implementation

        # to decide which object to keep when detected in multiple crops, we compute
        # the IoU (for objects of the same class) between the crop under consideration and the rest
        # then we keep objects in the crop under consideration that have
        # IoU <= nms_threshold_for_combining_crop_results with objects detected in the rest
        # we also keep objects with IoU > nms_threshold_for_combining_crop_results, if the detection
        # score for the object in the crop is higher than the objects in the other crops
        for idx, crop_id in enumerate(crop_ids_with_detection):
            crop_labels: np.ndarray = results["labels"][idx]
            crop_scores: np.ndarray = results["scores"][idx]
            crop_boxes: np.ndarray = results["boxes"][idx]

            num_detections_in_rest = 0
            for i in range(len(crop_ids_with_detection)):
                if i != idx:
                    num_detections_in_rest += len(results["labels"][i])

            if num_detections_in_rest == 0:
                # keep all detections in this crop
                for i, label in enumerate(crop_labels):
                    boxes.append(crop_boxes[i])
                    labels.append(label)
                    scores.append(crop_scores[i])
                continue

            # labels for detections in other cropped sub-images
            rest_labels: np.array = np.hstack(
                [
                    results["labels"][i]
                    for i in range(len(crop_ids_with_detection))
                    if i != idx
                ]
            )
            # scores for detections in other cropped sub-images
            rest_scores: np.array = np.hstack(
                [
                    results["scores"][i]
                    for i in range(len(crop_ids_with_detection))
                    if i != idx
                ]
            )
            # boxes for detections in other cropped sub-images
            rest_boxes: np.array = np.vstack(
                [
                    results["boxes"][i]
                    for i in range(len(crop_ids_with_detection))
                    if i != idx
                ]
            )

            for label in set(list(crop_labels) + list(rest_labels)):
                # the indexes of detections of the same label in each crop and rest set
                crop_class_idxs: np.array = np.where(crop_labels == label)
                rest_class_idxs: np.array = np.where(rest_labels == label)

                if len(crop_class_idxs[0]) == 0:
                    continue

                if len(rest_class_idxs[0]) == 0:
                    # keep all detections for this label in this crop
                    idx_to_keep: np.array = np.array(
                        [i for i in range(len(crop_class_idxs[0]))], dtype=int
                    )
                else:
                    # compute the IoU matrix, use torchvision implementation for efficiency
                    iou_matrix: np.array = iou_batch(
                        crop_boxes[crop_class_idxs], rest_boxes[rest_class_idxs]
                    )

                    # keep the bounding boxes from the crop that have
                    # 1) IoU <= nms_threshold_for_combining_crop_results with the boxes in the rest
                    #    of the crops, or
                    # 2) IoU > nms_threshold_for_combining_crop_results and the detection score for
                    #    the box in the crop is higher than the detection scores for the boxes in the rest

                    # IoU <= nms_threshold_for_combining_crop_results
                    idx_to_keep = np.where(
                        np.max(iou_matrix, axis=1)
                        <= nms_threshold_for_combining_crop_results
                    )[0]

                for i in range(len(crop_class_idxs[0])):
                    # detection score of this object
                    crop_det_score = crop_scores[crop_class_idxs[0][i]]
                    if i in idx_to_keep:
                        boxes.append(crop_boxes[crop_class_idxs[0][i]])
                        labels.append(crop_labels[crop_class_idxs[0][i]])
                        scores.append(crop_det_score)
                    else:
                        # there is some boxes in the rest of the crops with IoU more than the threshold
                        # find the maximum detection scores among the objects with IoU more than the threshold
                        # also find the area of the bounding box for that detection (needed to break a tie in case
                        # scores are equal)
                        high_iou_idxs = np.where(
                            iou_matrix[i] > nms_threshold_for_combining_crop_results
                        )[0]
                        scores_to_check = [
                            rest_scores[rest_class_idxs[0][j]] for j in high_iou_idxs
                        ]
                        max_index = np.argmax(scores_to_check)
                        rest_det_score = scores_to_check[max_index]

                        # areas of the matching boxes
                        crop_box_area: float = box_area(
                            crop_boxes[crop_class_idxs[0][i]]
                        )
                        rest_box_area: float = box_area(
                            rest_boxes[rest_class_idxs[0][high_iou_idxs[max_index]]]
                        )

                        # in case of a tie, pick the box with a larger area
                        # a tie can happen specially when the objects are both near a common boundary (e.g., a side of
                        # the image) of the crops (we reduce the scores for both to the threshold score and they become
                        # equal)
                        if crop_det_score > rest_det_score or (
                            crop_det_score == rest_det_score
                            and crop_box_area >= rest_box_area
                        ):
                            # keep this object as it has the highest score among all
                            boxes.append(crop_boxes[crop_class_idxs[0][i]])
                            labels.append(crop_labels[crop_class_idxs[0][i]])
                            scores.append(crop_det_score)
                            if crop_det_score == rest_det_score:
                                # this is added to break the tie if both areas are equal
                                # so we will not add the same box twice when considering
                                # in another crop
                                results["scores"][idx][crop_class_idxs[0][i]] += 1e-5

        # filter the detections
        if classnames_to_return is not None:
            detection_ids: List[int] = []
            for idx, label in enumerate(labels):
                if label not in class_ids_to_return:
                    continue
                detection_ids.append(idx)

            boxes: list = [boxes[i] for i in detection_ids]
            labels: list = [labels[i] for i in detection_ids]
            scores: list = [scores[i] for i in detection_ids]

        elap: float = time.time() - start
        if log_time:
            logging.info(
                "RT-DETR object detection after cropping the image to "
                "{} sub-images took {:.4f} seconds".format(len(crop_corners), elap)
            )

        out: dict = {
            "boxes": [
                [box[0], box[1], box[2], box[3]] for box in boxes
            ],  # convert to list
            "scores": scores,
            "labels": labels,
        }

        return out

    def postprocess(self, outputs, org_img_dims):
        processed_outputs = self.hg_preprocessor.post_process_object_detection(
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

        return results
