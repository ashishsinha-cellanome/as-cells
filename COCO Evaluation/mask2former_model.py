import os
import time
import logging
from typing import Tuple, List, Final, Optional, Dict, Union
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.transforms import functional as F
from transformers import (
    Dinov2Config,
    Dinov2Model,
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)

BASE_PATH: Final[str] = "/home/cellareye/Cellanome/dl-mehdi/Mask RCNN/checkpoints"
MODEL_WEIGHTS_PATH: Final[str] = (
    "/home/cellareye/Cellanome/dl-mehdi/Mask RCNN/checkpoints/20250312_mask2former_sets_1_2_3_6_to_41_0p1_bbox_0p7_1_rs_0p25_blur_2_bs_8_epochs_1cl_lrs.pt"
)
DEFAULT_DETECTION_CONFIDENCE: Final[float] = 0.5
MODEL_INPUT_SIZE: Final[Tuple[int, int]] = (1022, 798)
TRANSFORM_MEAN: Final[List[float]] = [0.485, 0.456, 0.406]
TRANSFORM_STD: Final[List[float]] = [0.229, 0.224, 0.225]


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


def iou_mask_pair(
    box1: np.ndarray, mask1: np.ndarray, box2: np.ndarray, mask2: np.ndarray
) -> float:
    """
    Given two np.uint8 M1xN1 and M2xN2 numpy arrays (mask1, mask2) for two object masks and
    two (4,) (4-element) integer numpy arrays of the top-left/bottom-right corners of the
    bounding boxes around these objects (box1, box2), the code returns the IoU between the masks of the two objects.
    The passed masks should be defined within the passed bounding boxes, and should have
    values set to 1 for the object. The bounding boxes should be passed in xtl, ytl, xbr, ybr order, e.g.,
    if [xtl, ytl, xbr, ybr] are the passed integer values of the top-left and bottle-right corner of the bounding box
    for an object, the passed masks should be a numpy array of type np.uint8 and size (ybr - ytl, xbr - xtl)
    with values set to 1 for the object.
    """
    # union of the box coordinates, make sure the coordinates are integers
    xtl_1, ytl_1, xbr_1, ybr_1 = box1.astype(int)
    xtl_2, ytl_2, xbr_2, ybr_2 = box2.astype(int)

    xtl: int = min(xtl_1, xtl_2)
    ytl: int = min(ytl_1, ytl_2)
    xbr: int = max(xbr_1, xbr_2)
    ybr: int = max(ybr_1, ybr_2)
    union_mask_1: np.ndarray = np.zeros((ybr - ytl, xbr - xtl), np.uint8)
    union_mask_1[(ytl_1 - ytl) : (ybr_1 - ytl), (xtl_1 - xtl) : (xbr_1 - xtl)] = mask1
    union_mask_2: np.ndarray = np.zeros((ybr - ytl, xbr - xtl), np.uint8)
    union_mask_2[(ytl_2 - ytl) : (ybr_2 - ytl), (xtl_2 - xtl) : (xbr_2 - xtl)] = mask2
    union: int = cv2.bitwise_or(union_mask_1, union_mask_2).sum()
    intersection: int = cv2.bitwise_and(union_mask_1, union_mask_2).sum()

    return intersection / (float(union) + 1e-30)


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
    masks = predictions["masks"]

    for i in range(len(masks)):
        # the bounding box
        (xtl, ytl, xbr, ybr) = boxes[i]
        if labels[i] not in class_ids:
            # use black for incorrect label
            color = (0, 0, 0)
            text = "Unknown label %s" % labels[i]
        else:
            color = COLORS[labels[i] % len(COLORS)]
            text = label_map[labels[i]]

        color_mask = color * np.repeat(np.expand_dims(masks[i], axis=2), 3, axis=2)
        blended = 0.4 * color_mask
        blended[color_mask == 0] = image[ytl:ybr, xtl:xbr][color_mask == 0]
        blended[color_mask > 0] += 0.6 * image[ytl:ybr, xtl:xbr][color_mask > 0]

        # store the blended ROI in the original image
        image[ytl:ybr, xtl:xbr] = blended.astype(np.uint8)
        # add the bounding box with yellow color
        color = (255, 255, 0)
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


class Dinov2WithSFP(torch.nn.Module):
    """
    Simple Feature Pyramid (SFP) adapter to convert DINOv2's flat stride-14
    feature maps into a multi-scale FPN (stride ~4, 8, 16, 32) expected by Mask2Former.
    """

    def __init__(self, original_encoder):
        super().__init__()
        self.original_encoder = original_encoder
        self.channels = original_encoder.channels

        # DINOv2 channels (e.g., 768)
        c = self.channels[0]

        # Stride 4 (Upscale 4x from stride 14)
        self.fpn1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(c, c, kernel_size=2, stride=2),
            torch.nn.SyncBatchNorm(c)
            if torch.cuda.device_count() > 1
            else torch.nn.BatchNorm2d(c),
            torch.nn.GELU(),
            torch.nn.ConvTranspose2d(c, c, kernel_size=2, stride=2),
        )
        # Stride 8 (Upscale 2x from stride 14)
        self.fpn2 = torch.nn.ConvTranspose2d(c, c, kernel_size=2, stride=2)
        # Stride 16 (Identity ~ Stride 14)
        self.fpn3 = torch.nn.Identity()
        # Stride 32 (Downscale 2x from stride 14)
        self.fpn4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, pixel_values):
        outputs = self.original_encoder(pixel_values)
        feats = outputs.feature_maps

        f1 = self.fpn1(feats[0])
        f2 = self.fpn2(feats[1])
        f3 = self.fpn3(feats[2])
        f4 = self.fpn4(feats[3])

        outputs.feature_maps = (f1, f2, f3, f4)
        return outputs


def get_mask2former_instance_segmentation_model_with_dinov2_backbone(
    id2label: Dict[int, str], model_type: str, with_registers: bool
):

    # transformer layer outputs to use
    output_indices_map: Dict[str, List[int]] = {
        "small": [6, 8, 10, 12],
        "base": [6, 8, 10, 12],
        "large": [18, 20, 22, 24],
        "giant": [34, 36, 38, 40],
    }

    if model_type.lower() in output_indices_map.keys():
        if with_registers:
            dinov2_checkpoint_str: str = "dinov2-with-registers-" + model_type.lower()
        else:
            dinov2_checkpoint_str: str = "dinov2-" + model_type.lower()

        output_indices: List[int] = output_indices_map[model_type.lower()]
    else:
        dinov2_checkpoint_str: str = "dinov2-base"
        output_indices: List[int] = output_indices_map["base"]
        print(
            f"[ERROR] Incorrect model type passed {model_type}! Using the base model by default."
        )

    # store Dinov2 weights locally to reload them again, only do it if already not loaded locally
    if not os.path.exists(os.path.join(BASE_PATH, dinov2_checkpoint_str + ".pth")):
        dinov2_model = Dinov2Model.from_pretrained(
            "facebook/" + dinov2_checkpoint_str, out_indices=output_indices
        )
        torch.save(
            dinov2_model.state_dict(),
            os.path.join(BASE_PATH, dinov2_checkpoint_str + ".pth"),
        )

    # create Mask2Former config for semantic segmentation with Dinov2 backbone

    mask2former_checkpoint = "facebook/mask2former-swin-large-coco-instance"

    model_config = Mask2FormerConfig.from_pretrained(mask2former_checkpoint)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        mask2former_checkpoint, id2label=id2label, ignore_mismatched_sizes=True
    )
    model_config = model.config
    model_config.backbone_config = Dinov2Config.from_pretrained(
        "facebook/" + dinov2_checkpoint_str, out_indices=output_indices
    )

    # instantiate Mask2Former model with Dinov2 backbone (random weights)
    model = Mask2FormerForUniversalSegmentation(model_config)

    # load Dinov2 weights into Mask2Former backbone
    dinov2_backbone = model.model.pixel_level_module.encoder
    dinov2_backbone.load_state_dict(
        torch.load(os.path.join(BASE_PATH, dinov2_checkpoint_str + ".pth"))
    )

    # freeze all the weights in Dinov2 backbone
    # for param in dinov2_backbone.parameters():
    #     param.requires_grad_(False)

    # this is for freezing the backbone in Mask2Former, it should be the same as above
    for param in model.model.pixel_level_module.encoder.parameters():
        param.requires_grad_(False)

    model.model.pixel_level_module.encoder = Dinov2WithSFP(
        model.model.pixel_level_module.encoder
    )

    return model


# Mask2Former instance segmentation class
class Mask2FormerInstanceSegmentation:
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
                f"Loading Hugging Face Mask2Former model from from {self._weights_path}. Setting to run on {self.device.type}."
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
                        f"The weights file should contain the label map but it missing. "
                        f"This should never happen..."
                    )
                    loaded_label_map = None

                if (
                    "resize_dict" in saved_model_param
                    and "crop_corners_dict" in saved_model_param
                ):
                    # resize and crop corners are also provided in the weights file
                    logging.info(
                        f"The resize and crop_corners dictionary are also provided in the weights file."
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
                    f"The mapping between the class IDs and class names is required for the model and is "
                    f"neither provided during class instantiation nor available in the weights file! Returning ..."
                )
                return
            else:
                logging.info(
                    f"Mapping between class IDs and class names is provided in the weights file."
                )
                self._label_map: Dict[int, str] = loaded_label_map
        else:
            logging.info(
                f"Mapping between class IDs and class names is passed during class instantiation! "
                f"It will overwrite the label map passed in the weights file (if provided)."
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

        # Mask2Former model
        self.model = get_mask2former_instance_segmentation_model_with_dinov2_backbone(
            id2label=self._label_map, model_type="base", with_registers=False
        )

        # loading the PyTorch model
        try:
            self.model.load_state_dict(model_state_dict)
            self.model.to(self.device)
            self.model.eval()

        except Exception as ex:
            logging.error(f"Failed to load Mask2Former model: {repr(ex)}.")

        self.hg_preprocessor = Mask2FormerImageProcessor(
            ignore_index=-1,
            do_resize=True,
            size=MODEL_INPUT_SIZE,
            size_divisor=14,
            reduce_labels=False,
            do_rescale=True,
            image_mean=TRANSFORM_MEAN,
            image_std=TRANSFORM_STD,
            do_normalize=True,
        )

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
                "masks": List of masks for the detected objects. Each mask is a numpy array of the same size
                    as the bounding box with width and height (xbr - xtl, ybr - ytl).
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
                "masks": List of masks for the detected objects. Each mask is a numpy array of the same size
                    as the bounding box with width and height (xbr - xtl, ybr - ytl).
        """

        if self.model is None:
            logging.error(
                "Mask2Former model has not been initialized. Please initialize the class before detect()."
            )

            out: List[Dict[str, list]] = [
                {"boxes": [], "scores": [], "labels": [], "masks": []}
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
            processed_outputs = self.hg_preprocessor.post_process_instance_segmentation(
                outputs, target_sizes=org_img_dims, return_binary_maps=True
            )

        if len(processed_outputs) == 0:
            elap: float = time.time() - start
            if log_time:
                logging.info(
                    f"Mask2Former instance segmentation took {elap:.4f} seconds"
                )
            # this should not happen and is not expected, return as if the model has not detected anything (for the whole list of images)
            return [{"boxes": [], "labels": [], "scores": [], "masks": []}] * len(
                images_list
            )

        results = []

        for sample_index, processed_output in enumerate(processed_outputs):
            sample_dict = {}
            instance_to_label_map = {
                segment["id"]: segment["label_id"]
                for segment in processed_output["segments_info"]
            }
            instance_to_score_map = {
                segment["id"]: segment["score"]
                for segment in processed_output["segments_info"]
            }
            sorted_instance_ids = sorted(instance_to_label_map.keys())

            if len(sorted_instance_ids) > 0:
                # processed_output['segmentation'] is of dimension num_detections x H x W
                num_instances = processed_output["segmentation"].shape[0]

                if num_instances != len(instance_to_label_map):
                    print(
                        f"[WARN]: # of instance masks {num_instances} is not equal to the number of labels {len(instance_to_label_map)}!"
                    )
                    sorted_instance_ids = [
                        i for i in sorted_instance_ids if i < num_instances
                    ]

                # masks should be of dimension num_detections x 1 x H x W
                sample_dict["labels"] = [
                    instance_to_label_map[i] for i in sorted_instance_ids
                ]
                sample_dict["scores"] = [
                    instance_to_score_map[i] for i in sorted_instance_ids
                ]
                sample_dict["masks"] = to_numpy(
                    processed_output["segmentation"][sorted_instance_ids]
                )
                boxes = []
                masks = []  # masks after restricting them to the size of the bounding box
                for i in range(sample_dict["masks"].shape[0]):
                    pos = np.where(sample_dict["masks"][i])
                    xtl = pos[1].min()
                    xbr = pos[1].max()
                    ytl = pos[0].min()
                    ybr = pos[0].max()
                    boxes.append([xtl, ytl, xbr, ybr])
                    masks.append(
                        sample_dict["masks"][i, ytl:ybr, xtl:xbr].astype(float)
                    )

                sample_dict["boxes"] = boxes
                sample_dict["masks"] = masks

                # remap the output class IDs if needed
                if self._detected_class_ids_remap is not None and len(out["labels"]):
                    sample_dict["labels"] = np.vectorize(
                        lambda x: (
                            self._detected_class_ids_remap[x]
                            if x in self._detected_class_ids_remap
                            else x
                        )
                    )(sample_dict["labels"])

            else:
                sample_dict = {"boxes": [], "labels": [], "scores": [], "masks": []}

            results.append(sample_dict)

        # Clear the CUDA cache
        torch.cuda.empty_cache()
        elap: float = time.time() - start
        if log_time:
            logging.info(f"Mask2Former instance segmentation took {elap:.4f} seconds")

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

        invalid_out: dict = {"boxes": [], "scores": [], "labels": [], "masks": []}

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
            "masks": [],
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
            # we leave the masks as a list of mask numpy arrays as each mask has different sizes below
            masks: List[np.ndarray] = preds["masks"]

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
            results["masks"].append(masks)

        # no object detected, return
        if len(crop_ids_with_detection) == 0:
            return invalid_out

        # list to contain the detections
        boxes: list = []
        labels: list = []
        scores: list = []
        masks: list = []

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
            crop_masks: List[np.ndarray] = results["masks"][idx]

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
                    masks.append(crop_masks[i])
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
                        masks.append(crop_masks[crop_class_idxs[0][i]])
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
                            masks.append(crop_masks[crop_class_idxs[0][i]])
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
            masks: list = [masks[i] for i in detection_ids]

        elap: float = time.time() - start
        if log_time:
            logging.info(
                "Mask2Former instance segmentation after cropping the image to "
                "{} sub-images took {:.4f} seconds".format(len(crop_corners), elap)
            )

        out: dict = {
            "boxes": [
                [box[0], box[1], box[2], box[3]] for box in boxes
            ],  # convert to list
            "scores": scores,
            "labels": labels,
            "masks": masks,
        }

        return out


detector = Mask2FormerInstanceSegmentation(weights_path=MODEL_WEIGHTS_PATH)

# A dictionary with keys as the input (original) image size (width, height) tuple and
# values as the list of coordinates (xtl, ytl, xbr, ybr) of sub-images/crops
# to run YOLOv5 on each
# note that the crop coordinates are with respect to resized image dimensions specified above
RESIZE: Final[Dict[Tuple[int, int], Tuple[int, int]]] = {
    (2000, 1600): (1280, 1024),
    (4512, 4512): (2148, 2148),
}
# A dictionary with keys as the input (original) image size (width, height) tuple and
# values as the list of coordinates (xtl, ytl, xbr, ybr) of sub-images/crops
# to run YOLOv5 on each
# note that the crop coordinates are with respect to resized image dimensions specified above
CROP_CORNERS: Final[Dict[Tuple[int, int], List[List[int]]]] = {
    (2000, 1600): [[0, 0, 800, 1024], [480, 0, 1280, 1024]],
    (4512, 4512): [
        [0, 0, 1022, 798],
        [0, 450, 1022, 1248],
        [0, 900, 1022, 1698],
        [0, 1350, 1022, 2148],
        [563, 0, 1585, 798],
        [563, 450, 1585, 1248],
        [563, 900, 1585, 1698],
        [563, 1350, 1585, 2148],
        [1126, 0, 2148, 798],
        [1126, 450, 2148, 1248],
        [1126, 900, 2148, 1698],
        [1126, 1350, 2148, 2148],
    ],
}

# threshold to apply on masks
# threshold for post-processing cells and remove the ones consisting of multiple smaller cells
OVER_LAP_THRESHOLD: Final[float] = 0.75


def run_mask2former(
    input_image: np.ndarray,
    normalize_image: bool = False,
    bit_depth: int = 8,
    crop: bool = True,
    classnames_mapping_dict=None,
    post_process_class_names: List[str] = list(detector.get_label_map().values()),
    plot_results: bool = False,
    detector=detector,
) -> Tuple[Dict[str, list], float, Optional[np.ndarray]]:
    # make a copy to not modify the input image
    img = input_image.copy()

    if len(img.shape) > 2:
        logging.warning(
            "Warning Mask2Former model may suffer loss in precision due to conversion from RGB to grayscale"
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

    if crop and (image_width, image_height) not in resize_dict:
        logging.error(
            "The input image size {} is not supported! Returning no cells!".format(
                image_width, image_height
            )
        )
        out = {
            "boxes": np.zeros((0, 4), dtype=int),
            "labels": np.zeros((0,), dtype=int),
            "scores": np.zeros((0,), dtype=float),
            "masks": [],
        }
        if plot_results:
            return (out, 0, np.zeros((image_height, image_width), dtype=np.uint8))
        else:
            return (out, 0)

    if crop:
        # we keep the aspect ratio in RESIZE dictionary, scale_factor is the same for both dimensions
        scale_factor: float = image_width / resize_dict[(image_width, image_height)][0]
        resized_width, resized_height = resize_dict[(image_width, image_height)]
    else:
        larger_side_size: int = max(image_width, image_height)
        smaller_side_size: int = min(image_width, image_height)
        scale_factor: float = max(smaller_side_size / 800.0, larger_side_size / 1333.0)
        # do not scale the image if the larger and the smaller sides are smaller than the model input sizes (1333, 800)
        scale_factor = max(1, scale_factor)
        resized_width, resized_height = (
            int(image_width / scale_factor),
            int(image_height / scale_factor),
        )

    if scale_factor != 1:
        resized_img: np.ndarray = cv2.resize(
            img, (resized_width, resized_height), interpolation=cv2.INTER_AREA
        )
    else:
        resized_img: np.ndarray = img

    st = time.time()

    if crop:
        crop_corners: List[List[int]] = crop_corners_dict[(image_width, image_height)]
        out: Dict[str, list] = detector.detect_by_cropping(
            img=resized_img,
            crop_corners=crop_corners,
        )
    else:
        out: Dict[str, list] = detector.detect(img=resized_img)

    if scale_factor != 1:
        # scale the detections back to original image resolution
        out["boxes"] = (scale_factor * np.array(out["boxes"])).astype(int)
        # convert to a list to be consistent with the rest
        out["boxes"] = [box for box in out["boxes"]]
    else:
        out["boxes"] = [np.array(box) for box in out["boxes"]]

    for idx in range(len(out["boxes"])):
        if scale_factor != 1:
            xtl, ytl, xbr, ybr = out["boxes"][idx]
            # note that mask here is not a probability mask, but a binary mask
            out["masks"][idx] = cv2.resize(
                out["masks"][idx],
                (xbr - xtl, ybr - ytl),
                interpolation=cv2.INTER_NEAREST,
            )

    if classnames_mapping_dict is not None:
        classnames_to_exclude: List[str] = [
            name
            for name, mapped_name in classnames_mapping_dict.items()
            if mapped_name == "bg"
        ]
        class_ids_to_exclude: List[int] = [
            detector._reverse_label_map[name] for name in classnames_to_exclude
        ]
        class_ids_mapping_dict = {
            detector._reverse_label_map[name]: detector._reverse_label_map[mapped_name]
            for name, mapped_name in classnames_mapping_dict.items()
            if mapped_name != "bg"
        }

        labels: List[int] = []
        idxs_to_keep: List[int] = []

        for idx, label in enumerate(out["labels"]):
            if label in class_ids_to_exclude:
                continue
            if label in class_ids_mapping_dict:
                labels.append(class_ids_mapping_dict[label])
            else:
                labels.append(label)
            idxs_to_keep.append(idx)

        out["boxes"] = [
            box for idx, box in enumerate(out["boxes"]) if idx in idxs_to_keep
        ]
        out["labels"] = labels
        out["scores"] = [
            score for idx, score in enumerate(out["scores"]) if idx in idxs_to_keep
        ]
        out["masks"] = [
            mask for idx, mask in enumerate(out["masks"]) if idx in idxs_to_keep
        ]

    # post-process the results
    # in the following, "larger" objects that consist of a number of already detected smaller objects of the same type are invalidated
    # this can happen mainly for 'cell', 'nucleus' and 'cell-adhered'/'cytoplasm' classes
    # list of indexes of objects for each class name to be included in post processing
    post_process_class_idxs: Dict[str, List[int]] = {}
    # list of bounding boxes for each class name to be included in post processing
    post_process_class_boxes: Dict[str, List[np.ndarray]] = {}
    for i, box in enumerate(out["boxes"]):
        for class_name in post_process_class_names:
            if (
                class_name in detector._reverse_label_map
                and out["labels"][i] == detector._reverse_label_map[class_name]
            ):
                if class_name in post_process_class_idxs:
                    post_process_class_idxs[class_name].append(i)
                    post_process_class_boxes[class_name].append(box)
                else:
                    post_process_class_idxs[class_name] = [i]
                    post_process_class_boxes[class_name] = [box]

    # list of detection indexes to be excluded (this is with respect to all detected objects and not only the class under consideration)
    obj_idxs_to_remove: List[int] = []
    for key in post_process_class_boxes:
        # convert to a numpy array
        post_process_class_boxes[key]: np.ndarray = np.array(
            post_process_class_boxes[key]
        )

        if len(post_process_class_idxs[key]) == 0:
            continue

        overlap: np.ndarray = overlap_batch(
            post_process_class_boxes[key], post_process_class_boxes[key], True
        )
        # remove diagonal elements (as each box has a complete overlap with itself)
        overlap = overlap - np.eye(len(post_process_class_boxes[key]))
        # index of larger objects (row indexes) covering some smaller already detected cells (column index)
        # by more than OVER_LAP_THRESHOLD
        # these smaller objects are most probably redundant objects
        covering_obj_idxs, covered_obj_idxs = np.where(overlap > OVER_LAP_THRESHOLD)
        # now double-check the coverage using the masks
        for i, j in zip(covering_obj_idxs, covered_obj_idxs):
            large_obj_index: int = post_process_class_idxs[key][i]
            small_obj_index: int = post_process_class_idxs[key][j]
            # larger box coordinates
            xl1, yl1, xl2, yl2 = out["boxes"][large_obj_index]
            # smaller box coordinates
            xs1, ys1, xs2, ys2 = out["boxes"][small_obj_index]
            # union of the two boxes
            x1: int = min(xl1, xs1)
            y1: int = min(yl1, ys1)
            x2: int = max(xl2, xs2)
            y2: int = max(yl2, ys2)
            large_obj_mask: np.ndarray = np.zeros((y2 - y1, x2 - x1), np.uint8)
            large_obj_mask[(yl1 - y1) : (yl2 - y1), (xl1 - x1) : (xl2 - x1)] = out[
                "masks"
            ][large_obj_index]
            small_obj_mask: np.ndarray = np.zeros((y2 - y1, x2 - x1), np.uint8)
            small_obj_mask[(ys1 - y1) : (ys2 - y1), (xs1 - x1) : (xs2 - x1)] = out[
                "masks"
            ][small_obj_index]

            if np.sum(small_obj_mask * large_obj_mask) > OVER_LAP_THRESHOLD * np.sum(
                small_obj_mask
            ):
                # add row index i to the list of object indexes to be removed
                if small_obj_index not in obj_idxs_to_remove:
                    obj_idxs_to_remove.append(small_obj_index)

    if len(obj_idxs_to_remove) > 0:
        out["boxes"] = [
            box for i, box in enumerate(out["boxes"]) if i not in obj_idxs_to_remove
        ]
        out["labels"] = [
            label
            for i, label in enumerate(out["labels"])
            if i not in obj_idxs_to_remove
        ]
        out["scores"] = [
            score
            for i, score in enumerate(out["scores"])
            if i not in obj_idxs_to_remove
        ]
        out["masks"] = [
            mask for i, mask in enumerate(out["masks"]) if i not in obj_idxs_to_remove
        ]

    et = time.time()

    if plot_results:
        return out, et - st, show_detections(img, out, detector._label_map)

    return out, et - st


# a more generic function to do post-processing
# this function accepts a list of "object type" groups (defined as a list of classnames) and applies post-processing on all
# objects belonging to the same type group
# a type group should contains objects that the model may confuse but in practice, differentiating them is not important;
# clearly a type group can be defined as a single class
def post_process_detections(
    detections: Dict[str, np.ndarray],
    post_process_type_groups: List[List[str]],
    min_diam_in_pixels_dict: Dict[str, int],
    label_map: Dict[int, str],
):
    # post-process the results
    # in the following, "larger" objects that consist of a number of already detected smaller objects from the same "type" group are
    # invalidated
    # list of indexes of objects for each object type group to be included in post processing
    reverse_label_map: Dict[str, int] = {v: k for k, v in label_map.items()}
    post_process_class_idxs: Dict[str, List[int]] = {}
    # list of bounding boxes for each each object type group to be included in post processing
    post_process_class_boxes: Dict[str, List[np.ndarray]] = {}
    # list of detection indexes to be excluded (this is with respect to all detected objects and not only the class under consideration)
    obj_idxs_to_remove: List[int] = []
    for i, box in enumerate(detections["boxes"]):
        # filter small objects
        class_id = detections["labels"][i]
        if class_id not in label_map:
            # this should never happen
            obj_idxs_to_remove.append(i)
            continue
        if (
            label_map[class_id] in min_diam_in_pixels_dict
            and max(box[3] - box[1], box[2] - box[0])
            < min_diam_in_pixels_dict[label_map[class_id]]
        ):
            obj_idxs_to_remove.append(i)
            continue
        for type_group_class_names in post_process_type_groups:
            # an identifier for the type group formed by concatenating all the class names in the group
            type_group_id: str = "_".join(type_group_class_names)
            for class_name in type_group_class_names:
                if (
                    class_name in reverse_label_map
                    and detections["labels"][i] == reverse_label_map[class_name]
                ):
                    if type_group_id in post_process_class_idxs:
                        post_process_class_idxs[type_group_id].append(i)
                        post_process_class_boxes[type_group_id].append(box)
                    else:
                        post_process_class_idxs[type_group_id] = [i]
                        post_process_class_boxes[type_group_id] = [box]

    for key in post_process_class_boxes:
        if len(post_process_class_idxs[key]) == 0:
            continue
        # convert to a numpy array
        post_process_class_boxes[key]: np.ndarray = np.array(
            post_process_class_boxes[key]
        )

        overlap: np.ndarray = overlap_batch(
            post_process_class_boxes[key], post_process_class_boxes[key], True
        )
        # remove diagonal elements (as each box has a complete overlap with itself)
        overlap = overlap - np.eye(len(post_process_class_boxes[key]))
        # index of larger objects (row indexes) covering some smaller already detected cells (column index)
        # by more than OVER_LAP_THRESHOLD
        # these smaller objects are most probably redundant objects
        covering_obj_idxs, covered_obj_idxs = np.where(overlap > OVER_LAP_THRESHOLD)
        # now double-check the coverage using the masks
        for i, j in zip(covering_obj_idxs, covered_obj_idxs):
            large_obj_index: int = post_process_class_idxs[key][i]
            small_obj_index: int = post_process_class_idxs[key][j]
            # larger box coordinates
            xl1, yl1, xl2, yl2 = detections["boxes"][large_obj_index]
            # smaller box coordinates
            xs1, ys1, xs2, ys2 = detections["boxes"][small_obj_index]
            # union of the two boxes
            x1: int = min(xl1, xs1)
            y1: int = min(yl1, ys1)
            x2: int = max(xl2, xs2)
            y2: int = max(yl2, ys2)
            large_obj_mask: np.ndarray = np.zeros((y2 - y1, x2 - x1), np.uint8)
            large_obj_mask[(yl1 - y1) : (yl2 - y1), (xl1 - x1) : (xl2 - x1)] = (
                detections["masks"][large_obj_index]
            )
            small_obj_mask: np.ndarray = np.zeros((y2 - y1, x2 - x1), np.uint8)
            small_obj_mask[(ys1 - y1) : (ys2 - y1), (xs1 - x1) : (xs2 - x1)] = (
                detections["masks"][small_obj_index]
            )

            if np.sum(small_obj_mask * large_obj_mask) > OVER_LAP_THRESHOLD * np.sum(
                small_obj_mask
            ):
                # add row index i to the list of object indexes to be removed
                if small_obj_index not in obj_idxs_to_remove:
                    obj_idxs_to_remove.append(small_obj_index)

    if len(obj_idxs_to_remove) > 0:
        detections["boxes"] = [
            box
            for i, box in enumerate(detections["boxes"])
            if i not in obj_idxs_to_remove
        ]
        detections["labels"] = [
            label
            for i, label in enumerate(detections["labels"])
            if i not in obj_idxs_to_remove
        ]
        detections["scores"] = [
            score
            for i, score in enumerate(detections["scores"])
            if i not in obj_idxs_to_remove
        ]
        detections["masks"] = [
            mask
            for i, mask in enumerate(detections["masks"])
            if i not in obj_idxs_to_remove
        ]
        if "features" in detections:
            detections["features"] = [
                box_features
                for i, box_features in enumerate(detections["features"])
                if i not in obj_idxs_to_remove
            ]

    return detections
