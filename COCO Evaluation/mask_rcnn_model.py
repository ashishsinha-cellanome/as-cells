import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import os
import time
import logging
from typing import Tuple, List, Final, Optional, Dict, Union

from torchvision.transforms import functional as F

# MODEL_WEIGHTS_PATH: Final[str] = '/home/cellareye/Cellanome/dl-mehdi/Mask RCNN/checkpoints/analysis_caging_cells_set_2_crop_2_0p1_bbox_0p8_2_random_scale_2_bs_8_epochs.pt'
# MODEL_WEIGHTS_PATH: Final[str] = '/home/cellareye/Cellanome/dl-mehdi/Mask RCNN/checkpoints/cell_bead_cage_nucl_mix_crop_0p1_bbox_0p7_1_rs_0p25_blur_2_bs_8_epochs.pt'
# MODEL_WEIGHTS_PATH: Final[str] = '/home/cellareye/Cellanome/dl-mehdi/Mask RCNN/checkpoints/cell_bead_cage_nucl_cyto_mix_crop_0p1_bbox_0p7_1_rs_0p25_blur_2_bs_8_epochs_2.pt'
# MODEL_WEIGHTS_PATH: Final[str] = '/home/cellareye/Cellanome/dl-mehdi/Mask RCNN/checkpoints/cell_bead_cage_nucl_cyto_hela_mix_crop_0p1_bbox_0p7_1_rs_0p25_blur_2_bs_8_epochs_1cl_lrs.pt'
MODEL_WEIGHTS_PATH: Final[str] = '/home/cellareye/Cellanome/dl-mehdi/Mask RCNN/checkpoints/sets_1_2_3_4_5_6_7_8_9_0p1_bbox_0p7_1_rs_0p25_blur_2_bs_8_epochs_1cl_lrs.pt'
# MODEL_WEIGHTS_PATH: Final[str] = '/home/cellareye/Cellanome/dl-mehdi/Mask RCNN/checkpoints/nuclei_bf_crop_2_0p1_bbox_0p8_1_rs_0p25_blur_2_bs_14_epochs.pt'
# MODEL_WEIGHTS_PATH: Final[str] = '/home/cellareye/Cellanome/dl-mehdi/Mask RCNN/checkpoints/nucl_cage_bf_crop_3_0p1_bbox_0p6_1_rs_0p25_blur_2_bs_14_epochs.pt'

DEFAULT_ANCHOR_SIZES: Tuple[Tuple[int]] = ((12,), (24,), (36,), (48,), (60,))
DEFAULT_DETECTION_CONFIDENCE: Final[float] = 0.5
DEFAULT_MASK_THRESHOLD_FOR_BBOX_EXPANSION: Final[float] = 0.1
DEFAULT_BBOX_EXPANSION_FACTOR: Final[float] = 0.2
MAX_BBOX_EXPANSION_FACTOR: Final[float] = 0.25



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

def overlap_batch(bboxes1: np.ndarray, bboxes2: np.ndarray, ordered: bool = False) -> np.ndarray:
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
    inter_ws = np.maximum(0., inter_x2s - inter_x1s)  # pairwise width of intersection rectangle NxM
    inter_hs = np.maximum(0., inter_y2s - inter_y1s)  # pairwise height of intersection rectangle NxM
    inter_areas = inter_ws * inter_hs  # pairwise intersection area NxM
    if ordered:
        # use the box area of bboxes2 as the denominator
        smallest_bb_areas = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    else:
        smallest_bb_areas = (np.minimum((bboxes1[..., 2] - bboxes1[..., 0])
                                        * (bboxes1[..., 3] - bboxes1[..., 1]),
                                        (bboxes2[..., 2] - bboxes2[..., 0])
                                        * (bboxes2[..., 3] - bboxes2[..., 1]))
                             + 1e-30)  # smallest bb area of each paired box NXM
    return inter_areas / smallest_bb_areas  # pairwise intersection divided by smallest box (overlap) NxM


def iou_mask_pair(box1: np.ndarray, mask1: np.ndarray, box2: np.ndarray, mask2: np.ndarray) -> float:
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
    union_mask_1[(ytl_1 - ytl):(ybr_1 - ytl), (xtl_1 - xtl):(xbr_1 - xtl)] = mask1
    union_mask_2: np.ndarray = np.zeros((ybr - ytl, xbr - xtl), np.uint8)
    union_mask_2[(ytl_2 - ytl):(ybr_2 - ytl), (xtl_2 - xtl):(xbr_2 - xtl)] = mask2
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
    COLORS = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    class_ids = list(label_map.keys())
    if isinstance(input_image, np.ndarray):
        image = input_image.copy()
    else:
        # convert to a numpy array
        image = np.array(input_image)
    
    # convert to 3-channels
    if len(image.shape) < 3:
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2)
    
    boxes = predictions['boxes']
    labels = predictions['labels']
    masks = predictions['masks']

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


def get_instance_segmentation_model(
    num_classes: int = 2,
    anchor_sizes: Tuple[Tuple[int]] = DEFAULT_ANCHOR_SIZES,
) -> torchvision.models.detection.mask_rcnn.MaskRCNN:
    """
    A function to return a Mask R-CNN model for training
    a Resent50 backbone. The backbone can be modified.
    The backbone, RoI pooling, anchor generator and classifier
    layers are redefined/customized for this network.

    Args:
        num_classes (integer): Number of object classes for detection (add +1 for background).
        anchor_sizes (Tuple[Tuple[int]]): Anchor sizes for each feature map (1, 0.5 and 2 is used for
        aspect ratios)
    Returns:
        A Mask R-CNN model for detection of num_classes objects
        (and their bounding boxes and masks).
    """
    # load an instance segmentation model pre-trained on COCO
    # (Resnet50 backbone with FPN, there are other options available )
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=torchvision.models.detection.mask_rcnn.MaskRCNN_ResNet50_FPN_Weights.COCO_V1
    )

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one for the given number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    # specify the anchors per spatial location for RPN
    # 3 anchors with the same size and 3 different aspect
    # ratios for each feature map
    # the format Tuple[Tuple[int]] for anchor_sizes and aspect_ratios is because each feature
    # map could potentially have different sizes and
    # aspect ratios
    # Note: If a different backbone is used, the anchor_generator should be updated
    # as the number of elements in anchor_sizes and aspect_ratios should both be equal to the
    # number of feature maps
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes, aspect_ratios=((0.5, 1.0, 2.0),) * len(anchor_sizes)
    )
    # update the anchor generator
    model.rpn.anchor_generator = anchor_generator

    # increase the number of proposals to keep before applying NMS and after
    # applying NMS during training and testing
    # we target for 500 cells in an image, so we need to make sure
    # enough region proposals are considered specially during testing/eval
    # (default values for both pre and post are 2000 and 1000 for training
    # and testing, respectively)
    model.rpn._pre_nms_top_n["training"] = 8000
    model.rpn._pre_nms_top_n["testing"] = 4000
    model.rpn._post_nms_top_n["training"] = 8000
    model.rpn._post_nms_top_n["testing"] = 4000

    # increase the total number of anchors (positive and negative) that are
    # sampled during training of RPN (for computing loss, default is 256; by
    # default 0.5 will be positive anchors)
    model.rpn.fg_bg_sampler.batch_size_per_image = 1024

    # increase the total number of anchors (positive and negative) that are
    # sampled during training of classification head (for computing loss,
    # default is 512; by default 0.25 will be positive anchors)
    model.roi_heads.fg_bg_sampler.batch_size_per_image = 2048

    # increase the number of detections per image to a larger number (default is
    # 100)
    model.roi_heads.detections_per_img = 1000

    return model


# Mask RCNN instance segmentation class
class MaskRCNNInstanceSegmentation:
    def __init__(
        self,
        weights_path: Optional[str],
        label_map: Optional[Dict[int, str]] = None,
        confidence: float = DEFAULT_DETECTION_CONFIDENCE,
        mask_threshold_for_bbox_expansion: float = DEFAULT_MASK_THRESHOLD_FOR_BBOX_EXPANSION,
    ):
        
        self.model: torchvision.models.detection.mask_rcnn.MaskRCNN = None
        self._weights_path: str = str(weights_path)
        self._confidence: float = confidence
        self._mask_threshold_for_bbox_expansion: float = (
            mask_threshold_for_bbox_expansion
        )
            
        # available device
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        
        # the state dictionary of the model to be read from the weights file
        model_state_dict: OrderedDict = None
        # the model's label map if available in the weight file
        loaded_label_map: Dict[int, str] = None   
        
        anchor_sizes: Tuple[Tuple[int]] =  DEFAULT_ANCHOR_SIZES
        # loading the PyTorch weights and the label map
        try:
            logging.info(
                f"Loading PyTorch Mask RCNN model from from {self._weights_path}. Setting to run on {self.device.type}."
            )
            
            saved_model_param: Union[OrderedDict, list] = torch.load(self._weights_path)
            if isinstance(saved_model_param, list):
                # label map and potentially other model related configs are also provided in the weights file, 
                # use them
                # in case a list is provided, the order of elemets is as following:
                # - model state dictionary (mandatory)
                # - label map (mandatory)
                # - anchor sizes (optional)
                # ... other optional fields to be added in future
                model_state_dict, loaded_label_map = saved_model_param[:2]
                if len(saved_model_param) >= 3:
                    # anchor sizes is also provided
                    logging.info(f"The model anchor sizes is provided in the weights file.") 
                    anchor_sizes = saved_model_param[2]
            else:
                model_state_dict = saved_model_param
                loaded_label_map = None

        except Exception as ex:
            logging.error(
                f"Failed to load Mask RCNN model. Likely the paths to model .pt weights "
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
                logging.info(f"Mapping between class IDs and class names is provided in the weights file.") 
                self._label_map: Dict[int, str] = loaded_label_map
        else:
            logging.info(f"Mapping between class IDs and class names is passed during class instantiation! "
                        f"It will overwrite the label map passed in the weights file (if provided).")        
            self._label_map: Dict[int, str] = label_map

        
        logging.info(f"Mapping between class IDs and class names: {self._label_map}") 
        self._reverse_label_map: Dict[str, int] = {
            value: key for key, value in self._label_map.items()
        }
        
        # Mask RCNN model
        self.model = get_instance_segmentation_model(
            num_classes=len(self._label_map) + 1, 
            anchor_sizes=anchor_sizes
        )

        # loading the PyTorch model
        try:
            self.model.load_state_dict(model_state_dict)
            self.model.to(self.device)
            self.model.eval()

        except Exception as ex:
            logging.error(
                f"Failed to load Mask RCNN model: {repr(ex)}."
            )

    # note that the passed image can be also a numpy array returned by
    # cv2.imread(img_path, cv2.IMREAD_UNCHANGED), it does not necessarily have to be a PIL image
    # in fact OpenCV is slightly more efficient in reading the images
    def detect(
        self, img: Union[Image.Image, np.ndarray], log_time: bool = False
    ) -> Dict[str, List]:
        """
        The main function to detect the bounding box and masks for objects in the input image.

        Args:
           img (PIL.Image or numpy array): Input image, should have 8 bits per channel bit depth (np.uint8 in
               case of a numpy array).
           log_time (bool): A flag to log the model run time.

        Returns:
           A dictionary with keys and values as below:
               "boxes": List of 4-tuples or 4-element lists for the detected objects' bounding boxes in
                xtl, ytl, xbr, ybr format/order
               "labels": List of integer class IDs for the detected objects
               "scores": List of float detection scores, after the threshold by self._confidence
               "masks": List of masks for the detected objects. Each mask is a numpy array of the same size
               as the bounding box (xbr - xtl, ybr - ytl)
        """

        if self.model is None:
            logging.error(
                "Mask RCNN model has not been initialized. Please initialize the class before detect()."
            )
            out: dict = {"boxes": [], "scores": [], "labels": [], "masks": []}
            return out

        start: float = time.time()

        # convert the input image to a tensor and scale it to [0, 1]
        # F.to_tensor takes care of it, however, make sure the passed img has bit-depth = 8 (is np.uint8 if numpy array)
        with torch.no_grad(), torch.cuda.amp.autocast():
            img_tensor = F.to_tensor(img).unsqueeze(0).to(self.device)
            # run inference
            out = self.model(img_tensor)[0]
        # before moving the results to CPU, crop the masks within the detection boxes
        # to significantly reduce their sizes
        # for a large number of detected cells, 2/3 of the model runtime is
        # spent on moving these image-sized masks from GPU to CPU, reduce their sizes in GPU to
        # save time moving them back to CPU

        # predictions (out) is a dictionary of four keys, 'boxes', 'labels', 'scores' and 'masks'
        out["boxes"] = to_numpy(out["boxes"]).astype(int)
        out["labels"] = to_numpy(out["labels"])
        out["scores"] = to_numpy(out["scores"])

        # masks will no longer be the same size for each detection, hence we return a list
        # of numpy arrays
        # to be consistent, we do the same (returning a list) for the rest
        masks: List[np.ndarray] = []
        boxes: List[List[int, int, int, int]] = []
        labels: List[int] = []
        scores: List[float] = []

        for i in range(out["boxes"].shape[0]):
            # skip unreliable or invalid detections
            (xtl, ytl, xbr, ybr) = out["boxes"][i]
            if out["scores"][i] < self._confidence or ytl >= ybr or xtl >= xbr:
                continue
            if self._mask_threshold_for_bbox_expansion > 0:
                # threshold the mask to keep only the values more than the passed threshold
                # then update the bounding box according to the remaining values
                pos = torch.where(
                    out["masks"][i, 0] >= self._mask_threshold_for_bbox_expansion
                )
                xmin: int = pos[1].min().item()
                xmax: int = pos[1].max().item()
                ymin: int = pos[0].min().item()
                ymax: int = pos[0].max().item()
                # apply some sanity checks on the expanded bounding box coordinates to avoid over expanding
                if 0 < (xmax - xmin) <= (1 + MAX_BBOX_EXPANSION_FACTOR) * (
                    xbr - xtl
                ) and 0 < (ymax - ymin) <= (1 + MAX_BBOX_EXPANSION_FACTOR) * (
                    ybr - ytl
                ):
                    xtl = xmin
                    ytl = ymin
                    xbr = xmax
                    ybr = ymax
                else:
                    # default expansion by 20% of the size of the bounding box
                    delta_x: int = int(DEFAULT_BBOX_EXPANSION_FACTOR * (xbr - xtl) / 2)
                    delta_y: int = int(DEFAULT_BBOX_EXPANSION_FACTOR * (ybr - ytl) / 2)

                    xtl = max(0, xtl - delta_x)
                    ytl = max(0, ytl - delta_y)
                    xbr = min(out["masks"][i, 0].shape[1], xbr + delta_x)
                    ybr = min(out["masks"][i, 0].shape[0], ybr + delta_y)

            boxes.append([xtl, ytl, xbr, ybr])
            labels.append(out["labels"][i])
            scores.append(out["scores"][i])
            # cast as float because with autocast, the masks will be float16, which may not
            # be supported by some OpenCV functions
            masks.append(to_numpy(out["masks"][i, 0, ytl:ybr, xtl:xbr]).astype(float))

        elap: float = time.time() - start
        if log_time:
            logging.info(f"Mask RCNN instance segmentation took {elap:.4f} seconds")

        out["boxes"] = boxes
        out["scores"] = scores
        out["labels"] = labels
        out["masks"] = masks
        return out
    
    def get_label_map(self):
        return self._label_map
    
    def get_reverse_label_map(self):
        return self._reverse_label_map
    
    def set_confidence(self, confidence):
        self._confidence = confidence

    def detect_by_cropping(
        self,
        image: Union[Image.Image, np.ndarray],
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
            image (PIL.Image or numpy array): Input image, should have 8 bits per channel
                bit depth (np.uint8 in case of a numpy array).
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
            log_time(bool): A flag to print the runtime of the function.
        Returns:
            A dictionary with keys and values as below:
               "boxes": List of 4-tuples or 4-element lists for the detected objects' bounding boxes in
               xtl, ytl, xbr, ybr format/order
               "labels": List of integer class IDs for the detected objects
               "scores": List of float detection scores, thresholded by self._confidence
               "masks": List of masks for the detected objects. Each mask is a numpy array of the same size
               as the bounding box (xbr - xtl, ybr - ytl)
        """

        start = time.time()
        if len(crop_corners) == 0:
            logging.error(
                "No crop corners are provided for running YOLOv5 model on sub-images. "
                "Returning no detections"
            )
            out: dict = {"boxes": [], "scores": [], "labels": [], "masks": []}
            return out
        
        
        if classnames_to_return is not None:
            class_ids_to_return: List[int] = []
            for class_name in classnames_to_return:
                if class_name not in self._reverse_label_map:
                    logging.warn(
                        f"Classname '{class_name}' is not included in the model label map! Skipping this name ..."
                    )
                else:
                    class_ids_to_return.append(self._reverse_label_map[class_name])
            
            if len(class_ids_to_return) == 0:
                # nothing to return
                logging.warn(
                        f"No valid classnames was provided in classnames_to_return: {classnames_to_return}!"
                        f"Returning no detections"
                    )
                out: dict = {"boxes": [], "scores": [], "labels": [], "masks": []}
                return out
        else:
            # this list will not be used and is not really needed, define it just in case
            class_ids_to_return: List[int] = list(self._label_map.keys())
        

        H, W = image.shape[:2]

        # check if all the crop sub-images are of the same size,
        # if not, make them equal size for batch processing
        crop_widths: List[int] = [min(c[2], W) - max(c[0], 0) for c in crop_corners]
        crop_heights: List[int] = [min(c[3], H) - max(c[1], 0) for c in crop_corners]

        if min(crop_widths) <= 0 or min(crop_heights) <= 0:
            logging.error(
                "Incorrect corners are provided for running Mask RCNN model on sub-images. "
                "Returning no detections"
            )
            out: dict = {"boxes": [], "scores": [], "labels": [], "masks": []}
            return out

        crop_width: int = max(crop_widths)
        crop_height: int = max(crop_heights)

        # combine the results, filter them based on the score,
        # and update the coordinates of the bounding boxes
        # for applying NMS later
        results: Dict = {"scores": [], "boxes": [], "labels": [], "masks": []}

        # a list to keep track of cropped sub-images with at least one object detection
        crop_ids_with_detection: List[int] = []

        for crop_id, corners in enumerate(crop_corners):
            (x1c, y1c, x2c, y2c) = corners
            # enlarge the crop if necessary to make all the same size
            x2c = x1c + crop_width
            y2c = y1c + crop_height

            # crop the image and run the model
            cropped_image = image[y1c:y2c, x1c:x2c]
            preds = self.detect(cropped_image)

            boxes: np.array = np.array(preds["boxes"])
            labels: np.arary = np.array(preds["labels"])
            scores: np.array = np.array(preds["scores"])
            # we leave the masks as a list of mask numpy arrays as each mask has different sizes below
            masks: List[np.array] = preds["masks"]

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
            out: dict = {"boxes": [], "scores": [], "labels": [], "masks": []}
            return out

        # list to contain the detections
        boxes, labels, scores, masks = [], [], [], []

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
            crop_labels: np.array = results["labels"][idx]
            crop_scores: np.array = results["scores"][idx]
            crop_boxes: np.array = results["boxes"][idx]
            crop_masks: List[np.array] = results["masks"][idx]

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

            # masks for detections in other cropped sub-images
            rest_masks: List[np.ndarray] = [
                results["masks"][i]
                for i in range(len(crop_ids_with_detection))
                if i != idx
            ]

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
                        crop_box_area: int = box_area(crop_boxes[crop_class_idxs[0][i]])
                        rest_box_area: int = box_area(
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

            boxes = [boxes[i] for i in detection_ids]
            labels = [labels[i] for i in detection_ids]
            scores = [scores[i] for i in detection_ids]
            masks = [masks[i] for i in detection_ids]

        elap: float = time.time() - start
        if log_time:
            logging.info(
                "Mask RCNN instance segmentation after cropping the image to "
                "{} sub-images took {:.4f} seconds in OpenCV".format(
                    len(crop_corners), elap
                )
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

        
detector = MaskRCNNInstanceSegmentation(weights_path=MODEL_WEIGHTS_PATH)

# A dictionary with keys as the input (original) image size (width, height) tuple and
# values as the list of coordinates (xtl, ytl, xbr, ybr) of sub-images/crops
# to run YOLOv5 on each
# note that the crop coordinates are with respect to resized image dimensions specified above
RESIZE: Final[Dict[Tuple[int, int], Tuple[int, int]]] = {
    (2000, 1600): (1280, 1024),
    (4512, 4512): (2144, 2144),
}
# A dictionary with keys as the input (original) image size (width, height) tuple and
# values as the list of coordinates (xtl, ytl, xbr, ybr) of sub-images/crops
# to run YOLOv5 on each
# note that the crop coordinates are with respect to resized image dimensions specified above
CROP_CORNERS: Final[Dict[Tuple[int, int], List[List[int]]]] = {
    (2000, 1600): [
        [0, 0, 800, 1024],
        [480, 0, 1280, 1024]
    ],
    (4512, 4512): [
        [0, 0, 1200, 800],
        [0, 448, 1200, 1248],
        [0, 896, 1200, 1696],
        [0, 1344, 1200, 2144],
        [472, 0, 1672, 800],
        [472, 448, 1672, 1248],
        [472, 896, 1672, 1696],
        [472, 1344, 1672, 2144],
        [944, 0, 2144, 800],
        [944, 448, 2144, 1248],
        [944, 896, 2144, 1696],
        [944, 1344, 2144, 2144]
    ]
}

# threshold to apply on masks
MASK_THRESHOLD: Final[float] = 0.1
# threshold for post-processing cells and remove the ones consisting of multiple smaller cells
OVER_LAP_THRESHOLD: Final[float] = 0.75

def run_mask_rcnn(
        input_image: np.ndarray, normalize_image: bool = True, bit_depth: int = 8, crop: bool = True, post_process_class_names: List[str] = ['cell', 'nucleus'], plot_results: bool = False, 
) -> Tuple[Dict[str, list], float, Optional[np.ndarray]]:
    # make a copy to not modify the input image
    img = input_image.copy()

    if len(img.shape) > 2:
        logging.warning(
            "Warning Mask R-CNN model may suffer loss in precision due to conversion from RGB to grayscale"
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = (255 * img.astype(float) / (2 ** bit_depth - 1)).astype(np.uint8)

    if normalize_image:
        img = cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    image_height, image_width = img.shape[:2]

    if crop and (image_width, image_height) not in RESIZE:
        logging.error(
            "The input image size {} is not supported! Returning no cells!".format(
                image_width, image_height
            )
        )
        out = {'boxes': np.zeros((0, 4), dtype=int),
               'labels': np.zeros((0,), dtype=int),
               'scores': np.zeros((0,), dtype=float),
               'masks': [],
               }
        if plot_results:
            return (out, 0, np.zeros((image_height, image_width), dtype=np.uint8))
        else:
            return (out, 0)

    if crop:
        # we keep the aspect ratio in RESIZE dictionary, scale_factor is the same for both dimensions
        scale_factor: float = image_width / RESIZE[(image_width, image_height)][0]
        resized_width, resized_height = RESIZE[(image_width, image_height)]
    else:
        larger_side_size: int = max(image_width, image_height)
        smaller_side_size: int = min(image_width, image_height)
        scale_factor: float = max(smaller_side_size / 800.0, larger_side_size / 1333.0)
        # do not scale the image if the larger and the smaller sides are smaller than the model input sizes (1333, 800)
        scale_factor = max(1, scale_factor)
        resized_width, resized_height = int(image_width / scale_factor), int(image_height / scale_factor)

    if scale_factor != 1:
        resized_img: np.ndarray = cv2.resize(
            img, (resized_width, resized_height), interpolation=cv2.INTER_AREA
        )
    else:
        resized_img: np.ndarray = img

    st = time.time()

    if crop:
        crop_corners: List[List[int]] = CROP_CORNERS[(image_width, image_height)]
        out: Dict[str, list] = detector.detect_by_cropping(
            resized_img, crop_corners
        )
    else:
        out: Dict[str, list] = detector.detect(resized_img)

    if scale_factor != 1:
        # scale the detections back to original image resolution
        out['boxes'] = (scale_factor * np.array(out['boxes'])).astype(int)
        # convert to a list to be consistent with the rest
        out['boxes'] = [box for box in out['boxes']]
    else:
        out['boxes'] = [np.array(box) for box in out['boxes']]   

    for idx in range(len(out['boxes'])):
        if scale_factor != 1:
            xtl, ytl, xbr, ybr = out['boxes'][idx]
            # note that mask here is a probability mask and interpolation does not have to be nearest neighbor
            out['masks'][idx] = cv2.resize(out['masks'][idx], (xbr - xtl, ybr - ytl), interpolation=cv2.INTER_LINEAR)
        mask_this_cell: np.ndarray = np.zeros(out['masks'][idx].shape, dtype=np.uint8)
        mask_this_cell[out['masks'][idx] >= MASK_THRESHOLD] = 1
        out['masks'][idx] = mask_this_cell.astype(np.uint8)
    
    # post-process the results
    # in the following, "larger" objects that consist of a number of already detected smaller objects of the same type are invalidated
    # this can happen mainly for 'cell', 'nucleus' and 'cell-adhered'/'cytoplasm' classes
    # list of indexes of objects for each class name to be included in post processing
    post_process_class_idxs: Dict[str, List[int]] = {}
    # list of bounding boxes for each class name to be included in post processing
    post_process_class_boxes: Dict[str, List[np.ndarray]] = {}
    for i, box in enumerate(out['boxes']):
        for class_name in post_process_class_names:
            if class_name in detector._reverse_label_map and out['labels'][i] == detector._reverse_label_map[class_name]:
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
        post_process_class_boxes[key]: np.ndarray = np.array(post_process_class_boxes[key])
        
        if len(post_process_class_idxs[key]) == 0:
            continue
        
        overlap: np.ndarray = overlap_batch(post_process_class_boxes[key], post_process_class_boxes[key], True)
        # remove diagonal elements (as each box has a complete overlap with itself)
        overlap = overlap - np.eye(len(post_process_class_boxes[key]))
        # index of larger objects (row indexes) covering some smaller already detected cells (column index)
        # by more than OVER_LAP_THRESHOLD
        # these smaller objects are most probably redundant objects
        covering_obj_idxs, covered_obj_idxs = np.where(overlap > OVER_LAP_THRESHOLD)
        # now double-check the coverage using the masks
        for (i, j) in zip(covering_obj_idxs, covered_obj_idxs):
            large_obj_index: int = post_process_class_idxs[key][i]
            small_obj_index: int = post_process_class_idxs[key][j]
            # larger box coordinates
            xl1, yl1, xl2, yl2 = out['boxes'][large_obj_index]
            # smaller box coordinates
            xs1, ys1, xs2, ys2 = out['boxes'][small_obj_index]
            # union of the two boxes
            x1: int = min(xl1, xs1)
            y1: int = min(yl1, ys1)
            x2: int = max(xl2, xs2)
            y2: int = max(yl2, ys2)
            large_obj_mask: np.ndarray = np.zeros((y2 - y1, x2 - x1), np.uint8)
            large_obj_mask[(yl1 - y1):(yl2 - y1), (xl1 - x1):(xl2 - x1)] = out['masks'][large_obj_index]
            small_obj_mask: np.ndarray = np.zeros((y2 - y1, x2 - x1), np.uint8)
            small_obj_mask[(ys1 - y1):(ys2 - y1), (xs1 - x1):(xs2 - x1)] = out['masks'][small_obj_index]
            
            if np.sum(small_obj_mask * large_obj_mask) > OVER_LAP_THRESHOLD * np.sum(small_obj_mask):
                # add row index i to the list of object indexes to be removed
                if small_obj_index not in obj_idxs_to_remove:
                    obj_idxs_to_remove.append(small_obj_index)
    
    if len(obj_idxs_to_remove) > 0:
        out['boxes'] = [box for i, box in enumerate(out['boxes']) if i not in obj_idxs_to_remove]
        out['labels'] = [label for i, label in enumerate(out['labels']) if i not in obj_idxs_to_remove]
        out['scores'] = [score for i, score in enumerate(out['scores']) if i not in obj_idxs_to_remove]
        out['masks'] = [mask for i, mask in enumerate(out['masks']) if i not in obj_idxs_to_remove]
    
    et = time.time()

    if plot_results:
        return out, et - st, show_detections(img, out, detector._label_map)
    
    return out, et - st
