import os
import time
import logging
from typing import Tuple, List, Final, Optional, Dict, Union
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image
import torch
from rfdetr import RFDETRBase

MODEL_WEIGHTS_PATH: Final[str] = '/home/cellareye/Cellanome/dl-mehdi/Mask RCNN/output/checkpoint_best_total.pth'
DEFAULT_DETECTION_CONFIDENCE: Final[float] = 0.45
DEFAULT_LABEL_MAP: Final[Dict[int, str]] = {0: 'cell', 1: 'bead', 2: 'soma', 3: 'cell-adhered'}
MODEL_INPUT_SIZE: Final[int] = 672
# the aspect ratio of the passed image to the model should be within this threshold (percentagewise)
# of the aspect ratio of the input image size specified above
# if the deviation is more than this threshold, the results may not be reliable
INPUT_IMAGE_ASPECT_RATIO_DIFF_DEV_THRESH: Final[float] = 0.1

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
    
    for i in range(len(boxes)):
        # the bounding box
        (xtl, ytl, xbr, ybr) = boxes[i].astype(int)
        if labels[i] not in class_ids:
            # use black for incorrect label
            color = (0, 0, 0)
            text = "Unknown label %s" % labels[i]
        else:
            color = COLORS[(labels[i] + 1) % len(COLORS)] # add one to start from 1 for consistency with Mask R-CNN
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
class RfDetrObjectDetector:
    def __init__(
        self,
        weights_path: Optional[str] = MODEL_WEIGHTS_PATH,
        label_map: Dict[int, str] = DEFAULT_LABEL_MAP,
        confidence: float = DEFAULT_DETECTION_CONFIDENCE,
    ):
        
        self.model = None
        self._weights_path: str = str(weights_path)
        self._confidence: float = confidence
            
        # available device
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # TODO: to be included in the weights file
        self._label_map: Dict[int, str] = label_map
        self._detected_class_names_remap: Dict[str, str] = None
        self._resize_dict: Dict[Tuple[int, int, str], Tuple[int, int]] = None
        self._crop_corners_dict: Dict[Tuple[int, int, str], List[List[int]]] = None
        self._model_input_size = MODEL_INPUT_SIZE
         
        
        logging.info(f"Mapping between class IDs and class names: {self._label_map}") 
        self._reverse_label_map: Dict[str, int] = {
            value: key for key, value in self._label_map.items()
        }
        
        # loading the PyTorch model
        try:
            self.model = RFDETRBase(pretrain_weights=self._weights_path)
            # will be taken care of in .predict function
            # self.model.to(self.device)
            # self.model.eval()

        except Exception as ex:
            logging.error(
                f"Failed to load RF-DETR model: {repr(ex)}."
            )
        
        self._metadata = {'resolution': self._model_input_size,
                          'release_date': '20250410',
                          'model_type': 'Transformer Detector',
                          'model_name': 'RF-DETR',
                          'model_extra_info': 'None',
                          'names': self._label_map,
                          'magnification': '10x'
                         }
    def detect(self,
               img: np.ndarray,
               log_time: bool = False
              ) -> Dict[str, list]:
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

        if self.model is None:
            logging.error(
                "RF-DETR model has not been initialized. Please initialize the class before detect()."
            )
            
            out: Dict[str, list] = {"boxes": [], "scores": [], "labels": [],}
            return out

        start: float = time.time()
        # convert to 3-channel images if needed, and store the original image dimensions for 
        # post processing
        org_img_shape: tuple = img.shape
        if len(org_img_shape) < 3:
            input_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            input_img = img

        # check if the aspect ratio of the input image is almost the same as the aspect ratio of
        # the model input size
        # if not, then the input image will be resized without keeping its aspect ratio when it
        # is passed to the model, and this may lead to inaccurate detection
        aspect_ratio_diff: float = float(org_img_shape[1]) / float(org_img_shape[0]) - 1.0

        if np.abs(aspect_ratio_diff) > INPUT_IMAGE_ASPECT_RATIO_DIFF_DEV_THRESH:
            logging.warning(
                f"The input image has a different aspect ratio: {aspect_ratio_diff + 1} than the model: 1! The results may not be accurate"
            )

        # resize the input image to match the model input size
        if aspect_ratio_diff != 0:
            input_img = cv2.resize(input_img, (self._model_input_size, self._model_input_size))
        
        out: dict = {}
        with torch.no_grad():
            detections = self.model.predict(input_img, threshold=self._confidence)

        if aspect_ratio_diff != 0:
            out['boxes'] = (
                detections.xyxy * np.array([org_img_shape[1] / self._model_input_size, org_img_shape[0] / self._model_input_size] * 2)
            ).astype(int)
        else:
            out['boxes'] = detections.xyxy.astype(int)
            
        out['scores'] = detections.confidence
        out['labels'] = detections.class_id
        # Clear the CUDA cache
        torch.cuda.empty_cache()
        elap: float = time.time() - start
        if log_time:
            logging.info(f"RF-DETR object detection took {elap:.4f} seconds")

        return out
    
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
        results: Dict = {"scores": [], "boxes": [], "labels": [],}

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
                        crop_box_area: float = box_area(crop_boxes[crop_class_idxs[0][i]])
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
                "RF-DETR object detection after cropping the image to "
                "{} sub-images took {:.4f} seconds".format(
                    len(crop_corners), elap
                )
            )

        out: dict = {
            "boxes": [
                [box[0], box[1], box[2], box[3]] for box in boxes
            ],  # convert to list
            "scores": scores,
            "labels": labels,
        }

        return out

        
detector = RfDetrObjectDetector(weights_path=MODEL_WEIGHTS_PATH)

"""
RESIZE: Final[Dict[Tuple[int, int, str], Tuple[int, int]]] = {
    (2000, 1600, "10x"): (1924, 1540),
    (4512, 4512, "10x"): (4342, 4342),
    (4512, 4512, "4x"): (4342, 4342),
}

CROP_CORNERS: Final[Dict[Tuple[int, int, str], List[List[int]]]] = {
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

RESIZE: Final[Dict[Tuple[int, int, str], Tuple[int, int]]] = {
    (2000, 1600, "10x"): (2000, 1600),
    (4512, 4512, "10x"): (4512, 4512),
    (4512, 4512, "4x"): (4512, 4512),
}
# A dictionary with keys as the input (original) image size (width, height, magnification)
# tuple and values as the list of coordinates (xtl, ytl, xbr, ybr) of sub-images/crops
# to run YOLOv5 on each
# note that the crop coordinates are with respect to resized image dimensions specified above
CROP_CORNERS: Final[Dict[Tuple[int, int, str], List[List[int]]]] = {
    (2000, 1600, "10x"): [
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
        [1328, 928, 2000, 1600]
    ],
    (4512, 4512, "10x"): [
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
        [3840, 3840, 4512, 4512]
    ],
    (4512, 4512, "4x"): [
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
        [3840, 3840, 4512, 4512]
    ],
}

def run_rf_detr(
    input_image: np.ndarray, 
    bit_depth: int = 12, 
    normalize_image: bool=True, 
    is_4x: bool=False, 
    detector: RfDetrObjectDetector=detector
) -> Tuple[np.array, np.array, np.array, float]:
    
    # make a copy to not modify the input image
    img = input_image.copy()

    if len(img.shape) > 2:
        print(
            "Warning RT-DETR model may suffer loss in precision due to conversion from RGB to grayscale"
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = (255 * img.astype(float) / (2 ** bit_depth - 1)).astype(np.uint8)
    
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
                0
            )

    # we keep the aspect ratio in RESIZE dictionary, scale_factor is the same for both dimensions
    
    scale_factor: float = image_width / resize_dict[resize_dict_key][0]
    resized_img: np.ndarray = cv2.resize(
        img, resize_dict[resize_dict_key], interpolation=cv2.INTER_AREA
    )

    crop_corners: List[List[int]] = crop_corners_dict[resize_dict_key]

    st = time.time()

    out = detector.detect_by_cropping(
        resized_img, crop_corners
    )
    boxes: np.ndarray =  np.zeros((0, 4), dtype=int)
    labels: np.ndarray = np.zeros((0,), dtype=int)
    scores: np.ndarray = np.zeros((0,), dtype=float)
    
    if len(out['labels']) > 0:
        boxes, labels, scores = np.array(out['boxes']), np.array(out['labels']), np.array(out['scores'])
    
    # scale the detections back to original image resolution
    boxes = (scale_factor * boxes).astype(int)

    et = time.time()

    return boxes, labels, scores, et - st

