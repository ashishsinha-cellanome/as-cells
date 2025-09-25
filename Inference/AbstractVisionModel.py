from __future__ import annotations
from model_utils import iou_batch, overlap_batch, box_area, show_detections

import numpy as np
import cv2
import os
import time
import logging
from PIL import Image
import torch

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Optional, Union, Final



# threshold on the proximity of the bounding box corners to the crop boundaries to declare the
# detected object is potentially cropped between the two
PROXIMITY_TO_CROP_BOUNDARIES_TO_DECLARE_CROPPED_DETECTIONS_IN_PIXEL: Final[int] = 4
# threshold for post-processing cells and remove the ones consisting of multiple smaller cells
OVER_LAP_THRESHOLD: Final[float] = 0.75
# the default detection threshold
DEFAULT_DETECTION_CONFIDENCE: Final[float] = 0.5

class VisionModel(ABC):
    """
    Generic CV model interface using the Template Method pattern.
    Subclasses override load(), preprocess(), infer(), and postprocess().
    This class also provides a generic detect_by_cropping() implementation.
    """

    def __init__(
        self,
        weights_path: str,
        model_name: str, 
        label_map: Optional[Dict[int, str]] = None,
        model_input_size: Optional[Tuple[int, int]] = None, 
        confidence: float = DEFAULT_DETECTION_CONFIDENCE,
        device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    ):
        self._model_name: str = model_name
        self._loaded: bool = False
        self._model: Any = None
        
        # the model input size in (width, height) if the model expect a fixed input size
        # this can be passed or included in the weights file
        self._model_input_size: Tuple[int, int] = None

        # resize and crop corners can be included in the model file and can be loaded below
        self._resize_dict: Dict[Tuple[int, int], Tuple[int, int]] | None = None
        self._crop_corners_dict: Dict[Tuple[int, int], List[List[int]]] | None = None
        
        # the model file usually includes the class ID to class name mapping, but it can be overwritten by 
        # the passed label_map above
        self._label_map: Dict[int, str] | None = None
        self._reverse_label_map: Dict[str, int] | None = None
        # the model file can also include some class ID remapping, or exclusion
        self._detected_class_ids_remap: Dict[int, int] | None = None
        self._detected_class_names_remap: Dict[str, str] | None = None
        self._class_ids_to_exclude_from_dets: List[int] = []

        
        self._weights_path = weights_path
        self._model_name = model_name
        self._confidence = confidence
        self._device = device
        
        # a dictionary with some details about the model that is populated by self.load()
        # for now use some default values
        self._metadata: dict = {
            'predict_masks': True, 
            'magnification': '10x',
        }
        
        # the state dictionary of the model to be read from the weights file
        self._model_state_dict: OrderedDict = None
        # the model's label map if available in the weight file
        loaded_label_map: Dict[int, str] = None   
        
        # loading the PyTorch weights and the label map
        try:
            logging.info(
                f"Loading {self._model_name} model from {self._weights_path}. Setting to run on {self._device.type}."
            )

            saved_model_param: Union[OrderedDict, dict] = torch.load(
                self._weights_path, map_location=self._device
            )
            if (isinstance(saved_model_param, dict) and "model_state_dict" in saved_model_param):
                # the weights file contains the model state dictionary (the weights) and the label map (both are
                # mandatory) and potentially other model related configs
                # in case a dictionary is provided, the keys and values are as following:
                # - 'model_state_dict': model state dictionary (mandatory)
                # - 'label_map': label map (mandatory)
                # - 'model_input_size': model input size tuple (optional, required for models with fixed input sizes)
                # - 'resize_dict': resize dictionary (optional, for resizing and cropping images during inference)
                # - 'crop_corners_dict': crop corners dictionary (optional, for cropping images during inference)
                # ... other optional fields to be added in future
                self._model_state_dict = saved_model_param["model_state_dict"]
                if "label_map" in saved_model_param:
                    loaded_label_map = saved_model_param["label_map"]
                else:
                    logging.warning(
                        f"The weights file should contain the label map but it missing. "
                        f"This should never happen..."
                    )
                    loaded_label_map = None

                if "model_input_size" in saved_model_param:
                    # resize and crop corners are also provided in the weights file
                    logging.info(
                        f"The model input size is also provided in the weights file."
                    )
                    self._model_input_size = saved_model_param["model_input_size"]
                    
                if ("resize_dict" in saved_model_param and "crop_corners_dict" in saved_model_param):
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
                self._model_state_dict = saved_model_param
                loaded_label_map = None

        except Exception as ex:
            logging.error(
                f"Failed to load {self._model_name} model. Likely the paths to model .pt weights "
                f"{self._weights_path} is incorrect or the model is not in the expected format! This may work though : {repr(ex)}."
            )

        if model_input_size is not None:
            if self._model_input_size is not None:
                logging.warning(
                        f"The model input size from the weights file {self._model_input_size} is overwritten by "
                        f"{model_input_size} during class instantiation! Make sure this replacement is intentional."
                    )    
            self._model_input_size = model_input_size
            
        
        # take care of the mapping between class IDs and class names
        if label_map is None:
            # check if the label map is provided in the model file
            if loaded_label_map is None:
                logging.error(
                    f"The mapping between the class IDs and class names is required for the model and is "
                    f"neither provided during class instantiation nor available in the weights file! Returning ..."
                )
            else:
                logging.info(f"The mapping between class IDs and class names is provided in the weights file.")
                self._label_map: Dict[int, str] = loaded_label_map
        else:
            logging.info(f"The mapping between class IDs and class names is passed during class instantiation! "
                        f"It will overwrite the label map passed in the weights file (if provided).")        
            self._label_map: Dict[int, str] = label_map

        # load the model
        if self._label_map is not None: # no need to check for self._model_state_dict as if None, self.load() below returns self._loaded False
            # load the model, we need the label map to define/load the model, if self._load() succeeds, it sets self._loaded to True 
            self.load()
        
        # proceed with the rest
        if self._loaded:
            logging.info(f"The mapping between class IDs and class names: {self._label_map}") 
            self._reverse_label_map: Dict[str, int] = {
                value: key for key, value in self._label_map.items()
            }
        
            if self._detected_class_names_remap is not None:
                # extract the classes that will be mapped to 'bg' and should be excluded from the detections
                class_names_to_exclude_from_dets = [k for k, v in self._detected_class_names_remap.items() if v == 'bg']
                # update the passed mapping and removed the ones that are going to be mapped to 'bg' (should be excluded)
                self._detected_class_names_remap = {k: v for k, v in self._detected_class_names_remap.items() if v != 'bg'}
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
                    f"The mapping between class IDs and class names will be updated according to the following "
                    f"mapping: {self._detected_class_names_remap}"
                )
                logging.info(
                    f"The following class names will be excluded from detections (mapped to 'bg' in the passed mapping): "
                    f"{class_names_to_exclude_from_dets}"
                )
        

    # abtract method to be overwritten for specific models

    @abstractmethod
    def load(self) -> None:
        """
        Load the model weights from the self._model_state_dict and specific model implementation, modifies self._model
        """


    @abstractmethod
    def detect_batch(
        self,
        input_images_list: List[Union[PIL.Image, np.ndarray]],
    ) -> List[Dict[str, list]]:
        """
        The main function to detect the bounding box and masks for objects in a list of inputs images (batch processing). 

        Args:
            input_images_list (list of PIL.Image or numpy arrays): Input images, each should have 8 bits per channel bit-depth 
            (np.uint8 numpy array). 

        Returns:
            A list of dictionaries with keys and values as below:
                "boxes": List of 4-tuples or 4-element lists for the detected objects' bounding boxes in
                    xtl, ytl, xbr, ybr format/order.
                "labels": List of integer class IDs for the detected objects.
                "scores": List of float detection scores, after applying the threshold self._confidence.
                "masks": Optional list of bindary masks (not probability) for the detected objects. Each mask is a numpy array of the same size
                    as the bounding box with width and height (xbr - xtl, ybr - ytl). This key will not be included in the results for
                    object detection models. 
        """

    @abstractmethod
    def postprocess(self, **kwargs: Any) -> List[Dict[str, list]]:
        """
        post-processing the results
        """
    
    def detect(
        self,
        img: Union[Image.Image, np.ndarray],
    ) -> Dict[str, list]:
        """
        The main function to detect the bounding box and masks for objects in the input image.

        Args:
            img (PIL.Image or numpy array): Input image, should have 8 bits per channel bit-depth (np.uint8 numpy array). 

        Returns:
            A dictionary with keys and values as below:
                "boxes": List of 4-tuples or 4-element lists for the detected objects' bounding boxes in
                    xtl, ytl, xbr, ybr format/order.
                "labels": List of integer class IDs for the detected objects.
                "scores": List of float detection scores, after applying the threshold self._confidence.
                "masks": Optional list of bindary masks (not probability) for the detected objects. Each mask is a numpy array of the same size
                    as the bounding box with width and height (xbr - xtl, ybr - ytl). This key will not be included in the results for
                    object detection models. 
        """
        return self.detect_batch(input_images_list=[img])[0]
    
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

    def get_model_name(self):
        return self._model_name

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
        specified by crop_corners), running the detector on each sub-image and
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
            log_time (bool): A flag to log the runtime of the function.
        Returns:
            A dictionary with keys and values as below:
                "boxes": List of 4-tuples or 4-element lists for the detected objects' bounding boxes in
                    xtl, ytl, xbr, ybr format/order.
                "labels": List of integer class IDs for the detected objects.
                "scores": List of float detection scores, after applying the threshold self._confidence.
                "masks": Optional list of bindary masks (not probability) for the detected objects. Each mask is a numpy array of the same size
                    as the bounding box with width and height (xbr - xtl, ybr - ytl). This key will not be included in the results for
                    object detection models. 
        """

        start = time.time()

        if self._metadata['predict_masks']:
            invalid_out: Dict[str, list] = {"boxes": [], "scores": [], "labels": [], "masks": []}
        else:
            invalid_out: Dict[str, list] = {"boxes": [], "scores": [], "labels": []}

        if len(crop_corners) == 0:
            logging.error(
                f"No crop corners are provided for running {self._model_name} model on sub-images. "
                f"Returning no detections ..."
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
            class_ids_to_return: List[int] = list(self.get_label_map().keys())

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
                f"Incorrect corners are provided for running {self._model_name} model on sub-images. "
                f"Returning no detections ..."
            )
            return invalid_out

        crop_width: int = max(crop_widths)
        crop_height: int = max(crop_heights)

        # combine the results, filter them based on the score,
        # and update the coordinates of the bounding boxes
        # for applying NMS later
        if self._metadata['predict_masks']:
            results: Dict = {"scores": [], "boxes": [], "labels": [], "masks": [],}
        else:
            results: Dict = {"scores": [], "boxes": [], "labels": []}
            
            
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
            if self._metadata['predict_masks']:
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
                (boxes[:, 0] < PROXIMITY_TO_CROP_BOUNDARIES_TO_DECLARE_CROPPED_DETECTIONS_IN_PIXEL)
                | (boxes[:, 1] < PROXIMITY_TO_CROP_BOUNDARIES_TO_DECLARE_CROPPED_DETECTIONS_IN_PIXEL)
                | (boxes[:, 2] >= crop_width - PROXIMITY_TO_CROP_BOUNDARIES_TO_DECLARE_CROPPED_DETECTIONS_IN_PIXEL)
                | (boxes[:, 3] >= crop_height - PROXIMITY_TO_CROP_BOUNDARIES_TO_DECLARE_CROPPED_DETECTIONS_IN_PIXEL)
                ] = self._confidence

            crop_ids_with_detection.append(crop_id)
            results["scores"].append(scores)
            results["boxes"].append(boxes + np.array([x1c, y1c, x1c, y1c], dtype=int))
            results["labels"].append(labels)
            if self._metadata['predict_masks']:
                results["masks"].append(masks)

        # no object detected, return
        if len(crop_ids_with_detection) == 0:
            return invalid_out

        # list to contain the detections
        boxes: list = []
        labels: list = []
        scores: list = []
        if self._metadata['predict_masks']:
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
            if self._metadata['predict_masks']:
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
                    if self._metadata['predict_masks']:
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
                    # compute the IoU matrix, not using torchvision implementation as iou_batch is more efficient
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
                        if self._metadata['predict_masks']:
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
                            if self._metadata['predict_masks']:
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
            if self._metadata['predict_masks']:
                masks: list = [masks[i] for i in detection_ids]
        
        elap: float = time.time() - start
        if log_time:
            logging.info(
                "{} instance segmentation model after cropping the image to "
                "{} sub-images took {:.4f} seconds".format(
                    self._model_name, len(crop_corners), elap
                )
            )

        out: dict = {
            "boxes": [
                [box[0], box[1], box[2], box[3]] for box in boxes
            ],  # convert to list
            "scores": scores,
            "labels": labels,
        }
        
        if self._metadata['predict_masks']:
            out["masks"] = masks

        return out

# end of class definition

   
def post_process_detections(
    detections: List[str, list], 
    post_process_class_names: List[str], 
    classnames_to_class_ids_map: Dict[str, int], 
    overlap_threshold: float,
):
    """
    in this function, "smaller" detections that are part of larger objects of the same type are detected, and invalidated
    this can happen mainly for 'cell', 'nucleus' and 'cell-adhered'/'cytoplasm' classes
    """
    predict_masks: bool = 'masks' in detections
    # list of indexes of objects for each class name to be included in post processing
    post_process_class_idxs: Dict[str, List[int]] = {}
    # list of bounding boxes for each class name to be included in post processing
    post_process_class_boxes: Dict[str, List[np.ndarray]] = {}
    for i, box in enumerate(detections['boxes']):
        for class_name in post_process_class_names:
            if class_name in classnames_to_class_ids_map and detections['labels'][i] == classnames_to_class_ids_map[class_name]:
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

        # find the overlap between the detections, overlap[i, j] is defined below as the area of the intersection between
        # the boxes of the detected object i and j divided by the area of the box of object j (column index)
        # note that this matrix is not symmetric:
        # - a large overlap[i, j] but small overlap[j, i] indicate that object i is larger than object j and is mostly covering it
        # - large overlap[i, j] and overlap[j, i] indicates both objects have high IoU (for an overlap threshold of 0.75, the IoU will be
        #   at least 0.6)
        overlap: np.ndarray = overlap_batch(post_process_class_boxes[key], post_process_class_boxes[key], True)
        # remove diagonal elements (as each box has a complete overlap with itself)
        overlap = overlap - np.eye(len(post_process_class_boxes[key]))
        
        # index of larger objects (row indexes) covering some smaller already detected cells (column index)
        # by more than OVER_LAP_THRESHOLD
        # these smaller objects are most probably redundant objects
        covering_obj_idxs, covered_obj_idxs = np.where(overlap > overlap_threshold)
        # now double-check the coverage using the masks
        for (i, j) in zip(covering_obj_idxs, covered_obj_idxs):
            # index with respect to all detected and not object of class key
            large_obj_index: int = post_process_class_idxs[key][i]
            small_obj_index: int = post_process_class_idxs[key][j]
            if predict_masks:
                # larger box coordinates
                xl1, yl1, xl2, yl2 = detections['boxes'][large_obj_index] # the same as post_process_class_boxes[key][i]
                # smaller box coordinates
                xs1, ys1, xs2, ys2 = detections['boxes'][small_obj_index] # the same as post_process_class_boxes[key][j]
                # union of the two boxes, needed to compute the masks intersection efficiently within this union
                # as the masks are defined within the boxes
                x1: int = min(xl1, xs1)
                y1: int = min(yl1, ys1)
                x2: int = max(xl2, xs2)
                y2: int = max(yl2, ys2)
                large_obj_mask: np.ndarray = np.zeros((y2 - y1, x2 - x1), np.uint8)
                large_obj_mask[(yl1 - y1):(yl2 - y1), (xl1 - x1):(xl2 - x1)] = detections['masks'][large_obj_index]
                small_obj_mask: np.ndarray = np.zeros((y2 - y1, x2 - x1), np.uint8)
                small_obj_mask[(ys1 - y1):(ys2 - y1), (xs1 - x1):(xs2 - x1)] = detections['masks'][small_obj_index]

                # check if the intersection of the masks are larger than the threshold, the box check and mask check allows us 
                # to run this code more efficiently 
                if np.sum(small_obj_mask * large_obj_mask) > overlap_threshold * np.sum(small_obj_mask):
                    # add column index j to the list of object indexes to be removed
                    if small_obj_index not in obj_idxs_to_remove:
                        obj_idxs_to_remove.append(small_obj_index)
            else:
                # add column index j to the list of object indexes to be removed without further checking as the masks are not
                # available
                if small_obj_index not in obj_idxs_to_remove:
                    obj_idxs_to_remove.append(small_obj_index)
                    
   
    if len(obj_idxs_to_remove) > 0:
        detections['boxes'] = [box for i, box in enumerate(detections['boxes']) if i not in obj_idxs_to_remove]
        detections['labels'] = [label for i, label in enumerate(detections['labels']) if i not in obj_idxs_to_remove]
        detections['scores'] = [score for i, score in enumerate(detections['scores']) if i not in obj_idxs_to_remove]
        if predict_masks:
            detections['masks'] = [mask for i, mask in enumerate(detections['masks']) if i not in obj_idxs_to_remove] 

    return detections


def run_model(
    detector: VisionModel,
    input_image: np.ndarray,
    input_resize: Dict[Tuple[int, int], Tuple[int, int]] | None = None, 
    input_crop_corners: Dict[Tuple[int, int], List[List[int]]] | None = None,
    normalize_image: bool = False, 
    bit_depth: int = 8, 
    crop: bool = True, 
    post_process_class_names: List[str] | None = None, 
    overlap_threshold: float = OVER_LAP_THRESHOLD,
    plot_results: bool = False, 
) -> Tuple[Dict[str, list], float, Optional[np.ndarray]]:

    """
    The wrapper function
    """

    predict_masks: bool = detector.get_metadata()['predict_masks']
    # make a copy to not modify the input image
    img = input_image.copy()

    if len(img.shape) > 2:
        logging.warning(
            f"Warning {detector.get_model_name()} model may suffer loss in precision due to conversion from RGB to grayscale"
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if bit_depth != 8:
        img = (255.0 * img.astype(float) / (2 ** bit_depth - 1)).astype(np.uint8)

    if normalize_image:
        img = cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    image_height, image_width = img.shape[:2]
    resize_dict: Dict[Tuple[int, int], Tuple[int, int]] | None = None
    crop_corners_dict: Dict[Tuple[int, int], List[List[int]]] | None = None

    # check if the model includes the resize and crop_corners dictionaries
    try:
        resize_dict, crop_corners_dict = detector.get_cropping_info()
    except Exception as ex:
        # the model class is old and not supporting the get_cropping_info()
        logging.info(
            f"{detector.__class__.__name__} class does not implement get_cropping_info(): {repr(ex)}"
        )
        
    if input_resize is not None and input_crop_corners is not None:
        # use the passed values is provided (they will overwrite if the model class also includes them), first warn the user
        if resize_dict is not None and crop_corners_dict is not None:
            logging.warning(
                "The cropping info provided in the model weights file are overwritten by the passed values"
            )
        resize_dict = input_resize
        crop_corners_dict = input_crop_corners

    if crop and (resize_dict is None or crop_corners_dict is None):
        logging.error("No cropping information to run! Unable to detect any objects")
        out = {'boxes': np.zeros((0, 4), dtype=int),
               'labels': np.zeros((0,), dtype=int),
               'scores': np.zeros((0,), dtype=float),
               }
        if predict_masks:
           out['masks'] = []
        
        if plot_results:
            return (out, 0, np.zeros((image_height, image_width), dtype=np.uint8))
        else:
            return (out, 0)


    if crop and (image_width, image_height) not in resize_dict:
        logging.error(
            "The input image size {} is not supported! Returning no cells!".format(
                image_width, image_height
            )
        )
        out = {'boxes': np.zeros((0, 4), dtype=int),
               'labels': np.zeros((0,), dtype=int),
               'scores': np.zeros((0,), dtype=float),
               }
        if predict_masks:
           out['masks'] = []
            
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
        resized_width, resized_height = int(image_width / scale_factor), int(image_height / scale_factor)

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
            img=resized_img, crop_corners=crop_corners, 
        )
    else:
        out: Dict[str, list] = detector.detect(img=resized_img)

    # post-process the results in the resized resolution
    out = post_process_detections(
        detections=out,  
        post_process_class_names=post_process_class_names,
        classnames_to_class_ids_map=detector.get_reverse_label_map(),
        overlap_threshold=overlap_threshold,
    )

    # resize back the detections
    if scale_factor != 1:
        # scale the detections back to original image resolution
        out['boxes'] = (scale_factor * np.array(out['boxes'])).astype(int)
        # convert to a list to be consistent with the rest
        out['boxes'] = [box for box in out['boxes']]
    else:
        out['boxes'] = [np.array(box) for box in out['boxes']]   

    if predict_masks and scale_factor != 1:
        for idx in range(len(out['boxes'])):
            xtl, ytl, xbr, ybr = out['boxes'][idx]
            # note that mask return by the detector is not a probability mask, but a binary mask
            out['masks'][idx] = cv2.resize(out['masks'][idx], (xbr - xtl, ybr - ytl), interpolation=cv2.INTER_NEAREST)
    
    
    et = time.time()

    if plot_results:
        return out, et - st, show_detections(img, out, detector._label_map)
    
    return out, et - st