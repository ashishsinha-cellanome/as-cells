from AbstractVisionModel import VisionModel
from model_utils import to_numpy

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F

import os
import time
import logging
from collections import OrderedDict
from typing import Tuple, List, Final, Optional, Dict, Union

DEFAULT_DETECTION_CONFIDENCE: Final[float] = 0.5
DEFAULT_ANCHOR_SIZES: Final[Tuple[Tuple[int]]] = ((12,), (24,), (36,), (48,), (60,))
DEFAULT_MODEL_INPUT_SIZE: Final[Tuple[int, int]] = (1200, 800)
DEFAULT_MASK_THRESHOLD_FOR_BBOX_EXPANSION: Final[float] = 0.1
DEFAULT_MASK_THRESHOLD_FOR_BINARY_CONVERSION: Final[float] = 0.1
DEFAULT_BBOX_EXPANSION_FACTOR: Final[float] = 0.2
MAX_BBOX_EXPANSION_FACTOR: Final[float] = 0.25


# A dictionary with keys as the input (original) image size (width, height) tuple and
# values as the list of coordinates (xtl, ytl, xbr, ybr) of sub-images/crops
# to run YOLOv5 on each
# note that the crop coordinates are with respect to resized image dimensions specified above
DEFAULT_RESIZE: Final[Dict[Tuple[int, int], Tuple[int, int]]] = {
    (2000, 1600): (1280, 1024),
    (4512, 4512): (2144, 2144),
}
# A dictionary with keys as the input (original) image size (width, height) tuple and
# values as the list of coordinates (xtl, ytl, xbr, ybr) of sub-images/crops
# to run YOLOv5 on each
# note that the crop coordinates are with respect to resized image dimensions specified above
DEFAULT_CROP_CORNERS: Final[Dict[Tuple[int, int], List[List[int]]]] = {
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
class MaskRCNNInstanceSegmentation(VisionModel):
    def __init__(
        self,
        weights_path: str,
        model_name: str = 'Mask R-CNN', 
        label_map: Optional[Dict[int, str]] = None, # to be read from the weights file
        model_input_size: Optional[Tuple[int, int]] = DEFAULT_MODEL_INPUT_SIZE, # not included in the weights, not really needed for Mask R-CNN 
        confidence: float = DEFAULT_DETECTION_CONFIDENCE,
        device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        mask_threshold_for_binary_conversion: float = DEFAULT_MASK_THRESHOLD_FOR_BINARY_CONVERSION, 
        mask_threshold_for_bbox_expansion: float = DEFAULT_MASK_THRESHOLD_FOR_BBOX_EXPANSION
    ):
        self._mask_threshold_for_binary_conversion = mask_threshold_for_binary_conversion
        self._mask_threshold_for_bbox_expansion = mask_threshold_for_bbox_expansion
        
        super().__init__(
            weights_path,
            model_name, 
            label_map,
            model_input_size, 
            confidence,
            device, 
        )
         
    def load(self):

        # check if anchor sizes are specified in the model weights file
        saved_model_param: Union[OrderedDict, dict] = torch.load(
                self._weights_path, map_location=self._device
            )

        if isinstance(saved_model_param, dict) and 'anchor_sizes' in saved_model_param:
            self._anchor_sizes: Tuple[Tuple[int]] = saved_model_param['anchor_sizes']
        else:
            self._anchor_sizes = DEFAULT_ANCHOR_SIZES
        
        # Mask RCNN model
        self._model = get_instance_segmentation_model(
            num_classes=len(self._label_map) + 1, 
            anchor_sizes=self._anchor_sizes
        )

        # loading the PyTorch model
        try:
            self._model.load_state_dict(self._model_state_dict)
            self._model.to(self._device)
            self._model.eval()

        except Exception as ex:
            logging.error(
                f"Failed to load {self._model_name} model from the weights file: {repr(ex)}."
            )
            return

        self._metadata = {
            'predict_masks': True,
            'resolution': self._model_input_size, # not used in our code, included from the cropping sizes for completeness
            'release_date': os.path.basename(self._weights_path).split('_')[0], # the model name starts with the release date
            'model_type': 'Instance Segmentation',
            'model_name': self._model_name,
            'model_extra_info': 'Torchvision implementation',
            'names': self._label_map,
            'magnification': '4x' if '4x' in os.path.basename(self._weights_path) else '10x' # 4x should be specified in the name
        }
        
        self._loaded = True
    

    def detect_batch(self,
                     input_images_list: List[Union[Image.Image, np.ndarray]],
                    ) -> List[Dict[str, list]]:

        if not self._loaded:
            logging.error(
                "Mask R-CNN model has not been initialized. Please initialize the class before detect()."
            )
            
            out: List[Dict[str, list]] = [{"boxes": [], "scores": [], "labels": [], "masks": []}] * len(input_images_list)
            return out

        # convert the input images to tensors and scale them to [0, 1]
        # F.to_tensor takes care of it, however, make sure the passed images have bit-depth = 8
        # (is np.uint8 if numpy array)
        img_tensor_list: List[torch.tensor] = [F.to_tensor(im).unsqueeze(0).to(self._device) for im in input_images_list]
        img_tensors: torch.tensor = torch.cat(img_tensor_list, dim=0)
        with torch.no_grad(), torch.amp.autocast(self._device.type):
            outputs = self._model(img_tensors)

        # outputs is a list of dictionaries of four keys, 'boxes', 'labels', 'scores' and 'masks', with
        # each element in the list corresponding to one input image
        results = self.postprocess(outputs) 
        # clear the CUDA cache
        torch.cuda.empty_cache()
        return results
    
    def postprocess(
        self, 
        outputs, 
    ):
        
        results: List[Dict[str, list]] = []
        
        out: List[Dict[str, np.ndarray]] = [
            {"boxes": to_numpy(prediction["boxes"]).astype(int),
             "labels": to_numpy(prediction["labels"]),
             "scores": to_numpy(prediction["scores"])
            } 
            for prediction in outputs
        ]

        for sample_index in range(len(outputs)):
            
            sample_dict = {}
            # remap the output class IDs if needed
            if self._detected_class_ids_remap is not None and len(out[sample_index]["labels"]):
                out[sample_index]["labels"] = np.vectorize(
                    lambda x: (
                        self._detected_class_ids_remap[x]
                        if x in self._detected_class_ids_remap
                        else x
                    )
                )(out[sample_index]["labels"])
        
            # before moving the results to CPU for the masks, crop the masks within the detection boxes
            # to significantly reduce their sizes
            # for a large number of detected cells, 2/3 of the model runtime is
            # spent on moving these image-sized masks from GPU to CPU, reduce their sizes in GPU to
            # save time moving them back to CPU
            all_mask_tensors: torch.tensor = outputs[sample_index]["masks"]

            # lists to store final results (after thresholding and post-processing)
            # masks will no longer be the same size for each detection, hence we return a list
            # of numpy arrays to be consistent, we do the same (returning a list) for the rest
            boxes: List[List[int]] = []
            labels: List[int] = []
            scores: List[float] = []
            masks: List[np.ndarray] = []

            for i in range(out[sample_index]["boxes"].shape[0]):
                # skip unreliable or invalid detections
                (xtl, ytl, xbr, ybr) = out[sample_index]["boxes"][i]
                if out[sample_index]["scores"][i] < self._confidence or ytl >= ybr or xtl >= xbr:
                    continue
                if out[sample_index]["labels"][i] in self._class_ids_to_exclude_from_dets:
                    continue
                if self._mask_threshold_for_bbox_expansion > 0:
                    # threshold the mask to keep only the values more than the passed threshold
                    # then update the bounding box according to the remaining values
                    pos = torch.where(
                        all_mask_tensors[i, 0] >= self._mask_threshold_for_bbox_expansion
                    )
                    # TODO: Add this to the Mask R-CNN code when adding support for overlaid images
                    if pos[0].numel() == 0:
                        continue
                    
                    xmin: int = pos[1].min().item()
                    xmax: int = pos[1].max().item()
                    ymin: int = pos[0].min().item()
                    ymax: int = pos[0].max().item()
                    # apply some sanity checks on the expanded bounding box coordinates to avoid over expanding
                    if (0 < (xmax - xmin) <= (1 + MAX_BBOX_EXPANSION_FACTOR) * (xbr - xtl) and
                        0 < (ymax - ymin) <= (1 + MAX_BBOX_EXPANSION_FACTOR) * (ybr - ytl)):
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
                        xbr = min(all_mask_tensors[i, 0].shape[1], xbr + delta_x)
                        ybr = min(all_mask_tensors[i, 0].shape[0], ybr + delta_y)

                boxes.append([xtl, ytl, xbr, ybr])
                labels.append(out[sample_index]["labels"][i])
                scores.append(out[sample_index]["scores"][i])
                # cast as float because with autocast, the masks will be float16, which may not
                # be supported by some OpenCV functions
                prob_mask_this_cell = to_numpy(all_mask_tensors[i, 0, ytl:ybr, xtl:xbr]).astype(float)
                # threshold the probability mask to get a binary one
                mask_this_cell: np.ndarray = np.zeros(prob_mask_this_cell.shape, dtype=np.uint8)
                mask_this_cell[prob_mask_this_cell >= self._mask_threshold_for_binary_conversion] = 1
                masks.append(mask_this_cell.astype(np.uint8))
            
            sample_dict: Dict[str, List] = {
                "boxes": boxes,
                "scores" : scores,
                "labels": labels,
                "masks": masks
            }
            results.append(sample_dict)
        
        return results

