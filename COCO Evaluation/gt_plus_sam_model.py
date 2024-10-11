import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.transforms import functional as F
import os
import time
import logging

from segment_anything import sam_model_registry, SamPredictor
from typing import Tuple, List, Final, Optional, Dict, Union

DEFAULT_CLASS_IDS_TO_CLASSNAMES_MAP: Final[Dict[int, str]] = {1: 'cell', 2: 'bead',  3: 'cage', 5: 'cell-adhered', 6: 'soma'}
DEFAULT_PERCENTAGE_TO_EXPAND_BBOX_BOUNDARIES: Final[float] = 0.0

SAM_MODEL_CHECKPOINTS_PATH: Final[str] = '/home/cellareye/Cellanome/dl-mehdi/SAM/checkpoints'

SAM_MODEL_TYPE_TO_CHECKPOINT_MAP: Dict[str, str] = {'vit_b': 'sam_vit_b_01ec64.pth', 
                                                    'vit_l': 'sam_vit_l_0b3195.pth',
                                                    'vit_h': 'sam_vit_h_4b8939.pth'}

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

# crop function
def crop_annotations(sample, crop_coords, labels_of_interest=None, keep_area_threshold=0.95):
    """
    Crop the image in the passed sample for a given crop coordinates and keep or discard
    partial bounding boxes that lies inside the passed crop based on the passed keep_area_threshold. 

    Args:
        sample (dictionary): Input data sample to be cropped. The dictionary
            should include "name", "image", "annotations" and optionally "masks" keys 
            for passing the image name, the image in np.uint8 1/3-channel numpy array, 
            the bounding boxes pandas DataFrame (with columns 'xtl', 'ytl', 'xbr', 'ybr' and 'label') 
            and optionally the list of masks for annotated each object (each as a 
            numpy array of the same size the the bounding box specified in "annotations").
        crop_coords (4-tuple or 4-element list of int): xtl, ytl, xbr, and ybr 
            box coordinates for cropping.
        labels_of_interest (list of integers or strings or None): List of classnames or 
            class IDs of interest depending on the values reporeted in annotations_df['label'] 
            (names or IDs). Any annotated object outside this list will be removed from the 
            annotations. If None passed, all the object classes will be considereed. 
        keep_area_threshold (float): The threshold on the ratio of the area of the 
            bounding boxes that lie inside the cropped image to keep. Bounding boxes 
            with at least keep_area_threshold of their area inside the cropped image 
            will be kept. Otherwise, all the  bounding boxes crossing the boundaries 
            of the cropped image will be removed. 
    """
    
    if keep_area_threshold < 0.33:
        keep_area_threshold = 0.33
    
    x1, y1, x2, y2 = crop_coords
    # make a copy of the input to make sure it is not modified
    name, image, df = sample['name'], sample['image'].copy(), sample['annotations'].copy()
    
    h, w = image.shape[:2]
    
    xc1 = int(max(x1, 0))
    yc1 = int(max(y1, 0))
    xc2 = int(min(x2, w))
    yc2 = int(min(y2, h))
    
    if xc2 <= xc1 or yc2 <= yc1:
        # incorrect input dimensions
        return None
    
    # sizes of cropped image
    crop_width = xc2 - xc1
    crop_height = yc2 - yc1
    
    # remove the bounding boxes that are totally outside the cropped image
    # keep the ones that have some non-zero overlap 
    df = df[df.apply(lambda row: True if 
                     max(0, min(row['xbr'] - xc1, crop_width) - max(row['xtl'] - xc1, 0)) * \
                     max(0, min(row['ybr'] - yc1, crop_height) - max(row['ytl'] - yc1, 0)) > 0\
                     else False, axis = 1)]
    
    if 'masks' in sample:
    
        # identify the masks that would lie inside the newly cropped
        # image by more than keep_area_threshold; these masks and boxes are kept 
    
        # list of df indices to keep 
        idxs_to_keep = []
        # list of masks to keep (should correspond to the same indices in the df to keep)
        masks = []
        
        for obj_id in df.index:
        
            # ignore the object if not in the labels_of_interest (if specified) or if it should not be blocked
            if labels_of_interest is not None and df.loc[obj_id, 'label'] not in (labels_of_interest):
                continue
            
            # find the overlapping part between the object's bounding box (where the mask is defined within)
            # and the crop
            box_xtl, box_ytl, box_xbr, box_ybr = df.loc[obj_id, ['xtl', 'ytl', 'xbr', 'ybr']].values
            # the upper bound for xmin, ymin (the outher min) is not really needed becuase
            # the DataFrame is already filtered to keep overlapping bounding boxes with the crop 
            # with xc1 < box_xbr and yc1 < box_ybr for df.index
            # similarly, the lower bound of 0 for xmax, ymax (the inner max) is not needed 
            # becuase the DataFrame is already filtered and box_xtl < xbr2 and box_ytl < yc2 for df.index
            xmin = min(max(0, xc1 - box_xtl), box_xbr - box_xtl)
            xmax = min(max(0, xc2 - box_xtl), box_xbr - box_xtl)
            ymin = min(max(0, yc1 - box_ytl), box_ybr - box_ytl)
            ymax = min(max(0, yc2 - box_ytl), box_ybr - box_ytl)
            
            cropped_mask = sample['masks'][obj_id][ymin: ymax, xmin: xmax]
            
            # now update the bounding boxes as the mask confined to the crop may be smaller, hence different box 
            pos = np.where(cropped_mask)
            if len(pos[0]) == 0 or len(pos[1]) == 0:
                continue
            
            # Note that the configuration percentage_to_expand_bbox_boundaries might have been applied
            # to the bounding boxes of the original annotations (with respect to the mask) in parse_json_annotations
            # function
            # the if conditions are included below to only update the bounding box if it crosses the crop boundary
            # we only include +1 pixel expansion that is always applied by default
            if xmin > 0:
                # the left side of the bounding box crosses the crop boundary, expand by -1 as we always do by default
                delta_x1 = max(0, np.min(pos[1]) - 1)
            else:
                delta_x1 = xmin

            if xmax < box_xbr - box_xtl:
                # the right side of the bounding box crosses the crop boundary, expand by +1 as we always do by default
                delta_x2 = min(np.max(pos[1]) + 1 + 1,
                               box_xbr - box_xtl)  # the first +1 is included as we need to include this point
            else:
                delta_x2 = xmax

            if ymin > 0:
                # the top side of the bounding box crosses the crop boundary, expand by -1 as we always do by default
                delta_y1 = max(0, np.min(pos[0]) - 1)
            else:
                delta_y1 = ymin

            if ymax < box_ybr - box_ytl:
                # the bottom side of the bounding box crosses the crop boundary, expand by +1 as we always do by default
                delta_y2 = min(np.max(pos[0]) + 1 + 1,
                               box_ybr - box_ytl)  # the first +1 is included as we need to include this point
            else:
                delta_y2 = ymax
            
            if delta_x1 >= delta_x2 or delta_y1 >= delta_y2:
                continue
                
            cropped_mask_area = cropped_mask[delta_y1:delta_y2, delta_x1:delta_x2].sum()
            mask_area = sample['masks'][obj_id].sum()
            
            if cropped_mask_area >= keep_area_threshold * mask_area:
                idxs_to_keep.append(obj_id)
                masks.append(cropped_mask[delta_y1:delta_y2, delta_x1:delta_x2].copy())
            
            # update the bounding box around the mask part that lies inside the crop (even for block_label objects, 
            # we do not want to block more than needed)    
            new_box_xtl = box_xtl + xmin + delta_x1
            new_box_xbr = box_xtl + xmin + delta_x2
            new_box_ytl = box_ytl + ymin + delta_y1
            new_box_ybr = box_ytl + ymin + delta_y2
            
            df.at[obj_id, 'xtl'] = new_box_xtl
            df.at[obj_id, 'ytl'] = new_box_ytl
            df.at[obj_id, 'xbr'] = new_box_xbr
            df.at[obj_id, 'ybr'] = new_box_ybr
                
        
        # transform the bounding boxes (needed below)
        df[['xtl', 'xbr']] = df[['xtl', 'xbr']] - xc1
        df[['ytl', 'ybr']] = df[['ytl', 'ybr']] - yc1   
    else: 
        # only bounding boxes are included in the annotations
        # use them to identify the objects that would lie in the cropped image       
        # transform the bounding boxes
        df[['xtl', 'xbr']] = df[['xtl', 'xbr']] - xc1
        df[['ytl', 'ybr']] = df[['ytl', 'ybr']] - yc1

        # identify bounding boxes that should be kept, left or blocked following 
        # the same logic as masks
    
        if labels_of_interest is None:
            # consider all objects
            idxs_to_keep = df.apply(lambda row: True \
                                    if (min(row['xbr'], crop_width) - max(row['xtl'], 0)) * 
                                    (min(row['ybr'], crop_height) - max(row['ytl'], 0)) >= keep_area_threshold * 
                                    (row['xbr'] - row['xtl']) * (row['ybr'] - row['ytl']) else False, axis=1)
        else:
            # if labels_of_interest is provided, use it to only keep the ones we need to keep, IT SHOULD NOT INCLUDE block_label
            idxs_to_keep = df.apply(lambda row: True \
                                    if (min(row['xbr'], crop_width) - max(row['xtl'], 0)) * 
                                    (min(row['ybr'], crop_height) - max(row['ytl'], 0)) >= keep_area_threshold * 
                                    (row['xbr'] - row['xtl']) * (row['ybr'] - row['ytl']) and 
                                    row['label'] in labels_of_interest else False, axis=1)
        # convert to indices of df                             
        idxs_to_keep = df[idxs_to_keep].index
    
    # limit the bounding boxes to image coordinates
    df.loc[df['xtl'] < 0, 'xtl'] = 0
    df.loc[df['ytl'] < 0, 'ytl'] = 0
    df.loc[df['xbr'] > crop_width, 'xbr'] = crop_width
    df.loc[df['ybr'] > crop_height, 'ybr'] = crop_height
    
    
    # reset the indexes in the annotations DataFrame
    df = df.loc[idxs_to_keep].reset_index(drop=True)
    
    if len(image.shape) > 2:
        # 3 channel image
        crop_mask = crop_mask[:, :, np.newaxis]
    
    if 'masks' in sample:
        return  {'name': name, 'image': image[yc1: yc2, xc1: xc2], 
                 'annotations': df, 'masks': masks} 
    
    return  {'name': name, 'image': image[yc1: yc2, xc1: xc2], 'annotations': df}

def resize_annotations(sample: dict, scale_factor: float):
    
    if scale_factor != 1:
        img = sample['image'].copy()
        annotations = sample['annotations'].copy()
        image_height, image_width = img.shape[:2]
        
        # for decimating an image, cv2.INTER_AREA is the preferred method (scale_factor is always > 1)
        img = cv2.resize(img, (int(image_width / scale_factor), int(image_height / scale_factor)), interpolation=cv2.INTER_AREA)
        # update all the masks and annotations
        annotations[['xtl', 'ytl', 'xbr', 'ybr']] = annotations[['xtl', 'ytl', 'xbr', 'ybr']].div(scale_factor).astype(int)

        # make sure no box width/height becomes zero after the resize
        # only keep boxes with positive width and height
        annotations = annotations[(annotations['ybr'] - annotations['ytl'] > 0) & (annotations['xbr'] - annotations['xtl'] > 0)]

        # keep the corresponding masks after resizing and make a copy
        masks: List[np.ndarray] = [sample['masks'][i].copy() for i in annotations.index]
        # reset the index
        annotations.reset_index(inplace=True, drop=True)

        # now resize the masks (note that they are defined within the bounding boxes)
        for idx in range(len(masks)): 
            box_xtl, box_ytl, box_xbr, box_ybr = annotations.loc[idx, ['xtl', 'ytl', 'xbr', 'ybr']].values
            masks[idx] = cv2.resize(masks[idx], (box_xbr - box_xtl, box_ybr - box_ytl), interpolation=cv2.INTER_NEAREST)

        resized_sample = {}
        resized_sample['name'] = sample['name']
        resized_sample['image'] = img
        resized_sample['annotations'] = annotations
        resized_sample['masks'] = masks
        return resized_sample

    return sample


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
    yolo_input_size: int = 640,
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
        yolo_input_size (int): The model input size (assuming it is YOLO and accept square images).
    Returns:
        List of 4-tuples or 4-elements integer coordinates of the sub-mages/crops covering the original image.
    """
    # the step size for the starting point of each crop in x and y dimension
    crop_start_step_x: int = yolo_input_size - overlap_in_x
    crop_start_step_y: int = yolo_input_size - overlap_in_y
    crop_corners: List = []

    # overlapping crops
    for x_start in range(0, image_width - overlap_in_x, crop_start_step_x):
        for y_start in range(0, image_height - overlap_in_y, crop_start_step_y):
            # crop coordinates
            xc_tl = x_start
            yc_tl = y_start
            xc_br = x_start + yolo_input_size
            yc_br = y_start + yolo_input_size
            # make sure we always crop the image with the given size
            # if we get to the boundaries, extend the crop
            # size inside the image to always get the same size crop
            # this is not really needed, but help with capturing more
            # annotations toward the low/right parts of the image
            if xc_br > image_width:
                xc_br = image_width
                xc_tl = xc_br - yolo_input_size
            if yc_br > image_height:
                yc_br = image_height
                yc_tl = yc_br - yolo_input_size

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

def load_sam_model(sam_checkpoints_path: str, model_type: str):
    """ A function to load the SAM model 
    Args:
        sam_checkpoints_path (str): Path to the SAM model checkpoints. 
        model_type (str): The encoder model architecture, can be 'vit_b', 'vit_l' or 'vit_h'.
    Returns the segment_anything.modeling.sam.Sam object
    """
    if model_type not in SAM_MODEL_TYPE_TO_CHECKPOINT_MAP:
        logging.error(f"Invalid SAM model_type: {model_type}! Impossible to instantiate the SAM model. Returning None ...")
        return None
    
    sam_checkpoint: str = os.path.join(sam_checkpoints_path, SAM_MODEL_TYPE_TO_CHECKPOINT_MAP[model_type])
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    return sam


# Ground truths + SAM segmentation class
class GtPlusSamInstanceSegmentation:
    def __init__(
        self,
        label_map: Optional[Dict[int, str]] = DEFAULT_CLASS_IDS_TO_CLASSNAMES_MAP,
        sam_checkpoints_path: Optional[str] = SAM_MODEL_CHECKPOINTS_PATH,
        sam_model_type: Optional[str] = 'vit_b',
        percentage_to_expand_bbox_boundaries: float = DEFAULT_PERCENTAGE_TO_EXPAND_BBOX_BOUNDARIES,
    ):
        self._sam_checkpoints_path: str = str(sam_checkpoints_path)
        self._sam_model_type: str = str(sam_model_type)
        self._percentage_to_expand_bbox_boundaries: float = percentage_to_expand_bbox_boundaries
            
        # available device
        self._device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
              
        self._label_map: Dict[int, str] = label_map

        logging.info(f"Passed mapping between class IDs and class names: {self._label_map}") 
        if self._label_map is not None:
            self._reverse_label_map: Dict[str, int] = {
                value: key for key, value in self._label_map.items()
            }
        else:
            self._reverse_label_map = None
        
        # SAM model
        self._sam = load_sam_model(self._sam_checkpoints_path, self._sam_model_type)

        if self._sam is None:
            self._sam_detector = None
            logging.error(
                f"Failed to load SAM model: {repr(ex)}."
            )
            return
        
        self._sam.to(device=self._device)
        self._sam_detector = SamPredictor(self._sam)

    def detect(self,
               sample: dict,
               log_time: bool = False
               ) -> Dict[str, list]:
        """
        The main function to extract masks for the image and bounding boxes passed to the function. 

        Args:
            sample (dictionary): Input data sample, an image with the annotations. The dictionary
                should include "name", "image", "annotations" and optionally "masks" keys 
                for passing the image name, the image in np.uint8 1/3-channel numpy array, 
                the bounding boxes pandas DataFrame (with columns 'xtl', 'ytl', 'xbr', 'ybr' and 'label') 
                and optionally the list of masks for annotated each object (each as a 
                numpy array of the same size the the bounding box specified in "annotations"). The bounding
                boxes from the annotations are extracted and passed as prompts to SAM for extracting masks. 
            log_time (bool): A flag to log the model run time.

        Returns:
            A dictionary with keys and values as below:
                "boxes": List of 4-tuples or 4-element lists for the detected objects' bounding boxes in
                    xtl, ytl, xbr, ybr format/order. These are the same as bounding boxes passed in sample['annotations'].
                "labels": List of integer class IDs for the detected objects. These are the same as labels passed 
                    in sample['annotations'].
                "scores": List of float detection scores. Always 1 in this case, we we use the ground truths from 
                    annotations
                "masks": List of masks extracted by SAM for the passed prompt bounding boxes. Each mask is a numpy array 
                    of the same size as the bounding box with width and height (xbr - xtl, ybr - ytl).
        """

        if self._sam_detector is None:
            logging.error(
                "SAM model has not been initialized. Please initialize the class before detect()."
            )
            out: Dict[str, list] = {"boxes": [], "scores": [], "labels": [], "masks": []}
            return out

        start: float = time.time()

        # extract the bounding boxes and labels from the annotations
        boxes: np.ndarray = sample['annotations'][['xtl', 'ytl', 'xbr', 'ybr']].values
        # expand the bounding boxes if self._percentage_to_expand_bbox_boundaries is more than 0
        if self._percentage_to_expand_bbox_boundaries > 0:
            # expand the box boundaries by a few pixels
            delta_x: np.ndarray = (self._percentage_to_expand_bbox_boundaries * (boxes[:, 2] - boxes[:, 0]) / 2.0).astype(int)
            delta_y: np.ndarray = (self._percentage_to_expand_bbox_boundaries * (boxes[:, 3] - boxes[:, 1]) / 2.0).astype(int)
            
            boxes[:, 0] = np.maximum(0, boxes[:, 0] - delta_x)
            boxes[:, 1] = np.maximum(0, boxes[:, 1] - delta_y)
            boxes[:, 2] = np.minimum(image_width, boxes[:, 2] + delta_x)
            boxes[:, 3] = np.minimum(image_height, boxes[:, 3] + delta_y)
         
        # class IDs from the annotations
        labels: np.ndarray = sample['annotations']['label'].values
        scores: np.ndarray = np.array([1.0] * len(labels))

        out: Dict[str, np.ndarray] = {"boxes": boxes.astype(int),
                                      "labels": labels,
                                      "scores": scores}
        
        masks: List[np.ndarray] = []

        # SAM expects a 3 channel image
        if len(sample['image'].shape) < 3:
            input_image: np.ndarray = cv2.cvtColor(sample['image'], cv2.COLOR_GRAY2RGB)
        else:
            input_image = sample['image']

        elap: float = time.time() - start
        check_point: float = time.time()
        if log_time:
            logging.info(f"Preprocessing took {elap:.4f} seconds")

        # extract the SAM model embedding (SAM encoder)
        self._sam_detector.set_image(input_image)

        elap: float = time.time() - check_point
        check_point: float = time.time()
        if log_time:
            logging.info(f"Extracting SAM's embeddings took {elap:.4f} seconds")

        
        # SAM decoder
        # extract masks for 100 boxes at a time to make sure we are not running out of GPU memory
        num_boxes_step_size: int = 100
        for i in range(0, len(scores), num_boxes_step_size):
            start_index : int = i
            end_index = min(i + num_boxes_step_size, len(scores))
            input_boxes = torch.tensor(boxes[start_index:end_index, :], device=self._device)
            transformed_boxes = self._sam_detector.transform.apply_boxes_torch(input_boxes, input_image.shape[:2])  
            mask_tensors, iou_predictions, low_res_masks = self._sam_detector.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            for i, box_tensor in enumerate(input_boxes):
                # confine the mask to the bounding box and move to CPU before converting to numpy arrays
                masks.append(mask_tensors[i, 0, box_tensor[1]: box_tensor[3], box_tensor[0]: box_tensor[2]].cpu().numpy().astype(np.uint8))
        

        elap: float = time.time() - check_point
        check_point: float = time.time()
        if log_time:
            logging.info(f"SAM mask prediction took {elap:.4f} seconds")

        
        elap: float = time.time() - start
        if log_time:
            logging.info(f"GT + SAM instance segmentation took {elap:.4f} seconds")

        out: Dict[str, List] = {"boxes": boxes,
                                "scores" : scores,
                                "labels": labels,
                                "masks": masks}
        return out

    def get_label_map(self):
        return self._label_map
    
    def get_reverse_label_map(self):
        return self._reverse_label_map
    
    
    def detect_by_cropping(
            self,
            sample: dict,
            crop_corners: List[List[int]],
            nms_threshold_for_combining_crop_results: float = 0.15,
            classnames_to_return: Optional[List[str]] = None,
            log_time=False,
    ) -> Dict[str, List]:
        """
        A function to apply the SAM model on a high resolution image. If the
        image is high resolution with many objects, after resizing the image
        to match the models input sizes, the objects may become too small for
        reliable detection. This function evaluate an image by first cropping
        the image into smaller and potentially overlapping sub-images (as
        specified by cropCorners), running the detector on each sub-image and
        then combining the detections over multiple overlapping sub-images
        by applying NMS
        Args:
            sample (dictionary): Input data sample, an image with the annotations. The dictionary
                should include "name", "image", "annotations" and optionally "masks" keys 
                for passing the image name, the image in np.uint8 1/3-channel numpy array, 
                the bounding boxes pandas DataFrame (with columns 'xtl', 'ytl', 'xbr', 'ybr' and 'label') 
                and optionally the list of masks for annotated each object (each as a 
                numpy array of the same size the the bounding box specified in "annotations"). The bounding
                boxes from the annotations are extracted and passed as prompts to SAM for extracting masks. 
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
                    xtl, ytl, xbr, ybr format/order. These are the same as bounding boxes passed in sample['annotations'].
                "labels": List of integer class IDs for the detected objects. These are the same as labels passed 
                    in sample['annotations'].
                "scores": List of float detection scores. Always 1 in this case, we we use the ground truths from 
                    annotations
                "masks": List of masks extracted by SAM for the passed prompt bounding boxes. Each mask is a numpy array 
                    of the same size as the bounding box with width and height (xbr - xtl, ybr - ytl).
        """

        start = time.time()

        # invalid output
        invalid_out: dict = {"boxes": [], "scores": [], "labels": [], "masks": []}

        if len(sample['annotations']) == 0:
            # crop_annotations function requires non-empty sample['annotations']
            # no bounding boxes here, return no results
            return invalid_out

        if len(crop_corners) == 0:
            logging.error(
                "No crop corners are provided for running YOLOv5 model on sub-images. "
                "Returning no detections ..."
            )
            return invalid_out

        if classnames_to_return is not None and self._reverse_label_map is not None:
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
            if self._label_map is not None:
                class_ids_to_return: List[int] = list(self._label_map.keys())
            else:
                class_ids_to_return = None

        img: np.ndarray = sample['image']

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
        results: Dict = {"scores": [], "boxes": [], "labels": [], "masks": []}

        # a list to keep track of cropped sub-images with at least one object detection
        crop_ids_with_detection: List[int] = []

        for crop_id, corners in enumerate(crop_corners):
            (x1c, y1c, x2c, y2c) = corners
            # enlarge the crop if necessary to make all the same size
            x2c = x1c + crop_width
            y2c = y1c + crop_height

            cropped_sample: dict = crop_annotations(sample=sample, 
                                                    crop_coords=(x1c, y1c, x2c, y2c), 
                                                    labels_of_interest=class_ids_to_return, 
                                                    keep_area_threshold=0.95)

            cropped_image = cropped_sample['image']

            preds = self.detect(cropped_sample)

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
                ] = 0.5

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
                            masks.append(crop_masks[crop_class_idxs[0][i]])
                            if crop_det_score == rest_det_score:
                                # this is added to break the tie if both areas are equal
                                # so we will not add the same box twice when considering
                                # in another crop
                                results["scores"][idx][crop_class_idxs[0][i]] += 1e-5

        elap: float = time.time() - start
        if log_time:
            logging.info(
                "Mask R-CNN instance segmentation after cropping the image to "
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

        
detector = GtPlusSamInstanceSegmentation()

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

# threshold for post-processing cells and remove the ones consisting of multiple smaller cells
OVER_LAP_THRESHOLD: Final[float] = 0.75

def run_gt_plus_sam(
    sample: dict,
    classnames_mapping_dict = None, 
    post_process_class_names: List[str] = [], 
    plot_results: bool = False, 
) -> Tuple[Dict[str, list], float, Optional[np.ndarray]]:
    # make a copy to not modify the input image
    img = sample['image'].copy()

    image_height, image_width = img.shape[:2]

    if (image_width, image_height) not in RESIZE:
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

    
    # we keep the aspect ratio in RESIZE dictionary, scale_factor is the same for both dimensions
    scale_factor: float = image_width / RESIZE[(image_width, image_height)][0]
    resized_width, resized_height = RESIZE[(image_width, image_height)]
   
    resized_sample: dict = resize_annotations(sample, scale_factor)

    st = time.time()

    crop_corners: List[List[int]] = CROP_CORNERS[(image_width, image_height)]
    out: Dict[str, list] = detector.detect_by_cropping(sample=resized_sample, crop_corners=crop_corners)
   

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
            # note that mask here is NOT a probability mask and interpolation does not have to be nearest neighbor
            out['masks'][idx] = cv2.resize(out['masks'][idx], (xbr - xtl, ybr - ytl), interpolation=cv2.INTER_NEAREST)

    if classnames_mapping_dict is not None and detector._reverse_label_map is not None:
        classnames_to_exclude: List[str] = [name for name, mapped_name in classnames_mapping_dict.items() if mapped_name == 'bg']
        class_ids_to_exclude: List [int] = [detector._reverse_label_map[name] for name in classnames_to_exclude]
        class_ids_mapping_dict = {detector._reverse_label_map[name]: detector._reverse_label_map[mapped_name] 
                                  for name, mapped_name in classnames_mapping_dict.items() if mapped_name != 'bg'}


        labels: List[int] = []
        idxs_to_keep: List[int] = []

        for idx, label in enumerate(out['labels']):
            if label in class_ids_to_exclude:
                continue
            if label in class_ids_mapping_dict:
                labels.append(class_ids_mapping_dict[label])
            else:
                labels.append(label)
            idxs_to_keep.append(idx)
    
        out['boxes'] = [box for idx, box in enumerate(out['boxes']) if idx in idxs_to_keep]
        out['labels'] = labels
        out['scores'] = [score for idx, score in enumerate(out['scores']) if idx in idxs_to_keep]
        out['masks'] = [mask for idx, mask in enumerate(out['masks']) if idx in idxs_to_keep]
    
    # post-process the results
    # in the following, "larger" objects that consist of a number of already detected smaller objects of the same type are invalidated
    # this can happen mainly for 'cell', 'nucleus' and 'cell-adhered'/'cytoplasm' classes
    # list of indexes of objects for each class name to be included in post processing
    post_process_class_idxs: Dict[str, List[int]] = {}
    # list of bounding boxes for each class name to be included in post processing
    post_process_class_boxes: Dict[str, List[np.ndarray]] = {}
    for i, box in enumerate(out['boxes']):
        for class_name in post_process_class_names:
            if detector._reverse_label_map is not None and class_name in detector._reverse_label_map and out['labels'][i] == detector._reverse_label_map[class_name]:
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
        return out, et - st, show_detections(img, out, detector.get_label_map())
    
    return out, et - st