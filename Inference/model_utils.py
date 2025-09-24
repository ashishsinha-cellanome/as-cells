import numpy as np
import torch
import cv2

from typing import List, Tuple, Dict, Final


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
