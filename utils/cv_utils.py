import cv2
import numpy as np
from PIL import Image
from typing import Tuple, List, Final, Optional, Dict, Union
# from .precision_recall_eval import AnnotationFilter
# from .pairing_utils import pair_gts_dets_bbox, pair_gts_dets_mask

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


def overlap_mask_pair(box1: np.ndarray,
                      mask1: np.ndarray,
                      box2: np.ndarray,
                      mask2: np.ndarray,
                      ordered: bool = False) -> float:
    """
    Given two np.uint8 M1xN1 and M2xN2 numpy arrays (mask1, mask2) for two object masks and two (4,) (4-element)
    integer numpy arrays of the top-left/bottom-right corners of the bounding boxes around these objects (box1, box2),
    the code returns the overlap between the masks of the two objects. The overlap between the two objects is defined
    as:
    - the intersection area between the masks over the smallest area between the two masks when ordered set to False
    - the intersection area between the masks over the area of mask2 when ordered set to True
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
    intersection: int = cv2.bitwise_and(union_mask_1, union_mask_2).sum()

    if ordered:
        # use the mask area of mask2/bboxes2 as the denominator
        smallest_mask_area: float = mask2.sum()
    else:
        smallest_mask_area: float = min(mask1.sum(), mask2.sum()) + 1e-30

    return intersection / smallest_mask_area


def get_num_partial_objects(boxes: np.ndarray,
                            masks: List[np.ndarray],
                            labels: Union[List[int], List[str]],
                            crop_coords: Union[List[int], Tuple[int]],
                            w: int, h: int,
                            labels_of_interest: Union[List[int], List[str]],
                            partial_object_area_threshold: float = 0.25) -> int:
    (x1, y1, x2, y2) = crop_coords
    # make sure the crop is within the image and corners are integer
    x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return -1

    crop_box: np.ndarray = np.array([x1, y1, x2, y2]).astype(int)
    crop_mask = np.ones((y2 - y1, x2 - x1), dtype=np.uint8)

    # to enable the code to run faster, we first compute the ovarlap between the crop and the boxes,
    # find the potentially overlapping ones and then calculate the overlap between the crop and the masks
    overlap_box: np.ndarray = overlap_batch(np.array([crop_box]), boxes, True)

    _, overlapping_idxs = np.where(overlap_box > 0)

    num_partial_objects: int = 0

    for idx in overlapping_idxs:

        if labels[idx] not in labels_of_interest:
            continue

        overlap = overlap_mask_pair(box1=crop_box,
                                    mask1=crop_mask,
                                    box2=boxes[idx],
                                    mask2=masks[idx],
                                    ordered=True)

        if overlap >= 1 - partial_object_area_threshold or overlap < partial_object_area_threshold:
            continue

        num_partial_objects += 1

    return num_partial_objects

# a function to return the area of bounding box
def box_area(box: np.array) -> float:
    """
    Args:
        box (numpy array of size (4,) or (4, 1) or (1, 4) or a 4-tuple or a 4-elements list): The box.
    Return the area.
    """
    return (box[3] - box[1]) * (box[2] - box[0])


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


def show_detections(input_image: Union[Image.Image, np.array], predictions: Dict[str, Union[list, np.array]],
                    label_map: Dict[int, str]):
    """
    A function to display the predictions from the YOLO or Mask R-CNN model. 
    
    """

    # colors for displaying bounding boxes
    COLORS: List[Tuple[int, int, int]] = [
        (0, 0, 255),
        (255, 0, 0),
        (0, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 0),
    ]

    class_ids: List[int] = list(label_map.keys())
    if isinstance(input_image, np.ndarray):
        image: np.array = input_image.copy()
    else:
        # convert to a numpy array
        image: np.array = np.array(input_image)

    # convert to 3-channels
    if len(image.shape) < 3:
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2)

    boxes = predictions['boxes']
    labels = predictions['labels']
    if 'masks' in predictions:
        masks = predictions['masks']

    for i in range(len(labels)):
        # the bounding box
        (xtl, ytl, xbr, ybr) = boxes[i]
        if labels[i] not in class_ids:
            # use black for incorrect label
            color = (0, 0, 0)
            text = "Unknown class ID %s" % labels[i]
        else:
            color = COLORS[labels[i] % len(COLORS)]
            text = label_map[labels[i]]

        if 'masks' in predictions:
            color_mask = color * np.repeat(np.expand_dims(masks[i], axis=2), 3, axis=2)
            blended = 0.4 * color_mask
            blended[color_mask == 0] = image[ytl:ybr, xtl:xbr][color_mask == 0]
            blended[color_mask > 0] += 0.6 * image[ytl:ybr, xtl:xbr][color_mask > 0]

            # store the blended ROI in the original image
            image[ytl:ybr, xtl:xbr] = blended.astype(np.uint8)

        cv2.rectangle(image, (xtl, ytl), (xbr, ybr), color, 1)
        cv2.putText(
            image, text, (xtl, ytl + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )
    return image

def show_detections_with_gt(
    input_image: Union[Image.Image, np.array],
    detections: Dict[str, Union[list, np.array]],
    label_map: Dict[int, str],
    pred: bool = False
):
    """
    Displays detections on an image with a single color.

    Uses GREEN for Ground Truth (pred=False) and RED for Predictions (pred=True).

    Args:
        input_image: The image to draw on.
        detections: A dictionary containing 'boxes', 'labels', and optionally 'masks'.
        label_map: A dictionary mapping class IDs to class names.
        pred (bool): If True, detections are drawn in red. If False (default), 
                     they are drawn in green.
    """
    # Define colors in BGR format for OpenCV
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)

    # Select the color based on the 'pred' flag
    color = RED if pred else GREEN
    if isinstance(input_image, np.ndarray):
        image = input_image.copy()
    else:
        # Convert PIL Image to a numpy array
        image = np.array(input_image)

    # Convert to 3-channels if the image is grayscale
    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Ensure image is in BGR format if it's from PIL (which loads as RGB)
    if isinstance(input_image, Image.Image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    boxes = detections['boxes']
    labels = detections['labels']

    for i in range(len(labels)):
        # Get integer coordinates for the bounding box
        xtl, ytl, xbr, ybr = map(int, boxes[i])
        
        # Get the class name for the label text, with a fallback for unknown IDs
        text = label_map.get(labels[i], f"Unknown ID: {labels[i]}")

        # Handle masks if they exist
        if 'masks' in detections:
            mask = detections['masks'][i]
            # Ensure mask is a boolean array for indexing
            bool_mask = mask.astype(bool)
            
            # Create a solid color overlay
            color_overlay = np.zeros_like(image[ytl:ybr, xtl:xbr])
            color_overlay[bool_mask, :] = color
            
            # Blend the overlay with the image region
            roi = image[ytl:ybr, xtl:xbr]
            blended_roi = cv2.addWeighted(roi, 0.6, color_overlay, 0.4, 0)
            
            # Apply the blended result only where the mask is active
            roi[bool_mask] = blended_roi[bool_mask]

        # Draw the bounding box and the label text
        line_thickness = 1 if pred==False else 2
        cv2.rectangle(image, (xtl, ytl), (xbr, ybr), color, line_thickness)
        cv2.putText(
            image, text, (xtl, ytl - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, line_thickness
        )
        
    return image

# Helper function to draw a dashed rectangle (no changes needed)
def _draw_dashed_rectangle(img, pt1, pt2, color, thickness=2, dash_length=10):
    """Draws a dashed rectangle on an image."""
    x1, y1 = pt1
    x2, y2 = pt2
    # Top and bottom lines
    for i in range(x1, x2, dash_length * 2):
        cv2.line(img, (i, y1), (min(i + dash_length, x2), y1), color, thickness)
        cv2.line(img, (i, y2), (min(i + dash_length, x2), y2), color, thickness)
    # Left and right lines
    for i in range(y1, y2, dash_length * 2):
        cv2.line(img, (x1, i), (x1, min(i + dash_length, y2)), color, thickness)
        cv2.line(img, (x2, i), (x2, min(i + dash_length, y2)), color, thickness)

def visualize_model_errors_with_official_pairing(
    image: Union[Image.Image, np.ndarray],
    ground_truth_original: Dict,
    predictions: Dict,
    label_map: Dict[int, str],
    min_iou: float = 0.5,
    use_mask: bool = False,
    annotation_filter = None,
    show_original_labels: bool = False
) -> np.ndarray:
    """
    visualization function to pair bboxes (masks if available) and plt TP, FP, FN

    Usage:
    datasample = test_dataset[idx]
    prediction = model(datasample['image']))

    err_img = visualize_model_errors_with_official_pairing(
        datasample['image'], # image
        datasample, # gt sampled from CellMaskDataset
        prediction, # prediction from the model
        label_map = model.get_label_map(),
        min_iou=0.5,
        annotation_filter=None, # optional AnnotationFilter object to filter ground truth annotations 
        show_original_labels=False
    )
    Image.fromarray(err_img) to view the image
    """
    COLORS: List[Tuple[int, int, int]] = [
        (0, 0, 255), (255, 0, 0), (0, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 255, 0), (128, 0, 128)
    ]
    vis_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) if isinstance(image, Image.Image) else image.copy()
    if len(vis_image.shape) < 3:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)

    ground_truth_eval = annotation_filter.apply(ground_truth_original) if annotation_filter else ground_truth_original

    gt_boxes_eval = ground_truth_eval['annotations'][['xtl', 'ytl', 'xbr', 'ybr']].values.astype(int)
    gt_labels_eval = ground_truth_eval['annotations']['label'].values
    gt_masks_eval = ground_truth_eval.get('masks')
    
    original_indices = ground_truth_eval['annotations'].index
    original_labels_for_eval_set = ground_truth_original['annotations'].loc[original_indices, 'label'].values

    pred_boxes = np.array(predictions['boxes']).astype(int) if 'boxes' in predictions and len(predictions['boxes']) > 0 else np.zeros((0, 4), dtype=int)
    pred_labels = np.array(predictions['labels']) if 'labels' in predictions and len(predictions['labels']) > 0 else np.zeros((0,), dtype=int)
    pred_scores = np.array(predictions['scores']) if 'scores' in predictions and len(predictions['scores']) > 0 else np.zeros((0,), dtype=float)
    pred_masks = predictions.get('masks')

    class_ids_to_filter = list(label_map.keys())
    
    all_paired_gt_indices = set()
    all_paired_det_indices = set()
    all_unpaired_gt_indices = set(range(len(gt_labels_eval)))
    all_unpaired_det_indices = set(range(len(pred_labels)))

    for class_id in class_ids_to_filter:
        if class_id == label_map.get('bg', -1): continue # Skip bg class

        gt_class_indices = np.where(gt_labels_eval == class_id)[0]
        det_class_indices = np.where(pred_labels == class_id)[0]

        if len(gt_class_indices) == 0 and len(det_class_indices) == 0:
            continue
            
        gt_boxes_class = gt_boxes_eval[gt_class_indices]
        det_boxes_class = pred_boxes[det_class_indices]

        if use_mask and gt_masks_eval and pred_masks:
            gt_masks_class = [gt_masks_eval[i] for i in gt_class_indices]
            det_masks_class = [pred_masks[i] for i in det_class_indices]
            paired, unpaired_gts, unpaired_dets = pair_gts_dets_mask(
                gt_boxes_class, gt_masks_class, det_boxes_class, det_masks_class, min_iou
            )
        else:
            paired, unpaired_gts, unpaired_dets = pair_gts_dets_bbox(
                gt_boxes_class, det_boxes_class, min_iou
            )

        for gt_local_idx, det_local_idx in paired:
            global_gt_idx = gt_class_indices[gt_local_idx]
            global_det_idx = det_class_indices[det_local_idx]
            
            all_paired_gt_indices.add(global_gt_idx)
            all_paired_det_indices.add(global_det_idx)
            
            if global_gt_idx in all_unpaired_gt_indices:
                all_unpaired_gt_indices.remove(global_gt_idx)
            if global_det_idx in all_unpaired_det_indices:
                all_unpaired_det_indices.remove(global_det_idx)

    # Draw True Positives (Green)
    for det_idx in all_paired_det_indices:
        box = pred_boxes[det_idx]
        score = pred_scores[det_idx]
        label_text = f"TP: {label_map.get(pred_labels[det_idx], 'N/A')} ({score:.2f})"
        cv2.rectangle(vis_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(vis_image, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw False Negatives (Red Dashed)
    for gt_idx in all_unpaired_gt_indices:
        box = gt_boxes_eval[gt_idx]
        display_label_id = original_labels_for_eval_set[gt_idx] if show_original_labels else gt_labels_eval[gt_idx]
        if display_label_id not in label_map or label_map[display_label_id] == 'bg':
            continue # Don't draw FNs for 'bg' or excluded classes
        label_text = f"FN: {label_map.get(display_label_id, 'N/A')}"
        _draw_dashed_rectangle(vis_image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cv2.putText(vis_image, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Draw False Positives (Yellow)
    for det_idx in all_unpaired_det_indices:
        box = pred_boxes[det_idx]
        score = pred_scores[det_idx]
        if pred_labels[det_idx] not in label_map or label_map[pred_labels[det_idx]] == 'bg':
            continue # Don't draw FPs for 'bg'
        label_text = f"FP: {label_map.get(pred_labels[det_idx], 'N/A')} ({score:.2f})"
        cv2.rectangle(vis_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
        cv2.putText(vis_image, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return vis_image