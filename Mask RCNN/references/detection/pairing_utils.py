from scipy.optimize import linear_sum_assignment  # for Hungarian algorithm

import numpy as np
import cv2
from typing import List, Tuple

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


def pair_gts_dets_bbox(gt_boxes: np.ndarray, det_boxes: np.ndarray, min_iou: float):
    """
    A function to pair ground truth and detection bounding boxes. The code uses Hungarian algorithm
    to match the ground truth and detection bounding boxes. The boxes from the sets are paired in a way
    to maximize the sum of IoUs for the pairs. Pairing is only allowed if IoU is greater than a passed
    minimum.

    Args
        gt_boxes (np.ndarray): A (num_ground_truths, 4) numpy array of ground truth bounding boxes in
            (xtl, ytl, xbr, ybr) format.
        det_boxes (np.ndarray): A (num_detections, 4) numpy array of detection bounding boxes in
            (xtl, ytl, xbr, ybr) format.
        min_iou (float): The minimum IoU to pair a ground truth box with a detection box.

    Returns
        A list of 2-tuples (i1, i2) for the paired boxes; i1 is row index of the ground truth box from
            gt_boxes and i2 is the the row index of the detection from det_boxes.
        A list of row indexes i1 from gt_boxes that were not matched to any detections.
        A list of row indexes i2 from det_boxes that were not matched to any ground truth.


    """

    # calculate the IoU between all possible combination of pairs from each set
    num_gts: int = len(gt_boxes)
    num_dets: int = len(det_boxes)
    iou_matrix: np.ndarray = iou_batch(gt_boxes, det_boxes)

    # set values less than the threshold to zero
    iou_matrix[iou_matrix < min_iou] = 0

    # run Hungarian pairing algorithm
    row_ind, col_ind = linear_sum_assignment(-1 * iou_matrix)
    # remove the assignments that are less than the passed min_iou
    unpaired1: List[int] = list(set([i for i in range(num_gts)]) - set(row_ind))
    unpaired2: List[int] = list(set([i for i in range(num_dets)]) - set(col_ind))
    paired_idx: List[Tuple[int, int]] = []
    for i, j in list(zip(row_ind, col_ind)):
        if iou_matrix[i, j] > 0:
            # > 0 means at least the IoU is equal to min_iou threshold
            paired_idx.append((i, j))
        else:
            unpaired1.append(i)
            unpaired2.append(j)

    return paired_idx, unpaired1, unpaired2


def pair_gts_dets_mask(
    gt_boxes: np.ndarray,
    gt_masks: List[np.ndarray],
    det_boxes: np.ndarray,
    det_masks: List[np.ndarray],
    min_iou: float,
):
    """
    A function to pair ground truth and detection masks. The code uses Hungarian algorithm
    to match the ground truth and detection masks based on their IoU value. The masks from the sets are
    paired in a way to maximize the sum of IoUs for the pairs. Pairing is only allowed if IoU is greater than
    a passed minimum.

    Args
        gt_boxes (np.ndarray): A (num_ground_truths, 4) numpy array of ground truth bounding boxes in
            (xtl, ytl, xbr, ybr) format.
        gt_masks (list of numpy arrays): A num_ground_truths list of numpy arrays. The i-th element is
            the mask for the i-th ground truch object defined within the passed bounding box for the
            object (xtl, ytl, xbr, ybr) and it should be a np.uint8 (ybr - ytl, xbr - xtl) numpy array
            with mask values set to 1 for the ground truth object.
        det_boxes (np.ndarray): A (num_detections, 4) numpy array of detection bounding boxes in
            (xtl, ytl, xbr, ybr) format.
        det_masks (list of numpy arrays): A num_detections list of numpy arrays. The i-th element is
            the mask for the i-th detected object defined within the passed bounding box for the
            object (xtl, ytl, xbr, ybr) and it should be a np.uint8 (ybr - ytl, xbr - xtl) numpy array
            with mask values set to 1 for the ground truth object.
        min_iou (float): The minimum IoU to pair a ground truth box with a detection box.

    Returns
        A list of 2-tuples (i1, i2) for the paired boxes; i1 is row index of the ground truth box from
            gt_boxes and i2 is the the row index of the detection from det_boxes.
        A list of row indexes i1 from gt_boxes that were not matched to any detections.
        A list of row indexes i2 from det_boxes that were not matched to any ground truth.


    """

    # calculate the IoU between all possible combination of pairs from each set
    num_gts: int = len(gt_boxes)
    num_dets: int = len(det_boxes)
    # to enable the code to run faster, we first compute the box IoUs, find the potentially
    # overlapping ones and then calculate the IoU between the masks
    iou_box_matrix: np.ndarray = iou_batch(gt_boxes, det_boxes)
    overlapping_gt_idxs, overlapping_det_idxs = np.where(iou_box_matrix > 0)

    iou_mask_matrix: np.ndarray = np.zeros((num_gts, num_dets))

    for i, j in zip(overlapping_gt_idxs, overlapping_det_idxs):
        iou_mask_matrix[i, j] = iou_mask_pair(
            gt_boxes[i], gt_masks[i], det_boxes[j], det_masks[j]
        )

    # set values less than the threshold to zero
    iou_mask_matrix[iou_mask_matrix < min_iou] = 0

    # run Hungarian pairing algorithm
    row_ind, col_ind = linear_sum_assignment(-1 * iou_mask_matrix)
    # remove the assignments that are less than the passed min_iou
    unpaired1: List[int] = list(set([i for i in range(num_gts)]) - set(row_ind))
    unpaired2: List[int] = list(set([i for i in range(num_dets)]) - set(col_ind))
    paired_idx: List[Tuple[int, int]] = []
    for i, j in list(zip(row_ind, col_ind)):
        if iou_mask_matrix[i, j] > 0:
            # > 0 means at least the IoU is equal to min_iou threshold
            paired_idx.append((i, j))
        else:
            unpaired1.append(i)
            unpaired2.append(j)

    return paired_idx, unpaired1, unpaired2
