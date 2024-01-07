from scipy.optimize import linear_sum_assignment  # for Hungarian algorithm
from cv_utils import iou_batch, iou_mask_pair
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Final, Union


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
    for (i, j) in list(zip(row_ind, col_ind)):
        if iou_matrix[i, j] > 0:
            # > 0 means at least the IoU is equal to min_iou threshold
            paired_idx.append((i, j))
        else:
            unpaired1.append(i)
            unpaired2.append(j)

    return paired_idx, unpaired1, unpaired2
    
    
def pair_gts_dets_mask(gt_boxes: np.ndarray, gt_masks: List[np.ndarray], det_boxes: np.ndarray, det_masks: List[np.ndarray], min_iou: float):
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
    
    for (i, j) in zip(overlapping_gt_idxs, overlapping_det_idxs):        
        iou_mask_matrix[i, j] = iou_mask_pair(gt_boxes[i], gt_masks[i], det_boxes[j], det_masks[j])
    
    # set values less than the threshold to zero
    iou_mask_matrix[iou_mask_matrix < min_iou] = 0
    
    # run Hungarian pairing algorithm
    row_ind, col_ind = linear_sum_assignment(-1 * iou_mask_matrix)
    # remove the assignments that are less than the passed min_iou
    unpaired1: List[int] = list(set([i for i in range(num_gts)]) - set(row_ind)) 
    unpaired2: List[int] = list(set([i for i in range(num_dets)]) - set(col_ind))  
    paired_idx: List[Tuple[int, int]] = []
    for (i, j) in list(zip(row_ind, col_ind)):
        if iou_mask_matrix[i, j] > 0:
            # > 0 means at least the IoU is equal to min_iou threshold
            paired_idx.append((i, j))
        else:
            unpaired1.append(i)
            unpaired2.append(j)

    return paired_idx, unpaired1, unpaired2
