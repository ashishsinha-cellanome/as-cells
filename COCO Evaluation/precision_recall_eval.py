from typing import Dict, List, Tuple, Optional, Final, Union
from pairing_utils import pair_gts_dets_bbox, pair_gts_dets_mask
from json_parser import enforce_one_to_one_mapping
import numpy as np
    
def get_bboxes_from_cellpose_mask(in_mask: Union[str, np.ndarray]) -> dict:
    """
    A helper function to take CellPose instance segmentation results and return bounding boxes 
    for each object instance in the mask. This code only supports masks from objects of one class. 
    Args:
        in_mask (a numpy array or a string): The numpy array of the mask returned by CellPose, or
        the path to the saved mask (should include the path and the filename). Note that the mask 
        should be the same size as the input image to CellPose. 
    
    Returns
        A (num_detections, 4) numpy array of object bounding boxes. 
    """
    
    if isinstance(in_mask, str):
        mask: np.ndarray = np.array(Image.open(in_mask))
    else:
        mask = in_mask.copy()
        
    # the assumption here is instances are encoded as different levels
    # with 0 being the background
    # each element in mask is an np.uint16 (unsigned 16 bits)
    
    # instances are encoded as different gray levels
    # the results are sorted
    obj_ids = np.unique(mask)
    # first id [0] is the background, so remove it
    obj_ids = obj_ids[1:]
        
    # get bounding box coordinates for each mask
    num_objs = len(obj_ids)
        
       
    # split the color-encoded mask into a set
    # of binary masks
    masks = mask == obj_ids[:, None, None]

    boxes = []
    for i in range(num_objs):
        pos = np.where(masks[i])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        if xmin < xmax and ymin < ymax:
            boxes.append([xmin, ymin, xmax, ymax])
        
    return np.array(boxes)    
    
def evaluate_cellpose_pr(cellpose_model, dataset, min_iou=0.5):
    """
    Args:
        cellpose_model: CellPose model to be called on test images; this should be the same code in 
            segmentation_cellpose.py from microscope-automation-gui/util accepting an image. 
        dataset: Custom dataset object. Make sure the dataset only includes 'Cell' annotations. 
        min_iou (float): Minimum IoU between the bounding box of a ground truth and that of a detection 
            to declare a detection correct. 
    Returns
        Precision
        Recall
    """
    
    num_true_positives: int = 0
    num_false_positives: int = 0
    num_false_negatives: int = 0
    
    for idx in range(len(dataset)):        
        annots = dataset[idx]
        
        # run CellPose
        mask = cellpose_model(annots['image'])
        
        gt_boxes = annots['annotations'][['xtl', 'ytl', 'xbr', 'ybr']].values.astype(float)
        # construct detection bounding boxes
        det_boxes = get_bboxes_from_cellpose_mask(mask).astype(float)
       
        paired_idx, unpaired_gts, unpaired_dets = pair_gts_dets_bbox(gt_boxes, det_boxes, min_iou)
        
        num_true_positives += len(paired_idx)
        num_false_positives += len(unpaired_dets)
        num_false_negatives += len(unpaired_gts)
        
        
        
        if (idx + 1) % 100 == 0:
            print(f"[INFO] Completed {idx+1} images out of {len(dataset)}")
        
    return num_true_positives / (num_true_positives + num_false_positives + 1e-30), num_true_positives / (num_true_positives + num_false_negatives + 1e-30)
    
    
def evaluate_yolo_pr(predictions, dataset, class_ids_of_interest=None, min_iou=0.5, child_parent_map=None):
    """
    Args:
        predictions (List of dictionaries): The i-th dictionary in the list is the detection results for
            the i-th image (dataset[i]) with keys as "boxes", "labels" and "scores" and values as
            (num_detections, 4) numpy array for bounding boxes, (num_detections, ) numpy array for 
            labels, and (num_detections, ) numpy array for scores (not used).
        dataset: Custom dataset object. 
        class_ids_of_interest (list or 1-D np.ndarray): A list of class IDs for labels to consider in 
            precision/recall evaluation. 
        min_iou (float): Minimum IoU between the bounding box of a ground truth and that of a detection 
            to declare a detection correct. 
        child_parent_map (dictionary): A dictionary with integer keys and values specifying the one-to-one 
            relationships between child class IDs and parent class IDs (to be enforced). 
    Returns
    	Precision
    	Recall
    """
    
    
    num_true_positives: int = 0
    num_false_positives: int = 0
    num_false_negatives: int = 0
    
    for idx in range(len(dataset)):        
        annots = dataset[idx]
        if child_parent_map is not None:
            annots = enforce_one_to_one_mapping(data_sample=annots, child_parent_map=child_parent_map)
        
        boxes = predictions[idx]['boxes']
        labels = predictions[idx]['labels']
        
        if class_ids_of_interest is None:
            # use the union of all the class IDs from the detections and the annotations
            class_ids_to_filter = list(annots['annotations']['label'].unique())
            class_ids_to_filter += list(np.unique(labels))
            # remove duplicates
            class_ids_to_filter = list(set(class_ids_to_filter))
        else:
            class_ids_to_filter = class_ids_of_interest
            
        for class_id in class_ids_to_filter:
            # filter the detections and ground truths for the given label
            gt_boxes = annots['annotations'].loc[annots['annotations']['label'] == class_id, ['xtl', 'ytl', 'xbr', 'ybr']].values.astype(int)
            det_boxes = boxes[labels == class_id, :]
            # pair
            paired_idx, unpaired_gts, unpaired_dets = pair_gts_dets_bbox(gt_boxes, det_boxes, min_iou)
        
            num_true_positives += len(paired_idx)
            num_false_positives += len(unpaired_dets)
            num_false_negatives += len(unpaired_gts)
        
        if (idx + 1) % 100 == 0:
            print(f"[INFO] Completed {idx+1} images out of {len(dataset)}")
        
    return num_true_positives / (num_true_positives + num_false_positives + 1e-30), num_true_positives / (num_true_positives + num_false_negatives + 1e-30)
    
    
def evaluate_mask_rcnn_pr(predictions, dataset, class_ids_of_interest=None, min_iou=0.5, child_parent_map=None):
    """
    Args:
        predictions (List of dictionaries): The i-th dictionary in the list is the detection results for
            the i-th image (dataset[i]) with keys as "boxes", "labels", "scores" and "masks" and values as
            (num_detections, 4) numpy array for bounding boxes, (num_detections, ) numpy array for 
            labels, (num_detections, ) numpy array for scores (not used) and a list of num_detections numpy
            arrays for each mask. Each mask should be defined within the passed bounding boxes for the 
            detected object, hence the masks are not of the same size. 
        dataset: Custom dataset object. 
        class_ids_of_interest (list or 1-D np.ndarray): A list of class IDs for labels to consider in 
            precision/recall evaluation. 
        min_iou (float): Minimum IoU between the mask of a ground truth and that of a detection 
            to declare a detection correct. 
        child_parent_map (dictionary): A dictionary with integer keys and values specifying the one-to-one 
            relationships between child class IDs and parent class IDs (to be enforced). 
    Returns
        Precision
        Recall
    """
    
    
    num_true_positives: int = 0
    num_false_positives: int = 0
    num_false_negatives: int = 0
    
    for idx in range(len(dataset)): 
        annots = dataset[idx]
        if child_parent_map is not None:
            annots = enforce_one_to_one_mapping(data_sample=annots, child_parent_map=child_parent_map)
        
        boxes = predictions[idx]['boxes']
        labels = predictions[idx]['labels']
        masks = predictions[idx]['masks']
        
        if class_ids_of_interest is None:
            # use the union of all the class IDs from the detections and the annotations
            class_ids_to_filter = list(annots['annotations']['label'].unique())
            class_ids_to_filter += list(np.unique(labels))
            # remove duplicates
            class_ids_to_filter = list(set(class_ids_to_filter))
        else:
            class_ids_to_filter = class_ids_of_interest
            
        for class_id in class_ids_to_filter:
            # filter the detections and ground truths for the given label
            idxs = annots['annotations'][annots['annotations']['label'] == class_id].index.values
            gt_boxes = annots['annotations'].loc[idxs, ['xtl', 'ytl', 'xbr', 'ybr']].values.astype(int)
            gt_masks = [annots['masks'][ind] for ind in idxs]
            
            det_boxes = boxes[labels == class_id, :]
            det_masks = [mask for i, mask in enumerate(masks) if labels[i] == class_id]
            # pair
            paired_idx, unpaired_gts, unpaired_dets = pair_gts_dets_mask(gt_boxes, gt_masks, det_boxes, det_masks, min_iou)
        
            num_true_positives += len(paired_idx)
            num_false_positives += len(unpaired_dets)
            num_false_negatives += len(unpaired_gts)
        
        if (idx + 1) % 100 == 0:
            print(f"[INFO] Completed {idx+1} images out of {len(dataset)}")
        
    return num_true_positives / (num_true_positives + num_false_positives + 1e-30), num_true_positives / (num_true_positives + num_false_negatives + 1e-30)    

def calculate_confusion_matrix(predictions, dataset, use_mask=False, min_iou=0.5, child_parent_map=None):
    """
    Args:
        predictions (List of dictionaries): The i-th dictionary in the list is the detection results for
            the i-th image (dataset[i]) with keys as "boxes", "labels", "scores" and "masks" and values as
            (num_detections, 4) numpy array for bounding boxes, (num_detections, ) numpy array for 
            labels, (num_detections, ) numpy array for scores (not used) and a list of num_detections numpy
            arrays for each mask. Each mask should be defined within the passed bounding boxes for the 
            detected object, hence the masks are not of the same size. 
        dataset: Custom dataset object. 
        use_mask (bool): Use IoU between masks if set to True. In this case, predictions[idx] and dataset dictionaries
            both should include a string "mask" key to report the mask. 
        min_iou (float): Minimum IoU between the mask of a ground truth and that of a detection 
            to declare a detection correct. 
        child_parent_map (dictionary): A dictionary with integer keys and values specifying the one-to-one 
            relationships between child class IDs and parent class IDs (to be enforced). 
    Returns
        confusion_matrix: A (num_classes + 1) x (num_classes + 1) matrix with (i, j)th element being the number
            of objects from the jth class that are being detected by the model as being from the ith class. The
            rightmost column is incorrect detections (FPs) by the model for the ith detection class, and the 
            last row is the miss detections (FNs) by  the model with the jth object class. 
    """

    gt_class_ids: List[int] = list(set(dataset.class_names_to_ids_map.values()))
    
    confusion_matrix: np.ndarray = np.zeros((len(gt_class_ids) + 1, len(gt_class_ids) + 1), dtype=int)
    matrix_idx_to_class_id_map: Dict[int, int] = {i: gt_class_ids[i] for i in range(len(gt_class_ids))}
    class_id_to_matrix_idx_map: Dict[int, int] = {gt_class_ids[i]: i for i in range(len(gt_class_ids))}
    
    for idx in range(len(dataset)): 
        annots = dataset[idx]
        if child_parent_map is not None:
            annots = enforce_one_to_one_mapping(data_sample=annots, child_parent_map=child_parent_map)
        
        det_boxes = predictions[idx]['boxes']
        det_labels = predictions[idx]['labels']
       
        gt_boxes = annots['annotations'][['xtl', 'ytl', 'xbr', 'ybr']].values.astype(int)
        gt_labels = annots['annotations']['label'].values.astype(int)
            
        if use_mask:

            gt_masks = annots['masks']
            det_masks = predictions[idx]['masks']
            paired_idx, unpaired_dets, unpaired_gts = pair_gts_dets_mask(det_boxes, det_masks, gt_boxes, gt_masks, min_iou)
        else:
            paired_idx, unpaired_dets, unpaired_gts = pair_gts_dets_bbox(det_boxes, gt_boxes, min_iou)

        for i in unpaired_gts:
            confusion_matrix[len(gt_class_ids), class_id_to_matrix_idx_map[gt_labels[i]]] += 1

        for i in unpaired_dets:
            confusion_matrix[class_id_to_matrix_idx_map[det_labels[i]], len(gt_class_ids)] += 1
            
        for (i, j) in paired_idx:
            confusion_matrix[class_id_to_matrix_idx_map[det_labels[i]], class_id_to_matrix_idx_map[gt_labels[j]]] += 1
            
        
        if (idx + 1) % 100 == 0:
            print(f"[INFO] Completed {idx+1} images out of {len(dataset)}")
        
    return confusion_matrix




