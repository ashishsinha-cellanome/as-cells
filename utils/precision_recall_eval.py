from .pairing_utils import pair_gts_dets_bbox, pair_gts_dets_mask
from .json_parser import CellMaskDataset

import numpy as np
import pandas as pd

from typing import Dict, List, Tuple, Optional, Final, Union
from tqdm import tqdm

class AnnotationFilter:
    def __init__(
        self, 
        classnames_mapping_dict: Dict[str, str], 
        class_ids_to_class_names_map: Dict[int, str]
    ):
        """
        A class to modify the annotations of samples of our custom dataset class for original annotations (e.g, CellMaskDataset, 
        TestDataset) to map some object classes to another (for example, map cell-adhered class object annotations 
        to cell class annotations), or completely exclude some. 
        Args:
            - classnames_mapping_dict: A dictionary with keys as the class name to be modified and values and the 
              target class name to which the key class should be mapped. A special target class of 'bg' means the 
              key should be ignored.
            - class_ids_to_class_names_map: A dictionary providing the mapping between class IDs and class names. It is needed
              because our dataset classes include class IDs in the annotations. 
        """
        self.classnames_mapping_dict = classnames_mapping_dict
        self.class_ids_to_class_names_map = class_ids_to_class_names_map
        reverse_label_map = {v:k for k, v in self.class_ids_to_class_names_map.items()}
        classnames_to_exclude: List[str] = [name for name, mapped_name in self.classnames_mapping_dict.items() if mapped_name == 'bg']
        self.class_ids_to_exclude: List [int] = [reverse_label_map[name] for name in classnames_to_exclude] 
        self.class_ids_mapping_dict = {reverse_label_map[name]: reverse_label_map[mapped_name] 
                                       for name, mapped_name in classnames_mapping_dict.items() if mapped_name != 'bg'}
    
    def apply(self, annots: dict) -> dict:
        class_ids_mapping_dict = {k: v for k, v in  self.class_ids_mapping_dict.items()}
        all_class_ids: List[int] = annots['annotations']['label'].unique()
        for class_id in all_class_ids:
            if class_id in class_ids_mapping_dict: 
                continue
            # a class that is mapped to itself, add it to the mapping
            class_ids_mapping_dict[class_id] = class_id
        annotations_df = annots['annotations'].copy()
        annotations_df['label'] = annots['annotations']['label'].apply(lambda x: class_ids_mapping_dict[x])
        annotations_df = annotations_df[annotations_df['label'].apply(lambda x: False if x in self.class_ids_to_exclude else True)]
        idxs_to_keep = annotations_df.index.values
        annotations_df = annotations_df.reset_index(drop=True)
        if len(annotations_df) == 0:
            annotations_df = pd.DataFrame(columns = ['xtl', 'ytl', 'xbr', 'ybr', 'label'])
        masks: List[np.ndarray] = [mask.copy() for idx, mask in enumerate(annots['masks']) if idx in idxs_to_keep]
        return {'name': annots['name'], 'image': annots['image'].copy(), 'annotations': annotations_df, 'masks': masks}

def evaluate_pr_per_image(
    predictions: List[Dict],
    dataset: CellMaskDataset,
    class_ids_of_interest: List[int],
    min_iou: float = 0.5,
    use_mask: bool = False,
    annotation_filter: Optional[AnnotationFilter] = None,
) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict, float, float, float, float, float, float, List[Dict]]:
    """
    Evaluates precision, recall, and F1 score, returning both aggregate
    (per-class and micro) metrics and per-image metrics.
    
    The return signature is modeled after the original `evaluate_pr` function,
    returning a tuple of metrics.

    Args:
        predictions: List of detection dictionaries for each image.
        dataset: The dataset object to get ground truth annotations.
        class_ids_of_interest: A list of class IDs to evaluate.
        min_iou: The IoU threshold for a detection to be a True Positive.
        use_mask: Whether to use mask IoU (True) or bounding box IoU (False).
        annotation_filter: An optional filter to apply to ground truth annotations.

    Returns:
        A tuple containing:
        - (dict) num_true_positives_per_class: Aggregate TPs, keyed by class_id.
        - (dict) num_false_positives_per_class: Aggregate FPs, keyed by class_id.
        - (dict) num_false_negatives_per_class: Aggregate FNs, keyed by class_id.
        - (dict) precision_per_class: Aggregate Precision, keyed by class_id.
        - (dict) recall_per_class: Aggregate Recall, keyed by class_id.
        - (dict) f1_per_class: Aggregate F1-score, keyed by class_id.
        - (float) total_true_positives: Total TPs across all classes.
        - (float) total_false_positives: Total FPs across all classes.
        - (float) total_false_negatives: Total FNs across all classes.
        - (float) micro_precision: Micro-average precision.
        - (float) micro_recall: Micro-average recall.
        - (float) micro_f1: Micro-average F1-score.
        - (list) per_image_metrics: A list of dictionaries, one per image, 
                                   detailing TP, FP, FN, P, R, and F1 for each class.
    """
    total_true_positives = {class_id: 0 for class_id in class_ids_of_interest}
    total_false_positives = {class_id: 0 for class_id in class_ids_of_interest}
    total_false_negatives = {class_id: 0 for class_id in class_ids_of_interest}
    
    per_image_metrics = []

    print("Running Precision/Recall Evaluation...")
    for i in tqdm(range(len(predictions))):
        preds = predictions[i]
        annots = dataset[i]
        
        if annotation_filter:
            annots = annotation_filter.apply(annots)
            
        boxes = np.array(preds['boxes']) if 'boxes' in preds and len(preds['boxes']) > 0 else np.zeros((0, 4))
        labels = np.array(preds['labels']) if 'labels' in preds and len(preds['labels']) > 0 else np.zeros((0,))
        
        if boxes.ndim == 1 and boxes.shape[0] > 0:
            boxes = np.expand_dims(boxes, 0)
        elif boxes.shape[0] == 0:
            boxes = np.zeros((0, 4))
            
        masks = preds.get('masks')
        gt_annotations = annots["annotations"]
        gt_masks = annots.get("masks")
        
        image_metrics = {
            'name': annots.get('name', f'image_{i}'),
            'tp': {class_id: 0 for class_id in class_ids_of_interest},
            'fp': {class_id: 0 for class_id in class_ids_of_interest},
            'fn': {class_id: 0 for class_id in class_ids_of_interest},
            'precision': {class_id: 0.0 for class_id in class_ids_of_interest},
            'recall': {class_id: 0.0 for class_id in class_ids_of_interest},
            'f1': {class_id: 0.0 for class_id in class_ids_of_interest},
        }

        for class_id in class_ids_of_interest:
            gt_idxs = gt_annotations[gt_annotations['label'] == class_id].index.values
            gt_boxes = gt_annotations.loc[gt_idxs, ['xtl', 'ytl', 'xbr', 'ybr']].values.astype(int)
            
            det_idxs = np.where(labels == class_id)[0]
            det_boxes = boxes[det_idxs, :]
            
            num_tp_class, num_fp_class, num_fn_class = 0, 0, 0

            if use_mask and gt_masks and masks:
                gt_masks_class = [gt_masks[idx] for idx in gt_idxs]
                det_masks_class = [masks[idx] for idx in det_idxs]
                paired, unpaired_gts, unpaired_dets = pair_gts_dets_mask(
                    gt_boxes, gt_masks_class, det_boxes, det_masks_class, min_iou
                )
            else:
                paired, unpaired_gts, unpaired_dets = pair_gts_dets_bbox(
                    gt_boxes, det_boxes, min_iou
                )
            
            num_tp_class = len(paired)
            num_fp_class = len(unpaired_dets)
            num_fn_class = len(unpaired_gts)

            total_true_positives[class_id] += num_tp_class
            total_false_positives[class_id] += num_fp_class
            total_false_negatives[class_id] += num_fn_class
            
            image_metrics['tp'][class_id] = num_tp_class
            image_metrics['fp'][class_id] = num_fp_class
            image_metrics['fn'][class_id] = num_fn_class

        # Calculate per-image metrics (P, R, F1)
        for cid in class_ids_of_interest:
            tp, fp, fn = image_metrics['tp'][cid], image_metrics['fp'][cid], image_metrics['fn'][cid]
            precision = tp / (tp + fp + 1e-30)
            recall = tp / (tp + fn + 1e-30)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-30)
            image_metrics['precision'][cid] = precision
            image_metrics['recall'][cid] = recall
            image_metrics['f1'][cid] = f1
            
        per_image_metrics.append(image_metrics)

    # Calculate aggregate metrics (per-class and micro)
    agg_precision, agg_recall, agg_f1 = {}, {}, {}
    total_tp_all_classes = 0.0
    total_fp_all_classes = 0.0
    total_fn_all_classes = 0.0

    for class_id in class_ids_of_interest:
        tp = total_true_positives[class_id]
        fp = total_false_positives[class_id]
        fn = total_false_negatives[class_id]

        # Add to micro totals
        total_tp_all_classes += tp
        total_fp_all_classes += fp
        total_fn_all_classes += fn
        
        # Per-class (macro-style) metrics
        agg_precision[class_id] = tp / (tp + fp + 1e-30)
        agg_recall[class_id] = tp / (tp + fn + 1e-30)
        agg_f1[class_id] = 2 * (agg_precision[class_id] * agg_recall[class_id]) / (agg_precision[class_id] + agg_recall[class_id] + 1e-30)
                           
    # Calculate micro-average metrics (across all classes)
    micro_precision = total_tp_all_classes / (total_tp_all_classes + total_fp_all_classes + 1e-30)
    micro_recall = total_tp_all_classes / (total_tp_all_classes + total_fn_all_classes + 1e-30)
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-30)

    return (
        total_true_positives,   # num_true_positives_per_class
        total_false_positives,  # num_false_positives_per_class
        total_false_negatives,  # num_false_negatives_per_class
        agg_precision,          # precision_per_class
        agg_recall,             # recall_per_class
        agg_f1,                 # f1_per_class (new)
        total_tp_all_classes,   # num_true_positives
        total_fp_all_classes,   # num_false_positives
        total_fn_all_classes,   # num_false_negatives
        micro_precision,        # micro precision
        micro_recall,           # micro recall
        micro_f1,               # micro f1 (new)
        per_image_metrics,      # per_image_metrics (new)
    )

def evaluate_pr_updated(
    predictions: List[Dict[str, list]], 
    dataset: CellMaskDataset, 
    class_ids_of_interest: List[int],
    min_iou: float = 0.5, 
    use_mask: bool = False, 
    annotation_filter: AnnotationFilter = None
):
    """
    The function calculates macro and micro precision and recall for a given list of class IDs. 
    ...
    """
    
    
    num_true_positives_per_class: Dict[int, int] = {c_id: 0 for c_id in class_ids_of_interest}
    num_false_positives_per_class: Dict[int, int] = {c_id: 0 for c_id in class_ids_of_interest}
    num_false_negatives_per_class: Dict[int, int] = {c_id: 0 for c_id in class_ids_of_interest}
    precision_per_class: Dict[int, float] = {}
    recall_per_class: Dict[int, float] = {}
    
    for idx in range(len(dataset)): 
        annots = dataset[idx]
        # Get the original predictions
        preds = predictions[idx].copy() # Use .copy() to avoid modifying the original list
            
        if annotation_filter is not None:
            # 1. Apply filter to Ground Truth (as you did before)
            annots = annotation_filter.apply(annots)
            
            # 2. Apply the same remapping to Predictions 
            remap_dict = annotation_filter.class_ids_mapping_dict
            if remap_dict and 'labels' in preds and len(preds['labels']) > 0:
                original_labels = np.array(preds['labels'])
                # Create new labels array by remapping
                new_labels = np.array([remap_dict.get(label, label) for label in original_labels])
                preds['labels'] = new_labels

        # Now 'preds' and 'annots' labels are on the same remapped basis
        boxes = preds['boxes']
        labels = preds['labels']
        if use_mask:
            masks = preds['masks']
            
        for class_id in class_ids_of_interest:
            # filter the detections and ground truths for the given label
            idxs = annots['annotations'][annots['annotations']['label'] == class_id].index.values
            
            gt_boxes = annots['annotations'].loc[idxs, ['xtl', 'ytl', 'xbr', 'ybr']].values.astype(int)
            
            # Ensure det_boxes is 2D, even if empty
            det_boxes_all = np.array(boxes)
            if det_boxes_all.ndim == 1 and det_boxes_all.shape[0] > 0:
                det_boxes_all = np.expand_dims(det_boxes_all, 0)
            elif det_boxes_all.shape[0] == 0:
                det_boxes_all = np.zeros((0, 4))
                
            det_boxes = det_boxes_all[labels == class_id, :]
            
            if use_mask:
                gt_masks = [annots['masks'][ind] for ind in idxs]
                det_masks = [mask for i, mask in enumerate(masks) if labels[i] == class_id]
                paired_idx, unpaired_gts, unpaired_dets = pair_gts_dets_mask(gt_boxes, gt_masks, det_boxes, det_masks, min_iou)
            else:
                paired_idx, unpaired_gts, unpaired_dets = pair_gts_dets_bbox(gt_boxes, det_boxes, min_iou)
                
            num_true_positives_per_class[class_id] += len(paired_idx)
            num_false_positives_per_class[class_id] += len(unpaired_dets)
            num_false_negatives_per_class[class_id] += len(unpaired_gts)
            
        if (idx + 1) % 100 == 0:
            print(f"[INFO] Completed {idx+1} images out of {len(dataset)}")

    # ... (rest of your function is correct) ...
    
    num_true_positives: float = 0
    num_false_positives: float = 0
    num_false_negatives: float = 0
    
    for class_id in class_ids_of_interest:
        # total counts for micro precision and recal
        num_true_positives += num_true_positives_per_class[class_id]
        num_false_positives += num_false_positives_per_class[class_id]
        num_false_negatives += num_false_negatives_per_class[class_id]
        
        precision_per_class[class_id] = (
            num_true_positives_per_class[class_id] / (num_true_positives_per_class[class_id] + num_false_positives_per_class[class_id] + 1e-30)
        ) # precision
        recall_per_class[class_id] = (
            num_true_positives_per_class[class_id] / (num_true_positives_per_class[class_id] + num_false_negatives_per_class[class_id] + 1e-30)
        ) # recall
       
    return (
        num_true_positives_per_class, 
        num_false_positives_per_class, 
        num_false_negatives_per_class,
        precision_per_class,
        recall_per_class,
        num_true_positives,
        num_false_positives,
        num_false_negatives,
        num_true_positives / (num_true_positives + num_false_positives + 1e-30), # micro precision
        num_true_positives / (num_true_positives + num_false_negatives + 1e-30), # micro recall
    )

def evaluate_pr(
    predictions: List[Dict[str, list]], 
    dataset: CellMaskDataset, 
    class_ids_of_interest: List[int],
    min_iou: float = 0.5, 
    use_mask: bool = False, 
    annotation_filter: AnnotationFilter = None
):
    """
    The function calculates macro and micro precision and recall for a given list of class IDs. 
    Args:
        predictions (List of dictionaries): The i-th dictionary in the list is the detection results for
            the i-th image (dataset[i]) with keys as "boxes", "labels", "scores" and optionally "masks" and values as
            (num_detections, 4) numpy array for bounding boxes, (num_detections, ) numpy array for 
            labels, (num_detections, ) numpy array for scores (not used) and optionally a list of num_detections numpy
            arrays for each mask. Each mask should be defined within the passed bounding boxes for the 
            detected object, hence the masks are not of the same size. 
        dataset: Custom dataset class for original annotations (e.g, CellMaskDataset, TestDataset).
        class_ids_of_interest (list or 1-D np.ndarray): A list of class IDs to consider in precision/recall evaluation. 
        min_iou (float): Minimum IoU between the mask of a ground truth and that of a detection 
            to declare a detection correct. 
        annotation_filter (AnnotationFilter class): A class to post process the sample from the dataset and exclude or merge some classes.
    Returns
        - 3 dictionaries with keys as the class IDs of interest and values as the number of TP, FP, FN detections per class, 
        - 2 dictionaries of precision and recall per calls
        - 3 counts of total number of TP, FP, FN detections over all class IDs of interest
        - micro precision and recall
    """
    num_true_positives_per_class: Dict[int, int] = {c_id: 0 for c_id in class_ids_of_interest}
    num_false_positives_per_class: Dict[int, int] = {c_id: 0 for c_id in class_ids_of_interest}
    num_false_negatives_per_class: Dict[int, int] = {c_id: 0 for c_id in class_ids_of_interest}
    precision_per_class: Dict[int, float] = {}
    recall_per_class: Dict[int, float] = {}
    
    for idx in range(len(dataset)): 
        annots = dataset[idx]

        # remaps the GT labels based on the mapping defined in annotation filter
        if annotation_filter is not None:
            annots = annotation_filter.apply(annots)
            # remap the preds
            # 3 -> 0, 4 -> 0
            # detr [0,...4] _>3,4
            # detr -> soma (pred), GT (soma -> cell, )
        
            # TODO: Apply the same remapping to Predictions 
            # remap_dict = annotation_filter.class_ids_mapping_dict
            # if remap_dict and 'labels' in preds and len(preds['labels']) > 0:
            #     original_labels = np.array(preds['labels'])
            #     # Create new labels array by remapping
            #     new_labels = np.array([remap_dict.get(label, label) for label in original_labels])
            #     preds['labels'] = new_labels
        
        
        boxes = predictions[idx]['boxes']
        labels = predictions[idx]['labels']
        if use_mask:
            masks = predictions[idx]['masks']
            
        for class_id in class_ids_of_interest:
            # filter the detections and ground truths for the given label
            idxs = annots['annotations'][annots['annotations']['label'] == class_id].index.values
            
            gt_boxes = annots['annotations'].loc[idxs, ['xtl', 'ytl', 'xbr', 'ybr']].values.astype(int)
            det_boxes = boxes[labels == class_id, :]
            
            if use_mask:
                gt_masks = [annots['masks'][ind] for ind in idxs]
                det_masks = [mask for i, mask in enumerate(masks) if labels[i] == class_id]
                paired_idx, unpaired_gts, unpaired_dets = pair_gts_dets_mask(gt_boxes, gt_masks, det_boxes, det_masks, min_iou)
            else:
                paired_idx, unpaired_gts, unpaired_dets = pair_gts_dets_bbox(gt_boxes, det_boxes, min_iou)
                
            num_true_positives_per_class[class_id] += len(paired_idx)
            num_false_positives_per_class[class_id] += len(unpaired_dets)
            num_false_negatives_per_class[class_id] += len(unpaired_gts)
        if (idx + 1) % 100 == 0:
            print(f"[INFO] Completed {idx+1} images out of {len(dataset)}")

    num_true_positives: float = 0
    num_false_positives: float = 0
    num_false_negatives: float = 0
    
    for class_id in class_ids_of_interest:
        # total counts for micro precision and recal
        num_true_positives += num_true_positives_per_class[class_id]
        num_false_positives += num_false_positives_per_class[class_id]
        num_false_negatives += num_false_negatives_per_class[class_id]
        
        precision_per_class[class_id] = (
            num_true_positives_per_class[class_id] / (num_true_positives_per_class[class_id] + num_false_positives_per_class[class_id] + 1e-30)
        ) # precision
        recall_per_class[class_id] = (
            num_true_positives_per_class[class_id] / (num_true_positives_per_class[class_id] + num_false_negatives_per_class[class_id] + 1e-30)
        ) # recall
       
    return (
        num_true_positives_per_class, 
        num_false_positives_per_class, 
        num_false_negatives_per_class,
        precision_per_class,
        recall_per_class,
        num_true_positives,
        num_false_positives,
        num_false_negatives,
        num_true_positives / (num_true_positives + num_false_positives + 1e-30), # micro precision
        num_true_positives / (num_true_positives + num_false_negatives + 1e-30), # micro recall
    )

def calculate_confusion_matrix(
    predictions: List[Dict[str, list]], 
    dataset: CellMaskDataset, 
    use_mask: bool = False, 
    min_iou: float = 0.5
):
    """
    Args:
        predictions (List of dictionaries): The i-th dictionary in the list is the detection results for
            the i-th image (dataset[i]) with keys as "boxes", "labels", "scores" and "masks" and values as
            (num_detections, 4) numpy array for bounding boxes, (num_detections, ) numpy array for 
            labels, (num_detections, ) numpy array for scores (not used) and a list of num_detections numpy
            arrays for each mask. Each mask should be defined within the passed bounding boxes for the 
            detected object, hence the masks are not of the same size. 
        dataset: Custom dataset class for original annotations (e.g, CellMaskDataset, TestDataset).
        use_mask (bool): Use IoU between masks if set to True. In this case, predictions[idx] and dataset dictionaries
            both should include a string "mask" key to report the mask. 
        min_iou (float): Minimum IoU between the mask of a ground truth and that of a detection 
            to declare a detection correct. 
    Returns
        confusion_matrix: A (num_classes + 1) x (num_classes + 1) matrix with (i, j)th element being the number
            of objects from the jth class that are being detected by the model as being from the ith class. The
            rightmost column is incorrect detections (FPs) by the model for the ith detection class, and the 
            last row is the miss detections (FNs) by the model with the jth object class. 
    """

    gt_class_ids: List[int] = list(set(dataset.class_names_to_ids_map.values()))
    
    confusion_matrix: np.ndarray = np.zeros((len(gt_class_ids) + 1, len(gt_class_ids) + 1), dtype=int)
    matrix_idx_to_class_id_map: Dict[int, int] = {i: gt_class_ids[i] for i in range(len(gt_class_ids))}
    class_id_to_matrix_idx_map: Dict[int, int] = {gt_class_ids[i]: i for i in range(len(gt_class_ids))}
    
    for idx in range(len(dataset)): 
        annots = dataset[idx]        
        confusion_matrix += calculate_sample_confusion_matrix(predictions[idx], annots, gt_class_ids, use_mask, min_iou)
            
        if (idx + 1) % 100 == 0:
            print(f"[INFO] Completed {idx+1} images out of {len(dataset)}")
        
    return confusion_matrix
    
    
def calculate_sample_confusion_matrix(
    detections: Dict[str, list], 
    data_sample: dict, 
    gt_class_ids: List[int], 
    use_mask: bool = False, 
    min_iou: float = 0.5
):
 
    confusion_matrix: np.ndarray = np.zeros((len(gt_class_ids) + 1, len(gt_class_ids) + 1), dtype=int)
    matrix_idx_to_class_id_map: Dict[int, int] = {i: gt_class_ids[i] for i in range(len(gt_class_ids))}
    class_id_to_matrix_idx_map: Dict[int, int] = {gt_class_ids[i]: i for i in range(len(gt_class_ids))}
    
        
    det_boxes = detections['boxes']
    det_labels = detections['labels']
       
    gt_boxes = data_sample['annotations'][['xtl', 'ytl', 'xbr', 'ybr']].values.astype(int)
    gt_labels = data_sample['annotations']['label'].values.astype(int)
            
    if use_mask:
        gt_masks = data_sample['masks']
        det_masks = detections['masks']
        paired_idx, unpaired_dets, unpaired_gts = pair_gts_dets_mask(det_boxes, det_masks, gt_boxes, gt_masks, min_iou)
    else:
        paired_idx, unpaired_dets, unpaired_gts = pair_gts_dets_bbox(det_boxes, gt_boxes, min_iou)

    for i in unpaired_gts:
        confusion_matrix[len(gt_class_ids), class_id_to_matrix_idx_map[gt_labels[i]]] += 1

    for i in unpaired_dets:
        confusion_matrix[class_id_to_matrix_idx_map[det_labels[i]], len(gt_class_ids)] += 1
            
    for (i, j) in paired_idx:
        confusion_matrix[class_id_to_matrix_idx_map[det_labels[i]], class_id_to_matrix_idx_map[gt_labels[j]]] += 1
            
        
    return confusion_matrix


def p_r_based_on_c_m(
    predictions: List[dict], 
    dataset: CellMaskDataset, 
    class_ids_to_classnames_map: Dict[int, str], 
    model_label_map: Dict[int, str], 
    annotation_filter: AnnotationFilter = None,
    use_masks: bool = False, 
    min_iou: float = 0.5
):
    """
    This function uses the confusion matrix for computing precision and recall. It is slightly different than the evaluate_pr
    method in the following senses:
    - The pairing between ground truths and detections is done at once for all the classes, unlike evaluate_pr where the 
      pairing is restricted to objects of a given class. Duplicate detections of the same object in different classes 
      (otherwise NMS would remove the duplicate one) by the model have different impacts here. In evaluate_pr, duplicates from
      the wrong class only contributes towards the FP of this class. In this code, the duplicate detection with the wrong class
      can have a better IoU with the GTs. This event contributes toward an FN for the correct class, and an FP for both classes. 
      So the difference indicates the level of duplicate detections (that wrongly get paired)
    - We may want to combine a number of classes (e.g., cell and cell-adhered) and train the model on the consolidated class
      for multiple reasons (for example noisy annotations between the classes). This function allows us to compute the recall
      (the percentage of the GTs that the model can correctly detect) for each contributing class (rather than the combined). 
      This can be used to check how the model performs on such classes when consolidated. The code uses the passed annotation_filter
      to understand which classes are combined for evaluation. 
    """

    # classnames and class IDs 
    dataset_class_ids: List[int] = list(set(dataset.class_names_to_ids_map.values()))
    dataset_class_names: List[str] = []
    # list of class IDs that the model can predict (from its label map and also exists in dataset_class_ids)
    # if a model predicts a class ID outside the dataset_class_ids, it will not be considered in the analysis
    ids_predicted_by_model: List[int] = []
    
    for class_id in dataset_class_ids:
        dataset_class_names.append(class_ids_to_classnames_map[class_id])
        if class_id in model_label_map:
            ids_predicted_by_model.append(class_id)

    c_m: np.ndarray = calculate_confusion_matrix(predictions, dataset, use_masks, min_iou)
    # confusion matrix is calculated for all the class IDs present in the dataset (dataset_class_ids above)
    # keep rows of the confusion matrix that are included in the model's label_map and remove the rest
    row_idxs_to_keep: List[int] = [i for i, c_id in enumerate(dataset_class_ids) if c_id in ids_predicted_by_model]
    model_classnames_to_keep : List[str] = [class_ids_to_classnames_map[c_id] for c_id in dataset_class_ids 
                                            if c_id in ids_predicted_by_model]
    # include the last row, which is the FNs
    c_m = c_m[np.array(row_idxs_to_keep + [c_m.shape[0] - 1]), :]
    # put the confusion matrix in a Pandas DataFrame
    c_m_df = pd.DataFrame(index = model_classnames_to_keep + ['FN'], 
                          columns = dataset_class_names + ['FP'], data=c_m) # the last column is FPs

    # in the following, if the model uses one consolidated class for mutiple class IDs in the dataset (multiple dataset class IDs are 
    # mapped to one model class ID specified in the annotation_filter), we identify them before calculating precision and recall correctly
    model_class_id_to_dataset_classnames_map: Dict[int, List[str]] = {}
    for class_id in model_label_map:
        if class_id in dataset_class_ids:
            mapped_ids: List[int] = [class_id]
        else:
            continue
        if annotation_filter is not None:
            # annotation_filter.class_ids_mapping_dict is a dictionary specifying the mapping between dataset class IDs
            # to model class IDs
            for k, v in annotation_filter.class_ids_mapping_dict.items():
                if v == class_id:
                    # I think we can just append k, which is a class ID?
                    mapped_ids.append(dataset.class_names_to_ids_map[class_ids_to_classnames_map[k]])
        mapped_ids = list(set(mapped_ids))
        model_class_id_to_dataset_classnames_map[class_id] = [class_ids_to_classnames_map[k] for k in mapped_ids]   

    precision: Dict[str, float] = {}
    recall: Dict[str, float] = {}
    recall_w_break_down: Dict[str, float] = {}
    
    for class_id in model_label_map:
        if class_id not in dataset_class_ids:
            continue
        # precision
        fp_col_names: List[str] = [col for col in c_m_df.columns.tolist() if col not in model_class_id_to_dataset_classnames_map[class_id]]
        row = c_m_df.loc[class_ids_to_classnames_map[class_id], model_class_id_to_dataset_classnames_map[class_id] + fp_col_names].values
        tp = np.sum(row[:len(model_class_id_to_dataset_classnames_map[class_id])])
        if np.sum(row) > 0:
            precision[class_ids_to_classnames_map[class_id]] = np.round(tp / np.sum(row), 3)
        # recall
        fn_row_names: List[str] = [ro for ro in c_m_df.index.tolist() if ro != class_ids_to_classnames_map[class_id]]
        fn = c_m_df.loc[fn_row_names, model_class_id_to_dataset_classnames_map[class_id]].values.sum()
        if tp + fn > 0:
            recall[class_ids_to_classnames_map[class_id]] = np.round(tp / (tp + fn), 3)
    
    for class_id in dataset_class_ids:
        if annotation_filter is not None and class_id in annotation_filter.class_ids_mapping_dict:
            model_class_id: int = annotation_filter.class_ids_mapping_dict[class_id]
        else:
            model_class_id: int = class_id
        
        if class_ids_to_classnames_map[model_class_id] not in c_m_df.index:
            # this class ID is mapped to bg
            continue
        tp = c_m_df.loc[class_ids_to_classnames_map[model_class_id],  class_ids_to_classnames_map[class_id]]
        fn_row_names: List[str] = [ro for ro in c_m_df.index.tolist() if ro != class_ids_to_classnames_map[model_class_id]]
        fn = c_m_df.loc[fn_row_names,  class_ids_to_classnames_map[class_id]].sum()
        if tp + fn > 0:
            recall_w_break_down[class_ids_to_classnames_map[class_id]] = np.round(tp / (tp + fn), 3)
        else:
            recall_w_break_down[class_ids_to_classnames_map[class_id]] = 0

    p_r_df = pd.DataFrame(data=[precision, recall], index = ['Precision', 'Recall']).transpose()
    p_r_with_break_down_df = pd.DataFrame(data=[precision, recall_w_break_down], index = ['Precision', 'Recall']).transpose()
    
    return c_m_df, p_r_df, p_r_with_break_down_df
