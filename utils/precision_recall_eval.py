from typing import Dict, List, Tuple, Optional, Final, Union
import numpy as np
import pandas as pd
from .pairing_utils import pair_gts_dets_bbox, pair_gts_dets_mask
from .json_parser import enforce_one_to_one_mapping, TestDataSet

class AnnotationFilter:
    def __init__(self, classnames_mapping_dict: Dict[str, str], class_ids_to_class_names_map: Dict[int, str]):
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
    
    
def evaluate_mask_rcnn_pr(predictions, dataset, class_ids_of_interest=None, min_iou=0.5, child_parent_map=None, annotation_filter=None):
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
        annotation_filter (AnnotationFilter class): A class to post process the datasample and exclude or merge some classes.
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

        if annotation_filter is not None:
            annots = annotation_filter.apply(annots)
        
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
        
        confusion_matrix += calculate_sample_confusion_matrix(annots, predictions[idx], gt_class_ids, use_mask, min_iou)
            
        
        if (idx + 1) % 100 == 0:
            print(f"[INFO] Completed {idx+1} images out of {len(dataset)}")
        
    return confusion_matrix
    
    
def calculate_sample_confusion_matrix(data_sample, detections, gt_class_ids, use_mask=False, min_iou=0.5):
 
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


def p_r_based_on_c_m(predictions: List[dict], 
                     dataset: TestDataSet, 
                     class_ids_to_classnames_map: Dict[int, str], 
                     model_label_map: Dict[int, str], 
                     annotation_filter: AnnotationFilter = None,
                     use_masks: bool = True, 
                     min_iou: float = 0.5):
    
    dataset_class_ids: List[int] = list(set(dataset.class_names_to_ids_map.values()))
    dataset_class_names: List[str] = []
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
    # include the last row, which is the FNs, and the last column, which is the FPs
    c_m = c_m[np.array(row_idxs_to_keep + [c_m.shape[0] - 1]), :]
    # put the confusion matrix in a Pandas DataFrame
    c_m_df = pd.DataFrame(index = model_classnames_to_keep + ['FN'], 
                          columns = dataset_class_names + ['FP'], data=c_m)

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
        tp = c_m_df.loc[class_ids_to_classnames_map[model_class_id],  class_ids_to_classnames_map[class_id]]
        fn_row_names: List[str] = [ro for ro in c_m_df.index.tolist() if ro != class_ids_to_classnames_map[model_class_id]]
        fn = c_m_df.loc[fn_row_names,  class_ids_to_classnames_map[class_id]].sum()
        if tp + fn > 0:
            recall_w_break_down[class_ids_to_classnames_map[class_id]] = np.round(tp / (tp + fn ), 3)
        else:
            recall_w_break_down[class_ids_to_classnames_map[class_id]] = 0

    p_r_df = pd.DataFrame(data=[precision, recall], index = ['Precision', 'Recall']).transpose()
    # p_r_df['Class'] = p_r_df.index
    p_r_with_break_down_df = pd.DataFrame(data=[precision, recall_w_break_down], index = ['Precision', 'Recall']).transpose()
    
    return c_m_df, p_r_df, p_r_with_break_down_df

