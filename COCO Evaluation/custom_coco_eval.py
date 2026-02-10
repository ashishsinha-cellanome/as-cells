import time
import torch
import numpy as np

from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from coco_eval import CocoEvaluator
from typing import Union, Dict, List

import utils


def convert_to_coco_api(ds, segmentation_model):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {'images': [], 'categories': [], 'annotations': []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        annots = ds[img_idx]
        img = annots['image']
        
        image_id = annots['name']
        img_dict = {}
        img_dict['id'] = image_id
        img_dict['height'] = img.shape[0]
        img_dict['width'] = img.shape[1]
        dataset['images'].append(img_dict)
        bboxes = annots['annotations'][['xtl', 'ytl', 'xbr', 'ybr']].values
        bboxes[:, 2:] -= bboxes[:, :2]
        areas = (bboxes[:, 2] * bboxes[:, 3]).tolist()
        bboxes = bboxes.tolist()
        labels = annots['annotations']['label'].values.tolist()
        iscrowd = [0] * len(labels)
       
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann['image_id'] = image_id
            ann['bbox'] = bboxes[i]
            ann['category_id'] = labels[i]
            categories.add(labels[i])
            ann['area'] = areas[i]
            ann['iscrowd'] = iscrowd[i]
            ann['id'] = ann_id
            
            if 'masks' in annots and segmentation_model:
                # make masks Fortran contiguous for coco_mask (first convert to a torch tensor, then back to numpy array)
                box_xtl, box_ytl, box_xbr, box_ybr = annots['annotations'].loc[i, ['xtl', 'ytl', 'xbr', 'ybr']].values
                numpy_mask = np.zeros((img_dict['width'] , img_dict['height']), np.uint8)
                numpy_mask[box_ytl:box_ybr, box_xtl:box_xbr] = annots['masks'][i].copy()
                mask = torch.tensor(numpy_mask)
                mask = mask.permute(1, 0).contiguous().permute(1, 0).numpy()
                ann["segmentation"] = coco_mask.encode(mask)
            
            dataset['annotations'].append(ann)
            ann_id += 1
    dataset['categories'] = [{'id': i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds

@torch.no_grad()
def evaluate_cellpose(cellpose_model, dataset, max_dets=100):
    """
    Args:
        cellpose_model: CellPose model to be called on test images; this should be the same code in 
        segmentation_cellpose.py from microscope-automation-gui/util accepting an image. 
        dataset: Custom dataset object.
        max_dets: The maximum of maxDets parameter in COCOeval. 
    """
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = convert_to_coco_api(dataset, True)
    iou_types = ["bbox"]
    iou_types.append("segm")
        
    coco_evaluator = CocoEvaluator(coco, iou_types)
    # set the maxDets
    for iou_type in iou_types:
        coco_evaluator.coco_eval[iou_type].params.maxDets = [1, 10, max_dets]

    for idx in range(len(dataset)):        
        annots = dataset[idx]
        
        if torch.cuda.is_available():
           torch.cuda.synchronize()
        
        model_time = time.time()
        # run CellPose
        mask = cellpose_model(annots['image'])
        
        # construct detections in Mask RCNN format
        output = contruct_predictions_from_cellpose_mask(mask)
        
        model_time = time.time() - model_time
        
        res = {annots["name"]: output}
        
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        
        if (idx + 1) % 100 == 0:
            print(f"[INFO] Completed {idx+1} images out of {len(dataset)}")
        

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
    
@torch.no_grad()
def evaluate_mask_rcnn(mask_rcnn_model, dataset, max_dets=100):
    """
    Args:
        mask_rcnn_model: Mask RCNN model to be called on test images; this code should only 
        accept an image. 
        dataset: Custom dataset object.
        max_dets: The maximum of maxDets parameter in COCOeval. 
    """
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = convert_to_coco_api(dataset, True)
    iou_types = ["bbox"]
    iou_types.append("segm")
        
    coco_evaluator = CocoEvaluator(coco, iou_types)
    # set the maxDets
    for iou_type in iou_types:
        coco_evaluator.coco_eval[iou_type].params.maxDets = [1, 10, max_dets]

    for idx in range(len(dataset)):        
        annots = dataset[idx]
        
        if torch.cuda.is_available():
           torch.cuda.synchronize()
        
        # run Mask RCNN
        preds, model_time = mask_rcnn_model(annots['image'])
        
        # construct detections in Mask RCNN format
        output = contruct_predictions_from_mask_rcnn(preds, annots['image'].shape[1], annots['image'].shape[0])
        
        
        res = {annots["name"]: output}
        
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        
        if (idx + 1) % 100 == 0:
            print(f"[INFO] Completed {idx+1} images out of {len(dataset)}")
        

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
    
    
def evaluate_detection_model(predictions, dataset, max_dets=100):
    """
    Args:
        predictions: List of dictionaries; the i-th dictionary in the list should include the 
            predictions for dataset[i] image (and annotations)  with keys as 'boxes', 'labels', 'scores'
            and values as numpy arrays or torch tensors of (num_detections, 4), 
            (num_detections,), (num_detections,), sizes for bounding boxes, labels, and scores.
        dataset: Custom dataset object.
        max_dets: The maximum of maxDets parameter in COCOeval. 
    """
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = convert_to_coco_api(dataset, False)
    iou_types = ["bbox"]

    coco_evaluator = CocoEvaluator(coco, iou_types)
    coco_evaluator.coco_eval["bbox"].params.maxDets = [1, 10, max_dets]

    for idx in range(len(dataset)):
        annots = dataset[idx]
        
        # convert to torch.tensor if they are not already
        output: dict = {}
        for k, v in predictions[idx].items():
            output[k] = torch.tensor(v) if isinstance(v, np.ndarray) else v
        
        res = {annots["name"]: output}
        
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=0, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
    
    
def contruct_predictions_from_cellpose_mask(in_mask: Union[str, np.ndarray]) -> dict:
    """
    A helper function to take CellPose instance segmentation results and construct an object
    (a dictionary) with the same format expected from torchvision's Mask RCNN model. This
    allows using the same COCO evaluation function from torchvision. 
    Args:
        in_mask (a numpy array or a string): The numpy array of the mask returned by CellPose, or
        the path to the saved mask (should include the path and the filename). Note that the mask 
        should be the same size as the input image to CellPose. 
    
    Returns
        A dictionary with keys as "boxes", "labels", "scores" and "masks" and values as
        (num_detections, 4) torch.float32 bounding boxes tensor, (num_detections, ) torch.int64 
        labels tensor, (num_detections, ) torch.float32 scores tensor, and 
        (num_detections, 1, image_height, image_width) torch.float32 masks tensor.
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
    valid_ids = []
    for i in range(num_objs):
        pos = np.where(masks[i])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        if xmin < xmax and ymin < ymax:
            valid_ids.append(i)
            boxes.append([xmin, ymin, xmax, ymax])
        
    num_objs = len(valid_ids)
    # there is only one class (cell), make sure "Cell" class name is always mapped to 1 
    # when annotations are passed to COCO evaluation
    labels = torch.ones((num_objs,), dtype=torch.int64)
            
    # convert everything into a torch.Tensor
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
    # masks, at this point the masks are binary (0, 1) but the 
    # results from the Mask RCNN model are torch.float32
    # so we keep them the same
    
    masks = torch.as_tensor(masks[valid_ids], dtype=torch.float32)
    
    masks = torch.cat([mask.unsqueeze(0).unsqueeze(0) for mask in masks])
    
    predictions = {}
    predictions["boxes"] = boxes
    predictions["labels"] = labels
    predictions["masks"] = masks
    predictions["scores"] = torch.ones((num_objs,), dtype=torch.float32)
       
    return predictions
    
def contruct_predictions_from_mask_rcnn(preds: Dict[str, List], image_width: int, image_height: int) -> dict:
    """
    A helper function to take our Mask RCNN instance segmentation results and construct an object
    (a dictionary) with the same format expected from torchvision's Mask RCNN model. This
    allows using the same COCO evaluation function from torchvision. 
    Args:
        preds (dictionary): A dictionary with keys and values as below:
        "boxes": List of 4-tuples or 4-element lists for the detected objects' bounding boxes in 
            xtl, ytl, xbr, ybr format/order
        "labels": List of integer class IDs for the detected objects
        "scores": List of float detection scores, thresholded by self._confidence
        "masks": List of masks for the detected objects. Each mask is a numpy array of the same size
            as the bounding box (xbr - xtl, ybr - ytl)
        image_width (int): Image width.
        image_height (int): Image height. 
    Returns
        A dictionary with keys as "boxes", "labels", "scores" and "masks" and values as
        (num_detections, 4) torch.float32 bounding boxes tensor, (num_detections, ) torch.int64 
        labels tensor, (num_detections, ) torch.float32 scores tensor, and 
        (num_detections, 1, image_height, image_width) torch.float32 masks tensor.
    """
    
    num_objs = len(preds["labels"])
    
    # convert boxes, labels and scores into a torch.Tensor
    boxes = torch.as_tensor(np.array(preds["boxes"]), dtype=torch.float32)
    labels = torch.as_tensor(np.array(preds["labels"]), dtype=torch.int64)
    # scores = torch.as_tensor(np.array(preds["scores"]), dtype=torch.float32)
    scores = torch.ones((num_objs,), dtype=torch.float32)
    
    masks = np.zeros((num_objs, image_height, image_width))
    for i in range(len(preds["masks"])):
        xtl, ytl, xbr, ybr = preds["boxes"][i]
        masks[i, ytl:ybr, xtl:xbr] = preds["masks"][i]
    
    masks = torch.as_tensor(masks, dtype=torch.float32)
    
    masks = torch.cat([mask.unsqueeze(0).unsqueeze(0) for mask in masks])
    
    predictions = {}
    predictions["boxes"] = boxes
    predictions["labels"] = labels
    predictions["masks"] = masks
    predictions["scores"] = scores
       
    return predictions
    
    
    
def contruct_predictions_from_detections(boxes: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> dict:
    """
    A helper function to take results from an object detection model (e.g., YOLO) and construct an object
    (a dictionary) with the same format expected from torchvision's Faster RCNN model. This
    allows using the same COCO evaluation function from torchvision. 
    Args:
        boxes: (num_detection, 4) numpy array of bounding boxes.
        labels: (num_detection,) numpy array of labels.
        scores: (num_detection,) numpy array of scores.
    
    Returns
        A dictionary with keys as "boxes", "labels" and "scores" and values as
        (num_detections, 4) torch.float32 bounding boxes tensor, (num_detections, ) torch.int64 
        labels tensor, and (num_detections, ) torch.float32 scores tensor.
    """     
    
    predictions = {}
    predictions["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
    predictions["labels"] = torch.as_tensor(labels, dtype=torch.int64)
    predictions["scores"] = torch.as_tensor(scores, dtype=torch.float32)
       
    return predictions
    
