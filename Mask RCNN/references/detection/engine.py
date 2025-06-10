import math
import sys
import time
import numpy as np
import torch
import torchvision.models.detection.mask_rcnn
import utils

from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
import pycocotools.mask as mask_util
from pycocotools.cocoeval import COCOeval
from coco_eval import convert_to_xywh
from tqdm import tqdm


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

def train_one_epoch_one_cycle_lrs(model, optimizer, in_lr_scheduler, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    
    if in_lr_scheduler is None:
        if epoch == 0:
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, len(data_loader) - 1)

            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=warmup_factor, total_iters=warmup_iters
            )
        else:
            lr_scheduler = None
    else:
        lr_scheduler = in_lr_scheduler

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device, max_dets=100):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    
    # set the maxDets
    for iou_type in iou_types:
        coco_evaluator.coco_eval[iou_type].params.maxDets = [1, 10, max_dets]

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
    
    
@torch.inference_mode()
def evaluate_coco_segm(model, data_loader, device, max_dets=100):

    new_test_dataset_format: bool = False
    if not isinstance(data_loader.dataset, torchvision.datasets.CocoDetection):
        if hasattr(data_loader.dataset, 'dataset_coco'):
            # CellMaskDataset class, we can add a specific check for the type instead if import the class here (not done for simplicity)
            new_test_dataset_format = True
            
        else:
            print(f"[ERROR]: evaluate_coco_segm only supports COCO dataset format (torchvision.datasets.CocoDetection)! \n"
                  f"The passed data_loader's dataset type is {type(data_loader.dataset)}. No evaluation is possible")
            return COCOeval(), COCOeval()
    
    if new_test_dataset_format:
        coco_dataset = data_loader.dataset.dataset_coco.coco
    else:
        coco_dataset = data_loader.dataset.coco
        
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()

    
    all_results = []
    all_image_ids = []
    model_time = 0
    for images, targets in tqdm(data_loader):
        images = list(img.to(device) for img in images)
         
        start_time = time.time()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        results = {target["image_id"]: output for target, output in zip(targets, outputs)}
        results = convert_preds_to_coco(results)
        all_results.extend(results)
        all_image_ids += [target["image_id"] for target in targets]
        model_time += time.time() - start_time

    if new_test_dataset_format:
        # image IDs in CocoDetection are integers, while in the target are tensors, comvert them
        # this is needed only for the new format
        all_image_ids = [int(image_id.item()) for image_id in all_image_ids]
    
        
    coco_gt = coco_dataset
    coco_dt = coco_gt.loadRes(all_results)  # init predictions api
    
    evaluator_time = time.time()
    
    # bounding box evaluation
    coco_evaluator_bbox = COCOeval(coco_gt, coco_dt, "bbox")
    coco_evaluator_bbox.params.maxDets = [1, 10, max_dets]
    coco_evaluator_bbox.params.imgIds = all_image_ids
    coco_evaluator_bbox.evaluate()
    coco_evaluator_bbox.accumulate()
    coco_evaluator_bbox.summarize()
    
    # segmentation evaluation
    coco_evaluator_segm = COCOeval(coco_gt, coco_dt, "segm")
    coco_evaluator_segm.params.maxDets = [1, 10, max_dets]
    coco_evaluator_segm.params.imgIds = all_image_ids
    coco_evaluator_segm.evaluate()
    coco_evaluator_segm.accumulate()
    coco_evaluator_segm.summarize()
    
    
    evaluator_time = time.time() - evaluator_time
    
    print("model_time:", model_time)
    print("evaluator_time:", evaluator_time)

    torch.set_num_threads(n_threads)
    return coco_evaluator_bbox, coco_evaluator_segm
    

def convert_preds_to_coco(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue
        
        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()
         
        masks = prediction["masks"]
        masks = masks > 0.5

        rles = [
            mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0] for mask in masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": boxes[k],
                    "segmentation": rle,
                    "score": scores[k],
                }
                for k, rle in enumerate(rles)
            ]
        )
    return coco_results
