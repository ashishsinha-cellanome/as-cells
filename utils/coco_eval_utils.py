import torch
import torch.distributed as dist
from pycocotools.cocoeval import COCOeval


def to_cpu_device(data):
    """Detach/move nested tensors to CPU."""
    if isinstance(data, dict):
        return {k: to_cpu_device(v) for k, v in data.items()}
    if isinstance(data, list):
        return [to_cpu_device(v) for v in data]
    if isinstance(data, torch.Tensor):
        return data.detach().cpu()
    return data


def convert_to_xywh(boxes):
    """Convert Nx4 boxes from xyxy to xywh."""
    if len(boxes) == 0:
        return boxes
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def convert_preds_to_coco(predictions):
    """
    Convert predictions from:
      {image_id: {"boxes": Tensor[N,4], "scores": Tensor[N], "labels": Tensor[N]}}
    to COCO list[dict].
    """
    coco_results = []
    for image_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": int(image_id),
                    "category_id": int(labels[idx]),
                    "bbox": boxes[idx],
                    "score": float(scores[idx]),
                }
                for idx in range(len(scores))
            ]
        )
    return coco_results


def gather_outputs_across_processes(local_outputs):
    """Gather Python objects from all ranks and flatten."""
    if not dist.is_available() or not dist.is_initialized():
        return local_outputs

    world_size = dist.get_world_size()
    if world_size <= 1:
        return local_outputs

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_outputs)
    return [item for rank_outputs in gathered for item in rank_outputs]


def broadcast_object(obj, src=0):
    """Broadcast a Python object from src rank to all ranks."""
    if not dist.is_available() or not dist.is_initialized():
        return obj
    object_list = [obj]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


def compute_coco_metrics(coco_gt, predictions, image_ids, max_detections=100, label_map=None):
    """
    Compute aggregate + per-class COCO bbox metrics.
    Returns a dict containing map/map_50/... and per-class map_* keys.
    """
    if coco_gt is None or len(predictions) == 0:
        return {}

    metrics = {
        "map": -1.0,
        "map_50": -1.0,
        "map_75": -1.0,
        "map_small": -1.0,
        "map_medium": -1.0,
        "map_large": -1.0,
        "mar_1": -1.0,
        "mar_10": -1.0,
        f"mar_{max_detections}": -1.0,
        "mar_small": -1.0,
        "mar_medium": -1.0,
        "mar_large": -1.0,
    }

    try:
        coco_dt = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.maxDets = [1, 10, max_detections]
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        keys = list(metrics.keys())
        for idx, key in enumerate(keys):
            if idx < len(coco_eval.stats):
                metrics[key] = round(float(coco_eval.stats[idx]), 4)

        if hasattr(coco_eval, "eval") and "precision" in coco_eval.eval:
            precisions = coco_eval.eval["precision"]
            import numpy as np

            for class_idx, cat_id in enumerate(coco_eval.params.catIds):
                class_name = None
                if label_map is not None:
                    class_name = label_map.get(int(cat_id)) or label_map.get(str(cat_id))
                if class_name is None and int(cat_id) in coco_gt.cats:
                    class_name = coco_gt.cats[int(cat_id)]["name"]
                if class_name is None:
                    class_name = f"class_{cat_id}"

                p_all = precisions[:, :, class_idx, 0, -1]
                valid_all = p_all[p_all > -1]
                if len(valid_all) > 0:
                    metrics[f"map_{class_name}"] = round(float(np.mean(valid_all)), 4)

                p_50 = precisions[0, :, class_idx, 0, -1]
                valid_50 = p_50[p_50 > -1]
                if len(valid_50) > 0:
                    metrics[f"map_50_{class_name}"] = round(float(np.mean(valid_50)), 4)
    except Exception:
        return metrics

    return metrics
