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


def convert_preds_to_coco(predictions, model_to_coco=None):
    """
    Convert predictions from:
      {image_id: {"boxes": Tensor[N,4], "scores": Tensor[N], "labels": Tensor[N]}}
    to COCO list[dict].
    """
    model_to_coco = {int(k): int(v) for k, v in (model_to_coco or {}).items()}
    coco_results = []
    for image_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()
        segmentations = prediction.get("segmentation", None)

        for idx in range(len(scores)):
            res = {
                "image_id": int(image_id),
                "category_id": int(
                    model_to_coco.get(int(labels[idx]), int(labels[idx]))
                ),
                "bbox": boxes[idx],
                "score": float(scores[idx]),
            }
            if segmentations is not None and idx < len(segmentations):
                res["segmentation"] = segmentations[idx]

            coco_results.append(res)

    return coco_results


def gather_outputs_across_processes(local_outputs):
    """Gather Python objects from all ranks and flatten on rank 0."""
    if not dist.is_available() or not dist.is_initialized():
        return local_outputs

    world_size = dist.get_world_size()
    if world_size <= 1:
        return local_outputs

    gathered = [None for _ in range(world_size)]
    dist.gather_object(local_outputs, gathered if dist.get_rank() == 0 else None, dst=0)
    
    if dist.get_rank() == 0:
        return [item for rank_outputs in gathered for item in rank_outputs]
    return []


def broadcast_object(obj, src=0):
    """Broadcast a Python object from src rank to all ranks."""
    if not dist.is_available() or not dist.is_initialized():
        return obj
    object_list = [obj]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


def compute_coco_metrics(
    coco_gt,
    predictions,
    image_ids,
    max_detections=100,
    label_map=None,
    prefix="Performance",
    iou_type="bbox",
    metric_prefix="",
):
    """
    Compute aggregate + per-class COCO metrics.
    Returns a dict containing map/map_50/... and per-class map_* keys.
    When metric_prefix is provided, all returned keys are prefixed, e.g. segm_map.
    """
    base_metrics = {
        "map": 0.0,
        "map_50": 0.0,
        "map_75": 0.0,
        "map_small": 0.0,
        "map_medium": 0.0,
        "map_large": 0.0,
        "mar_1": 0.0,
        "mar_10": 0.0,
        f"mar_{max_detections}": 0.0,
        "mar_small": 0.0,
        "mar_medium": 0.0,
        "mar_large": 0.0,
    }
    metric_prefix = str(metric_prefix or "").strip()
    key_prefix = f"{metric_prefix}_" if metric_prefix else ""
    metrics = {f"{key_prefix}{key}": value for key, value in base_metrics.items()}

    if coco_gt is None:
        if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
            print(f"\n{prefix}: No GT to evaluate.")
        return metrics

    import copy
    import contextlib
    import io

    is_dist = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if is_dist else 1
    rank = dist.get_rank() if is_dist else 0

    try:
        # 1. Evaluate locally
        local_eval_imgs = []
        if len(image_ids) > 0:
            coco_dt = coco_gt.loadRes(predictions if len(predictions) > 0 else [])
            coco_eval_local = COCOeval(coco_gt, coco_dt, iou_type)
            coco_eval_local.params.maxDets = [1, 10, max_detections]
            coco_eval_local.params.imgIds = image_ids
            # Suppress eval prints per rank
            with contextlib.redirect_stdout(io.StringIO()):
                coco_eval_local.evaluate()
            local_eval_imgs = coco_eval_local.evalImgs

        # 2. Gather correctly
        if is_dist and world_size > 1:
            gathered_eval_imgs = [None for _ in range(world_size)]
            dist.gather_object(local_eval_imgs, gathered_eval_imgs if rank == 0 else None, dst=0)
            
            gathered_preds = [None for _ in range(world_size)]
            dist.gather_object(predictions, gathered_preds if rank == 0 else None, dst=0)
            
            gathered_img_ids = [None for _ in range(world_size)]
            dist.gather_object(image_ids, gathered_img_ids if rank == 0 else None, dst=0)
            
            if rank != 0:
                return metrics
                
            all_eval_imgs = [img for sublist in gathered_eval_imgs if sublist for img in sublist]
            all_preds = [p for sublist in gathered_preds if sublist for p in sublist]
            
            # Since DistributedSampler splits datasets, different ranks have DIFFERENT image_ids
            # We must concatenate them, not just take the unique subset if they overlap (though they shouldn't)
            all_img_ids = list(set([i for sublist in gathered_img_ids if sublist for i in sublist]))
            all_img_ids.sort() # sort for deterministic ordering
            
            if len(all_img_ids) == 0:
                print(f"\n{prefix}: No images to evaluate globally.")
                return metrics
                
            master_coco_dt = coco_gt.loadRes(all_preds if len(all_preds) > 0 else [])
            coco_eval = COCOeval(coco_gt, master_coco_dt, iou_type)
            coco_eval.params.maxDets = [1, 10, max_detections]
            coco_eval.params.imgIds = all_img_ids
            
            # We must call evaluate() on rank 0 just to populate `_paramsEval` exactly as pycocotools expects.
            # But we replace the empty `evalImgs` it produces (since it evaluated nothing valid or too slow if it did)
            # with our pre-computed distributed `evalImgs` immediately after.
            # However, evaluating on Rank 0 again defeats the purpose.
            # PyCOCOtools expects `_paramsEval` to have `catIds`, `areaRng`, `imgIds`.
            # Let's populate it manually without running evaluate()!
            import numpy as np
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
            
            # The exact logic from pycocotools COCOeval.evaluate() that sets up `_paramsEval`:
            p = coco_eval._paramsEval
            p.imgIds = list(np.unique(p.imgIds))
            if p.useCats:
                p.catIds = list(np.unique(p.catIds))
            p.maxDets = sorted(p.maxDets)
            coco_eval.params = p
            
            # Reorder evalImgs so accumulate() indexes them properly
            lookup = {}
            for res in all_eval_imgs:
                if res is None: continue
                # pycocotools structure is: res = {'image_id': ..., 'category_id': ..., 'aRng': [...], 'maxDet': ...}
                # But notice areaRng might be a list, make it tuple for dict key
                key = (res['category_id'], tuple(res['aRng']), res['image_id'])
                lookup[key] = res
                
            ordered_eval_imgs = []
            for catId in p.catIds:
                for areaRng in p.areaRng:
                    for imgId in p.imgIds:
                        key = (catId, tuple(areaRng), imgId)
                        # We append the eval result or None if it's missing (pycocotools handles None)
                        ordered_eval_imgs.append(lookup.get(key, None))
                        
            coco_eval.evalImgs = ordered_eval_imgs
        else:
            if len(image_ids) == 0:
                print(f"\n{prefix}: No images to evaluate locally.")
                return metrics
            coco_eval = coco_eval_local

        # 3. Accumulate and summarize on Rank 0
        with contextlib.redirect_stdout(io.StringIO()):
            # By default PyCOCOtools populate self.eval internally when accumulate() runs
            # We explicitly check that evalImgs is not fully empty or None
            if not any(coco_eval.evalImgs):
                print(f"\n{prefix}: Warning, no valid evalImgs gathered, returning empty metrics.")
                return metrics
                
            coco_eval.accumulate()
            
        coco_eval.summarize()

        keys = list(base_metrics.keys())
        for idx, key in enumerate(keys):
            if idx < len(coco_eval.stats):
                metrics[f"{key_prefix}{key}"] = round(float(coco_eval.stats[idx]), 4)

        if hasattr(coco_eval, "eval") and "precision" in coco_eval.eval:
            precisions = coco_eval.eval["precision"]
            recalls = coco_eval.eval.get("recall")
            import numpy as np

            # For formatting YOLO style
            num_images = len(coco_eval.params.imgIds)

            # Find number of labels per class from coco_gt annotations
            labels_per_class = {}
            total_labels = 0
            eval_img_ids_set = set(coco_eval.params.imgIds)
            for ann in coco_gt.dataset.get("annotations", []):
                # Only count annotations in the evaluated images
                if ann["image_id"] in eval_img_ids_set:
                    c_id = ann["category_id"]
                    labels_per_class[c_id] = labels_per_class.get(c_id, 0) + 1
                    total_labels += 1

            table_rows = []

            for class_idx, cat_id in enumerate(coco_eval.params.catIds):
                class_name = None
                if int(cat_id) in coco_gt.cats:
                    class_name = coco_gt.cats[int(cat_id)]["name"]
                if class_name is None and label_map is not None:
                    class_name = label_map.get(int(cat_id)) or label_map.get(
                        str(cat_id)
                    )
                if class_name is None:
                    class_name = f"class_{cat_id}"

                p_all = precisions[:, :, class_idx, 0, -1]
                valid_all = p_all[p_all > -1]
                if len(valid_all) > 0:
                    metrics[f"{key_prefix}map_{class_name}"] = round(
                        float(np.mean(valid_all)), 4
                    )

                p_50 = precisions[0, :, class_idx, 0, -1]
                valid_50 = p_50[p_50 > -1]
                if len(valid_50) > 0:
                    metrics[f"{key_prefix}map_50_{class_name}"] = round(
                        float(np.mean(valid_50)), 4
                    )

                if recalls is not None:
                    for recall_idx, recall_val in enumerate([1, 10, max_detections]):
                        if recall_idx >= recalls.shape[3]:
                            continue
                        r_all = recalls[:, class_idx, 0, recall_idx]
                        valid_r_all = r_all[r_all > -1]
                        if len(valid_r_all) > 0:
                            metrics[f"{key_prefix}mar_{recall_val}_{class_name}"] = (
                                round(float(np.mean(valid_r_all)), 4)
                            )

                # Compute best P and R at IoU 0.5 (index 0)
                best_p, best_r = 0.0, 0.0
                if len(valid_50) > 0:
                    # precisions for IoU=0.5, area=All, maxDets=max_detections
                    # shape is (101,) corresponding to 101 recall points
                    p_curve = precisions[0, :, class_idx, 0, -1]
                    # The corresponding recalls used by COCOeval
                    r_curve = np.linspace(0.0, 1.0, len(p_curve))

                    # Compute F1 for all points, ignoring invalid points (-1)
                    valid_mask = p_curve > -1
                    if np.any(valid_mask):
                        valid_p = p_curve[valid_mask]
                        valid_r = r_curve[valid_mask]
                        # Avoid division by zero
                        denominator = valid_p + valid_r + 1e-16
                        f1_curve = 2 * valid_p * valid_r / denominator
                        best_idx = np.argmax(f1_curve)
                        best_p = valid_p[best_idx]
                        best_r = valid_r[best_idx]

                best_f1 = 2 * best_p * best_r / (best_p + best_r + 1e-16)

                table_rows.append(
                    {
                        "Class": class_name,
                        "Images": num_images,
                        "Labels": labels_per_class.get(cat_id, 0),
                        "P": best_p,
                        "R": best_r,
                        "F1": best_f1,
                        "mAP@.5": metrics.get(f"{key_prefix}map_50_{class_name}", 0.0),
                        "mAP@.5:.95": metrics.get(f"{key_prefix}map_{class_name}", 0.0),
                    }
                )

            # Calculate and add 'all' row
            # For "all", we can use the overall stats from coco_eval
            # stats[0] = AP @.5:.95, stats[1] = AP @.5, stats[8] = AR @maxDets
            # To get P and R for "all" at best F1, we could average the class curves, but
            # Pycocotools doesn't provide a single curve for "all". So we take average of best P and R.
            avg_p = np.mean([row["P"] for row in table_rows]) if table_rows else 0.0
            avg_r = np.mean([row["R"] for row in table_rows]) if table_rows else 0.0
            avg_f1 = np.mean([row["F1"] for row in table_rows]) if table_rows else 0.0

            all_row = {
                "Class": "all",
                "Images": num_images,
                "Labels": total_labels,
                "P": avg_p,
                "R": avg_r,
                "F1": avg_f1,
                "mAP@.5": metrics.get(f"{key_prefix}map_50", 0.0),
                "mAP@.5:.95": metrics.get(f"{key_prefix}map", 0.0),
            }

            # Build markdown table output
            md_table = f"| Class | P | R | F1 | mAP@0.5 | mAP@0.5-0.95 |\n"
            md_table += f"|---|---|---|---|---|---|\n"
            md_table += f"| **{all_row['Class']}** | {all_row['P']:.3f} | {all_row['R']:.3f} | {all_row['F1']:.3f} | {all_row['mAP@.5']:.3f} | {all_row['mAP@.5:.95']:.3f} |\n"
            for r in table_rows:
                md_table += f"| {r['Class']} | {r['P']:.3f} | {r['R']:.3f} | {r['F1']:.3f} | {r['mAP@.5']:.3f} | {r['mAP@.5:.95']:.3f} |\n"
            
            metrics["_markdown_table"] = md_table

            # Print the formatted table
            print(f"\n{prefix}")
            header = f"{'Class':<14}{'Images':<8}{'Labels':<11}{'P':<8}{'R':<9}{'F1':<8}{'mAP@.5':<11}{'mAP@.5:.95':<11}"
            print(header)

            def print_row(r):
                print(
                    f"{r['Class']:<14}{r['Images']:<8}{r['Labels']:<11}{r['P']:<8.3f}{r['R']:<9.3f}{r['F1']:<8.3f}{r['mAP@.5']:<11.3f}{r['mAP@.5:.95']:<11.3f}"
                )

            print_row(all_row)
            for r in table_rows:
                print_row(r)
            print()

    except Exception as e:
        import traceback

        print(f"[COCOEval Error] {e}")
        traceback.print_exc()
        return metrics

    return metrics
