from typing import List, Dict, Union, Optional
from utils.json_parser import CellMaskDataset
from torch.utils.data import ConcatDataset
from utils.precision_recall_eval import AnnotationFilter
from utils.dataset_utils import create_dataset_classes
from tqdm import tqdm
from models.AbstractVisionModel import run_model
from utils.pairing_utils import pair_gts_dets_bbox, pair_gts_dets_mask
import numpy as np
import cv2
from PIL import Image
import torch
from models.yolo_model import (
    DEFAULT_CROP_CORNERS_10x,
    DEFAULT_RESIZE_10x,
    DEFAULT_CROP_CORNERS_4x,
    DEFAULT_RESIZE_4x,
)
import json


def get_unique_labels(json_path):
    """
    A function to check class labels across all datasets

    Loads a JSON file, parses it to find all "name" labels within
    the "annotations" list, and returns a set of unique labels.

    Args:
        json_path (str): The file path to the JSON file.

    Returns:
        set: A set containing all unique "name" values found.
             Returns an empty set if the file isn't found,
             is invalid JSON, or the expected structure is missing.
    """
    unique_labels = set()

    try:
        # Open and load the JSON file
        with open(json_path, "r") as f:
            data = json.load(f)

            # Check if 'annotations' key exists and is a list
            if "annotations" in data and isinstance(data["annotations"], list):
                # Iterate through each item in the 'annotations' list
                for annotation in data["annotations"]:
                    # Check if the item is a dictionary and has a 'name' key
                    if isinstance(annotation, dict) and "name" in annotation:
                        # Add the 'name' value to our set
                        unique_labels.add(annotation["name"])
            else:
                print(
                    f"Warning: 'annotations' key not found or is not a list in {json_path}"
                )

    except FileNotFoundError:
        print(f"Error: The file '{json_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{json_path}' contains invalid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return unique_labels


def get_test_set(DATASET_PATHS, CLASS_NAME2ID_MAPPING):
    test_dataset_class_list: List[CellMaskDataset] = []
    for dataset_path in tqdm(DATASET_PATHS):
        _, test_dataset = create_dataset_classes(
            dataset_path=dataset_path,
            class_names_to_class_ids_map=CLASS_NAME2ID_MAPPING,
            max_images_to_consider_for_each_annotation=1,
            only_use_best_focus_image=True,  # only consider the best in-focus image for now, we can randomize this later
        )
        test_dataset_class_list.append(test_dataset)

    # concatenate datasetΩ
    test_dataset = ConcatDataset(test_dataset_class_list)
    # carry over the mapping as ConcatDataset is not carrying it
    test_dataset.class_names_to_ids_map = CLASS_NAME2ID_MAPPING

    print(
        f"{len(DATASET_PATHS)} test datasets with a total of {len(test_dataset)} samples for evaluation."
    )
    return test_dataset


@torch.no_grad()
def infer(detector, test_dataset):
    """
    takes the model and concatenated test set as input
    and returns the predictions and run times for computing the PR metrics

    """
    predictions: List[Dict[str, list]] = []
    runtimes: List[float] = []

    # run them in a loop, it is not inefficient at least for Mask R-CNN where batching inputs does not improve speed much
    for datasample in tqdm(test_dataset):
        if "yolo" in detector.get_model_name().lower():
            detections, runtime = run_model(
                detector=detector,
                input_image=datasample["image"],
                input_resize=DEFAULT_RESIZE_4x
                if detector.get_metadata()["magnification"] == "4x"
                else DEFAULT_RESIZE_10x,  # need to pass them as the weights files does not provide them
                input_crop_corners=DEFAULT_CROP_CORNERS_4x
                if detector.get_metadata()["magnification"] == "4x"
                else DEFAULT_CROP_CORNERS_10x,  # need to pass them as the weights files does not provide them
                normalize_image=False,  # the images are already normalized with bit-depth converted to 8
                bit_depth=8,
                post_process_class_names=list(detector.get_reverse_label_map().keys()),
                plot_results=False,
            )
        else:
            detections, runtime = run_model(
                detector=detector,
                input_image=datasample["image"],
                normalize_image=False,  # the images are already normalized with bit-depth converted to 8
                bit_depth=8,
                post_process_class_names=list(detector.get_reverse_label_map().keys()),
                plot_results=False,
            )
        # the PR codes assume the boxes, labels and scores are all numpy arrays (faster pairing, ...)
        # convert them below
        if detector.get_metadata()["predict_masks"]:
            masks = detections["masks"]
        else:
            masks = None
        detections = {k: np.array(v) for k, v in detections.items() if k != "masks"}
        if masks:
            detections["masks"] = masks

        predictions.append(detections)
        runtimes.append(runtime)
    print(
        f"Running the {detector.get_model_name()} model took {np.round(np.mean(runtimes) * 1000)}ms on average per image"
    )
    return predictions, runtimes


def show_detections_updated(
    input_image: Union[Image.Image, np.array],
    detections: Dict[str, Union[list, np.array]],
    label_map: Dict[int, str],
    pred: bool = False,
):
    """
    Displays detections on an image with a single color.
    Uses GREEN for Ground Truth (pred=False) and RED for Predictions (pred=True).
    """
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    color = RED if pred else GREEN

    if isinstance(input_image, np.ndarray):
        image = input_image.copy()
    else:
        image = np.array(input_image)

    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if isinstance(input_image, Image.Image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    boxes = detections["boxes"]
    labels = detections["labels"]

    for i in range(len(labels)):
        xtl, ytl, xbr, ybr = map(int, boxes[i])
        text = label_map.get(labels[i], f"Unknown ID: {labels[i]}")

        if (
            "masks" in detections
            and detections["masks"] is not None
            and len(detections["masks"]) > i
        ):
            mask = detections["masks"][i]
            bool_mask = mask.astype(bool)

            # Ensure ROI is valid
            if ytl < ybr and xtl < xbr:
                color_overlay = np.zeros_like(image[ytl:ybr, xtl:xbr])
                color_overlay[bool_mask, :] = color

                roi = image[ytl:ybr, xtl:xbr]
                blended_roi = cv2.addWeighted(roi, 0.6, color_overlay, 0.4, 0)

                roi[bool_mask] = blended_roi[bool_mask]

        line_thickness = 1 if pred == False else 2
        cv2.rectangle(image, (xtl, ytl), (xbr, ybr), color, line_thickness)
        cv2.putText(
            image,
            text,
            (xtl, ytl - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            line_thickness,
        )

    return image


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
    annotation_filter: Optional[AnnotationFilter] = None,
    show_original_labels: bool = False,
) -> np.ndarray:
    """Visualization function to pair bboxes and plot them."""
    vis_image = (
        cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        if isinstance(image, Image.Image)
        else image.copy()
    )
    if len(vis_image.shape) < 3:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)

    ground_truth_eval = (
        annotation_filter.apply(ground_truth_original)
        if annotation_filter
        else ground_truth_original
    )

    gt_boxes_eval = ground_truth_eval["annotations"][
        ["xtl", "ytl", "xbr", "ybr"]
    ].values.astype(int)
    gt_labels_eval = ground_truth_eval["annotations"]["label"].values
    gt_masks_eval = ground_truth_eval.get("masks")

    original_indices = ground_truth_eval["annotations"].index
    original_labels_for_eval_set = (
        ground_truth_original["annotations"].loc[original_indices, "label"].values
    )

    pred_boxes = (
        np.array(predictions["boxes"]).astype(int)
        if "boxes" in predictions and len(predictions["boxes"]) > 0
        else np.zeros((0, 4), dtype=int)
    )
    pred_labels = (
        np.array(predictions["labels"])
        if "labels" in predictions and len(predictions["labels"]) > 0
        else np.zeros((0,), dtype=int)
    )
    pred_scores = (
        np.array(predictions["scores"])
        if "scores" in predictions and len(predictions["scores"]) > 0
        else np.zeros((0,), dtype=float)
    )
    pred_masks = predictions.get("masks")

    class_ids_to_filter = list(label_map.keys())

    all_paired_gt_indices = set()
    all_paired_det_indices = set()
    all_unpaired_gt_indices = set(range(len(gt_labels_eval)))
    all_unpaired_det_indices = set(range(len(pred_labels)))

    for class_id in class_ids_to_filter:
        if class_id == label_map.get("bg", -1):
            continue  # Skip bg class

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
                gt_boxes_class,
                gt_masks_class,
                det_boxes_class,
                det_masks_class,
                min_iou,
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
        cv2.putText(
            vis_image,
            label_text,
            (box[0], box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    # Draw False Negatives (Red Dashed)
    for gt_idx in all_unpaired_gt_indices:
        box = gt_boxes_eval[gt_idx]
        display_label_id = (
            original_labels_for_eval_set[gt_idx]
            if show_original_labels
            else gt_labels_eval[gt_idx]
        )
        if display_label_id not in label_map or label_map[display_label_id] == "bg":
            continue  # Don't draw FNs for 'bg' or excluded classes
        label_text = f"FN: {label_map.get(display_label_id, 'N/A')}"
        _draw_dashed_rectangle(
            vis_image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2
        )
        cv2.putText(
            vis_image,
            label_text,
            (box[0], box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    # Draw False Positives (Yellow)
    for det_idx in all_unpaired_det_indices:
        box = pred_boxes[det_idx]
        score = pred_scores[det_idx]
        if (
            pred_labels[det_idx] not in label_map
            or label_map[pred_labels[det_idx]] == "bg"
        ):
            continue  # Don't draw FPs for 'bg'
        label_text = f"FP: {label_map.get(pred_labels[det_idx], 'N/A')} ({score:.2f})"
        cv2.rectangle(vis_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
        cv2.putText(
            vis_image,
            label_text,
            (box[0], box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

    return vis_image
