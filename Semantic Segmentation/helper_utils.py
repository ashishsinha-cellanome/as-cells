import os
import pickle
from typing import Dict, List, Final, Tuple

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as coco_mask_util
from scipy.optimize import linear_sum_assignment, brentq
from scipy.signal import find_peaks
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

import sys

sys.path.append("../coco_eval")
from mask_rcnn_model import run_mask_rcnn, MaskRCNNInstanceSegmentation
from SemanticSegmentation import SemanticSegmentator

# threshold to apply on masks
MASK_THRESHOLD: Final[float] = 0.55
FOV_SIZE: Final[float] = 1117.0
PIXEL_SIZE: Final[float] = 2.74
IMAGE_SIZE: Final[int] = 4512
MAGNIFICATION: float = PIXEL_SIZE * IMAGE_SIZE / FOV_SIZE


def distance_batch(in_centers1: np.ndarray, in_centers2: np.ndarray) -> np.ndarray:
    """Given N x 2 & M x 2 ndarrays of box centers, compute pairwise euclidean distances"""
    # expand dims to allow computing pairwise distance (creates NxM below)
    centers1 = np.expand_dims(in_centers1, 1)
    centers2 = np.expand_dims(in_centers2, 0)

    # for each center coordinate, compute distance in x & y directions
    x_distances = centers1[..., 0] - centers2[..., 0]  # NxM
    y_distances = centers1[..., 1] - centers2[..., 1]  # NxM
    # compute euclidean distance
    distances: np.ndarray = np.sqrt(
        x_distances * x_distances + y_distances * y_distances
    )  # NxM
    return distances


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


def pair_cages_using_distance(
    cage_center_coordinates: np.ndarray,
    reference_cage_center_coordinates: np.ndarray,
    max_allowed_distance: int = 500,
):
    # get the distance measure between centers (of the cages)
    distances: np.ndarray = distance_batch(
        in_centers1=cage_center_coordinates,
        in_centers2=reference_cage_center_coordinates,
    )

    # for finding the best matching, we use the Hungarian algorithm, unpaired1 and unpaired2 will contain
    # indexes that are not paired after the matching as below

    # Hungarian algorithm for finding the matching with the maximum total cost (maximizing the sum of the distances between paired cages)
    row_ind, col_ind = linear_sum_assignment(distances)
    # remove the assignments that are farther than max_allowed_distance
    paired_idx: List[Tuple[int, int]] = []
    unpaired1: List[int] = list(
        set([i for i in range(cage_center_coordinates.shape[0])]) - set(row_ind)
    )  # type: ignore
    unpaired2: List[int] = list(
        set([i for i in range(reference_cage_center_coordinates.shape[0])])
        - set(col_ind)
    )  # type: ignore
    for i, j in list(zip(row_ind, col_ind)):
        if distances[i, j] <= max_allowed_distance:
            paired_idx.append((i, j))
        else:
            unpaired1.append(i)
            unpaired2.append(j)

    return paired_idx, unpaired1, unpaired2


def pair_cages_using_iou(
    cage_boxes: np.ndarray,
    reference_cage_boxes: np.ndarray,
    min_iou_for_pairing: float = 0.5,
):
    # get the distance measure between centers (of the cages)
    iou_matrix: np.ndarray = iou_batch(bboxes1=cage_boxes, bboxes2=reference_cage_boxes)

    # for finding the best matching, we use the Hungarian algorithm, unpaired1 and unpaired2 will contain
    # indexes that are not paired after the matching as below

    # Hungarian algorithm for finding the matching with the maximum total cost (maximizing the sum of IoUs)
    # linear_sum_assignment minimizes the cost, so we multiply by -1
    row_ind, col_ind = linear_sum_assignment(-1 * iou_matrix)
    # remove the assignments that are farther than max_allowed_distance
    paired_idx: List[Tuple[int, int]] = []
    unpaired1: List[int] = list(
        set([i for i in range(cage_boxes.shape[0])]) - set(row_ind)
    )  # type: ignore
    unpaired2: List[int] = list(
        set([i for i in range(reference_cage_boxes.shape[0])]) - set(col_ind)
    )  # type: ignore
    for i, j in list(zip(row_ind, col_ind)):
        if iou_matrix[i, j] >= min_iou_for_pairing:
            paired_idx.append((i, j))
        else:
            unpaired1.append(i)
            unpaired2.append(j)

    return paired_idx, unpaired1, unpaired2


def get_list_of_raw_bf_images(exp_base_path: str, scan_id: int, lane_id: int):
    raw_images_path: str = os.path.join(
        exp_base_path, "scan_" + str(scan_id), "lane_" + str(lane_id)
    )
    raw_images: List[str] = os.listdir(raw_images_path)
    raw_bf_images: List[str] = [
        img_name for img_name in raw_images if "_White.png" in img_name
    ]
    return raw_bf_images


def save_masks_in_coco_rle_format(
    out: Dict[str, List],
    model_label_map: Dict[int, str],
    model_info: str,
    image_width: int,
    image_height: int,
    filename: str,
) -> None:
    """
    A function to save the masks and bounding boxes for the detected objects
    as a dictionary (pickle file) after compressing the mask using COCO RLE format

    Args:
        out: The model output in a dictionary format with the following keys and values:
            - "boxes": List of numpy arrays, or 4-element lists for each bounding box
            - "labels": List of integer labels (class IDs)
            - "scores": List of detection scores/confidence values (not used in this function)
            - "masks":  List of numpy array for object masks within the bounding boxes
        model_label_map: A dictionary with keys as class IDs and values as class names. This is
            saved to be used in case the class names are needed (only IDs are stored).
        model_info: The type of the model used to produce the results, e.g., "mask_rcnn", "point_rend"
        image_width (integer): The width of the input image to the model.
        image_height (integer): The height of the input image to the model.
        filename: the filename in string format.

    """
    record: dict = {}
    record["annotations"]: List[dict] = []
    # save the model label map
    record["model_label_map"]: Dict[int, str] = model_label_map
    # image height and width
    record["width"]: int = image_width
    record["height"]: int = image_height
    record["model_info"]: str = model_info

    for obj_id, current_mask in enumerate(out["masks"]):
        # check the area first, make sure to include only large enough objects
        # object's bounding box
        box_xtl, box_ytl, box_xbr, box_ybr = out["boxes"][obj_id]
        # object's label
        label = int(out["labels"][obj_id])

        # we do not expand the mask to the input image's resolution and
        # only save the mask within the objects bounding box to save space

        # annotations in COCO RLE format
        annots: dict = {
            "bbox": [box_xtl, box_ytl, box_xbr, box_ybr],
            "category_id": label,
            "segmentation": coco_mask_util.encode(np.asarray(current_mask, order="F")),
        }
        record["annotations"].append(annots)

    filehandler = open(filename, "wb")
    pickle.dump(record, filehandler)
    filehandler.close()


def load_masks_in_coco_rle_format(
    filename: str,
) -> (Dict[str, Dict[int, np.ndarray]], Dict[str, np.ndarray], Dict[str, List]):
    """
    A function to load the masks and bounding boxes for the detected objects from a file
    (pickle file) after the results were compressed using COCO RLE format

    Args:
        filename: the filename in string format. It should be '.pkl'. A FileNotFoundError exception
            is thrown if the file does not exist (should be handled by the caller).

    Outputs:
        The bounding boxes of the objects, which is a dictionary with keys as the string class names
            (from the model's label map passed) and values as dictionaries with keys as the object ID
            and values as the bounding box for the detected objects (np.array). The object IDs are unique for
            all the objects and are not reused between objects of different classes.
        The combined_masks, which is a dictionary with keys as the string class names
            (from the model's label map passed) and values as the instance masks of the objects of that
            class; for each class, the mask values of a given object (of that class) are the unique
            integer object ID of that object.  The object IDs are unique for all the objects and are
            not reused between objects of different classes
            NOTE: Each pixel in the mask as defined for each class below can only belong to one object
            (e.g., cell), so this mask will not clearly indicate the boundaries of overlapping objects
            (while the masks from the instance segmentation model do).
        The model output in a dictionary format with the following keys and values:
            - "boxes": List of numpy arrays, or 4-element lists for each bounding box
            - "labels": List of integer labels (class IDs)
            - "masks":  List of numpy array for object masks within the bounding boxes
        The model info string (e.g., "mask_rcnn")
    """

    # model outputs to be read from file
    boxes: List[np.ndarray] = []
    labels: List[int] = []
    masks: List[np.ndarray] = []

    # load the annotations
    filehandler = open(filename, "rb")
    annots = pickle.load(filehandler)
    filehandler.close()

    # input image dimensions for the results
    image_width: int = annots["width"]
    image_height: int = annots["height"]

    # model's label map and type
    model_label_map: Dict[int, str] = annots["model_label_map"]
    model_info: str = annots["model_info"]

    # below is the combined instance masks for each class
    # the combined_masks is a dictionary with keys as the string class names (from the model's label map)
    # and values as instance masks of the objects of that class; for each class, the mask values of a given
    # object (of that class) are the unique integer object ID of that object
    # the object IDs are unique for all the objects and are not reused between objects of different classes
    # NOTE: Each pixel in the mask as defined for each class below can only belong to one object (e.g., cell)
    # so this mask will not clearly indicate the boundaries of overlapping objects
    # (while the masks from the instance segmentation model do)
    combined_masks: Dict[str, np.ndarray] = {
        class_name: np.zeros((image_height, image_width), dtype=np.uint16)
        for class_name in model_label_map.values()
    }

    id_boxes: Dict[str, Dict[int, np.ndarray]] = {
        class_name: {} for class_name in model_label_map.values()
    }

    for obj_id, record in enumerate(annots["annotations"]):
        xtl, ytl, xbr, ybr = record["bbox"]
        boxes.append(np.array([xtl, ytl, xbr, ybr]))
        labels.append(int(record["category_id"]))
        mask_this_object: np.ndarray = coco_mask_util.decode(record["segmentation"])
        masks.append(mask_this_object)

        class_name: str = model_label_map[int(record["category_id"])]
        # threshold the mask
        mask_this_object[mask_this_object >= MASK_THRESHOLD] = 1
        mask_this_object = mask_this_object.astype(np.uint16)
        # it is possible to have the bounding box of that object that is being added now
        # (only its box and not its mask) overlap with the mask of a previously included
        # cell (already annotated in masks), in this case, the box removes the valid mask
        # of the cell (replace it with zeros) while there is no object content there
        # in the following, we prevent this from happening
        mask_copy: np.ndarray = combined_masks[class_name][ytl:ybr, xtl:xbr].copy()
        mask_copy[mask_this_object > 0] = obj_id + 1
        combined_masks[class_name][ytl:ybr, xtl:xbr] = mask_copy
        id_boxes[class_name][obj_id + 1] = np.array([xtl, ytl, xbr, ybr])

    return (
        id_boxes,
        combined_masks,
        {"boxes": boxes, "labels": labels, "masks": masks},
        model_info,
    )


def convert_to_log_scale(fl_img: np.ndarray):
    # make a copy as we modify the input array
    img: np.ndarray = fl_img.astype(float).copy()
    img[img <= 1e-30] = 1e-30
    return np.log10(img)


def get_fov_coords(filename: str):
    parsed_filename: List[str] = filename.strip().split("_")
    return (int(parsed_filename[5]), int(parsed_filename[6]))


def get_fl_channel_filenames(
    bf_img_name: str, all_img_names: List[str], fl_channel_str_identifiers: List[str]
):
    parsed_bf_img_name: List[str] = bf_img_name.strip().split("_")
    x_coor_str, y_coor_str = parsed_bf_img_name[5:7]

    # if a matching filename for the passed fluorescent channel identifier in fl_channel_str_identifiers cannot be found
    # the identifier will not be included in the output dictionary as a key
    out: Dict[str, str] = {}

    for img_name in all_img_names:
        parsed_img_name: List[str] = img_name.strip().split("_")
        channel_identifier: str = ".".join(parsed_img_name[-1].split(".")[:-1])
        if (
            parsed_img_name[5] == x_coor_str
            and parsed_img_name[6] == y_coor_str
            and channel_identifier in fl_channel_str_identifiers
        ):
            out[channel_identifier] = img_name

    return out


def load_images(
    bf_img_filename: str,
    raw_images_folder: str,
    fl_channel_identifiers: List[str],
):
    # load all the images (brightfield and fluorescent) from disk
    fl_img_filenames: Dict[str, str] = get_fl_channel_filenames(
        bf_img_name=bf_img_filename,
        all_img_names=os.listdir(raw_images_folder),
        fl_channel_str_identifiers=fl_channel_identifiers,
    )

    out_images: Dict[str, np.ndarray] = {
        "White": cv2.imread(
            os.path.join(raw_images_folder, bf_img_filename), cv2.IMREAD_UNCHANGED
        )
    }

    for channel_identifier, fl_img_name in fl_img_filenames.items():
        # load fluorescent channels
        out_images[channel_identifier]: np.ndarray = cv2.imread(
            os.path.join(raw_images_folder, fl_img_name), cv2.IMREAD_UNCHANGED
        )
        # apply a median filter to fluorescent channels to fill the holes and smooth the images
        out_images[channel_identifier] = cv2.medianBlur(
            src=out_images[channel_identifier], ksize=3
        )

    return out_images


def load_images_and_mask_rcnn_results(
    bf_img_filename: str,
    raw_images_folder: str,
    mask_rcnn_results_folder: str,
    fl_channel_identifiers: List[str],
    inst_segmentor_model: MaskRCNNInstanceSegmentation,
):

    out_images: Dict[str, np.ndarray] = load_images(
        bf_img_filename=bf_img_filename,
        raw_images_folder=raw_images_folder,
        fl_channel_identifiers=fl_channel_identifiers,
    )

    run_the_model: bool = False
    if not os.path.exists(mask_rcnn_results_folder):
        run_the_model: bool = True
        os.mkdir(mask_rcnn_results_folder)

    # load the Mask R-CNN results for each FoV from disk
    results_filename: str = ".".join(bf_img_filename.strip().split(".")[:-1]) + ".pkl"
    if (
        not os.path.exists(os.path.join(mask_rcnn_results_folder, results_filename))
        or run_the_model
    ):
        # run Mask R-CNN and save the results
        preds, run_time = run_mask_rcnn(
            input_image=out_images["White"],
            normalize_image=True,
            bit_depth=12,
            crop=True,
            classnames_mapping_dict=None,
            post_process_class_names=list(
                inst_segmentor_model.get_label_map().values()
            ),
            return_features=False,
            plot_results=False,
            detector=inst_segmentor_model,
        )

        save_masks_in_coco_rle_format(
            out=preds,
            model_label_map=inst_segmentor_model.get_label_map(),
            model_info="mask_rcnn",
            image_width=4512,
            image_height=4512,
            filename=os.path.join(mask_rcnn_results_folder, results_filename),
        )

    # and load the results to get the combined_mask
    _, combined_obj_masks, preds, _ = load_masks_in_coco_rle_format(
        filename=os.path.join(mask_rcnn_results_folder, results_filename)
    )

    return out_images, {"predictions": preds, "combined_obj_masks": combined_obj_masks}


def get_avg_fl_signal_per_cell_for_fov(
    bf_img_filename: str,
    raw_images_folder: str,
    mask_rcnn_results_folder: str,
    fl_channel_identifiers: List[str],
    inst_segmentor_model: MaskRCNNInstanceSegmentation,
    return_avg_of_log_fl_values: bool = False,
):

    # load the images and the Mask R-CNN results
    out_images, mask_rcnn_results = load_images_and_mask_rcnn_results(
        bf_img_filename=bf_img_filename,
        raw_images_folder=raw_images_folder,
        mask_rcnn_results_folder=mask_rcnn_results_folder,
        fl_channel_identifiers=fl_channel_identifiers,
        inst_segmentor_model=inst_segmentor_model,
    )

    preds: dict = mask_rcnn_results["predictions"]

    # first, we calculate the total and average flurscent signal readouts per detected cell objects from Mask R-CNN
    # this is done over all cells and not only the caged ones

    fl_channel_identifiers_list: List[str] = [
        channel_identifier
        for channel_identifier in out_images.keys()
        if channel_identifier.lower() != "white"
    ]

    total_fl_signal_per_cell: Dict[str, List[int]] = {
        channel_identifier: [] for channel_identifier in fl_channel_identifiers_list
    }
    avg_fl_signal_per_cell: Dict[str, List[float]] = {
        channel_identifier: [] for channel_identifier in fl_channel_identifiers_list
    }

    for obj_id, box in enumerate(preds["boxes"]):
        # only consider detected objects of class cell
        if (
            preds["labels"][obj_id]
            != inst_segmentor_model.get_reverse_label_map()["cell"]
        ):
            continue
        # box coordinates for a cell object
        xtl, ytl, xbr, ybr = box
        # cell area
        cell_area = float(np.sum(preds["masks"][obj_id]))
        if cell_area == 0:
            # this should never happen
            continue
        for channel_identifier in fl_channel_identifiers_list:
            if return_avg_of_log_fl_values:
                fl_img: np.ndarray = convert_to_log_scale(
                    out_images[channel_identifier][ytl:ybr, xtl:xbr]
                )
            else:
                fl_img: np.ndarray = out_images[channel_identifier][
                    ytl:ybr, xtl:xbr
                ].astype(float)

            masked_fl_img: np.ndarray = fl_img * preds["masks"][obj_id]
            # total signal
            total_fl_signal_per_cell[channel_identifier].append(np.sum(masked_fl_img))
            # average values
            avg_fl_signal_per_cell[channel_identifier].append(
                np.sum(masked_fl_img) / cell_area
            )

    if not return_avg_of_log_fl_values:
        for channel_identifier in fl_channel_identifiers_list:
            # replace the zero values with the smallest none-zero
            data = np.array(total_fl_signal_per_cell[channel_identifier])
            min_non_zero_data_value = np.min(data[data > 0])
            data[data == 0] = min_non_zero_data_value
            # logarithmic scale
            total_fl_signal_per_cell[channel_identifier] = np.log10(data).tolist()

            data = np.array(avg_fl_signal_per_cell[channel_identifier])
            min_non_zero_data_value = np.min(data[data > 0])
            data[data == 0] = min_non_zero_data_value
            # logarithmic scale
            avg_fl_signal_per_cell[channel_identifier] = np.log10(data).tolist()

    return total_fl_signal_per_cell, avg_fl_signal_per_cell


def get_avg_fl_signal_per_cell_for_scan_and_lane(
    exp_base_path: str,
    scan_id: int,
    lane_id: int,
    fl_channel_identifiers: List[str],
    inst_segmentor_model: MaskRCNNInstanceSegmentation,
    return_avg_of_log_fl_values: bool = False,
):

    # check if the results already exists
    if return_avg_of_log_fl_values:
        saved_filename_avg: str = (
            "avg_fl_signal_per_cell_scan_"
            + str(scan_id)
            + "_lane_"
            + str(lane_id)
            + "_avg_log.csv"
        )
        saved_filename_total: str = (
            "total_fl_signal_per_cell_scan_"
            + str(scan_id)
            + "_lane_"
            + str(lane_id)
            + "_avg_log.csv"
        )
    else:
        saved_filename_avg: str = (
            "avg_fl_signal_per_cell_scan_"
            + str(scan_id)
            + "_lane_"
            + str(lane_id)
            + ".csv"
        )
        saved_filename_total: str = (
            "total_fl_signal_per_cell_scan_"
            + str(scan_id)
            + "_lane_"
            + str(lane_id)
            + ".csv"
        )

    if not os.path.exists(os.path.join(exp_base_path, saved_filename_avg)):
        total_fl_signal_per_cell = {
            channel_identifier: [] for channel_identifier in fl_channel_identifiers
        }
        avg_fl_signal_per_cell = {
            channel_identifier: [] for channel_identifier in fl_channel_identifiers
        }

        raw_bf_images: List[str] = get_list_of_raw_bf_images(
            exp_base_path, scan_id, lane_id
        )
        raw_images_path: str = os.path.join(
            exp_base_path, "scan_" + str(scan_id), "lane_" + str(lane_id)
        )
        mask_rcnn_results_path: str = os.path.join(
            exp_base_path, "scan_" + str(scan_id), "lane_" + str(lane_id) + "_results"
        )

        for bf_img_name in raw_bf_images:
            total_fl_signal, avg_fl_signal = get_avg_fl_signal_per_cell_for_fov(
                bf_img_filename=bf_img_name,
                raw_images_folder=raw_images_path,
                mask_rcnn_results_folder=mask_rcnn_results_path,
                fl_channel_identifiers=fl_channel_identifiers,
                inst_segmentor_model=inst_segmentor_model,
                return_avg_of_log_fl_values=return_avg_of_log_fl_values,
            )
            for key in avg_fl_signal_per_cell:
                total_fl_signal_per_cell[key] += total_fl_signal[key]
                avg_fl_signal_per_cell[key] += avg_fl_signal[key]
        # save them to disk
        avg_fl_signal_per_cell = pd.DataFrame(avg_fl_signal_per_cell)
        avg_fl_signal_per_cell.to_csv(
            os.path.join(exp_base_path, saved_filename_avg), index=False
        )

        total_fl_signal_per_cell = pd.DataFrame(total_fl_signal_per_cell)
        total_fl_signal_per_cell.to_csv(
            os.path.join(exp_base_path, saved_filename_total), index=False
        )
    else:
        print(
            f"[INFO] Fluorescent signal stats per cell are already available for scan {scan_id}, lane {lane_id}! loading from file"
        )
        avg_fl_signal_per_cell = pd.read_csv(
            os.path.join(exp_base_path, saved_filename_avg)
        )
        total_fl_signal_per_cell = pd.read_csv(
            os.path.join(exp_base_path, saved_filename_total)
        )

    return total_fl_signal_per_cell, avg_fl_signal_per_cell


def get_fl_signal_per_cell_pixel_for_scan_and_lane(
    exp_base_path: str,
    scan_id: int,
    lane_id: int,
    fl_channel_identifiers: List[str],
    inst_segmentor_model: MaskRCNNInstanceSegmentation,
):

    # check if the results already exists
    if not os.path.exists(
        os.path.join(
            exp_base_path,
            "fl_signal_per_cell_pixel_scan_"
            + str(scan_id)
            + "_lane_"
            + str(lane_id)
            + ".csv",
        )
    ):
        # the histogram of pixel values for each fl channel is returned as a series
        fl_signal_per_cell_pixel = {
            channel_identifier: pd.Series(data=0, index=np.arange(0, 2**12))
            for channel_identifier in fl_channel_identifiers
        }

        # give a name to the index of these series
        for pd_series in fl_signal_per_cell_pixel.values():
            pd_series.index.name = "fl_pixel_value"

        raw_bf_images: List[str] = get_list_of_raw_bf_images(
            exp_base_path, scan_id, lane_id
        )
        raw_images_path: str = os.path.join(
            exp_base_path, "scan_" + str(scan_id), "lane_" + str(lane_id)
        )
        mask_rcnn_results_path: str = os.path.join(
            exp_base_path, "scan_" + str(scan_id), "lane_" + str(lane_id) + "_results"
        )

        for bf_img_filename in raw_bf_images:
            # load the images and the Mask R-CNN results
            out_images, mask_rcnn_results = load_images_and_mask_rcnn_results(
                bf_img_filename=bf_img_filename,
                raw_images_folder=raw_images_path,
                mask_rcnn_results_folder=mask_rcnn_results_path,
                fl_channel_identifiers=fl_channel_identifiers,
                inst_segmentor_model=inst_segmentor_model,
            )

            # mask of all detected cells
            combined_obj_masks: dict = mask_rcnn_results["combined_obj_masks"]
            all_cells_mask: np.ndarray = np.zeros(
                combined_obj_masks["cell"].shape, np.uint8
            )
            all_cells_mask[combined_obj_masks["cell"] > 0] = 1

            for channel_identifier in fl_channel_identifiers:
                masked_fl_img: np.ndarray = out_images[channel_identifier][
                    all_cells_mask > 0
                ]
                # convert to pandas DataFrame for each of value count calculations
                masked_fl_img_df: pd.DataFrame = pd.DataFrame(
                    data=masked_fl_img.flatten(), columns=["fl_pixel_value"]
                )
                pixel_value_counts: pd.Series = masked_fl_img_df.value_counts(
                    subset="fl_pixel_value"
                )

                fl_signal_per_cell_pixel[channel_identifier].loc[
                    pixel_value_counts.index
                ] += pixel_value_counts.values

        # combine all in a DataFrame
        fl_signal_per_cell_pixel = pd.DataFrame(fl_signal_per_cell_pixel)
        # save them to disk
        fl_signal_per_cell_pixel.to_csv(
            os.path.join(
                exp_base_path,
                "fl_signal_per_cell_pixel_scan_"
                + str(scan_id)
                + "_lane_"
                + str(lane_id)
                + ".csv",
            ),
            index=False,
        )
    else:
        print(
            f"[INFO] Fluorescent signal stats per cell are already available for scan {scan_id}, lane {lane_id}! loading from file"
        )
        fl_signal_per_cell_pixel = pd.read_csv(
            os.path.join(
                exp_base_path,
                "fl_signal_per_cell_pixel_scan_"
                + str(scan_id)
                + "_lane_"
                + str(lane_id)
                + ".csv",
            )
        )

    return fl_signal_per_cell_pixel


def get_cages_metrics_for_fov(
    bf_img_filename: str,
    raw_images_folder: str,
    mask_rcnn_results_folder: str,
    fl_channel_identifiers: List[str],
    log_gating_threshold_dict: Dict[str, float],
    fl_channels_list_for_metric_extraction: List[str],
    inst_segmentor_model: MaskRCNNInstanceSegmentation,
    sem_segmentor_model,
    return_avg_of_log_fl_values: bool = False,
    skip_mask_rcnn_metrics: bool = False,
    starting_cage_id=0,
):
    if return_avg_of_log_fl_values:
        # keep the thresholds in DB
        gating_threshold_dict: Dict[str, float] = log_gating_threshold_dict
    else:
        # convert the thresholds to linear to be applied in fl channel values
        gating_threshold_dict = {k: 10**v for k, v in log_gating_threshold_dict.items()}

    # get FOV coordinates in um for later cage matching
    fov_start_x, fov_start_y = get_fov_coords(bf_img_filename)
    # convert the FoV coordinates to pixels
    fov_start_x = int(np.round(fov_start_x * MAGNIFICATION / PIXEL_SIZE))
    fov_start_y = int(np.round(fov_start_y * MAGNIFICATION / PIXEL_SIZE))

    out_images, mask_rcnn_results = load_images_and_mask_rcnn_results(
        bf_img_filename=bf_img_filename,
        raw_images_folder=raw_images_folder,
        mask_rcnn_results_folder=mask_rcnn_results_folder,
        fl_channel_identifiers=fl_channel_identifiers,
        inst_segmentor_model=inst_segmentor_model,
    )

    # load BF, Blue and Red images
    bf_img: np.ndarray = out_images["White"]

    # and load the results to get the combined_mask
    combined_obj_masks, preds = (
        mask_rcnn_results["combined_obj_masks"],
        mask_rcnn_results["predictions"],
    )

    cage_boxes: List[np.ndarray] = [
        box
        for i, box in enumerate(preds["boxes"])
        if preds["labels"][i] == inst_segmentor_model.get_reverse_label_map()["cage"]
    ]

    cage_boxes = (
        np.array(cage_boxes) if len(cage_boxes) > 0 else np.zeros((0, 4), dtype=int)
    )
    # a mapping between the index of a cage in the cage_boxes above, and Mask R-CNN results preds
    cage_box_idx_to_mask_rcnn_results_idx_map: Dict[int, int] = {
        k: v
        for k, v in enumerate(
            [
                i
                for i, class_id in enumerate(preds["labels"])
                if class_id == inst_segmentor_model.get_reverse_label_map()["cage"]
            ]
        )
    }

    # normalization is not needed for sem_segmentor_model.predict_cages as it takes care of it inside the method
    cage_semantic_masks, _ = sem_segmentor_model.predict_cages(
        input_image=bf_img,
        cage_bounding_boxes=cage_boxes,
        percentage_to_expand_cage_box_boundaries=0.2,
    )

    # initialize the output dictionary
    # for all calls/pixels without any gating
    cage_metrics: Dict[str, dict] = {}
    cage_metrics["all"]: dict = {}
    if not skip_mask_rcnn_metrics:
        cage_metrics["all"]["num_cells"]: Dict[int, int] = {}
        cage_metrics["all"]["num_cell_pixels"]: Dict[int, int] = {}
        for metric_channel_identifier in fl_channels_list_for_metric_extraction:
            cage_metrics["all"]["total_" + metric_channel_identifier + "_cell"]: Dict[
                int, int
            ] = {}
            cage_metrics["all"]["avg_" + metric_channel_identifier + "_cell"]: Dict[
                int, float
            ] = {}
            if metric_channel_identifier in gating_threshold_dict:
                cage_metrics["all"][
                    "num_" + metric_channel_identifier + "_positive_cell"
                ]: Dict[int, float] = {}

    cage_metrics["all"]["num_pixels"]: Dict[int, int] = {}
    for metric_channel_identifier in fl_channels_list_for_metric_extraction:
        # metrics to be extracted (total and average only defined for now)
        cage_metrics["all"]["total_" + metric_channel_identifier + "_pixel"]: Dict[
            int, int
        ] = {}
        cage_metrics["all"]["avg_" + metric_channel_identifier + "_pixel"]: Dict[
            int, float
        ] = {}
        if metric_channel_identifier in gating_threshold_dict:
            cage_metrics["all"][
                "num_" + metric_channel_identifier + "_positive_pixel"
            ]: Dict[int, float] = {}

    # cage position
    cage_metrics["all"]["fov_filename"]: Dict[int, str] = {}
    cage_metrics["all"]["cage_xtl"]: Dict[int, float] = {}
    cage_metrics["all"]["cage_ytl"]: Dict[int, float] = {}
    cage_metrics["all"]["cage_xbr"]: Dict[int, float] = {}
    cage_metrics["all"]["cage_ybr"]: Dict[int, float] = {}
    cage_metrics["all"]["cage_center_x"]: Dict[int, float] = {}
    cage_metrics["all"]["cage_center_y"]: Dict[int, float] = {}

    # for positive/negative gated calls/pixels using the provided list of channels and the thresholds
    for gating_channel_identifier in gating_threshold_dict:
        cage_metrics[gating_channel_identifier + "_positive"]: dict = {}
        cage_metrics[gating_channel_identifier + "_negative"]: dict = {}

        if not skip_mask_rcnn_metrics:
            cage_metrics[gating_channel_identifier + "_positive"]["num_cells"]: Dict[
                int, int
            ] = {}
            cage_metrics[gating_channel_identifier + "_negative"]["num_cells"]: Dict[
                int, int
            ] = {}

            cage_metrics[gating_channel_identifier + "_positive"][
                "num_cell_pixels"
            ]: Dict[int, int] = {}
            cage_metrics[gating_channel_identifier + "_negative"][
                "num_cell_pixels"
            ]: Dict[int, int] = {}

            for metric_channel_identifier in fl_channels_list_for_metric_extraction:
                cage_metrics[gating_channel_identifier + "_positive"][
                    "total_" + metric_channel_identifier + "_cell"
                ]: Dict[int, int] = {}
                cage_metrics[gating_channel_identifier + "_negative"][
                    "total_" + metric_channel_identifier + "_cell"
                ]: Dict[int, int] = {}

                cage_metrics[gating_channel_identifier + "_positive"][
                    "avg_" + metric_channel_identifier + "_cell"
                ]: Dict[int, float] = {}
                cage_metrics[gating_channel_identifier + "_negative"][
                    "avg_" + metric_channel_identifier + "_cell"
                ]: Dict[int, float] = {}
                if metric_channel_identifier in gating_threshold_dict:
                    cage_metrics[gating_channel_identifier + "_positive"][
                        "num_" + metric_channel_identifier + "_positive_cell"
                    ]: Dict[int, float] = {}
                    cage_metrics[gating_channel_identifier + "_negative"][
                        "num_" + metric_channel_identifier + "_positive_cell"
                    ]: Dict[int, float] = {}

        cage_metrics[gating_channel_identifier + "_positive"]["num_pixels"]: Dict[
            int, int
        ] = {}
        cage_metrics[gating_channel_identifier + "_negative"]["num_pixels"]: Dict[
            int, int
        ] = {}

        for metric_channel_identifier in fl_channels_list_for_metric_extraction:
            cage_metrics[gating_channel_identifier + "_positive"][
                "total_" + metric_channel_identifier + "_pixel"
            ]: Dict[int, int] = {}
            cage_metrics[gating_channel_identifier + "_negative"][
                "total_" + metric_channel_identifier + "_pixel"
            ]: Dict[int, int] = {}

            cage_metrics[gating_channel_identifier + "_positive"][
                "avg_" + metric_channel_identifier + "_pixel"
            ]: Dict[int, float] = {}
            cage_metrics[gating_channel_identifier + "_negative"][
                "avg_" + metric_channel_identifier + "_pixel"
            ]: Dict[int, float] = {}

            if metric_channel_identifier in gating_threshold_dict:
                cage_metrics[gating_channel_identifier + "_positive"][
                    "num_" + metric_channel_identifier + "_positive_pixel"
                ]: Dict[int, float] = {}
                cage_metrics[gating_channel_identifier + "_negative"][
                    "num_" + metric_channel_identifier + "_positive_pixel"
                ]: Dict[int, float] = {}

        # cage position
        cage_metrics[gating_channel_identifier + "_positive"]["fov_filename"]: Dict[
            int, str
        ] = {}
        cage_metrics[gating_channel_identifier + "_positive"]["cage_xtl"]: Dict[
            int, float
        ] = {}
        cage_metrics[gating_channel_identifier + "_positive"]["cage_ytl"]: Dict[
            int, float
        ] = {}
        cage_metrics[gating_channel_identifier + "_positive"]["cage_xbr"]: Dict[
            int, float
        ] = {}
        cage_metrics[gating_channel_identifier + "_positive"]["cage_ybr"]: Dict[
            int, float
        ] = {}
        cage_metrics[gating_channel_identifier + "_positive"]["cage_center_x"]: Dict[
            int, float
        ] = {}
        cage_metrics[gating_channel_identifier + "_positive"]["cage_center_y"]: Dict[
            int, float
        ] = {}

        cage_metrics[gating_channel_identifier + "_negative"]["fov_filename"]: Dict[
            int, str
        ] = {}
        cage_metrics[gating_channel_identifier + "_negative"]["cage_xtl"]: Dict[
            int, float
        ] = {}
        cage_metrics[gating_channel_identifier + "_negative"]["cage_ytl"]: Dict[
            int, float
        ] = {}
        cage_metrics[gating_channel_identifier + "_negative"]["cage_xbr"]: Dict[
            int, float
        ] = {}
        cage_metrics[gating_channel_identifier + "_negative"]["cage_ybr"]: Dict[
            int, float
        ] = {}
        cage_metrics[gating_channel_identifier + "_negative"]["cage_center_x"]: Dict[
            int, float
        ] = {}
        cage_metrics[gating_channel_identifier + "_negative"]["cage_center_y"]: Dict[
            int, float
        ] = {}

    # go over all cages
    for i, cage_semantic_mask in enumerate(cage_semantic_masks):
        # cage_semantic_masks[i] is corresponding to cage_boxes[i], they are the same size and from the same area of the image
        xtl, ytl, xbr, ybr = cage_boxes[i]
        cage_mask_from_mask_rcnn: np.ndarray = preds["masks"][
            cage_box_idx_to_mask_rcnn_results_idx_map[i]
        ]

        # number of cell objects detected inside this cage from Mask R-CNN
        caged_cell_instances_mask: np.ndarray = combined_obj_masks["cell"][
            ytl:ybr, xtl:xbr
        ] * cage_mask_from_mask_rcnn.astype(int)
        cell_ids_in_cage: np.ndarray = np.unique(
            caged_cell_instances_mask[caged_cell_instances_mask > 0]
        )

        # store the cage location
        cage_metrics["all"]["fov_filename"][i + starting_cage_id] = bf_img_filename
        cage_metrics["all"]["cage_xtl"][i + starting_cage_id] = xtl + fov_start_x
        cage_metrics["all"]["cage_ytl"][i + starting_cage_id] = ytl + fov_start_y
        cage_metrics["all"]["cage_xbr"][i + starting_cage_id] = xbr + fov_start_x
        cage_metrics["all"]["cage_ybr"][i + starting_cage_id] = ybr + fov_start_y
        cage_metrics["all"]["cage_center_x"][i + starting_cage_id] = (
            xtl + xbr
        ) / 2.0 + fov_start_x
        cage_metrics["all"]["cage_center_y"][i + starting_cage_id] = (
            ytl + ybr
        ) / 2.0 + fov_start_y

        for gating_channel_identifier in gating_threshold_dict.keys():
            cage_metrics[gating_channel_identifier + "_positive"]["fov_filename"][
                i + starting_cage_id
            ] = bf_img_filename
            cage_metrics[gating_channel_identifier + "_positive"]["cage_xtl"][
                i + starting_cage_id
            ] = xtl + fov_start_x
            cage_metrics[gating_channel_identifier + "_positive"]["cage_ytl"][
                i + starting_cage_id
            ] = ytl + fov_start_y
            cage_metrics[gating_channel_identifier + "_positive"]["cage_xbr"][
                i + starting_cage_id
            ] = xbr + fov_start_x
            cage_metrics[gating_channel_identifier + "_positive"]["cage_ybr"][
                i + starting_cage_id
            ] = ybr + fov_start_y
            cage_metrics[gating_channel_identifier + "_positive"]["cage_center_x"][
                i + starting_cage_id
            ] = (xtl + xbr) / 2.0 + fov_start_x
            cage_metrics[gating_channel_identifier + "_positive"]["cage_center_y"][
                i + starting_cage_id
            ] = (ytl + ybr) / 2.0 + fov_start_y

            cage_metrics[gating_channel_identifier + "_negative"]["fov_filename"][
                i + starting_cage_id
            ] = bf_img_filename
            cage_metrics[gating_channel_identifier + "_negative"]["cage_xtl"][
                i + starting_cage_id
            ] = xtl + fov_start_x
            cage_metrics[gating_channel_identifier + "_negative"]["cage_ytl"][
                i + starting_cage_id
            ] = ytl + fov_start_y
            cage_metrics[gating_channel_identifier + "_negative"]["cage_xbr"][
                i + starting_cage_id
            ] = xbr + fov_start_x
            cage_metrics[gating_channel_identifier + "_negative"]["cage_ybr"][
                i + starting_cage_id
            ] = ybr + fov_start_y
            cage_metrics[gating_channel_identifier + "_negative"]["cage_center_x"][
                i + starting_cage_id
            ] = (xtl + xbr) / 2.0 + fov_start_x
            cage_metrics[gating_channel_identifier + "_negative"]["cage_center_y"][
                i + starting_cage_id
            ] = (ytl + ybr) / 2.0 + fov_start_y

        # generate metrics (total and average) using the provided list of FL channels, and if needed, gated cells/pixels
        for metric_channel_identifier in fl_channels_list_for_metric_extraction:
            # cropped metric mask
            if return_avg_of_log_fl_values:
                metric_mask: np.ndarray = convert_to_log_scale(
                    fl_img=out_images[metric_channel_identifier][ytl:ybr, xtl:xbr]
                )
            else:
                metric_mask: np.ndarray = out_images[metric_channel_identifier][
                    ytl:ybr, xtl:xbr
                ].astype(float)

            # the mask of the metric that is more than the threshold
            if metric_channel_identifier in gating_threshold_dict:
                metric_positive_mask: np.ndarray = np.zeros(metric_mask.shape, np.uint8)
                metric_positive_mask[
                    metric_mask >= gating_threshold_dict[metric_channel_identifier]
                ] = 1

            # in the following, we compute the total number of cells, the gated cells and the number of pixels
            # belonging to each category using Mask R-CNN results and the passed threshold
            if not skip_mask_rcnn_metrics:
                # calculate the mask for all the objects, we do it this way to avoid double counting pixel in overlapping cell areas
                foreground_cells_mask_from_mask_rcnn: np.ndarray = np.zeros(
                    caged_cell_instances_mask.shape, np.uint8
                )

                for cell_id in cell_ids_in_cage:
                    # this_cell_mask can be partially or completely outside the cage, we do not include cage_mask_from_mask_rcnn
                    # yet because first we need to classify the cell as target/effector
                    this_cell_mask: np.ndarray = np.zeros(
                        caged_cell_instances_mask.shape, np.uint8
                    )
                    this_cell_mask[caged_cell_instances_mask == cell_id] = 1
                    # take the union, we multiply by cage_mask_from_mask_rcnn to only consider parts inside the cage
                    # we do not skip cells that are partially (even very small portion) inside the cage
                    foreground_cells_mask_from_mask_rcnn = np.maximum(
                        foreground_cells_mask_from_mask_rcnn,
                        this_cell_mask * cage_mask_from_mask_rcnn,
                    )

                cage_metrics["all"]["num_cells"][i + starting_cage_id] = len(
                    cell_ids_in_cage
                )
                cage_metrics["all"]["num_cell_pixels"][i + starting_cage_id] = int(
                    np.sum(foreground_cells_mask_from_mask_rcnn)
                )

                cage_metrics["all"]["total_" + metric_channel_identifier + "_cell"][
                    i + starting_cage_id
                ] = np.sum(metric_mask * foreground_cells_mask_from_mask_rcnn)
                cage_metrics["all"]["avg_" + metric_channel_identifier + "_cell"][
                    i + starting_cage_id
                ] = (
                    np.sum(metric_mask * foreground_cells_mask_from_mask_rcnn)
                    / float(np.sum(foreground_cells_mask_from_mask_rcnn))
                    if np.sum(foreground_cells_mask_from_mask_rcnn) > 0
                    else np.nan
                )

                if not return_avg_of_log_fl_values:
                    temp = cage_metrics["all"][
                        "total_" + metric_channel_identifier + "_cell"
                    ][i + starting_cage_id]
                    temp = np.log10(temp) if temp > 0 else np.nan
                    cage_metrics["all"]["total_" + metric_channel_identifier + "_cell"][
                        i + starting_cage_id
                    ] = temp

                    temp = cage_metrics["all"][
                        "avg_" + metric_channel_identifier + "_cell"
                    ][i + starting_cage_id]
                    temp = np.log10(temp) if temp > 0 else np.nan
                    cage_metrics["all"]["avg_" + metric_channel_identifier + "_cell"][
                        i + starting_cage_id
                    ] = temp

                if metric_channel_identifier in gating_threshold_dict:
                    cage_metrics["all"][
                        "num_" + metric_channel_identifier + "_positive_cell"
                    ][i + starting_cage_id] = (
                        # the result will be int any way and casting is not needed
                        int(
                            np.sum(
                                metric_positive_mask
                                * foreground_cells_mask_from_mask_rcnn
                            )
                        )
                    )

                # now compute the metrics after gating cells using Mask R-CNN results
                for (
                    gating_channel_identifier,
                    threshold,
                ) in gating_threshold_dict.items():
                    # mask for gating the cells or pixels
                    if return_avg_of_log_fl_values:
                        gating_mask: np.ndarray = convert_to_log_scale(
                            fl_img=out_images[gating_channel_identifier][
                                ytl:ybr, xtl:xbr
                            ]
                        )
                    else:
                        gating_mask: np.ndarray = out_images[gating_channel_identifier][
                            ytl:ybr, xtl:xbr
                        ].astype(float)

                    num_positive_cells: int = 0
                    num_negative_cells: int = 0

                    positive_cells_mask_from_mask_rcnn: np.ndarray = np.zeros(
                        caged_cell_instances_mask.shape, np.uint8
                    )
                    negative_cells_mask_from_mask_rcnn: np.ndarray = np.zeros(
                        caged_cell_instances_mask.shape, np.uint8
                    )

                    for cell_id in cell_ids_in_cage:
                        # this_cell_mask can be partially or completely outside the cage, we do not include cage_mask_from_mask_rcnn
                        # yet because first we need to classify the cell as target/effector
                        this_cell_mask: np.ndarray = np.zeros(
                            caged_cell_instances_mask.shape, np.uint8
                        )
                        this_cell_mask[caged_cell_instances_mask == cell_id] = 1
                        # take the union, we multiply by cage_mask_from_mask_rcnn to only consider parts inside the cage
                        # we do not skip cells that are partially (even very small portion) inside the cage
                        if np.sum(gating_mask * this_cell_mask) >= threshold * np.sum(
                            this_cell_mask
                        ):
                            num_positive_cells += 1
                            positive_cells_mask_from_mask_rcnn = np.maximum(
                                positive_cells_mask_from_mask_rcnn,
                                this_cell_mask * cage_mask_from_mask_rcnn,
                            )
                        else:
                            num_negative_cells += 1
                            negative_cells_mask_from_mask_rcnn = np.maximum(
                                negative_cells_mask_from_mask_rcnn,
                                this_cell_mask * cage_mask_from_mask_rcnn,
                            )

                    cage_metrics[gating_channel_identifier + "_positive"]["num_cells"][
                        i + starting_cage_id
                    ] = num_positive_cells
                    cage_metrics[gating_channel_identifier + "_negative"]["num_cells"][
                        i + starting_cage_id
                    ] = num_negative_cells

                    cage_metrics[gating_channel_identifier + "_positive"][
                        "num_cell_pixels"
                    ][i + starting_cage_id] = int(
                        np.sum(positive_cells_mask_from_mask_rcnn)
                    )
                    cage_metrics[gating_channel_identifier + "_negative"][
                        "num_cell_pixels"
                    ][i + starting_cage_id] = int(
                        np.sum(negative_cells_mask_from_mask_rcnn)
                    )

                    cage_metrics[gating_channel_identifier + "_positive"][
                        "total_" + metric_channel_identifier + "_cell"
                    ][i + starting_cage_id] = np.sum(
                        metric_mask * positive_cells_mask_from_mask_rcnn
                    )
                    cage_metrics[gating_channel_identifier + "_negative"][
                        "total_" + metric_channel_identifier + "_cell"
                    ][i + starting_cage_id] = np.sum(
                        metric_mask * negative_cells_mask_from_mask_rcnn
                    )

                    cage_metrics[gating_channel_identifier + "_positive"][
                        "avg_" + metric_channel_identifier + "_cell"
                    ][i + starting_cage_id] = (
                        np.sum(metric_mask * positive_cells_mask_from_mask_rcnn)
                        / float(np.sum(positive_cells_mask_from_mask_rcnn))
                        if np.sum(positive_cells_mask_from_mask_rcnn) > 0
                        else np.nan
                    )
                    cage_metrics[gating_channel_identifier + "_negative"][
                        "avg_" + metric_channel_identifier + "_cell"
                    ][i + starting_cage_id] = (
                        np.sum(metric_mask * negative_cells_mask_from_mask_rcnn)
                        / float(np.sum(negative_cells_mask_from_mask_rcnn))
                        if np.sum(negative_cells_mask_from_mask_rcnn) > 0
                        else np.nan
                    )

                    if not return_avg_of_log_fl_values:
                        temp = cage_metrics[gating_channel_identifier + "_positive"][
                            "total_" + metric_channel_identifier + "_cell"
                        ][i + starting_cage_id]
                        temp = np.log10(temp) if temp > 0 else np.nan
                        cage_metrics[gating_channel_identifier + "_positive"][
                            "total_" + metric_channel_identifier + "_cell"
                        ][i + starting_cage_id] = temp

                        temp = cage_metrics[gating_channel_identifier + "_negative"][
                            "total_" + metric_channel_identifier + "_cell"
                        ][i + starting_cage_id]
                        temp = np.log10(temp) if temp > 0 else np.nan
                        cage_metrics[gating_channel_identifier + "_negative"][
                            "total_" + metric_channel_identifier + "_cell"
                        ][i + starting_cage_id] = temp

                        temp = cage_metrics[gating_channel_identifier + "_positive"][
                            "avg_" + metric_channel_identifier + "_cell"
                        ][i + starting_cage_id]
                        temp = np.log10(temp) if temp > 0 else np.nan
                        cage_metrics[gating_channel_identifier + "_positive"][
                            "avg_" + metric_channel_identifier + "_cell"
                        ][i + starting_cage_id] = temp

                        temp = cage_metrics[gating_channel_identifier + "_negative"][
                            "avg_" + metric_channel_identifier + "_cell"
                        ][i + starting_cage_id]
                        temp = np.log10(temp) if temp > 0 else np.nan
                        cage_metrics[gating_channel_identifier + "_negative"][
                            "avg_" + metric_channel_identifier + "_cell"
                        ][i + starting_cage_id] = temp

                    if metric_channel_identifier in gating_threshold_dict:
                        cage_metrics[gating_channel_identifier + "_positive"][
                            "num_" + metric_channel_identifier + "_positive_cell"
                        ][i + starting_cage_id] = int(
                            np.sum(
                                metric_positive_mask
                                * positive_cells_mask_from_mask_rcnn
                            )
                        )
                        cage_metrics[gating_channel_identifier + "_negative"][
                            "num_" + metric_channel_identifier + "_positive_cell"
                        ][i + starting_cage_id] = int(
                            np.sum(
                                metric_positive_mask
                                * negative_cells_mask_from_mask_rcnn
                            )
                        )

            # in the following, we use the extracted semantic mask for cells inside the cage to calculate
            # the number of pixels belonging to cells, target and effector cells

            # mask the area with the foreground cell mask (cage_mask includes cages, cells and beads semantic segments)
            foreground_cells_mask = np.zeros(cage_semantic_mask.shape, np.uint8)
            # beads overlapping will cells (if detected properly) will be excluded below
            foreground_cells_mask[
                cage_semantic_mask
                == sem_segmentor_model.get_reverse_label_map()["cell"]
            ] = 1
            # only consider the semantic masks for cell inside the cage mask from Mask R-CNN
            foreground_cells_mask = foreground_cells_mask * cage_mask_from_mask_rcnn

            cage_metrics["all"]["num_pixels"][i + starting_cage_id] = int(
                np.sum(foreground_cells_mask)
            )

            cage_metrics["all"]["total_" + metric_channel_identifier + "_pixel"][
                i + starting_cage_id
            ] = np.sum(metric_mask * foreground_cells_mask)
            cage_metrics["all"]["avg_" + metric_channel_identifier + "_pixel"][
                i + starting_cage_id
            ] = (
                np.sum(metric_mask * foreground_cells_mask)
                / float(np.sum(foreground_cells_mask))
                if np.sum(foreground_cells_mask) > 0
                else np.nan
            )

            if not return_avg_of_log_fl_values:
                temp = cage_metrics["all"][
                    "total_" + metric_channel_identifier + "_pixel"
                ][i + starting_cage_id]
                temp = np.log10(temp) if temp > 0 else np.nan
                cage_metrics["all"]["total_" + metric_channel_identifier + "_pixel"][
                    i + starting_cage_id
                ] = temp

                temp = cage_metrics["all"][
                    "avg_" + metric_channel_identifier + "_pixel"
                ][i + starting_cage_id]
                temp = np.log10(temp) if temp > 0 else np.nan
                cage_metrics["all"]["avg_" + metric_channel_identifier + "_pixel"][
                    i + starting_cage_id
                ] = temp

            if metric_channel_identifier in gating_threshold_dict:
                cage_metrics["all"][
                    "num_" + metric_channel_identifier + "_positive_pixel"
                ][i + starting_cage_id] = int(
                    np.sum(metric_positive_mask * foreground_cells_mask)
                )

            for gating_channel_identifier, threshold in gating_threshold_dict.items():
                # mask for gating the cells or pixels
                if return_avg_of_log_fl_values:
                    gating_mask: np.ndarray = convert_to_log_scale(
                        fl_img=out_images[gating_channel_identifier][ytl:ybr, xtl:xbr]
                    )
                else:
                    gating_mask: np.ndarray = out_images[gating_channel_identifier][
                        ytl:ybr, xtl:xbr
                    ].astype(float)

                # create a mask for pixels corresponding to target cells
                positive_pixels_mask: np.ndarray = np.zeros(
                    cage_semantic_mask.shape, np.uint8
                )
                positive_pixels_mask[
                    gating_mask * foreground_cells_mask >= threshold
                ] = 1

                # do the same for effector cells (to find dead effectors)
                negative_pixels_mask: np.ndarray = np.zeros(
                    cage_semantic_mask.shape, np.uint8
                )
                negative_pixels_mask[gating_mask < threshold] = 1
                # this step is needed to make sure 0/small blue value non-cell pixels are not classified as effector cell pixels
                negative_pixels_mask = negative_pixels_mask * foreground_cells_mask

                cage_metrics[gating_channel_identifier + "_positive"]["num_pixels"][
                    i + starting_cage_id
                ] = int(np.sum(positive_pixels_mask))
                cage_metrics[gating_channel_identifier + "_negative"]["num_pixels"][
                    i + starting_cage_id
                ] = int(np.sum(negative_pixels_mask))

                cage_metrics[gating_channel_identifier + "_positive"][
                    "total_" + metric_channel_identifier + "_pixel"
                ][i + starting_cage_id] = np.sum(metric_mask * positive_pixels_mask)
                cage_metrics[gating_channel_identifier + "_negative"][
                    "total_" + metric_channel_identifier + "_pixel"
                ][i + starting_cage_id] = np.sum(metric_mask * negative_pixels_mask)

                cage_metrics[gating_channel_identifier + "_positive"][
                    "avg_" + metric_channel_identifier + "_pixel"
                ][i + starting_cage_id] = (
                    np.sum(metric_mask * positive_pixels_mask)
                    / float(np.sum(positive_pixels_mask))
                    if np.sum(positive_pixels_mask) > 0
                    else np.nan
                )
                cage_metrics[gating_channel_identifier + "_negative"][
                    "avg_" + metric_channel_identifier + "_pixel"
                ][i + starting_cage_id] = (
                    np.sum(metric_mask * negative_pixels_mask)
                    / float(np.sum(negative_pixels_mask))
                    if np.sum(negative_pixels_mask) > 0
                    else np.nan
                )

                if not return_avg_of_log_fl_values:
                    temp = cage_metrics[gating_channel_identifier + "_positive"][
                        "total_" + metric_channel_identifier + "_pixel"
                    ][i + starting_cage_id]
                    temp = np.log10(temp) if temp > 0 else np.nan
                    cage_metrics[gating_channel_identifier + "_positive"][
                        "total_" + metric_channel_identifier + "_pixel"
                    ][i + starting_cage_id] = temp

                    temp = cage_metrics[gating_channel_identifier + "_negative"][
                        "total_" + metric_channel_identifier + "_pixel"
                    ][i + starting_cage_id]
                    temp = np.log10(temp) if temp > 0 else np.nan
                    cage_metrics[gating_channel_identifier + "_negative"][
                        "total_" + metric_channel_identifier + "_pixel"
                    ][i + starting_cage_id] = temp

                    temp = cage_metrics[gating_channel_identifier + "_positive"][
                        "avg_" + metric_channel_identifier + "_pixel"
                    ][i + starting_cage_id]
                    temp = np.log10(temp) if temp > 0 else np.nan
                    cage_metrics[gating_channel_identifier + "_positive"][
                        "avg_" + metric_channel_identifier + "_pixel"
                    ][i + starting_cage_id] = temp

                    temp = cage_metrics[gating_channel_identifier + "_negative"][
                        "avg_" + metric_channel_identifier + "_pixel"
                    ][i + starting_cage_id]
                    temp = np.log10(temp) if temp > 0 else np.nan
                    cage_metrics[gating_channel_identifier + "_negative"][
                        "avg_" + metric_channel_identifier + "_pixel"
                    ][i + starting_cage_id] = temp

                if metric_channel_identifier in gating_threshold_dict:
                    cage_metrics[gating_channel_identifier + "_positive"][
                        "num_" + metric_channel_identifier + "_positive_pixel"
                    ][i + starting_cage_id] = int(
                        np.sum(metric_positive_mask * positive_pixels_mask)
                    )

                    cage_metrics[gating_channel_identifier + "_negative"][
                        "num_" + metric_channel_identifier + "_positive_pixel"
                    ][i + starting_cage_id] = int(
                        np.sum(metric_positive_mask * negative_pixels_mask)
                    )

    out: Dict[str, pd.DataFrame] = {}
    out["all"] = pd.DataFrame(cage_metrics["all"])
    for gating_channel_identifier in gating_threshold_dict:
        out[gating_channel_identifier + "_positive"] = pd.DataFrame(
            cage_metrics[gating_channel_identifier + "_positive"]
        )
        out[gating_channel_identifier + "_negative"] = pd.DataFrame(
            cage_metrics[gating_channel_identifier + "_negative"]
        )

    return out


def get_cages_metrics_for_scan_and_lane(
    exp_base_path: str,
    scan_id: int,
    lane_id: int,
    fl_channel_identifiers: List[str],
    log_gating_threshold_dict: Dict[str, float],
    fl_channels_list_for_metric_extraction: List[str],
    inst_segmentor_model: MaskRCNNInstanceSegmentation,
    sem_segmentor_model,
    return_avg_of_log_fl_values: bool = False,
    skip_mask_rcnn_metrics: bool = False,
):
    cages_metrics_dict: Dict[str, pd.DataFrame] = {}

    run_the_lane: bool = False

    positive_filenames: Dict[str, str] = {}
    negative_filenames: Dict[str, str] = {}

    base_filename: str = (
        "cage_metrics_scan_" + str(scan_id) + "_lane_" + str(lane_id) + "_"
    )
    all_filename: str = os.path.join(exp_base_path, base_filename + "all.csv")
    if return_avg_of_log_fl_values:
        all_filename: str = os.path.join(
            exp_base_path, base_filename + "all_avg_log.csv"
        )

    # check if the results already exists
    if not os.path.exists(all_filename):
        run_the_lane: bool = True

    for gating_id, threshold in log_gating_threshold_dict.items():
        # check if the metrics have already been calculated for the scan/lane/threshold
        thr_str: str = str(np.round(10**threshold, 2)).replace(".", "p")

        if return_avg_of_log_fl_values:
            thr_str = thr_str + "_avg_log"

        positive_filenames[gating_id]: str = os.path.join(
            exp_base_path, base_filename + gating_id + "_positive_" + thr_str + ".csv"
        )
        negative_filenames[gating_id]: str = os.path.join(
            exp_base_path, base_filename + gating_id + "_negative_" + thr_str + ".csv"
        )

        if not os.path.exists(positive_filenames[gating_id]):
            run_the_lane: bool = True

        if not os.path.exists(negative_filenames[gating_id]):
            run_the_lane: bool = True

    if run_the_lane:
        raw_bf_images: List[str] = get_list_of_raw_bf_images(
            exp_base_path, scan_id, lane_id
        )
        raw_images_path: str = os.path.join(
            exp_base_path, "scan_" + str(scan_id), "lane_" + str(lane_id)
        )
        mask_rcnn_results_path: str = os.path.join(
            exp_base_path, "scan_" + str(scan_id), "lane_" + str(lane_id) + "_results"
        )

        cages_metrics_dict["all"] = pd.DataFrame()
        for gating_channel_identifier in log_gating_threshold_dict.keys():
            cages_metrics_dict[gating_channel_identifier + "_positive"] = pd.DataFrame()
            cages_metrics_dict[gating_channel_identifier + "_negative"] = pd.DataFrame()

        num_cages_so_far: int = 0

        for bf_img_name in raw_bf_images:
            cages_metrics_fov = get_cages_metrics_for_fov(
                bf_img_filename=bf_img_name,
                raw_images_folder=raw_images_path,
                mask_rcnn_results_folder=mask_rcnn_results_path,
                fl_channel_identifiers=fl_channel_identifiers,
                log_gating_threshold_dict=log_gating_threshold_dict,
                fl_channels_list_for_metric_extraction=fl_channels_list_for_metric_extraction,
                inst_segmentor_model=inst_segmentor_model,
                sem_segmentor_model=sem_segmentor_model,
                return_avg_of_log_fl_values=return_avg_of_log_fl_values,
                skip_mask_rcnn_metrics=skip_mask_rcnn_metrics,
                starting_cage_id=num_cages_so_far,
            )
            cages_metrics_dict["all"] = pd.concat(
                [cages_metrics_dict["all"], cages_metrics_fov["all"]]
            )

            # the number of cages is the same over all metrics dataframes (all, gated positive or negative)
            num_cages_so_far: int = len(cages_metrics_dict["all"])

            for gating_id in log_gating_threshold_dict:
                cages_metrics_dict[gating_id + "_positive"] = pd.concat(
                    [
                        cages_metrics_dict[gating_id + "_positive"],
                        cages_metrics_fov[gating_id + "_positive"],
                    ]
                )
                cages_metrics_dict[gating_id + "_negative"] = pd.concat(
                    [
                        cages_metrics_dict[gating_id + "_negative"],
                        cages_metrics_fov[gating_id + "_negative"],
                    ]
                )

        # save the results to disk
        cages_metrics_dict["all"].to_csv(all_filename, index=False)
        for gating_id in log_gating_threshold_dict:
            cages_metrics_dict[gating_id + "_positive"].to_csv(
                positive_filenames[gating_id], index=False
            )
            cages_metrics_dict[gating_id + "_negative"].to_csv(
                negative_filenames[gating_id], index=False
            )
    else:
        print(
            f"[INFO] Cage metrics are already available for scan {scan_id}, lane {lane_id}! loading from file"
        )

        cages_metrics_dict["all"] = pd.read_csv(all_filename)
        for gating_id in log_gating_threshold_dict:
            cages_metrics_dict[gating_id + "_positive"] = pd.read_csv(
                positive_filenames[gating_id]
            )
            cages_metrics_dict[gating_id + "_negative"] = pd.read_csv(
                negative_filenames[gating_id]
            )

    return cages_metrics_dict


def prepare_data_for_scan(
    exp_base_path: str,
    scan_id: int,
    lane_ids_list: List[int],
    fl_channels_list: List[str],
    fl_channels_list_for_gating: List[str],
    fl_channels_list_for_metric_extraction: List[str],
    control_lane_to_experiment_lane_map: Dict[int, int],
    inst_segmentor_model: MaskRCNNInstanceSegmentation,
    sem_segmentor_model: SemanticSegmentator,
    return_avg_of_log_fl_values: bool = False,
    log_gating_threshold_overwrite_dict: Dict[str, float] = {},
):

    # first, go over all the lanes to run Mask R-CNN to get the results (needed for the analysis, mainly the number of cells
    # in the cage initially, and some metrics based on Mask R-CNN)
    avg_fl_signal_per_cell: Dict[int, Dict[str, np.ndarray]] = {}

    for lane_id in lane_ids_list:
        # Mask R-CNN is run (if the results not already available) as part of the
        # function below, so we shoule not skip this for contorl lanes
        _, avg_fl_signal_per_cell[lane_id] = (
            get_avg_fl_signal_per_cell_for_scan_and_lane(
                exp_base_path=exp_base_path,
                scan_id=scan_id,
                lane_id=lane_id,
                fl_channel_identifiers=fl_channels_list,
                inst_segmentor_model=inst_segmentor_model,
                return_avg_of_log_fl_values=return_avg_of_log_fl_values,
            )
        )

    # now extract the threshold for gating using the experiment lanes only
    # the same thresholds will be used for control lanes mapped to the experiment lanes
    log_gating_threshold_dicts_for_exp_lanes: Dict[int, Dict[str, float]] = {}
    # go over all experiment lanes
    for exp_lane_id in np.unique(list(control_lane_to_experiment_lane_map.values())):
        # this is an experiment channel, use it
        log_gating_threshold_dict: Dict[str, float] = {}
        for gating_channel_identifier in fl_channels_list_for_gating:
            if gating_channel_identifier in log_gating_threshold_overwrite_dict:
                log_threshold = log_gating_threshold_overwrite_dict[
                    gating_channel_identifier
                ]
            else:
                data = avg_fl_signal_per_cell[exp_lane_id][
                    gating_channel_identifier
                ].values
                # drop the lowest 5th percentile
                target_percentile = np.percentile(data, 5)
                target_percentile = min(target_percentile, -5)
                data[data <= target_percentile] = target_percentile
                avg_fl_signal_per_cell[exp_lane_id][gating_channel_identifier] = data
                log_threshold = gmm_fit_2(
                    avg_fl_signal_per_cell[exp_lane_id], gating_channel_identifier
                )
                # the return threshold is in logarithmic scale
            log_gating_threshold_dict[gating_channel_identifier] = log_threshold
        # save the thresholds for the experiment lane
        log_gating_threshold_dicts_for_exp_lanes[exp_lane_id] = (
            log_gating_threshold_dict
        )

    cages_metrics_dict: Dict[int, Dict[str, pd.DataFrame]] = {}
    # just for returning the thresholds used for each lane and FL channel
    fl_thresholds: Dict[int, Dict[str, float]] = {}
    # now extract all the metrics
    for lane_id in lane_ids_list:
        if lane_id in control_lane_to_experiment_lane_map:
            # this is a control lane, get the associated experiment lane
            exp_lane = control_lane_to_experiment_lane_map[lane_id]
        else:
            # this is an experiment lane
            exp_lane = lane_id

        # make sure the experiment lane is already covered in control_lane_to_experiment_lane_map:
        if exp_lane not in control_lane_to_experiment_lane_map.values():
            # skip the lanes not specified as either an experiment or a control
            continue

        # get the threshold for gating
        log_gating_threshold_dict = log_gating_threshold_dicts_for_exp_lanes[exp_lane]

        # just for returning the thresholds used for each lane and FL channel
        fl_thresholds[lane_id] = log_gating_threshold_dict.copy()

        cages_metrics_dict[lane_id] = get_cages_metrics_for_scan_and_lane(
            exp_base_path=exp_base_path,
            scan_id=scan_id,
            lane_id=lane_id,
            fl_channel_identifiers=fl_channels_list,
            log_gating_threshold_dict=log_gating_threshold_dict,
            fl_channels_list_for_metric_extraction=fl_channels_list_for_metric_extraction,
            inst_segmentor_model=inst_segmentor_model,
            sem_segmentor_model=sem_segmentor_model,
            return_avg_of_log_fl_values=return_avg_of_log_fl_values,
            skip_mask_rcnn_metrics=False,
        )

    return cages_metrics_dict, fl_thresholds


def get_cage_crop(
    merged_cages_metrics: pd.DataFrame,
    exp_base_path: str,
    ref_scan_id: int,
    scan_id: int,
    lane_id: int,
    cage_id: int,
    fl_channel_identifiers: List[str],
    thresholds_for_pixel_classification: Dict[int, Dict[int, Dict[str, float]]],
    inst_segmentor_model: MaskRCNNInstanceSegmentation,
    sem_segmentor_model: SemanticSegmentator,
    colors: Dict[str, np.ndarray],
):

    # the key 'all' always exists in the merged_cages_metrics, and in general, the coordinates are all the same in other gating dataframes
    cage_location_info: pd.DataFrame = merged_cages_metrics["all"].loc[
        (merged_cages_metrics["all"]["cage_id"] == cage_id)
        & (merged_cages_metrics["all"]["scan_id"] == scan_id),
        [
            "cage_xtl",
            "cage_ytl",
            "cage_xbr",
            "cage_ybr",
            "fov_filename",
            "fov_filename_ref",
        ],
    ]

    if len(cage_location_info) == 0:
        print(
            f"[ERROR] Cage ID {cage_id} does not exists in the passed merged_cages_metrics! Returning empty ..."
        )
        return np.array([], np.uint8), np.array([], np.uint8), np.array([], np.uint8)

    bf_img_name: str = cage_location_info["fov_filename"].iloc[0]
    cage_coords: np.ndarray = (
        cage_location_info[["cage_xtl", "cage_ytl", "cage_xbr", "cage_ybr"]]
        .values[0]
        .astype(int)
    )

    fov_start_x, fov_start_y = get_fov_coords(bf_img_name)
    # convert these coordinates to pixels later when you correctly compute them

    fov_start_x = int(np.round(fov_start_x * MAGNIFICATION / PIXEL_SIZE))
    fov_start_y = int(np.round(fov_start_y * MAGNIFICATION / PIXEL_SIZE))

    # cage crop coordinates
    xc_tl, yc_tl, xc_br, yc_br = cage_coords - np.array([fov_start_x, fov_start_y] * 2)

    raw_images_path: str = os.path.join(
        exp_base_path, "scan_" + str(scan_id), "lane_" + str(lane_id)
    )

    # read all the images
    out_images: Dict[str, np.ndarray] = load_images(
        bf_img_filename=bf_img_name,
        raw_images_folder=raw_images_path,
        fl_channel_identifiers=fl_channel_identifiers,
    )

    # read the Mask R-CNN results from the reference scan
    ref_bf_img_name: str = cage_location_info["fov_filename_ref"].iloc[0]
    ref_raw_images_path: str = os.path.join(
        exp_base_path, "scan_" + str(ref_scan_id), "lane_" + str(lane_id)
    )
    ref_mask_rcnn_results_path: str = os.path.join(
        exp_base_path, "scan_" + str(ref_scan_id), "lane_" + str(lane_id) + "_results"
    )

    ref_out_images, ref_mask_rcnn_results = load_images_and_mask_rcnn_results(
        bf_img_filename=ref_bf_img_name,
        raw_images_folder=ref_raw_images_path,
        mask_rcnn_results_folder=ref_mask_rcnn_results_path,
        fl_channel_identifiers=fl_channel_identifiers,
        inst_segmentor_model=inst_segmentor_model,
    )

    cage_boxes: np.ndarray = np.array(
        [
            box
            for i, box in enumerate(ref_mask_rcnn_results["predictions"]["boxes"])
            if ref_mask_rcnn_results["predictions"]["labels"][i]
            == inst_segmentor_model.get_reverse_label_map()["cage"]
        ]
    )

    cage_idxs: np.ndarray = [
        i
        for i, label in enumerate(ref_mask_rcnn_results["predictions"]["labels"])
        if label == inst_segmentor_model.get_reverse_label_map()["cage"]
    ]

    iou_matrix: np.ndarray = iou_batch(
        bboxes1=np.array([[xc_tl, yc_tl, xc_br, yc_br]]), bboxes2=cage_boxes
    )

    cage_mask_from_mask_rcnn: np.ndarray = ref_mask_rcnn_results["predictions"][
        "masks"
    ][cage_idxs[np.argmax(iou_matrix[0])]]

    # update the bounding boxes as well
    xc_tl, yc_tl, xc_br, yc_br = ref_mask_rcnn_results["predictions"]["boxes"][
        cage_idxs[np.argmax(iou_matrix[0])]
    ]

    cnts, _ = cv2.findContours(
        cage_mask_from_mask_rcnn.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # contours of objects detected by Mask R-CNN
    mask_rcnn_obj_contours: List[np.ndarray] = [cnts[0]]

    caged_cell_instances_mask: np.ndarray = ref_mask_rcnn_results["combined_obj_masks"][
        "cell"
    ][yc_tl:yc_br, xc_tl:xc_br] * cage_mask_from_mask_rcnn.astype(int)
    cell_ids_in_cage: np.ndarray = np.unique(
        caged_cell_instances_mask[caged_cell_instances_mask > 0]
    )

    target_cells_mask_from_mask_rcnn: np.ndarray = np.zeros(
        cage_mask_from_mask_rcnn.shape, np.uint8
    )
    # threshold to use for classifying the target cells for the reference Mask R-CNN scan
    threshold_for_pixel_classification_per_fl_channel: Dict[str, float] = (
        thresholds_for_pixel_classification[ref_scan_id][lane_id]
    )

    num_target_cells: int = 0
    for cell_id in cell_ids_in_cage:
        this_cell_mask: np.ndarray = np.zeros(caged_cell_instances_mask.shape, np.uint8)
        this_cell_mask[caged_cell_instances_mask == cell_id] = 1
        # keep the cells only in the cage
        # this_cell_mask = this_cell_mask * cage_mask_from_mask_rcnn
        if (
            np.sum(ref_out_images["Blue"][yc_tl:yc_br, xc_tl:xc_br] * this_cell_mask)
            >= np.sum(this_cell_mask)
            * threshold_for_pixel_classification_per_fl_channel["Blue"]
        ):
            target_cells_mask_from_mask_rcnn = np.maximum(
                target_cells_mask_from_mask_rcnn, this_cell_mask
            )
            num_target_cells += 1
        # find the contour
        cnts, _ = cv2.findContours(
            this_cell_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # pick the contour with the maximum area
        max_cell_area: float = 0
        cell_contour: np.ndarray = None
        for contour in cnts:
            area: float = cv2.contourArea(contour)
            if area > max_cell_area:
                max_cell_area = area
                cell_contour = contour
        if max_cell_area > 0:
            mask_rcnn_obj_contours += [cell_contour]

    print(f"Num cells: {len(cell_ids_in_cage)}, num targets: {num_target_cells}")

    # normalize the White channel image, for visualization
    ref_img_with_mask_rcnn_masks: np.ndarray = ref_out_images["White"].copy()
    ref_img_with_mask_rcnn_masks = (
        255 * ref_img_with_mask_rcnn_masks.astype(float) / (2**12 - 1)
    ).astype(np.uint8)
    ref_img_with_mask_rcnn_masks = cv2.normalize(
        ref_img_with_mask_rcnn_masks,
        ref_img_with_mask_rcnn_masks,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
    )
    ref_img_with_mask_rcnn_masks = ref_img_with_mask_rcnn_masks[
        yc_tl:yc_br, xc_tl:xc_br
    ]
    ref_img_with_mask_rcnn_masks = cv2.cvtColor(
        ref_img_with_mask_rcnn_masks, cv2.COLOR_GRAY2RGB
    )

    # identify the target cells in the Mask R-CNN results with the Blue stain
    ref_img_with_mask_rcnn_masks = (
        ref_img_with_mask_rcnn_masks * 0.6
        + cv2.cvtColor(target_cells_mask_from_mask_rcnn, cv2.COLOR_GRAY2RGB)
        * colors["Blue"]
        * 0.4
    ).astype(np.uint8)

    cv2.drawContours(
        ref_img_with_mask_rcnn_masks, mask_rcnn_obj_contours, -1, (255, 255, 0), 2
    )

    # normalization is not needed for sem_segmentor_model.predict_cages as it takes care of it inside the method
    cage_semantic_masks, _ = sem_segmentor_model.predict_cages(
        input_image=out_images["White"],
        cage_bounding_boxes=np.array([[xc_tl, yc_tl, xc_br, yc_br]]),
        percentage_to_expand_cage_box_boundaries=0.2,
    )

    cage_semantic_mask = cage_semantic_masks[0]
    foreground_cells_mask = np.zeros(cage_semantic_mask.shape, np.uint8)
    # beads overlapping will cells (if detected properly) will be excluded below
    foreground_cells_mask[
        cage_semantic_mask == sem_segmentor_model.get_reverse_label_map()["cell"]
    ] = 1
    # only consider the semantic masks for cell inside the cage mask from Mask R-CNN
    foreground_cells_mask = foreground_cells_mask * cage_mask_from_mask_rcnn

    normalized_cropped_img: np.ndarray = (
        255 * out_images["White"][yc_tl:yc_br, xc_tl:xc_br].astype(float) / (2**12 - 1)
    ).astype(np.uint8)
    normalized_cropped_img = cv2.normalize(
        normalized_cropped_img,
        normalized_cropped_img,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
    )
    normalized_cropped_img = normalized_cropped_img * cage_mask_from_mask_rcnn
    normalized_cropped_img = cv2.cvtColor(normalized_cropped_img, cv2.COLOR_GRAY2RGB)

    img_with_cell_masks: np.ndarray = normalized_cropped_img.copy()
    # add the semantic segmentation mask in Green color
    img_with_cell_masks = (
        img_with_cell_masks * 0.9
        + cv2.cvtColor(foreground_cells_mask, cv2.COLOR_GRAY2RGB)
        * np.array([0, 255, 0])
        * 0.1
    ).astype(np.uint8)

    img_overlay: np.ndarray = normalized_cropped_img.copy()

    # threshold to use for classifying the pixels for the current scan
    threshold_for_pixel_classification_per_fl_channel: Dict[str, float] = (
        thresholds_for_pixel_classification[scan_id][lane_id]
    )

    for fl_channel_identifier in fl_channel_identifiers:
        if (
            fl_channel_identifier
            not in threshold_for_pixel_classification_per_fl_channel
        ):
            continue
        img: np.ndarray = out_images[fl_channel_identifier][yc_tl:yc_br, xc_tl:xc_br]
        mask: np.ndarray = np.zeros(foreground_cells_mask.shape, np.uint8)
        mask[
            img
            >= threshold_for_pixel_classification_per_fl_channel[fl_channel_identifier]
        ] = 1
        mask = mask * foreground_cells_mask

        img_overlay = (
            img_overlay * 0.6
            + cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            * colors[fl_channel_identifier]
            * 0.4
        ).astype(np.uint8)

    return ref_img_with_mask_rcnn_masks, img_with_cell_masks, img_overlay


def merge_cages_metrics_dataframes_using_cage_coords(
    cages_metrics: Dict[str, pd.DataFrame],
    ref_cages_metrics: Dict[str, pd.DataFrame],
    columns_to_include: List[str],
    columns_to_include_from_ref: List[str],
):

    # merge the cages metrics for the provided scan with the Mask R-CNN metrics, for both experiment and control lanes
    # use 'all', but all are the same, this key should always exist for the cages metrics dictionary
    key_0 = "all"
    paired_idxs, unpaired_1, unpaired_2 = pair_cages_using_iou(
        cage_boxes=cages_metrics[key_0][
            ["cage_xtl", "cage_ytl", "cage_xbr", "cage_ybr"]
        ].values.astype(float),
        reference_cage_boxes=ref_cages_metrics[key_0][
            ["cage_xtl", "cage_ytl", "cage_xbr", "cage_ybr"]
        ].values.astype(float),
        min_iou_for_pairing=0.8,
    )

    print(
        f"Successfully paired {len(paired_idxs)} cages out of {len(cages_metrics[key_0])}!"
    )
    # merge the two results
    merged_cages_metrics = {}
    index_mapping = {i: j for (i, j) in paired_idxs}

    for key in cages_metrics:
        df: pd.DataFrame = cages_metrics[key][columns_to_include].copy()
        ref_df: pd.DataFrame = ref_cages_metrics[key][
            columns_to_include_from_ref
        ].copy()
        # aad a column to each DataFrame based on the index mapping
        # map df index to ref_df index
        df["cage_id"] = df.index.map(index_mapping)
        # use ref_df index as the key for merging
        ref_df["cage_id"] = ref_df.index
        # Merge based on the custom mapping key
        merged_cages_metrics[key] = df.merge(
            ref_df, on="cage_id", suffixes=("", "_ref")
        )

    return merged_cages_metrics


def gmm_fit(
    avg_fl_signal_per_cell: Dict[str, List[float]],
    fl_channel_identifier: str,
    show_plot: bool = True,
):

    data = np.array(avg_fl_signal_per_cell[fl_channel_identifier])

    # each FL channel identifies one cell type
    n_components = 2
    # create and fit the Gaussian Mixture Model (GMM)
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(data.reshape(-1, 1))  # reshape data to (n_samples, n_features)

    # get GMM parameters
    means = gmm.means_.flatten()
    variances = gmm.covariances_.flatten()
    weights = gmm.weights_

    def get_pdf(i, x):
        return weights[i] * norm.pdf(x, means[i], np.sqrt(variances[i]))

    # find the intersection point of the two PDFs
    # this is the threshold that best separates the two components (maximum likelihood)
    try:
        threshold = brentq(
            lambda x: get_pdf(1, x) - get_pdf(0, x), min(data), max(data)
        )
    except ValueError:
        threshold = min(data)

    print(f"Threshold (log): {threshold}, (linear): {10**threshold}")
    print(f"Number of effectors: {len(data[data <= threshold]) / len(data)}")
    print(f"Number of targets: {len(data[data > threshold]) / len(data)}")
    print(f"Weights: {weights}")

    if show_plot:
        # Plot the results
        x_val = np.linspace(min(data), max(data), 1000)
        pdf = np.zeros_like(x_val)

        for i, (mean, var, weight) in enumerate(zip(means, variances, weights)):
            pdf += weight * norm.pdf(x_val, mean, np.sqrt(var))

        plt.hist(
            data,
            bins=100,
            density=True,
            alpha=0.5,
            color="skyblue",
            label=f"Log10(Mean {fl_channel_identifier} Signal) over Cells",
        )
        plt.plot(x_val, pdf, color="red", label="GMM Fit")
        plt.axvline(x=threshold, color="red", linestyle="--")
        plt.title("Gaussian Mixture Model Fit")
        plt.xlabel(f"Log(Mean {fl_channel_identifier} Signal)")
        plt.ylabel("Density")
        plt.legend()
        plt.show()
        print(f"Threshold (log): {threshold}, (linear): {10**threshold}")
        print(f"Number of effectors: {len(data[data <= threshold]) / len(data)}")
        print(f"Number of targets: {len(data[data > threshold]) / len(data)}")

    return threshold


def kde_fit(
    avg_fl_signal_per_cell: Dict[str, List[float]],
    fl_channel_identifier: str,
    show_plot: bool = True,
):

    data = np.array(avg_fl_signal_per_cell[fl_channel_identifier])

    # fit KDE
    kde = KernelDensity(bandwidth=1.0)
    kde.fit(data.reshape(-1, 1))

    # generate density over a range of values
    x_vals = np.linspace(min(data), max(data), 1000)
    log_density = kde.score_samples(x_vals[:, None])

    # detect peaks in the KDE
    peaks, _ = find_peaks(np.exp(log_density))
    peak_positions = x_vals[peaks]
    if len(peaks) >= 2:
        two_highest_peak_idxs = np.argsort(-log_density[peaks])[:2]
        peak_positions = peak_positions[two_highest_peak_idxs]
        threshold = (peak_positions[1] + peak_positions[0]) / 2.0
    else:
        threshold = min(data)

    if show_plot:
        # Plot the results
        pdf = np.zeros_like(x_vals)

        plt.hist(
            data,
            bins=100,
            density=True,
            alpha=0.5,
            color="skyblue",
            label=f"Log10(Mean {fl_channel_identifier} Signal) over Cells",
        )

        plt.axvline(x=threshold, color="red", linestyle="--")
        plt.title("Kernel Density Estimation Fit")
        plt.xlabel(f"Log(Mean {fl_channel_identifier} Signal)")
        plt.ylabel("Density")
        plt.legend()
        plt.show()
        print(f"Threshold (log): {threshold}, (linear): {10**threshold}")
        print(f"Number of effectors: {len(data[data <= threshold]) / len(data)}")
        print(f"Number of targets: {len(data[data > threshold]) / len(data)}")

    return threshold


def kmeans_fit(
    avg_fl_signal_per_cell: Dict[str, List[float]],
    fl_channel_identifier: str,
    show_plot: bool = True,
):
    data = np.array(avg_fl_signal_per_cell[fl_channel_identifier])

    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
    kmeans.fit(data.reshape(-1, 1))
    labels = kmeans.labels_
    unique_labels = np.unique(labels)
    unique_label_counts = np.zeros_like(unique_labels)
    for i, label in enumerate(unique_labels):
        unique_label_counts[i] = len(np.where(labels == label)[0])

    if len(unique_label_counts) >= 2:
        two_highest_density_idxs = np.argsort(-unique_label_counts)[:2]
        mean_1 = np.mean(data[labels == unique_labels[two_highest_density_idxs[0]]])
        mean_2 = np.mean(data[labels == unique_labels[two_highest_density_idxs[1]]])

        std_1 = np.std(data[labels == unique_labels[two_highest_density_idxs[0]]])
        std_2 = np.std(data[labels == unique_labels[two_highest_density_idxs[1]]])

        threshold = (std_1 * mean_2 + std_2 * mean_1) / (std_1 + std_2)

    else:
        threshold = min(data)

    # Plot the results
    if show_plot:
        plt.hist(
            data,
            bins=100,
            density=True,
            alpha=0.5,
            color="skyblue",
            label=f"Log10(Mean {fl_channel_identifier} Signal) over Cells",
        )

        plt.axvline(x=threshold, color="red", linestyle="--")
        plt.title("Kernel Density Estimation Fit")
        plt.xlabel(f"Log(Mean {fl_channel_identifier} Signal)")
        plt.ylabel("Density")
        plt.legend()
        plt.show()
        print(f"Threshold (log): {threshold}, (linear): {10**threshold}")
        print(f"Number of effectors: {len(data[data <= threshold]) / len(data)}")
        print(f"Number of targets: {len(data[data > threshold]) / len(data)}")

    return threshold


def gmm_fit_2(
    avg_fl_signal_per_cell: Dict[str, List[float]],
    fl_channel_identifier: str,
    show_plot: bool = True,
):

    data = np.array(avg_fl_signal_per_cell[fl_channel_identifier])

    aic_min = 1e30
    best_n_components = 2
    # each FL channel identifies one cell type
    for n_components in [2, 3, 4]:
        # create and fit the Gaussian Mixture Model (GMM)
        gmm = GaussianMixture(n_components=n_components, random_state=0)
        gmm.fit(data.reshape(-1, 1))  # reshape data to (n_samples, n_features)
        aic = gmm.aic(data.reshape(-1, 1))
        if aic < aic_min:
            aic_min = aic
            best_n_components = n_components

    # get GMM parameters
    gmm = GaussianMixture(n_components=best_n_components, random_state=0)
    gmm.fit(data.reshape(-1, 1))
    means = gmm.means_.flatten()
    variances = gmm.covariances_.flatten()
    weights = gmm.weights_

    if show_plot:
        # Plot the results
        x_val = np.linspace(min(data), max(data), 1000)
        pdf = np.zeros_like(x_val)

        for i, (mean, var, weight) in enumerate(zip(means, variances, weights)):
            pdf += weight * norm.pdf(x_val, mean, np.sqrt(var))

        plt.hist(
            data,
            bins=100,
            density=True,
            alpha=0.5,
            color="skyblue",
            label=f"Log10(Mean {fl_channel_identifier} Signal) over Cells",
        )
        plt.plot(x_val, pdf, color="red", label="GMM Fit")
        plt.title("Gaussian Mixture Model Fit")
        plt.xlabel(f"Log(Mean {fl_channel_identifier} Signal)")
        plt.ylabel("Density")
        plt.legend()

    # find the threshold between disjoint classes below
    # sort them from smallest to largest mean values
    sorted_idxs = np.argsort(means)
    means = means[sorted_idxs]
    variances = variances[sorted_idxs]
    weights = weights[sorted_idxs]

    # drop low weight distributions
    high_density_idxs = np.where(weights >= 0.05)[0]
    means = means[high_density_idxs]
    variances = variances[high_density_idxs]
    weights = weights[high_density_idxs]

    # merge the close ones
    while True:
        normalized_distances = []
        for i in range(len(means) - 1):
            normalized_distances.append(
                (means[i + 1] - means[i])
                / (np.sqrt(variances[i + 1]) + np.sqrt(variances[i]))
            )

        normalized_distances = np.array(normalized_distances)
        if np.min(normalized_distances) < 2:
            i = np.argmin(normalized_distances)
            merged_mean = (weights[i] * means[i] + weights[i + 1] * means[i + 1]) / (
                weights[i] + weights[i + 1]
            )
            merged_var = (
                weights[i] * (variances[i] + means[i] ** 2)
                + weights[i + 1] * (variances[i + 1] + means[i + 1] ** 2)
            ) / (weights[i] + weights[i + 1]) - merged_mean**2
            means = np.array(
                means[: max(0, i)].tolist()
                + [merged_mean]
                + means[min(i + 2, len(means)) :].tolist()
            )
            variances = np.array(
                variances[: max(0, i)].tolist()
                + [merged_var]
                + variances[min(i + 2, len(variances)) :].tolist()
            )
            weights = np.array(
                weights[: max(0, i)].tolist()
                + [weights[i] + weights[i + 1]]
                + weights[min(i + 2, len(weights)) :].tolist()
            )

        else:
            break

        if len(weights) <= 2:
            break

    two_highest_density_idxs = np.argsort(-weights)[:2]
    mean_1 = means[two_highest_density_idxs[0]]
    mean_2 = means[two_highest_density_idxs[1]]

    std_1 = np.sqrt(variances[two_highest_density_idxs[0]])
    std_2 = np.sqrt(variances[two_highest_density_idxs[1]])

    threshold = (std_1 * mean_2 + std_2 * mean_1) / (std_1 + std_2)

    if show_plot:
        plt.axvline(x=threshold, color="red", linestyle="--")
        plt.show()

        print(f"Threshold (log): {threshold}, (linear): {10**threshold}")
        print(f"Number of effectors: {len(data[data <= threshold]) / len(data)}")
        print(f"Number of targets: {len(data[data > threshold]) / len(data)}")

    return threshold


###########################################################
#
#    OLD CODE
#
###########################################################
def get_cages_metrics_old(
    bf_img_filename: str,
    raw_images_folder: str,
    mask_rcnn_results_folder: str,
    inst_segmentor_model,
    sem_segmentor_model,
    starting_cage_id=0,
):

    out_images, mask_rcnn_results = load_images_and_mask_rcnn_results(
        bf_img_filename=bf_img_filename,
        raw_images_folder=raw_images_folder,
        mask_rcnn_results_folder=mask_rcnn_results_folder,
        fl_channel_identifiers=["Blue", "Red"],
        inst_segmentor_model=inst_segmentor_model,
    )

    # load BF, Blue and Red images
    bf_img: np.ndarray = out_images["White"]
    red_img: np.ndarray = out_images["Red"]
    blue_img: np.ndarray = out_images["Blue"]

    # and load the results to get the combined_mask
    combined_obj_masks, preds = (
        mask_rcnn_results["combined_obj_masks"],
        mask_rcnn_results["predictions"],
    )

    # first, we calculate the total and average red and blue signal readouts per detected cell objects from Mask R-CNN
    # this is done over all cells and not only the caged ones
    # the average blue histogram will be used to find a threshold to differentiate between target and effector cells
    # (THRESHOLD_ON_AVG_BLUE_FOR_TARGET_CELLS already set below as have already found the value and target/effector cell
    # classification is also done below)
    # we also save the blue channel pixel values for all the target/effector cells (after classifications and in separate list)
    # to use for finding the right threshold on blue for classifying a pixel as a pixel of a target or effector cell
    # these histograms are already used to calculate THRESHOLD_ON_BLUE_FOR_TARGET_CELL_PIXELS that is used further below

    avg_red_per_cell: List[float] = []
    avg_blue_per_cell: List[float] = []

    blue_pixel_values_over_target_cells: List[int] = []
    blue_pixel_values_over_effector_cells: List[int] = []

    for i, box in enumerate(preds["boxes"]):
        # only consider detected objects of class cell
        if preds["labels"][i] != inst_segmentor_model.get_reverse_label_map()["cell"]:
            continue
        # a cell object
        xtl, ytl, xbr, ybr = box
        masked_red: np.ndarray = red_img[ytl:ybr, xtl:xbr] * preds["masks"][i]
        masked_blue: np.ndarray = blue_img[ytl:ybr, xtl:xbr] * preds["masks"][i]

        cell_area = np.sum(preds["masks"][i])
        # average values
        if cell_area > 0:
            avg_red_per_cell.append(float(np.sum(masked_red)) / cell_area)
            avg_blue_per_cell.append(float(np.sum(masked_blue)) / cell_area)
            # save the blue pixel values in a long list depending on the cell classification
            if (
                np.sum(masked_blue)
                >= THRESHOLD_ON_AVG_BLUE_FOR_TARGET_CELLS * cell_area
            ):
                # this is a target cell
                blue_pixel_values_over_target_cells += (
                    masked_blue[preds["masks"][i] > 0].flatten().astype(int).tolist()
                )
            else:
                # this is an effector cell
                blue_pixel_values_over_effector_cells += (
                    masked_blue[preds["masks"][i] > 0].flatten().astype(int).tolist()
                )
        else:
            # this should not really happen
            avg_red_per_cell.append(np.nan)
            avg_blue_per_cell.append(np.nan)

    # normalized_bf_img: np.ndarray = (bf_img.astype(float) / (2**12 - 1) * 255).astype(np.uint8)
    # normalized_bf_img: np.ndarray = cv2.normalize(normalized_bf_img, normalized_bf_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    cage_boxes: List[np.ndarray] = [
        box
        for i, box in enumerate(preds["boxes"])
        if preds["labels"][i] == inst_segmentor_model.get_reverse_label_map()["cage"]
    ]

    cage_boxes = (
        np.array(cage_boxes) if len(cage_boxes) > 0 else np.zeros((0, 4), dtype=int)
    )
    # a mapping between the index of a cage in the cage_boxes above, and Mask R-CNN results preds
    cage_box_idx_to_mask_rcnn_results_idx_map: Dict[int, int] = {
        k: v
        for k, v in enumerate(
            [
                i
                for i, class_id in enumerate(preds["labels"])
                if class_id == inst_segmentor_model.get_reverse_label_map()["cage"]
            ]
        )
    }

    # crops_list: List[np.ndarray] = generate_cage_crops(img=normalized_img,
    #                                                    in_cage_boxes=cage_boxes,
    #                                                    percentage_to_expand_cage_box_boundaries=0.2,
    #                                                    crop_size=sem_segmentor_model.get_model_input_size())

    # masks = sem_segmentor_model.predict_batch(crops_list)

    # normalization is not needed for sem_segmentor_model.predict_cages as it takes care of it inside the method
    cage_semantic_masks, _ = sem_segmentor_model.predict_cages(
        input_image=bf_img,
        cage_bounding_boxes=cage_boxes,
        percentage_to_expand_cage_box_boundaries=0.2,
    )

    num_cells_per_cage: Dict[int, int] = {}
    num_target_cells_per_cage: Dict[int, int] = {}
    num_effector_cells_per_cage: Dict[int, int] = {}

    num_foreground_pixels_mask_rcnn: Dict[int, int] = {}
    num_target_pixels_mask_rcnn: Dict[int, int] = {}
    num_effector_pixels_mask_rcnn: Dict[int, int] = {}

    num_foreground_pixels: Dict[int, int] = {}
    num_target_pixels: Dict[int, int] = {}
    num_effector_pixels: Dict[int, int] = {}

    total_red_signal_all_pixels: Dict[int, int] = {}
    total_red_signal_target_pixels: Dict[int, int] = {}
    total_red_signal_effector_pixels: Dict[int, int] = {}

    for i, cage_semantic_mask in enumerate(cage_semantic_masks):
        # cage_semantic_masks[i] is corresponding to cage_boxes[i], they are the same size and from the same area of the image
        xtl, ytl, xbr, ybr = cage_boxes[i]
        cage_mask_from_mask_rcnn: np.ndarray = preds["masks"][
            cage_box_idx_to_mask_rcnn_results_idx_map[i]
        ]

        # in the following, we compute the total number of cells, the target and effector cells and the number of pixels
        # belonging to each category using Mask R-CNN results and the threshold found on the average blue channel over cells
        # to classify them (into target/effector)

        # number of cell objects detected inside this cage
        caged_cell_instances_mask: np.ndarray = combined_obj_masks["cell"][
            ytl:ybr, xtl:xbr
        ]
        cell_ids_in_cage: np.ndarray = np.unique(
            caged_cell_instances_mask[caged_cell_instances_mask > 0]
        )
        num_cells_in_cage: int = len(cell_ids_in_cage)
        # number of effector/target cells in this cage
        num_targets_in_cage: int = 0
        num_effectors_in_cage: int = 0

        # calculate the mask for all the objects, we do it this way to avoid double counting pixel in overlapping cell areas
        foreground_mask_from_mask_rcnn: np.ndarray = np.zeros(
            caged_cell_instances_mask.shape, np.uint8
        )
        target_cells_mask_from_mask_rcnn: np.ndarray = np.zeros(
            caged_cell_instances_mask.shape, np.uint8
        )
        effector_cells_mask_from_mask_rcnn: np.ndarray = np.zeros(
            caged_cell_instances_mask.shape, np.uint8
        )

        for cell_id in cell_ids_in_cage:
            # this_cell_mask can be partially or completely outside the cage, we do not include cage_mask_from_mask_rcnn
            # yet because first we need to classify the cell as target/effector
            this_cell_mask: np.ndarray = np.zeros(
                caged_cell_instances_mask.shape, np.uint8
            )
            this_cell_mask[caged_cell_instances_mask == cell_id] = 1
            # take the union, we multiply by cage_mask_from_mask_rcnn to only consider parts inside the cage
            # we do not skip cells that are partially (even very small portion) inside the cage
            foreground_mask_from_mask_rcnn = np.maximum(
                foreground_mask_from_mask_rcnn,
                this_cell_mask * cage_mask_from_mask_rcnn,
            )
            # check if this is a target cell using the average blue channel and THRESHOLD_ON_AVG_BLUE_FOR_TARGET_CELLS
            if (
                np.sum(blue_img[ytl:ybr, xtl:xbr] * this_cell_mask)
                >= THRESHOLD_ON_AVG_BLUE_FOR_TARGET_CELLS * this_cell_mask.sum()
            ):
                num_targets_in_cage += 1
                target_cells_mask_from_mask_rcnn = np.maximum(
                    target_cells_mask_from_mask_rcnn,
                    this_cell_mask * cage_mask_from_mask_rcnn,
                )
            else:
                num_effectors_in_cage += 1
                effector_cells_mask_from_mask_rcnn = np.maximum(
                    effector_cells_mask_from_mask_rcnn,
                    this_cell_mask * cage_mask_from_mask_rcnn,
                )

        num_foreground_pixels_in_cage: int = int(foreground_mask_from_mask_rcnn.sum())
        num_target_pixels_in_cage: int = int(target_cells_mask_from_mask_rcnn.sum())
        num_effector_pixels_in_cage: int = int(effector_cells_mask_from_mask_rcnn.sum())

        # save the number of cells for this cage
        num_cells_per_cage[starting_cage_id + i] = num_cells_in_cage
        num_target_cells_per_cage[starting_cage_id + i] = num_targets_in_cage
        num_effector_cells_per_cage[starting_cage_id + i] = num_effectors_in_cage

        num_foreground_pixels_mask_rcnn[starting_cage_id + i] = (
            num_foreground_pixels_in_cage
        )
        num_target_pixels_mask_rcnn[starting_cage_id + i] = num_target_pixels_in_cage
        num_effector_pixels_mask_rcnn[starting_cage_id + i] = (
            num_effector_pixels_in_cage
        )

        # in the following, we use the extracted semantic mask for cells inside the cage to calculate
        # the number of pixels belonging to cells, target and effector cells

        # mask the area with the foreground cell mask (cage_mask includes cages, cells and beads semantic segments)
        foreground_cell_mask = np.zeros(cage_semantic_mask.shape, np.uint8)
        # beads overlapping will cells (if detected properly) will be excluded below
        foreground_cell_mask[
            cage_semantic_mask == sem_segmentor_model.get_reverse_label_map()["cell"]
        ] = 1
        # only consider the semantic masks for cell inside the cage mask from Mask R-CNN
        foreground_cell_mask = foreground_cell_mask * cage_mask_from_mask_rcnn

        masked_red: np.ndarray = red_img[ytl:ybr, xtl:xbr] * foreground_cell_mask
        masked_blue: np.ndarray = blue_img[ytl:ybr, xtl:xbr] * foreground_cell_mask
        # create a mask for pixels corresponding to target cells
        target_cell_pixels_mask: np.ndarray = np.zeros(
            cage_semantic_mask.shape, np.uint8
        )
        target_cell_pixels_mask[
            masked_blue >= THRESHOLD_ON_BLUE_FOR_TARGET_CELL_PIXELS
        ] = 1

        # do the same for effector cells (to find dead effectors)
        effector_cell_pixels_mask: np.ndarray = np.zeros(
            cage_semantic_mask.shape, np.uint8
        )
        effector_cell_pixels_mask[
            masked_blue < THRESHOLD_ON_BLUE_FOR_TARGET_CELL_PIXELS
        ] = 1
        # this step is needed to make sure 0/small blue value non-cell pixels are not classified as effector cell pixels
        effector_cell_pixels_mask = effector_cell_pixels_mask * foreground_cell_mask

        total_red_signal_all_pixels[starting_cage_id + i] = int(masked_red.sum())
        total_red_signal_target_pixels[starting_cage_id + i] = int(
            np.sum(masked_red * target_cell_pixels_mask)
        )
        total_red_signal_effector_pixels[starting_cage_id + i] = int(
            np.sum(masked_red * effector_cell_pixels_mask)
        )

        num_target_pixels[starting_cage_id + i] = int(target_cell_pixels_mask.sum())
        num_effector_pixels[starting_cage_id + i] = int(effector_cell_pixels_mask.sum())
        num_foreground_pixels[starting_cage_id + i] = int(foreground_cell_mask.sum())

    return (
        avg_red_per_cell,
        avg_blue_per_cell,
        blue_pixel_values_over_target_cells,
        blue_pixel_values_over_effector_cells,
        pd.DataFrame(
            {
                "num_cells_per_cage": num_cells_per_cage,
                "num_target_cells_per_cage": num_target_cells_per_cage,
                "num_effector_cells_per_cage": num_effector_cells_per_cage,
                "total_red_signal_all_pixels": total_red_signal_all_pixels,
                "total_red_signal_target_pixels": total_red_signal_target_pixels,
                "total_red_signal_effector_pixels": total_red_signal_effector_pixels,
                "num_foreground_pixels": num_foreground_pixels,
                "num_target_pixels": num_target_pixels,
                "num_effector_pixels": num_effector_pixels,
                "num_foreground_pixels_mask_rcnn": num_foreground_pixels_mask_rcnn,
                "num_target_pixels_mask_rcnn": num_target_pixels_mask_rcnn,
                "num_effector_pixels_mask_rcnn": num_effector_pixels_mask_rcnn,
            }
        ),
    )


def generate_cage_plots(
    bf_img_filename: str,
    raw_images_folder: str,
    mask_rcnn_results_folder: str,
    inst_segmentor_model,
    sem_segmentor_model,
):

    # load the Mask R-CNN results for each FoV from disk
    results_filename: str = ".".join(bf_img_filename.strip().split(".")[:-1]) + ".pkl"
    _, combined_obj_masks, preds, _ = load_masks_in_coco_rle_format(
        filename=os.path.join(mask_rcnn_results_folder, results_filename)
    )

    # load all the images (BF, Red and Blue) from disk
    # in this expriment, Red is dead cell stain, and Blue is target cell stain
    fl_img_filenames: Tuple[str] = get_fl_channel_filenames(
        bf_img_filename, {"Red": RED_OFFSET_TO_BF, "Blue": BLUE_OFFSET_TO_BF}
    )

    red_img_filename: str = fl_img_filenames[0]
    blue_img_filename: str = fl_img_filenames[1]

    # load BF, Blue and Red images
    bf_img: np.ndarray = cv2.imread(
        os.path.join(raw_images_folder, bf_img_filename), cv2.IMREAD_UNCHANGED
    )
    red_img: np.ndarray = cv2.imread(
        os.path.join(raw_images_folder, red_img_filename), cv2.IMREAD_UNCHANGED
    )
    blue_img: np.ndarray = cv2.imread(
        os.path.join(raw_images_folder, blue_img_filename), cv2.IMREAD_UNCHANGED
    )

    # Plot Mask R-CNN results
    normalized_bf_img: np.ndarray = (bf_img.astype(float) / (2**12 - 1) * 255).astype(
        np.uint8
    )
    normalized_bf_img: np.ndarray = cv2.normalize(
        normalized_bf_img,
        normalized_bf_img,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
    )
    debug_img: np.ndarray = show_detections(
        normalized_bf_img, preds, inst_segmentor_model.get_label_map()
    )

    # apply a median filter to Red and Blue channels to fill the holes and smooth the images
    red_img = cv2.medianBlur(src=red_img, ksize=3)
    blue_img = cv2.medianBlur(src=blue_img, ksize=3)

    cage_boxes: List[np.ndarray] = [
        box
        for i, box in enumerate(preds["boxes"])
        if preds["labels"][i] == inst_segmentor_model.get_reverse_label_map()["cage"]
    ]

    cage_boxes = (
        np.array(cage_boxes) if len(cage_boxes) > 0 else np.zeros((0, 4), dtype=int)
    )
    # a mapping between the index of a cage in the cage_boxes above, and Mask R-CNN results preds
    cage_box_idx_to_mask_rcnn_results_idx_map: Dict[int, int] = {
        k: v
        for k, v in enumerate(
            [
                i
                for i, class_id in enumerate(preds["labels"])
                if class_id == inst_segmentor_model.get_reverse_label_map()["cage"]
            ]
        )
    }

    # normalization is not needed for sem_segmentor_model.predict_cages as it takes care of it inside the method
    cage_semantic_masks, _ = sem_segmentor_model.predict_cages(
        input_image=bf_img,
        cage_bounding_boxes=cage_boxes,
        percentage_to_expand_cage_box_boundaries=0.2,
    )

    red_channels: List[np.ndarray] = []
    blue_channels: List[np.ndarray] = []
    obj_masks_from_mask_rcnn: List[np.ndarray] = []
    cage_masks_from_mask_rcnn: List[np.ndarray] = []

    foreground_cell_masks: List[np.ndarray] = []
    target_cell_pixels_masks: List[np.ndarray] = []
    effector_cell_pixels_masks: List[np.ndarray] = []

    foreground_masks_from_mask_rcnn: List[np.ndarray] = []
    target_cells_masks_from_mask_rcnn: List[np.ndarray] = []
    effector_cells_masks_from_mask_rcnn: List[np.ndarray] = []

    for i, cage_semantic_mask in enumerate(cage_semantic_masks):
        # cage_semantic_masks[i] is corresponding to cage_boxes[i], they are the same size and from the same area of the image
        xtl, ytl, xbr, ybr = cage_boxes[i]
        cage_mask_from_mask_rcnn: np.ndarray = preds["masks"][
            cage_box_idx_to_mask_rcnn_results_idx_map[i]
        ]
        cage_masks_from_mask_rcnn.append(cage_mask_from_mask_rcnn)

        # number of cell objects detected inside this cage
        caged_cell_instances_mask: np.ndarray = combined_obj_masks["cell"][
            ytl:ybr, xtl:xbr
        ]
        cell_ids_in_cage: np.ndarray = np.unique(
            caged_cell_instances_mask[caged_cell_instances_mask > 0]
        )
        # calculate the mask for all the objects, we do it this way to avoid double counting pixel in overlapping cell areas
        foreground_mask_from_mask_rcnn: np.ndarray = np.zeros(
            caged_cell_instances_mask.shape, np.uint8
        )
        target_cells_mask_from_mask_rcnn: np.ndarray = np.zeros(
            caged_cell_instances_mask.shape, np.uint8
        )
        effector_cells_mask_from_mask_rcnn: np.ndarray = np.zeros(
            caged_cell_instances_mask.shape, np.uint8
        )

        for cell_id in cell_ids_in_cage:
            this_cell_mask: np.ndarray = np.zeros(
                caged_cell_instances_mask.shape, np.uint8
            )
            this_cell_mask[caged_cell_instances_mask == cell_id] = 1
            # take the union
            foreground_mask_from_mask_rcnn = np.maximum(
                foreground_mask_from_mask_rcnn,
                this_cell_mask * cage_mask_from_mask_rcnn,
            )
            # check if this is a target cell using the average blue channel and THRESHOLD_ON_AVG_BLUE_FOR_TARGET_CELLS
            if (
                np.sum(blue_img[ytl:ybr, xtl:xbr] * this_cell_mask)
                >= THRESHOLD_ON_AVG_BLUE_FOR_TARGET_CELLS * this_cell_mask.sum()
            ):
                target_cells_mask_from_mask_rcnn = np.maximum(
                    target_cells_mask_from_mask_rcnn,
                    this_cell_mask * cage_mask_from_mask_rcnn,
                )

                debug_img[ytl:ybr, xtl:xbr][
                    this_cell_mask * cage_mask_from_mask_rcnn > 0
                ] = (
                    debug_img[ytl:ybr, xtl:xbr][
                        this_cell_mask * cage_mask_from_mask_rcnn > 0
                    ]
                    * 0.6
                    + (
                        cv2.cvtColor(
                            (this_cell_mask * cage_mask_from_mask_rcnn),
                            cv2.COLOR_GRAY2RGB,
                        )
                        * np.array([255, 0, 0])
                    )[this_cell_mask * cage_mask_from_mask_rcnn > 0]
                    * 0.4
                ).astype(np.uint8)

            else:
                effector_cells_mask_from_mask_rcnn = np.maximum(
                    effector_cells_mask_from_mask_rcnn,
                    this_cell_mask * cage_mask_from_mask_rcnn,
                )

                debug_img[ytl:ybr, xtl:xbr][
                    this_cell_mask * cage_mask_from_mask_rcnn > 0
                ] = (
                    debug_img[ytl:ybr, xtl:xbr][
                        this_cell_mask * cage_mask_from_mask_rcnn > 0
                    ]
                    * 0.6
                    + (
                        cv2.cvtColor(
                            (this_cell_mask * cage_mask_from_mask_rcnn),
                            cv2.COLOR_GRAY2RGB,
                        )
                        * np.array([0, 255, 0])
                    )[this_cell_mask * cage_mask_from_mask_rcnn > 0]
                    * 0.4
                ).astype(np.uint8)

        foreground_masks_from_mask_rcnn.append(foreground_mask_from_mask_rcnn)
        target_cells_masks_from_mask_rcnn.append(target_cells_mask_from_mask_rcnn)
        effector_cells_masks_from_mask_rcnn.append(effector_cells_mask_from_mask_rcnn)
        #
        obj_masks_from_mask_rcnn.append(debug_img[ytl:ybr, xtl:xbr])

        # in the following, we use the extracted semantic mask for cells inside the cage to calculate
        # the number of pixels belonging to cells, target and effector cells

        # mask the area with the foreground cell mask (cage_mask includes cages, cells and beads semantic segments)
        foreground_cell_mask = np.zeros(cage_semantic_mask.shape, np.uint8)
        # beads overlapping will cells (if detected properly) will be excluded below
        foreground_cell_mask[
            cage_semantic_mask == sem_segmentor_model.get_reverse_label_map()["cell"]
        ] = 1
        # only consider the semantic masks for cell inside the cage mask from Mask R-CNN
        foreground_cell_mask = foreground_cell_mask * cage_mask_from_mask_rcnn
        foreground_cell_masks.append(foreground_cell_mask)

        masked_red: np.ndarray = red_img[ytl:ybr, xtl:xbr] * foreground_cell_mask
        masked_blue: np.ndarray = blue_img[ytl:ybr, xtl:xbr] * foreground_cell_mask

        red_channels.append(masked_red)
        blue_channels.append(masked_blue)

        # create a mask for pixels corresponding to target cells
        target_cell_pixels_mask: np.ndarray = np.zeros(
            cage_semantic_mask.shape, np.uint8
        )
        target_cell_pixels_mask[
            masked_blue >= THRESHOLD_ON_BLUE_FOR_TARGET_CELL_PIXELS
        ] = 1
        target_cell_pixels_masks.append(target_cell_pixels_mask)

        # do the same for effector cells (to find dead effectors)
        effector_cell_pixels_mask: np.ndarray = np.zeros(
            cage_semantic_mask.shape, np.uint8
        )
        effector_cell_pixels_mask[
            masked_blue < THRESHOLD_ON_BLUE_FOR_TARGET_CELL_PIXELS
        ] = 1
        # this step is needed to make sure 0/small blue value non-cell pixels are not classified as effector cell pixels
        effector_cell_pixels_mask = effector_cell_pixels_mask * foreground_cell_mask
        effector_cell_pixels_masks.append(effector_cell_pixels_mask)

    return (
        red_channels,
        blue_channels,
        obj_masks_from_mask_rcnn,
        cage_masks_from_mask_rcnn,
        foreground_cell_masks,
        target_cell_pixels_masks,
        effector_cell_pixels_masks,
        foreground_masks_from_mask_rcnn,
        target_cells_masks_from_mask_rcnn,
        effector_cells_masks_from_mask_rcnn,
    )


def create_image_grid(images, grid_size, image_size):
    """
    Resizes a list of NumPy array images to a given size and arranges them in an m x n grid.

    Args:
        images (list of np.ndarray): List of images as NumPy arrays.
        grid_size (tuple): Grid dimensions (m, n) where m is rows and n is columns.
        image_size (tuple): New size for each image (height, width).

    Returns:
        np.ndarray: The resulting grid image as a NumPy array.
    """
    m, n = grid_size
    grid_height = m * image_size[0]
    grid_width = n * image_size[1]

    # Create an empty grid canvas (assuming RGB images, 3 channels)
    if images[0].ndim == 3:
        grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    else:
        grid_image = np.zeros((grid_height, grid_width), dtype=np.uint8)

    # Resize and place each image in the grid
    for idx, image in enumerate(images):
        if idx >= m * n:  # Stop if there are more images than grid cells
            break

        # Resize image
        resized_image = cv2.resize(image, image_size)  # cv2 uses (width, height)

        # Calculate position in the grid
        row = idx // n
        col = idx % n

        # Calculate the top-left corner of the placement
        y_start = row * image_size[0]
        x_start = col * image_size[1]

        # Place the resized image into the grid
        grid_image[
            y_start : y_start + image_size[0], x_start : x_start + image_size[1]
        ] = resized_image

    return grid_image


"""
avg_red_per_cell: List[float] = []
avg_blue_per_cell: List[float] = []

blue_pixel_values_over_target_cells: List[int] = []
blue_pixel_values_over_effector_cells: List[int] = []

cages_metrics_df: pd.DataFrame = pd.DataFrame()

total_num_cages: int = 0
cage_ids_per_bf_img_name: Dict[str, Tuple[int, int]] = {}

for bf_img_name in raw_bf_images:
    (avg_red_per_cell_fov, 
     avg_blue_per_cell_fov, 
     blue_pixel_values_over_target_cells_fov, 
     blue_pixel_values_over_effector_cells_fov,
     cages_metrics_df_fov) = get_cages_metrics_old(
         bf_img_filename=bf_img_name, 
         raw_images_folder=raw_images_path,
         mask_rcnn_results_folder=mask_rcnn_results_path, 
         inst_segmentor_model=inst_segmentor, 
         sem_segmentor_model=sem_segmentor, 
         starting_cage_id=total_num_cages
     )
    avg_red_per_cell += avg_red_per_cell_fov
    avg_blue_per_cell += avg_blue_per_cell_fov
    blue_pixel_values_over_target_cells += blue_pixel_values_over_target_cells_fov
    blue_pixel_values_over_effector_cells += blue_pixel_values_over_effector_cells_fov
    cages_metrics_df = pd.concat([cages_metrics_df, cages_metrics_df_fov])
    cage_ids_per_bf_img_name[bf_img_name] = (total_num_cages, len(cages_metrics_df) - 1)
    total_num_cages = len(cages_metrics_df)
"""
