from pycocotools import mask as coco_mask_util
from typing import Dict, List, Final
import pickle
import numpy as np

# threshold to apply on masks
MASK_THRESHOLD: Final[float] = 0.55


def save_masks_in_coco_rle_format(
    out: Dict[str, List], image_width: int, image_height: int, filename: str
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
        image_width (integer): The width of the input image to the model.
        image_height (integer): The height of the input image to the model.
        filename: the filename in string format. If an extension is provided in the filename,
            it should be '.pkl'. Otherwise, it will be overwritten.

    """
    record: dict = {}
    record["annotations"]: List[dict] = []
    # image height and width
    record["width"]: int = image_width
    record["height"]: int = image_height

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

    if filename is None or len(filename) == 0:
        filename = "output.pkl"

    # save the record dictionary as a pickle file
    parsed_filename: List[str] = filename.strip().split(".")

    if len(parsed_filename) == 1:
        # no extension has been provided in the filename
        # add extension
        filename = filename + ".pkl"
    else:
        name: str = ".".join(parsed_filename[:-1])
        extension: str = parsed_filename[-1]
        if extension != "pkl":
            filename = name + ".pkl"

    filehandler = open(filename, "wb")
    pickle.dump(record, filehandler)
    filehandler.close()


def load_masks_in_coco_rle_format(
    filename: str, model_label_map: Dict[int, str]
) -> (Dict[str, np.ndarray], Dict[str, List]):
    """
    A function to load the masks and bounding boxes for the detected objects from a file
    (pickle file) after the results were compressed using COCO RLE format

    Args:
        filename: the filename in string format. It should be '.pkl'. A FileNotFoundError exception
            is thrown if the file does not exist (should be handled by the caller).
        model_label_map: A dictionary with keys as class IDs and values as class names. This is
            only used to produce the combined mask of objects per class with keys as class names.

    Outputs:

        The model output in a dictionary format with the following keys and values:
            - "boxes": List of numpy arrays, or 4-element lists for each bounding box
            - "labels": List of integer labels (class IDs)
            - "masks":  List of numpy array for object masks within the bounding boxes
        The combined_masks, which is a dictionary with keys as the string class names
            (from the model's label map passed) and values as the instance masks of the objects of that
            class; for each class, the mask values of a given object (of that class) are the unique
            integer object ID of that object.  The object IDs are unique for all the objects and are
            not reused between objects of different classes
            NOTE: Each pixel in the mask as defined for each class below can only belong to one object
            (e.g., cell), so this mask will not clearly indicate the boundaries of overlapping objects
            (while the masks from the instance segmentation model do).

    """

    # model outputs to be read from file
    boxes: List[List] = []
    labels: List[int] = []
    masks: List[np.ndarray] = []

    # load the annotations
    filehandler = open(filename, "rb")
    annots = pickle.load(filehandler)
    filehandler.close()

    # input image dimensions for the results
    image_width: int = annots["width"]
    image_height: int = annots["height"]

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
        mask_copy: np.ndrray = combined_masks[class_name][ytl:ybr, xtl:xbr].copy()
        mask_copy[mask_this_object > 0] = obj_id + 1
        combined_masks[class_name][ytl:ybr, xtl:xbr] = mask_copy

    return combined_masks, {"boxes": boxes, "labels": labels, "masks": masks}


def save_combined_masks_in_coco_rle_format(
    combined_masks: Dict[str, np.ndarray],
    model_label_map: Dict[int, str],
    filename: str,
) -> None:
    """
    A function to load the masks and bounding boxes for the detected objects from a file
    (pickle file) after the results were compressed using COCO RLE format

    Args:
        combined_masks: A dictionary with keys as the string class names and values as the instance
            masks of the objects of that class; for each class, the mask values of a given object
            (of that class) are the unique integer object ID of that object. The object IDs are unique
            for all the objects and are not reused between objects of different classes.
            NOTE: Each pixel in the mask as defined for each class below can only belong to one object
            (e.g., cell), so this mask will not clearly indicate the boundaries of overlapping objects
            (while the masks from the instance segmentation model do).
        model_label_map: A dictionary with keys as class IDs and values as class names. This is
            only used to produce the combined mask of objects per class with keys as class names.
        filename: The filename in string format. If an extension is provided in the filename,
            it should be '.pkl'. Otherwise, it will be overwritten.

    """

    reverse_label_map: Dict[str, int] = {
        value: key for key, value in model_label_map.items()
    }

    boxes: List[List[int]] = []
    labels: List[int] = []
    masks: List[np.ndarray] = []
    for class_name, c_mask in combined_masks.items():
        image_height, image_width = c_mask.shape[:2]

        # object IDs for objects of this class (class_name)
        obj_ids: np.ndarray = np.unique(c_mask)
        # first id (0) is the background, so remove it
        obj_ids: np.ndarray = obj_ids[1:]

        # get bounding box coordinates for each mask
        num_objs: int = len(obj_ids)

        # split the gray-scale-encoded mask into a set
        # of binary masks
        full_res_masks: np.ndarray = c_mask == obj_ids[:, None, None]

        for i in range(num_objs):
            pos = np.where(full_res_masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1]) + 1
            ymin = np.min(pos[0])
            ymax = np.max(pos[0]) + 1
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(reverse_label_map[class_name])
            masks.append(full_res_masks[i][ymin:ymax, xmin:xmax].astype(np.uint8))

    save_masks_in_coco_rle_format(
        {"boxes": boxes, "labels": labels, "masks": masks},
        image_width,
        image_height,
        filename,
    )
