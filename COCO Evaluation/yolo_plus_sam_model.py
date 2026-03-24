import cv2
from PIL import Image
import numpy as np
import os
import time
import logging
import onnxruntime
import onnx
import torch
from segment_anything import sam_model_registry, SamPredictor
from typing import Tuple, List, Final, Optional, Dict, Union

# YOLO_MODEL_WEIGHTS_PATH: Final[str] = '/home/cellareye/Development/yolov5/runs/train/batch_1_batch_2_batch_3_images_20_epochs_m/weights/batch_1_batch_2_batch_3_images_20_epochs_m.onnx'
# YOLO_MODEL_WEIGHTS_PATH: Final[str] = '/home/cellareye/Development/yolov5/runs/train/batch_1_batch_2_batch_3_images_20_epochs_m/weights/batch_1_4_oof_images_20_epochs_m.onnx'
# YOLO_MODEL_WEIGHTS_PATH: Final[str] = '/home/cellareye/Development/yolov5/runs/train/sets_1_2_3_8_to_26_10x_bf_10_epochs_m/weights/20240511_sets_1_2_3_8_to_26_10x_bf_10_epochs_m.onnx'
YOLO_MODEL_WEIGHTS_PATH: Final[str] = (
    "/home/cellareye/Development/yolov5/runs/train/sets_1_2_3_8_to_27_10x_bf_10_epochs_m/weights/20240624_sets_1_2_3_8_to_27_10x_bf_10_epochs_m.onnx"
)
# YOLO_MODEL_WEIGHTS_PATH: Final[str] = '/home/cellareye/Development/yolov5/runs/train/sets_1_to_5_4x_bf_10_epochs_m/weights/20240611_sets_1_to_5_4x_bf_10_epochs_m.onnx'
# YOLO_MODEL_WEIGHTS_PATH: Final[str] = '/home/cellareye/Development/darknet/backup/nk92_cells_beads_10000_d.onnx'
# YOLO_MODEL_WEIGHTS_PATH: Final[str] = '/home/cellareye/Development/yolov5/runs/train/fluorescent_images_batch_1_20_epochs_m/weights/fluorescent_images_batch_1_20_epochs_m.onnx'
# YOLO_MODEL_WEIGHTS_PATH: Final[str] = '/home/cellareye/Development/yolov5/runs/train/nuclei_bf_25_epochs_m/weights/nuclei_bf_25_epochs_m.onnx'
DEFAULT_CLASS_NAMES_PATH: Final[str] = (
    "/home/cellareye/Development/yolov5/runs/train/batch_1_and_batch_2_images_20_epochs/weights/caging_analysis_cells.names"
)
# CLASS_NAMES_PATH: Final[str] = '/home/cellareye/Development/yolov5/runs/train/fluorescent_images_batch_1_20_epochs/weights/fluorescent_cells.names'
# CLASS_NAMES_PATH: Final[str] = '/home/cellareye/Development/yolov5/runs/train/nuclei_bf_25_epochs_m/weights/nuclei.names'


SAM_MODEL_CHECKPOINTS_PATH: Final[str] = (
    "/home/cellareye/Cellanome/dl-mehdi/SAM/checkpoints"
)

SAM_MODEL_TYPE_TO_CHECKPOINT_MAP: Dict[str, str] = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}

# mapping the YOLO model label map of {0: 'cell', 1: 'bead', 2: 'soma'} from YOLO to a
# Mask R-CNN compatible label map of {1: 'cell', 2: 'bead', 6: 'soma'}
YOLO_TO_SAM_CLASS_IDS_MAP: Dict[int, int] = {0: 1, 1: 2, 2: 3}

DEFAULT_DETECTION_CONFIDENCE: Final[float] = 0.4
# use higher confidence if running microscope image with original 2000x1600 resolution
# DEFAULT_DETECTION_CONFIDENCE: Final[float] = 0.7
# use lower confidence if running on breaboard images with smaller resizing
# DEFAULT_DETECTION_CONFIDENCE: Final[float] = 0.35

DEFAULT_NMS_THRESHOLD: Final[float] = 0.3
INPUT_IMAGE_SIZE: Final[Tuple[int, int]] = (640, 640)  # width x height
# the aspect ratio of the passed image to the model should be within this threshold (percentagewise)
# of the aspect ratio of the input image size specified above
# if the deviation is more than this threshold, the results may not be reliable
INPUT_IMAGE_ASPECT_RATIO_DIFF_DEV_THRESH: Final[float] = 0.1


# Utility functions
# very efficient batch IoU calculation
# needed for combining detection results from different image crops on
# the overlapping parts; we use this function instead of
# torchvision.ops.box_iou(bboxes1, bboxes2) to remove dependency on torch
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


def overlap_batch(
    bboxes1: np.ndarray, bboxes2: np.ndarray, ordered: bool = False
) -> np.ndarray:
    """Given Nx4 & Mx4 ndarrays of bounding boxes, compute pairwise overlaps defined as the intersection over
    the smallest box area (ordered set to False); a small box fully enclosed by a large box has an "overlap" of 1.
    if ordered is set to 1, the overlap is the intersection over the area of the box from bboxes2 set. In this case,
    overlap is close to 1 only if the box from bboxes2 lies inside the box from bboxes1"""

    # expand dims to allow computing pairwise overlap via outerproducts (creates NxM below)
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
    if ordered:
        # use the box area of bboxes2 as the denominator
        smallest_bb_areas = (bboxes2[..., 2] - bboxes2[..., 0]) * (
            bboxes2[..., 3] - bboxes2[..., 1]
        )
    else:
        smallest_bb_areas = (
            np.minimum(
                (bboxes1[..., 2] - bboxes1[..., 0])
                * (bboxes1[..., 3] - bboxes1[..., 1]),
                (bboxes2[..., 2] - bboxes2[..., 0])
                * (bboxes2[..., 3] - bboxes2[..., 1]),
            )
            + 1e-30
        )  # smallest bb area of each paired box NXM
    return (
        inter_areas / smallest_bb_areas
    )  # pairwise intersection divided by smallest box (overlap) NxM


def iou_mask_pair(
    box1: np.ndarray, mask1: np.ndarray, box2: np.ndarray, mask2: np.ndarray
) -> float:
    """
    Given two np.uint8 M1xN1 and M2xN2 numpy arrays (mask1, mask2) for two object masks and
    two (4,) (4-element) integer numpy arrays of the top-left/bottom-right corners of the
    bounding boxes around these objects (box1, box2), the code returns the IoU between the masks of the two objects.
    The passed masks should be defined within the passed bounding boxes, and should have
    values set to 1 for the object. The bounding boxes should be passed in xtl, ytl, xbr, ybr order, e.g.,
    if [xtl, ytl, xbr, ybr] are the passed integer values of the top-left and bottle-right corner of the bounding box
    for an object, the passed masks should be a numpy array of type np.uint8 and size (ybr - ytl, xbr - xtl)
    with values set to 1 for the object.
    """
    # union of the box coordinates, make sure the coordinates are integers
    xtl_1, ytl_1, xbr_1, ybr_1 = box1.astype(int)
    xtl_2, ytl_2, xbr_2, ybr_2 = box2.astype(int)

    xtl: int = min(xtl_1, xtl_2)
    ytl: int = min(ytl_1, ytl_2)
    xbr: int = max(xbr_1, xbr_2)
    ybr: int = max(ybr_1, ybr_2)
    union_mask_1: np.ndarray = np.zeros((ybr - ytl, xbr - xtl), np.uint8)
    union_mask_1[(ytl_1 - ytl) : (ybr_1 - ytl), (xtl_1 - xtl) : (xbr_1 - xtl)] = mask1
    union_mask_2: np.ndarray = np.zeros((ybr - ytl, xbr - xtl), np.uint8)
    union_mask_2[(ytl_2 - ytl) : (ybr_2 - ytl), (xtl_2 - xtl) : (xbr_2 - xtl)] = mask2
    union: int = cv2.bitwise_or(union_mask_1, union_mask_2).sum()
    intersection: int = cv2.bitwise_and(union_mask_1, union_mask_2).sum()

    return intersection / (float(union) + 1e-30)


# a function to return the area of bounding box
def box_area(box: np.array) -> float:
    """
    Args:
        box (numpy array of size (4,) or (4, 1) or (1, 4) or a 4-tuple or a 4-elements list): The box.
    Return the area.
    """
    return (box[3] - box[1]) * (box[2] - box[0])


def show_detections(input_image, predictions, label_map):

    # colors for displaying bounding boxes
    COLORS = [
        (0, 0, 255),
        (255, 0, 0),
        (0, 255, 0),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]

    class_ids = list(label_map.keys())
    if isinstance(input_image, np.ndarray):
        image = input_image.copy()
    else:
        # convert to a numpy array
        image = np.array(input_image)

    # convert to 3-channels
    if len(image.shape) < 3:
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2)

    boxes = predictions["boxes"]
    labels = predictions["labels"]
    masks = predictions["masks"]

    for i in range(len(masks)):
        # the bounding box
        (xtl, ytl, xbr, ybr) = boxes[i]
        if labels[i] not in class_ids:
            # use black for incorrect label
            color = (0, 0, 0)
            text = "Unknown label %s" % labels[i]
        else:
            color = COLORS[labels[i] % len(COLORS)]
            text = label_map[labels[i]]

        color_mask = color * np.repeat(np.expand_dims(masks[i], axis=2), 3, axis=2)
        blended = 0.4 * color_mask
        blended[color_mask == 0] = image[ytl:ybr, xtl:xbr][color_mask == 0]
        blended[color_mask > 0] += 0.6 * image[ytl:ybr, xtl:xbr][color_mask > 0]

        # store the blended ROI in the original image
        image[ytl:ybr, xtl:xbr] = blended.astype(np.uint8)
        # add the bounding box with yellow color
        color = (255, 255, 0)
        cv2.rectangle(image, (xtl, ytl), (xbr, ybr), color, 1)
        cv2.putText(
            image, text, (xtl, ytl + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )
    return image


def get_crop_corners(
    image_width: int,
    image_height: int,
    overlap_in_x: int = 80,
    overlap_in_y: int = 80,
    yolo_input_size: int = 640,
) -> List[List[int]]:
    """
    A function to get the coordinates of the sub-images/crops given the image dimensions, the model input size
    and the desired overlap between the crops in the x and y dimensions.
    Args:
        image_width (int): Input image width.
        image_height (int): Input image height.
        overlap_in_x (int): The desired overlap between the crops in x (this is the minimum, some crops may have more
            than this number of pixels overlap).
        overlap_in_y (int): The desired overlap between the crops in y.
        yolo_input_size (int): The model input size (assuming it is YOLO and accept square images).
    Returns:
        List of 4-tuples or 4-elements integer coordinates of the sub-mages/crops covering the original image.
    """
    # the step size for the starting point of each crop in x and y dimension
    crop_start_step_x: int = yolo_input_size - overlap_in_x
    crop_start_step_y: int = yolo_input_size - overlap_in_y
    crop_corners: List = []

    # overlapping crops
    for x_start in range(0, image_width - overlap_in_x, crop_start_step_x):
        for y_start in range(0, image_height - overlap_in_y, crop_start_step_y):
            # crop coordinates
            xc_tl = x_start
            yc_tl = y_start
            xc_br = x_start + yolo_input_size
            yc_br = y_start + yolo_input_size
            # make sure we always crop the image with the given size
            # if we get to the boundaries, extend the crop
            # size inside the image to always get the same size crop
            # this is not really needed, but help with capturing more
            # annotations toward the low/right parts of the image
            if xc_br > image_width:
                xc_br = image_width
                xc_tl = xc_br - yolo_input_size
            if yc_br > image_height:
                yc_br = image_height
                yc_tl = yc_br - yolo_input_size

            crop_coords = [xc_tl, yc_tl, xc_br, yc_br]
            crop_corners.append(crop_coords)

    return crop_corners


def load_onnx_model_and_metadata(
    weights_path: str, use_onnx_runtime: bool, classnames_path: str = None
):
    # loading the ONNX model
    logging.info(f"Loading YOLOv5 ONNX weights from {weights_path}.")
    try:
        if use_onnx_runtime:
            logging.info("Using ONNX Runtime for running the model.")

            net = onnxruntime.InferenceSession(
                weights_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )

            if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
                logging.warning(
                    "CUDAExecutionProvider is not available in ONNX Runtime! The model will be run on CPU."
                )
        else:
            logging.info(
                "Using OpenCV dnn for running the model. Setting dnn to use CUDA."
            )
            net = cv2.dnn.readNetFromONNX(weights_path)

            try:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            except Exception:
                logging.warning(
                    "Failed to set dnn to use CUDA! Installed OpenCV is not compiled with CUDA support! "
                    "The model will be run on CPU."
                )

    except Exception as ex:
        logging.error(
            f"Failed to initialize YOLOv5 model with CUDA. Likely the path to model ONNX weights "
            f"{weights_path} is incorrect or ONNX Runtime is not installed: {repr(ex)}."
        )
        return None, None

    # read class names and other metadata from ONNX file and create the label map
    try:
        if use_onnx_runtime:
            metadata: Dict[str, str] = net.get_modelmeta().custom_metadata_map
            # convert string values to dicts, or integers or strings
            metadata = {k: eval(v) for k, v in metadata.items()}
            # metadata will be a dictionary with keys as 'resolution', 'release_date', 'model_type', 'model_name', 'model_extra_info', 'names', 'stride'
            # example: {'resolution': 640,
            #           'release_date': '20240415',
            #           'model_type': 'YOLO Detector',
            #           'model_name': 'YOLOv5m',
            #           'model_extra_info': 'V5 Medium',
            #           'names': {0: 'cell', 1: 'bead'},
            #           'stride': 32}
        else:
            # reading the model with OpenCV dnn does not capture the metadata above, try onnx directly
            onnx_model = onnx.load(weights_path)
            metadata: Dict[str, str] = {}
            for field in onnx_model.metadata_props:
                metadata[field.key] = eval(field.value)

    except Exception:
        # try reading from a defualt location
        try:
            # read class names and create the label map
            with open(
                classnames_path, "r"
            ) as f:  # if fails to read then blow with error
                class_names: List[str] = [cname.strip() for cname in f.readlines()]

            label_map: Dict[int, str] = {i: c for i, c in enumerate(class_names)}
            metadata = {"names": label_map}

        except Exception as ex:
            logging.error(
                "Failed to initialize YOLOv5 model with CUDA. The model does not include label map in its metadata and a valid path to classnames file "
                f"{classnames_path} is not provided: {repr(ex)}."
            )
            return net, None

    logging.info("Class names were successfully loaded")

    return net, metadata


def load_sam_model(sam_checkpoints_path: str, model_type: str):
    """A function to load the SAM model
    Args:
        sam_checkpoints_path (str): Path to the SAM model checkpoints.
        model_type (str): The encoder model architecture, can be 'vit_b', 'vit_l' or 'vit_h'.
    Returns the segment_anything.modeling.sam.Sam object
    """
    if model_type not in SAM_MODEL_TYPE_TO_CHECKPOINT_MAP:
        logging.error(
            f"Invalid SAM model_type: {model_type}! Impossible to instantiate the SAM model. Returning None ..."
        )
        return None

    sam_checkpoint: str = os.path.join(
        sam_checkpoints_path, SAM_MODEL_TYPE_TO_CHECKPOINT_MAP[model_type]
    )
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    return sam


# YOLOv5 detector followed by a SAM segmentor class
class Yolov5PlusSamSegmentor:
    def __init__(
        self,
        yolo_weights_path: Optional[str] = YOLO_MODEL_WEIGHTS_PATH,
        sam_checkpoints_path: Optional[str] = SAM_MODEL_CHECKPOINTS_PATH,
        sam_model_type: Optional[str] = "vit_b",
        yolo_to_sam_class_ids_map: Optional[Dict[int, int]] = YOLO_TO_SAM_CLASS_IDS_MAP,
        model_input_size: Tuple[int, int] = INPUT_IMAGE_SIZE,
        confidence: float = DEFAULT_DETECTION_CONFIDENCE,
        nms_threshold: float = DEFAULT_NMS_THRESHOLD,
        use_onnx_runtime: bool = False,
    ):

        self._net = None
        self._weights_path: str = str(yolo_weights_path)
        self._sam_checkpoints_path: str = str(sam_checkpoints_path)
        self._sam_model_type: str = str(sam_model_type)
        self._yolo_to_sam_class_ids_map: Dict[int, int] = yolo_to_sam_class_ids_map
        self._model_input_size: Tuple[int, int] = model_input_size
        self._confidence: float = confidence
        self._nms_threshold: float = nms_threshold
        self._use_onnx_runtime: bool = use_onnx_runtime

        logging.info(
            f"Loading the model with confidence {self._confidence} and"
            f" NMS threshold {self._nms_threshold} ..."
        )

        self._net, self._metadata = load_onnx_model_and_metadata(
            weights_path=self._weights_path,
            use_onnx_runtime=self._use_onnx_runtime,
            classnames_path=DEFAULT_CLASS_NAMES_PATH,
        )

        yolo_label_map: Dict[int, str] = self._metadata["names"]

        # map the YOLO label map to a Mask R-CNN compatible label map for comparison
        self._label_map: Dict[int, str] = {}
        for k, v in yolo_label_map.items():
            # if two class IDs from YOLO are mapped to one class ID, the first name is the list is used for the classname
            if (
                k in self._yolo_to_sam_class_ids_map
                and self._yolo_to_sam_class_ids_map[k] not in self._label_map
            ):
                self._label_map[self._yolo_to_sam_class_ids_map[k]] = v

        self._reverse_label_map: Dict[str, int] = {
            v: k for k, v in self._label_map.items()
        }

        self._device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self._sam = load_sam_model(self._sam_checkpoints_path, self._sam_model_type)

        if self._net is None or self._metadata is None or self._sam is None:
            self._sam_detector = None
            return

        self._sam.to(device=self._device)
        self._sam_detector = SamPredictor(self._sam)

    def detect(self, image: np.ndarray, log_time=False) -> Dict[str, list]:
        """
        The main function to detect the bounding box and masks for objects in the input image.
        """
        if self._net is None:
            logging.error(
                "YOLOv5 CV model has not been initialized. Please initialize the class before detect()."
            )
            out: Dict[str, list] = {
                "boxes": [],
                "scores": [],
                "labels": [],
                "masks": [],
            }
            return out

        if self._sam_detector is None:
            logging.error(
                "SAM model has not been initialized. Please initialize the class before detect()."
            )
            out: Dict[str, list] = {
                "boxes": [],
                "scores": [],
                "labels": [],
                "masks": [],
            }
            return out

        start: float = time.time()

        image_height, image_width = image.shape[:2]

        # check if the aspect ratio of the input image is almost the same as the aspect ratio of
        # the model input size
        # if not, then the input image will be resized without keeping its aspect ratio when it
        # is passed to the model, and this may lead to inaccurate detection
        aspect_ratio_diff: float = (image_width * self._model_input_size[1]) / (
            image_height * self._model_input_size[0]
        ) - 1

        if np.abs(aspect_ratio_diff) > INPUT_IMAGE_ASPECT_RATIO_DIFF_DEV_THRESH:
            logging.warning(
                f"The input image has a different aspect ratio: {image_width / image_height} than the "
                f"YOLOv5 model: {self._model_input_size[0] / self._model_input_size[1]}! "
                f"The detection results may not be accurate"
            )

        # prepare input blob and perform inference; the model expect the image in 3 channel format
        # also, no need for scaling as the range is expected to be in 0-255
        if len(image.shape) < 3:
            # input_image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2)
            input_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            input_image = image

        elap: float = time.time() - start
        check_point: float = time.time()

        if log_time:
            logging.info(f"Pre-processing took {elap:.4f} seconds")

        # extract the SAM model embedding (SAM encoder)
        self._sam_detector.set_image(input_image)

        elap: float = time.time() - check_point
        check_point: float = time.time()
        if log_time:
            logging.info(f"Extracting SAM's embeddings took {elap:.4f} seconds")

        # run YOLOv5 to detect objects' bounding boxes
        if self._use_onnx_runtime:
            # transpose the input image to C x H x W and add batch dimension
            input_image = np.expand_dims(np.transpose(input_image, (2, 0, 1)), axis=0)
            # construct the input dictionary
            ort_inputs: dict = {
                self._net.get_inputs()[0].name: input_image.astype(np.float32) / 255.0
            }
            # run the forward pass to get output
            outputs: np.array = self._net.run(None, ort_inputs)
        else:
            input_blob = cv2.dnn.blobFromImage(
                input_image,
                scalefactor=1.0 / 255,
                size=self._model_input_size,
                mean=(0, 0, 0),
                swapRB=True,
                crop=False,
            )
            # set the input to the model
            self._net.setInput(input_blob)
            # run the forward pass to get output of the output layers
            outputs: np.ndarray = self._net.forward(
                self._net.getUnconnectedOutLayersNames()
            )

        # only one output layer, pick the first element in the tuple then remove the batch dimension
        outputs = outputs[0][0]
        boxes, labels, scores = self._post_process_yolo(
            outputs, (image_width, image_height)
        )

        for i in range(len(boxes)):
            box = boxes[i]
            if (
                box[0] < 0
                or box[1] < 0
                or box[2] > image_width
                or box[3] > image_height
            ):
                print(
                    f"[ERROR] Returned box coodinates from YOLO: {box} are out of the image! This should never happen ..."
                )
                boxes[i] = np.array(
                    [
                        max(0, box[0]),
                        max(0, box[1]),
                        min(box[2], image_width),
                        min(box[3], image_height),
                    ]
                )

        # convert the YOLO labels to a compatible Mask R-CNN labels
        valid_idxs: List[int] = []
        for i in range(len(labels)):
            if labels[i] in self._yolo_to_sam_class_ids_map:
                labels[i] = self._yolo_to_sam_class_ids_map[labels[i]]
                valid_idxs.append(i)

        boxes = np.array([box for i, box in enumerate(boxes) if i in valid_idxs])
        labels = np.array([label for i, label in enumerate(labels) if i in valid_idxs])
        scores = np.array([score for i, score in enumerate(scores) if i in valid_idxs])

        elap: float = time.time() - check_point
        check_point: float = time.time()
        if log_time:
            logging.info(f"YOLO V5 object detection took {elap:.4f} seconds")

        masks: List[np.ndarray] = []
        # extract masks for 100 boxes at a time to make sure we are not running out of GPU memory
        num_boxes_step_size: int = 100
        for i in range(0, len(scores), num_boxes_step_size):
            start_index: int = i
            end_index = min(i + num_boxes_step_size, len(scores))
            input_boxes = torch.tensor(
                boxes[start_index:end_index, :], device=self._device
            )
            transformed_boxes = self._sam_detector.transform.apply_boxes_torch(
                input_boxes, image.shape[:2]
            )
            mask_tensors, iou_predictions, low_res_masks = (
                self._sam_detector.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
            )
            for i, box_tensor in enumerate(input_boxes):
                # confine the mask to the bounding box and move to CPU before converting to numpy arrays
                masks.append(
                    mask_tensors[
                        i,
                        0,
                        box_tensor[1] : box_tensor[3],
                        box_tensor[0] : box_tensor[2],
                    ]
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )

        elap: float = time.time() - check_point
        check_point: float = time.time()
        if log_time:
            logging.info(f"SAM mask prediction took {elap:.4f} seconds")

        elap: float = time.time() - start
        if log_time:
            logging.info(f"YOLOv5 + SAM segmentation took {elap:.4f} seconds")

        return {
            "boxes": [box for box in boxes],
            "labels": list(labels),
            "scores": list(scores),
            "masks": masks,
        }

    def set_confidence(self, confidence):
        self._confidence = confidence

    def set_nms_threshold(self, threshold):
        self._nms_threshold = threshold

    def get_label_map(self):
        return self._label_map

    def get_reverse_label_map(self):
        return self._reverse_label_map

    def get_metadata(self):
        return self._metadata

    def _post_process_yolo(
        self, outputs: np.ndarray, org_image_shape: Tuple[int, int]
    ) -> Tuple[np.array, np.array, np.array]:
        """Post process outputs, discarding unreliable detections & performing NMS"""
        # discard unreliable detections before NMS to reduce NMS computations
        outputs = outputs[np.where(outputs[:, 4] >= min(0.5, self._confidence))[0]]
        # class IDs
        labels: np.ndarray = np.argmax(outputs[:, 5:], axis=1)
        # scores
        scores: np.ndarray = np.max(outputs[:, 5:], axis=1)
        # center (x, y) and width and height of each box
        cx: np.array = outputs[:, 0]
        cy: np.array = outputs[:, 1]
        w: np.array = outputs[:, 2]
        h: np.array = outputs[:, 3]
        # convert the boxes from (cx, cy, w, h) format to (xtl, ytl, w, h) before running NMS
        boxes: np.ndarray = np.vstack(
            [np.maximum(0, cx - w / 2), np.maximum(cy - h / 2, 0), w, h]
        ).T * np.array(
            [
                org_image_shape[0] / self._model_input_size[0],
                org_image_shape[1] / self._model_input_size[1],
            ]
            * 2
        )

        # run NMS per class
        index_list_to_return: List[int] = []
        # we are interested in unique class IDs in the output, hence set(classes)
        unique_label_ids = list(set(labels))
        for label_id in unique_label_ids:
            class_indexes = np.where(labels == label_id)[0]
            valid_det_indexes = cv2.dnn.NMSBoxes(
                boxes[class_indexes],
                scores[class_indexes],
                self._confidence,
                self._nms_threshold,
            )

            index_list_to_return += list(class_indexes[valid_det_indexes])

        # NMS removes repeated detections (may return empty ndarrays)
        boxes = boxes[index_list_to_return, :]
        scores = scores[index_list_to_return]
        labels = labels[index_list_to_return]

        # convert from xtl, ytl, w, h) to (xtl, ytl, xbr, ybr) for each box
        boxes[:, 2] = np.minimum(boxes[:, 0] + boxes[:, 2], org_image_shape[0])
        boxes[:, 3] = np.minimum(boxes[:, 1] + boxes[:, 3], org_image_shape[1])
        # convert the coordinates to int
        boxes = boxes.astype(int)

        return boxes, labels, scores

    def detect_by_cropping(
        self,
        img: Union[Image.Image, np.ndarray],
        crop_corners: List[List[int]],
        nms_threshold_for_combining_crop_results: float = 0.15,
        classnames_to_return: Optional[List[str]] = None,
        log_time=False,
    ) -> Dict[str, List]:
        """
        A function to apply the model on a high resolution image. If the
        image is high resolution with many objects, after resizing the image
        to match the models input sizes, the objects may become too small for
        reliable detection. This function evaluate an image by first cropping
        the image into smaller and potentially overlapping sub-images (as
        specified by cropCorners), running the detector on each sub-image and
        then combining the detections over multiple overlapping sub-images
        by applying NMS
        Args:
            img (PIL.Image or numpy array): Input image, should have 8 bits per channel
                bit-depth (np.uint8 in case of a numpy array).
            crop_corners (list of 4-tuples (x1, y1, x2, y2)): Each element of
                this list specifies a cropped sub-image of the input image with
                top-left corner (x1, y1) and bottom-right corner (x2, y2). All
                the coordinates should be with respect to input image sizes.
                The input image is divided into len(crop_corners) sub-images
                before running the model on each.
            nms_threshold_for_combining_crop_results (float): NMS threshold to be used
                for combining detections of cropped sub-images over the overlapping areas.
            classnames_to_return (optional list of string): The class names to return. If
                None is passed (default), all classes detected will be returned.
            log_time (bool): A flag to print the runtime of the function.
        Returns:
            A dictionary with keys and values as below:
                "boxes": List of 4-tuples or 4-element lists for the detected objects' bounding boxes in
                    xtl, ytl, xbr, ybr format/order.
                "labels": List of integer class IDs for the detected objects.
                "scores": List of float detection scores, after applying the threshold self._confidence.
                "masks": List of masks for the detected objects. Each mask is a numpy array of the same size
                    as the bounding box with width and height (xbr - xtl, ybr - ytl).
        """

        start = time.time()

        # invalid output
        invalid_out: dict = {"boxes": [], "scores": [], "labels": [], "masks": []}

        if len(crop_corners) == 0:
            logging.error(
                "No crop corners are provided for running YOLOv5 model on sub-images. "
                "Returning no detections ..."
            )
            return invalid_out

        if classnames_to_return is not None:
            class_ids_to_return: List[int] = []
            for class_name in classnames_to_return:
                if class_name not in self._reverse_label_map:
                    logging.warning(
                        f"Classname '{class_name}' is not included in the model label map! Skipping this name ..."
                    )
                else:
                    class_ids_to_return.append(self._reverse_label_map[class_name])

            if len(class_ids_to_return) == 0:
                # nothing to return
                logging.warning(
                    f"No valid classnames was provided in classnames_to_return: {classnames_to_return}!"
                    f"Returning no detections"
                )
                return invalid_out
        else:
            # this list will not be used and is not really needed, define it just in case
            class_ids_to_return: List[int] = list(self._label_map.keys())

        if isinstance(img, Image.Image):
            W, H = img.size
        else:
            H, W = img.shape[:2]

        # check if all the crop sub-images are of the same size,
        # if not, make them equal size for batch processing
        crop_widths: List[int] = [min(c[2], W) - max(c[0], 0) for c in crop_corners]
        crop_heights: List[int] = [min(c[3], H) - max(c[1], 0) for c in crop_corners]

        if min(crop_widths) <= 0 or min(crop_heights) <= 0:
            logging.error(
                "Incorrect corners are provided for running Mask RCNN model on sub-images. "
                "Returning no detections ..."
            )
            return invalid_out

        crop_width: int = max(crop_widths)
        crop_height: int = max(crop_heights)

        # combine the results, filter them based on the score,
        # and update the coordinates of the bounding boxes
        # for applying NMS later
        results: Dict = {
            "scores": [],
            "boxes": [],
            "labels": [],
            "masks": [],
            "features": [],
        }

        # a list to keep track of cropped sub-images with at least one object detection
        crop_ids_with_detection: List[int] = []

        for crop_id, corners in enumerate(crop_corners):
            (x1c, y1c, x2c, y2c) = corners
            # enlarge the crop if necessary to make all the same size
            x2c = x1c + crop_width
            y2c = y1c + crop_height

            # crop the image and run the model
            if isinstance(img, Image.Image):
                cropped_image = img.crop((x1c, y1c, x2c, y2c))
            else:
                cropped_image = img[y1c:y2c, x1c:x2c]

            preds = self.detect(cropped_image)

            boxes: np.ndarray = np.array(preds["boxes"])
            labels: np.ndarray = np.array(preds["labels"])
            scores: np.ndarray = np.array(preds["scores"])
            # we leave the masks as a list of mask numpy arrays as each mask has different sizes below
            masks: List[np.ndarray] = preds["masks"]

            # combine the detection results
            if len(scores) == 0:
                continue

            # find the bounding boxes close to the boundaries of the cropped image
            # these boxes most probably are truncated (because they are close to the boundary)
            # for a properly designed image crops, the overlapping section (in x or y dimensions)
            # between two adjacent crops is larger than the largest object (in each dimension)
            # hence an object can only cross one boundary of an overlapping part and will
            # definitely lie completely in another cropped image
            # modify the score for these detected boxes (assign the minimum score of self._confidence)
            # to give them lower priority during NMS when the results in the overlapping parts
            # of cropped images are combined

            scores[
                (boxes[:, 0] < 4)
                | (boxes[:, 1] < 4)
                | (boxes[:, 2] > crop_width - 4)
                | (boxes[:, 3] > crop_height - 4)
            ] = self._confidence

            crop_ids_with_detection.append(crop_id)
            results["scores"].append(scores)
            results["boxes"].append(boxes + np.array([x1c, y1c, x1c, y1c], dtype=int))
            results["labels"].append(labels)
            results["masks"].append(masks)

        # no object detected, return
        if len(crop_ids_with_detection) == 0:
            return invalid_out

        # list to contain the detections
        boxes: list = []
        labels: list = []
        scores: list = []
        masks: list = []
        appearance_features: list = []

        # now compare the detection results of one crop with the detections in the
        # rest of the image to identify objects that are uniquely detected in the
        # crop and should be kept
        # this part is needed to pick one object in the overlapping crop areas when
        # detected by the detector in multiple crops
        # note we need to compare the detections in one crop only with overlapping
        # crops; but here we are doing it for all for simplicity of implementation

        # to decide which object to keep when detected in multiple crops, we compute
        # the IoU (for objects of the same class) between the crop under consideration and the rest
        # then we keep objects in the crop under consideration that have
        # IoU <= nms_threshold_for_combining_crop_results with objects detected in the rest
        # we also keep objects with IoU > nms_threshold_for_combining_crop_results, if the detection
        # score for the object in the crop is higher than the objects in the other crops
        for idx, crop_id in enumerate(crop_ids_with_detection):
            crop_labels: np.ndarray = results["labels"][idx]
            crop_scores: np.ndarray = results["scores"][idx]
            crop_boxes: np.ndarray = results["boxes"][idx]
            crop_masks: List[np.ndarray] = results["masks"][idx]

            num_detections_in_rest = 0
            for i in range(len(crop_ids_with_detection)):
                if i != idx:
                    num_detections_in_rest += len(results["labels"][i])

            if num_detections_in_rest == 0:
                # keep all detections in this crop
                for i, label in enumerate(crop_labels):
                    boxes.append(crop_boxes[i])
                    labels.append(label)
                    scores.append(crop_scores[i])
                    masks.append(crop_masks[i])
                continue

            # labels for detections in other cropped sub-images
            rest_labels: np.array = np.hstack(
                [
                    results["labels"][i]
                    for i in range(len(crop_ids_with_detection))
                    if i != idx
                ]
            )
            # scores for detections in other cropped sub-images
            rest_scores: np.array = np.hstack(
                [
                    results["scores"][i]
                    for i in range(len(crop_ids_with_detection))
                    if i != idx
                ]
            )
            # boxes for detections in other cropped sub-images
            rest_boxes: np.array = np.vstack(
                [
                    results["boxes"][i]
                    for i in range(len(crop_ids_with_detection))
                    if i != idx
                ]
            )

            for label in set(list(crop_labels) + list(rest_labels)):
                # the indexes of detections of the same label in each crop and rest set
                crop_class_idxs: np.array = np.where(crop_labels == label)
                rest_class_idxs: np.array = np.where(rest_labels == label)

                if len(crop_class_idxs[0]) == 0:
                    continue

                if len(rest_class_idxs[0]) == 0:
                    # keep all detections for this label in this crop
                    idx_to_keep: np.array = np.array(
                        [i for i in range(len(crop_class_idxs[0]))], dtype=int
                    )
                else:
                    # compute the IoU matrix, use torchvision implementation for efficiency
                    iou_matrix: np.array = iou_batch(
                        crop_boxes[crop_class_idxs], rest_boxes[rest_class_idxs]
                    )

                    # keep the bounding boxes from the crop that have
                    # 1) IoU <= nms_threshold_for_combining_crop_results with the boxes in the rest
                    #    of the crops, or
                    # 2) IoU > nms_threshold_for_combining_crop_results and the detection score for
                    #    the box in the crop is higher than the detection scores for the boxes in the rest

                    # IoU <= nms_threshold_for_combining_crop_results
                    idx_to_keep = np.where(
                        np.max(iou_matrix, axis=1)
                        <= nms_threshold_for_combining_crop_results
                    )[0]

                for i in range(len(crop_class_idxs[0])):
                    # detection score of this object
                    crop_det_score = crop_scores[crop_class_idxs[0][i]]
                    if i in idx_to_keep:
                        boxes.append(crop_boxes[crop_class_idxs[0][i]])
                        labels.append(crop_labels[crop_class_idxs[0][i]])
                        scores.append(crop_det_score)
                        masks.append(crop_masks[crop_class_idxs[0][i]])
                    else:
                        # there is some boxes in the rest of the crops with IoU more than the threshold
                        # find the maximum detection scores among the objects with IoU more than the threshold
                        # also find the area of the bounding box for that detection (needed to break a tie in case
                        # scores are equal)
                        high_iou_idxs = np.where(
                            iou_matrix[i] > nms_threshold_for_combining_crop_results
                        )[0]
                        scores_to_check = [
                            rest_scores[rest_class_idxs[0][j]] for j in high_iou_idxs
                        ]
                        max_index = np.argmax(scores_to_check)
                        rest_det_score = scores_to_check[max_index]

                        # areas of the matching boxes
                        crop_box_area: float = box_area(
                            crop_boxes[crop_class_idxs[0][i]]
                        )
                        rest_box_area: float = box_area(
                            rest_boxes[rest_class_idxs[0][high_iou_idxs[max_index]]]
                        )

                        # in case of a tie, pick the box with a larger area
                        # a tie can happen specially when the objects are both near a common boundary (e.g., a side of
                        # the image) of the crops (we reduce the scores for both to the threshold score and they become
                        # equal)
                        if crop_det_score > rest_det_score or (
                            crop_det_score == rest_det_score
                            and crop_box_area >= rest_box_area
                        ):
                            # keep this object as it has the highest score among all
                            boxes.append(crop_boxes[crop_class_idxs[0][i]])
                            labels.append(crop_labels[crop_class_idxs[0][i]])
                            scores.append(crop_det_score)
                            masks.append(crop_masks[crop_class_idxs[0][i]])
                            if crop_det_score == rest_det_score:
                                # this is added to break the tie if both areas are equal
                                # so we will not add the same box twice when considering
                                # in another crop
                                results["scores"][idx][crop_class_idxs[0][i]] += 1e-5

        # filter the detections
        if classnames_to_return is not None:
            detection_ids: List[int] = []
            for idx, label in enumerate(labels):
                if label not in class_ids_to_return:
                    continue
                detection_ids.append(idx)

            boxes: list = [boxes[i] for i in detection_ids]
            labels: list = [labels[i] for i in detection_ids]
            scores: list = [scores[i] for i in detection_ids]
            masks: list = [masks[i] for i in detection_ids]

        elap: float = time.time() - start
        if log_time:
            logging.info(
                "Mask R-CNN instance segmentation after cropping the image to "
                "{} sub-images took {:.4f} seconds in OpenCV".format(
                    len(crop_corners), elap
                )
            )

        out: dict = {
            "boxes": [
                [box[0], box[1], box[2], box[3]] for box in boxes
            ],  # convert to list
            "scores": scores,
            "labels": labels,
            "masks": masks,
        }

        return out


detector = Yolov5PlusSamSegmentor(use_onnx_runtime=False)


RESIZE: Final[Dict[Tuple[int, int], Tuple[int, int]]] = {
    (2000, 1600): (1000, 800),
    (4512, 4512): (2440, 2440),
}
# A dictionary with keys as the input (original) image size (width, height)
# tuple and values as the list of coordinates (xtl, ytl, xbr, ybr) of sub-images/crops
# to run YOLOv5 and SAM model on each
# note that the crop coordinates are with respect to resized image dimensions specified above
CROP_CORNERS: Final[Dict[Tuple[int, int], List[List[int]]]] = {
    (2000, 1600): [
        [0, 0, 640, 640],
        [0, 160, 640, 800],
        [360, 0, 1000, 640],
        [360, 160, 1000, 800],
    ],
    (4512, 4512): [
        [0, 0, 640, 640],
        [0, 600, 640, 1240],
        [0, 1200, 640, 1840],
        [0, 1800, 640, 2440],
        [600, 0, 1240, 640],
        [600, 600, 1240, 1240],
        [600, 1200, 1240, 1840],
        [600, 1800, 1240, 2440],
        [1200, 0, 1840, 640],
        [1200, 600, 1840, 1240],
        [1200, 1200, 1840, 1840],
        [1200, 1800, 1840, 2440],
        [1800, 0, 2440, 640],
        [1800, 600, 2440, 1240],
        [1800, 1200, 2440, 1840],
        [1800, 1800, 2440, 2440],
    ],
}


# threshold for post-processing cells and remove the ones consisting of multiple smaller cells
OVER_LAP_THRESHOLD: Final[float] = 0.75


def run_yolo_plus_sam(
    input_image: np.ndarray,
    normalize_image: bool = False,
    bit_depth: int = 8,
    classnames_mapping_dict=None,
    post_process_class_names: List[str] = list(detector.get_label_map().values()),
    plot_results: bool = False,
) -> Tuple[Dict[str, list], float, Optional[np.ndarray]]:
    # make a copy to not modify the input image
    img = input_image.copy()

    if len(img.shape) > 2:
        logging.warning(
            "Warning Mask R-CNN model may suffer loss in precision due to conversion from RGB to grayscale"
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = (255 * img.astype(float) / (2**bit_depth - 1)).astype(np.uint8)

    if normalize_image:
        img = cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    image_height, image_width = img.shape[:2]

    if (image_width, image_height) not in RESIZE:
        logging.error(
            "The input image size {} is not supported! Returning no cells!".format(
                image_width,
            )
        )
        out = {
            "boxes": np.zeros((0, 4), dtype=int),
            "labels": np.zeros((0,), dtype=int),
            "scores": np.zeros((0,), dtype=float),
            "masks": [],
        }
        if plot_results:
            return (out, 0, np.zeros((image_height, image_width), dtype=np.uint8))
        else:
            return (out, 0)

    # we keep the aspect ratio in RESIZE dictionary, scale_factor is the same for both dimensions
    scale_factor: float = image_width / RESIZE[(image_width, image_height)][0]
    resized_width, resized_height = RESIZE[(image_width, image_height)]

    if scale_factor != 1:
        resized_img: np.ndarray = cv2.resize(
            img, (resized_width, resized_height), interpolation=cv2.INTER_AREA
        )
    else:
        resized_img: np.ndarray = img

    st = time.time()

    crop_corners: List[List[int]] = CROP_CORNERS[(image_width, image_height)]
    out: Dict[str, list] = detector.detect_by_cropping(
        img=resized_img, crop_corners=crop_corners
    )

    if scale_factor != 1:
        # scale the detections back to original image resolution
        out["boxes"] = (scale_factor * np.array(out["boxes"])).astype(int)
        # convert to a list to be consistent with the rest
        out["boxes"] = [box for box in out["boxes"]]
    else:
        out["boxes"] = [np.array(box) for box in out["boxes"]]

    for idx in range(len(out["boxes"])):
        if scale_factor != 1:
            xtl, ytl, xbr, ybr = out["boxes"][idx]
            # note that mask here is NOT a probability mask and interpolation does not have to be nearest neighbor
            out["masks"][idx] = cv2.resize(
                out["masks"][idx],
                (xbr - xtl, ybr - ytl),
                interpolation=cv2.INTER_NEAREST,
            )

    if classnames_mapping_dict is not None:
        classnames_to_exclude: List[str] = [
            name
            for name, mapped_name in classnames_mapping_dict.items()
            if mapped_name == "bg"
        ]
        class_ids_to_exclude: List[int] = [
            detector._reverse_label_map[name] for name in classnames_to_exclude
        ]
        class_ids_mapping_dict = {
            detector._reverse_label_map[name]: detector._reverse_label_map[mapped_name]
            for name, mapped_name in classnames_mapping_dict.items()
            if mapped_name != "bg"
        }

        labels: List[int] = []
        idxs_to_keep: List[int] = []

        for idx, label in enumerate(out["labels"]):
            if label in class_ids_to_exclude:
                continue
            if label in class_ids_mapping_dict:
                labels.append(class_ids_mapping_dict[label])
            else:
                labels.append(label)
            idxs_to_keep.append(idx)

        out["boxes"] = [
            box for idx, box in enumerate(out["boxes"]) if idx in idxs_to_keep
        ]
        out["labels"] = labels
        out["scores"] = [
            score for idx, score in enumerate(out["scores"]) if idx in idxs_to_keep
        ]
        out["masks"] = [
            mask for idx, mask in enumerate(out["masks"]) if idx in idxs_to_keep
        ]

    # post-process the results
    # in the following, "larger" objects that consist of a number of already detected smaller objects of the same type are invalidated
    # this can happen mainly for 'cell', 'nucleus' and 'cell-adhered'/'cytoplasm' classes
    # list of indexes of objects for each class name to be included in post processing
    post_process_class_idxs: Dict[str, List[int]] = {}
    # list of bounding boxes for each class name to be included in post processing
    post_process_class_boxes: Dict[str, List[np.ndarray]] = {}
    for i, box in enumerate(out["boxes"]):
        for class_name in post_process_class_names:
            if (
                class_name in detector._reverse_label_map
                and out["labels"][i] == detector._reverse_label_map[class_name]
            ):
                if class_name in post_process_class_idxs:
                    post_process_class_idxs[class_name].append(i)
                    post_process_class_boxes[class_name].append(box)
                else:
                    post_process_class_idxs[class_name] = [i]
                    post_process_class_boxes[class_name] = [box]

    # list of detection indexes to be excluded (this is with respect to all detected objects and not only the class under consideration)
    obj_idxs_to_remove: List[int] = []
    for key in post_process_class_boxes:
        # convert to a numpy array
        post_process_class_boxes[key]: np.ndarray = np.array(
            post_process_class_boxes[key]
        )

        if len(post_process_class_idxs[key]) == 0:
            continue

        overlap: np.ndarray = overlap_batch(
            post_process_class_boxes[key], post_process_class_boxes[key], True
        )
        # remove diagonal elements (as each box has a complete overlap with itself)
        overlap = overlap - np.eye(len(post_process_class_boxes[key]))
        # index of larger objects (row indexes) covering some smaller already detected cells (column index)
        # by more than OVER_LAP_THRESHOLD
        # these smaller objects are most probably redundant objects
        covering_obj_idxs, covered_obj_idxs = np.where(overlap > OVER_LAP_THRESHOLD)
        # now double-check the coverage using the masks
        for i, j in zip(covering_obj_idxs, covered_obj_idxs):
            large_obj_index: int = post_process_class_idxs[key][i]
            small_obj_index: int = post_process_class_idxs[key][j]
            # larger box coordinates
            xl1, yl1, xl2, yl2 = out["boxes"][large_obj_index]
            # smaller box coordinates
            xs1, ys1, xs2, ys2 = out["boxes"][small_obj_index]
            # union of the two boxes
            x1: int = min(xl1, xs1)
            y1: int = min(yl1, ys1)
            x2: int = max(xl2, xs2)
            y2: int = max(yl2, ys2)
            large_obj_mask: np.ndarray = np.zeros((y2 - y1, x2 - x1), np.uint8)
            large_obj_mask[(yl1 - y1) : (yl2 - y1), (xl1 - x1) : (xl2 - x1)] = out[
                "masks"
            ][large_obj_index]
            small_obj_mask: np.ndarray = np.zeros((y2 - y1, x2 - x1), np.uint8)
            small_obj_mask[(ys1 - y1) : (ys2 - y1), (xs1 - x1) : (xs2 - x1)] = out[
                "masks"
            ][small_obj_index]

            if np.sum(small_obj_mask * large_obj_mask) > OVER_LAP_THRESHOLD * np.sum(
                small_obj_mask
            ):
                # add row index i to the list of object indexes to be removed
                if small_obj_index not in obj_idxs_to_remove:
                    obj_idxs_to_remove.append(small_obj_index)

    if len(obj_idxs_to_remove) > 0:
        out["boxes"] = [
            box for i, box in enumerate(out["boxes"]) if i not in obj_idxs_to_remove
        ]
        out["labels"] = [
            label
            for i, label in enumerate(out["labels"])
            if i not in obj_idxs_to_remove
        ]
        out["scores"] = [
            score
            for i, score in enumerate(out["scores"])
            if i not in obj_idxs_to_remove
        ]
        out["masks"] = [
            mask for i, mask in enumerate(out["masks"]) if i not in obj_idxs_to_remove
        ]

    et = time.time()

    if plot_results:
        return out, et - st, show_detections(img, out, detector.get_label_map())

    return out, et - st
