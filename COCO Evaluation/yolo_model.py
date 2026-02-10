import cv2
import numpy as np
import time
import logging
import onnxruntime
import onnx
from typing import Tuple, List, Final, Optional, Dict

# MODEL_WEIGHTS_PATH: Final[str] = '/home/cellareye/Development/yolov5/runs/train/batch_1_batch_2_batch_3_images_20_epochs_m/weights/batch_1_batch_2_batch_3_images_20_epochs_m.onnx'
# MODEL_WEIGHTS_PATH: Final[str] = '/home/cellareye/Development/yolov5/runs/train/batch_1_batch_2_batch_3_images_20_epochs_m/weights/batch_1_4_oof_images_20_epochs_m.onnx'
# MODEL_WEIGHTS_PATH: Final[str] = '/home/cellareye/Development/yolov5/runs/train/sets_1_2_3_8_to_26_10x_bf_10_epochs_m/weights/20240511_sets_1_2_3_8_to_26_10x_bf_10_epochs_m.onnx'
MODEL_WEIGHTS_PATH: Final[str] = '/home/cellareye/Development/yolov5/runs/train/sets_1_2_3_8_to_27_10x_bf_10_epochs_m/weights/20240624_sets_1_2_3_8_to_27_10x_bf_10_epochs_m.onnx'
# MODEL_WEIGHTS_PATH: Final[str] = '/home/cellareye/Development/yolov5/runs/train/sets_1_to_5_4x_bf_10_epochs_m/weights/20240611_sets_1_to_5_4x_bf_10_epochs_m.onnx'
# MODEL_WEIGHTS_PATH: Final[str] = '/home/cellareye/Development/darknet/backup/nk92_cells_beads_10000_d.onnx'
# MODEL_WEIGHTS_PATH: Final[str] = '/home/cellareye/Development/yolov5/runs/train/fluorescent_images_batch_1_20_epochs_m/weights/fluorescent_images_batch_1_20_epochs_m.onnx'
# MODEL_WEIGHTS_PATH: Final[str] = '/home/cellareye/Development/yolov5/runs/train/nuclei_bf_25_epochs_m/weights/nuclei_bf_25_epochs_m.onnx'
DEFAULT_CLASS_NAMES_PATH: Final[str] = '/home/cellareye/Development/yolov5/runs/train/batch_1_and_batch_2_images_20_epochs/weights/caging_analysis_cells.names'
# CLASS_NAMES_PATH: Final[str] = '/home/cellareye/Development/yolov5/runs/train/fluorescent_images_batch_1_20_epochs/weights/fluorescent_cells.names'
# CLASS_NAMES_PATH: Final[str] = '/home/cellareye/Development/yolov5/runs/train/nuclei_bf_25_epochs_m/weights/nuclei.names'


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


# a function to return the area of bounding box
def box_area(box: np.array) -> float:
    """
    Args:
        box (numpy array of size (4,) or (4, 1) or (1, 4) or a 4-tuple or a 4-elements list): The box.
    Return the area.
    """
    return (box[3] - box[1]) * (box[2] - box[0])


# checks on the results
def show_detections(input_image, boxes, labels, label_map):
    # colors for displaying bounding boxes
    COLORS = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 0),
    ]

    class_ids = list(label_map.keys())
    if len(input_image.shape) < 3:
        image = np.repeat(np.expand_dims(input_image, axis=2), 3, axis=2)
    else:
        image = input_image.copy()

    H, W = image.shape[:2]
    # add the bounding boxes to the image
    for i, box in enumerate(boxes):
        (xtl, ytl, xbr, ybr) = box
        if labels[i] not in class_ids:
            # use black for incorrect label
            color = (0, 0, 0)
            text = "Unknown label %s" % labels[i]
        else:
            color = COLORS[labels[i] % len(COLORS)]
            text = label_map[labels[i]]

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

def load_onnx_model_and_metadata(weights_path: str, use_onnx_runtime: bool, classnames_path: str=None):
    # loading the ONNX model
    logging.info(
        f"Loading YOLOv5 ONNX weights from {weights_path}."
    )
    try:
        if use_onnx_runtime:
            logging.info(
                "Using ONNX Runtime for running the model."
            )

            net = onnxruntime.InferenceSession(weights_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                
            if 'CUDAExecutionProvider' not in onnxruntime.get_available_providers():
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
            with open(classnames_path, 'r') as f:  # if fails to read then blow with error
                class_names: List[str] = [cname.strip() for cname in f.readlines()]

            label_map: Dict[int, str] = {i: c for i, c in enumerate(class_names)}
            metadata = {'names': label_map}

        except Exception as ex:
            logging.error(
                "Failed to initialize YOLOv5 model with CUDA. The model does not include label map in its metadata and a valid path to classnames file "
                    f"{classnames_path} is not provided: {repr(ex)}."
                )
            return net, None
                

    logging.info("Class names were successfully loaded")
    
    return net, metadata


# YOLOv5 detector class
class Yolov5ObjectDetector:
    def __init__(
            self,
            weights_path: Optional[str] = MODEL_WEIGHTS_PATH,
            model_input_size: Tuple[int, int] = INPUT_IMAGE_SIZE,
            confidence: float = DEFAULT_DETECTION_CONFIDENCE,
            nms_threshold: float = DEFAULT_NMS_THRESHOLD,
            use_onnx_runtime: bool = False
    ):

        self._net = None
        self._weights_path: str = str(weights_path)
        self._model_input_size: Tuple[int, int] = model_input_size
        self._confidence: float = confidence
        self._nms_threshold: float = nms_threshold
        self._use_onnx_runtime: bool = use_onnx_runtime

        logging.info(
            f"Loading the model with confidence {self._confidence} and"
            f" NMS threshold {self._nms_threshold} ..."
        )
        
        self._net, self._metadata = load_onnx_model_and_metadata(weights_path=self._weights_path, 
                                                                 use_onnx_runtime=self._use_onnx_runtime, 
                                                                 classnames_path=DEFAULT_CLASS_NAMES_PATH)
        if self._net is None or self._metadata is None:
            return
        
        self._label_map: Dict[int, str] = self._metadata['names']
        self._reverse_label_map: Dict[str, int] = {v: k for k, v in self._label_map.items()}

    def detect(
            self, image: np.ndarray, log_time=False
    ) -> Tuple[np.array, np.array, np.array]:
        """
        The main function to detect the bounding box and masks for persons in the input image.
        """
        if self._net is None:
            logging.error(
                "YOLOv5 CV model has not been initialized. Please initialize the class before detect()."
            )
            return (
                np.zeros((0, 4), dtype=int),
                np.zeros((0,), dtype=int),
                np.zeros((0,), dtype=float),
            )

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
                f"model: {self._model_input_size[0] / self._model_input_size[1]}! "
                f"The results may not be accurate"
            )

        # prepare input blob and perform inference; the model expect the image in 3 channel format
        # also, no need for scaling as the range is expected to be in 0-255
        if len(image.shape) < 3:
            input_image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2)
        else:
            input_image = image

        if self._use_onnx_runtime:
            # transpose the input image to C x H x W and add batch dimension
            input_image = np.expand_dims(np.transpose(input_image, (2, 0, 1)), axis=0)
            # construct the input dictionary
            ort_inputs: dict = {self._net.get_inputs()[0].name: input_image.astype(np.float32) / 255.0}
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
        boxes, labels, scores = self._post_process(outputs, (image_width, image_height))

        elap: float = time.time() - start
        if log_time:
            logging.info(f"YOLO V5 object detection took {elap:.4f} seconds")

        return boxes, labels, scores

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
        
    def _post_process(
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
        boxes: np.ndarray = np.vstack([cx - w / 2, cy - h / 2, w, h]).T * np.array(
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
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        # convert the coordinates to int
        boxes = boxes.astype(int)

        return boxes, labels, scores

    def detect_by_cropping(
        self,
        image: np.ndarray,
        crop_corners: List[List[int]],
        nms_threshold_for_combining_crop_results: float = 0.15,
        only_report_cells: bool = False,
        log_time=False,
    ) -> Tuple[np.array, np.array, np.array]:

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
            image (numpy array): Input image; it should be an OpenCV np.uint8 numpy
                array either in BGR format or Grayscale (output of cv2.imread).
            crop_corners (list of 4-tuples (x1, y1, x2, y2)): Each element of
                this list specifies a cropped sub-image of the input image with
                top-left corner (x1, y1) and bottom-right corner (x2, y2). All
                the coordinates should be with respect to input image sizes.
                The input image is divided into len(crop_corners) sub-images
                before running the model on each.
            nms_threshold_for_combining_crop_results (float): NMS threshold to be used
                for combining detections of cropped sub-images over the overlapping areas.
            only_report_cells (bool): If set to True, the detector only report objects
                with classname 'cell' (should be included in self._label_map and classnames).
            log_time(bool): A flag to print the runtime of the function.
        Returns:
            numpy array of bounding boxes for detected objects; each bounding box
                is represented as (x1, y1, x2, y2) where (x1, y1) and (x2, y2)
                are the coordinates of the top-left and bottom-right corners
                of the bounding box, respectively.
            numpy array of labels for detected objects; the label index to class name
                mapping is specified by self.labelMap.
            numpy of detection scores for detected objects.
        """

        start = time.time()
        if len(crop_corners) == 0:
            logging.error(
                "No crop corners are provided for running YOLOv5 model on sub-images. "
                "Returning no detections"
            )
            return (
                np.zeros((0, 4), dtype=int),
                np.zeros((0,), dtype=int),
                np.zeros((0,), dtype=float),
            )

        if only_report_cells and "cell" not in self._reverse_label_map:
            logging.error(
                "'cell' classname is not included in the model classnames. Returning no detections"
            )
            return (
                np.zeros((0, 4), dtype=int),
                np.zeros((0,), dtype=int),
                np.zeros((0,), dtype=float),
            )

        H, W = image.shape[:2]

        # check if all the crop sub-images are of the same size,
        # if not, make them equal size for batch processing
        crop_widths: List[int] = [min(c[2], W) - max(c[0], 0) for c in crop_corners]
        crop_heights: List[int] = [min(c[3], H) - max(c[1], 0) for c in crop_corners]

        if min(crop_widths) <= 0 or min(crop_heights) <= 0:
            logging.error(
                "Incorrect corners are provided for running YOLOv5 model on sub-images. "
                "Returning no detections"
            )
            return (
                np.zeros((0, 4), dtype=int),
                np.zeros((0,), dtype=int),
                np.zeros((0,), dtype=float),
            )

        crop_width: int = max(crop_widths)
        crop_height: int = max(crop_heights)

        # combine the results, filter them based on the score,
        # and update the coordinates of the bounding boxes
        # for applying NMS later
        results: Dict = {"scores": [], "boxes": [], "labels": []}

        # a list to keep track of cropped sub-images with at least one object detection
        crop_ids_with_detection: List[int] = []

        for crop_id, corners in enumerate(crop_corners):
            (x1c, y1c, x2c, y2c) = corners
            # enlarge the crop if necessary to make all the same size
            x2c = x1c + crop_width
            y2c = y1c + crop_height

            # crop the image and run the model
            cropped_image = image[y1c:y2c, x1c:x2c]
            boxes, labels, scores = self.detect(cropped_image)

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

        # no object detected, return
        if len(crop_ids_with_detection) == 0:
            return (
                np.zeros((0, 4), dtype=int),
                np.zeros((0,), dtype=int),
                np.zeros((0,), dtype=float),
            )

        # list to contain the detections
        boxes, labels, scores = [], [], []

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

            crop_labels: np.array = results["labels"][idx]
            crop_scores: np.array = results["scores"][idx]
            crop_boxes: np.array = results["boxes"][idx]

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
                continue

            # labels  for detections in other cropped sub-images
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
                        crop_box_area: int = box_area(crop_boxes[crop_class_idxs[0][i]])
                        rest_box_area: int = box_area(
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
                            if crop_det_score == rest_det_score:
                                # this is added to break the tie if both areas are equal
                                # so we will not add the same box twice when considering
                                # in another crop
                                results["scores"][idx][crop_class_idxs[0][i]] += 1e-5

        # change the coordinates from numpy array for each box to a list
        boxes = np.array(boxes)
        labels = np.array(labels)
        scores = np.array(scores)

        if only_report_cells:
            cell_class_id: int = self._reverse_label_map["cell"]
            cell_detection_ids: List[int] = []
            for idx, label in enumerate(labels):
                if label != cell_class_id:
                    continue
                cell_detection_ids.append(idx)

            boxes = boxes[cell_detection_ids]
            labels = labels[cell_detection_ids]
            scores = scores[cell_detection_ids]

        elap = time.time() - start
        if log_time:
            logging.info(
                "YOLOv5 object detection after cropping the image to "
                "{} sub-images took {:.4f} seconds in OpenCV".format(
                    len(crop_corners), elap
                )
            )
        return boxes, labels, scores
    
    
detector = Yolov5ObjectDetector(use_onnx_runtime=False)


RESIZE: Final[Dict[Tuple[int, int, str], Tuple[int, int]]] = {
    (2000, 1600, "10x"): (1000, 800),
    (4512, 4512, "10x"): (2440, 2440),
    (4512, 4512, "4x"): (4512, 4512),
}
# A dictionary with keys as the input (original) image size (width, height, magnification)
# tuple and values as the list of coordinates (xtl, ytl, xbr, ybr) of sub-images/crops
# to run YOLOv5 on each
# note that the crop coordinates are with respect to resized image dimensions specified above
CROP_CORNERS: Final[Dict[Tuple[int, int, str], List[List[int]]]] = {
    (2000, 1600, "10x"): [
        [0, 0, 640, 640],
        [0, 160, 640, 800],
        [360, 0, 1000, 640],
        [360, 160, 1000, 800],
    ],
    (4512, 4512, "10x"): [
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
    (4512, 4512, "4x"): [
        [0, 0, 640, 640],
        [0, 560, 640, 1200],
        [0, 1120, 640, 1760],
        [0, 1680, 640, 2320],
        [0, 2240, 640, 2880],
        [0, 2800, 640, 3440],
        [0, 3360, 640, 4000],
        [0, 3872, 640, 4512],
        [560, 0, 1200, 640],
        [560, 560, 1200, 1200],
        [560, 1120, 1200, 1760],
        [560, 1680, 1200, 2320],
        [560, 2240, 1200, 2880],
        [560, 2800, 1200, 3440],
        [560, 3360, 1200, 4000],
        [560, 3872, 1200, 4512],
        [1120, 0, 1760, 640],
        [1120, 560, 1760, 1200],
        [1120, 1120, 1760, 1760],
        [1120, 1680, 1760, 2320],
        [1120, 2240, 1760, 2880],
        [1120, 2800, 1760, 3440],
        [1120, 3360, 1760, 4000],
        [1120, 3872, 1760, 4512],
        [1680, 0, 2320, 640],
        [1680, 560, 2320, 1200],
        [1680, 1120, 2320, 1760],
        [1680, 1680, 2320, 2320],
        [1680, 2240, 2320, 2880],
        [1680, 2800, 2320, 3440],
        [1680, 3360, 2320, 4000],
        [1680, 3872, 2320, 4512],
        [2240, 0, 2880, 640],
        [2240, 560, 2880, 1200],
        [2240, 1120, 2880, 1760],
        [2240, 1680, 2880, 2320],
        [2240, 2240, 2880, 2880],
        [2240, 2800, 2880, 3440],
        [2240, 3360, 2880, 4000],
        [2240, 3872, 2880, 4512],
        [2800, 0, 3440, 640],
        [2800, 560, 3440, 1200],
        [2800, 1120, 3440, 1760],
        [2800, 1680, 3440, 2320],
        [2800, 2240, 3440, 2880],
        [2800, 2800, 3440, 3440],
        [2800, 3360, 3440, 4000],
        [2800, 3872, 3440, 4512],
        [3360, 0, 4000, 640],
        [3360, 560, 4000, 1200],
        [3360, 1120, 4000, 1760],
        [3360, 1680, 4000, 2320],
        [3360, 2240, 4000, 2880],
        [3360, 2800, 4000, 3440],
        [3360, 3360, 4000, 4000],
        [3360, 3872, 4000, 4512],
        [3872, 0, 4512, 640],
        [3872, 560, 4512, 1200],
        [3872, 1120, 4512, 1760],
        [3872, 1680, 4512, 2320],
        [3872, 2240, 4512, 2880],
        [3872, 2800, 4512, 3440],
        [3872, 3360, 4512, 4000],
        [3872, 3872, 4512, 4512],
    ],
}


def run_yolo_v5(
    input_image: np.ndarray, normalize_image: bool=False, is_4x: bool=False
) -> Tuple[np.array, np.array, np.array, float]:
    
    # make a copy to not modify the input image
    img = input_image.copy()

    if len(img.shape) > 2:
        print(
            "Warning YOLO model may suffer loss in precision due to conversion from RGB to grayscale"
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if normalize_image:
        img = cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    image_height, image_width = img.shape[:2]
    
    if is_4x:
        resize_dict_key: Tuple[int, int, str] = (image_width, image_height, "4x")
    else:
        resize_dict_key: Tuple[int, int, str] = (image_width, image_height, "10x")
    
    if resize_dict_key not in RESIZE:
        logging.error(
            f"The input image size {(image_width, image_height)} is not supported for {'4x' if is_4x else '10x'}! Returning no cells!"
            )
        return (
                np.zeros((0, 4), dtype=int),
                np.zeros((0,), dtype=int),
                np.zeros((0,), dtype=float),
                0
            )

    # we keep the aspect ratio in RESIZE dictionary, scale_factor is the same for both dimensions
    
    scale_factor: float = image_width / RESIZE[resize_dict_key][0]
    resized_img: np.ndarray = cv2.resize(
        img, RESIZE[resize_dict_key], interpolation=cv2.INTER_AREA
    )

    crop_corners: List[List[int]] = CROP_CORNERS[resize_dict_key]


    # only declarations
    boxes: np.ndarray = np.zeros((0, 4), dtype=int)
    labels: np.ndarray = np.zeros((0,), dtype=int)
    scores: np.ndarray = np.zeros((0,), dtype=float)

    st = time.time()

    boxes, labels, scores = detector.detect_by_cropping(
        resized_img, crop_corners
    )

    # scale the detections back to original image resolution
    boxes = (scale_factor * boxes).astype(int)

    et = time.time()

    return boxes, labels, scores, et - st
