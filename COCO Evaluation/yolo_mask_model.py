import cv2
from PIL import Image
import numpy as np
import time
import logging
import onnxruntime
from typing import Tuple, List, Final, Optional, Dict, Union

MODEL_WEIGHTS_PATH: Final[str] = '/home/cellareye/Cellanome/dl-mehdi/runs/segment/batch_1_batch_2_batch_3_images_seg_10_epochs/weights/batch_1_batch_2_batch_3_images_seg_10_epochs.onnx'
CLASS_NAMES_PATH: Final[str] = '/home/cellareye/Cellanome/dl-mehdi/runs/segment/batch_1_batch_2_batch_3_images_seg_10_epochs/weights/caging_analysis_cells_seg.names'


DEFAULT_DETECTION_CONFIDENCE: Final[float] = 0.4
DEFAULT_NMS_THRESHOLD: Final[float] = 0.3
INPUT_IMAGE_SIZE: Final[Tuple[int, int]] = (640, 640)  # width x height
MASK_THRESHOLD: Final[float] = 0.45


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


def show_detections(input_image, predictions, label_map):

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
    if isinstance(input_image, np.ndarray):
        image = input_image.copy()
    else:
        # convert to a numpy array
        image = np.array(input_image)
    
    # convert to 3-channels
    if len(image.shape) < 3:
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2)
    
    boxes = predictions['boxes']
    labels = predictions['labels']
    masks = predictions['masks']

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


# A utility function
def to_numpy(tensor):
    """
    A function to convert a torch input to numpy array.
    Args:
        tensor (torch tensor).
    Returns:
        Converted to numpy array.
    """
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# YOLOv8 (Yolact) instance segmentation class
class Yolov8InstanceSegmentation:
    def __init__(
            self,
            weights_path: Optional[str] = MODEL_WEIGHTS_PATH,
            names_path: Optional[str] = CLASS_NAMES_PATH,
            model_input_size: Tuple[int, int] = INPUT_IMAGE_SIZE,
            confidence: float = DEFAULT_DETECTION_CONFIDENCE,
            nms_threshold: float = DEFAULT_NMS_THRESHOLD,
            use_onnx_runtime: bool = True
    ):

        self._net = None
        self._weights_path: str = str(weights_path)
        self._names_path: str = str(names_path)
        self._model_input_size: Tuple[int, int] = model_input_size
        self._confidence: float = confidence
        self._nms_threshold: float = nms_threshold
        self._use_onnx_runtime: bool = use_onnx_runtime

        logging.info(
            f"Loading class names from {self._names_path} with confidence {self._confidence} and"
            f" NMS threshold {self._nms_threshold} ..."
        )

        # read class names and create the label map
        try:
            with open(
                    self._names_path, "r"
            ) as f:  # if fails to read then blow with error
                class_names: List[str] = [cname.strip() for cname in f.readlines()]

            self._label_map: Dict[int, str] = {i: c for i, c in enumerate(class_names)}
            self._reverse_label_map: Dict[str, int] = {
                value: key for key, value in self._label_map.items()
            }
        except Exception as ex:
            logging.error(
                f"Failed to initialize YOLOv8 model with CUDA. Likely the paths to model ONNX weights "
                f"{self._names_path} is incorrect: {repr(ex)}."
            )
            return

        logging.info("Class names were successfully loaded")

        # loading the ONNX model
        logging.info(
            f"Loading YOLOv8 ONNX weights from {self._weights_path}."
        )
        try:
            if self._use_onnx_runtime:
                logging.info(
                    "Using ONNX Runtime for running the model."
                )

                self._net = onnxruntime.InferenceSession(self._weights_path,
                                                         providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                if 'CUDAExecutionProvider' not in onnxruntime.get_available_providers():
                    logging.warning(
                        "CUDAExecutionProvider is not available in ONNX Runtime! The model will be run on CPU."
                    )
            else:
                logging.info(
                    "Using OpenCV dnn for running the model. Setting dnn to use CUDA."
                )
                self._net = cv2.dnn.readNetFromONNX(self._weights_path)

                try:
                    self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
                except Exception:
                    logging.warning(
                        "Failed to set dnn to use CUDA! Installed OpenCV is not compiled with CUDA support! "
                        "The model will be run on CPU."
                    )

        except Exception as ex:
            logging.error(
                f"Failed to initialize YOLOv8 model with CUDA. Likely the path to model ONNX weights "
                f"{self._weights_path} is incorrect or ONNX Runtime is not installed: {repr(ex)}."
            )

    # note that the passed image can be also a numpy array returned by
    # cv2.imread(img_path, cv2.IMREAD_UNCHANGED), it does not necessarily have to be a PIL image
    # in fact OpenCV is slightly more efficient in reading the images
    def detect(
            self, image: np.ndarray, log_time=False
    ) -> Tuple[np.array, np.array, np.array, List[np.ndarray]]:
        """
        The main function to detect the bounding box and masks for persons in the input image.
        """
        if self._net is None:
            logging.error(
                "YOLOv8 CV model has not been initialized. Please initialize the class before detect()."
            )
            return (
                np.zeros((0, 4), dtype=int),
                np.zeros((0,), dtype=int),
                np.zeros((0,), dtype=float),
                [],
            )

        start: float = time.time()

        image_height, image_width = image.shape[:2]

        # the ONNX model expects input images of exactly specified sizes
        if image_height != self._model_input_size[1] or image_width != self._model_input_size[0]:
            logging.error(
                f"Incorrect input image size {(image_width, image_height)}! The input image should be  "
                f"of size {self._model_input_size}"
            )
            return (
                np.zeros((0, 4), dtype=int),
                np.zeros((0,), dtype=int),
                np.zeros((0,), dtype=float),
                []
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
            # TODO: check if this part works with OpenCV dnn, currently my OpenCV dnn is old and does not
            #  support the ONNX ops used in this model
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

        boxes, labels, scores, masks = self._post_process(outputs)

        elap: float = time.time() - start
        if log_time:
            logging.info(f"YOLOv8 instance segmentation took {elap:.4f} seconds")

        return boxes, labels, scores, masks

    def set_confidence(self, confidence):
        self._confidence = confidence

    def set_nms_threshold(self, threshold):
        self._nms_threshold = threshold

    def _post_process(
            self, outputs: np.ndarray
    ) -> Tuple[np.array, np.array, np.array, List[np.ndarray]]:
        """Post process outputs, discarding unreliable detections & performing NMS"""

        # masks protos, remove the batch dimension
        protos: np.ndarray = outputs[1][0]
        # transpose to have the channels in the axis=2 (needed later for mask calculation, see below)
        protos = np.transpose(protos, (1, 2, 0))
        # predictions, remove the batch dimension and transpose the predictions so
        # we have the detections in axis=0
        preds: np.ndarray = outputs[0][0].T

        # number of classes
        num_classes: int = len(self._label_map)
        # score for all classes
        scores_all_classes: np.ndarray = preds[:, 4:4 + num_classes]
        # bounding boxes and masks after transposing the matrix
        boxes: np.ndarray = preds[:, :4]
        masks: np.ndarray = preds[:, 4 + num_classes:]
        # find the score of the most probable class for each detection (row) across the classes (columns)
        # to filter out unreliable detections (rows) before running NMS
        scores: np.ndarray = np.max(scores_all_classes, axis=1)
        # filter out unreliable detections (rows)
        valid_det_idxs: np.ndarray = np.where(scores >= min(0.5, self._confidence))[0]
        # get the label (class ID) of each detection
        labels: np.ndarray = np.argmax(scores_all_classes[valid_det_idxs, :], axis=1)
        # only keep the confident ones
        # discard unreliable detections before NMS to reduce NMS computations
        scores = scores[valid_det_idxs]
        # remove batch index and keep reliable bounding boxes
        boxes = boxes[valid_det_idxs, :]
        # remove batch index and keep reliable masks
        masks = masks[valid_det_idxs, :]

        # now run NMS on the detections independently on each class
        # the order of the elements in boxes is [center_x, center_y, width, height] without
        # any normalization
        # first, convert the boxes to [x, y, w, h] format with integer values before applying NMS
        # (this is the expected input format for NMS implementation in OpenCV)
        boxes[:, 0] -= boxes[:, 2] / 2.0
        boxes[:, 1] -= boxes[:, 3] / 2.0
        boxes = boxes.astype(int)
        # make sure the boxes are within the image dimensions
        boxes[:, 0] = np.maximum(0, boxes[:, 0])
        boxes[:, 1] = np.maximum(0, boxes[:, 1])

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
        boxes = boxes[index_list_to_return]
        scores = scores[index_list_to_return]
        labels = labels[index_list_to_return]
        masks = masks[index_list_to_return]

        # convert from xtl, ytl, w, h) to (xtl, ytl, xbr, ybr) for each box
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]

        # make sure the boxes are within the image dimensions
        boxes[:, 0] = np.maximum(0, boxes[:, 0])
        boxes[:, 1] = np.maximum(0, boxes[:, 1])
        boxes[:, 2] = np.minimum(self._model_input_size[0], boxes[:, 2])
        boxes[:, 3] = np.minimum(self._model_input_size[1], boxes[:, 3])

        # convert the masks for the detected boxes using the protos
        # the mask for detection m is a 160 by 160 pixels matrix obtained as
        # mask[x, y] = sigmoid(protos[x, y, :] * masks[m, :])
        # the mask covers the whole image (input_image resolution) and should be confined later
        # to the detection bounding box
        actual_masks: List[np.ndarray] = []

        for i, mask in enumerate(masks):
            # get the probability map
            actual_mask = 1 / (1 + np.exp(-np.sum(protos * mask, axis=2)))
            # resize it to the input image, also scale it to [0, 255] to be able to resize
            actual_mask = cv2.resize((actual_mask * 255).astype(np.uint8),
                                     dsize=(self._model_input_size[0], self._model_input_size[1]))
            # confine it to the bounding box for the detection, convert to a probability between [0, 1]
            actual_mask = actual_mask[boxes[i, 1]: boxes[i, 3], boxes[i, 0]: boxes[i, 2]].astype(float) / 255.0
            # threshold the mask to obtain a binary mask
            # actual_mask = cv2.threshold(actual_mask / 255.0, self.mask_threshold, 1.0, cv2.THRESH_BINARY)[1]
            actual_masks.append(actual_mask)

        return boxes, labels, scores, actual_masks

    def detect_by_cropping(
            self,
            image: Union[Image.Image, np.ndarray],
            crop_corners: List[List[int]],
            nms_threshold_for_combining_crop_results: float = 0.15,
            only_report_cells: bool = False,
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
            image (PIL.Image or numpy array): Input image, should have 8 bits per channel
                bit depth (np.uint8 in case of a numpy array).
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
            A dictionary with keys and values as below:
               "boxes": List of 4-tuples or 4-element lists for the detected objects' bounding boxes in
               xtl, ytl, xbr, ybr format/order
               "labels": List of integer class IDs for the detected objects
               "scores": List of float detection scores, thresholded by self._confidence
               "masks": List of masks for the detected objects. Each mask is a numpy array of the same size
               as the bounding box (xbr - xtl, ybr - ytl)
        """

        start = time.time()
        if len(crop_corners) == 0:
            logging.error(
                "No crop corners are provided for running YOLOv8 model on sub-images. "
                "Returning no detections"
            )
            out: dict = {'boxes': [], 'scores': [], 'labels': [], 'masks': []}
            return out

        if only_report_cells and "cell" not in self._reverse_label_map:
            logging.error(
                "'cell' classname is not included in the model label map. Returning no detections"
            )
            out: dict = {'boxes': [], 'scores': [], 'labels': [], 'masks': []}
            return out

        H, W = image.shape[:2]

        # check if all the crop sub-images are of the same size,
        # if not, make them equal size for batch processing
        crop_widths: List[int] = [min(c[2], W) - max(c[0], 0) for c in crop_corners]
        crop_heights: List[int] = [min(c[3], H) - max(c[1], 0) for c in crop_corners]

        if min(crop_widths) <= 0 or min(crop_heights) <= 0:
            logging.error(
                "Incorrect corners are provided for running YOLOv8 model on sub-images. "
                "Returning no detections"
            )
            out: dict = {'boxes': [], 'scores': [], 'labels': [], 'masks': []}
            return out

        crop_width: int = max(crop_widths)
        crop_height: int = max(crop_heights)

        # combine the results, filter them based on the score,
        # and update the coordinates of the bounding boxes
        # for applying NMS later
        results: Dict = {"scores": [], "boxes": [], "labels": [], "masks": []}

        # a list to keep track of cropped sub-images with at least one object detection
        crop_ids_with_detection: List[int] = []

        for crop_id, corners in enumerate(crop_corners):
            (x1c, y1c, x2c, y2c) = corners
            # enlarge the crop if necessary to make all the same size
            x2c = x1c + crop_width
            y2c = y1c + crop_height

            # crop the image and run the model
            cropped_image = image[y1c:y2c, x1c:x2c]
            boxes, labels, scores, masks = self.detect(cropped_image)
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
            out: dict = {'boxes': [], 'scores': [], 'labels': [], 'masks': []}
            return out

        # list to contain the detections
        boxes, labels, scores, masks = [], [], [], []

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
            crop_masks: List[np.array] = results["masks"][idx]

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

            # masks for detections in other cropped sub-images
            rest_masks: List[np.ndarray] = [
                results["masks"][i]
                for i in range(len(crop_ids_with_detection))
                if i != idx
            ]

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
                            masks.append(crop_masks[crop_class_idxs[0][i]])
                            if crop_det_score == rest_det_score:
                                # this is added to break the tie if both areas are equal
                                # so we will not add the same box twice when considering
                                # in another crop
                                results["scores"][idx][crop_class_idxs[0][i]] += 1e-5

        if only_report_cells:
            cell_class_id: int = self._reverse_label_map["cell"]
            cell_detection_ids: List[int] = []
            for idx, label in enumerate(labels):
                if label != cell_class_id:
                    continue
                cell_detection_ids.append(idx)

            boxes = [boxes[i] for i in cell_detection_ids]
            labels = [labels[i] for i in cell_detection_ids]
            scores = [scores[i] for i in cell_detection_ids]
            masks = [masks[i] for i in cell_detection_ids]

        elap: float = time.time() - start
        if log_time:
            logging.info(
                "YOLOv8 instance segmentation after cropping the image to "
                "{} sub-images took {:.4f} seconds in OpenCV".format(
                    len(crop_corners), elap
                )
            )

        out: dict = {'boxes': [[box[0], box[1], box[2], box[3]] for box in boxes],  # convert to list
                     'scores': scores, 'labels': labels, 'masks': masks}
        return out
    
    
detector = Yolov8InstanceSegmentation(use_onnx_runtime=True)

RESIZE: Final[Dict[Tuple[int, int], Tuple[int, int]]] = {
    (2000, 1600): (2000, 1600),
    (4512, 4512): (2880, 2880),
}
# A dictionary with keys as the input (original) image size (width, height) tuple and
# values as the list of coordinates (xtl, ytl, xbr, ybr) of sub-images/crops
# to run YOLOv5 on each
# note that the crop coordinates are with respect to resized image dimensions specified above
CROP_CORNERS: Final[Dict[Tuple[int, int], List[List[int]]]] = {
    (2000, 1600): [
         [0, 0, 640, 640],
         [0, 480, 640, 1120],
         [0, 960, 640, 1600],
         [454, 0, 1094, 640],
         [454, 480, 1094, 1120],
         [454, 960, 1094, 1600],
         [908, 0, 1548, 640],
         [908, 480, 1548, 1120],
         [908, 960, 1548, 1600],
         [1360, 0, 2000, 640],
         [1360, 480, 2000, 1120],
         [1360, 960, 2000, 1600],
    ],
    
    (4512, 4512): [
        [0, 0, 640, 640],
        [0, 374, 640, 1014],
        [0, 748, 640, 1388],
        [0, 1122, 640, 1762],
        [0, 1496, 640, 2136],
        [0, 1870, 640, 2510],
        [0, 2240, 640, 2880],
        [374, 0, 1014, 640],
        [374, 374, 1014, 1014],
        [374, 748, 1014, 1388],
        [374, 1122, 1014, 1762],
        [374, 1496, 1014, 2136],
        [374, 1870, 1014, 2510],
        [374, 2240, 1014, 2880],
        [748, 0, 1388, 640],
        [748, 374, 1388, 1014],
        [748, 748, 1388, 1388],
        [748, 1122, 1388, 1762],
        [748, 1496, 1388, 2136],
        [748, 1870, 1388, 2510],
        [748, 2240, 1388, 2880],
        [1122, 0, 1762, 640],
        [1122, 374, 1762, 1014],
        [1122, 748, 1762, 1388],
        [1122, 1122, 1762, 1762],
        [1122, 1496, 1762, 2136],
        [1122, 1870, 1762, 2510],
        [1122, 2240, 1762, 2880],
        [1496, 0, 2136, 640],
        [1496, 374, 2136, 1014],
        [1496, 748, 2136, 1388],
        [1496, 1122, 2136, 1762],
        [1496, 1496, 2136, 2136],
        [1496, 1870, 2136, 2510],
        [1496, 2240, 2136, 2880],
        [1870, 0, 2510, 640],
        [1870, 374, 2510, 1014],
        [1870, 748, 2510, 1388],
        [1870, 1122, 2510, 1762],
        [1870, 1496, 2510, 2136],
        [1870, 1870, 2510, 2510],
        [1870, 2240, 2510, 2880],
        [2240, 0, 2880, 640],
        [2240, 374, 2880, 1014],
        [2240, 748, 2880, 1388],
        [2240, 1122, 2880, 1762],
        [2240, 1496, 2880, 2136],
        [2240, 1870, 2880, 2510],
        [2240, 2240, 2880, 2880],
    ],
}

def run_yolo_v8(input_image: np.ndarray, 
                normalize_image: bool = True, 
                bit_depth: int = 8, 
                plot_results: bool = False, 
) -> Tuple[Dict[str, list], float, Optional[np.ndarray]]:
    
    # make a copy to not modify the input image
    img = input_image.copy()

    if len(img.shape) > 2:
        print(
            "Warning YOLO model may suffer loss in precision due to conversion from RGB to grayscale"
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    img = (255 * img.astype(float) / (2 ** bit_depth - 1)).astype(np.uint8)

    if normalize_image:
        img = cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    image_height, image_width = img.shape[:2]

    if (image_width, image_height) not in RESIZE:
        logging.error(
            "The input image size {} is not supported! Returning no cells!".format(
                image_width, )
        )
        out = {'boxes': [],
               'labels': [],
               'scores': [],
               'masks': [],
               }
        if plot_results:
            return (out, 0, np.zeros((image_height, image_width), dtype=np.uint8))
        else:
            return (out, 0)

    # we keep the aspect ratio in RESIZE dictionary, scale_factor is the same for both dimensions
    scale_factor: float = image_width / RESIZE[(image_width, image_height)][0]
    resized_img: np.ndarray = cv2.resize(
        img, RESIZE[(image_width, image_height)], interpolation=cv2.INTER_AREA
    )
    
    st = time.time()
    crop_corners: List[List[int]] = CROP_CORNERS[(image_width, image_height)]
    out: Dict[str, list] = detector.detect_by_cropping(
            resized_img, crop_corners, only_report_cells=False)

    if scale_factor != 1:
        # scale the detections back to original image resolution
        out['boxes'] = (scale_factor * np.array(out['boxes'])).astype(int)
        # convert to a list to be consistent with the rest
        out['boxes'] = [box for box in out['boxes']]
        
    for idx in range(len(out['boxes'])):
        if scale_factor != 1:
            xtl, ytl, xbr, ybr = out['boxes'][idx]
            # note that mask here is a probability mask and interpolation does not have to be nearest neighbor
            out['masks'][idx] = cv2.resize(out['masks'][idx], (xbr - xtl, ybr - ytl), interpolation=cv2.INTER_LINEAR)
        mask_this_cell: np.ndarray = np.zeros(out['masks'][idx].shape, dtype=np.uint8)
        mask_this_cell[out['masks'][idx] >= MASK_THRESHOLD] = 1
        out['masks'][idx] = mask_this_cell.astype(np.uint8)

    et = time.time()

    if plot_results:
        return out, et - st, show_detections(img, out, detector._label_map)
    
    return out, et - st
