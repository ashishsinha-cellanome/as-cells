import logging

import cv2
import numpy as np

import torch
import torchvision
from torchvision.models import segmentation as seg_models
from typing import List, Dict, Tuple, Union, Final, Optional

# these should go into model configurations included in the weights file (like in Mask R-CNN)
DEFAULT_LABEL_MAP: Dict[int, str] = {1: "cage", 2: "cell", 3: "bead"}
# model input image large/small-side sizes
DEFAULT_MODEL_INPUT_SIZE: Final[int] = 512
# the default batch size for running the model on multiple images
DEFAULT_BATCH_SIZE: Final[int] = 16
# path to the model's weights
WEIGHTS_PATH: Final[str] = (
    "/home/cellareye/Cellanome/dl-mehdi/Semantic Segmentation/checkpoints/20241023_sets_1_2_3_6_to_38_0p2_bbox_0p2_b_c_adj_16_bs_10_epochs_1cl_lrs.pt"
)


def get_sem_seg_model(num_classes: int, model_type: str):

    if model_type == "fcn_resnet50":
        model = seg_models.fcn_resnet50(weights=seg_models.FCN_ResNet50_Weights.DEFAULT)
    elif model_type == "deeplabv3_resnet50":
        model = seg_models.deeplabv3_resnet50(
            weights=seg_models.DeepLabV3_ResNet50_Weights.DEFAULT
        )
    else:
        logging.error(
            f"[ERROR]: Invalid model type: {model_type}! The passed type should be either 'fcn_resnet50' or 'deeplabv3_resnet50'"
        )
        return None
    # Change final layer to 2 classes
    # note that FCN and DeepLab models have different number of channels for this layer (512 vs 256 for DeepLab)
    num_channels: int = model.classifier[4].state_dict()["weight"].shape[1]
    model.classifier[4] = torch.nn.Conv2d(
        num_channels, num_classes, kernel_size=(1, 1), stride=(1, 1)
    )
    return model


# FCN or DeepLabv3 semantic segmentation model class
class SemanticSegmentator:
    def __init__(
        self,
        weights_path: Optional[str] = WEIGHTS_PATH,
        label_map: Optional[Dict[int, str]] = None,
        model_type: str = "fcn_resnet50",
    ):

        self._model: Union[
            torchvision.models.segmentation.fcn.FCN,
            torchvision.models.segmentation.deeplabv3.DeepLabV3,
        ] = None
        self._weights_path: str = str(weights_path)

        # available device
        self._device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # the state dictionary of the model to be read from the weights file
        model_state_dict: OrderedDict = None
        # the model's label map if available in the weight file
        loaded_label_map: Dict[int, str] = None
        self._model_input_size: int = DEFAULT_MODEL_INPUT_SIZE

        # loading the PyTorch weights and the label map
        try:
            logging.info(
                f"Loading PyTorch FCN/DeepLabV3 model from {self._weights_path}. Setting to run on {self._device.type}."
            )

            saved_model_param: Union[OrderedDict, list] = torch.load(
                self._weights_path, map_location=self._device
            )
            if (
                isinstance(saved_model_param, dict)
                and "model_state_dict" in saved_model_param
            ):
                # the weights file contains the model state dictionary (the weights) and the label map (both are
                # mandatory) and potentially other model related configs
                # in case a dictionary is provided, the keys and values are as following:
                # - 'model_state_dict': model state dictionary (mandatory)
                # - 'label_map': label map (mandatory)
                # - 'input_size': anchor sizes (mandatory)
                # ... other optional fields to be added in future
                model_state_dict = saved_model_param["model_state_dict"]
                if "label_map" in saved_model_param:
                    loaded_label_map = saved_model_param["label_map"]
                else:
                    logging.warning(
                        "The weights file should contain the label map but it is missing. "
                        "This should never happen..."
                    )
                    loaded_label_map = None

                if "input_size" in saved_model_param:
                    self._model_input_size = saved_model_param["input_size"]
                else:
                    logging.warning(
                        "The weights file should contain the model input size but it is missing. "
                        "This should never happen..."
                    )
                    # we use the default value here
            else:
                # the file only contains the model state dictionary
                model_state_dict = saved_model_param
                loaded_label_map = None

        except Exception as ex:
            logging.error(
                f"Failed to load the instance segmentation model. Likely the paths to model .pt weights "
                f"{self._weights_path} is incorrect: {repr(ex)}."
            )

        if label_map is None:
            if loaded_label_map is None:
                logging.error(
                    "The mapping between the class IDs and class names is required for the model and is "
                    "neither provided during class instantiation nor available in the weights file! Returning ..."
                )
                return
            else:
                logging.info(
                    "Mapping between class IDs and class names is provided in the weights file."
                )
                self._label_map: Dict[int, str] = loaded_label_map
        else:
            logging.info(
                "Mapping between class IDs and class names is passed during class instantiation! "
                "It will overwrite the label map passed in the weights file (if provided)."
            )
            self._label_map: Dict[int, str] = label_map

        logging.info(f"Mapping between class IDs and class names: {self._label_map}")
        self._reverse_label_map: Dict[str, int] = {
            value: key for key, value in self._label_map.items()
        }

        # instance segmentation model
        self._model = get_sem_seg_model(
            num_classes=len(self._label_map) + 1, model_type=model_type
        )
        if self._model is None:
            # invalid model_type
            return

        # loading the PyTorch model
        try:
            self._model.load_state_dict(model_state_dict)
            self._model.to(self._device)
            self._model.eval()

        except Exception as ex:
            logging.error(f"Failed to load instance segmentation model: {repr(ex)}.")
        # image transform to apply during inference
        # if the passed crops are not sqaure, the aspect ratio will be modified
        self._transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(
                    size=(self._model_input_size, self._model_input_size)
                ),
                torchvision.transforms.Normalize([0.449], [0.226]),
            ]
        )

    def get_label_map(self):
        return self._label_map

    def get_reverse_label_map(self):
        return self._reverse_label_mape

    def get_class_names(self):
        return list(self.get_reverse_label_map().keys())

    def get_model_input_size(self):
        return self._model_input_size

    def predict(
        self,
        crop: np.ndarray,
    ):
        return self.predict_batch([crop])[0]

    def predict_batch(
        self, crops_list: List[np.ndarray], batch_size: int = DEFAULT_BATCH_SIZE
    ):
        if len(crops_list) == 0:
            return []
        # make sure the images are with 3 channels (expected by the model)
        adj_crops_list: List[np.ndarray] = [
            cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR)
            if np.ndim(cropped_img) == 2
            else cropped_img
            for cropped_img in crops_list
        ]
        # input sizes of the crops, save them to resize the results back to the original size before returning
        org_crop_sizes: List[Tuple[int, int]] = [
            (cropped_img.shape[1], cropped_img.shape[0]) for cropped_img in crops_list
        ]
        output_masks: List[np.ndarray] = []
        # if the passed crops are not sqaure, the aspect ratio will be modified below in self._transform
        for i in range(0, len(adj_crops_list), batch_size):
            image_tensors: torch.tensor = torch.cat(
                [
                    self._transform(cropped_img).unsqueeze(0)
                    for cropped_img in adj_crops_list[
                        i : min(i + batch_size, len(adj_crops_list))
                    ]
                ],
                dim=0,
            )
            image_tensors = image_tensors.to(self._device)
            with torch.no_grad():
                batch_output = self._model(image_tensors)["out"]
                batch_masks = (
                    torch.argmax(batch_output, dim=1).cpu().numpy().astype(np.uint8)
                )
            output_masks.append(batch_masks)

        # merge all the batch results over axis=0 (batch dimension)
        output_masks: np.ndarray = np.concatenate(output_masks, axis=0)
        # convert back to original sizes
        return [
            cv2.resize(mask, org_crop_sizes[i], cv2.INTER_NEAREST)
            for i, mask in enumerate(output_masks)
        ]

    def predict_cages(
        self,
        input_image: np.ndarray,
        cage_bounding_boxes: np.ndarray,
        percentage_to_expand_cage_box_boundaries: float,
        bit_depth: int = 12,
        normalize_image: bool = True,
    ) -> Tuple[List[np.array], np.ndarray]:

        # make a copy of the input image
        img = input_image.copy()
        # adjust the bit-depth
        img = (255 * img.astype(float) / (2**bit_depth - 1)).astype(np.uint8)

        if normalize_image:
            img = cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        img_height, img_width = img.shape[:2]

        # make a copy to not modify the input
        if isinstance(cage_bounding_boxes, list):
            # the past bounding boxes are in the form of List[np.array] (the output of Mask R-CNN), convert them to a numpy array
            cage_boxes: np.ndarray = (
                np.array(cage_bounding_boxes).copy()
                if len(cage_bounding_boxes) > 0
                else np.zeros((0, 4), dtype=int)
            )
        else:
            cage_boxes: np.ndarray = cage_bounding_boxes.copy()
        # expand the cage bounding boxes by the provided percentage
        cage_boxes_widths: np.ndarray = (cage_boxes[:, 2] - cage_boxes[:, 0]) * (
            1 + percentage_to_expand_cage_box_boundaries
        )
        cage_boxes_heights: np.ndarray = (cage_boxes[:, 3] - cage_boxes[:, 1]) * (
            1 + percentage_to_expand_cage_box_boundaries
        )

        # the center of cages (defined as the center of cage boxes)
        cage_centers_x: np.ndarray = (cage_boxes[:, 2] + cage_boxes[:, 0]) / 2.0
        cage_centers_y: np.ndarray = (cage_boxes[:, 3] + cage_boxes[:, 1]) / 2.0

        # form an sqaure crop around the cages
        half_cage_sizes: np.ndarray = (
            np.maximum(cage_boxes_heights, cage_boxes_widths) / 2.0
        )

        # expand the bbox to create a square bounding box around the cage
        # we do this because the input to the sematic segmentation model should be sqaure and we want to keep the aspect ratio
        # note that the bounding boxes for cages at the image boundaries may not be sqaure, we will fix it later
        cage_boxes[:, 0] = np.maximum((cage_centers_x - half_cage_sizes), 0).astype(int)
        cage_boxes[:, 1] = np.maximum((cage_centers_y - half_cage_sizes), 0).astype(int)
        cage_boxes[:, 2] = np.minimum(
            (cage_centers_x + half_cage_sizes), img_width
        ).astype(int)
        cage_boxes[:, 3] = np.minimum(
            (cage_centers_y + half_cage_sizes), img_height
        ).astype(int)

        # store an expanded sqaure crop of the cages in a list of processing
        cage_crops_list: List[np.ndarray] = []
        for cage_id in range(cage_boxes.shape[0]):
            # adjust the boundaries for boxes near the image boundaries to make sure they are square
            # if the size of the cage is more than one image dimension, the result will not be sqaure, but this will not happen
            # in practice
            half_cage_size: int = int(half_cage_sizes[cage_id])
            if cage_boxes[cage_id, 0] == 0:
                # xtl is at the image boundary, adjust xbr
                cage_boxes[cage_id, 2] = min(img_width, 2 * half_cage_size)
            if cage_boxes[cage_id, 1] == 0:
                # ytl is at the image boundary, adjust ybr
                cage_boxes[cage_id, 3] = min(img_height, 2 * half_cage_size)
            if cage_boxes[cage_id, 2] == img_width:
                # xbr is at the image boundary, adjust xtl
                cage_boxes[cage_id, 0] = max(0, img_width - 2 * half_cage_size)
            if cage_boxes[cage_id, 3] == img_height:
                # ybr is at the image boundary, adjust ytl
                cage_boxes[cage_id, 1] = max(0, img_height - 2 * half_cage_size)

            xtl, ytl, xbr, ybr = cage_boxes[cage_id]
            # resizing and add channel dimension is done in predict_batch
            cage_crops_list.append(img[ytl:ybr, xtl:xbr])

        # run the model
        masks_list: List[np.ndarray] = self.predict_batch(crops_list=cage_crops_list)
        # combine all the semantic masks
        semantic_mask: np.ndarray = np.zeros(img.shape[:2], np.uint8)
        for cage_id in range(cage_boxes.shape[0]):
            xtl, ytl, xbr, ybr = cage_boxes[cage_id]
            # we are taking the maximum here to maintain the order of overlapping objects in overlapping areas between cage boxes
            # if an object semantic mask is detected in one cage box but not in another overlapping one, we OR them here
            # if a cage is detected in one, and a cell in another, we use the cell mask
            semantic_mask[ytl:ybr, xtl:xbr] = np.maximum(
                semantic_mask[ytl:ybr, xtl:xbr], masks_list[cage_id]
            )
        # mask per cage bounding box, use the original boxes
        cage_masks: List[np.ndarray] = []
        for box in cage_bounding_boxes:
            xtl, ytl, xbr, ybr = box
            cage_masks.append(semantic_mask[ytl:ybr, xtl:xbr])

        return cage_masks, semantic_mask


# TODO: REMOVE after testing
def generate_cage_crops(
    img: np.ndarray,
    in_cage_boxes: np.ndarray,
    percentage_to_expand_cage_box_boundaries: float,
    crop_size: int,
) -> List[np.ndarray]:

    # image height and width
    img_height, img_width = img.shape[:2]

    # outputs
    img_crops: List[np.ndarray] = []

    half_crop_size: int = int(np.ceil(crop_size / 2.0))

    # make a copy to not modify the input
    cage_boxes: np.ndarray = in_cage_boxes.copy()
    # expand the cage bounding boxes by the provided percentage
    cage_boxes_widths: np.ndarray = (cage_boxes[:, 2] - cage_boxes[:, 0]) * (
        1 + percentage_to_expand_cage_box_boundaries
    )
    cage_boxes_heights: np.ndarray = (cage_boxes[:, 3] - cage_boxes[:, 1]) * (
        1 + percentage_to_expand_cage_box_boundaries
    )

    # the center of cages (defined as the center of cage boxes)
    cage_centers_x: np.ndarray = (cage_boxes[:, 2] + cage_boxes[:, 0]) / 2.0
    cage_centers_y: np.ndarray = (cage_boxes[:, 3] + cage_boxes[:, 1]) / 2.0

    # form an sqaure crop around the cages
    half_cage_sizes: np.ndarray = (
        np.maximum(cage_boxes_heights, cage_boxes_widths) / 2.0
    )

    # expand the bbox to create a square bounding box around the cage
    # we do this because the input to the sematic segmentation model should be sqaure and we want to keep the aspect ratio
    # note that the bounding boxes for cages at the image boundaries may not be sqaure, we will fix it later
    cage_boxes[:, 0] = np.maximum((cage_centers_x - half_cage_sizes), 0).astype(int)
    cage_boxes[:, 1] = np.maximum((cage_centers_y - half_cage_sizes), 0).astype(int)
    cage_boxes[:, 2] = np.minimum((cage_centers_x + half_cage_sizes), img_width).astype(
        int
    )
    cage_boxes[:, 3] = np.minimum(
        (cage_centers_y + half_cage_sizes), img_height
    ).astype(int)

    for cage_id, (xtl, ytl, xbr, ybr) in enumerate(cage_boxes):
        # adjust the boundaries for boxes near the image boundaries to make sure they are square
        # if the size of the cage is more than one image dimension, the result will not be sqaure, but this will not happen
        # in practice
        half_cage_size: int = int(half_cage_sizes[cage_id])
        if xtl == 0:
            xbr = min(img_width, 2 * half_cage_size)
        if ytl == 0:
            ybr = min(img_height, 2 * half_cage_size)
        if xbr == img_width:
            xtl = max(0, img_width - 2 * half_cage_size)
        if ybr == img_height:
            ytl = max(0, img_height - 2 * half_cage_size)

        # resize the cage crop within the provided crop size
        if half_cage_size > half_crop_size:
            interpolation_scheme = cv2.INTER_AREA
        else:
            interpolation_scheme = cv2.INTER_CUBIC

        cropped_img: np.ndarray = cv2.resize(
            img[ytl:ybr, xtl:xbr], (crop_size, crop_size), interpolation_scheme
        )
        # make the crop a 3-channel image (expected by the model) if it is not
        cropped_img = (
            cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR)
            if np.ndim(cropped_img) == 2
            else cropped_img
        )

        img_crops.append(cropped_img)

    return img_crops


def prepare_crop(crop: np.ndarray):
    if crop is None:
        return None

    # normalization parameters
    mean: Final[float] = 0.449
    std: Final[float] = 0.226
    # make sure the image is a gray scale image (BGR or RGB does not matter)
    # then normalize
    if len(crop.shape) > 2:
        img: np.ndarray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        img: np.ndarray = (crop / 255.0 - mean) / std
    else:
        img: np.ndarray = (crop / 255.0 - mean) / std

    if max(img.shape[:2]) > MODEL_INPUT_SIZE:
        interpolation_scheme = cv2.INTER_AREA
    else:
        interpolation_scheme = cv2.INTER_CUBIC
    img = cv2.resize(img, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation_scheme)

    # make a 3 channel image as the model expects it
    img = np.repeat(np.expand_dims(img, axis=2), 3, axis=2)

    return img
