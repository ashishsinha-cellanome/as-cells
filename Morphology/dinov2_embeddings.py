import torch, torchvision
import numpy as np
import cv2
import logging
from typing import List, Tuple, Dict, Final

# constants
# the minimum ratio of the area of a cell mask to the area of the DINOv2 crop for
# reliable embedding extraction
# embeddings for objects smaller than this threshold are mark as unreliable
MIN_CROP_COVERAGE_RATIO: Final[float] = 0.1
# objects with corners within this threshold to the image boundaries are likely to be
# cut/divided between images and are their embeddings are marked as unreliable
NUM_PIXELS_TO_DECLARE_BORDER_PROXIMITY: Final[int] = 5
# the input size for DINOv2 model (crop size)
DINOV2_INPUT_SIZE: Final[int] = 84


def build_dinov2_model(model_type: str = "small", with_registers: bool = False):
    model_type_map: Dict[str, str] = {
        "small": "dinov2_vits14",
        "base": "dinov2_vitb14",
        "large": "dinov2_vitl14",
        "giant": "dinov2_vitg14",
    }
    if model_type in model_type_map:
        model_type_str: str = model_type_map[model_type]

    else:
        model_type_str: str = "dinov2_vitb14"
        logging.error(
            f"Incorrect model type passed {model_type}! Using the base model by default."
        )
    dinov2_model = torch.hub.load("facebookresearch/dinov2", model_type_str)

    for param in dinov2_model.parameters():
        param.requires_grad_(False)

    return dinov2_model


# build the model (to use in function below)
dinov2 = build_dinov2_model(model_type="large", with_registers=True)


def extract_dinov2_embeddings(
    img: np.ndarray,
    predictions: dict,
    class_ids_of_interest: List[int],
    crop_size: int = DINOV2_INPUT_SIZE,
    batch_size: int = 32,
    keep_obj_sizes: bool = False,
    dinov2_model=dinov2,
) -> Tuple[List[int], np.ndarray, List[bool]]:
    """
    A function to extraction the embeddings from an input image given the masks and bounding boxes of the objects
    or interest.
    Args:
        img (numpy array): The input image with np.uint8 dtype (with bit-depth already converted to 8 bits) and after
            any required normalization. This is the same input image to the Mask R-CNN model.
        predictions (dictionary): Mask R-CNN output results dictionary with keys as "boxes", "labels", "scores" and
            "masks" and values as a list of num_detections (4,) numpy array for bounding boxes, list of num_detections
            integer class IDs (labels), list of num_detections float detection scores and a list of num_detections numpy
            arrays for each mask. Each mask should be defined within the passed bounding boxes for the
            detected object, hence the masks are not of the same size.
        class_ids_of_interest (list of integers): List of class IDs for cells to use for embedding extraction. If None
            is passed, the embeddings will be extracted for all objects.
        crop_size (integer): The input size (square size) of DINOv2 model, and the size of the crops for the objects.
        batch_size (integer): The batch size to use for passing the cropped images to the model. Can be used to
            trade-off speed and memory requirement.
        keep_obj_sizes (bool): If set to True, the detected objects will be confined in the crop without any resizing.
            Objects larger than the crop size will be resized to fit in. Their embeddings will be marked as unreliable.
        dinov2_model: The DINOv2 model (without any head) to extract the embeddings (given a crop). This is the
            model that is returned by the function build_dinov2_model() above (torch.hub.load function) for which
            embeddings = dinov2_model(normalized_img_crop_tensor). The required normalization on the crop
            and conversion to tensor is done below and not in extract_dinov2_embeddings.
    Returns:
         crops_obj_ids: The list of object indexes (with respect to detections in predictions['boxes'],
            predictions['labels'], ...) for which the embeddings were extracted (the objects with class IDs/labels in
            class_ids_of_interest).
         The numpy array of (len(crops_obj_ids), embedding_size) of the extracted embeddings.
         A list of boolean of the same size as crops_obj_ids, each indicating whether the extracted embeddings are
            reliable or not for that object. Embeddings for objects that are too small with respect to the DINOv2 input
            crop size, or too large to fit in the crop when keep_obj_sizes is set to True, or objects near the image
            boundaries are flagged as unreliable.

    """

    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    half_crop_size: int = int(np.ceil(crop_size / 2.0))
    # mark objects that their areas are smaller than 10% of the crop area as potentially having
    # unreliable and inaccurate embeddings
    min_mask_area: int = int(MIN_CROP_COVERAGE_RATIO * crop_size**2)

    img_height, img_width = img.shape[:2]

    if isinstance(predictions["boxes"], list):
        obj_boxes: np.ndarray = np.array(predictions["boxes"]).astype(int)
    else:
        obj_boxes: np.ndarray = predictions["boxes"].astype(int)

    boxes_widths: np.ndarray = obj_boxes[:, 2] - obj_boxes[:, 0]
    boxes_heights: np.ndarray = obj_boxes[:, 3] - obj_boxes[:, 1]

    boxes_centers_x: np.ndarray = (obj_boxes[:, 2] + obj_boxes[:, 0]) / 2.0
    boxes_centers_y: np.ndarray = (obj_boxes[:, 3] + obj_boxes[:, 1]) / 2.0

    half_obj_sizes: np.ndarray = np.maximum(boxes_heights, boxes_widths) / 2.0

    obj_masks: List[np.ndarray] = predictions["masks"]
    obj_areas: List[int] = [np.sum(mask) for mask in obj_masks]

    crops_obj_ids: List[int] = []
    is_reliable: List[bool] = []
    crops_list: List[np.ndarray] = []

    for obj_id in range(obj_boxes.shape[0]):

        reliable_embeddings_flag: bool = True
        resize_this_obj: bool = not keep_obj_sizes

        # box coordinates
        xtl, ytl, xbr, ybr = obj_boxes[obj_id]
        class_id: int = predictions["labels"][obj_id]

        if class_ids_of_interest is not None and class_id not in class_ids_of_interest:
            continue

        # mark tiny and narrow objects as potentially unreliable
        if obj_areas[obj_id] < min_mask_area:
            reliable_embeddings_flag: bool = False

        # declare the potentially cut and incomplete objects near the image boundary as unreliable
        if (
            xtl < NUM_PIXELS_TO_DECLARE_BORDER_PROXIMITY
            or ytl < NUM_PIXELS_TO_DECLARE_BORDER_PROXIMITY
            or xbr > img_width - NUM_PIXELS_TO_DECLARE_BORDER_PROXIMITY
            or ybr > img_height - NUM_PIXELS_TO_DECLARE_BORDER_PROXIMITY
        ):
            reliable_embeddings_flag: bool = False

        # if we are supposed to keep the object sizes and the object does not fit in the crop, resize it
        # but reflect this in the embedding reliablitly flag
        if keep_obj_sizes and half_obj_sizes[obj_id] > half_crop_size:
            resize_this_obj: bool = True
            reliable_embeddings_flag: bool = False

        # form a square crop around the object and centered around the center of the box
        # note that the bounding box of the object is already expanded by a factor
        # PERCENTAGE_TO_EXPAND_BBOX_BOUNDARIES in Mask R-CNN detections if needed
        # so no need to further expand here to have some margin around the object
        start_pixel_in_x: int = int(
            max(0, boxes_centers_x[obj_id] - half_obj_sizes[obj_id])
        )
        end_pixel_in_x: int = int(start_pixel_in_x + 2 * half_obj_sizes[obj_id])

        # adjust for the objects on the image boundaries
        if end_pixel_in_x >= img_width:
            end_pixel_in_x = img_width
            start_pixel_in_x = int(img_width - 2 * half_obj_sizes[obj_id])

        start_pixel_in_y: int = int(
            max(0, boxes_centers_y[obj_id] - half_obj_sizes[obj_id])
        )
        end_pixel_in_y: int = int(start_pixel_in_y + 2 * half_obj_sizes[obj_id])

        if end_pixel_in_y >= img_height:
            end_pixel_in_y = img_height
            start_pixel_in_y = int(img_height - 2 * half_obj_sizes[obj_id])

        # resize the object within the provided crop size, we do that to ensure the crop only
        # contains the object class as annotated
        if half_obj_sizes[obj_id] > half_crop_size:
            interpolation_scheme = cv2.INTER_AREA
        else:
            interpolation_scheme = cv2.INTER_CUBIC

        # make a copy of the image, and then mask it with dark background
        img_copy = np.zeros(img.shape, np.uint8)
        # with this option, xtl, ytl, xbr, ybr have not been extended to for a square crop
        img_copy[ytl:ybr, xtl:xbr] = img[ytl:ybr, xtl:xbr] * obj_masks[obj_id]

        if resize_this_obj:
            cropped_img: np.ndarray = cv2.resize(
                img_copy[
                    start_pixel_in_y:end_pixel_in_y, start_pixel_in_x:end_pixel_in_x
                ],
                (crop_size, crop_size),
                interpolation_scheme,
            )
        else:
            cropped_img: np.ndarray = np.zeros((crop_size, crop_size), dtype=np.uint8)
            obj_crop: np.ndarray = img_copy[
                start_pixel_in_y:end_pixel_in_y, start_pixel_in_x:end_pixel_in_x
            ]
            # center the object crop inside the crop
            # if we are here, for sure half_obj_sizes[obj_id] <= half_crop_size
            offset: int = int(half_crop_size - half_obj_sizes[obj_id])
            cropped_img[
                offset : offset + obj_crop.shape[1], offset : offset + obj_crop.shape[0]
            ] = obj_crop

        if np.ndim(cropped_img) == 2:
            # for gray scale channel, make them a 3-D image expected by the model
            cropped_img: np.ndarray = np.repeat(
                np.expand_dims(cropped_img, axis=2), 3, axis=2
            )
        crops_list.append(cropped_img)
        crops_obj_ids.append(obj_id)
        is_reliable.append(reliable_embeddings_flag)

    dinov2_model.to(device)
    dinov2_model.eval()

    # extract the embeddings
    embeddings: List[torch.tensor] = []

    image_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(DINOV2_INPUT_SIZE),
            # torchvision.transforms.CenterCrop(84),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )

    for i in range(0, len(crops_list), batch_size):
        image_tensors = torch.cat(
            [
                image_transform(cropped_img).unsqueeze(0)
                for cropped_img in crops_list[i : min(i + batch_size, len(crops_list))]
            ],
            dim=0,
        )
        image_tensors = image_tensors.to(device)
        with torch.no_grad():
            features: torch.tesnor = dinov2_model(image_tensors)
        embeddings.append(features.cpu())

    return crops_obj_ids, torch.cat(embeddings, dim=0).numpy(), is_reliable
