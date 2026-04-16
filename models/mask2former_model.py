from models.AbstractVisionModel import VisionModel
from utils.model_utils import to_numpy

import numpy as np
import cv2
from PIL import Image
import torch
from transformers import (
    Dinov2Config,
    Dinov2Model,
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)

import os
import logging
from typing import Tuple, List, Final, Optional, Dict, Union


def _from_pretrained_cached_first(loader_fn, *, repo_id: str, **kwargs):
    """Try loading from local cache first; fall back to downloading if not cached."""
    # Remove local_files_only from kwargs since we manage it internally
    kwargs.pop("local_files_only", None)
    try:
        return loader_fn(repo_id, local_files_only=True, **kwargs)
    except OSError:
        print(f"[HF] Model not in cache, downloading: {repo_id}")
        return loader_fn(repo_id, local_files_only=False, **kwargs)


# constants and default values
TRANSFORM_MEAN: Final[List[float]] = [0.485, 0.456, 0.406]
TRANSFORM_STD: Final[List[float]] = [0.229, 0.224, 0.225]

# MODEL_REPO_PATH: Final[str] = '/home/cellareye/Cellanome/dl-mehdi/Mask RCNN/checkpoints/mask2former_checkpoints'
MODEL_REPO_PATH: Final[str] = (
    "/global/home/ashish.sinha/cellanome/models/mask2former_checkpoints/"
)

# default model parameters
# in case the weights file does not include them, they can be used
DEFAULT_MODEL_INPUT_SIZE: Final[Tuple[int, int]] = (1022, 798)
DEFAULT_DETECTION_CONFIDENCE: Final[float] = 0.5

# A dictionary with keys as the input (original) image size (width, height) tuple and
# values as the list of coordinates (xtl, ytl, xbr, ybr) of sub-images/crops
# to run YOLOv5 on each
# note that the crop coordinates are with respect to resized image dimensions specified above
DEFAULT_RESIZE: Final[Dict[Tuple[int, int], Tuple[int, int]]] = {
    (2000, 1600): (1280, 1024),
    (4512, 4512): (2148, 2148),
}
# A dictionary with keys as the input (original) image size (width, height) tuple and
# values as the list of coordinates (xtl, ytl, xbr, ybr) of sub-images/crops
# to run YOLOv5 on each
# note that the crop coordinates are with respect to resized image dimensions specified above
DEFAULT_CROP_CORNERS: Final[Dict[Tuple[int, int], List[List[int]]]] = {
    (2000, 1600): [[0, 0, 800, 1024], [480, 0, 1280, 1024]],
    (4512, 4512): [
        [0, 0, 1022, 798],
        [0, 450, 1022, 1248],
        [0, 900, 1022, 1698],
        [0, 1350, 1022, 2148],
        [563, 0, 1585, 798],
        [563, 450, 1585, 1248],
        [563, 900, 1585, 1698],
        [563, 1350, 1585, 2148],
        [1126, 0, 2148, 798],
        [1126, 450, 2148, 1248],
        [1126, 900, 2148, 1698],
        [1126, 1350, 2148, 2148],
    ],
}


def _default_dinov2_out_indices(backbone_name: str) -> List[int]:
    name = backbone_name.lower()
    if "small" in name or "base" in name:
        return [6, 8, 10, 12]
    if "large" in name:
        return [18, 20, 22, 24]
    if "giant" in name:
        return [34, 36, 38, 40]
    return [6, 8, 10, 12]


def _apply_backbone_training_mode(
    model: Mask2FormerForUniversalSegmentation,
    training_mode: str,
) -> None:
    mode = str(training_mode).lower()
    encoder = model.model.pixel_level_module.encoder

    if mode == "frozen":
        for param in encoder.parameters():
            param.requires_grad = False
        return

    if mode == "full_finetune":
        for param in encoder.parameters():
            param.requires_grad = True
        return

    if mode == "lora":
        for param in encoder.parameters():
            param.requires_grad = False
        return

    raise ValueError(f"Unsupported Mask2Former backbone training_mode: {training_mode}")


def _apply_backbone_lora(
    model: Mask2FormerForUniversalSegmentation,
    *,
    rank: int,
    alpha: int,
    dropout: float,
    bias: str,
    target_modules: List[str],
    modules_to_save: Optional[List[str]] = None,
) -> Mask2FormerForUniversalSegmentation:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "LoRA training was requested, but the 'peft' package is not installed. "
            "Install project dependencies again after adding 'peft'."
        ) from exc

    encoder = model.model.pixel_level_module.encoder
    for param in encoder.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=int(rank),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        bias=str(bias),
        target_modules=[str(module) for module in target_modules],
        modules_to_save=list(modules_to_save or []),
    )
    model.model.pixel_level_module.encoder = get_peft_model(encoder, lora_config)
    return model


def summarize_trainable_parameters(
    model: Mask2FormerForUniversalSegmentation,
) -> Dict[str, int]:
    encoder = model.model.pixel_level_module.encoder
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    trainable_backbone_params = sum(
        param.numel() for param in encoder.parameters() if param.requires_grad
    )
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "trainable_backbone_params": int(trainable_backbone_params),
    }


from models.dinov2_backbone_with_fpn import _init_fpn_weights


class Dinov2WithSFP(torch.nn.Module):
    """
    Simple Feature Pyramid (SFP) adapter to convert DINOv2's flat stride-14
    feature maps into a multi-scale FPN (stride ~4, 8, 16, 32) expected by Mask2Former.
    """

    def __init__(self, original_encoder):
        super().__init__()
        self.original_encoder = original_encoder
        self.channels = original_encoder.channels

        # DINOv2 channels (e.g., 768)
        c = self.channels[0]

        # Stride 4 (Upscale 4x from stride 14)
        self.fpn1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(c, c, kernel_size=2, stride=2),
            torch.nn.SyncBatchNorm(c)
            if torch.cuda.device_count() > 1
            else torch.nn.BatchNorm2d(c),
            torch.nn.GELU(),
            torch.nn.ConvTranspose2d(c, c, kernel_size=2, stride=2),
        )
        # Stride 8 (Upscale 2x from stride 14)
        self.fpn2 = torch.nn.ConvTranspose2d(c, c, kernel_size=2, stride=2)
        # Stride 16 (Identity ~ Stride 14)
        self.fpn3 = torch.nn.Identity()
        # Stride 32 (Downscale 2x from stride 14)
        self.fpn4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.apply(_init_fpn_weights)

    def forward(self, pixel_values):
        outputs = self.original_encoder(pixel_values)
        feats = outputs.feature_maps

        f1 = self.fpn1(feats[0])
        f2 = self.fpn2(feats[1])
        f3 = self.fpn3(feats[2])
        f4 = self.fpn4(feats[3])

        outputs.feature_maps = (f1, f2, f3, f4)
        return outputs


def build_mask2former_with_dinov2_backbone(
    *,
    id2label: Dict[int, str],
    mask2former_pretrained_name_or_path: str,
    backbone_pretrained_name_or_path: str,
    training_mode: str = "frozen",
    lora_config: Optional[Dict[str, Union[int, float, str, bool, List[str]]]] = None,
    num_queries: Optional[int] = None,
    out_indices: Optional[List[int]] = None,
    local_files_only: bool = False,
    fpn_type: str = "fused",
    intermediate_resolutions: Optional[List[int]] = None,
) -> Mask2FormerForUniversalSegmentation:

    label2id = {str(name): int(idx) for idx, name in id2label.items()}
    out_indices = out_indices or _default_dinov2_out_indices(
        backbone_pretrained_name_or_path
    )

    # 1. Load the original Mask2Former config to get Swin dimensions
    orig_config = _from_pretrained_cached_first(
        Mask2FormerConfig.from_pretrained,
        repo_id=mask2former_pretrained_name_or_path,
        local_files_only=local_files_only,
    )

    # Extract native embed_dim from Swin config (128 for base, 192 for large)
    embed_dim = orig_config.backbone_config.embed_dim
    # Automatically generate [embed_dim, 2*embed_dim, 4*embed_dim, 8*embed_dim]
    intermediate_channel_sizes = [embed_dim * (2**i) for i in range(4)]

    # 2. Import your generic custom config and backbone
    from models.custom_mask2former_with_dinov2_backbone import (
        Mask2FormerConfigWithCustomBackBone,
        Mask2FormerSegmentationWithCustomBackbone,
    )
    from models.dinov2_backbone_with_fpn import Dinov2BackBoneWithFPNConfig

    model_config = _from_pretrained_cached_first(
        Mask2FormerConfigWithCustomBackBone.from_pretrained,
        repo_id=mask2former_pretrained_name_or_path,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    if num_queries is not None:
        model_config.num_queries = int(num_queries)

    model_config.backbone_config = Dinov2BackBoneWithFPNConfig.from_pretrained(
        backbone_pretrained_name_or_path,
        output_indices_for_fpn=out_indices,
        intermediate_channel_sizes=intermediate_channel_sizes,
        intermediate_resolutions=intermediate_resolutions,
        fpn_type=fpn_type,
    )

    # 3. Instantiate your custom model wrapper
    model = Mask2FormerSegmentationWithCustomBackbone(model_config)

    # 4. Transfer the pre-trained COCO Mask2Former decoder weights
    pretrained_model = _from_pretrained_cached_first(
        Mask2FormerForUniversalSegmentation.from_pretrained,
        repo_id=mask2former_pretrained_name_or_path,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    pretrained_state = pretrained_model.state_dict()
    current_state = model.state_dict()

    transferable_state = {
        key: value
        for key, value in pretrained_state.items()
        if key in current_state
        and value.shape == current_state[key].shape
        and not key.startswith("model.pixel_level_module.encoder.")
    }
    model.load_state_dict(transferable_state, strict=False)

    _apply_backbone_training_mode(model, training_mode)

    # Handle LoRA if enabled
    if str(training_mode).lower() == "lora":
        if not lora_config or not bool(lora_config.get("enabled", False)):
            raise ValueError(
                "Mask2Former LoRA requires model.backbone.training_mode=lora and model.lora.enabled=true."
            )
        model = _apply_backbone_lora(
            model,
            rank=int(lora_config["rank"]),
            alpha=int(lora_config["alpha"]),
            dropout=float(lora_config["dropout"]),
            bias=str(lora_config["bias"]),
            target_modules=[str(x) for x in lora_config["target_modules"]],
            modules_to_save=[
                str(x) for x in lora_config.get("modules_to_save", []) or []
            ],
        )

    return model


def build_original_mask2former(
    *,
    id2label: Dict[int, str],
    mask2former_pretrained_name_or_path: str,
    training_mode: str = "full_finetune",
    num_queries: Optional[int] = None,
    local_files_only: bool = False,
    out_indices: Optional[List[int]] = None,
) -> Mask2FormerForUniversalSegmentation:
    """Build original Mask2Former with its native Swin backbone (no DINOv2 replacement).

    The pretrained Mask2Former checkpoint already contains a Swin backbone trained
    end-to-end on COCO. This function loads it directly and adapts the classification
    head for the target number of classes.
    """
    # out_indices is unused here since the Swin encoder is already configured in the checkpoint
    del out_indices

    label2id = {str(name): int(idx) for idx, name in id2label.items()}

    model = _from_pretrained_cached_first(
        Mask2FormerForUniversalSegmentation.from_pretrained,
        repo_id=mask2former_pretrained_name_or_path,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        local_files_only=local_files_only,
    )

    model_config = model.config
    model_config.num_labels = len(id2label)
    model_config.id2label = id2label
    model_config.label2id = label2id
    if num_queries is not None:
        model_config.num_queries = int(num_queries)

    _apply_backbone_training_mode(model, training_mode)

    # We deliberately don't apply LoRA here for the Swin backbone yet,
    # as it requires verifying the target_modules for Swin architecture.
    if str(training_mode).lower() == "lora":
        raise NotImplementedError(
            "LoRA training is not yet supported for the original Swin backbone. "
            "Please use training_mode='frozen' or 'full_finetune'."
        )

    return model


def get_mask2former_instance_segmentation_model_with_dinov2_backbone(
    id2label: Dict[int, str], model_type: str, with_registers: bool
):
    model_type = str(model_type).lower()
    if with_registers:
        backbone_name = f"facebook/dinov2-with-registers-{model_type}"
    else:
        backbone_name = f"facebook/dinov2-{model_type}"
    return build_mask2former_with_dinov2_backbone(
        id2label=id2label,
        mask2former_pretrained_name_or_path="facebook/mask2former-swin-large-coco-instance",
        backbone_pretrained_name_or_path=backbone_name,
        training_mode="frozen",
        out_indices=_default_dinov2_out_indices(backbone_name),
    )


def get_mask2former_processor(
    model_input_size: Tuple[int, int] | int,
) -> Mask2FormerImageProcessor:
    if isinstance(model_input_size, int):
        model_input_size = (model_input_size, model_input_size)
    return Mask2FormerImageProcessor(
        ignore_index=-1,
        do_resize=True,
        size={"height": int(model_input_size[0]), "width": int(model_input_size[1])},
        size_divisor=14,
        reduce_labels=False,
        do_rescale=True,
        image_mean=TRANSFORM_MEAN,
        image_std=TRANSFORM_STD,
        do_normalize=True,
    )


def largest_blob_by_area(mask: np.ndarray):
    """
    This function finds the largest blob (by area) in the passed binary mask and keeps the
    largest blob by area (if more than 1 was found)
    Args:
        mask (np.ndarray): Binary image (H,W) dtype uint8 with values 0 and 1.
    returns:
        The the binary mask for the largest block, the bounding box for the blob (xtl, ytl, xbr, ybr).

    """

    m = (mask > 0).astype(np.uint8) * 255

    # find only outer contours (ignore holes)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = mask.shape[:, 2]
        return mask, (0, 0, w, h)

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # mask of the largest blob
    largest_mask = np.zeros_like(mask)
    cv2.drawContours(
        largest_mask, [largest_contour], contourIdx=-1, color=1, thickness=cv2.FILLED
    )

    return largest_mask * mask, (x, y, x + w, y + h)


# Mask2Former instance segmentation class
class Mask2FormerInstanceSegmentation(VisionModel):
    def __init__(
        self,
        weights_path: str,
        model_name: str = "Mask2Former",
        label_map: Optional[Dict[int, str]] = None,  # to be read from the weights file
        model_input_size: Optional[
            Tuple[int, int]
        ] = None,  # to be read from the weights file
        confidence: float = DEFAULT_DETECTION_CONFIDENCE,
        device: torch.device = torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
        adjust_masks: bool = True,
    ):
        self._adjust_masks = adjust_masks
        super().__init__(
            weights_path,
            model_name,
            label_map,
            model_input_size,
            confidence,
            device,
        )

    def load(self) -> None:

        if self._model_input_size is None:
            # we need a valid model input size for Mask2Former model
            logging.error(
                f"Missing model input size! It was neither included in the weights file nor passed during instantiation! "
                f"Failed to instantiate {self._model_name} class."
            )
            return

        # Mask2Former model
        self._model = get_mask2former_instance_segmentation_model_with_dinov2_backbone(
            id2label=self._label_map, model_type="base", with_registers=False
        )

        # loading the PyTorch model
        try:
            self._model.load_state_dict(self._model_state_dict)
            self._model.to(self._device)
            self._model.eval()

        except Exception as ex:
            logging.error(
                f"Failed to load {self._model_name} model from the weights file: {repr(ex)}."
            )
            return

        self._hg_preprocessor = get_mask2former_processor(self._model_input_size)

        self._metadata = {
            "predict_masks": True,
            "resolution": self._model_input_size,
            "release_date": os.path.basename(self._weights_path).split("_")[
                0
            ],  # the model name starts with the release date
            "model_type": "Instance Segmentation",
            "model_name": self._model_name,
            "model_extra_info": "DINOv2 backbone",
            "names": self._label_map,
            "magnification": "4x"
            if "4x" in os.path.basename(self._weights_path)
            else "10x",  # 4x should be specified in the name
        }

        self._loaded = True

    def detect_batch(
        self,
        input_images_list: List[Union[Image.Image, np.ndarray]],
    ) -> List[Dict[str, list]]:

        if not self._loaded:
            logging.error(
                f"{self._model_name} model has not been initialized. Please initialize the class before detect()."
            )

            out: List[Dict[str, list]] = [
                {"boxes": [], "scores": [], "labels": [], "masks": []}
            ] * len(input_images_list)
            return out

        # convert to 3-channel images if needed, and store the original image dimensions for
        # post processing
        images_list: List[np.ndarray] = []
        org_img_dims: List[Tuple[int, int]] = []

        for img in input_images_list:
            if isinstance(img, Image.Image):
                img_array: np.ndarray = np.array(img)
            else:
                img_array: np.ndarray = img

            img_shape: tuple = img_array.shape
            if len(img_shape) < 3:
                images_list.append(cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB))
            else:
                images_list.append(img_array)
            org_img_dims.append(img_shape[:2])

        processed_imgs_dict = self._hg_preprocessor(images_list, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(
                pixel_values=processed_imgs_dict["pixel_values"].to(self._device)
            )

        results = self.postprocess(outputs, org_img_dims)
        # clear the CUDA cache
        torch.cuda.empty_cache()

        return results

    def postprocess(self, outputs, org_img_dims):

        processed_outputs = self._hg_preprocessor.post_process_instance_segmentation(
            outputs,
            threshold=self._confidence,
            target_sizes=org_img_dims,
            return_binary_maps=True,
        )

        if len(processed_outputs) == 0:
            # this should not happen and is not expected, return as if the model has not detected anything (for the whole list of images)
            return [{"boxes": [], "labels": [], "scores": [], "masks": []}] * len(
                processed_outputs
            )

        results = []

        for sample_index, processed_output in enumerate(processed_outputs):
            sample_dict = {}
            instance_to_label_map = {
                segment["id"]: segment["label_id"]
                for segment in processed_output["segments_info"]
            }
            instance_to_score_map = {
                segment["id"]: segment["score"]
                for segment in processed_output["segments_info"]
            }
            sorted_instance_ids = sorted(instance_to_label_map.keys())

            if len(sorted_instance_ids) > 0:
                # processed_output['segmentation'] is of dimension num_detections x H x W
                num_instances = processed_output["segmentation"].shape[0]

                if num_instances != len(instance_to_label_map):
                    logging.warning(
                        f"# of instance masks {num_instances} is not equal to the number of labels {len(instance_to_label_map)}!"
                    )
                    sorted_instance_ids = [
                        i for i in sorted_instance_ids if i < num_instances
                    ]

                # masks should be of dimension num_detections x 1 x H x W
                sample_dict["labels"] = [
                    instance_to_label_map[i] for i in sorted_instance_ids
                ]
                sample_dict["scores"] = [
                    instance_to_score_map[i] for i in sorted_instance_ids
                ]
                sample_dict["masks"] = to_numpy(
                    processed_output["segmentation"][sorted_instance_ids]
                )
                boxes = []
                masks = []  # masks after restricting them to the size of the bounding box
                for i in range(sample_dict["masks"].shape[0]):
                    instance_mask: np.ndarray = sample_dict["masks"][i]
                    if self._adjust_masks:
                        # post processing the mask for this instance, for this model, we have seen masks as disjoints blobs,
                        # check if there are more than one blob in the reported mask and keep the largest one
                        largest_mask, (xtl, ytl, xbr, ybr) = largest_blob_by_area(
                            instance_mask
                        )
                    else:
                        largest_mask = instance_mask
                        pos = np.where(largest_mask)
                        xtl = pos[1].min()
                        xbr = pos[1].max()
                        ytl = pos[0].min()
                        ybr = pos[0].max()

                    boxes.append([xtl, ytl, xbr, ybr])
                    masks.append(largest_mask[ytl:ybr, xtl:xbr].astype(float))

                sample_dict["boxes"] = boxes
                sample_dict["masks"] = masks

                # remap the output class IDs if needed
                if self._detected_class_ids_remap is not None and len(
                    sample_dict["labels"]
                ):
                    sample_dict["labels"] = np.vectorize(
                        lambda x: (
                            self._detected_class_ids_remap[x]
                            if x in self._detected_class_ids_remap
                            else x
                        )
                    )(sample_dict["labels"]).tolist()
            else:
                sample_dict = {"boxes": [], "labels": [], "scores": [], "masks": []}

            results.append(sample_dict)

        return results
