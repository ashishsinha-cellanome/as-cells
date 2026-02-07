from models.AbstractVisionModel import VisionModel
from utils.model_utils import to_numpy

import numpy as np
import cv2
from PIL import Image
import torch
from transformers import RTDetrForObjectDetection, RTDetrV2ForObjectDetection, RTDetrImageProcessor
# RT-DETRv2 with our custom DINOv2 + FPN backbone using output layers 2, 7 and 12
from models.dinov2_backbone_with_fpn import Dinov2BackBoneWithFPN
from models.custom_rt_detr_with_dinov2_backbone import RTDetrV2ConfigWithCustomBackBone, RTDetrV2ForObjectDetectionWithCustomBackbone
# DINOv2 + FPN backbone (the FPN is not trained, and will be loaded with the full model later)

import os
import logging
from typing import Tuple, List, Final, Optional, Dict, Union


DEFAULT_DETECTION_CONFIDENCE: Final[float] = 0.4
DEFAULT_MODEL_INPUT_SIZE: Final[int] = (640, 640) # for the RT-DETR model with the default Resnet50 backbone 

DEFAULT_RESIZE: Final[Dict[Tuple[int, int, str], Tuple[int, int]]] = {
    (2000, 1600): (1000, 800),
    (4512, 4512): (4512, 4512),
}

DEFAULT_CROP_CORNERS: Final[Dict[Tuple[int, int, str], List[List[int]]]] = {
    (2000, 1600): [
        [0, 0, 640, 640],
        [0, 160, 640, 800],
        [360, 0, 1000, 640],
        [360, 160, 1000, 800],
    ],
    (4512, 4512): [
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

DINOV2_MODEL_INPUT_SIZE: Final[int] = (672, 672) # for the RT-DETR model with DINOv2 backbone

DINOV2_RESIZE: Final[Dict[Tuple[int, int, str], Tuple[int, int]]] = {
    (2000, 1600): (2000, 1600),
    (4512, 4512): (4512, 4512),
}

DINOV2_CROP_CORNERS: Final[Dict[Tuple[int, int, str], List[List[int]]]] = {
    (2000, 1600): [
        [0, 0, 672, 672],
        [0, 464, 672, 1136],
        [0, 928, 672, 1600],
        [444, 0, 1116, 672],
        [444, 464, 1116, 1136],
        [444, 928, 1116, 1600],
        [888, 0, 1560, 672],
        [888, 464, 1560, 1136],
        [888, 928, 1560, 1600],
        [1328, 0, 2000, 672],
        [1328, 464, 2000, 1136],
        [1328, 928, 2000, 1600]
    ],
    (4512, 4512): [
        [0, 0, 672, 672],
        [0, 480, 672, 1152],
        [0, 960, 672, 1632],
        [0, 1440, 672, 2112],
        [0, 1920, 672, 2592],
        [0, 2400, 672, 3072],
        [0, 2880, 672, 3552],
        [0, 3360, 672, 4032],
        [0, 3840, 672, 4512],
        [480, 0, 1152, 672],
        [480, 480, 1152, 1152],
        [480, 960, 1152, 1632],
        [480, 1440, 1152, 2112],
        [480, 1920, 1152, 2592],
        [480, 2400, 1152, 3072],
        [480, 2880, 1152, 3552],
        [480, 3360, 1152, 4032],
        [480, 3840, 1152, 4512],
        [960, 0, 1632, 672],
        [960, 480, 1632, 1152],
        [960, 960, 1632, 1632],
        [960, 1440, 1632, 2112],
        [960, 1920, 1632, 2592],
        [960, 2400, 1632, 3072],
        [960, 2880, 1632, 3552],
        [960, 3360, 1632, 4032],
        [960, 3840, 1632, 4512],
        [1440, 0, 2112, 672],
        [1440, 480, 2112, 1152],
        [1440, 960, 2112, 1632],
        [1440, 1440, 2112, 2112],
        [1440, 1920, 2112, 2592],
        [1440, 2400, 2112, 3072],
        [1440, 2880, 2112, 3552],
        [1440, 3360, 2112, 4032],
        [1440, 3840, 2112, 4512],
        [1920, 0, 2592, 672],
        [1920, 480, 2592, 1152],
        [1920, 960, 2592, 1632],
        [1920, 1440, 2592, 2112],
        [1920, 1920, 2592, 2592],
        [1920, 2400, 2592, 3072],
        [1920, 2880, 2592, 3552],
        [1920, 3360, 2592, 4032],
        [1920, 3840, 2592, 4512],
        [2400, 0, 3072, 672],
        [2400, 480, 3072, 1152],
        [2400, 960, 3072, 1632],
        [2400, 1440, 3072, 2112],
        [2400, 1920, 3072, 2592],
        [2400, 2400, 3072, 3072],
        [2400, 2880, 3072, 3552],
        [2400, 3360, 3072, 4032],
        [2400, 3840, 3072, 4512],
        [2880, 0, 3552, 672],
        [2880, 480, 3552, 1152],
        [2880, 960, 3552, 1632],
        [2880, 1440, 3552, 2112],
        [2880, 1920, 3552, 2592],
        [2880, 2400, 3552, 3072],
        [2880, 2880, 3552, 3552],
        [2880, 3360, 3552, 4032],
        [2880, 3840, 3552, 4512],
        [3360, 0, 4032, 672],
        [3360, 480, 4032, 1152],
        [3360, 960, 4032, 1632],
        [3360, 1440, 4032, 2112],
        [3360, 1920, 4032, 2592],
        [3360, 2400, 4032, 3072],
        [3360, 2880, 4032, 3552],
        [3360, 3360, 4032, 4032],
        [3360, 3840, 4032, 4512],
        [3840, 0, 4512, 672],
        [3840, 480, 4512, 1152],
        [3840, 960, 4512, 1632],
        [3840, 1440, 4512, 2112],
        [3840, 1920, 4512, 2592],
        [3840, 2400, 4512, 3072],
        [3840, 2880, 4512, 3552],
        [3840, 3360, 4512, 4032],
        [3840, 3840, 4512, 4512]
    ],
}

# RT-DETR object detection class
class RtDetrObjectDetector(VisionModel):
    def __init__(
        self,
        weights_path: str,
        model_name: str = 'RT-DETR', 
        label_map: Optional[Dict[int, str]] = None, # to be read from the weights file
        model_input_size: Optional[Tuple[int, int]] = None, # to be read from the weights file
        confidence: float = DEFAULT_DETECTION_CONFIDENCE,
        device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        backbone_name_str: str = "default"
    ): 
        self._backbone = backbone_name_str
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

        if self._backbone.lower() == 'default':
            # RT-DETRv1 model with the default Resnet50 backbone
            # for the default backbone, we have trained RT-DETRv1
            # TODO: update the model to RT_DETRv2 and retrain
            
            # pretrained RT-DETRv1 model loaded from the original checkpoint
            pre_trained_rt_detr_model_checkpoint: str = "PekingU/rtdetr_r50vd_coco_o365"
            self._model = RTDetrForObjectDetection.from_pretrained(
                pre_trained_rt_detr_model_checkpoint,
                id2label=self._label_map,
                label2id=self._reverse_label_map,
                anchor_image_size=None,
                ignore_mismatched_sizes=True,
            )
        elif self._backbone.lower() == 'dinov2':
            
            dinov2_backbone = Dinov2BackBoneWithFPN.from_pretrained(
                "facebook/dinov2-base", 
                output_indices_for_fpn = [2, 7, 12]
            )
            dinov2_backbone_config = dinov2_backbone.config

            # pretrained RT-DETRv2 model loaded from the original checkpoint
            pretrained_rt_detr_model_checkpoint: str = "PekingU/rtdetr_v2_r18vd"
            pretrained_rt_detr_model = RTDetrV2ForObjectDetection.from_pretrained(
                    pretrained_rt_detr_model_checkpoint,
                    id2label = self._label_map, 
                    label2id = self._reverse_label_map,
                    ignore_mismatched_sizes=True
            )
            
            # update the config of pretrained_rt_detr_model to configure Dinov2 backbone with FPN
            # convert to RTDetrV2ConfigWithCustomBackBone class before replacing the backbone
            pretrained_model_config_dict = pretrained_rt_detr_model.config.to_dict()
            rt_detr_model_with_dinov2_backbone_config = RTDetrV2ConfigWithCustomBackBone(**pretrained_model_config_dict)
            # replace the backbone_config with that of DINOv2 with backbone
            rt_detr_model_with_dinov2_backbone_config.backbone_config = dinov2_backbone_config
    
            # build the model with random weights from the config
            self._model = RTDetrV2ForObjectDetectionWithCustomBackbone(rt_detr_model_with_dinov2_backbone_config)
            
        # loading the PyTorch model
        try:
            self._model.load_state_dict(self._model_state_dict)
            # this is for freezing the backbone, not really needed for inference time
            for param in self._model.model.backbone.parameters():
                param.requires_grad_(False)
            self._model.to(self._device)
            self._model.eval()

        except Exception as ex:
            logging.error(
                f"Failed to load {self._model_name} model from the weights file: {repr(ex)}."
            )
            return 

        self._hg_preprocessor = RTDetrImageProcessor(
            do_convert_annotations=True,
            do_resize=True,
            size={"width": self._model_input_size[0], "height": self._model_input_size[1]},
            reduce_labels=False,
            do_rescale=True, 
            do_normalize=True
        )
        
        # added for consistency with the YOLO model
        # metadata will be a dictionary with keys as 'predict_masks', 'resolution', 'release_date', 'model_type', 
        # 'model_name', 'model_extra_info', 'names', 'stride', ...
        # example: {'predict_masks': False
        #           'resolution': 640,
        #           'release_date': '20240415',
        #           'model_type': 'YOLO Detector',
        #           'model_name': 'YOLOv5m',
        #           'model_extra_info': 'V5 Medium',
        #           'names': {0: 'cell', 1: 'bead'},
        self._metadata = {
            'predict_masks': False, # detector model
            'resolution': self._model_input_size[0], # square input
            'release_date': os.path.basename(self._weights_path).split('_')[0], # the model name starts with the release date
            'model_type': 'Transformer Detector',
            'model_name': self._model_name,
            'model_extra_info': (
                'RT-DETRv1 With Resnet50 backbone' if self._backbone.lower() == 'default' else 
                (
                    'RT-DETRv2 With DINOv2 backbone' if self._backbone.lower() == 'dinov2' else ''
                )
            ),
            'names': self._label_map,
            'magnification': '4x' if '4x' in os.path.basename(self._weights_path) else '10x' # 4x should be specified in the name
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
            
            out: List[Dict[str, list]] = [{"boxes": [], "scores": [], "labels": []}] * len(input_images_list)
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
            outputs = self._model(pixel_values=processed_imgs_dict["pixel_values"].to(self._device))
        
        results = self.postprocess(outputs, org_img_dims)
        # clear the CUDA cache
        torch.cuda.empty_cache()
        
        return results

    def postprocess(self, outputs, org_img_dims):
        processed_outputs = self._hg_preprocessor.post_process_object_detection(
            outputs, threshold=self._confidence, target_sizes=org_img_dims
        )

        # processed_outputs is a list of len(input_images_list) dictionary elements, each dictionary containing the detections
        # for the input image in the input list with keys as 'boxes', 'labels' and 'scores', and values as
        # - 'boxes': a (num_detection, 4) torch.float32 tensor of bounding boxes in (xtl, ytl, xbr, ybr) format
        # - 'labels': a (num_detection, 1) torch.int64 tensor of class IDs
        # - 'scores': a (num_detection, 1) torch.float32 tensor of detection confidences
        
        if len(processed_outputs) == 0:
            # this should not happen and is not expected, return as if the model has not detected anything (for the whole list of images)
            results =  [
                {'boxes': [],
                 'labels': [],
                 'scores': [],
                }
            ] * len(outputs)
        else:
            # move to CPU and convert to numpy arrays before returning
            results = [{k: list(to_numpy(v)) for k, v in result.items()} for result in processed_outputs]

        # convert the bounding box coordinates to int
        for result in results:
            result['boxes'] = [box.astype(int) for box in result['boxes']]
            # map the class IDs
            if self._detected_class_ids_remap is not None and len(result["labels"]):
                result["labels"] = np.vectorize(
                    lambda x: (
                        self._detected_class_ids_remap[x]
                        if x in self._detected_class_ids_remap
                        else x
                    )
                )(result["labels"]).tolist()

        return results
