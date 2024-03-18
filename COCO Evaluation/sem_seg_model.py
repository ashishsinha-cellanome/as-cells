import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.models import segmentation as seg_models
from torchvision.transforms import functional as F
import os
import time
import logging
from typing import Tuple, List, Final, Optional, Dict, Union


MODEL_WEIGHTS_PATH: Final[str] = '/home/cellareye/Cellanome/dl-mehdi/Semantic Segmentation/checkpoints/bf_fcn_resnet50_1cycle_lrs_4_bs_10_epochs.pt'
LABEL_MAP: Final[Dict[int, str]] = {1: 'cytoplasm'}

MODEL_INPUT_MAX_SIZE: Final[int] = 1024
MODEL_INPUT_MIN_SIZE: Final[int] = 1024 

INPUT_MEAN: Final[float] = 0.449 
INPUT_STD:  Final[float] = 0.226 

# module-level variables and constants
OPEN_CV_BLACK: Final[Tuple[int, int, int]] = (0, 0, 0)
OPEN_CV_WHITE: Final[Tuple[int, int, int]] = (255, 255, 255)
OPEN_CV_RED: Final[Tuple[int, int, int]] = (0, 0, 255)
OPEN_CV_BLUE: Final[Tuple[int, int, int]] = (255, 0, 0)
OPEN_CV_GREEN: Final[Tuple[int, int, int]] = (0, 195, 0)
OPEN_CV_BRIGHT_GREEN: Final[Tuple[int, int, int]] = (0, 255, 0)
OPEN_CV_MAGENTA: Final[Tuple[int, int, int]] = (255, 0, 255)
OPEN_CV_YELLOW: Final[Tuple[int, int, int]] = (0, 255, 255)
OPEN_CV_CYAN: Final[Tuple[int, int, int]] = (255, 255, 0)
OPEN_CV_ORANGE: Final[Tuple[int, int, int]] = (0, 165, 255)
OPEN_CV_GRAY: Final[Tuple[int, int, int]] = (169, 169, 169)
GREEN_COLOR_MULTIPLIER: Final[Tuple[float, float, float]] = (0.6, 1.0, 0.6)
# all colors in a specific order for debug image coloring
COLORS: Final[List[Tuple[int, int, int]]] = [
    OPEN_CV_BLACK,
    OPEN_CV_GREEN,
    OPEN_CV_WHITE,
    OPEN_CV_BLUE,
    OPEN_CV_RED,
    OPEN_CV_BRIGHT_GREEN,
    OPEN_CV_MAGENTA,
    OPEN_CV_CYAN,
    OPEN_CV_YELLOW]

def show_detections(img: np.ndarray, mask: np.ndarray):
    
    image = img.copy()
    class_ids: List[int] = np.unique(mask)[1:]
    if len(image.shape) < 3:
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2)
    
    mask = np.repeat(np.expand_dims(mask, axis=2), 3, axis=2)
    
    for class_id in class_ids:
        class_pixels: Tuple[np.ndarray, np.ndarray] = np.where(mask==class_id)
        image[class_pixels] = 0.6 * image[class_pixels] + 0.4 * (mask * COLORS[class_id])[class_pixels]
    
    return image.astype(np.uint8)
    
# Semantic segmentation class (FCN or DeepLab V3 with Resnet50 backbones are supported)
class SemanticSegmentation:
    def __init__(
        self,
        model_type: Optional[str] = 'FCN', # 'FCN' or 'DLV3' are currently supported
        weights_path: Optional[str] = MODEL_WEIGHTS_PATH,
        label_map: Optional[Dict[int, str]] = LABEL_MAP,
    ):
        
        self.model_type = model_type
        if model_type not in ['FCN', 'DLV3']:
            # the default model is the FCN model (in case an invalid model type is passed)
            logging.warning(f"Invalid model type was passed: {model_type}! Trying to use the default FCN model")
            self.model_type = 'FCN'
        self._weights_path: str = str(weights_path)
        self._label_map: Dict[int, str] = label_map
        self._reverse_label_map: Dict[str, int] = {
                value: key for key, value in self._label_map.items()
                }
        
        # mean and variance to normalize the input
        self._input_mean: float = INPUT_MEAN
        self._input_std: float = INPUT_STD 
        
        # the larger and the smaller side sizes of the input image to the model
        self._max_input_size: int = MODEL_INPUT_MAX_SIZE
        self._min_input_size: int = MODEL_INPUT_MIN_SIZE
        
        logging.info(f"Mapping between class IDs and class names: {self._label_map}")
        
        # Semantic segmentation model
        if model_type == 'DLV3':
            self._model = seg_models.deeplabv3_resnet50(weights=seg_models.DeepLabV3_ResNet50_Weights.DEFAULT)
        else:   
            self._model = seg_models.fcn_resnet50(weights=seg_models.FCN_ResNet50_Weights.DEFAULT)

        # Change final layer to number of classes + 1 
        # note that FCN and DeepLab models have different number of channels for this layer 
        # (512 vs 256 for DeepLab)
        num_channels: int =  self._model.classifier[4].state_dict()['weight'].shape[1]
        self._model.classifier[4] = torch.nn.Conv2d(num_channels, len(self._label_map) + 1, 
                                                    kernel_size=(1, 1), stride=(1, 1)) 
       
        # available device
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # loading the PyTorch model
        try:
            logging.info(
                f"Loading PyTorch {model_type} Semantic Segmentation model from from {self._weights_path}. Setting to run on {self.device.type}."
            )
            
            self._model.load_state_dict(torch.load(self._weights_path))
            self._model.to(self.device)
            self._model.eval()
            
        except Exception as ex:
            self._model = None
            logging.error(
                f"Failed to load Semantic Segmentation model. Likely the paths to model .pt weights "
                f"{self._weights_path} is incorrect: {repr(ex)}."
            )
    
    
    # note that the passed image can be also a numpy array returned by 
    # cv2.imread(img_path, cv2.IMREAD_UNCHANGED), it does not necessarily have to be a PIL image
    # in fact OpenCV is slightly more efficient in reading the images
    def detect(self, image: Union[Image.Image, np.ndarray], log_time: bool = False, post_process: bool = False) -> np.ndarray:
        """
        The main function to detect the segmantic masks for the objects in the image.
        
        Args:
           mage (PIL.Image or numpy array): Input image, should have 8 bits per channel bit depth (np.uint8 in
               case of a numpy array). 
           log_time (bool): A flag to log the model run time. 
        
        Returns:
           A numpy array with np.uint16 elements for the semantic mask. Each pixel takes 0 ('bg') or the
               class_id of the object as specified in the label mask
        """
        
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()
        
        image_height, image_width = img.shape[:2]
        
        if self._model is None:
            logging.error(
                "Semantic Segmentation model has not been initialized. Please initialize the class before detect()."
            )
            return np.zeros(img.shape[:2], np.uint16)
        
        
        start: float = time.time()
        
        # make sure the image is a gray scale image (BGR or RGB does not matter)
        # then normalize
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
           
        img = (img / 255.0 - self._input_mean) / self._input_std
        
        # scale the image while keeping the aspect ratios
        larger_side_size: int = max(image_height, image_width)
        smaller_side_size: int = min(image_height, image_width)
        
        scale: float = max(larger_side_size / self._max_input_size, smaller_side_size / self._min_input_size)
        if scale > 1:
            img = cv2.resize(img, (int(image_width / scale), int(image_height / scale)))
        
        # the model expects a 3 channel input
        # FIXME: check how to improve this to accept a Gray scale image
        img = np.repeat(np.expand_dims(img, axis=2), 3, axis=2)
        # convert to tensor and add batch dimension
        image_tensor = F.to_tensor(img).unsqueeze(dim=0).to(self.device).float()
    
        with torch.no_grad(), torch.cuda.amp.autocast():
            output = self._model(image_tensor)["out"]
            mask = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        
        if scale > 1:
            mask = cv2.resize(mask, (image_width, image_height), interpolation = cv2.INTER_NEAREST)
        
        if post_process:
      	    pass
        
        elap: float = time.time() - start 
        if log_time:
            logging.info(f"{self.model_type} Semantic Segmentation took {elap:.4f} seconds")
        
        return mask


detector = SemanticSegmentation()

def run_sem_seg(
        input_image: np.ndarray, normalize_image: bool = True, bit_depth: int = 8, post_process: bool = True, plot_results: bool = False, 
) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
    
    # make a copy to not modify the input image
    img = input_image.copy()

    if len(img.shape) > 2:
        logging.warning(
            "Warning Semantic Segmentation model may suffer loss in precision due to conversion from RGB to grayscale"
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = (255 * img.astype(float) / (2 ** bit_depth - 1)).astype(np.uint8)

    if normalize_image:
        img = cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    
    st = time.time()    
    mask = detector.detect(image=img, post_process=post_process)
    
    et = time.time()

    if plot_results:
        return mask, et - st, show_detections(img, mask)
    
    return mask, et - st
