import os
import time
import pickle
from typing import Tuple, Union, List, Dict, Final

import numpy as np
from PIL import Image
import cv2
import torch
from pycocotools import mask as coco_mask_util
import albumentations as A
from transformers import AutoImageProcessor
# sys.path.append('references/detection')
# from references.detection.coco_utils  import CocoDetection
from torchvision.datasets import CocoDetection


# ADD profiling support 
from transformers import TrainingArguments, Trainer
import torch.profiler

## Pre-processing configurations
# The following configurations are used for pre-processing the images during the training. 
# probability of adding random blur and salt-and-pepper or additive gaussian noise to the training images
# see the image Transforms section below
P_NOISE = 0.25

# the lower and upper bounds for random scaling the images (and annotated masks) for training augmentation
MIN_RANDOM_SCALE = 0.7
MAX_RANDOM_SCALE = 1.0

# model input image size (should be square and divisable by 14 because of DINOv2 backbone)
MODEL_INPUT_SIZE: Final[int] = 672 # will not be used without Dinov2 backbone

MIN_AREA: Final[int] = 16

def get_transform(train: bool = True) -> A.core.composition.Compose:
    # no resizing is needed here as the training images are already cropped with the model input size, 
    # and also the Hugging face processor will handle any needed resizing
    if train:
        # random noise addition and random scale as defined above, 
        # we call these before PILToTensor as these classes 
        # operates on PIL images
        trsfms = [
            A.RandomScale(scale_limit=(MIN_RANDOM_SCALE - 1.0, MAX_RANDOM_SCALE - 1.0), p=1.0),
            A.PadIfNeeded(min_height = MODEL_INPUT_SIZE, min_width = MODEL_INPUT_SIZE, position = 'random'),
            A.RandomCrop(height = MODEL_INPUT_SIZE, width=MODEL_INPUT_SIZE, pad_if_needed=False, p=1.0),
            A.RandomRotate90(),
            A.Perspective(p=P_NOISE),
            A.RandomBrightnessContrast(p=P_NOISE),
            A.HueSaturationValue(p=P_NOISE),
            A.AdditiveNoise(p=P_NOISE, spatial_mode='per_pixel')
        ]        
    else:
        # test set 
        trsfms = [A.NoOp()]

    # bbox format is defined as COCO xywh
    return A.Compose(trsfms, bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=MIN_AREA))

# hugging face 
# we need to resize the input images (not needed as they are already in the correct MODEL_INPUT_SIZE x MODEL_INPUT_SIZE input size, 
# and normalize them, we use the already implemented Hugging Face preprocessor for this conversion
# index 0 will be used 

from transformers import RTDetrImageProcessor
hg_preprocessor: RTDetrImageProcessor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
# make the RT-DETR processor to be consistent with the DinoV2 backbone in terms of normalization and resizing
hg_preprocessor.do_normalize = True
hg_preprocessor.resample = 3
hg_preprocessor.size = {
    "height": MODEL_INPUT_SIZE,
    "width": MODEL_INPUT_SIZE
}

DATASET_BASE_PATH: str = '/global/home/ashish.sinha/cellanome/TRAINING_DATA/'
CROPPED_TRAIN_IMAGES_PATH: str = os.path.join(DATASET_BASE_PATH, 'images', 'train')
CROPPED_TEST_IMAGES_PATH: str = os.path.join(DATASET_BASE_PATH, 'images', 'valid')

# mapping between the class IDs and class names for the annotated data 
# labels start from 0
# TODO: changed the labels here
LABEL_MAP = {0: 'cell', 1: 'bead', 2: 'cell-adhered', 3: 'soma'}
REVERESE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


class CellMaskDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 dataset_coco: CocoDetection, 
                 processor: AutoImageProcessor, 
                 instance_segmentation: bool = False,
                 transforms: A.core.composition.Compose = None):
        self.dataset_coco = dataset_coco
        self.processor = processor
        self.instance_segmentation = instance_segmentation
        self.transforms = transforms

   
    def __len__(self):
        return len(self.dataset_coco)

    def __getitem__(self, idx):
        image, annotations = self.dataset_coco[idx]
        
        # Convert image to RGB numpy array
        image_array: np.ndarray = np.array(image.convert("RGB"))
        
        if isinstance(annotations, dict):
            # references.detection.coco_utils.CocoDetection class, annotations is a dictionary with keys image_id, annotations
            # annotations['annotations'] is a list of annotation dictionaries
            image_id: int = annotations['image_id']
            boxes: np.ndarray = np.array([record['bbox'] for record in annotations['annotations']])
            labels: List[int] = [record['category_id'] for record in annotations['annotations']]
            if self.instance_segmentation:
                masks: List[np.ndarray] = [coco_mask_util.decode(record['segmentation']) for record in annotations['annotations']]
        else:
            # torchvision.datasets.CocoDetection class, annotations is a list of dictionaries for each annotated object
            if len(annotations) > 0:
                image_id: int = annotations[0]['image_id']
            else:
                image_id: int = idx + 1
            boxes: np.ndarray = np.array([record['bbox'] for record in annotations])
            labels: List[int] = [record['category_id'] for record in annotations]
            if self.instance_segmentation:
                masks: List[np.ndarray] = [coco_mask_util.decode(record['segmentation']) for record in annotations]
        
        # apply augmentations, the second condition should not happen (all preprocessed images should at least have one object)
        if self.transforms and len(boxes) > 0:
            if self.instance_segmentation:
                # TODO: check to make sure the below mask transform would work
                transformed = self.transforms(image=image_array, bboxes=boxes, masks=masks, category=labels)
                img = transformed["image"]
                boxes = transformed["bboxes"]
                masks = transformed["masks"]
                labels = transformed["category"]
            else:
                transformed = self.transforms(image=image_array, bboxes=boxes, category=labels)
                img = transformed["image"]
                boxes = transformed["bboxes"]
                labels = transformed["category"]

        # reformat annotations
        annotations: List[dict] = []
        for i, bbox in enumerate(boxes):
            record = {
                "image_id": image_id,
                "category_id": int(labels[i]),
                "bbox": np.array([int(v) for v in bbox]),
                "iscrowd": 0,
                "area": bbox[2] * bbox[3],
            }
            if self.instance_segmentation:
                record["segmentation"] = coco_mask_util.encode(np.asarray(masks[i], order="F"))
                record["segmentation"]['counts'] = record["segmentation"]['counts'].decode('utf8')
            annotations.append(record)

        formatted_annotation: dict = {"image_id": image_id, "annotations": annotations,}
        if self.processor is not None:
            results = self.processor(images=img, annotations=formatted_annotation, return_tensors="pt")

            # image processor expands batch dimension, lets squeeze it
            results = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k, v in results.items()}
            return results
         
        return img, formatted_annotation

from tqdm import tqdm
# A flag to indicate whether the cropped images/annotations are for Mask RCNN or YOLO
CROPPED_FOR_MASK_RCNN: bool = True
if CROPPED_FOR_MASK_RCNN:
    annots_folder_name: str = "masks"
else:
    annots_folder_name: str = "labels"

CROPPED_TRAIN_ANNOTATIONS_PATH = os.path.join(DATASET_BASE_PATH, annots_folder_name, 'train')
CROPPED_TEST_ANNOTATIONS_PATH = os.path.join(DATASET_BASE_PATH, annots_folder_name, 'test')



def expand_bbox(
    bbox: Union[List[int], Tuple[int], np.ndarray], 
    image_width: int, 
    image_height: int, 
    percentage_to_expand_bbox_boundaries: float
) -> List[int]:
    
    xtl, ytl, xbr, ybr = bbox
    delta_x: int = int(percentage_to_expand_bbox_boundaries * (xbr - xtl) / 2)
    delta_y: int = int(percentage_to_expand_bbox_boundaries * (ybr - ytl) / 2)
    # expand by one pixel on each side at least to cover boundaries            
    delta_x = max(1, delta_x)
    delta_y = max(1, delta_y)
            
    xmin: int = max(0, xtl - delta_x)
    ymin: int = max(0, ytl - delta_y)
    xmax: int = min(image_width, xbr + delta_x)
    ymax: int = min(image_height, ybr + delta_y)
    
    return [xmin, ymin, xmax, ymax]

def convert_to_coco_api(images_path: str, 
                        annotations_path: str, 
                        id2label: Dict[int, str], # this is for the converted dataset, should start with 0 for YOLO, DETR models (RT/RF), 
                                                  # and 1 for Mask R-CNN/Mask2Fomer
                        instance_segmentation: bool = False,
                        percentage_to_expand_bbox_boundaries: float = 0.0):
    # load all images and annotations
    # the assumption is the image and its annotation use the same name
    imgs = list(sorted(os.listdir(images_path)))
    annotations = list(sorted(os.listdir(annotations_path)))
    start_class_id: int = min(list(id2label.keys()))
    
    if len(imgs) != len(annotations):
        print("[ERROR]: The list of images and masks are not consistent")
        return False, {}
    
    for i, img_filename in enumerate(imgs):
        # drop the image/mask filename extension 
        # (anything after the last '.' in the filename is considered as extension)
        img_name = ".".join(img_filename.strip().split('.')[:-1])
        annots_name = ".".join(annotations[i].strip().split('.')[:-1])
        if img_name != annots_name:
            print("[ERROR]: Inconsistent annotations file :{} found for image file: {}".format(annots_name, img_name))
            return False, {}
    # the index for annotations starts at 1
    annots_id = 1
    categories = set()
    json_annotations = {"images": [], "categories": [], "annotations": []}
    for idx in tqdm(range(len(imgs))):
        # load the image
        img_path = os.path.join(images_path, imgs[idx])
        # read the image, we only read the image to get the size of it
        # so no need to change the format (BGR to RGB) or convert to PIL 
        opencv_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image_height, image_width = opencv_img.shape[:2]
        img_dict = {}
        img_dict["id"] = idx + 1
        img_dict["file_name"] = imgs[idx]
        img_dict["height"] = image_height
        img_dict["width"] = image_width
        json_annotations["images"].append(img_dict)


        # annotations
        boxes: List[List[int]] = []
        labels: List[int] = []
        masks: List[np.ndarray] = []
        # load the annotations (masks)
        annots_path = os.path.join(annotations_path, annotations[idx])

        if instance_segmentation:
            # load the annotations
            filehandler = open(annots_path, 'rb')
            annots = pickle.load(filehandler)
            filehandler.close()
            
            for record in annots['annotations']:
                xtl, ytl, xbr, ybr = record['bbox']
                # no need to check the validity 
                if xtl >= xbr or ytl >= ybr:
                    continue
                # subtract 1 - start_class_id to adjust the labels
                # class IDs start from 1 in Mask R-CNN annotations, so if training YOLO/DETR, subtract 1
                # otherwise, leave the labels as is
                labels.append(int(record['category_id'] - 1 + start_class_id))
                mask: np.ndarray = coco_mask_util.decode(record['segmentation'])
                # mask in full image resolution
                full_res_mask: np.ndarray = np.zeros((image_height, image_width), np.uint8)
                full_res_mask[ytl:ybr, xtl:xbr] = mask
                
                masks.append(full_res_mask)
                
                expanded_bbox = expand_bbox(
                    bbox=[xtl, ytl, xbr, ybr],
                    image_width=image_width, 
                    image_height=image_height, 
                    percentage_to_expand_bbox_boundaries=percentage_to_expand_bbox_boundaries
                )
                # will convert bbox to xtl, ytl, w, h format below
                boxes.append(expanded_bbox)
        else:
            with open(annots_path,'r') as annot_file:
                for line in annot_file:
                    fields = line.strip().split(' ')
                    (label, center_x, center_y, w, h) = fields
                    xtl = int((float(center_x) - float(w) / 2.0) * image_width)
                    ytl = int((float(center_y) - float(h) / 2.0) * image_height)
                    xbr = int((float(center_x) + float(w) / 2.0) * image_width)
                    ybr = int((float(center_y) + float(h) / 2.0) * image_height)
                    if xtl >= xbr or ytl >= ybr:
                        continue
                    # convert the label to an integer from string
                    # for YOLO, labels start from 0, adjust the starting class IDs if needed
                    label = int(label + start_class_id) 
                    expanded_bbox: List[int] = expand_bbox(
                        bbox=[xtl, ytl, xbr, ybr],
                        image_width=image_width, 
                        image_height=image_height, 
                        percentage_to_expand_bbox_boundaries=percentage_to_expand_bbox_boundaries
                    )
                    # will convert bbox to xtl, ytl, w, h format below
                    boxes.append(expanded_bbox)
                    labels.append(label)
        
        for i, box in enumerate(boxes):
            record = {}
            record["image_id"] = idx + 1
            record['category_id'] = labels[i] 
            categories.add(record['category_id'])
            if instance_segmentation:
                record["segmentation"] = coco_mask_util.encode(np.asarray(masks[i], order="F"))
                record["segmentation"]['counts'] = record["segmentation"]['counts'].decode('utf8')
            xmin, ymin, xmax, ymax = [int(v) for v in box]
            #convert to xywh
            record['bbox'] = [xmin, ymin, xmax - xmin, ymax - ymin]
            record["area"] = (ymax - ymin) * (xmax - xmin)
            record["iscrowd"] = 0
            record["id"] = annots_id
                
            json_annotations["annotations"].append(record)
            annots_id += 1 
            
    json_annotations["categories"] = [{"id": i, "name": id2label[i], "supercategory": "biology"} for i in sorted(categories)]
    return True, json_annotations


# success, json_annots = convert_to_coco_api(images_path=CROPPED_TRAIN_IMAGES_PATH, 
#                                            annotations_path=CROPPED_TRAIN_ANNOTATIONS_PATH, 
#                                            id2label=LABEL_MAP,
#                                            instance_segmentation = CROPPED_FOR_MASK_RCNN)
# with open(os.path.join(DATASET_BASE_PATH, 'train_annotations.json'), 'w') as file:
#     json.dump(json_annots, file)

# success, json_annots = convert_to_coco_api(images_path=CROPPED_TEST_IMAGES_PATH, 
#                                            annotations_path=CROPPED_TEST_ANNOTATIONS_PATH, 
#                                            id2label=LABEL_MAP,
#                                            instance_segmentation = CROPPED_FOR_MASK_RCNN)
# with open(os.path.join(DATASET_BASE_PATH, 'test_annotations.json'), 'w') as file:
#     json.dump(json_annots, file)

# read the prepared json annotations in COCO format as COCO dataset objects
train_dataset_coco = CocoDetection(
    root=CROPPED_TRAIN_IMAGES_PATH, 
    annFile=os.path.join(DATASET_BASE_PATH, 'train_annotations.json'), 
    transforms=None
)
test_dataset_coco = CocoDetection(
    root=CROPPED_TEST_IMAGES_PATH, 
    annFile=os.path.join(DATASET_BASE_PATH, 'valid_annotations.json'), # valid_annotations.json
    transforms=None
)

train_dataset = CellMaskDataset(
    dataset_coco=train_dataset_coco, 
    processor=hg_preprocessor,
    instance_segmentation=False, # no need to set to True for training a DETR model as it adds to runtime computation! set True for testing
    transforms=get_transform(True)
)

test_dataset = CellMaskDataset(
    dataset_coco=test_dataset_coco, 
    processor=hg_preprocessor,
    instance_segmentation=False, # no need to set to True for training a DETR model as it adds to runtime computation
    transforms=get_transform(False)
)


# the path for loading and saving model checkpoints
MODEL_PATH: str = 'checkpoints/rt_detr_HF'
DINOV2_BACKBONE_INITIAL_CHECKPOINT: str = 'dinov2_backbone_with_fpn'
CUSTOM_RT_DETR_WITH_DINOV2_BACKBONE_INITIAL_CHECKPOINT: str = 'custom_rt_detr_with_dinov2_backbone'

from models.dinov2_backbone_with_fpn import Dinov2BackBoneWithFPNConfig, Dinov2BackBoneWithFPN
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
from models.custom_rt_detr_with_dinov2_backbone import RTDetrV2ConfigWithCustomBackBone, RTDetrV2ForObjectDetectionWithCustomBackbone

if not os.path.exists(os.path.join(MODEL_PATH, DINOV2_BACKBONE_INITIAL_CHECKPOINT)):
    dinov2_backbone = Dinov2BackBoneWithFPN.from_pretrained(
        "facebook/dinov2-base", 
        # first_layer_dims = (48, 48), 
        output_indices_for_fpn = [4, 8, 12]
    )
    dinov2_backbone.save_pretrained(os.path.join(MODEL_PATH, DINOV2_BACKBONE_INITIAL_CHECKPOINT))
else:
    print("Dinov2 with FPN backbone initial checkpoint already created!")

if not os.path.exists(os.path.join(MODEL_PATH, CUSTOM_RT_DETR_WITH_DINOV2_BACKBONE_INITIAL_CHECKPOINT)):
    id2label = LABEL_MAP
    label2id = {v: k for k, v in id2label.items()}
    pretrained_rt_detr_model_checkpoint: str = "PekingU/rtdetr_v2_r18vd"
    pretrained_rt_detr_model = RTDetrV2ForObjectDetection.from_pretrained(
            pretrained_rt_detr_model_checkpoint,
            id2label = id2label, 
            label2id = label2id,
            ignore_mismatched_sizes=True
        )
    
    # check the backnone output dims
    print("Pretrained models encoder input dims: ", [pretrained_rt_detr_model.model.encoder_input_proj[i][0].weight.shape[1] 
                                  for i in range(len(pretrained_rt_detr_model.model.encoder_input_proj))])
    print("Pretrained config encoder input dims: ", pretrained_rt_detr_model.model.backbone.intermediate_channel_sizes)
    
    # load our custom DINOv2 bakbone with FPN from the initial checkpoint
    dinov2_backbone_config = Dinov2BackBoneWithFPNConfig.from_pretrained(os.path.join(MODEL_PATH, DINOV2_BACKBONE_INITIAL_CHECKPOINT))
    print("DINOv2 model checkpoint of the backbone: ", dinov2_backbone_config.dinov2_pretrained_backbone_name_or_path)
    dinov2_backbone = Dinov2BackBoneWithFPN.from_pretrained(os.path.join(MODEL_PATH, DINOV2_BACKBONE_INITIAL_CHECKPOINT))
    
    # update the config of pretrained_rt_detr_model to configure Dinov2 backbone with FPN
    # convert to RTDetrV2ConfigWithCustomBackBone class before replacing the backbone
    pretrained_model_config_dict = pretrained_rt_detr_model.config.to_dict()
    rt_detr_model_with_dinov2_backbone_config = RTDetrV2ConfigWithCustomBackBone(**pretrained_model_config_dict)
    # replace the backbone_config with that of DINOv2 with backbone
    rt_detr_model_with_dinov2_backbone_config.backbone_config = dinov2_backbone_config
    
    # build the model (with random weights), we do this to make sure the config is correct
    test_model = RTDetrV2ForObjectDetectionWithCustomBackbone(rt_detr_model_with_dinov2_backbone_config)
    print("Reconstructed models encoder input dims: ", [test_model.model.encoder_input_proj[i][0].weight.shape[1] 
                                                        for i in range(len(test_model.model.encoder_input_proj))])
    print("Modified config encoder input dims: ", test_model.model.backbone.intermediate_channel_sizes)
    
    # now replce the backbone in the model and update its config with the new one before saving
    pretrained_rt_detr_model.config = rt_detr_model_with_dinov2_backbone_config
    pretrained_rt_detr_model.model.backbone = dinov2_backbone
    # save
    pretrained_rt_detr_model.save_pretrained(os.path.join(MODEL_PATH, CUSTOM_RT_DETR_WITH_DINOV2_BACKBONE_INITIAL_CHECKPOINT))
    print("backbone_config type for the saved model: ", type(pretrained_rt_detr_model.config.backbone_config))
else:
    print("The custom RT-DETR with DINOv2 backbone initial checkpoint already created!")


TRAIN_BATCH_SIZE = 64
LEARNING_RATE = 5e-5
NUM_EPOCHS = 8

def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    return data

# define training and validation data loaders
train_data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size = TRAIN_BATCH_SIZE, shuffle = True, num_workers = 4,
    collate_fn = collate_fn)

test_data_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size = 1, shuffle = False, num_workers = 4,
    collate_fn = collate_fn)

print('Training data includes %d annotated images.' %len(train_dataset))
print('Test data includes %d annotated images.' %len(test_dataset))


from transformers.image_transforms import center_to_corners_format

def convert_bbox_yolo_to_pascal(boxes, image_size):
    """
    Convert bounding boxes from YOLO format (x_center, y_center, width, height) in range [0, 1]
    to Pascal VOC format (x_min, y_min, x_max, y_max) in absolute coordinates.

    Args:
        boxes (torch.Tensor): Bounding boxes in YOLO format
        image_size (Tuple[int, int]): Image size in format (height, width)

    Returns:
        torch.Tensor: Bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max)
    """
    # convert center to corners format
    boxes = center_to_corners_format(boxes)

    # convert to absolute coordinates
    height, width = image_size
    boxes = boxes * torch.tensor([[width, height, width, height]])
    return boxes

COLORS = [(0, 0, 0), (0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (255, 0, 255)]

def show_sample(idx, dataset):
    sample = dataset[idx]
    if isinstance(sample, tuple):
        image, annotations = sample
        boxes = np.array([t['bbox'] for t in annotations['annotations']])
        if len(boxes) > 0:
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]
        labels = np.array([record['category_id'] for record in annotations['annotations']])
        if len(annotations['annotations']) and 'segmentation' in  annotations['annotations'][0]:
            masks = np.array([coco_mask_util.decode(record['segmentation']) for record in annotations['annotations']])
        else:
            masks = None
    else:
        set_mean = torch.tensor(dataset.processor.image_mean)
        set_var = torch.tensor(dataset.processor.image_std)
        image = (set_mean + sample['pixel_values'].permute(1, 2, 0) * set_var).mul(255).byte().numpy().copy()
        boxes = convert_bbox_yolo_to_pascal(sample['labels']['boxes'], 
                                            (sample['labels']['size'][0].item(), sample['labels']['size'][0].item())).numpy().astype(int)
        labels = sample['labels']['class_labels'].numpy().tolist()
        masks = None
                   
                
    for i in range(len(labels)):
        # the bounding box
        (xtl, ytl, xbr, ybr) = boxes[i]
        # use green color for masks
        color = COLORS[(labels[i] + 1) % len(COLORS)] # add 1 to be consistent with Mask R-CNN colors/labels
        if masks is not None:
            color_mask = color * np.repeat(np.expand_dims(masks[i][ytl:ybr, xtl:xbr], axis=2), 3, axis=2)
            blended = 0.4 * color_mask
            blended[color_mask == 0] = image[ytl:ybr, xtl:xbr][color_mask == 0]
            blended[color_mask > 0] += 0.6 * image[ytl:ybr, xtl:xbr][color_mask > 0]

            # store the blended ROI in the original image
            image[ytl:ybr, xtl:xbr] = blended.astype(np.uint8)
        
        if labels[i] in LABEL_MAP:
            text = LABEL_MAP[labels[i]]
        else:
            print('Incorrect ID was found %s' %labels[i])
            text = 'Unknown'
        
        # add label
        cv2.putText(image, text, (xtl, ytl + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # add the bounding box with yellow color
        color = (255, 255, 0)
        cv2.rectangle(image, (xtl, ytl), (xbr, ybr), color, 1)
        
    print(f"Image size (W, H): {image.shape[1]}, {image.shape[0]}")
    # convert to PIL image to display
    return Image.fromarray(image)


pre_trained_model_checkpoint: str = os.path.join(MODEL_PATH, CUSTOM_RT_DETR_WITH_DINOV2_BACKBONE_INITIAL_CHECKPOINT)
# pre_trained_model_checkpoint: str = os.path.join(MODEL_PATH, "checkpoint-53574")
model = RTDetrV2ForObjectDetectionWithCustomBackbone.from_pretrained(pretrained_model_name_or_path=pre_trained_model_checkpoint)

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print('Device available:' , device)

# move model to the right device
model.train()
model.to(device)

"""
To test the model on some inputs, 
train_iter = iter(train_data_loader)
train_batch = next(train_iter)
train_batch['pixel_values'] = train_batch['pixel_values'].to(device)
train_batch['labels'] = [{k: v.to(device) for k, v in sample.items()} for sample in train_batch['labels']]
with torch.no_grad():
    out = model(**train_batch)
"""

from pycocotools.cocoeval import COCOeval

def to_cpu_device(tensor):
    """
    A function to move a CUDA torch input to CPU memory.
    Args:
        tensor (torch tensor).
    Returns:
        Moved to CPU.
    """
    return tensor.detach().cpu() if tensor.requires_grad else tensor.cpu()

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def convert_preds_to_coco(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue
        
        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()
         
        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": boxes[k],
                    "score": scores[k],
                }
                for k in range(len(scores))
            ]
        )
    return coco_results

from dataclasses import dataclass

@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


class MAPEvaluator:

    def __init__(self, data_loader, image_processor, threshold=0.4, max_dets=100):
        self.image_processor = image_processor
        self.threshold = threshold
        self.test_dataset_coco = data_loader.dataset.dataset_coco
        self.all_predictions = []
        self.all_image_ids = []
        self.max_dets = max_dets

    def collect_image_sizes(self, batch):
        """Collect image sizes across a batch of the dataset as a list of batch_size size (list of 2 elements, height and width)."""
        # we should use size and not 'org_size' this is because we directly use ground truth from the COCO test dataset, which already
        # have the images resized to size
        batch_image_sizes = [to_cpu_device(x["size"]).numpy().tolist() for x in batch]
        return batch_image_sizes

    def collect_predictions(self, batch_predictions, batch_image_sizes):
        post_processed_predictions = []
        batch_logits, batch_boxes = batch_predictions[1], batch_predictions[2]
        output = ModelOutput(logits=batch_logits, pred_boxes=batch_boxes)
        # post_processed_output is a list of batch_size dictionaty elements, each dictionary containing the detections
        # for the images in the batch with keys as 'boxes', 'labels' and 'scores', and values as
        # - 'boxes': a (num_detection, 4) torch.float32 tensor of bounding boxes in (xtl, ytl, xbr, ybr) format
        # - 'labels': a (num_detection, 1) torch.int64 tensor of class IDs
        # - 'scores': a (num_detection, 1) torch.float32 tensor of detection confidences
        post_processed_output = self.image_processor.post_process_object_detection(
            output, threshold=self.threshold, target_sizes=batch_image_sizes
        )        
        # move the detections to CPU
        post_processed_output = [{k: to_cpu_device(v) for k, v in outputs.items()} for outputs in post_processed_output]
        post_processed_predictions.extend(post_processed_output)
        return post_processed_predictions
    
    # metrics should be a dictionary with the following keys: 
    # 'map', 'map_50', 'map_75', 'map_small', 'map_medium', 'map_large', 'mar_1', 'mar_10', 'mar_100', 'mar_small', 'mar_medium', 'mar_large', 
    # 'map_cell', 'mar_100_cell', 'map_bead', 'mar_100_bead', 'map_soma', 'mar_100_soma'
    @torch.no_grad()
    def __call__(self, evaluation_results, compute_result):

        metrics = {
                'map': -1.0, 'map_50': -1.0, 'map_75': -1.0, 
                'map_small': -1.0, 'map_medium': -1.0, 'map_large': -1.0, 
                'mar_1': -1.0, 'mar_10': -1.0, 'mar_' + str(self.max_dets): -1.0, 
                'mar_small': -1.0, 'mar_medium': -1.0, 'mar_large': -1.0
            }

        batch_predictions, batch_targets = evaluation_results.predictions, evaluation_results.label_ids 
        
        batch_image_sizes = self.collect_image_sizes(batch_targets)
        post_processed_batch_predictions = self.collect_predictions(batch_predictions, batch_image_sizes)
        results = {int(target["image_id"].item()): output for target, output in zip(batch_targets, post_processed_batch_predictions)}    
        results = convert_preds_to_coco(results)
        self.all_predictions.extend(results)
        self.all_image_ids += [int(target["image_id"].item()) for target in batch_targets]
        
        if compute_result and len(self.all_predictions) > 0:
            n_threads = torch.get_num_threads()
            # FIXME remove this and make paste_masks_in_image run on the GPU
            torch.set_num_threads(1)
            cpu_device = torch.device("cpu")
    
            coco_gt = self.test_dataset_coco.coco  
            coco_gt.dataset['info'] = {}
            coco_dt = coco_gt.loadRes(self.all_predictions)  # init predictions api
    
            evaluator_time = time.time()
    
            # bounding box evaluation
            coco_evaluator_bbox = COCOeval(coco_gt, coco_dt, "bbox")
            coco_evaluator_bbox.params.maxDets = [1, 10, self.max_dets]
            coco_evaluator_bbox.params.imgIds = self.all_image_ids
            coco_evaluator_bbox.evaluate()
            coco_evaluator_bbox.accumulate()
            coco_evaluator_bbox.summarize()
            evaluator_time = time.time() - evaluator_time
    
            print("evaluator_time:", evaluator_time)

            torch.set_num_threads(n_threads)
            for i, key in enumerate(metrics.keys()):
                metrics[key] = coco_evaluator_bbox.stats[i]
            
            metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

            # clear up the history
            self.all_predictions = []
            self.all_image_ids = []

        if compute_result:
            # clear up the history
            self.all_predictions = []
            self.all_image_ids = []
            
                    
        return metrics


# we should use a very small threshold here to allow MAPEvaluator to use all the predictions with their scores
eval_compute_metrics_fn = MAPEvaluator(data_loader=test_data_loader, image_processor=hg_preprocessor, threshold=0.05, max_dets=100)


if __name__ == '__main__':

    PROFILER_LOG_DIR = "checkpoints/rt_detr_HF/profiler_logs"

    profiler_schedule = torch.profiler.schedule(
        skip_first=5,  # Warmup batches
        wait=0,
        warmup=1,
        active=10,     # Batches to profile
        repeat=1       # Only run this once
    )
    profiler_handler = torch.profiler.tensorboard_trace_handler(
        PROFILER_LOG_DIR
    )
    profiler_config = {
        "activities": [torch.profiler.ProfilerActivity.CPU, 
                    torch.profiler.ProfilerActivity.CUDA],
        "schedule": profiler_schedule,
        "on_trace_ready": profiler_handler,
        "record_shapes": True,
        "with_stack": True
    }
    training_args = TrainingArguments(
        output_dir=MODEL_PATH,
        num_train_epochs=NUM_EPOCHS,
        max_grad_norm=0.1,
        learning_rate=LEARNING_RATE,
        warmup_steps=300,
        lr_scheduler_type = "reduce_lr_on_plateau",
        per_device_train_batch_size=TRAIN_BATCH_SIZE ,
        per_device_eval_batch_size=1,
        torch_empty_cache_steps=int(len(train_dataset) / (5 * TRAIN_BATCH_SIZE)), 
        batch_eval_metrics=True,
        dataloader_num_workers=4,
        metric_for_best_model="eval_map",
        greater_is_better=True,
        load_best_model_at_end=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        bf16=True,
        report_to = 'wandb',
        # torch_profiler="pytorch",
        # torch_profiler_config=profiler_config,
        max_steps=20,
        seed=42,
    )
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=hg_preprocessor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,

    )

    trainer.train()