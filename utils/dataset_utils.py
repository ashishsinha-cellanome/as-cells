import os
import pickle
from typing import Tuple, Union, List, Dict, Final, Optional

from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision.datasets import CocoDetection
from pycocotools import mask as coco_mask_util
from torchvision.transforms import functional as F
import albumentations as A

MIN_AREA: Final[int] = 16

def get_transform(
    model_input_width: int, 
    model_input_height: int,
    min_random_scale: float,
    max_random_scale: float,
    p_noise: float, 
    org_images_in_model_input_size: bool = True,
    train: bool = True) -> A.core.composition.Compose:
    if train:
        # random noise addition and random scale as defined above, 
        # we call these before PILToTensor as these classes 
        # operates on numpy images
        trsfms = []
        if not org_images_in_model_input_size:
            trsfms = [A.Resize(
                height=model_input_height, width=model_input_width, 
                interpolation=cv2.INTER_CUBIC, mask_interpolation=cv2.INTER_NEAREST, area_for_downscale = "image", 
                p=1.0
            ), ] # needed if the cropped images were not created for the model input size
        trsfms.extend([
            A.RandomScale(
                scale_limit=(min_random_scale - 1.0, max_random_scale - 1.0), 
                interpolation=cv2.INTER_CUBIC, mask_interpolation=cv2.INTER_NEAREST, area_for_downscale = "image", 
                p=1.0
            ),
            A.RandomCrop(height = model_input_height, width=model_input_width, pad_if_needed=True, pad_position='random', p=1.0),
            A.RandomRotate90(),
            # A.Perspective(p=P_NOISE),
            A.RandomBrightnessContrast(p=p_noise),
            A.CoarseDropout(
                num_holes_range=(1, 0.02 * model_input_width * model_input_height),
                hole_height_range=(1, 1),
                hole_width_range=(1, 1), 
                p = p_noise
            ),
            A.GaussianBlur(
                sigma_limit=(0, 2.0),
                p = p_noise,
            ),
            A.AdditiveNoise(
                noise_type='gaussian', 
                noise_params={'mean_range': (0, 0), 'std_range': (0, 0.04)}, 
                p=p_noise, 
                spatial_mode='per_pixel'
            )
        ])        
    else:
        # test set 
        trsfms = [A.NoOp()]

    # bbox format is defined as COCO xywh
    return A.Compose(trsfms, bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=MIN_AREA))


def expand_bbox(
    bbox: Union[List[int], Tuple[int], np.ndarray], 
    image_width: int, 
    image_height: int, 
    percentage_to_expand_bbox_boundaries: float
) -> List[int]:
    
    xtl, ytl, xbr, ybr = bbox
    delta_x: int = int(percentage_to_expand_bbox_boundaries * (xbr - xtl) / 2.0)
    delta_y: int = int(percentage_to_expand_bbox_boundaries * (ybr - ytl) / 2.0)
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
                        annotations_in_mask_rcnn_format: bool = False,
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

        if annotations_in_mask_rcnn_format:
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
            if annotations_in_mask_rcnn_format:
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

class CellMaskDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 dataset_coco: CocoDetection, 
                 processor, 
                 classnames_to_class_ids_map: Dict[str, int] = None,
                 instance_segmentation: bool = True,
                 transforms: A.core.composition.Compose = None):
        self.dataset_coco = dataset_coco
        self.class_id_remap: Dict[int, int] = None
        if classnames_to_class_ids_map is not None:
            self.class_id_remap = {}
            for coco_class_id_to_classname_map in dataset_coco.coco.cats.values():
                coco_class_id: int = coco_class_id_to_classname_map['id']
                coco_classname: str = coco_class_id_to_classname_map['name']
                if coco_classname in classnames_to_class_ids_map:
                    # add the class ID to be remapped to the new class ID
                    # note that any object with a class ID different that what is listed in the values of classnames_to_class_ids_map
                    # or any class name in COCO not listed in classnames_to_class_ids_map keys will be ignored
                    self.class_id_remap[coco_class_id] = classnames_to_class_ids_map[coco_classname]
                else:
                    print(f"[WARN] Class ID: {coco_class_id} used in COCO datast with name: {coco_classname} is not included in the passed "
                          f"classnames_to_class_ids_map and objects of this class will be excluded from annotations!")
            print(f"[INFO] The class IDs in COCO will be remapped according to this mapping: {self.class_id_remap}")  
        self.processor = processor
        self.classnames_to_class_ids_map = classnames_to_class_ids_map
        self.instance_segmentation = instance_segmentation
        self.transforms = transforms

   
    def __len__(self):
        return len(self.dataset_coco)

    def __getitem__(self, idx):
        image, annotations = self.dataset_coco[idx]
        
        # Convert image to RGB numpy array (image is in PIL format returned by torchvision.datasets.CocoDetection)
        image_array: np.ndarray = np.array(image.convert("RGB"))
        
        
        # in torchvision.datasets.CocoDetection class, annotations is a list of dictionaries for each annotated object
        if len(annotations) > 0:
            image_id: int = annotations[0]['image_id']
        else:
            image_id: int = idx + 1

        
        annotations_to_keep: List[dict] = []
        if self.class_id_remap is not None:
            labels: List[int] = []
            for record in annotations:
                if record['category_id'] in self.class_id_remap:
                    annotations_to_keep.append(record)
                    labels.append(self.class_id_remap[record['category_id']])
        else:
            annotations_to_keep = annotations
            labels: List[int] = [record['category_id'] for record in annotations_to_keep]
        
        boxes: np.ndarray = np.array([record['bbox'] for record in annotations_to_keep])
        
        
        # NOTE: the following implementation for image augmentation, and then adjusting them for a specific model with the passed processor
        # is not efficient because we decode and then encode the masks from and to COCO RLE format multiple times
        # first we decode to apply the transforms (albumentation), then we encode again before passing to the processor function, in which 
        # we decode again (so one decoding and then encoding is redundant)
        # but this allows us to use the same CellMaskDataset and albumentations transforms for all different models, and only have the
        # processor as a model specific function
        if self.instance_segmentation:
            masks: List[np.ndarray] = [coco_mask_util.decode(record['segmentation']) for record in annotations_to_keep]
        
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
            if bbox[2] * bbox[3] == 0:
                # skip zero area invalid annotations post augmentation
                # NOTE: this may lead to having no annotated object in the image
                continue
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
            if isinstance(results, dict):
                # image processor for hugging face expands batch dimension, lets squeeze it
                results = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k, v in results.items()}
            return results
         
        return img, formatted_annotation


def mask_rcnn_processor(
    images: np.ndarray, # the name images is chosen to be consistent with hugging face processor, here images is only one numpy image
    annotations: dict, # annotations in COCO format, should have keys and 'image_id' and 'annotations'
    return_tensors: str="pt" # not used, included to be consistent with hugging face processor arguments
):
    # the following is the torchvision imeplementation, we can use the torch functional instead a below
    # import references.detection.transforms as T
    # transform = T.Compose([T.PILToTensor(), T.ConvertImageDtype(torch.float)])
    # img_tensor: torch.tensor = transform(Image.fromarray(images), target=None)[0]

    img_tensor: torch.tensor = F.pil_to_tensor(Image.fromarray(images))
    img_tensor = F.convert_image_dtype(img_tensor, torch.float)
    
    num_objs = len(annotations['annotations'])
    iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

    boxes = []
    labels = []
    masks = []
    areas = []
    for annots in annotations['annotations']:
        labels.append(annots['category_id'])
        areas.append(annots['area'])
        boxes.append(annots['bbox'])
        masks.append(np.expand_dims(coco_mask_util.decode(annots['segmentation']), axis=0))

    # combine all the masks
    masks = np.concatenate(masks, axis=0)
    labels = np.array(labels)
    boxes = np.array(boxes)
    areas = np.array(areas)

    # convert from xywh in COCO to xyxy format
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    
    
    # convert everything into a torch.Tensor
    labels = torch.as_tensor(labels, dtype=torch.int64)    
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    # masks, at this point the masks are binary (0, 1) so uint8 is fine
    masks = torch.as_tensor(masks, dtype=torch.uint8)
    areas = torch.as_tensor(areas, dtype=torch.float32)
    
    target = {}
    target["boxes"] = boxes
    target["labels"] = labels
    target["masks"] = masks
    target["image_id"] = torch.tensor(annotations['image_id'])
    target["area"] = areas
    target["iscrowd"] = iscrowd

    return img_tensor, target