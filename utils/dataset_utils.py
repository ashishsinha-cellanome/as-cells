import os
import pickle
from typing import Tuple, Union, List, Dict, Final, Optional

from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
import pandas as pd

import torch
from torchvision.datasets import CocoDetection
from pycocotools import mask as coco_mask_util
from torchvision.transforms import functional as F
import albumentations as A

from .json_parser import CellMaskDataset

MIN_AREA: Final[int] = 16
MAX_IMAGE_SIDE: Final[int] = 4512

def build_transforms_from_config(transform_list: List[Dict]) -> List[A.BasicTransform]:
    """Builds a list of Albumentations transforms from a configuration list."""
    transforms = []
    for t_config in transform_list:
        name = t_config['name']
        params = t_config.get('params', {})
        
        # Handle special cases or type conversions if necessary
        # For example, converting lists to tuples if strictly required by library (though A usually handles lists)
        
        if hasattr(A, name):
            transform_cls = getattr(A, name)
            # Filter out None params or handle them? usually **params is enough
            try:
                transforms.append(transform_cls(**params))
            except Exception as e:
                print(f"[ERROR] Failed to initialize transform {name} with params {params}: {e}")
        else:
            print(f"[WARN] Transform {name} not found in albumentations.")
            
    return transforms

def get_transform(
    model_input_width: int, 
    model_input_height: int,
    min_random_scale: float,
    max_random_scale: float,
    p_noise: float, 
    org_images_in_model_input_size: bool = True,
    train: bool = True,
    transforms_config: Optional[List[Dict]] = None) -> A.core.composition.Compose:
    
    trsfms = []
    
    if transforms_config is not None:
        # Use the provided configuration
        trsfms = build_transforms_from_config(transforms_config)
    
    elif train:
        # Legacy/Default hardcoded transforms
        # random noise addition and random scale as defined above, 
        # we call these before PILToTensor as these classes 
        # operates on numpy images
        trsfms = []
        # TODO: uncomment this if something happens
        # if not org_images_in_model_input_size:
        #     trsfms = [A.Resize(
        #         height=model_input_height, width=model_input_width, 
        #         interpolation=cv2.INTER_CUBIC, mask_interpolation=cv2.INTER_NEAREST, area_for_downscale = "image", 
        #         p=1.0
        #     ), ] # needed if the cropped images were not created for the model input size
        trsfms.extend([
            A.RandomScale(
                scale_limit=(min_random_scale - 1.0, max_random_scale - 1.0), 
                # interpolation=cv2.INTER_CUBIC, mask_interpolation=cv2.INTER_NEAREST, area_for_downscale = "image", 
                p=1.0
            ),
            A.PadIfNeeded(min_height = model_input_height, min_width = model_input_width, position = 'random'),
            A.RandomCrop(height = model_input_height, width=model_input_width, pad_if_needed=False, p=1.0),
            A.RandomRotate90(),
            A.Perspective(p=p_noise),
            A.RandomBrightnessContrast(p=p_noise),
            # breakpoint(),
            # TODO: changed the ratio of coarse dropout from 0.02 -> 0.0002
            # A.CoarseDropout(
            #     num_holes_range=(1, int(0.0002 * model_input_width * model_input_height)),
            #     hole_height_range=(1, 1),
            #     hole_width_range=(1, 1), 
            #     p = p_noise
            # ),
            # A.GaussianBlur(
            #     sigma_limit=(0, 2.0),
            #     p = p_noise,
            # ),
            A.HueSaturationValue(p=p_noise),
            A.AdditiveNoise(
                # noise_type='gaussian', 
                # noise_params={'mean_range': (0, 0), 'std_range': (0, 0.04)}, 
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

# class CellMaskDataset(torch.utils.data.Dataset):
#     def __init__(self, 
#                  dataset_coco: CocoDetection, 
#                  processor, 
#                  classnames_to_class_ids_map: Dict[str, int] = None,
#                  instance_segmentation: bool = True,
#                  transforms: A.core.composition.Compose = None):
#         self.dataset_coco = dataset_coco
#         self.class_id_remap: Dict[int, int] = None
#         if classnames_to_class_ids_map is not None:
#             self.class_id_remap = {}
#             for coco_class_id_to_classname_map in dataset_coco.coco.cats.values():
#                 coco_class_id: int = coco_class_id_to_classname_map['id']
#                 coco_classname: str = coco_class_id_to_classname_map['name']
#                 if coco_classname in classnames_to_class_ids_map:
#                     # add the class ID to be remapped to the new class ID
#                     # note that any object with a class ID different that what is listed in the values of classnames_to_class_ids_map
#                     # or any class name in COCO not listed in classnames_to_class_ids_map keys will be ignored
#                     self.class_id_remap[coco_class_id] = classnames_to_class_ids_map[coco_classname]
#                 else:
#                     print(f"[WARN] Class ID: {coco_class_id} used in COCO datast with name: {coco_classname} is not included in the passed "
#                           f"classnames_to_class_ids_map and objects of this class will be excluded from annotations!")
#             print(f"[INFO] The class IDs in COCO will be remapped according to this mapping: {self.class_id_remap}")  
#         self.processor = processor
#         self.classnames_to_class_ids_map = classnames_to_class_ids_map
#         self.instance_segmentation = instance_segmentation
#         self.transforms = transforms

   
#     def __len__(self):
#         return len(self.dataset_coco)

#     def __getitem__(self, idx):
#         image, annotations = self.dataset_coco[idx]
        
#         # Convert image to RGB numpy array (image is in PIL format returned by torchvision.datasets.CocoDetection)
#         image_array: np.ndarray = np.array(image.convert("RGB"))
        
        
#         # in torchvision.datasets.CocoDetection class, annotations is a list of dictionaries for each annotated object
#         if len(annotations) > 0:
#             image_id: int = annotations[0]['image_id']
#         else:
#             image_id: int = idx + 1

        
#         annotations_to_keep: List[dict] = []
#         if self.class_id_remap is not None:
#             labels: List[int] = []
#             for record in annotations:
#                 if record['category_id'] in self.class_id_remap:
#                     annotations_to_keep.append(record)
#                     labels.append(self.class_id_remap[record['category_id']])
#         else:
#             annotations_to_keep = annotations
#             labels: List[int] = [record['category_id'] for record in annotations_to_keep]
        
#         boxes: np.ndarray = np.array([record['bbox'] for record in annotations_to_keep])
        
        
#         # NOTE: the following implementation for image augmentation, and then adjusting them for a specific model with the passed processor
#         # is not efficient because we decode and then encode the masks from and to COCO RLE format multiple times
#         # first we decode to apply the transforms (albumentation), then we encode again before passing to the processor function, in which 
#         # we decode again (so one decoding and then encoding is redundant)
#         # but this allows us to use the same CellMaskDataset and albumentations transforms for all different models, and only have the
#         # processor as a model specific function
#         if self.instance_segmentation:
#             masks: List[np.ndarray] = [coco_mask_util.decode(record['segmentation']) for record in annotations_to_keep]
        
#         # apply augmentations, the second condition should not happen (all preprocessed images should at least have one object)
#         if self.transforms and len(boxes) > 0:
#             if self.instance_segmentation:
#                 # TODO: check to make sure the below mask transform would work
#                 transformed = self.transforms(image=image_array, bboxes=boxes, masks=masks, category=labels)
#                 img = transformed["image"]
#                 boxes = transformed["bboxes"]
#                 masks = transformed["masks"]
#                 labels = transformed["category"]
#             else:
#                 transformed = self.transforms(image=image_array, bboxes=boxes, category=labels)
#                 img = transformed["image"]
#                 boxes = transformed["bboxes"]
#                 labels = transformed["category"]

#         # reformat annotations
#         annotations: List[dict] = []
#         for i, bbox in enumerate(boxes):
#             if int(bbox[2]) * int(bbox[3]) == 0:
#                 # skip zero area invalid annotations post augmentation
#                 # NOTE: this may lead to having no annotated object in the image
#                 continue
#             record = {
#                 "image_id": image_id,
#                 "category_id": int(labels[i]),
#                 "bbox": np.array([int(v) for v in bbox]),
#                 "iscrowd": 0,
#                 "area": bbox[2] * bbox[3],
#             }
#             if self.instance_segmentation:
#                 record["segmentation"] = coco_mask_util.encode(np.asarray(masks[i], order="F"))
#                 record["segmentation"]['counts'] = record["segmentation"]['counts'].decode('utf8')
#             annotations.append(record)

#         formatted_annotation: dict = {"image_id": image_id, "annotations": annotations,}
#         if self.processor is not None:
#             results = self.processor(images=img, annotations=formatted_annotation, return_tensors="pt")
#             if isinstance(results, dict):
#                 # image processor for hugging face expands batch dimension, lets squeeze it
#                 results = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k, v in results.items()}
#             return results
         
#         return img, formatted_annotation


def create_dataset_classes(
        dataset_path: str,
        class_names_to_class_ids_map: Dict[str, int],
        percentage_to_expand_bbox_boundaries: float = 0.0,
        max_images_to_consider_for_each_annotation: int = 1,
        only_use_best_focus_image: bool = False,
        max_larger_side: int = MAX_IMAGE_SIDE,
        max_smaller_side: int = MAX_IMAGE_SIDE,
        fl_channel_id_to_color_map_for_overlay: Dict[str, str] = None,
):
    """
    A function to create a pytorch style dataset class for Cellanome datasets.

    Args:
        dataset_path (str): Base path to the dataset
        class_names_to_class_ids_map (dictionary): A dictionary with keys as string class names, e.g., 'cell', 'bead', and values
            as integer class IDs, e.g., 1, 2. This mapping is needed as the annotation files contain the class names that wiill be
            mapped to class IDs by the dataset class. Any class name not included in this dictionary will be ignored.
        percentage_to_expand_bbox_boundaries (float): A float factor to expand the bounding box around the object's mask, as a percentage
            of the box's height/width (equal expansion on both sides). For example 0.1 expands the box's width by 5% of the width, and the
            height by 5% of the height on both sides. During training the Mask R-CNN model, expaniding the bounding boxes around the
            objects' masks slightly was helping with more accurate detections. Otherwise, set to 0.0.
        max_images_to_consider_for_each_annotation (int): When a z-stack is available for each FoV, this settins specified the
            maximum number of z-stack images to use with one annotation (one FoV). This does not have any effect if multuple images
            for a FoV is not available.
        only_use_best_focus_image (bool): When a z-stack is available for each FoV, if max_images_to_consider_for_each_annotation is set
            to 1 and this flag is set to True, only the best focus image (dz0) will be used. Otherwise, no effect.
        max_larger_side (int): The upper bound on the larger side of the image. If the larger side is greater than this value, the
            image is resized (down) while keeping the aspect ratio to have the larger side of the image <= this value. This setting
            (and the following one on the smaller side of the image) can be used to resize the images and the annotations.
        max_smaller_side (int): The upper bound on the smaller side of the image. If the smaller side is greater than this value, the
            image is resized (down) while keeping the aspect ratio to have the smaller side of the image <= this value.
        fl_channel_id_to_color_map_for_overlay (dictionary): A dictionary with keys as the Fluorescent channel identifiers, e.g.,
            'Red', 'Blue' (they should be the same identifiers as used in the image names and containing folders) and values as the
            color codes (ONLY 'red', 'green', 'blue' are allowed). If the dataset includes Fluorescent channels as well as brightfield ones,
            this argument can be used to get an overlay of the brightfield image with the specified Fluorescent (the list of identifiers)
            for the FoV. For returning overlaid images, max_images_to_consider_for_each_annotation should be set to 1 if z-stack brightfield
            images are available. only_use_best_focus_image can be set to True or False.

    Returns:
        Two pytorch style dataset classes for train and test datasets.
        Example:

        # Note that early annotations uses different names, e.g., 'cell' and 'Cell' for the same class annotation. This can be addressed by
        # mapping them all to the same ID in class_names_to_class_ids_map

        class_names_2_ids_map: Dict[str, int]  = {
            'cell': 0, 'Cell': 0, 'dying/dead cells': 0, 'dead-cell': 0,
            'Bead': 1, 'bead': 1,
            'cell-adhered': 2, 'cytoplasm': 2,
            'soma': 3,
        }

        train_dataset, test_dataset = create_dataset_classes(
            dataset_path='base/path/to/dataset',
            class_names_to_class_ids_map=class_names_2_ids_map,
            max_images_to_consider_for_each_annotation: int = 2,
        )
    """

    images_path: str = os.path.join(dataset_path, 'images')
    annotations_path: str = os.path.join(dataset_path, 'annotations')

    annotations_images_map: pd.DataFrame = pd.read_csv(os.path.join(dataset_path, 'annotation_images_mapping.csv'))

    test_files: List[str] = []
    train_files: List[str] = []

    with open(os.path.join(dataset_path, 'test.txt')) as file:
        filenames: List[str] = file.readlines()
        filenames = [f.replace('\n', '') for f in filenames if len(f) > 0]
        test_files += filenames

    with open(os.path.join(dataset_path, 'train.txt')) as file:
        filenames: List[str] = file.readlines()
        filenames = [f.replace('\n', '') for f in filenames if len(f) > 0]
        train_files += filenames

    columns: List[str] = list(annotations_images_map.columns)

    overlay_color_map: Dict[str, str] = None
    if fl_channel_id_to_color_map_for_overlay is not None:
        overlay_color_map = {}
        for k, v in fl_channel_id_to_color_map_for_overlay.items():
            if k not in columns:
                print(
                    f"[WARN] FL channel {k} is not included in 'annotation_images_mapping.csv' file and will not be used for creating an overlaid image")
                continue
            if v.lower() not in ['red', 'green', 'blue']:
                print(
                    f"[WARN] FL channel {k} is not mapped to either of 'red', 'green' or 'blue' colors and will not be used for creating an overlaid image")
                continue
            overlay_color_map[k] = v.lower()

    image_columns: List[str] = [column_name for column_name in columns if 'white_dz' in column_name.lower()]

    num_images_to_consider_for_each_annotation: int = max_images_to_consider_for_each_annotation

    if len(image_columns) > 1:
        if only_use_best_focus_image:
            image_columns = [column_name for column_name in columns if 'white_dz0' in column_name.lower()]
            num_images_to_consider_for_each_annotation = min(1, num_images_to_consider_for_each_annotation)
            # print('[INFO]: This is a z-stack dataset, but the best focus distance image is selected because of the passed configuration.')
        else:
            pass
            # print(f"[INFO]: This is a z-stack dataset. {max_images_to_consider_for_each_annotation} randomly chosen images will be considered.")

    if len(image_columns) == 0:
        # this is not a focus sweep dataset, get the BF image
        for column_name in columns:
            if 'bf' in column_name.lower() or 'white' in column_name.lower() or 'images' in column_name.lower():
                image_columns = [column_name]
                num_images_to_consider_for_each_annotation = 1
                # print(f"[INFO]: This is NOT a z-stack dataset. {column_name} channel will be considered as the brightfield channel.")
                break

    if len(image_columns) == 0:
        print(
            '[ERROR]: No brightfield image folder could be extracted from annotation_images_mapping.csv file for the dataset.')

    train_map_dict: Dict[str, List[str]] = {}
    test_map_dict: Dict[str, List[str]] = {}
    train_overlay_images: Dict[str, Dict[str, str]] = None
    test_overlay_images: Dict[str, Dict[str, str]] = None
    if overlay_color_map is not None:
        train_overlay_images = {}
        test_overlay_images = {}

    for _, row in annotations_images_map.iterrows():
        annotations_filename = row['annotation_json']
        name = '.'.join(annotations_filename.strip().split('.')[:-1])
        if name in test_files:
            test_map_dict[annotations_filename] = list(row[image_columns])
            if overlay_color_map is not None and len(overlay_color_map) > 0:
                test_overlay_images[annotations_filename] = {}
                for k, v in overlay_color_map.items():
                    test_overlay_images[annotations_filename][v] = row[k]
        else:
            train_map_dict[annotations_filename] = list(row[image_columns])
            if overlay_color_map is not None and len(overlay_color_map) > 0:
                train_overlay_images[annotations_filename] = {}
                for k, v in overlay_color_map.items():
                    train_overlay_images[annotations_filename][v] = row[k]

    # dataset classes
    train_dataset = CellMaskDataset(images_path=images_path, annotations_path=annotations_path,
                                    annotations=train_map_dict,
                                    overlay_fl_images_per_annotation=train_overlay_images,
                                    max_images_to_consider_for_each_annotation=num_images_to_consider_for_each_annotation,
                                    labels_of_interest=list(class_names_to_class_ids_map.keys()),
                                    percentage_to_expand_bbox_boundaries=percentage_to_expand_bbox_boundaries,
                                    color_depth=8,
                                    min_object_diameter=6.0,
                                    scale_factor_dict={(2000, 1600): 1.11111111},
                                    # for ix-81 microscope images in old datasets
                                    max_larger_side=max_larger_side, max_smaller_side=max_smaller_side,
                                    normalize=False,
                                    # can set ignore_extremes_for_fl_normalization as well if normalize is set to True
                                    class_names_to_ids_map=class_names_to_class_ids_map)

    test_dataset = CellMaskDataset(images_path=images_path, annotations_path=annotations_path,
                                   annotations=test_map_dict,
                                   overlay_fl_images_per_annotation=test_overlay_images,
                                   max_images_to_consider_for_each_annotation=num_images_to_consider_for_each_annotation,
                                   labels_of_interest=list(class_names_to_class_ids_map.keys()),
                                   percentage_to_expand_bbox_boundaries=percentage_to_expand_bbox_boundaries,
                                   color_depth=8,
                                   min_object_diameter=6.0,
                                   scale_factor_dict={(2000, 1600): 1.11111111},
                                   # for ix-81 microscope images in old datasets
                                   max_larger_side=max_larger_side, max_smaller_side=max_smaller_side,
                                   normalize=False,
                                   # can set ignore_extremes_for_fl_normalization as well if normalize is set to True
                                   class_names_to_ids_map=class_names_to_class_ids_map)

    return train_dataset, test_dataset

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