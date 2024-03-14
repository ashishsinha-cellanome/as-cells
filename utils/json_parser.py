# load required libraries
from cv_utils import iou_batch, iou_mask_pair
import os
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import json
import requests
import random
from pycocotools import mask as coco_mask_util
from typing import List, Final, Dict, Union, Tuple

# a mapping between the class names and class IDs
# this dictionary should have identical string keys as the class names used in annotations
# the values (class IDs) should be consistent with results returned by the trained model
DEFAULT_CLASS_NAMES_TO_IDS_MAP: Final[Dict[str, int]] = {'Cell': 1, 'dying/dead cells': 2, 'Bead': 3, 'Cluster': 4} 

# optical characteristics of the images (used to map the minimum diameter of objects in um to pixels)
OPTICAL_CHARACTERISTICS: Final[Dict[Tuple, Dict[str, float]]] = {(2000, 1600): {'mag': 10.0, 'pixel_size': 4.54}, (4512, 4512): {'mag': 9.0, 'pixel_size': 2.74}}

# an upper limit on the number of annotations to cache
MAX_ANNOTATION_CACHE_LENGTH: Final[int] = 5


def parse_json_annotations(
    json_filename: str,
    labels_of_interest: Union[List[str], None] = None,
    download_image: bool = False,
    percentage_to_expand_bbox_boundaries: float = 0.0,
    min_object_diameter: float = 0.0,
    optical_characteristics: Dict[
        Tuple[int, int], Dict[str, float]
    ] = OPTICAL_CHARACTERISTICS,
    return_masks_in_coco_rle_format: bool = True,
) -> dict:
    """
    A function to parse the JSON annotation files and extract both the bounding boxes and
    the polygon annotations.

    Args:
        json_filename (str): The JSON filename (including the path) for extracting the
            annotations.
        labels_of_interest (list of strings or None): List of labels to return. Any annotated object with
            a label not included in this list will be ignored. If None passed, all the objects will be returned.
        download_image (bool): If set to True, the image will also be downloaded from the URL
            provided in the annotation file and returned in the 'image' field of the returned dictionary.
        percentage_to_expand_bbox_boundaries (float): Percentage to expand the bounding boxes around the mask to improve
            training the box regressor of Mask R-CNN
        min_object_diameter (float): The minimum diameter in um for objects to keep.
        optical_characteristics (dictionary): A dictionary with keys as image resolution tuples and values as
            dictionaries with two keys: "pixel_size" and "mag", the optical characteristics of the device; these
            are used to convert the min_object_diameter in micro meter to pixels
        return_masks_in_coco_rle_format (bool): If set, the masks are reported in COCO's RLE format to save
            memory.
    Returns:
        a dictionary with keys and values as
            'name'(str): image name,
            'image'(numpy.ndarray): H x W (Gray scale) or H x W x 3 of the downloaded image array in RGB format. This
                field is populated only if the input flag download_image is set to True, otherwise, it will be None.
            'annotations' (pandas DataFrame): A DataFrame with columns 'xtl', 'ytl', 'xbr', 'ybr', 'label'.
                The i-th row is the bounding box coordinates and the label for the i-th annotated object,
            'masks'(list of numpy.ndarrays or COCO RLE format dictionaries): If return_masks_in_coco_rle_format is
                set to False, masks[i] is the H x W array mask for the i-th object with value 1 for the object. If
                return_masks_in_coco_rle_format is set to True, the numpy mask above is converted to COCO RLE format
                for masks (a dictionary with keys as 'size, 'count' and values as 2-element list and bytes
                (encoded mask)).
    """

    #  drop the path from the json_filename to get the filename
    filename_wo_path: str = os.path.basename(json_filename)
    
    try:
        with open(json_filename, 'r') as annotations:
            json_annotations = json.load(annotations)
    except FileNotFoundError:
        print(f"[ERROR]: Annotation file {json_filename} was not found!")
        return {'name': 'UNKNOWN', 'image': None, 
                'annotations': pd.DataFrame(columns=['xtl', 'ytl', 'xbr', 'ybr', 'label']), 
                'masks': np.array((0, 0, 0), np.uint8)}
    except Exception as ex:
        print(f"[EXCEPTION]: Unable to open annotation file {json_filename} failed on {repr(ex)}")
        return {'name': 'UNKNOWN', 'image': None, 
                'annotations': pd.DataFrame(columns=['xtl', 'ytl', 'xbr', 'ybr', 'label']), 
                'masks': np.array((0, 0, 0), np.uint8)}
    
    image_info: dict = json_annotations.get('image')
    if image_info is None:
        # Darwin 2.0 (JSON) annotations format
        return parse_json_annotations_2p0(json_filename,
                                          labels_of_interest,
                                          download_image,
                                          percentage_to_expand_bbox_boundaries,
                                          min_object_diameter,
                                          optical_characteristics,
                                          return_masks_in_coco_rle_format)
                               
    image_width: int = image_info.get('width')
    image_height: int = image_info.get('height')
    image_name: str = image_info.get('original_filename')    
    if image_name is None:
        image_name = image_info.get('filename')    
    
    img = None
    if download_image:
        img_url = image_info.get('url')
        try:
            img = Image.open(requests.get(img_url, stream=True).raw)
            # convert to an array in RGB format
            img = np.array(img)
            if image_height is None:
                image_height = img.shape[0]
            if image_width is None:
                image_width = img.shape[1]
            
            if image_height != img.shape[0] or image_width != img.shape[1]:
                # image shapes reported in the 
                print(f"[ERROR]: Image dimensions: {img.shape[:2]} are not consistent with json " + 
                      f"file: {(image_height, image_width)}! Skipping the file")
                return {'name': 'UNKNOWN', 'image': None, 
                        'annotations': pd.DataFrame(columns=['xtl', 'ytl', 'xbr', 'ybr', 'label']), 
                        'masks': np.array((0, 0, 0), np.uint8)}
        except Exception as ex:
            print(f"[EXCEPTION]: Downloading image {image_name} failed on {repr(ex)}")

    # filter small objects from annotations
    if (image_width, image_height) in optical_characteristics:
        mag: float = optical_characteristics[(image_width, image_height)]['mag']
        pixel_size: float = optical_characteristics[(image_width, image_height)]['pixel_size']
    else:
        print(f"[WARNING]: The image resolution for {json_filename} is not supported! Not possible to filter small objects")
        mag: float = 0
        pixel_size: float = 1.0
    
    min_diameter_in_pixels: int = int(min_object_diameter * mag / pixel_size)
    
    # number of objects
    num_objects: int = len(json_annotations.get('annotations'))
        
    # print(f"[INFO]: {filename_wo_path} includes {num_objects} annotated objects")
    
    # create a pandas DataFrame for ease of processing
    # the i-th row includes the bounding box coordinates (top-left and bottom-right)
    # and the label for the i-th object
    annotations_df = pd.DataFrame(columns=['xtl', 'ytl', 'xbr', 'ybr', 'label'])
    # the masks array, masks[i] is a image_height x image_width np.uint8 array for the i-th
    # object with object mask values set to 1
    masks: List[np.ndarray] = []
    
    obj_id = 0
    complex_polygon_found = False
    for idx, annots in enumerate(json_annotations.get('annotations')):
        # the label for the annotated object
        label: str = annots.get('name')
        if label is None:
            print(f"[ERROR]: No label was provided for {idx}th object in {filename_wo_path}")
            continue
        # skip labels not included in the labels_of_interest list
        if labels_of_interest is not None and label not in labels_of_interest:
            continue
        # the bounding box for the object
        # we are not using these reported bounding boxes for now because of the issue
        # with complex_polygon
        bbox: dict = annots.get('bounding_box')
        if bbox is None or 'x' not in bbox or 'y' not in bbox or 'w' not in bbox or 'h' not in bbox:
            print(f"[ERROR]: No bounding box was provided for {idx}th object in {filename_wo_path}")
            continue   
            
        polygon: dict = annots.get('polygon')
        if polygon is None:
            # the polygon is reported as a union of polygons
            # it is not clear what his is happning, but for some cases, 
            # these are simply multiple objects (e.g., cells) reported as one object
            # with one bounding box (covering all polygons), one label and multiple polygons
            complex_polygon_found = True
            polygon = annots.get('complex_polygon')
            # in this case, polygon.get('path') returns a list of polygons
            # each polygon is a list of dictionaries with keys as 'x' and 'y'
            # the coordinates of the polygon vertices
            polygon_points_list: list = polygon.get('path')
        else:
            # for a simple polygon, we also create a list of polygons 
            # to process them identically below
            polygon_points_list = [polygon.get('path')]
            
        if polygon_points_list is None or polygon_points_list[0] is None:
            print(f"[ERROR]: No polygon was provided for {idx}th object in {filename_wo_path}")
            continue
        
        for poly in polygon_points_list:
            # pandas is just used for simplicity
            polygon_points: np.ndarray = pd.DataFrame(poly).values.astype(np.int32)
            
            # the bounding box around the polygon points
            # we draw the mask (polygon) within this confined box to save processing
            xmin: int = max(0, np.min(polygon_points[:, 0]))
            xmax: int = min(image_width, np.max(polygon_points[:, 0]))
            ymin: int = max(0, np.min(polygon_points[:, 1]))
            ymax: int = min(image_height, np.max(polygon_points[:, 1]))
            
            # filter the small objects here before modifying the bounding boxes
            if xmin >= xmax or ymin >= ymax or max(xmax - xmin, ymax - ymin) < min_diameter_in_pixels:
                continue
            
            # expand the box boundaries by a few pixels as the masks may not be covering the boundaries
            delta_x: int = int(percentage_to_expand_bbox_boundaries * (xmax - xmin) / 2)
            delta_y: int = int(percentage_to_expand_bbox_boundaries * (ymax - ymin) / 2)
            
            delta_x = max(1, delta_x)
            delta_y = max(1, delta_y)
            
            xmin = max(0, xmin - delta_x)
            ymin = max(0, ymin - delta_y)
            xmax = min(image_width, xmax + delta_x)
            ymax = min(image_height, ymax + delta_y)
            
            # draw the polygon mask within the bounding box
            # CV2 contours format
            polygon_points = np.expand_dims(polygon_points, axis=1)
            mask: np.ndarray = np.zeros((ymax - ymin, xmax - xmin), np.uint8)
            # create the mask for the object (the mask value is set to 1 for the object)
            cv2.drawContours(mask, [polygon_points - np.array([xmin, ymin])], 0, 1, -1)
            
            if return_masks_in_coco_rle_format:
                encoded_mask: dict = coco_mask_util.encode(np.asarray(mask, order="F"))    
                masks.append(encoded_mask)
            else:
                masks.append(mask)
            annotations_df = pd.concat([annotations_df, pd.DataFrame(data = 
                                                                    [{'xtl': xmin, 
                                                                      'ytl': ymin, 
                                                                      'xbr': xmax, 
                                                                      'ybr': ymax, 
                                                                      'label': label}], index=[len(annotations_df)])])

            obj_id += 1
    
    if complex_polygon_found:
        pass
        # print(f"[WARNING]: Check the annotations for {filename_wo_path} as complex polygon found")

    # remove duplicate objects (bounding boxes)
    # make sure the indexes are 0 1, ...
    annotations_df.reset_index(inplace=True, drop=True)
    annotations_df = annotations_df.drop_duplicates()
    masks = [masks[i] for i in annotations_df.index]
    annotations_df.reset_index(inplace=True, drop=True)
         
    return {'name': image_name, 'image': img, 'annotations': annotations_df, 'masks': masks}


def parse_json_annotations_2p0(
        json_filename: str,
        labels_of_interest: Union[List[str], None] = None,
        download_image: bool = False,
        percentage_to_expand_bbox_boundaries: float = 0.0,
        min_object_diameter: float = 0.0,
        optical_characteristics: Dict[
            Tuple[int, int], Dict[str, float]
        ] = OPTICAL_CHARACTERISTICS,
        return_masks_in_coco_rle_format: bool = True,
):
    """
    A function to parse the JSON annotation files and extract both the bounding boxes and
    the polygon annotations.

    Args:
        json_filename (str): The JSON filename (including the path) for extracting the
            annotations.
        labels_of_interest (list of strings or None): List of labels to return. Any annotated object with
            a label not included in this list will be ignored. If None passed, all the objects will be returned.
        download_image (bool): If set to True, the image will also be downloaded from the URL
            provided in the annotation file and returned in the 'image' field of the returned dictionary.
        percentage_to_expand_bbox_boundaries (float): Percentage to expand the bounding boxes around the mask to improve
            training the box regressor of Mask R-CNN
        min_object_diameter (float): The minimum diameter in um for objects to keep.
        optical_characteristics (dictionary): A dictionary with keys as image resolution tuples and values as
            dictionaries with two keys: "pixel_size" and "mag", the optical characteristics of the device; these
            are used to convert the min_object_diameter in micro meter to pixels
        return_masks_in_coco_rle_format (bool): If set, the masks are reported in COCO's RLE format to save
            memory.
    Returns:
        a dictionary with keys and values as
            'name'(str): image name,
            'image'(numpy.ndarray): H x W (Gray scale) or H x W x 3 of the downloaded image array in RGB format. This
                field is populated only if the input flag download_image is set to True, otherwise, it will be None.
            'annotations' (pandas DataFrame): A DataFrame with columns 'xtl', 'ytl', 'xbr', 'ybr', 'label'.
                The i-th row is the bounding box coordinates and the label for the i-th annotated object,
            'masks'(list of numpy.ndarrays or COCO RLE format dictionaries): If return_masks_in_coco_rle_format is
                set to False, masks[i] is the H x W array mask for the i-th object with value 1 for the object. If
                return_masks_in_coco_rle_format is set to True, the numpy mask above is converted to COCO RLE format
                for masks (a dictionary with keys as 'size, 'count' and values as 2-element list and bytes
                (encoded mask)).
    """
    
    #  drop the path from the json_filename to get the filename
    filename_wo_path: str = os.path.basename(json_filename)
    
    try:
        with open(json_filename, 'r') as annotations:
            json_annotations = json.load(annotations)
    except FileNotFoundError:
        print(f"[ERROR]: Annotation file {json_filename} was not found!")
        return {'name': 'UNKNOWN', 'image': None, 
                'annotations': pd.DataFrame(columns=['xtl', 'ytl', 'xbr', 'ybr', 'label']), 
                'masks': np.array((0, 0, 0), np.uint8)}
    except Exception as ex:
        print(f"[EXCEPTION]: Unable to open annotation file {json_filename} failed on {repr(ex)}")
        return {'name': 'UNKNOWN', 'image': None, 
                'annotations': pd.DataFrame(columns=['xtl', 'ytl', 'xbr', 'ybr', 'label']), 
                'masks': np.array((0, 0, 0), np.uint8)}
    
    image_info: dict = json_annotations.get('item')
    image_name: str = image_info.get('name') 
    details = image_info.get('slots')[0]
    image_width: int = details.get('width')
    image_height: int = details.get('height')    
    
    img = None
    if download_image:
        img_url = details.get('source_files')[0].get('url')
        try:
            img = Image.open(requests.get(img_url, stream = True).raw)
            # convert to an array in RGB format
            img = np.array(img)
            if image_height is None:
                image_height = img.shape[0]
            if image_width is None:
                image_width = img.shape[1]
            
            if image_height != img.shape[0] or image_width != img.shape[1]:
                # image shapes reported in the 
                print(f"[ERROR]: Image dimensions: {img.shape[:2]} are not consistent with json " + 
                      f"file: {(image_height, image_width)}! Skipping the file")
                return {'name': 'UNKNOWN', 'image': None, 
                        'annotations': pd.DataFrame(columns=['xtl', 'ytl', 'xbr', 'ybr', 'label']), 
                        'masks': np.array((0, 0, 0), np.uint8)}
        except Exception as ex:
            print(f"[EXCEPTION]: Downloading image {image_name} failed on {repr(ex)}")

    # filter small objects from annotations
    if (image_width, image_height) in optical_characteristics:
        mag: float = optical_characteristics[(image_width, image_height)]['mag']
        pixel_size: float = optical_characteristics[(image_width, image_height)]['pixel_size']
    else:
        print(f"[WARNING]: The image resolution for {json_filename} is not supported! Not possible to filter small objects")
        mag: float = 0
        pixel_size: float = 1.0
    
    min_diameter_in_pixels: int = int(min_object_diameter * mag / pixel_size)
    
    # number of objects
    num_objects: int = len(json_annotations.get('annotations'))
        
    # print(f"[INFO]: {filename_wo_path} includes {num_objects} annotated objects")
    
    # create a pandas DataFrame for ease of processing
    # the i-th row includes the bounding box coordinates (top-left and bottom-right)
    # and the label for the i-th object
    annotations_df = pd.DataFrame(columns=['xtl', 'ytl', 'xbr', 'ybr', 'label'])
    # the masks array, masks[i] is a image_height x image_width np.uint8 array for the i-th
    # object with object mask values set to 1
    masks: List[np.ndarray] = []
    
    obj_id = 0
    
    for idx, annots in enumerate(json_annotations.get('annotations')):
        # the label for the annotated object
        label: str = annots.get('name')
        if label is None:
            print(f"[ERROR]: No label was provided for {idx}th object in {filename_wo_path}")
            continue
        # skip labels not included in the labels_of_interest list
        if labels_of_interest is not None and label not in labels_of_interest:
            continue
        # the bounding box for the object
        # we are not using these reported bounding boxes for now because of the issue
        # with complex_polygon
        bbox: dict = annots.get('bounding_box')
        if bbox is None or 'x' not in bbox or 'y' not in bbox or 'w' not in bbox or 'h' not in bbox:
            print(f"[ERROR]: No bounding box was provided for {idx}th object in {filename_wo_path}")
            continue   
            
        polygon: dict = annots.get('polygon')
        if polygon is None:
            print(f"[ERROR]: No polygon was provided for {idx}th object in {filename_wo_path}")
            continue
        
        # for a simple polygon, we also create a list of polygons 
        # to process them identically below
        polygon_points_list = polygon.get('paths')
            
        if polygon_points_list is None or polygon_points_list[0] is None:
            print(f"[ERROR]: No polygon was provided for {idx}th object in {filename_wo_path}")
            continue
        
        # in v2.0, an object can be represented with multiple polygons if it has "holes" in it
        # the largest polygon (in terms of area) specifies the object's mask; the masks specified
        # by the rest of polygons should be excluded from this main mask
        max_area: int = 0
        main_index: int = -1
        main_mask_xmin: int = image_width
        main_mask_ymin: int = image_height
        main_mask_xmax: int = 0
        main_mask_ymax: int = 0
        
        for i, poly in enumerate(polygon_points_list):
            # pandas is just used for simplicity
            polygon_points: np.ndarray = pd.DataFrame(poly).values.astype(np.int32)
            
            # the bounding box around the polygon points
            # we draw the mask (polygon) within this confined box to save processing
            xmin: int = max(0, np.min(polygon_points[:, 0]))
            xmax: int = min(image_width, np.max(polygon_points[:, 0]))
            ymin: int = max(0, np.min(polygon_points[:, 1]))
            ymax: int = min(image_height, np.max(polygon_points[:, 1]))
            
            area: int = (xmax - xmin) * (ymax - ymin)
            
            if area > max_area:
                max_area = area
                main_index = i
            
            if xmin < main_mask_xmin:
                main_mask_xmin = xmin
            if ymin < main_mask_ymin:
                main_mask_ymin = ymin
            if xmax > main_mask_xmax:
                main_mask_xmax = xmax
            if ymax > main_mask_ymax:
                main_mask_ymax = ymax
                
        # filter the small objects here before modifying the bounding boxes
        if main_mask_xmin >= main_mask_xmax or main_mask_ymin >= main_mask_ymax or max(main_mask_xmax - main_mask_xmin, main_mask_ymax - main_mask_ymin) < min_diameter_in_pixels:
            continue
             
        # expand the box boundaries by a few pixels as the masks may not be covering the boundaries
        delta_x: int = int(percentage_to_expand_bbox_boundaries * (main_mask_xmax - main_mask_xmin) / 2)
        delta_y: int = int(percentage_to_expand_bbox_boundaries * (main_mask_ymax - main_mask_ymin) / 2)
            
        delta_x = max(1, delta_x)
        delta_y = max(1, delta_y)
            
        xmin = max(0, main_mask_xmin - delta_x)
        ymin = max(0, main_mask_ymin - delta_y)
        xmax = min(image_width, main_mask_xmax + delta_x)
        ymax = min(image_height, main_mask_ymax + delta_y)
        
        # the object's mask
        mask: np.ndarray = np.zeros((ymax - ymin, xmax - xmin), np.uint8)
        # plot the main mask
        polygon_points: np.ndarray = pd.DataFrame(polygon_points_list[main_index]).values.astype(np.int32)
        polygon_points = np.expand_dims(polygon_points, axis=1)
        cv2.drawContours(mask, [polygon_points - np.array([xmin, ymin])], 0, 1, -1)
        
        for i, poly in enumerate(polygon_points_list):
            if i == main_index:
                continue
            # pandas is just used for simplicity
            polygon_points: np.ndarray = pd.DataFrame(poly).values.astype(np.int32)    
       
            # draw the polygon mask within the bounding box
            # CV2 contours format
            polygon_points = np.expand_dims(polygon_points, axis=1)
            
            hole_mask: np.ndarray = np.zeros((ymax - ymin, xmax - xmin), np.uint8)
            # create the mask for the object (the mask value is set to 1 for the object)
            cv2.drawContours(hole_mask, [polygon_points - np.array([xmin, ymin])], 0, 1, -1)
            # make sure the "hole" is a subset of the main mask before subtraction
            hole_mask = cv2.bitwise_and(mask, hole_mask)
            mask = mask - hole_mask
            
        if return_masks_in_coco_rle_format:
            encoded_mask: dict = coco_mask_util.encode(np.asarray(mask, order="F"))    
            masks.append(encoded_mask)
        else:
            masks.append(mask)
        annotations_df = pd.concat([annotations_df, pd.DataFrame(data = 
                                                                [{'xtl': xmin, 
                                                                  'ytl': ymin, 
                                                                  'xbr': xmax, 
                                                                  'ybr': ymax, 
                                                                  'label': label}], index=[len(annotations_df)])])

        obj_id += 1
        
    # remove duplicate objects (bounding boxes)
    # make sure the indexes are 0 1, ...
    annotations_df.reset_index(inplace=True, drop=True)
    annotations_df = annotations_df.drop_duplicates()
    masks = [masks[i] for i in annotations_df.index]
    annotations_df.reset_index(inplace=True, drop=True)
         
    return {'name': image_name, 'image': img, 'annotations': annotations_df, 'masks': masks}


# a dataset classe to "parse" the masks and extract the bounding boxes from annotations
class CellMaskDataset:
    def __init__(self,
                 images_path: str,
                 annotations_path: str,
                 annotations: Union[List[str], Dict[str, str], Dict[str, List[str]]],
                 max_images_to_consider_for_each_annotation: int = -1,
                 labels_of_interest: Union[List[str], None] = None,
                 percentage_to_expand_bbox_boundaries: float = 0.2,
                 color_depth: int = 14,
                 min_object_diameter: float = 0.0,
                 optical_characteristics: Dict[Tuple[int, int], Dict[str, float]] = OPTICAL_CHARACTERISTICS,
                 scale_factor_dict: Dict[Tuple[int, int], float] = {},
                 max_larger_side: int = 2000,
                 max_smaller_side: int = 1600,
                 normalize: bool = False,
                 class_names_to_ids_map: dict = DEFAULT_CLASS_NAMES_TO_IDS_MAP) -> None:
        """_summary_

        Args:
            images_path (str): Path to images.
            annotations_path (str): Path to annotations.
            annotations (List[str], or Dict[str, str], or Dict[str, List[str]]: This field provides the list of
                train/test annotations files and the corresponding image files for the dataset. There are 3 options to
                provide information this information:
                    1) In case the annotation files and the images share the same name (excluding the extensions), a
                    list of annotations filenames without the extension can be passed. For a given 'name' in this list,
                    the code will search under the specified annotations_path for the annotation file 'name.json' (json
                    is the only allowed extension for annotations) and under the images_path for the image with the
                    name and one of the allowed image extensions ('jpg', 'jpeg', and 'png' all in lower case). This is
                    the legacy behaviour of the code.
                    2) In case the annotation files are not sharing the same names with the images, a dictionary with
                    keys as the annotation filenames WITH the extension and values as the image filenames WITH
                    the extension can be passed. The names for both images and annotations in this case can include an
                    additional path that will be considered with respect to the provided parent paths: images_path and
                    annotations_path. For a given key:value pair 'extra_annots_path/annots_name.json':
                    'extra_img_path/img_name.jpg' in the dictionary, the code will search the annotation file
                    'annotations_path/extra_annots_path/annots_name.json' and for the image
                    'images_path/extra_img_path/img_name.jpg'. Note that the file extensions for both annotations and
                    images should be provided.
                    3) There are scenarios in which one annotation file is applicable to multiple images. In this case,
                    a dictionary with keys as the annotation filenames (with the extension and potentially with an extra
                    path) and values as the list of all applicable image filenames (with the extensions and extra paths)
                    can be passed. For a given key:value pair 'extra_annots_path/annots_name.json':
                    ['extra_img_path_1/img_name_1.jpg', 'extra_img_path_2/img_name_2.jpg', ...] in the dictionary, the
                    code will search the annotation file 'annotations_path/extra_annots_path/annots_name.json' and for
                    the images 'images_path/extra_img_path_1/img_name_1.jpg',
                    'images_path/extra_img_path_2/img_name_2.jpg' , ... . Note that the file extensions for both
                    annotations and images should be provided.
            max_images_to_consider_for_each_annotation (int): This input is only applicable in case each annotation file
                is applicable to multiple images (scenario 3 above). Otherwise, it does not have any effect. In case
                it is set to special value -1 (or any value <= 0), all possible combinations of the annotations and the
                corresponding images will be considered and returned in the dataset class. Otherwise, the specified
                number of images are randomly selected to be is used with the annotation and are returned.
            labels_of_interest (Union[List[str], None], optional): _description_. Defaults to None.
            percentage_to_expand_bbox_boundaries (float, optional): _description_. Defaults to 0.2.
            color_depth (int, optional): _description_. Defaults to 14.
            min_object_diameter (float, optional): _description_. Defaults to 0.0.
            optical_characteristics (Dict[Tuple[int, int], Dict[str, float]]): _description_. Defaults to
                OPTICAL_CHARACTERISTICS.
            scale_factor_dict (Dict[Tuple[int, int], float]): _description_. Defaults to {}.
            max_larger_side (int, optional): _description_. Defaults to 2000.
            max_smaller_side (int, optional): _description_. Defaults to 1600.
            normalize (bool, optional): _description_. Defaults to False.
            class_names_to_ids_map (dict, optional): _description_. Defaults to DEFAULT_CLASS_NAMES_TO_IDS_MAP.
        """
        self.images_base_path = images_path
        self.annotations_base_path = annotations_path
        # annotations contain the list of train or test files without extension.
        self.annotations_images_map: List[Tuple[str, str]] = []
        if isinstance(annotations, list):
            for annotation_file in annotations:
                annotation_path: str = os.path.join(self.annotations_base_path, f"{annotation_file}.json")
                image_path: str = self.get_image_path(annotation_file)
                if os.path.exists(annotation_path):
                    # image_path will be None if the image does not exist, leave it as None for now
                    # the assumption here is the image and its mask/label annotation use the same name
                    # if the images are not already downloaded in the images_path, the function
                    # __getitem__ will also download the image to images_path folder from the location
                    # specified in the json annotation file
                    self.annotations_images_map.append((annotation_path, image_path))
        elif isinstance(annotations, dict):
            for annotation_file, value in annotations.items():
                annotation_path: str = os.path.join(self.annotations_base_path, annotation_file)
                if isinstance(value, list):
                    # value is a list of strings specifying the image names (with the extension) corresponding to the
                    # annotation file
                    image_paths: List[str] = [os.path.join(self.images_base_path, image_name) for image_name in value]
                    if 0 < max_images_to_consider_for_each_annotation < len(image_paths):
                        # TODO: do we want to set a fixed seed to the randomization here to be able to
                        #  reproduce the same everytime?
                        # random.seed(7)
                        random.shuffle(image_paths)
                        image_paths = image_paths[:max_images_to_consider_for_each_annotation]
                else:
                    # value is a string specifying the image name (with the extension)
                    image_paths: List[str] = [os.path.join(self.images_base_path, value)]

                for image_path in image_paths:
                    if os.path.exists(annotation_path) and os.path.exists(image_path):
                        self.annotations_images_map.append((annotation_path, image_path))
                    else:
                        print(f"No annotation file or image could be found for {annotation_path}:{image_path}! "
                              f"This should never happen! Check the passed 'annotations' for a correct mapping between "
                              f"the annotation files and images. Skipping ...")

        # the list of images (with paths) per annotation files (the same content as self.annotations_images_map but
        # formatted in a dictionary, it is used for caching and checking if all the images per annotation are covered
        # to clear the cache
        self.list_of_images_per_annotation: Dict[str, List[str]] = {}
        for (annotation_path, image_path) in self.annotations_images_map:
            if annotation_path in self.list_of_images_per_annotation:
                self.list_of_images_per_annotation[annotation_path].append(image_path)
            else:
                self.list_of_images_per_annotation[annotation_path] = [image_path]

        self.labels_of_interest = labels_of_interest
        # in order to reduce the memory required for the masks (for instance segmentation models)
        # the image and the annotations can be downsized by the passed scale_factor_dict
        # this is a dictionary with keys as the image resolutions for which the scaling should be
        # applied and values as the scaling factor
        # (values >= 1 are expected to downsize the image by the factor)
        self.scale_factor_dict = scale_factor_dict
        # the image is further resized (after applying the passed scale_factor above) to have the
        # larger side and the smaller side both smaller than these two maximum set values
        self.max_larger_side = max_larger_side
        self.max_smaller_side = max_smaller_side
        # the scaling factor for normalizing the channels after Tensor
        # conversion to get values in [0, 1]
        # this is 2 ^ color_depth - 1, where color_depth is the number of bits
        # use to represent the intensities for each channel
        self.channel_scale = 2 ** color_depth - 1
        self.normalize = normalize
        self.class_names_to_ids_map = class_names_to_ids_map
        self.percentage_to_expand_bbox_boundaries = percentage_to_expand_bbox_boundaries
        # minimum diameter of objects to keep in micro meter, objects smaller than this
        # minimum diameter are expected from the returned annotations
        # only 2000x1600 and 4512x4512 image resolutions are supported as this
        # diameter should be converted to a number of pixels based on magnification and sensor pixel size
        self.min_object_diameter = min_object_diameter
        # the optical characteristics of the device (used to convert the above min_object_diameter from um to pixels)
        self.optical_characteristics = optical_characteristics
        # a dictionary with keys as the annotations files paths, and values as the parsed dictionary of
        # annotations, this is used to cache the annotations in case one is used with multiple images and
        # reuse them
        self.cached_annotations: Dict[str, dict] = {}
        # dictionary with keys as the annotations files paths, and values as the list of images already
        # called, it is used to remove the parsed annotation from the cache (to save memory) once all the
        # images for a given annotation have been called
        self.list_of_images_covered_per_annotation: Dict[str, List[str]] = \
            {key: [] for key in self.list_of_images_per_annotation}
        # in case some annotations (with more than one corresponding images) are blocked from being cached (because
        # the cache is already full at the time of call), we add them to this list to block them from being cached
        # later (when called for another image) when some room becomes available in the cache
        # if cached, these will be left in the cache because we never complete a full-cycle read of images for them
        # this step is needed to avoid having some left over in the cache from these annotations
        self.blocked_annotations_from_caching: List[str] = []

    def __getitem__(self, idx: int):
        # load images and masks
        annotation_path: str = self.annotations_images_map[idx][0]
        img_path: str = self.annotations_images_map[idx][1]

        if img_path is not None:
            # read the image, do not change the format
            # depending on the set color_depth, the values will be in [0, 2^color_depth - 1]
            # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img = np.array(Image.open(img_path))
            download_image: bool = False
        else:
            # download the image as well, this can only happen when the annotation and the images use the same name
            annots_name: str = os.path.basename(annotation_path)
            image_name_without_ext: str = '.'.join(annots_name.strip().split('.')[:-1])
            print(f"An image with the same name as the annotation file {os.path.basename(annots_name)} was not "
                  f"found! Downloading the image from the URL in the annotations. This is not considered normal!")
            download_image: bool = True

        if annotation_path in self.cached_annotations:
            # make a (deep) copy
            # we are not caching the images (hence None below fpr key 'image')
            annotations: dict = {'name': self.cached_annotations[annotation_path]['name'],
                                 'image': None, # N/A, see parse_json_annotations with download_image = True
                                 'annotations': self.cached_annotations[annotation_path]['annotations'].copy(),
                                 'masks': [m.copy() for m in self.cached_annotations[annotation_path]['masks']]}
        else:
            # parse the annotations file, this is the first time an image/annotation pair is called
            annotations = parse_json_annotations(json_filename=annotation_path,
                                                 labels_of_interest=self.labels_of_interest,
                                                 download_image=download_image,
                                                 percentage_to_expand_bbox_boundaries=self.percentage_to_expand_bbox_boundaries,
                                                 min_object_diameter=self.min_object_diameter,
                                                 optical_characteristics=self.optical_characteristics,
                                                 return_masks_in_coco_rle_format=False)

            # cache a copy, only if the annotation has multiple images and has not been blocked before
            # make sure we are not growing the cache more than the limit (a safety check)
            if (len(self.list_of_images_per_annotation[annotation_path]) > 1 and
                    annotation_path not in self.blocked_annotations_from_caching):
                if len(self.cached_annotations) < MAX_ANNOTATION_CACHE_LENGTH:
                    self.cached_annotations[annotation_path]: dict = {'name': annotations['name'],
                                                                      'image': None,  # N/A
                                                                      'annotations': annotations['annotations'].copy(),
                                                                      'masks': [m.copy() for m in annotations['masks']]}
                else:
                    self.blocked_annotations_from_caching.append(annotation_path)

        if download_image:
            # download_image True means img_path is None, and we had to download the image
            # the first time parse_json_annotations is called with download_image set to True,
            # we save the image to the images_path and update the path in self.annotations_images_map,
            # so the next time the dataset is called with the same idx, we load the image
            # from disk and will not download it
            img: np.ndarray = annotations['image']
            img_path: str = os.path.join(self.images_base_path, image_name_without_ext + '.jpg')
            # save the image using PIL.Image
            Image.fromarray(img).save(img_path)
            # update the map so next time we do not download and read from the file
            self.annotations_images_map[idx] = (annotation_path, img_path)
            # update the list of images as well, this is not really needed as we are not caching one-to-one
            # image/annotations, but just to be precise
            self.list_of_images_per_annotation[annotation_path] = [img_path]

        # check if done with the images, img_path cannot be None if we are here
        if (annotation_path in self.cached_annotations and
                img_path not in self.list_of_images_covered_per_annotation[annotation_path]):
            # the first condition ensure we are caching this annotation, otherwise, no reason to track
            # we may not if the number of already cached annotations is more than the limit, or each annotation
            # is applicable to only one image
            self.list_of_images_covered_per_annotation[annotation_path].append(img_path)
            if self.covered_all_images_per_annotation(annotation_path):
                # all images have been called, delete the annotation from the cache
                del self.cached_annotations[annotation_path]
                # reset the tracking dictionary as well
                self.list_of_images_covered_per_annotation[annotation_path] = []

        # map the class names included in the annotations DataFrame to class IDs if a
        # mapping is passed
        # Note that class_names_to_ids_map should include all the class names used in the annotations
        # as the keys
        if self.class_names_to_ids_map is not None:
            annotations['annotations']['label'] = annotations['annotations']['label'].map(self.class_names_to_ids_map)

        # convert the returned np.unit32 image to float with values between 0, 1
        if self.normalize:
            # if this flag is set, normalize the image such the minimum intensity
            # is mapped to zero, and the maximum is mapped to one
            # convert the image to a numpy array
            img = cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        else:
            # the intensity of the images is color_depth bits, so we need to divide by 2^color_depth - 1
            img = (255 * img.astype(float) / self.channel_scale).astype(np.uint8)

        image_height, image_width = img.shape[:2]
        # scale factor
        if (image_width, image_height) in self.scale_factor_dict:
            scale_factor: float = self.scale_factor_dict[(image_width, image_height)]
        else:
            scale_factor: float = 1.0

        larger_side: int = max(int(image_width / scale_factor), int(image_height / scale_factor))
        smaller_side: int = min(int(image_width / scale_factor), int(image_height / scale_factor))

        if larger_side > self.max_larger_side or smaller_side > self.max_smaller_side:
            scale_factor *= max(float(larger_side) / self.max_larger_side, float(smaller_side) / self.max_smaller_side)

        if scale_factor != 1:
            # for decimating an image, cv2.INTER_AREA is the preferred method (scale_factor is always > 1)
            img = cv2.resize(img, (int(image_width / scale_factor), int(image_height / scale_factor)),
                             interpolation=cv2.INTER_AREA)
            # update all the masks and annotations
            annotations['annotations'][['xtl', 'ytl', 'xbr', 'ybr']] = annotations['annotations'][
                ['xtl', 'ytl', 'xbr', 'ybr']].div(scale_factor).astype(int)

            # make sure no box width/height becomes zero after the resize
            # only keep boxes with positive width and height
            annotations['annotations'] = annotations['annotations'][
                (annotations['annotations']['ybr'] - annotations['annotations']['ytl'] > 0) &
                (annotations['annotations']['xbr'] - annotations['annotations']['xtl'] > 0)]

            # keep the corresponding masks after resizing
            annotations['masks'] = [annotations['masks'][i] for i in annotations['annotations'].index]
            # reset the index
            annotations['annotations'].reset_index(inplace=True, drop=True)

            # now resize the masks (note that they are defined within the bounding boxes)
            for idx in range(len(annotations['masks'])):
                box_xtl, box_ytl, box_xbr, box_ybr = annotations['annotations'].loc[
                    idx, ['xtl', 'ytl', 'xbr', 'ybr']].values
                annotations['masks'][idx] = cv2.resize(annotations['masks'][idx],
                                                       (box_xbr - box_xtl, box_ybr - box_ytl),
                                                       interpolation=cv2.INTER_NEAREST)

        annotations['image'] = img
        # update the name of the image
        annotations['name']: str = os.path.basename(img_path)

        return annotations

    def __len__(self):
        return len(self.annotations_images_map)

    def get_image_path(self, img_name):
        jpg_file_path = os.path.join(self.images_base_path, f"{img_name}.jpg")
        jpeg_file_path = os.path.join(self.images_base_path, f"{img_name}.jpeg")
        png_file_path = os.path.join(self.images_base_path, f"{img_name}.png")
        if os.path.exists(jpg_file_path):
            return jpg_file_path
        elif os.path.exists(jpeg_file_path):
            return jpeg_file_path
        elif os.path.exists(png_file_path):
            return png_file_path

        return None

    def covered_all_images_per_annotation(self, annotation_key: str):
        uncovered_list: List[str] = [img_path for img_path in self.list_of_images_per_annotation[annotation_key] if
                                     img_path not in self.list_of_images_covered_per_annotation[annotation_key]]
        if len(uncovered_list) == 0:
            return True
        return False



# a dataset classe to "parse" the masks and extract the bounding boxes from annotations
# this function combines annotations provided in multiple files (assuming each annotation file set
# covers a different set of classes)
class MaskDatasetFromMultiAnnotations:
    def __init__(self,
                 images_path: str,
                 annotations_paths: List[str],
                 common_names_to_use: List[str],
                 annotation_files_exts: List[str], 
                 labels_of_interest: Union[List[str], None] = None,
                 percentage_to_expand_bbox_boundaries: float = 0.2,
                 color_depth: int = 12,
                 min_object_diameter: float = 0.0,
                 optical_characteristics: Dict[Tuple[int, int], Dict[str, float]] = OPTICAL_CHARACTERISTICS,
                 scale_factor_dict: Dict[Tuple[int, int], float] = {},
                 max_larger_side: int = 4512,
                 max_smaller_side: int = 4512,
                 normalize: bool = False,
                 class_names_to_ids_map: dict = DEFAULT_CLASS_NAMES_TO_IDS_MAP) -> None:
        """_summary_

        Args:
            images_path (str): Path to images.
            annotations_paths (List[str]): Path to multiple annotations folders. Each file should follow 
                the following convention: common_name + '_' + annotation_specific_ext + '.json', where
                common_name should be the same in the image name (common_name + '.jpg') and all different 
                annotations files of that image,  annotation_specific_ext should differentiate the filenames
                (e.g., 'cyto', 'nucl'). The list of annotation_specific_ext should be passed as well.
            common_names_to_use (List[str]): List of train/test annotations files without extension and 
                without the added extention to the filename (see annotation_files_exts) to use.
            annotation_files_exts (List[str]): List of annotation_specific_ext values (in the same oder 
                provided in annotations_paths).
            labels_of_interest (Union[List[str], None], optional): _description_. Defaults to None.
            percentage_to_expand_bbox_boundaries (float, optional): _description_. Defaults to 0.2.
            color_depth (int, optional): _description_. Defaults to 14.
            min_object_diameter (float, optional): _description_. Defaults to 0.0.
            optical_characteristics (Dict[Tuple[int, int], Dict[str, float]]): _description_. Defaults to
                OPTICAL_CHARACTERISTICS.
            scale_factor_dict (Dict[Tuple[int, int], float]): _description_. Defaults to {}.
            max_larger_side (int, optional): _description_. Defaults to 2000.
            max_smaller_side (int, optional): _description_. Defaults to 1600.
            normalize (bool, optional): _description_. Defaults to False.
            class_names_to_ids_map (dict, optional): _description_. Defaults to DEFAULT_CLASS_NAMES_TO_IDS_MAP.
        """
        self.images_path = images_path
        self.annotations_paths = annotations_paths
        if len(annotations_paths) != len(annotation_files_exts):
            print(f"[ERROR]: Args annotations_paths and annotation_files_exts should have the same length!")
            print(f"[ERROR]: Class was not instantiated")
            return 
    
        # annotations contain the list of train or test files without extension.
        # the assumption is the image and its multiple mask/label annotation use the same common
        # name; each annotation file name should follow the following convention: 
        # common_name + '_' + annotation_specific_ext + '.json', where common_name should be the 
        # same in the image name (common_name + '.jpg') and all different annotations files of that 
        # image,  annotation_specific_ext should differentiate the filenames
        # (e.g., 'cyto', 'nucl'). The list of annotation_specific_ext should be passed as well.
        # the images should already be downloaded in the images_path, 
        
        self.annotations: Dict[str, List[str]] = {key: [] for key in annotation_files_exts}
        self.images_names: List[str] = []
        self.common_names: List[str] = []
        for annotation_file in common_names_to_use:
            img_full_name: str = self.get_image_path(annotation_file)
            if img_full_name is None:
                # image could not be found
                continue
            
            found_all_annotation_files = True
            for i, annotation_specific_ext in enumerate(annotation_files_exts):
            
                annotation_path: str = os.path.join(self.annotations_paths[i], 
                                                    f"{annotation_file + '_' + annotation_specific_ext}.json")
                if not os.path.exists(annotation_path):
                    found_all_annotation_files = False
                    break
            
            if found_all_annotation_files:
                # we have found the image and all the annotation files
                self.images_names.append(img_full_name)
                self.common_names.append(annotation_file)
                for i, annotation_specific_ext in enumerate(annotation_files_exts):
            
                    annotation_path: str = os.path.join(self.annotations_paths[i], 
                                                        f"{annotation_file + '_' + annotation_specific_ext}.json")
                    self.annotations[annotation_specific_ext].append(annotation_path)
        
        
        self.labels_of_interest = labels_of_interest
        # in order to reduce the memory required for the masks (for instance segmentation models)
        # the image and the annotations can be downsized by the passed scale_factor_dict
        # this is a dictionary with keys as the image resolutions for which the scaling should be
        # applied and values as the scaling factor
        # (values >= 1 are expected to downsize the image by the factor)
        self.scale_factor_dict = scale_factor_dict
        # the image is further resized (after applying the passed scale_factor above) to have the
        # larger side and the smaller side both smaller than these two maximum set values
        self.max_larger_side = max_larger_side
        self.max_smaller_side = max_smaller_side
        # the scaling factor for normalizing the channels after Tensor
        # conversion to get values in [0, 1]
        # this is 2 ^ color_depth - 1, where color_depth is the number of bits
        # use to represent the intensities for each channel
        self.channel_scale = 2 ** color_depth - 1
        self.normalize = normalize
        self.class_names_to_ids_map = class_names_to_ids_map
        self.percentage_to_expand_bbox_boundaries = percentage_to_expand_bbox_boundaries
        # minimum diameter of objects to keep in micro meter, objects smaller than this
        # minimum diameter are expected from the returned annotations
        # only 2000x1600 and 4512x4512 image resolutions are supported as this
        # diameter should be converted to a number of pixels based on magnification and sensor pixel size
        self.min_object_diameter = min_object_diameter
        # the optical characteristics of the device (used to convert the above min_object_diameter from um to pixels)
        self.optical_characteristics = optical_characteristics

    def __getitem__(self, idx: int):
        # load images and masks
        
        img_path = self.images_names[idx]
        img = np.array(Image.open(img_path))
        
        # parse the annotations file
        annotations_df: pd.DataFrame = pd.DataFrame(columns=['xtl', 'ytl', 'xbr', 'ybr', 'label'])
        masks: List[np.ndarray] = []
        for annotation_specific_ext in self.annotations.keys():
            annotation_path: str = self.annotations[annotation_specific_ext][idx]
            annotations = parse_json_annotations(json_filename=annotation_path,
                                                 labels_of_interest=self.labels_of_interest,
                                                 download_image=False,
                                                 percentage_to_expand_bbox_boundaries=self.percentage_to_expand_bbox_boundaries,
                                                 min_object_diameter=self.min_object_diameter,
                                                 optical_characteristics=self.optical_characteristics,
                                                 return_masks_in_coco_rle_format=False)
            
            annotations_df = pd.concat([annotations_df, annotations['annotations']])
            masks += annotations['masks']
        
        # Note that the name in annotations['name'] is taken from the annotation files, since we are combining 
        # the annotations here, the name is referred to the FL image used in annotations and is not consistent
        # with the image name here, so use the image name directly
        annotations = {'name': self.common_names[idx] + '.jpg', 'annotations': annotations_df.reset_index(drop=True), 'masks': masks}
        # map the class names included in the annotations DataFrame to class IDs if a
        # mapping is passed
        # Note that class_names_to_ids_map should include all the class names used in the annotations
        # as the keys
        if self.class_names_to_ids_map is not None:
            annotations['annotations']['label'] = annotations['annotations']['label'].map(self.class_names_to_ids_map)

        # convert the returned np.unit32 image to float with values between 0, 1
        if self.normalize:
            # if this flag is set, normalize the image such the the minimum intensity
            # is mapped to zero, and the maximum is mapped to one
            # convert the image to a numpy array
            img = cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        else:
            # the intensity of the images is color_deptj bits, so we need to divide by 2^color_depth - 1
            img = (255 * img.astype(float) / self.channel_scale).astype(np.uint8)

        image_height, image_width = img.shape[:2]
        # scale factor
        if (image_width, image_height) in self.scale_factor_dict:
            scale_factor: float = self.scale_factor_dict[(image_width, image_height)]
        else:
            scale_factor: float = 1.0

        larger_side: int = max(int(image_width / scale_factor), int(image_height / scale_factor))
        smaller_side: int = min(int(image_width / scale_factor), int(image_height / scale_factor))

        if larger_side > self.max_larger_side or smaller_side > self.max_smaller_side:
            scale_factor *= max(float(larger_side) / self.max_larger_side, float(smaller_side) / self.max_smaller_side)
        
        
        
        if scale_factor != 1:
            # for decimating an image, cv2.INTER_AREA is the preferred method (scale_factor is always > 1)
            img = cv2.resize(img, (int(image_width / scale_factor), int(image_height / scale_factor)),
                             interpolation=cv2.INTER_AREA)
            # update all the masks and annotations
            annotations['annotations'][['xtl', 'ytl', 'xbr', 'ybr']] = annotations['annotations'][
                ['xtl', 'ytl', 'xbr', 'ybr']].div(scale_factor).astype(int)

            # make sure no box width/height becomes zero after the resize
            # only keep boxes with positive width and height
            annotations['annotations'] = annotations['annotations'][
                (annotations['annotations']['ybr'] - annotations['annotations']['ytl'] > 0) &
                (annotations['annotations']['xbr'] - annotations['annotations']['xtl'] > 0)]

            # keep the corresponding masks after resizing
            annotations['masks'] = [annotations['masks'][i] for i in annotations['annotations'].index]
            # reset the index
            annotations['annotations'].reset_index(inplace=True, drop=True)

            # now resize the masks (note that they are defined within the bounding boxes)
            for idx in range(len(annotations['masks'])):
                box_xtl, box_ytl, box_xbr, box_ybr = annotations['annotations'].loc[
                    idx, ['xtl', 'ytl', 'xbr', 'ybr']].values
                annotations['masks'][idx] = cv2.resize(annotations['masks'][idx],
                                                       (box_xbr - box_xtl, box_ybr - box_ytl),
                                                       interpolation=cv2.INTER_NEAREST)

        annotations['image'] = img

        return annotations

    def __len__(self):
        return len(self.images_names)

    def get_image_path(self, img_name):
        jpg_file_path = os.path.join(self.images_path, f"{img_name}.jpg")
        jpeg_file_path = os.path.join(self.images_path, f"{img_name}.jpeg")
        png_file_path = os.path.join(self.images_path, f"{img_name}.png")
        if os.path.exists(jpg_file_path):
            return jpg_file_path
        elif os.path.exists(jpeg_file_path):
            return jpeg_file_path
        elif os.path.exists(png_file_path):
            return png_file_path

        return None
# helper functions to crop large annotated images into smaller ones 
# for training


def optimize_crop(annotations_df: pd.DataFrame,
                  crop_coords: Union[List[int], Tuple[int]],
                  dx: int, dy: int,
                  w: int, h: int,
                  labels_of_interest: Union[List[int], List[str]],
                  masks: Union[None, List[np.ndarray]] = None):
    """
    This function finds the "best" center for the crop with corners (x1, y1),
    (x2, y2) within the specified range +/-dx and +/-dy such that the bounding boxes
    (specified by annotations_df) either lie completely in or completely out of the crop.

    Args:
        annotations_df (pandas.DataFrame): A DataFrame for the bounding boxes with
            columns 'xtl', 'ytl', 'xbr', 'ybr' and 'label'.
        crop_coords (4-tuple or 4-element list of int): xtl, ytl, xbr, and ybr
            box coordinates for the crop.
        dx (int): The limit on moving the center of the crop (+/-) in x.
        dy (int): The limit on moving the center of the crop (+/-) in y.
        w (int): The original image width.
        h (int): The original image height.
        labels_of_interest (list of integers or strings): List of classnames or class IDs
            of interest depending on the values reported in annotations_df['label']
            (names or IDs). These object classes will be used to optimize the crop
            corners. Any object not from these classes will be ignored.
        masks (optional, list of numpy arrays): The object masks. If provided, the code also
            tries to minimize the length of the crop boundaries cross the objects masks
    """

    # the minimum size of each side of the box to consider it for optimizing the crop
    min_bbox_size_to_consider: int = 4

    (x1, y1, x2, y2) = crop_coords

    # make sure the crop is within the image and corners are interger
    x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))
    dx = int(dx)
    dy = int(dy)
    w = int(w)
    h = int(h)

    # extend the crop by dx and dy in x and y dimensions on both sizes
    xc1: int = max(x1 - dx, 0)
    yc1: int = max(y1 - dy, 0)
    xc2: int = min(x2 + dx, w)
    yc2: int = min(y2 + dy, h)

    if xc2 <= xc1 or yc2 <= yc1:
        # incorrect input dimensions
        return None

    # make a copy of the annotations dataframe
    df = annotations_df.copy()

    # transform the bounding boxes
    df[['xtl', 'xbr']] = df[['xtl', 'xbr']] - xc1
    df[['ytl', 'ybr']] = df[['ytl', 'ybr']] - yc1
    
    crop_width: int = xc2 - xc1 + 1
    crop_height: int = yc2 - yc1 + 1

    # remove the bounding boxes that are totally outside the extended cropped image
    # keep the ones that have some non-zero overlap

    df = df[df.apply(lambda row: True if
    max(0, min(row['xbr'], crop_width) - max(row['xtl'], 0)) * \
    max(0, min(row['ybr'], crop_height) - max(row['ytl'], 0)) > 0 and \
    row['label'] in labels_of_interest
    else False, axis=1)]

    # save the indices of the retained rows (needed for filtering the masks if passes)
    idxs_of_retained_objects: np.ndarray = df.index.values

    # for each pixel x_c in x dimension with 0 <= x_c < crop_width, calculate the cost of having one of the
    # crop boundaries (vertical) at x_c as the number of bounding boxes that the vertical line at
    # x_c will cross
    # we assign a weight in [0, 1] to each box depending on where the line through x_c crosses the box:
    # 0 if the line is within 25% of the length of the box from either edges and 1 otherwise
    # similarly for the y dimension
    cost_in_x: np.ndarray = np.zeros(crop_width)
    cost_in_y: np.ndarray = np.zeros(crop_height)

    edge_score: float = 0.25
    center_score: float = 1.0

    for _, row in df.iterrows():
        xtl: int = int(row['xtl'])
        ytl: int = int(row['ytl'])
        xbr: int = int(row['xbr'])
        ybr: int = int(row['ybr'])

        if row['label'] not in labels_of_interest:
            continue

        # skip skinny boxes
        if (xbr - xtl) < min_bbox_size_to_consider or (ybr - ytl) < min_bbox_size_to_consider:
            continue

        # stepwise weight for the box in x dimension, box_score[0] is the weight at x = xtl and
        # box_score[xbr - xtl - 1] is the weight at x = xbr - 1
        # box_score[0:k] and box_score[xbr - xtl - k + 1:xbr - xtl] are 0.25 and the rest are 1
        box_edge_portion_1: int = max(1, int((xbr - xtl) / 10))
        box_edge_portion_2: int = int((xbr - xtl) / 4) - box_edge_portion_1
        box_edge_portion: int = box_edge_portion_1 + box_edge_portion_2
        box_score: List[float] = [0.0] * box_edge_portion_1 + [edge_score] * box_edge_portion_2 + \
                                 [center_score] * (xbr - xtl - 2 * box_edge_portion) + \
                                 [edge_score] * box_edge_portion_2 + [0.0] * box_edge_portion_1

        """
        # triangular weight for the box in x dimension, box_score[0] is the weight at x = xtl and 
        # box_score[xbr - xtl - 1] is the weight at x = xbr - 1 (both are zero)

        # center of the box
        if (xbr + xtl) % 2 == 0:
            mid_x: int = (xbr + xtl) // 2 - 1 
            box_score = [float(i - xtl) / (mid_x - xtl) for i in range(xtl, mid_x + 1)]
            box_score += [float(xbr - i - 1) / (xbr - mid_x - 2) for i in range(mid_x + 1, xbr)]
        else:
            mid_x: int = (xbr + xtl - 1) // 2 
            box_score = [float(i - xtl) / (mid_x - xtl) for i in range(xtl, mid_x + 1)]
            box_score += [float(xbr - i - 1) / (xbr - mid_x - 1) for i in range(mid_x + 1, xbr)]
        """

        # pixels of the bounding box in x dimension that overlaps with [0, crop_width]
        for i in range(xtl, xbr):
            if 0 <= i < crop_width:
                cost_in_x[i] += box_score[i - xtl]

                # y dimension
        box_edge_portion_1: int = max(1, int((ybr - ytl) / 10))
        box_edge_portion_2: int = int((ybr - ytl) / 4) - box_edge_portion_1
        box_edge_portion: int = box_edge_portion_1 + box_edge_portion_2
        box_score: List[float] = [0.0] * box_edge_portion_1 + [edge_score] * box_edge_portion_2 + \
                                 [center_score] * (ybr - ytl - 2 * box_edge_portion) + \
                                 [edge_score] * box_edge_portion_2 + [0.0] * box_edge_portion_1

        """
        # triangular weight for the box in y dimension, box_score[0] is the weight at y = ytl and 
        # box_score[ybr - ytl - 1] is the weight at y = ybr - 1 (both are zero)

        # center of the box
        if (ybr + ytl) % 2 == 0:
            mid_y: int = (ybr + ytl) // 2 - 1 
            box_score = [float(i - ytl) / (mid_y - ytl) for i in range(ytl, mid_y + 1)]
            box_score += [float(ybr - i - 1) / (ybr - mid_y - 2) for i in range(mid_y + 1, ybr)]
        else:
            mid_y: int = (ybr + ytl - 1) // 2 
            box_score = [float(i - ytl) / (mid_y - ytl) for i in range(ytl, midY + 1)]
            box_score += [float(ybr - i - 1) / (ybr - mid_y - 1) for i in range(mid_y + 1, ybr)]
        """

        # pixels of the bounding box in x dimension that overlaps with [0, crop_height]
        for i in range(ytl, ybr):
            if 0 <= i < crop_height:
                cost_in_y[i] += box_score[i - ytl]

    # the mask for all objects within the crop (after including adjustments)
    objects_mask: np.ndarray = np.zeros((crop_height, crop_width), np.uint8)
    if masks is not None:
        for idx in idxs_of_retained_objects:
            this_mask: np.ndarray = np.zeros((crop_height, crop_width), np.uint8)
            xtl, ytl, xbr, ybr = df.loc[idx, ['xtl', 'ytl', 'xbr', 'ybr']].values
            # confine the mask within the crop
            xtl_a = max(0, xtl)
            ytl_a = max(0, ytl)
            xbr_a = min(crop_width, xbr)
            ybr_a = min(crop_height, ybr)
            this_mask[ytl_a:ybr_a, xtl_a:xbr_a] = masks[idx][ytl_a - ytl: ybr_a - ytl, xtl_a - xtl: xbr_a - xtl]
            objects_mask = cv2.bitwise_or(objects_mask, this_mask)

    """
    # the mask for all objects within the crop (after including adjustments)
    objects_mask: np.ndarray = np.zeros((h, w), np.uint8)
    if masks is not None:
        for idx in idxs_of_retained_objects:
            this_mask: np.ndarray = np.zeros((h, w), np.uint8)
            xtl, ytl, xbr, ybr = df.loc[idx, ['xtl', 'ytl', 'xbr', 'ybr']].values + np.array([xc1, yc1, xc1, yc1])
            this_mask[ytl:ybr, xtl:xbr] = masks[idx]
            objects_mask = cv2.bitwise_or(objects_mask, this_mask)
    """

    # limit the search for the start of the box
    left_delta: int = min(dx, x1)
    right_delta: int = min(dx, w - x2)
    start_cost_x: np.ndarray = np.zeros(left_delta + right_delta + 1)

    for i in range(left_delta + right_delta + 1):
        start_cost_x[i] = cost_in_x[i] + cost_in_x[x2 - x1 + i - 1]

    # limit the search for the start of the box
    top_delta: int = min(dy, y1)
    bottom_delta: int = min(dy, h - y2)
    start_cost_y: np.ndarray = np.zeros(top_delta + bottom_delta + 1)

    for i in range(top_delta + bottom_delta + 1):
        start_cost_y[i] = cost_in_y[i] + cost_in_y[y2 - y1 + i - 1]

    x_start_candidates: np.ndarray = np.array([x1])
    if len(start_cost_x) > 0:
        x_start_candidates = np.where(start_cost_x == np.min(start_cost_x))[0] + x1 - left_delta

    y_start_candidates: np.ndarray = np.array([y1])
    if len(start_cost_y) > 0:
        y_start_candidates = np.where(start_cost_y == np.min(start_cost_y))[0] + y1 - top_delta

    # is mask is available, and there are multiple options for the crops, use the one with minimum length
    # for crossing the object masks
    x1_adjusted: int = x_start_candidates[0]
    if masks is not None and len(x_start_candidates) > 1:
        x1_adjusted = x_start_candidates[
            np.argmin(np.sum(objects_mask[:, x_start_candidates - x1 + left_delta], axis=0) +
                      np.sum(objects_mask[:, x_start_candidates - x1 + left_delta + x2 - x1], axis=0))]

    y1_adjusted: int = y_start_candidates[0]
    if masks is not None and len(y_start_candidates) > 1:
        y1_adjusted = y_start_candidates[
            np.argmin(np.sum(objects_mask[y_start_candidates - y1 + top_delta, :], axis=1) +
                      np.sum(objects_mask[y_start_candidates - y1 + top_delta + y2 - y1, :], axis=1))]

    return x1_adjusted, y1_adjusted, x1_adjusted + x2 - x1, y1_adjusted + y2 - y1


# crop function
def crop_and_block(sample, crop_coords, labels_of_interest=None, 
                   block_label='block', keep_area_threshold=0.9):
    """
    Crop the image in a sample for a given crop coordinates and black out 
    the partial bounding boxes the lies on the crop boundary.

    Args:
        sample (dictionary): Input data sample to be cropped. The dictionary
            should include "name", "image", "annotations" and optionally "masks" keys 
            for passing the image name, the image in np.uint8 1/3-channel numpy array, 
            the bounding boxes pandas DataFrame (with columns 'xtl', 'ytl', 'xbr', 'ybr') 
            and optionally the list of masks for annotated each object (each as a 
            numpy array of the same size the the bounding box specified in "annotations").
        crop_coords (4-tuple or 4-element list of int): xtl, ytl, xbr, and ybr 
            box coordinates for cropping.
        labels_of_interest (list of integers or strings or None): List of classnames or 
            class IDs of interest depending on the values reporeted in annotations_df['label'] 
            (names or IDs). Any annotated object outside this list will be removed from the 
            annotations. If None passed, all the object classed will be included. 
        block_label (integer or string): String class name or integer ID used for blocking 
            areas (hiding during training). This should be the same as the ID used in the 
            annotations (if used). 
        keep_area_threshold (float): The threshold on the ratio of the area of the 
            bounding boxes that lie inside the cropped image to keep. Bounding boxes 
            with at least keep_area_threshold of their area inside the cropped image 
            will be kept. Otherwise, all the  bounding boxes crossing the boundaries 
            of the cropped image will be removed. 'block' labels are kept to be 
            blacked out in the train/test images later. 
    """
    
    x1, y1, x2, y2 = crop_coords
    # make a copy of the input to make sure it is not modified
    name, image, df = sample['name'], sample['image'].copy(), sample['annotations'].copy()
    
    h, w = image.shape[:2]
    
    xc1 = int(max(x1, 0))
    yc1 = int(max(y1, 0))
    xc2 = int(min(x2, w))
    yc2 = int(min(y2, h))
    
    if xc2 <= xc1 or yc2 <= yc1:
        # incorrect input dimensions
        return None
    
    # sizes of cropped image
    crop_width = xc2 - xc1
    crop_height = yc2 - yc1
    
    # remove the bounding boxes that are totally outside the cropped image
    # keep the ones that have some non-zero overlap
        
    df = df[df.apply(lambda row: True if 
                     max(0, min(row['xbr'] - xc1, crop_width) - max(row['xtl'] - xc1, 0)) * \
                     max(0, min(row['ybr'] - yc1, crop_height) - max(row['ytl'] - yc1, 0)) > 0\
                     else False, axis = 1)]
    
    # crop the masks if available, and only keep the ones inside the crop
    # we further remove or block the partial objects
    if 'masks' in sample:
        masks = []
        idxs_to_keep = []
        for obj_id in df.index:
            # find the overlapping part between the object's bounding box (where the mask is defined within)
            # and the crop
            box_xtl, box_ytl, box_xbr, box_ybr = df.loc[obj_id, ['xtl', 'ytl', 'xbr', 'ybr']].values
            # the upper bound for xmin, ymin (the outher min) is not really needed becuase
            # the DataFrame is already filtered to keep overlapping bounding boxes with the crop 
            # with xc1 < box_xbr and yc1 < box_ybr for df.index
            # similarly, the lower bound of 0 for xmax, ymax (the inner max) is not needed 
            # becuase the DataFrame is already filtered and box_xtl < xbr2 and box_ytl < yc2 for df.index
            xmin = min(max(0, xc1 - box_xtl), box_xbr - box_xtl)
            xmax = min(max(0, xc2 - box_xtl), box_xbr - box_xtl)
            ymin = min(max(0, yc1 - box_ytl), box_ybr - box_ytl)
            ymax = min(max(0, yc2 - box_ytl), box_ybr - box_ytl)
            
            cropped_mask = sample['masks'][obj_id][ymin: ymax, xmin: xmax]
            
            # now update the bounding boxes as the mask confined to the crop may be smaller, hence different box 
            pos = np.where(cropped_mask)
            if len(pos[0]) == 0 or len(pos[1]) == 0:
                continue
            delta_x1 = np.min(pos[1])
            delta_x2 = np.max(pos[1])
            delta_y1 = np.min(pos[0])
            delta_y2 = np.max(pos[0])
            
            if delta_x1 >= delta_x2 or delta_y1 >= delta_y2:
                continue
            
            idxs_to_keep.append(obj_id)
            new_box_xtl = box_xtl + xmin + delta_x1
            new_box_xbr = box_xtl + xmin + delta_x2
            new_box_ytl = box_ytl + ymin + delta_y1
            new_box_ybr = box_ytl + ymin + delta_y2
            
            df.at[obj_id, 'xtl'] = new_box_xtl
            df.at[obj_id, 'ytl'] = new_box_ytl
            df.at[obj_id, 'xbr'] = new_box_xbr
            df.at[obj_id, 'ybr'] = new_box_ybr
            masks.append(cropped_mask[delta_y1:delta_y2, delta_x1:delta_x2].copy())
        df = df.loc[idxs_to_keep]    
            
    # transform the bounding boxes
    df[['xtl', 'xbr']] = df[['xtl', 'xbr']] - xc1
    df[['ytl', 'ybr']] = df[['ytl', 'ybr']] - yc1
    # reset the index
    df = df.reset_index(drop=True)

    # identify bounding boxes that would lie inside the newly cropped
    # image by more than keep_area_threshold; these boxes are kept 
    
    # also identify bounding boxes that would lie inside the newly cropped
    # image by less than keep_area_threshold and more than 30%; we are going 
    # to black out the area of these bounding boxes (together with objects 
    # identified with block_label in the annotations) in the image to prevent 
    # the model from seeing thes partial objects
    
    # for any object with overlap less than 30% with the cropped
    # sub-image, we only remove the bounding box. We allow the model to see the 
    # content of the partial object 
    
    if keep_area_threshold < 0.33:
        keep_area_threshold = 0.33
   
    
    if labels_of_interest is None:
        # consider all objects
        idxs_to_keep = df.apply(lambda row: True \
                                if (min(row['xbr'], crop_width) - max(row['xtl'], 0)) * 
                                (min(row['ybr'], crop_height) - max(row['ytl'], 0)) >= keep_area_threshold * 
                                (row['xbr'] - row['xtl']) * (row['ybr'] - row['ytl']) else False, axis=1)
        idxs_to_block = df.apply(lambda row: True \
                                 if ((min(row['xbr'], crop_width) - max(row['xtl'], 0)) * 
                                     (min(row['ybr'], crop_height) - max(row['ytl'], 0)) < keep_area_threshold * 
                                     (row['xbr'] - row['xtl']) * (row['ybr'] - row['ytl']) and 
                                     (min(row['xbr'], crop_width) - max(row['xtl'], 0)) * 
                                     (min(row['ybr'], crop_height) - max(row['ytl'], 0)) >= 0.3 * 
                                     (row['xbr'] - row['xtl']) * (row['ybr'] - row['ytl'])) or
                                 row['label'] == block_label else False, axis=1)
    else:
        # if labels_of_interest is provided, use it to only keep the ones we need to keep 
        idxs_to_keep = df.apply(lambda row: True \
                                if (min(row['xbr'], crop_width) - max(row['xtl'], 0)) * 
                                (min(row['ybr'], crop_height) - max(row['ytl'], 0)) >= keep_area_threshold * 
                                (row['xbr'] - row['xtl']) * (row['ybr'] - row['ytl']) and 
                                row['label'] in labels_of_interest else False, axis=1)
        idxs_to_block = df.apply(lambda row: True \
                                 if ((min(row['xbr'], crop_width) - max(row['xtl'], 0)) * 
                                     (min(row['ybr'], crop_height) - max(row['ytl'], 0)) < keep_area_threshold * 
                                     (row['xbr'] - row['xtl']) * (row['ybr'] - row['ytl']) and 
                                     (min(row['xbr'], crop_width) - max(row['xtl'], 0)) * 
                                     (min(row['ybr'], crop_height) - max(row['ytl'], 0)) >= 0.3 * 
                                     (row['xbr'] - row['xtl']) * (row['ybr'] - row['ytl']) and 
                                     row['label'] in labels_of_interest) or
                                 row['label'] == block_label  else False, axis=1)
        
    # limit the bounding boxes to image coordinates
    df.loc[df['xtl'] < 0, 'xtl'] = 0
    df.loc[df['ytl'] < 0, 'ytl'] = 0
    df.loc[df['xbr'] > crop_width, 'xbr'] = crop_width
    df.loc[df['ybr'] > crop_height, 'ybr'] = crop_height
    
    # now black out the image on the boxes that should be blacked out
    crop_mask = np.ones((crop_height, crop_width), dtype=np.uint8)
    
    # indicate the areas that should be blacked out from the bounding
    # boxes to be blocked
    for _, row in df[idxs_to_block].iterrows():
        crop_mask[int(row['ytl']):int(row['ybr']), int(row['xtl']):int(row['xbr'])]  = 0
    
    # then overwrite them by the annotated objects that we are going to keep
    # we do this to make sure we are not blacking out anything from the objects
    # that are going to be used for training, and only black out objects that we 
    # are not going to use (not annotated anymore)
    for _, row in df[idxs_to_keep].iterrows():
        crop_mask[int(row['ytl']):int(row['ybr']), int(row['xtl']):int(row['xbr'])]  = 1
    
    if 'masks' in sample:
        # keep the masks for the objects that we need to keep 
        # (lie in the crop)
        masks = [masks[i] for i in df[idxs_to_keep].index]
        for i, obj_id in enumerate(df[idxs_to_keep].index):
            box_xtl, box_ytl, box_xbr, box_ybr = df.loc[obj_id, ['xtl', 'ytl', 'xbr', 'ybr']].values
            masks[i] = (masks[i] * crop_mask[box_ytl:box_ybr, box_xtl:box_xbr]).astype(np.uint8)
    
    # reset the indexes in the annotations DataFrame
    df = df[idxs_to_keep].reset_index(drop=True)
    
    if len(image.shape) > 2:
        # 3 channel image
        crop_mask = crop_mask[:, :, np.newaxis]
    
    if 'masks' in sample:
        return  {'name': name, 'image': image[yc1: yc2, xc1: xc2] * crop_mask, 
                 'annotations': df, 'masks': masks} 
    
    return  {'name': name, 'image': image[yc1: yc2, xc1: xc2] * crop_mask, 'annotations': df}


def enforce_one_to_one_mapping(data_sample: dict, child_parent_map: Dict[int, int]) -> dict:
    """
    A function to clean up annotations and enforce a one-to-one mapping between child-parent pairs of objects.
    For some annotations, there is a hierarchy between objects of different classes. For example, the mask of an 
    object from class 'nucleus' should always be inside the mask of another object of class 'cytoplasm'/'cell-adhered'. 
    Furthermore, this mapping has to be one-to-one in certain scenarios. A 'cytoplasm'/'cell-adhered' object can only 
    have one 'nucleus', and a 'nucleus' object should belong to only one 'cytoplasm'/'cell-adhered'. 
    In many annotated images, different objects of class 'cytoplasm' cannot be accurately differentiated and multiple 
    objects are combined (hence have multiple nuclei). This function is used to remove these invalid annotations by 
    passing a set of child:parent relationships that should be enforced in 'child_parent_map' argument. 
    
    Args:
        - data_sample (dictionary): A dictionary with keys and values as 
            'name'(str): image name,
            'image'(numpy.ndarray): H x W (Gray scale) or H x W x 3 image array in BGR format,
            'annotations' (pandas DataFrame): A DataFrame with columns 'xtl', 'ytl', 'xbr', 'ybr', 'label'.
                The i-th row is the bounding box coordinates and the label for the i-th annotated object,
            'masks'(list of numpy.ndarrays): masks[i] is the H x W array mask for the i-th object with value 1 for 
                the object.
        - child_parent_map (dictionary): A dictionary with integer keys and values specifying the one-to-one relationship
            as between child class ID and parent class ID. 
    Returns:
        - The filtered data_sample with invalid objects removed and blacked-out in the image.  
    """
    
    # make sure the index in the DataFrame is the same as the row index
    annotations: pd.DataFrame  = data_sample['annotations'].reset_index(drop=True).copy()
    
    # the indexes of invalid objects from the child and parent classes
    # objects from the child classes (with class IDs specified by the keys in child_parent_map)
    # that have no objects from the mapped parent class including them, or have multiple objects from 
    # the mapped parent class covering them (not expected)
    invalid_child_idxs: List[int] = []
    # objects from the parent classes (with class IDs specified by the values in child_parent_map)
    # that have no objects from the mapped child class belonging to them, or cover multiple objects from 
    # the mapped child class 
    invalid_parent_idxs: List[int] = []
        
    # go over the child-parent pairs for which one-to-one mapping should be enforced
    for child_class_id, parent_class_id in child_parent_map.items():
    
    
        child_df: pd.DataFrame = annotations[annotations['label'] == child_class_id]
        parent_df: pd.DataFrame = annotations[annotations['label'] == parent_class_id]
        
        # indexes
        child_idxs: np.ndarray = child_df.index.values
        parent_idxs: np.ndarray = parent_df.index.values
            
        if len(child_idxs) == 0:
            if len(parent_idxs) > 0:
                invalid_parent_idxs += parent_idxs.tolist()    
            continue
        elif len(parent_idxs) == 0:
            invalid_child_idxs += child_idxs.tolist()
            continue    
        
        # bounding boxes
        child_boxes: np.ndarray = child_df[['xtl', 'ytl', 'xbr', 'ybr']].values
        parent_boxes: np.ndarray = parent_df[['xtl', 'ytl', 'xbr', 'ybr']].values

        # masks
        child_masks: List[np.ndarray] = [data_sample['masks'][idx] for idx in child_idxs]
        parent_masks: List[np.ndarray] = [data_sample['masks'][idx] for idx in parent_idxs]

        # the IoU matrix for the bounding boxes, element (i, j) in the matrix below is the 
        # IoU between the bounding box of the child object i (with index child_idxs[i] in the 
        # data_sample['annotations'] and data_sample['masks']) and the bounding box of the 
        # parent object j (with index parent_idxs[j])
        iou_box_matrix: np.ndarray = iou_batch(child_boxes, parent_boxes)
    
        overlapping_child_idxs, overlapping_parent_idxs = np.where(iou_box_matrix > 0)
    
        # the IoU matrix for the masks, for efficiency reasons, it is only calculated
        # for pairs with box IoU > 0
        iou_mask_matrix: np.ndarray = np.zeros((len(child_idxs), len(parent_idxs)))
    
        for (i, j) in zip(overlapping_child_idxs, overlapping_parent_idxs):        
            iou_mask_matrix[i, j] = iou_mask_pair(child_boxes[i], child_masks[i], 
                                                  parent_boxes[j], parent_masks[j])

        iou_mask_matrix[iou_mask_matrix > 0] = 1
        
        # the (columns) index of parent objects without an object from the child class or 
        # with multiple objects from the child class 
        invalid_col_idxs: np.ndarray = np.where(np.sum(iou_mask_matrix, axis=0) != 1)[0]
        # only keep objects from the parent class with one child, before finding invalid 
        # objects of class child
        valid_col_idxs: np.ndarray = np.array([idx for idx in range(len(parent_idxs)) 
                                               if idx not in invalid_col_idxs])
            
        if len(valid_col_idxs) == 0:
            invalid_row_idxs: np.ndarray = np.array([idx for idx in range(len(child_idxs))])
        else:
            # the (row) index of child objects without an object from the parent class or 
            # with multiple objects from the parent class 
            invalid_row_idxs: np.ndarray = np.where(np.sum(iou_mask_matrix[:, valid_col_idxs], axis=1) != 1)[0]    
            
        
        invalid_child_idxs += child_idxs[invalid_row_idxs].tolist()
        invalid_parent_idxs += parent_idxs[invalid_col_idxs].tolist()
    
    # black out invalid objects in the image, and remove them from the annotations and masks
    img: np.ndarray = data_sample['image'].copy()
    img_mask: np.ndarray = np.zeros(img.shape[:2], np.uint8)
    blacked_out_mask: np.ndarray = np.ones(img.shape[:2], np.uint8)

    valid_idxs: List[int] = []
    for idx, row in annotations.iterrows():
        xtl, ytl, xbr, ybr = row[['xtl', 'ytl', 'xbr', 'ybr']].values
        if idx in (invalid_child_idxs + invalid_parent_idxs):
            blacked_out_mask[ytl:ybr, xtl:xbr] = 0
        else:
            img_mask[ytl:ybr, xtl:xbr] = 1
            valid_idxs.append(idx)
    out: dict = {}
    out['name'] = data_sample['name']
    out['image']: np.ndarray = cv2.bitwise_or(blacked_out_mask, img_mask) * img
    out['annotations']: pd.DataFrame = annotations.loc[valid_idxs].reset_index(drop=True).copy()
    out['masks']: List[np.ndarray] = [data_sample['masks'][idx].copy() for idx in valid_idxs]
        
    return out

# a function to display the samples
COLORS = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

def show_sample(sample, class_id_to_name_mapping=None):
    """
    A function to display the annotations on the image. 
    
    Args:
        sample (dictionary): Input data sample to be displayed. The dictionary
            should include "name", "image", "annotations" and optionally "masks" keys 
            for passing the image name, the image in np.uint8 1/3-channel numpy array, 
            the bounding boxes pandas DataFrame (with columns 'xtl', 'ytl', 'xbr', 'ybr', 
            'label') and optionally the list of masks for annotated each object (each as a 
            numpy array of the same size as the object's bounding box returned in "annotations").
        class_id_to_name_mapping (dictionary or None): A dictionary with keys as class IDs 
            and values as class names. Should be passed if sample['annotations']['label']
            includes class IDs instead of names. Otherwise, None should be passed. 
    
    Returns numpy array of the image with annotations added
    
    """

    image = sample['image']
    # convert to 3-channels
    if len(image.shape) < 3:
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2)
    annotations_df = sample['annotations']
    boxes = annotations_df[['xtl', 'ytl', 'xbr', 'ybr']].values.astype(np.int32)
    labels  = annotations_df['label'].values
    if 'masks' in sample:
        masks = sample['masks']
    
    if class_id_to_name_mapping is None:
        # annotations_df include class names
        name_to_id_mapping = {key: idx for idx, key in enumerate(annotations_df['label'].unique())}
    else:
        name_to_id_mapping = {value: key for key, value in class_id_to_name_mapping.items()}
    
    for i in range(len(annotations_df)):
        # the bounding box
        (xtl, ytl, xbr, ybr) = boxes[i]
        if class_id_to_name_mapping is None:
            color = COLORS[name_to_id_mapping[labels[i]] % len(COLORS)]
            text = labels[i]
        else:
            if labels[i] in class_id_to_name_mapping:
                color = COLORS[labels[i] % len(COLORS)]
                text = class_id_to_name_mapping[labels[i]]
            else:
                print('Incorrect ID was found %s' %labels[i])
                # use black for incorrect label
                text = 'Unknown'
                color = (0, 0, 0)
   
                
        cv2.rectangle(image, (xtl, ytl), (xbr, ybr), color, 1)
        cv2.putText(image, text, (xtl, ytl + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        if 'masks' in sample:
            color_mask = color * np.repeat(np.expand_dims(masks[i], axis=2), 3, axis=2)
            blended = 0.4 * color_mask
            blended[color_mask == 0] = image[ytl:ybr, xtl:xbr][color_mask == 0]
            blended[color_mask > 0] += 0.6 * image[ytl:ybr, xtl:xbr][color_mask > 0]

            # store the blended ROI in the original image
            image[ytl:ybr, xtl:xbr] = blended.astype(np.uint8)
        
  
    return image
    
