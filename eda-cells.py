import os
import sys
import cv2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import supervision as sv

from PIL import Image
from glob import glob
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple, List, Final

from utils.json_parser import (
    CellMaskDataset,
    parse_json_annotations,
    parse_json_annotations_2p0,
    create_overlaid_img
)
from utils.cv_utils import (
    get_crop_corners,
    show_detections
)

# configs
from config import *

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

def analyze_cell_size(dataset, dataset_name, train=True):
    """
    Args:
        dataset: CellMaskDataset
        dataset_name: str
        train: bool
    Returns:
        mask_data: pd.DataFrame
    
    Use:
        analyze_cell_size(train_dataset, dataset_path.name, True)
    """
    mask_data = []
    for sample in tqdm(dataset, desc=f"Analyzing {dataset_name}"):
        masks = sample['masks']
        labels = sample['annotations'].label.values
        img_name = sample['name'].split('.')[0]
        for mask, label_id in zip(masks, labels):
            mask_area = np.sum(mask)
            label_name = CLASS_IDS_TO_CLASS_NAMES_MAP.get(label_id, 'unknown')
            mask_data.append({
                'dataset': dataset_name,
                'img_name': img_name,
                'label': label_name,
                'mask_area': mask_area
            })
    mask_data = pd.DataFrame(mask_data)

    return mask_data

def plot_mask_size_distribution(all_datasets, class_ids_to_class_names_map, **kwargs):
    """
    Analyzes and plots the size distribution of masks per label across datasets.

    Args:
        all_datasets (list): A list of Path objects for each dataset.
        class_ids_to_class_names_map (dict): A mapping from class IDs to names.
        **kwargs: Arguments for dataset loading functions.
    """
    print("Starting mask size distribution analysis...")

    # Data collection
    all_mask_data = []

    for dataset_path in tqdm(all_datasets, desc="Processing Datasets"):
        train_ds, test_ds = create_dataset_classes(dataset_path) # Use the dummy function
        
        for sample in tqdm(train_ds, desc=f"Analyzing {dataset_path.name}"):
            masks = sample['masks']
            labels = sample['annotations'].label.values
            
            for mask, label_id in zip(masks, labels):
                mask_area = np.sum(mask)
                label_name = CLASS_IDS_TO_CLASS_NAMES_MAP.get(label_id, 'unknown')
                all_mask_data.append({
                    'dataset': dataset_path.name,
                    'label': label_name,
                    'mask_area': mask_area
                })

    # Create master DataFrame from collected data
    df = pd.DataFrame(all_mask_data)

    if df.empty:
        print("[ERROR] No masks found to analyze.")
        return

    # Save the detailed analysis DataFrame
    output_df_path = "mask_size_distribution_data.csv"
    df.to_csv(output_df_path, index=False)
    print(f"Detailed analysis saved to: {output_df_path}")

    # Plotting
    os.makedirs('plots', exist_ok=True)
    
    ## Per-Dataset Plots
    for dataset_name in df['dataset'].unique():
        fig, ax = plt.subplots(figsize=(10, 6))
        dataset_df = df[df['dataset'] == dataset_name]
        sns.violinplot(x='label', y='mask_area', data=dataset_df, ax=ax)
        ax.set_title(f'Mask Size Distribution in {dataset_name}')
        ax.set_xlabel('Class Label')
        ax.set_ylabel('Mask Area (in pixels)')
        plt.tight_layout()
        output_plot_path = f"plots/mask_size_distribution_{dataset_name}.png"
        plt.savefig(output_plot_path)
        plt.close(fig)
        print(f"Plot for '{dataset_name}' saved to: {output_plot_path}")
    
    ## Across All Datasets Plot 
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.violinplot(x='label', y='mask_area', hue='dataset', data=df, ax=ax, split=True)
    ax.set_title('Mask Size Distribution Across All Datasets')
    ax.set_xlabel('Class Label')
    ax.set_ylabel('Mask Area (in pixels)')
    plt.legend(title='Dataset')
    plt.tight_layout()
    output_plot_path_all = "plots/mask_size_distribution_all_datasets.png"
    plt.savefig(output_plot_path_all)
    plt.close(fig)
    print(f"Combined plot saved to: {output_plot_path_all}")

def plot_cell_size_distribution(df, save_name, log=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    if log:
        ax.set_yscale('log')
    sns.violinplot(x='label', y='mean_mask_area', data=df, hue='dataset', ax=ax, split=False)
    ax.set_title('Cell Size (mask area) Distribution')
    ax.set_xlabel('Class Label')
    if log:
        ax.set_ylabel('Avg. Mask Area (in pixels) (log-scale)')
    else:
        ax.set_ylabel('Avg. Mask Area (in pixels)')
    plt.tight_layout()
    
    plt.savefig(save_name)
    plt.close(fig)
    print(f"Plot saved to: {save_name}")

if __name__ == '__main__':
    BASE_PATH = Path(ROOT_DATADIR)
    OUTPUT_DIR = (BASE_PATH.parent / 'eda-analysis')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_datasets = list(BASE_PATH.glob('*'))

    print ('Total datasets found: ', len(all_datasets))
    dataset_path = all_datasets[0]
    for dataset_path in tqdm(all_datasets, desc='Datasets'):
        print ('Using dataset: ', dataset_path)
        try:
            train_dataset, test_dataset = create_dataset_classes(
                dataset_path=dataset_path,
                class_names_to_class_ids_map=CLASS_NAMES_TO_CLASS_IDS_MAP,
                percentage_to_expand_bbox_boundaries=0.1,
                max_images_to_consider_for_each_annotation=5,
                only_use_best_focus_image=True,
                max_larger_side=MAX_IMAGE_SIDE,
                max_smaller_side=MAX_IMAGE_SIDE,
                fl_channel_id_to_color_map_for_overlay=FL_CH_IDENTIFIER_TO_COLOR_FOR_OVERLAY
            )
        except Exception as e:
            print (f'[ERROR] {e}')
            continue
        print(f"[INFO]: Number of training samples: {len(train_dataset)}")
        print(f"[INFO]: Number of testing samples: {len(test_dataset)}")

        sample_idx = 0
        sample = train_dataset[sample_idx]
        # import ipdb; ipdb.set_trace()
        
        
        df_name = OUTPUT_DIR / (dataset_path.name + '_mask-size-distro.csv')
        if df_name.exists():
            continue
        cell_df = analyze_cell_size(train_dataset, dataset_path.name, train=True)
        cell_df.to_csv(df_name, index=False)
        cell_df = pd.read_csv(df_name)
        image_stats_df = cell_df.groupby(['dataset', 'img_name', 'label']).agg(
            mean_mask_area=('mask_area', 'mean'),
            median_mask_area=('mask_area', 'median'),
            count=('mask_area', 'count')
        ).reset_index()
        # import pdb; pdb.set_trace()

        save_dset_plot_name = OUTPUT_DIR / (dataset_path.name+"_mask-size-distro-per-dataset.png")
        plot_cell_size_distribution(image_stats_df, save_dset_plot_name, log=True)

        print (f"[INFO] Keys: {','.join(sample.keys())}")
        print(f"[INFO] Sample image shape: {sample['image'].shape}")
        print(f"[INFO] Sample masks shape: {len(sample['masks'])}")
        print(f"[INFO] Sample annotation shape: {sample.get('annotations').shape}")
        print(f"[INFO] Unique labels present in the image: {sample['annotations'].label.unique().tolist()}")

        # visualize a sample
        anno = sample['annotations'] # Nx5
        labels = anno.label.values # N
        boxes = anno[anno.columns[:-1]].values# Nx4
        masks = sample['masks'] # List[[M1,N1], [M2,N2]...[M_N, N_N]]
        image = sample['image'] # H x W

        gt_preds = {
            'masks': masks,
            'labels': labels,
            'boxes': boxes
        }
        # import pdb; pdb.set_trace()
        img_w_gt_boxes = show_detections(input_image=image, predictions = gt_preds, label_map=CLASS_IDS_TO_CLASS_NAMES_MAP)
        save_name = OUTPUT_DIR / f'vis-{sample.get("name")}'
        cv2.imwrite(save_name, img_w_gt_boxes)