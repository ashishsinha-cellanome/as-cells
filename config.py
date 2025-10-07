from typing import Any, Final
import os
ROOT_DATADIR = '/global/home/ashish.sinha/cellanome/datasets'

MAX_IMAGE_SIDE: Final[int] = 4512
RESIZED_BB_IMAGE_SIZE = 3840 ###### 2720 for Mask R-CNN 10x
RATIO_OF_IMAGES_TO_USE_FOR_SUSPENSION_CAGED_DATASETS = 0.25   ###### was 0.2
RATIO_OF_IMAGES_TO_USE_FOR_SUSPENSION_UNCAGED_DATASETS = 0.05 ###### was 0.02
RATIO_OF_IMAGES_TO_USE_FOR_ADHERED_DATASETS = 1.0

CLASS_NAMES_TO_CLASS_IDS_MAP =  {'nucleus': 6,
                                 'soma': 5,
                                 'cytoplasm': 4,  'cell-adhered': 4,
                                 'cage': 3, 'cages': 3, 
                                 'bead': 2, 'Bead': 2,
                                 'cell': 1, 'Cell': 1, 'dead-cell': 1, 'dying/dead cells': 1}

CLASS_IDS_TO_CLASS_NAMES_MAP = {6: 'nucleus',
                                5: 'soma',
                                4: 'cytoplasm',
                                3: 'cage', 
                                2: 'bead', 
                                1: 'cell'}
                                
NUM_CLASSES = len(set(CLASS_NAMES_TO_CLASS_IDS_MAP.values()))

CLASS_NAMES_TO_USE_FOR_CROP_OVERLAPS = ['cell', 'cell-adhered', 'soma',] # 'cage']

NUM_RANDOM_IMAGES_TO_CONSIDER_IN_Z_STACK_SET: int = 2
# the minimum object mask area to keep the object in the data
MIN_MASK_AREA = 16

CHILD_PARENT_CLASS_MAP = None
# the limit on the larger and smaller sides of the input to Mask R-CNN model
# used for preparing un-cropped training images (whole images)
MASK_RCNN_INPUT_WIDTH = 1024
MASK_RCNN_INPUT_HEIGHT = 800
FL_CH_IDENTIFIER_TO_COLOR_FOR_OVERLAY = {'Violet': 'red', 'Blue': 'blue'}

# configs for running inference
MODEL_CKPT_DIR = '/global/home/ashish.sinha/cellanome/models'

# RT-detr related
RESIZE_DICT_10x = {
    (2000, 1600): (1000, 800),
    (4512, 4512): (2440, 2440),
}
RESIZE_DICT_4x = {
    (2000, 1600): (1000, 800),
    (4512, 4512): (4512, 4512),
}


MODELS = {
    'mask2former': {
        'model_class': 'Mask2FormerInstanceSegmentation',
        'weights_path': os.path.join(MODEL_CKPT_DIR, "mask2former_checkpoints/20250312_mask2former_sets_1_2_3_6_to_41_0p1_bbox_0p7_1_rs_0p25_blur_2_bs_8_epochs_1cl_lrs.pt"),
        'model_name': 'Mask2Former',
        'adjust_masks': True,
        'task_type': 'instance_segmentation',
    },
    'mask-rcnn': {
        'model_class': 'MaskRCNNInstanceSegmentation',
        'weights_path': os.path.join(MODEL_CKPT_DIR, "mask_rcnn_checkpoints/updated-20240923_sets_1_2_3_6_to_38_0p1_bbox_0p7_1_rs_0p25_blur_2_bs_8_epochs_1cl_lrs.pt"),
        'model_name': 'Mask R-CNN',
        'task_type': 'instance_segmentation',
    },
    'deformable-detr': {
        'model_class': 'DeformableDetrObjectDetector',
        'weights_path': os.path.join(MODEL_CKPT_DIR, "deformable_detr_with_sam_checkpoints/20250511_deformable_detr_sam.pt"),
        'model_name': 'Deformable DETR',
        'task_type': 'object_detection',
        'backbone_name_str': 'sam',
    },
    'rf-detr': {
        'model_class': 'RfDetrObjectDetector',
        'weights_path': os.path.join(MODEL_CKPT_DIR, "rf_detr_checkpoints/updated-20250410_rf_detr_best_total.pth"),
        'model_name': 'RF-DETR',
        'task_type': 'object_detection',
        'backbone_name_str': 'sam',
    },
    'rt-detr-v1': {
        'model_class': 'RTDeTRObjectDetector',
        'weights_path': os.path.join(MODEL_CKPT_DIR, "rt_detr_v1_default/20250331_sets_1_2_3_6_to_41_rt_detr_16_bs_10_epochs.pt"),
        'model_name': 'RT-DETR v1',
        'task_type': 'object_detection',
        'backbone_name_str': None,
    },
    'rt-detr-v2': {
        'model_class': 'RTDeTRObjectDetector',
        'weights_path': os.path.join(MODEL_CKPT_DIR, "rt_detr_v2_with_dinov2_fpn_2_7_12/20250603_sets_1_2_3_6_to_41_rt_detrv2_with_dinov2_fpn_2_7_12_16_bs_10_epochs.pt"),
        'model_name': 'RT-DETR v2',
        'task_type': 'object_detection',
        'backbone_name_str': 'dinov2',
    },
}