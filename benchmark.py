
import os
import sys
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from typing import List, Dict, Union, Tuple, Final
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import time

import torch
import random
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from config import (
    MODEL_CKPT_DIR,
    MODELS,
    ROOT_DATADIR,
    CLASS_IDS_TO_CLASS_NAMES_MAP,
    CLASS_NAMES_TO_CLASS_IDS_MAP,
    FL_CH_IDENTIFIER_TO_COLOR_FOR_OVERLAY,
    MAX_IMAGE_SIDE,
    RESIZE_DICT_10x,
    RESIZE_DICT_4x,
)
# seed everything
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# import all the models
from tqdm import tqdm
from models.AbstractVisionModel import run_model, VisionModel
from models.mask2former_model import Mask2FormerInstanceSegmentation
from models.mask_rcnn_model import MaskRCNNInstanceSegmentation
from models.deformable_detr_model import DeformableDetrObjectDetector
from models.rf_detr_model import (
    DEFAULT_CROP_CORNERS,
    DEFAULT_RESIZE, 
    DEFAULT_LABEL_MAP, 
    DEFAULT_MODEL_INPUT_SIZE,
    RfDetrObjectDetector,
    )
from models.rt_detr_model import RTDeTRObjectDetector

from utils.model_utils import get_crop_corners

from utils.dataset_utils import create_dataset_classes
from utils.json_parser import TestDataSet, OPTICAL_CHARACTERISTICS, CellMaskDataset
from utils.precision_recall_eval import evaluate_yolo_pr, calculate_confusion_matrix, p_r_based_on_c_m
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

TEST_DATASETS = [
    os.path.join(ROOT_DATADIR, '20240228_jurkat_10x_caged'),
    os.path.join(ROOT_DATADIR, '20240228_k562_10x_uncaged'),
    os.path.join(ROOT_DATADIR, '20240228_nk92_10x_uncaged'),
    os.path.join(ROOT_DATADIR, '20240305_pbmc-nobeads_10x_uncaged'),
    os.path.join(ROOT_DATADIR, '20240314_imr90-suspension_10x_caged'),
    os.path.join(ROOT_DATADIR, '20240703_neuron-adhered_10x_caged'),
    os.path.join(ROOT_DATADIR, '20240924_enteric-glia-adhered_10x_uncaged'),
]

class Benchmark:
    def __init__(self, model_key: str):
        if model_key not in MODELS:
            raise ValueError(f"Model key '{model_key}' not found in MODELS dictionary.")
        self.model_key = model_key
        model_info = MODELS.get(model_key)
        model = getattr(sys.modules[__name__], model_info['model_class'])
        if model_key == 'rf-detr':
            self.model = model(
                weights_path = model_info['weights_path'],
                label_map=DEFAULT_LABEL_MAP, 
                model_input_size=DEFAULT_MODEL_INPUT_SIZE
            )
        elif model_key in ['deformable-detr', 'rt-detr-v2']:
            self.model = model(
                weights_path = model_info['weights_path'],
                backbone_name_str = model_info['backbone_name_str']
            )
        else:
            self.model = model(
                weights_path=model_info['weights_path'],
            )
        print (f"Loaded model: {model_info['model_name']}")
        print (f'Model Metadata: {self.model.get_metadata()}')
        self._metadata = self.model.get_metadata()
        self.label_names = self.model.get_class_names()
        self.id2label_map = self.model.get_label_map()
        self.label2id_map = self.model.get_reverse_label_map()
    
    def create_dataset(self, dataset_path: str) -> Tuple[CellMaskDataset, CellMaskDataset]:
        """
        Given the path to a dataset, create the training and testing dataset objects.
        The dataset path should contain 'train' and 'test' subdirectories with COCO-style
        annotations and images.
        """
        train_dataset, test_dataset = create_dataset_classes(
            dataset_path = dataset_path,
            class_names_to_class_ids_map=self.label2id_map,
            percentage_to_expand_bbox_boundaries=0.1,
            max_images_to_consider_for_each_annotation=5,
            only_use_best_focus_image=True,
            max_larger_side=MAX_IMAGE_SIDE,
            max_smaller_side=MAX_IMAGE_SIDE,
            fl_channel_id_to_color_map_for_overlay=FL_CH_IDENTIFIER_TO_COLOR_FOR_OVERLAY
        )
        self.train_set = train_dataset
        self.test_set = test_dataset
        self.test_dataset_path = dataset_path

        return train_dataset, test_dataset
    
    def infer(self, normalize_image: bool = True, bit_depth: int = 12, plot_detections: bool = False, save_path: str = None) :
        if plot_detections:
            assert save_path is not None, f"Since plot_detections is set to True, Please provide a save_path to save the images with detections."
    
        self.predictions = []
        self.runtimes = []
        is_4x = self._metadata['magnification'].lower() == '4x'
        for idx in tqdm(range(len(self.test_set)), desc='Inference'):
        # for idx in tqdm(4, desc='Inference'):
            data = self.test_set[idx]
            # print (data['name'])
            img = data['image']
            if isinstance(self.model, RfDetrObjectDetector):
                outputs = run_model(
                    detector=self.model,
                    input_image=img,
                    input_resize=DEFAULT_RESIZE, # need to pass them as the weights files does not provide them
                    input_crop_corners=DEFAULT_CROP_CORNERS, # need to pass them as the weights files does not provide them
                    normalize_image=normalize_image,
                    bit_depth=bit_depth,
                    post_process_class_names=list(self.model.get_reverse_label_map().keys()),
                    plot_results=plot_detections,
                )
            elif isinstance(self.model, RTDeTRObjectDetector):
                CROP_CORNERS_DICT_10x = {
                    (2000, 1600): get_crop_corners(1000, 800, input_size = self.model._model_input_size),
                    (4512, 4512): get_crop_corners(2440, 2440, input_size = self.model._model_input_size)
                }
                CROP_CORNERS_DICT_4x = {
                    (2000, 1600): get_crop_corners(1000, 800, input_size = self.model._model_input_size),
                    (4512, 4512): get_crop_corners(4512, 4512, input_size = self.model._model_input_size)
                }
                outputs = run_model(
                    detector=self.model,
                    input_image=img,
                    input_resize=RESIZE_DICT_4x if is_4x else RESIZE_DICT_10x, # need to pass them as the weights files does not provide them
                    input_crop_corners=CROP_CORNERS_DICT_4x if is_4x else CROP_CORNERS_DICT_10x, # need to pass them as the weights files does not provide them
                    normalize_image=normalize_image,
                    bit_depth=bit_depth,
                    post_process_class_names=list(self.model.get_reverse_label_map().keys()),
                    plot_results=plot_detections,
                )
            else:
                outputs = run_model(
                    detector=self.model,
                    input_image=img,
                    normalize_image=normalize_image,
                    bit_depth=bit_depth ,
                    post_process_class_names=list(self.model.get_reverse_label_map().keys()),
                    plot_results=plot_detections,
                )
            if plot_detections:
                detections, runtime, debug_img = outputs
                os.makedirs(save_path, exist_ok=True)
                # save every 10 iterations
                if idx % 10 == 0:
                    img_save_path = os.path.join(save_path, f"{data['name'].split('.')[0]}_detections.png")
                    cv2.imwrite(img_save_path, debug_img)
            else:
                detections, runtime = outputs
            self.predictions.append(detections)
            self.runtimes.append(runtime)

        print (f"{self.model.get_model_name()} took on average {np.mean(self.runtimes):.4f} s/image for inference on {len(self.test_set)} images.")
        return self.predictions, self.runtimes
    
    def evaluate_predictions(self, predictions, dataset, model, iou_threshold: float = 0.5):
        """
        Compute precision, recall and F1 scores for predictions
        
        Args:
            predictions: List of prediction dictionaries
            dataset: TestDataSet instance 
            model: Model instance (for label mapping)
            iou_threshold: IoU threshold for matching
            
        Returns:
            Tuple of (precision_dict, recall_dict, f1_dict)
        """

        model_label_map = self.id2label_map.copy()
        precision = {}
        recall = {}
        f1 = {}
        # print (model_label_map)
        # import pdb; pdb.set_trace()
        # if predictions is not None and np.array(predictions[-1]['labels']).min() == 0:
        #     for pred in predictions:
        #         pred['labels'] = (np.array(pred['labels']) + 1).tolist()
        c_m_df, p_r_df, p_r_with_break_down_df = p_r_based_on_c_m(
            predictions=predictions,
            dataset=dataset,
            class_ids_to_classnames_map=self.id2label_map,
            model_label_map=self.id2label_map,
            annotation_filter=None,
            use_masks=self.model.get_metadata()['predict_masks']
        )
        p_r_df['Class'] = p_r_df.index
        print("\nConfusion Matrix:")
        print(c_m_df)
        print("\nPrecision-Recall Metrics:")
        print(p_r_df)
        
        # Extract metrics per class
        for class_name in model_label_map.values():
            if class_name not in p_r_df.index:
                print (f"Class {class_name} not found in precision-recall DataFrame")
                continue
            # import pdb; pdb.set_trace()
            
            class_metrics = p_r_df[p_r_df['Class'] == class_name].iloc[0]
            precision[class_name] = class_metrics['Precision']
            recall[class_name] = class_metrics['Recall']
            
            if precision[class_name] + recall[class_name] > 0:
                f1[class_name] = 2 * (precision[class_name] * recall[class_name]) / (precision[class_name] + recall[class_name])
            else:
                f1[class_name] = 0.0
                
            print(f"\n{class_name}:")
            print(f"Precision: {precision[class_name]:.4f}")
            print(f"Recall: {recall[class_name]:.4f}")
            print(f"F1 Score: {f1[class_name]:.4f}")
            
        return precision, recall, f1

    def compute_metrics(self, preds, gt_dataset, iou_threshold: float = 0.5):
        # Placeholder for metric computation logic
        precision, recall, f1 = self.evaluate_predictions(
            predictions=preds,
            dataset=gt_dataset,
            model=self.model,
            iou_threshold=iou_threshold
        )
        return precision, recall, f1
        
    def get_model_summary(self, results_dir: str, output_dir: str = None) -> pd.DataFrame:
        """
        Collect and summarize all metrics for this model across different datasets.
        
        Args:
            results_dir: Directory containing all results CSV files
            output_dir: Optional directory to save summary plots and CSV files
            
        Returns:
            DataFrame containing summarized metrics
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Collect all results for this model
        model_results = []
        for file in glob.glob(os.path.join(results_dir, '**/**.csv'), recursive=True):
            file_name = Path(file).name
            if file_name.startswith(f"{self.model_key}_") and file_name.endswith("_results.csv"):
                df = pd.read_csv(file)
                
                dataset_name = Path(file).parent.parent.name
                # dataset_name = file.replace(f"{self.model_key}_", "").replace("_results.csv", "")
                df['dataset'] = dataset_name
                model_results.append(df)
        
        if not model_results:
            raise ValueError(f"No results found for model {self.model_key} in {results_dir}")
            
        # Combine all results
        combined_df = pd.concat(model_results, ignore_index=True)
        
        # Compute summary statistics
        summary_df = combined_df.groupby('class_name').agg({
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'f1': ['mean', 'std'],
            'inference_time_ms': ['mean', 'std'],
            'dataset_size': 'sum'
        }).round(4)
        
        # Flatten column names
        summary_df.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] 
                            for col in summary_df.columns]
        
        if output_dir:
            # Save summary to CSV
            summary_path = os.path.join(output_dir, f"{self.model_key}_summary.csv")
            summary_df.to_csv(summary_path)
            print(f"Summary saved to {summary_path}")
            
            # Create and save visualizations
            self._plot_metric_distributions(combined_df, output_dir)
            self._plot_metric_correlations(combined_df, output_dir)
            self._plot_class_performance(summary_df, output_dir)
        
        return summary_df
    
    def _plot_metric_distributions(self, df: pd.DataFrame, output_dir: str):
        """Plot distributions of precision, recall, and F1 scores across datasets."""
        plt.figure(figsize=(15, 5))
        
        metrics = ['precision', 'recall', 'f1']
        for i, metric in enumerate(metrics, 1):
            plt.subplot(1, 3, i)
            sns.boxplot(data=df, x='class_name', y=metric)
            plt.xticks(rotation=45)
            plt.title(f'{metric.capitalize()} Distribution by Class')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{self.model_key}_metric_distributions.png"))
        plt.close()
        
    def _plot_metric_correlations(self, df: pd.DataFrame, output_dir: str):
        """Plot correlations between metrics."""
        metrics = ['precision', 'recall', 'f1', 'inference_time_ms']
        corr = df[metrics].corr()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        plt.title('Metric Correlations')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{self.model_key}_metric_correlations.png"))
        plt.close()
        
    def _plot_class_performance(self, summary_df: pd.DataFrame, output_dir: str):
        """Plot average performance metrics with error bars for each class."""
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(summary_df.index))
        width = 0.25
        
        metrics = ['precision', 'recall', 'f1']
        for i, metric in enumerate(metrics):
            means = summary_df[f'{metric}_mean']
            stds = summary_df[f'{metric}_std']
            
            plt.bar(x + i*width, means, width, 
                   label=metric.capitalize(),
                   yerr=stds, 
                   capsize=5)
        
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.title('Average Performance by Class')
        plt.xticks(x + width, summary_df.index, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{self.model_key}_class_performance.png"))
        plt.close()
    
    def compute_per_image_metrics(self, predictions: List[Dict], dataset: CellMaskDataset, iou_threshold: float = 0.5) -> pd.DataFrame:
        """
        Compute precision, recall, and F1 score for each image in the dataset
        
        Args:
            predictions: List of prediction dictionaries for each image
            dataset: Dataset containing ground truth annotations
            iou_threshold: IoU threshold for considering a match
            
        Returns:
            DataFrame with per-image metrics
        """
        records = []
        
        for idx, (pred, data) in enumerate(zip(predictions, dataset)):
            # Get ground truth annotations for this image
            gt_annots = data['annotations']

            gt_boxes = gt_annots[gt_annots.columns[:-1]].values
            gt_labels = gt_annots.label.values
            image_name = data['name']
            
            # Get predictions for this image
            pred_boxes = pred['boxes']
            pred_labels = np.array(pred['labels'])
            pred_scores = pred['scores']
            
            # Initialize metrics for each class
            class_metrics = {}
            
            # Get all unique classes in both GT and predictions
            gt_classes = set(gt_labels.tolist())
            pred_classes = set(pred_labels.tolist())
            unique_classes = gt_classes.union(pred_classes)
            
            for class_id in unique_classes:
                class_name = self.id2label_map[class_id]
                
                # Get ground truth and predictions for this class
                gt_mask = gt_labels == class_id
                pred_mask = pred_labels == class_id
                
                gt_class_boxes = gt_boxes[gt_mask]
                pred_class_boxes = np.array(pred_boxes)[pred_mask]
                pred_class_scores = np.array(pred_scores)[pred_mask] if len(pred_mask) > 0 else []
                
                n_gt = len(gt_class_boxes)
                n_pred = len(pred_class_boxes)
                
                if n_gt == 0 and n_pred == 0:
                    precision = 1.0
                    recall = 1.0
                    f1 = 1.0
                    tp, fp, fn = 0, 0, 0
                elif n_gt == 0:
                    precision = 0.0
                    recall = 1.0
                    f1 = 0.0
                    tp, fp, fn = 0, n_pred, 0
                elif n_pred == 0:
                    precision = 1.0
                    recall = 0.0
                    f1 = 0.0
                    tp, fp, fn = 0, 0, n_gt
                else:
                    # Calculate IoU matrix between all predictions and ground truths
                    ious = np.zeros((n_pred, n_gt))
                    for i in range(n_pred):
                        for j in range(n_gt):
                            ious[i, j] = self._compute_iou(
                                pred_class_boxes[i], 
                                gt_class_boxes[j]
                            )
                    
                    # Find matches using IoU threshold
                    matched_gt = set()
                    tp = 0  # True positives
                    fp = 0  # False positives
                    
                    # Sort predictions by confidence score (highest first)
                    pred_indices = np.argsort(pred_class_scores)[::-1]
                    
                    for pred_idx in pred_indices:
                        # Find best matching ground truth for this prediction
                        best_iou = 0
                        best_gt_idx = -1
                        
                        for gt_idx in range(n_gt):
                            if gt_idx not in matched_gt:  # Only consider unmatched ground truths
                                iou = ious[pred_idx, gt_idx]
                                if iou > iou_threshold and iou > best_iou:
                                    best_iou = iou
                                    best_gt_idx = gt_idx
                        
                        if best_gt_idx >= 0:
                            # Found a match
                            matched_gt.add(best_gt_idx)
                            tp += 1
                        else:
                            # No match found
                            fp += 1
                    
                    # Calculate false negatives (unmatched ground truths)
                    fn = n_gt - len(matched_gt)
                    
                    # Calculate metrics
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                records.append({
                    'image_name': image_name,
                    'class_name': class_name,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'true_positives': tp,
                    'false_positives': fp,
                    'false_negatives': fn,
                    'n_pred': n_pred,
                    'n_gt': n_gt,
                    'avg_confidence': np.mean(pred_class_scores) if len(pred_class_scores) > 0 else 0.0
                })
        
        # Convert to DataFrame
        df = pd.DataFrame.from_records(records)
        return df
    
    def _compute_iou(self, box1, box2):
        """
        Compute IoU between two boxes
        
        Args:
            box1: First bounding box [x1, y1, x2, y2]
            box2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU score
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _write_results(self, results, results_path: str):
        records = []
        for class_name in results['precision']:
            record = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_type': results['model_type'],
                'model_name': results['model_name'],
                'dataset': self.test_dataset_path,
                'dataset_size': results['dataset_size'],
                'class_name': class_name,
                'precision': results['precision'][class_name],
                'recall': results['recall'][class_name],
                'f1': results['f1'][class_name],
                'inference_time_ms': results['avg_time'],
                'magnification': results['model_metadata']['magnification'],
                # 'model_name': results['model_metadata'].get('model_name', ''),
                'predict_masks': str(results['model_metadata']['predict_masks'])
            }
            records.append(record)
        
        # Create DataFrame
        df = pd.DataFrame.from_records(records)
        df.to_csv(results_path, index=False)
        print(f"Results written to {results_path}")

    def save_results(self, output_dir: str, iou_threshold: float = 0.5):
        os.makedirs(output_dir, exist_ok=True)
        
        # Save overall results
        results_path = os.path.join(output_dir, f"{self.model_key}_results.csv")
        precision, recall, f1 = self.compute_metrics(
            preds=self.predictions,
            gt_dataset=self.test_set,
            iou_threshold=iou_threshold,
        )
        results_dict = {
            'model_type': self._metadata['model_type'],
            'model_name': self.model_key,
            'dataset_size': len(self.test_set),
            'avg_time': float(np.mean(self.runtimes)),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'model_metadata': self._metadata
        }
        self._write_results(results=results_dict, results_path=results_path)
        
        # Save per-image metrics
        per_image_df = self.compute_per_image_metrics(
            predictions=self.predictions,
            dataset=self.test_set,
            iou_threshold=iou_threshold
        )
        
        # Add model metadata to per-image results
        per_image_df['model_type'] = self._metadata['model_type']
        per_image_df['model_name'] = self.model_key
        per_image_df['magnification'] = self._metadata['magnification']
        per_image_df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        per_image_path = os.path.join(output_dir, f"{self.model_key}_per_image_results.csv")
        per_image_df.to_csv(per_image_path, index=False)
        print(f"Overall results saved to {results_path}")
        print(f"Per-image results saved to {per_image_path}")

sns.set_palette("colorblind")
def compare_model_performances(results_dir: str, output_dir: str = None) -> pd.DataFrame:
    """
    Compare and visualize performance metrics across all models.
    
    Args:
        results_dir: Directory containing all model summary CSV files (*_summary.csv)
        output_dir: Directory to save comparison plots. If None, uses results_dir
        
    Returns:
        DataFrame containing comparative statistics
    """
    if output_dir is None:
        output_dir = results_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all model summaries
    model_summaries = []
    for file in os.listdir(results_dir):
        if file.endswith('_summary.csv'):
            model_name = file.replace('_summary.csv', '')
            df = pd.read_csv(os.path.join(results_dir, file))
            df['model'] = model_name
            model_summaries.append(df)
    
    if not model_summaries:
        raise ValueError(f"No model summary files found in {results_dir}")
    
    combined_df = pd.concat(model_summaries, ignore_index=True)
    
    # 1. Overall Performance Comparison
    _plot_overall_model_comparison(combined_df, output_dir)
    
    # 2. Class-wise Performance Comparison
    _plot_class_wise_model_comparison(combined_df, output_dir)
    
    # 3. Model Performance Radar Plot
    _plot_model_radar_comparison(combined_df, output_dir)
    
    # 4. Statistical Significance Tests
    significance_df = _compute_statistical_significance(combined_df)
    significance_df.to_csv(os.path.join(output_dir, 'model_significance_tests.csv'))
    
    # 5. Generate comparison summary
    comparison_summary = _generate_comparison_summary(combined_df)
    comparison_summary.to_csv(os.path.join(output_dir, 'model_comparison_summary.csv'))
    
    return comparison_summary

def _plot_overall_model_comparison(df: pd.DataFrame, output_dir: str):
    """Plot overall performance comparison across models."""
    plt.figure(figsize=(12, 6))
    
    # Get unique models in a consistent order
    models = sorted(df['model'].unique())
    metrics = ['precision_mean', 'recall_mean', 'f1_mean']
    metric_labels = ['Precision', 'Recall', 'F1']
    
    x = np.arange(len(models))
    width = 0.25
    
    # Create colorblind-friendly palette
    colors = ['#0077BB', '#EE7733', '#009988']  # Blue, Orange, Teal - colorblind friendly
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        # Calculate means and stds for each model
        means = [df[df['model'] == model][metric].mean() for model in models]
        stds = [df[df['model'] == model][f'{metric.replace("_mean", "_std")}'].mean() for model in models]
        
        # Plot bars with consistent colors
        bars = plt.bar(x + i*width, means, width,
                      label=label,
                      yerr=stds,
                      capsize=5,
                      color=colors[i],
                      alpha=0.8)
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Overall Model Performance Comparison', fontsize=14, pad=20)
    
    # Center the xticks between the grouped bars
    plt.xticks(x + width, models, rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    # Add legend with better placement
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Customize grid
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.ylim(0, 1.0)  # Set y-axis from 0 to 1 for better comparison
    
    # Add value labels on top of bars with better formatting
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        means = [df[df['model'] == model][metric].mean() for model in models]
        for j, v in enumerate(means):
            # Determine text color based on bar height
            text_color = 'white' if v > 0.7 else 'black'
            plt.text(x[j] + i*width, v/2, 
                    f'{v:.3f}', 
                    ha='center', 
                    va='center',
                    fontsize=8,
                    color=text_color,
                    fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_model_comparison.png'))
    plt.close()

def _plot_class_wise_model_comparison(df: pd.DataFrame, output_dir: str):
    """Plot class-wise performance comparison across models using heatmaps."""
    metrics = ['precision_mean', 'recall_mean', 'f1_mean']
    
    # Get all unique classes and models in a consistent order
    all_classes = sorted(df['class_name'].unique())
    all_models = sorted(df['model'].unique())
    
    # Create figure with subplots for each metric
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Model Performance Metrics by Class', fontsize=16, y=1.05)
    
    # Create a colorblind-friendly colormap
    cmap = 'GnBu'  # Built-in colorblind-friendly colormap
    
    for i, metric in enumerate(metrics):
        # Create data matrix for heatmap
        data_matrix = np.zeros((len(all_models), len(all_classes)))
        
        for mi, model in enumerate(all_models):
            for ci, class_name in enumerate(all_classes):
                mask = (df['model'] == model) & (df['class_name'] == class_name)
                if mask.any():
                    data_matrix[mi, ci] = df.loc[mask, metric].iloc[0]
        
        # Plot heatmap
        ax = axes[i]
        im = ax.imshow(data_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        
        # Configure axes
        ax.set_xticks(np.arange(len(all_classes)))
        ax.set_yticks(np.arange(len(all_models)))
        ax.set_xticklabels(all_classes, rotation=45, ha='right')
        ax.set_yticklabels(all_models)
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        # Add value annotations
        for mi in range(len(all_models)):
            for ci in range(len(all_classes)):
                value = data_matrix[mi, ci]
                text_color = 'white' if value < 0.5 else 'black'
                ax.text(ci, mi, f'{value:.3f}', 
                       ha='center', va='center', 
                       color=text_color,
                       fontsize=8)
        
        ax.set_title(f'{metric.replace("_mean", "").capitalize()}')
        ax.set_xlabel('Class')
        if i == 0:
            ax.set_ylabel('Model')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_wise_heatmaps.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Define x and width for proper bar placement
    x = np.arange(len(all_classes))
    width = 0.8 / len(all_models)  # Adjust bar width based on number of models
    
    # Plot bars for each model
    bars = []
    for i, model in enumerate(all_models):
        model_data = df[df['model'] == model]
        model_means = []
        for class_name in all_classes:
            class_data = model_data[model_data['class_name'] == class_name]
            if len(class_data) > 0:
                model_means.append(class_data['f1_mean'].iloc[0])
            else:
                model_means.append(0)
        
        bar = plt.bar(x + i*width, model_means, width, label=model)
        bars.append(bar)
    
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.title('Model Performance Comparison by Class')
    plt.xticks(x + width * (len(all_models)-1)/2, 
               all_classes, 
               rotation=45, 
               ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 1.0)  # Set y-axis from 0 to 1 for better comparison
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_wise_comparison_bars.png'))
    plt.close()
    

    
    # 3. Line Plot with Error Bands - Shows trends across classes
    plt.figure(figsize=(15, 8))
    
    for model in all_models:
        # Create arrays with proper alignment to classes
        f1_means = []
        f1_stds = []
        for class_name in all_classes:
            class_data = df[(df['model'] == model) & (df['class_name'] == class_name)]
            if len(class_data) > 0:
                f1_means.append(class_data['f1_mean'].iloc[0])
                f1_stds.append(class_data['f1_std'].iloc[0])
            else:
                f1_means.append(0)
                f1_stds.append(0)
        
        plt.plot(all_classes, 
                f1_means, 
                'o-', 
                label=model, 
                linewidth=2, 
                markersize=8)
        
        plt.fill_between(all_classes,
                        np.array(f1_means) - np.array(f1_stds),
                        np.array(f1_means) + np.array(f1_stds),
                        alpha=0.2)
    
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.title('Model Performance Trends Across Classes')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_wise_comparison_trends.png'))
    plt.close()

def _plot_model_radar_comparison(df: pd.DataFrame, output_dir: str):
    """Create radar plot comparing models across different metrics."""
    metrics = ['precision_mean', 'recall_mean', 'f1_mean', 'inference_time_ms_mean']
    metric_labels = ['Precision', 'Recall', 'F1', 'Speed']  # More readable labels
    
    # Prepare data
    model_stats = df.groupby('model')[metrics].mean()
    
    # Keep original metrics for precision, recall, f1 (they're already 0-1)
    metrics_data = model_stats.copy()
    
    # Normalize inference time to 0-1 range (slower time = higher value)
    max_time = metrics_data['inference_time_ms_mean'].max()
    min_time = metrics_data['inference_time_ms_mean'].min()
    if max_time > min_time:
        # Normalize to 0-1 range where slowest (highest time) = 1
        metrics_data['inference_time_ms_mean'] = (metrics_data['inference_time_ms_mean'] - min_time) / (max_time - min_time)
    
    # Set up the radar plot
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Use a carefully selected colorblind-friendly palette
    colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02']  # ColorBrewer Set2 palette
    # Ensure we have enough colors for all models
    colors = colors[:len(metrics_data.index)]
    
    for idx, model in enumerate(sorted(metrics_data.index)):
        values = metrics_data.loc[model].values
        values = np.concatenate((values, [values[0]]))  # Close the polygon
        
        # Plot with GnBu colors and better line styling
        ax.plot(angles, values, 'o-', linewidth=2, 
               label=model, color=colors[idx], 
               markersize=8)
        ax.fill(angles, values, alpha=0.2, color=colors[idx])
        
        # No value labels
    
    # Customize the plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=12)
    
    # Add gridlines and adjust their style
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Customize legend
    plt.legend(loc='center left', bbox_to_anchor=(1.2, 0.5),
              title='Models', title_fontsize=12, fontsize=10)
    
    plt.title('Model Performance Comparison\nAcross Metrics', 
             y=1.05, fontsize=14, pad=20)
    
    # Add metric descriptions
    # desc_text = ("Metrics:\n"
    #             "Precision: Accuracy of positive predictions\n"
    #             "Recall: Coverage of actual positives\n"
    #             "F1: Harmonic mean of precision & recall\n"
    #             "Speed: Normalized inference speed")
    # plt.figtext(1.3, 0.02, desc_text, fontsize=8, 
    #             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_radar_comparison.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def _compute_statistical_significance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute statistical significance of performance differences between models."""
    from scipy import stats
    
    models = df['model'].unique()
    metrics = ['precision_mean', 'recall_mean', 'f1_mean']
    results = []
    
    for m1 in models:
        for m2 in models:
            if m1 >= m2:
                continue
                
            for metric in metrics:
                m1_scores = df[df['model'] == m1][metric]
                m2_scores = df[df['model'] == m2][metric]
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(m1_scores, m2_scores)
                
                results.append({
                    'model_1': m1,
                    'model_2': m2,
                    'metric': metric.replace('_mean', ''),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
    
    return pd.DataFrame(results)

def _generate_comparison_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a comprehensive summary of model comparisons."""
    metrics = ['precision_mean', 'recall_mean', 'f1_mean', 'inference_time_ms_mean']
    
    summary_data = []
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        
        row = {'model': model}
        # Overall metrics
        for metric in metrics:
            row[f'avg_{metric}'] = model_data[metric].mean()
            row[f'std_{metric}'] = model_data[metric].std()
        
        # Best performing classes
        for metric in metrics[:-1]:  # exclude inference time
            best_class = model_data.loc[model_data[metric].idxmax()]
            row[f'best_class_{metric}'] = best_class['class_name']
            row[f'best_score_{metric}'] = best_class[metric]
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df

if __name__ == "__main__":
    seed_everything(42)
    # model_key = 'rf-detr'  # Change this to test different models: 'mask2former', 'mask-rcnn', 'deformable-detr', 'rf-detr', 'rt-detr-v1', 'rt-detr-v2'
    # detector = Benchmark(model_key=model_key)
    for model_key in MODELS.keys():
        detector = Benchmark(model_key=model_key)
        for dataset_path in TEST_DATASETS:
            output_dir = os.path.join('benchmark_results', os.path.basename(dataset_path))
            print (f"Creating dataset from {dataset_path}...")
            train_set, test_set = detector.create_dataset(dataset_path=dataset_path)
            print (f"Dataset created with {len(train_set)} training images and {len(test_set)} testing images.")
            
            print (f"Running inference on test set of {dataset_path}...")
            predictions, runtimes = detector.infer(plot_detections=True, save_path=f'{output_dir}/{model_key}')
            print (f"Inference completed on {len(test_set)} images from {dataset_path}.\n")
            # precision, recall, f1 = detector.compute_metrics(
            #     preds=predictions,
            #     gt_dataset=test_set,
            #     iou_threshold=0.5
            # )
            detector.save_results(output_dir=f'{output_dir}/{model_key}', iou_threshold=0.5)

        # aggregate results and generate summary
        summary_df = detector.get_model_summary(results_dir='/global/home/ashish.sinha/cellanome/dl-mehdi/benchmark_results', output_dir='/global/home/ashish.sinha/cellanome/dl-mehdi/benchmark_results/summary')