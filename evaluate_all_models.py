"""
Run inference on a dataset and evaluate precision and recall.
"""

import sys
import pandas as pd
import cv2
import warnings
from tqdm import tqdm
from pathlib import Path
from utils.precision_recall_eval import AnnotationFilter, evaluate_pr_per_image

from models.rf_detr_model import (
    DEFAULT_LABEL_MAP as RF_DETR_DEFAULT_LABEL_MAP,
    DEFAULT_MODEL_INPUT_SIZE as RF_DETR_DEFAULT_INPUT_SIZE,
)

from config import (
    SAVE_PATH,
    MODELS,
    TEST_DATASETS,
    CLASS_IDS_TO_CLASS_NAMES_MAP,
    CLASS_NAMES_TO_CLASS_IDS_MAP,
    MASK_RCNN_CLASS_NAMES_TO_CLASS_IDS_MAP,
    MASK_RCNN_CLASS_IDS_TO_CLASS_NAMES_MAP,
    OTHER_CLASS_IDS_TO_CLASS_NAMES_MAP,
    OTHER_CLASS_NAMES_TO_CLASS_IDS_MAP,
)
from utils.benchmark_utils import (
    get_test_set,
    infer,
    show_detections_updated,
    visualize_model_errors_with_official_pairing,
)


def main(SAVE_PATH):
    """
    Main function to run the evaluation pipeline.
    """
    # Ensure base save directory exists
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    # Flag to control visualization
    plot_preds = True

    # List to store paths to all generated metric files
    all_metric_files = []

    # Iterate over each dataset path
    for TEST_DATASET in TEST_DATASETS:
        dataset_name = Path(TEST_DATASET).name
        print(f"[INFO] Processing Dataset: {dataset_name} ")

        # Iterate over each model
        for model_key in MODELS:
            model_info = MODELS.get(model_key)
            if model_key not in ["rf-detr", "yolo"]:
                continue
            model_name = model_info["model_name"]
            print(f"\n[INFO] Evaluating Model: {model_name} on {dataset_name} ")

            # --- 1. Initialize Model ---
            try:
                model_class = getattr(sys.modules[__name__], model_info["model_class"])

                if model_key == "rf-detr":
                    model = model_class(
                        weights_path=model_info["weights_path"],
                        label_map=RF_DETR_DEFAULT_LABEL_MAP,
                        model_input_size=RF_DETR_DEFAULT_INPUT_SIZE,
                    )
                elif model_key in ["deformable-detr", "rt-detr-v2"]:
                    model = model_class(
                        weights_path=model_info["weights_path"],
                        backbone_name_str=model_info["backbone_name_str"],
                    )
                else:
                    model = model_class(
                        weights_path=model_info["weights_path"],
                    )
            except Exception as e:
                print(f"[ERROR] Failed to initialize model {model_name}: {e}")
                continue

            print(
                f"Class ID to class name mapping for {model_key}: {model.get_label_map()}"
            )

            # --- 2. Set Class Mappings and Filters ---
            if model_key == "mask-rcnn":
                CLASS_MAPPING = MASK_RCNN_CLASS_IDS_TO_CLASS_NAMES_MAP.copy()
                CLASS_NAME2ID_MAPPING = MASK_RCNN_CLASS_NAMES_TO_CLASS_IDS_MAP.copy()
            elif model_key in ["yolo", "rf-detr", "rt-detr-v1", "rt-detr-v2"]:
                CLASS_MAPPING = OTHER_CLASS_IDS_TO_CLASS_NAMES_MAP.copy()
                CLASS_NAME2ID_MAPPING = OTHER_CLASS_NAMES_TO_CLASS_IDS_MAP.copy()
            else:
                CLASS_MAPPING = CLASS_IDS_TO_CLASS_NAMES_MAP.copy()
                CLASS_NAME2ID_MAPPING = CLASS_NAMES_TO_CLASS_IDS_MAP.copy()

            for k, v in model.get_label_map().items():
                if k not in CLASS_MAPPING or v != CLASS_MAPPING[k]:
                    print(
                        f"[WARN] (class_id, class_name) pair ({k}, {v}) is defined in the {model_key} label map but not covered in CLASS_IDS_TO_CLASS_NAMES_MAP"
                    )

            # Define annotation filter based on model's detected classes
            class_ids_of_interest = list(model.get_label_map().keys())

            if (
                "cell-adhered" in model.get_class_names()
                or "soma" in model.get_class_names()
            ):
                annotation_filter = None
                print(f"---> ONLY DETECT cell in {model.get_class_names()}")
            else:
                annotation_filter = AnnotationFilter(
                    classnames_mapping_dict={"soma": "cell", "cell-adhered": "cell"},
                    class_ids_to_class_names_map=CLASS_MAPPING,
                )

            print(f"CLASS IDS of interest: {class_ids_of_interest}")

            # --- 3. Load Dataset and Run Inference ---
            try:
                test_dataset = get_test_set([TEST_DATASET], CLASS_NAME2ID_MAPPING)
                predictions, runtimes = infer(model, test_dataset)
            except Exception as e:
                print(f"[ERROR] Failed during dataset loading or inference: {e}")
                continue

            # --- 4. Process Images: Log Metrics and Visuals ---

            # Prepare directory for this model/dataset
            output_dir = SAVE_PATH / dataset_name / model_key
            output_dir.mkdir(parents=True, exist_ok=True)

            # Prepare for metric logging
            image_metrics_data = []
            metrics_csv_path = output_dir / "image_metrics.csv"

            use_mask = model.get_metadata()["predict_masks"]

            print("\n[INFO] Computing metrics")
            eval_results = evaluate_pr_per_image(
                predictions=predictions,
                dataset=test_dataset,
                class_ids_of_interest=class_ids_of_interest,
                min_iou=0.5,
                use_mask=use_mask,
                annotation_filter=annotation_filter,
            )
            print(
                "\n[INFO] Processing results for logging metrics and visualization..."
            )
            # for idx, prediction in enumerate(tqdm(predictions, desc=f"Processing {model_name}/{dataset_name}")):
            for idx, img_metrics in enumerate(
                tqdm(eval_results[-1], desc=f"Logging {model_name}/{dataset_name}")
            ):
                image_name = img_metrics["name"]
                base_img_name = Path(image_name).stem
                datasample = test_dataset[idx]
                prediction = predictions[idx]
                try:
                    # Store metrics for each class
                    for class_id in class_ids_of_interest:
                        class_label = CLASS_MAPPING.get(class_id, f"ID_{class_id}")
                        # class_name = model.get_label_map().get(class_id, f"ID_{class_id}")
                        if class_label == "bg":
                            continue  # Don't log 'bg' stats
                        row = {
                            "model": model_key,
                            "dataset": dataset_name,
                            "image": image_name,
                            "class_id": class_id,
                            "class_name": class_label,
                            "tp": img_metrics["tp"][class_id],
                            "fp": img_metrics["fp"][class_id],
                            "fn": img_metrics["fn"][class_id],
                            "precision": img_metrics["precision"][class_id],
                            "recall": img_metrics["recall"][class_id],
                            "f1_score": img_metrics["f1"][class_id],
                            "inference_time_ms": runtimes[idx] * 1000,
                        }
                        image_metrics_data.append(row)

                    # --- 4b. Save Visualizations (if enabled) ---
                    if plot_preds and (idx % 10 == 0):  # Only plot every 5th image
                        gt_annots = datasample["annotations"]
                        gt_dets = {
                            "boxes": gt_annots[gt_annots.columns[:-1]].values,
                            "labels": gt_annots[gt_annots.columns[-1]].values,
                        }

                        # Plot preds with GT
                        pred_img1 = show_detections_updated(
                            datasample["image"],
                            gt_dets,
                            pred=False,
                            label_map=CLASS_MAPPING,
                        )  # model.get_label_map()
                        pred_img2 = show_detections_updated(
                            pred_img1, prediction, pred=True, label_map=CLASS_MAPPING
                        )
                        (output_dir / "preds").mkdir(parents=True, exist_ok=True)
                        (output_dir / "errors").mkdir(parents=True, exist_ok=True)
                        save_name_preds = output_dir / f"preds/{base_img_name}.png"
                        cv2.imwrite(str(save_name_preds), pred_img2)

                        # Plot errors with remapping
                        err_img1 = visualize_model_errors_with_official_pairing(
                            datasample["image"],
                            datasample,
                            prediction,
                            label_map=CLASS_MAPPING,
                            annotation_filter=annotation_filter,
                            show_original_labels=False,
                            min_iou=0.5,
                            use_mask=use_mask,
                        )
                        save_name_err1 = (
                            output_dir / f"errors/{base_img_name}_remap.png"
                        )
                        cv2.imwrite(str(save_name_err1), err_img1)

                        # Plot errors with original GT
                        err_img2 = visualize_model_errors_with_official_pairing(
                            datasample["image"],
                            datasample,
                            prediction,
                            label_map=CLASS_MAPPING,
                            annotation_filter=None,  # No filter
                            show_original_labels=False,
                            min_iou=0.5,
                            use_mask=use_mask,
                        )
                        save_name_err2 = output_dir / f"errors/{base_img_name}_OG.png"
                        cv2.imwrite(str(save_name_err2), err_img2)

                except Exception as e:
                    print(
                        f"[ERROR] Failed processing image {idx} ({datasample.get('name', 'N/A')}): {e}"
                    )

            # print aggregated results on the current dataset
            for class_id in tqdm(class_ids_of_interest):
                class_name = CLASS_MAPPING[class_id]
                precision_per_class = eval_results[3]
                recall_per_class = eval_results[4]
                f1_per_class = eval_results[5]
                print(
                    f"[INFO] {model_key.upper()} Precision: {precision_per_class[class_id]:.5f} | Recall: {recall_per_class[class_id]:.5f} | F1: {f1_per_class[class_id]:.5f} at IoU 0.5 for {class_name}"
                )
            # --- 5. Save Per-Model CSV ---
            if image_metrics_data:
                try:
                    metrics_df = pd.DataFrame(image_metrics_data)
                    metrics_df.to_csv(metrics_csv_path, index=False)
                    print(f"[INFO] Successfully saved metrics to {metrics_csv_path}")
                    all_metric_files.append(metrics_csv_path)
                except Exception as e:
                    print(f"[ERROR] Failed to save metrics CSV: {e}")
            else:
                print(
                    f"[ERROR] \nNo metric data generated for {model_name} on {dataset_name}."
                )

            #
    # --- 6. Final CSV Compilation ---
    print("\n [INFO] All processing complete. Combining all metric CSVs")

    all_dfs = []
    for csv_file in all_metric_files:
        if Path(csv_file).exists():
            try:
                df = pd.read_csv(csv_file)
                all_dfs.append(df)
            except pd.errors.EmptyDataError:
                print(f"[WARN] {csv_file} is empty, skipping.")
            except Exception as e:
                print(f"[ERROR] Failed to read {csv_file}: {e}")
        else:
            print(f"[WARN] {csv_file} not found, skipping.")

    if all_dfs:
        try:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            final_csv_path = SAVE_PATH / "all_models_all_datasets_metrics.csv"
            combined_df.to_csv(final_csv_path, index=False)
            print(
                f"\n✅ Successfully combined {len(all_dfs)} CSVs into {final_csv_path}"
            )
        except Exception as e:
            print(f"[ERROR] Failed to combine and save final CSV: {e}")
    else:
        print("\n[ERROR] No metric files were found to combine.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main(Path(SAVE_PATH))
