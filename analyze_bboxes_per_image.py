#!/usr/bin/env python3
import os
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image
from pycocotools.coco import COCO
import hydra
from omegaconf import DictConfig
from pathlib import Path


def plot_debug_grid(debug_samples, output_path, classes, splits):
    """
    Plots a debug grid where rows are splits and columns are class labels.
    debug_samples is a dict mapping (split, class_name) -> dict with keys:
    'img_path', 'bboxes', 'count'
    """
    rows = len(splits)
    cols = len(classes)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))

    # Ensure axes is 2D
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for row_idx, split_key in enumerate(splits):
        for col_idx, class_name in enumerate(classes):
            ax = axes[row_idx, col_idx]
            sample = debug_samples.get((split_key, class_name))

            if sample:
                img_path = sample["img_path"]
                bboxes = sample["bboxes"]
                count = sample["count"]

                try:
                    img = Image.open(img_path).convert("RGB")
                    ax.imshow(img)

                    for bbox in bboxes:
                        # COCO bbox: [x, y, w, h]
                        rect = patches.Rectangle(
                            (bbox[0], bbox[1]),
                            bbox[2],
                            bbox[3],
                            linewidth=2,
                            edgecolor="r",
                            facecolor="none",
                        )
                        ax.add_patch(rect)

                    title = f"{split_key.upper()} - {class_name}\nCount: {count}"
                    ax.set_title(title, fontsize=12)
                except Exception as e:
                    ax.text(
                        0.5,
                        0.5,
                        f"Error loading\n{os.path.basename(img_path)}",
                        ha="center",
                        va="center",
                    )
                    print(f"Error loading {img_path}: {e}")
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"No data for\n{split_key} - {class_name}",
                    ha="center",
                    va="center",
                )

            ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved debug grid plot to {output_path}")


def plot_statistics(df_stats, output_dir):
    """
    Plots mean and median bounding boxes per image per class across splits.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Plot Mean Bounding Boxes
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        x="Class", y="Mean", hue="Split", data=df_stats, palette="colorblind"
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=12)

    plt.title("Mean Bounding Boxes per Image", fontsize=16)
    plt.xlabel("Class Label", fontsize=14)
    plt.ylabel("Mean Count", fontsize=14)
    plt.legend(title="Split", title_fontsize=12, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mean_bboxes_per_image.png"), dpi=300)
    plt.close()

    # 2. Plot Median Bounding Boxes
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        x="Class", y="Median", hue="Split", data=df_stats, palette="colorblind"
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=3, fontsize=12)

    plt.title("Median Bounding Boxes per Image", fontsize=16)
    plt.xlabel("Class Label", fontsize=14)
    plt.ylabel("Median Count", fontsize=14)
    plt.legend(title="Split", title_fontsize=12, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "median_bboxes_per_image.png"), dpi=300)
    plt.close()

    print(f"Saved statistics plots to {output_dir}")


def print_table(df_stats):
    print("\n--- Bounding Box Statistics (Where Class is Present) ---")
    headers = [
        "Split",
        "Class",
        "Mean",
        "Median",
        "Max",
        "Images With BBoxes",
        "Total Images",
    ]

    widths = [len(h) for h in headers]
    for _, row in df_stats.iterrows():
        row_vals = [
            row["Split"],
            row["Class"],
            f"{row['Mean']:.2f}",
            f"{row['Median']:.1f}",
            str(row["Max"]),
            str(row["ImagesWithBBoxes"]),
            str(row["TotalImages"]),
        ]
        for i, val in enumerate(row_vals):
            widths[i] = max(widths[i], len(val))

    header_str = " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    print("-" * len(header_str))
    print(header_str)
    print("-" * len(header_str))

    for _, row in df_stats.iterrows():
        row_vals = [
            row["Split"].capitalize(),
            row["Class"],
            f"{row['Mean']:.2f}",
            f"{row['Median']:.1f}",
            str(int(row["Max"])),
            str(int(row["ImagesWithBBoxes"])),
            str(int(row["TotalImages"])),
        ]
        print(" | ".join(f"{str(v):<{w}}" for v, w in zip(row_vals, widths)))
    print("-" * len(header_str))
    print()


@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
    output_dir = Path("analysis_plots")
    output_dir.mkdir(exist_ok=True)

    data_root = Path(cfg.data.path)
    image_base_dir = data_root / "images"

    splits = {
        "train": getattr(cfg, "train_name", "train"),
        "val": getattr(cfg, "val_name", "valid"),
        "test": getattr(cfg, "test_name", "test"),
    }

    # We will fallback to COCO categories if label_map is not present, but let's try to get it
    if hasattr(cfg, "model") and hasattr(cfg.model, "label_map"):
        label_map = {int(k): v for k, v in cfg.model.label_map.items()}
    else:
        label_map = None

    all_stats = []
    debug_samples = {}
    max_debug_samples = {}

    split_keys_processed = []
    all_classes_found = set()

    is_debug = cfg.get("debug", False)

    for split_key, split_name in splits.items():
        print(f"\nProcessing {split_key} split ('{split_name}')...")

        anno_file = data_root / f"{split_name}_annotations.json"

        if not anno_file.exists():
            print(f"Annotation file not found: {anno_file}. Skipping.")
            continue

        import sys

        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            coco = COCO(str(anno_file))
        finally:
            sys.stdout = original_stdout

        split_keys_processed.append(split_key)

        # Get categories
        cat_ids = coco.getCatIds()
        cats = coco.loadCats(cat_ids)
        cat_name_to_id = {c["name"]: c["id"] for c in cats}

        for c in cats:
            all_classes_found.add(c["name"])

        img_ids = coco.getImgIds()
        total_images = len(img_ids)

        if total_images == 0:
            print(f"No images found in {split_key}.")
            continue

        print(f"Total images in {split_key}: {total_images}")

        for cat_name, coco_cat_id in cat_name_to_id.items():
            # For each image, count how many annotations of this category it has
            counts_per_image = []

            # To pick a debug sample, we want an image that has at least 1 bbox for this class
            candidate_images = []

            for img_id in img_ids:
                ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[coco_cat_id])
                count = len(ann_ids)
                counts_per_image.append(count)

                if count > 0:
                    candidate_images.append((img_id, ann_ids, count))

            counts_per_image = np.array(counts_per_image)
            images_with_bboxes = int(np.sum(counts_per_image > 0))

            # Filter out images with 0 annotations for accurate mean/median per class
            counts_present = counts_per_image[counts_per_image > 0]

            if len(counts_present) > 0:
                mean_count = float(np.mean(counts_present))
                median_count = float(np.median(counts_present))
                max_count = int(np.max(counts_present))
            else:
                mean_count = 0.0
                median_count = 0.0
                max_count = 0

            all_stats.append(
                {
                    "Split": split_key,
                    "Class": cat_name,
                    "Mean": mean_count,
                    "Median": median_count,
                    "Max": max_count,
                    "TotalImages": total_images,
                    "ImagesWithBBoxes": images_with_bboxes,
                }
            )

            # Pick a debug sample
            if is_debug and candidate_images:
                # Randomly select one image that has annotations for this class
                sample_img_id, sample_ann_ids, sample_count = random.choice(
                    candidate_images
                )

                img_info = coco.loadImgs(sample_img_id)[0]
                file_name = img_info["file_name"]

                img_path = image_base_dir / split_name / file_name
                if not img_path.exists():
                    fallback_path = image_base_dir / file_name
                    if fallback_path.exists():
                        img_path = fallback_path

                anns = coco.loadAnns(sample_ann_ids)
                bboxes = [ann["bbox"] for ann in anns]

                debug_samples[(split_key, cat_name)] = {
                    "img_path": str(img_path),
                    "bboxes": bboxes,
                    "count": sample_count,
                }

                # Select the image with the MAXIMUM annotations for this class
                max_img_tuple = max(candidate_images, key=lambda x: x[2])
                max_img_id, max_ann_ids, max_count = max_img_tuple

                max_img_info = coco.loadImgs(max_img_id)[0]
                max_file_name = max_img_info["file_name"]

                max_img_path = image_base_dir / split_name / max_file_name
                if not max_img_path.exists():
                    fallback_max_path = image_base_dir / max_file_name
                    if fallback_max_path.exists():
                        max_img_path = fallback_max_path

                max_anns = coco.loadAnns(max_ann_ids)
                max_bboxes = [ann["bbox"] for ann in max_anns]

                max_debug_samples[(split_key, cat_name)] = {
                    "img_path": str(max_img_path),
                    "bboxes": max_bboxes,
                    "count": max_count,
                }

    if not all_stats:
        print("No statistics gathered. Exiting.")
        return

    df_stats = pd.DataFrame(all_stats)

    # Sort classes for consistency
    classes_sorted = sorted(list(all_classes_found))

    # 1. Print Table
    print_table(df_stats)

    # 2. Plot Bar Charts
    plot_statistics(df_stats, str(output_dir))

    # 3. Plot Debug Grid
    if is_debug:
        debug_grid_path = output_dir / "debug_bbox_samples.png"
        plot_debug_grid(
            debug_samples, str(debug_grid_path), classes_sorted, split_keys_processed
        )

        # 4. Plot Max BBox Grid
        debug_max_grid_path = output_dir / "debug_max_bbox_samples.png"
        plot_debug_grid(
            max_debug_samples,
            str(debug_max_grid_path),
            classes_sorted,
            split_keys_processed,
        )


if __name__ == "__main__":
    main()
