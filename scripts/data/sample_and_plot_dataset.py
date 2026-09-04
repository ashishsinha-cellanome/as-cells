#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pycocotools.coco import COCO
import hydra
from omegaconf import DictConfig
from pathlib import Path


def sample_images_for_class(coco, category_id, num_samples):
    """Samples N image IDs that contain at least one instance of the given category."""
    ann_ids = coco.getAnnIds(catIds=[category_id])
    image_ids = list(set([coco.loadAnns(aid)[0]["image_id"] for aid in ann_ids]))

    if not image_ids:
        return []

    return random.sample(image_ids, min(len(image_ids), num_samples))


def plot_samples(image_paths, titles, output_path, bboxes_list=None, grid_size=(2, 4)):
    """Plots a grid of images and saves it."""
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    for i, (img_path, title) in enumerate(zip(image_paths, titles)):
        if i >= len(axes):
            break
        try:
            img = Image.open(img_path).convert("RGB")
            axes[i].imshow(img)

            if bboxes_list and i < len(bboxes_list):
                for bbox in bboxes_list[i]:
                    # COCO bbox: [x, y, w, h]
                    rect = patches.Rectangle(
                        (bbox[0], bbox[1]),
                        bbox[2],
                        bbox[3],
                        linewidth=2,
                        edgecolor="r",
                        facecolor="none",
                    )
                    axes[i].add_patch(rect)

            axes[i].set_title(title, fontsize=10)
            axes[i].axis("off")
        except Exception as e:
            axes[i].text(
                0.5,
                0.5,
                f"Error loading\n{os.path.basename(img_path)}",
                ha="center",
                va="center",
            )
            axes[i].axis("off")
            print(f"Error loading {img_path}: {e}")

    # Hide unused axes
    for i in range(len(image_paths), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")


@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
    # Number of samples per class (configurable via Hydra, e.g., sampling.n=10)
    N = cfg.get("sampling", {}).get("n", 8)
    num_rows = cfg.get("sampling", {}).get("rows", 2)
    num_cols = cfg.get("sampling", {}).get("cols", 4)
    grid_size = (num_rows, num_cols)

    output_dir = Path("dataset_samples")
    output_dir.mkdir(exist_ok=True)

    # Dataset path from config
    data_root = Path(cfg.data.path)
    # The user mentioned image subfolders are in data_root/images/{split}
    image_base_dir = data_root / "images"

    # Splits to process
    splits = {"train": cfg.train_name, "val": cfg.val_name, "test": cfg.test_name}

    # Label map from config
    label_map = {int(k): v for k, v in cfg.model.label_map.items()}

    # Local annotation files in ./dataset/
    # breakpoint()
    # local_anno_dir = Path("./dataset")
    local_anno_dir = Path(cfg.data.path)

    for split_key, split_name in splits.items():
        print(f"\nProcessing {split_key} split ('{split_name}')...")

        # Determine annotation file name
        # Based on list_dir, they are train_annotations.json, valid_annotations.json, test_annotations.json
        anno_file = local_anno_dir / f"{split_name}_annotations.json"

        if not anno_file.exists():
            print(f"Annotation file not found: {anno_file}. Skipping.")
            continue

        coco = COCO(str(anno_file))

        # Process each class
        for cat_id, cat_name in label_map.items():
            # Find category ID in COCO (might be different from label_map if names don't match exactly)
            cat_ids = coco.getCatIds(catNms=[cat_name])
            if not cat_ids:
                print(
                    f"  Category '{cat_name}' (id {cat_id}) not found in COCO annotations for {split_key}. Skipping."
                )
                continue

            coco_cat_id = cat_ids[0]
            sampled_ids = sample_images_for_class(coco, coco_cat_id, N)

            if not sampled_ids:
                print(f"  No images found for category '{cat_name}' in {split_key}.")
                continue

            print(f"  Sampling {len(sampled_ids)} images for '{cat_name}'...")

            image_paths = []
            titles = []
            bboxes_list = []

            for img_id in sampled_ids:
                img_info = coco.loadImgs(img_id)[0]
                file_name = img_info["file_name"]
                # Path construction based on user feedback
                img_path = image_base_dir / split_name / file_name
                image_paths.append(str(img_path))
                titles.append(f"{cat_name} | {file_name}")

                # Fetch annotations for this image and class
                ann_ids = coco.getAnnIds(
                    imgIds=img_id, catIds=[coco_cat_id], iscrowd=None
                )
                anns = coco.loadAnns(ann_ids)
                bboxes = [ann["bbox"] for ann in anns]
                bboxes_list.append(bboxes)

            # Create output plot
            plot_name = output_dir / f"{split_key}_{cat_name}_samples.png"
            plot_samples(
                image_paths,
                titles,
                str(plot_name),
                bboxes_list=bboxes_list,
                grid_size=grid_size,
            )


if __name__ == "__main__":
    main()
