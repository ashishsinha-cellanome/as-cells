#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import contextlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig
from pycocotools.coco import COCO

THRESHOLDS = (200, 300)


def load_coco_quiet(anno_file: Path) -> COCO:
    """Load a COCO annotation file while suppressing pycocotools console noise."""
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            return COCO(str(anno_file))


def resolve_image_path(image_base_dir: Path, split_name: str, file_name: str) -> Path:
    """Resolve image path with split-first lookup and a flat-folder fallback."""
    preferred = image_base_dir / split_name / file_name
    if preferred.exists():
        return preferred

    fallback = image_base_dir / file_name
    if fallback.exists():
        return fallback

    return preferred


def collect_counts_for_split(
    coco: COCO, split_name: str, image_base_dir: Path, thresholds: tuple[int, ...]
) -> dict[str, Any]:
    """Collect rows of images with bbox counts above each threshold for one split."""
    rows_by_threshold: dict[int, list[dict[str, Any]]] = {t: [] for t in thresholds}

    image_items = sorted(coco.imgs.items(), key=lambda item: int(item[0]))
    total_images = len(image_items)

    for image_id, image_info in image_items:
        file_name = str(image_info.get("file_name", ""))
        bbox_count = len(coco.imgToAnns.get(image_id, []))
        image_path = resolve_image_path(image_base_dir, split_name, file_name)

        row = {
            "image_id": int(image_id),
            "file_name": file_name,
            "bbox_count": int(bbox_count),
            "image_path": str(image_path),
        }

        for threshold in thresholds:
            if bbox_count > threshold:
                rows_by_threshold[threshold].append(row)

    for threshold, rows in rows_by_threshold.items():
        rows.sort(
            key=lambda r: (
                -int(r["bbox_count"]),
                str(r["file_name"]),
                int(r["image_id"]),
            )
        )
        rows_by_threshold[threshold] = rows

    return {
        "total_images": total_images,
        "rows_by_threshold": rows_by_threshold,
    }


def _md_escape(value: str) -> str:
    return value.replace("|", r"\|")


def render_markdown(report_data: dict[str, Any], meta: dict[str, Any]) -> str:
    """Render the final report as markdown."""
    lines: list[str] = []
    lines.append("# High-BBox Images Report")
    lines.append("")
    lines.append(f"- Generated At: `{meta['generated_at']}`")
    lines.append(f"- Dataset Path: `{meta['dataset_path']}`")
    lines.append(f"- Thresholds: `{', '.join(f'>{t}' for t in meta['thresholds'])}`")
    lines.append("- Split Mapping:")
    for split_key, split_name in meta["splits"].items():
        lines.append(f"  - `{split_key}` -> `{split_name}`")
    lines.append("")

    for split_key, split_name in meta["splits"].items():
        split_report = report_data.get(split_key, {})
        lines.append(f"## Split: {split_key} (name: {split_name})")
        lines.append("")

        if split_report.get("missing_annotation"):
            lines.append(
                f"> Warning: Annotation file not found: `{split_report['annotation_path']}`"
            )
            lines.append("")
            continue

        total_images = int(split_report["total_images"])
        for threshold in meta["thresholds"]:
            rows = split_report["rows_by_threshold"][threshold]
            lines.append(f"### Images with >{threshold} bboxes")
            lines.append("")
            lines.append(f"{len(rows)} matches out of {total_images} images.")
            lines.append("")

            if not rows:
                lines.append("No images matched this threshold.")
                lines.append("")
                continue

            lines.append("| Rank | Image ID | File Name | BBox Count | Image Path |")
            lines.append("| ---: | ---: | --- | ---: | --- |")
            for rank, row in enumerate(rows, start=1):
                lines.append(
                    "| "
                    f"{rank} | "
                    f"{int(row['image_id'])} | "
                    f"{_md_escape(str(row['file_name']))} | "
                    f"{int(row['bbox_count'])} | "
                    f"{_md_escape(str(row['image_path']))} |"
                )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = Path("analysis_plots")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "high_bbox_images_report.md"

    data_root = Path(cfg.data.path).expanduser()
    image_base_dir = data_root / "images"
    splits = {
        "train": getattr(cfg, "train_name", "train"),
        "val": getattr(cfg, "val_name", "valid"),
        "test": getattr(cfg, "test_name", "test"),
    }

    report_data: dict[str, Any] = {}

    print(f"Dataset root: {data_root}")
    print(f"Output report: {output_path}")

    for split_key, split_name in splits.items():
        anno_file = data_root / f"{split_name}_annotations.json"
        print(f"\nProcessing split '{split_key}' (name='{split_name}')")

        if not anno_file.exists():
            print(f"  [Warning] Missing annotation file: {anno_file}")
            report_data[split_key] = {
                "missing_annotation": True,
                "annotation_path": str(anno_file),
            }
            continue

        coco = load_coco_quiet(anno_file)
        split_stats = collect_counts_for_split(
            coco=coco,
            split_name=split_name,
            image_base_dir=image_base_dir,
            thresholds=THRESHOLDS,
        )
        report_data[split_key] = {
            "missing_annotation": False,
            "annotation_path": str(anno_file),
            **split_stats,
        }

        total_images = split_stats["total_images"]
        summary = ", ".join(
            f">{threshold}: {len(split_stats['rows_by_threshold'][threshold])}"
            for threshold in THRESHOLDS
        )
        print(f"  Total images: {total_images} | Matches -> {summary}")

    meta = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_path": str(data_root),
        "splits": splits,
        "thresholds": THRESHOLDS,
    }

    markdown_report = render_markdown(report_data=report_data, meta=meta)
    output_path.write_text(markdown_report, encoding="utf-8")
    print(f"\nReport written to: {output_path}")


if __name__ == "__main__":
    main()
