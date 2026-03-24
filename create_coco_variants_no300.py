#!/usr/bin/env python3
# USE this file only to create the splits
import json
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from pycocotools.coco import COCO


THRESHOLD = 300
VAL_OUTPUT_NAME = "valid_no300"
TEST_OUTPUT_NAME = "test_no300"
TRAIN_PROMOTED_OUTPUT_NAME = "train_plus_valgt300"


def _strip_split_prefix(file_name: str, split_name: str) -> str:
    if file_name.startswith(f"../{split_name}/"):
        return file_name[len(f"../{split_name}/") :]
    if file_name.startswith(f"{split_name}/"):
        return file_name[len(f"{split_name}/") :]
    return file_name


def _sanitize_file_name(file_name: str) -> str:
    if (
        "/" in file_name
        or "\\" in file_name
        or ".cache/datasets/" in file_name
        or Path(file_name).is_absolute()
    ):
        return Path(file_name).name
    return file_name


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload: dict[str, Any], pretty: bool = False) -> None:
    with path.open("w", encoding="utf-8") as fh:
        if pretty:
            json.dump(payload, fh, indent=2)
        else:
            json.dump(payload, fh, separators=(",", ":"))


def _file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _label_counts_str(label_counts: dict[str, int]) -> str:
    if not label_counts:
        return ""
    ordered = sorted(
        label_counts.items(), key=lambda item: (-int(item[1]), str(item[0]))
    )
    return ", ".join(f"{name}:{count}" for name, count in ordered)


def _md_escape(value: str) -> str:
    return value.replace("|", r"\|")


def _load_coco_quiet(path: Path) -> COCO:
    import contextlib
    import os

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            return COCO(str(path))


def analyze_and_filter_split(
    payload: dict[str, Any],
    split_name: str,
    threshold: int,
    normalize_file_names: bool = False,
) -> dict[str, Any]:
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    categories = payload.get("categories", [])

    cat_id_to_name = {int(cat["id"]): str(cat["name"]) for cat in categories}

    ann_count_by_image: Counter[int] = Counter()
    label_count_by_image: dict[int, Counter[int]] = defaultdict(Counter)

    for ann in annotations:
        image_id = int(ann["image_id"])
        cat_id = int(ann["category_id"])
        ann_count_by_image[image_id] += 1
        label_count_by_image[image_id][cat_id] += 1

    dropped_image_ids = {
        img_id for img_id, c in ann_count_by_image.items() if c > threshold
    }

    filtered_images = []
    dropped_images = []
    for image in images:
        img_id = int(image["id"])
        if img_id in dropped_image_ids:
            dropped_images.append(image)
        else:
            new_image = dict(image)
            if normalize_file_names:
                orig_file_name = str(image.get("file_name", ""))
                new_image["file_name"] = _sanitize_file_name(orig_file_name)
            filtered_images.append(new_image)

    filtered_annotations = []
    dropped_annotations = []
    for ann in annotations:
        img_id = int(ann["image_id"])
        if img_id in dropped_image_ids:
            dropped_annotations.append(ann)
        else:
            filtered_annotations.append(ann)

    filtered_payload = dict(payload)
    filtered_payload["images"] = filtered_images
    filtered_payload["annotations"] = filtered_annotations

    aggregate_label_counts: Counter[str] = Counter()
    excluded_rows = []
    for image in dropped_images:
        img_id = int(image["id"])
        per_label_by_name: dict[str, int] = {}
        for cat_id, count in label_count_by_image.get(img_id, Counter()).items():
            cat_name = cat_id_to_name.get(int(cat_id), f"cat_{cat_id}")
            per_label_by_name[cat_name] = int(count)
            aggregate_label_counts[cat_name] += int(count)

        excluded_rows.append(
            {
                "split": split_name,
                "source_image_id": img_id,
                "file_name": str(image.get("file_name", "")),
                "bbox_count": int(ann_count_by_image.get(img_id, 0)),
                "label_counts": dict(
                    sorted(per_label_by_name.items(), key=lambda item: item[0])
                ),
            }
        )

    excluded_rows.sort(
        key=lambda row: (
            -int(row["bbox_count"]),
            str(row["file_name"]),
            int(row["source_image_id"]),
        )
    )

    return {
        "split_name": split_name,
        "threshold": int(threshold),
        "source_images_count": int(len(images)),
        "source_annotations_count": int(len(annotations)),
        "excluded_images_count": int(len(dropped_images)),
        "excluded_annotations_count": int(len(dropped_annotations)),
        "kept_images_count": int(len(filtered_images)),
        "kept_annotations_count": int(len(filtered_annotations)),
        "dropped_image_ids": sorted(int(x) for x in dropped_image_ids),
        "dropped_images": dropped_images,
        "dropped_annotations": dropped_annotations,
        "aggregate_label_counts": dict(
            sorted(aggregate_label_counts.items(), key=lambda item: item[0])
        ),
        "excluded_rows": excluded_rows,
        "filtered_payload": filtered_payload,
    }


def build_train_plus_val_promoted(
    train_payload: dict[str, Any],
    dropped_val_images: list[dict[str, Any]],
    dropped_val_annotations: list[dict[str, Any]],
    val_split_name: str,
    train_split_name: str,
    copy_images: bool = False,
) -> dict[str, Any]:
    train_images = train_payload.get("images", [])
    train_annotations = train_payload.get("annotations", [])

    max_image_id = max((int(image["id"]) for image in train_images), default=0)
    max_annotation_id = max((int(ann["id"]) for ann in train_annotations), default=0)

    new_train_images = []
    for image in train_images:
        new_image = dict(image)
        if copy_images:
            orig_file_name = str(image.get("file_name", ""))
            new_image["file_name"] = _sanitize_file_name(orig_file_name)
        new_train_images.append(new_image)

    promoted_images = []
    source_to_new_image_id: dict[int, int] = {}
    next_image_id = max_image_id + 1
    for image in dropped_val_images:
        source_image_id = int(image["id"])
        new_image = dict(image)
        new_image["id"] = int(next_image_id)
        orig_file_name = str(image.get("file_name", ""))
        if copy_images:
            new_image["file_name"] = _sanitize_file_name(orig_file_name)
        else:
            new_image["file_name"] = f"../{val_split_name}/{orig_file_name}"
        promoted_images.append(new_image)
        source_to_new_image_id[source_image_id] = int(next_image_id)
        next_image_id += 1

    promoted_annotations = []
    next_annotation_id = max_annotation_id + 1
    for ann in dropped_val_annotations:
        source_image_id = int(ann["image_id"])
        if source_image_id not in source_to_new_image_id:
            continue
        new_ann = dict(ann)
        new_ann["id"] = int(next_annotation_id)
        new_ann["image_id"] = int(source_to_new_image_id[source_image_id])
        promoted_annotations.append(new_ann)
        next_annotation_id += 1

    out_payload = dict(train_payload)
    out_payload["images"] = new_train_images + promoted_images
    out_payload["annotations"] = list(train_annotations) + promoted_annotations

    return {
        "payload": out_payload,
        "promoted_images_count": int(len(promoted_images)),
        "promoted_annotations_count": int(len(promoted_annotations)),
        "promoted_source_image_ids": sorted(
            int(k) for k in source_to_new_image_id.keys()
        ),
        "promoted_new_image_ids": sorted(
            int(v) for v in source_to_new_image_id.values()
        ),
        "new_image_id_range": (
            [max_image_id + 1, max_image_id + len(promoted_images)]
            if promoted_images
            else []
        ),
        "new_annotation_id_range": (
            [max_annotation_id + 1, max_annotation_id + len(promoted_annotations)]
            if promoted_annotations
            else []
        ),
    }


def render_exclusion_markdown(
    *,
    dataset_path: Path,
    threshold: int,
    val_stats: dict[str, Any],
    test_stats: dict[str, Any],
    promoted_val_source_split: str,
    output_paths: dict[str, Path],
) -> str:
    rows = []
    for row in val_stats["excluded_rows"]:
        row_copy = deepcopy(row)
        row_copy["promoted_to_train"] = "yes"
        rows.append(row_copy)
    for row in test_stats["excluded_rows"]:
        row_copy = deepcopy(row)
        row_copy["promoted_to_train"] = "no"
        rows.append(row_copy)

    split_order = {
        val_stats["split_name"]: 0,
        test_stats["split_name"]: 1,
    }
    rows.sort(
        key=lambda row: (
            split_order.get(str(row["split"]), 99),
            -int(row["bbox_count"]),
            str(row["file_name"]),
            int(row["source_image_id"]),
        )
    )

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    lines: list[str] = []
    lines.append("# Excluded Image/Label Pairs (`>300` BBoxes)")
    lines.append("")
    lines.append(f"- Generated At: `{generated_at}`")
    lines.append(f"- Dataset Path: `{dataset_path}`")
    lines.append(f"- Threshold Rule: `bbox_count > {threshold}`")
    lines.append(f"- Promoted To Train Source: `{promoted_val_source_split}`")
    lines.append(f"- Filtered Val Annotation File: `{output_paths['val']}`")
    lines.append(f"- Filtered Test Annotation File: `{output_paths['test']}`")
    lines.append(f"- Promoted Train Annotation File: `{output_paths['train_plus']}`")
    lines.append("")

    lines.append("## Summary by Split")
    lines.append("")
    lines.append(
        "| Split | Excluded Images | Excluded Annotations | Kept Images | Kept Annotations | Promoted To Train |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    lines.append(
        f"| {val_stats['split_name']} | {val_stats['excluded_images_count']} | "
        f"{val_stats['excluded_annotations_count']} | {val_stats['kept_images_count']} | "
        f"{val_stats['kept_annotations_count']} | yes |"
    )
    lines.append(
        f"| {test_stats['split_name']} | {test_stats['excluded_images_count']} | "
        f"{test_stats['excluded_annotations_count']} | {test_stats['kept_images_count']} | "
        f"{test_stats['kept_annotations_count']} | no |"
    )
    lines.append("")

    lines.append("## Aggregate Excluded Label Counts")
    lines.append("")
    lines.append("| Split | Label | Count |")
    lines.append("| --- | --- | ---: |")
    for split_name, label_counts in (
        (val_stats["split_name"], val_stats["aggregate_label_counts"]),
        (test_stats["split_name"], test_stats["aggregate_label_counts"]),
    ):
        if not label_counts:
            lines.append(f"| {split_name} | _none_ | 0 |")
            continue
        for label, count in sorted(label_counts.items(), key=lambda item: item[0]):
            lines.append(f"| {split_name} | {_md_escape(str(label))} | {int(count)} |")
    lines.append("")

    total_promoted_images = int(val_stats["excluded_images_count"])
    total_not_promoted_images = int(test_stats["excluded_images_count"])
    total_promoted_annotations = int(val_stats["excluded_annotations_count"])
    total_not_promoted_annotations = int(test_stats["excluded_annotations_count"])

    lines.append("## Promotion Totals")
    lines.append("")
    lines.append(
        f"- Promoted Images (from `{val_stats['split_name']}`): `{total_promoted_images}`"
    )
    lines.append(
        f"- Promoted Annotations (from `{val_stats['split_name']}`): `{total_promoted_annotations}`"
    )
    lines.append(
        f"- Not Promoted Images (from `{test_stats['split_name']}`): `{total_not_promoted_images}`"
    )
    lines.append(
        f"- Not Promoted Annotations (from `{test_stats['split_name']}`): `{total_not_promoted_annotations}`"
    )
    lines.append("")

    lines.append("## Excluded Image/Label Pairs (Detailed)")
    lines.append("")
    lines.append(
        "| Split | Source Image ID | File Name | BBox Count | Label Counts | Promoted To Train |"
    )
    lines.append("| --- | ---: | --- | ---: | --- | --- |")
    for row in rows:
        lines.append(
            f"| {row['split']} | {int(row['source_image_id'])} | "
            f"{_md_escape(str(row['file_name']))} | {int(row['bbox_count'])} | "
            f"{_md_escape(_label_counts_str(row['label_counts']))} | "
            f"{row['promoted_to_train']} |"
        )
    lines.append("")

    return "\n".join(lines)


def validate_outputs(
    *,
    val_path: Path,
    test_path: Path,
    train_plus_payload: dict[str, Any],
    promoted_new_image_ids: list[int],
    threshold: int,
    promoted_val_split_name: str,
    train_split_name: str,
    copy_images: bool,
) -> dict[str, Any]:
    val_coco = _load_coco_quiet(val_path)
    test_coco = _load_coco_quiet(test_path)

    max_val_bbox_per_image = max(
        (len(val_coco.imgToAnns.get(img_id, [])) for img_id in val_coco.getImgIds()),
        default=0,
    )
    max_test_bbox_per_image = max(
        (len(test_coco.imgToAnns.get(img_id, [])) for img_id in test_coco.getImgIds()),
        default=0,
    )

    promoted_id_set = {int(x) for x in promoted_new_image_ids}
    promoted_image_path_violations = []
    base_train_image_path_violations = []
    bad_path_fragments = []
    for image in train_plus_payload.get("images", []):
        image_id = int(image["id"])
        file_name = str(image.get("file_name", ""))
        if copy_images:
            if "/" in file_name or "\\" in file_name:
                bad_path_fragments.append(file_name)
            if ".cache/datasets/" in file_name or Path(file_name).is_absolute():
                bad_path_fragments.append(file_name)
        else:
            if image_id in promoted_id_set and not file_name.startswith(
                f"../{promoted_val_split_name}/"
            ):
                promoted_image_path_violations.append(file_name)
            if image_id not in promoted_id_set and file_name.startswith("../"):
                base_train_image_path_violations.append(file_name)

    image_ids = [int(image["id"]) for image in train_plus_payload.get("images", [])]
    ann_ids = [int(ann["id"]) for ann in train_plus_payload.get("annotations", [])]
    image_id_set = set(image_ids)
    missing_image_refs = sorted(
        {
            int(ann["image_id"])
            for ann in train_plus_payload.get("annotations", [])
            if int(ann["image_id"]) not in image_id_set
        }
    )

    return {
        "max_val_bbox_per_image": int(max_val_bbox_per_image),
        "max_test_bbox_per_image": int(max_test_bbox_per_image),
        "val_filter_rule_satisfied": bool(max_val_bbox_per_image <= threshold),
        "test_filter_rule_satisfied": bool(max_test_bbox_per_image <= threshold),
        "unique_train_plus_image_ids": bool(len(image_ids) == len(set(image_ids))),
        "unique_train_plus_annotation_ids": bool(len(ann_ids) == len(set(ann_ids))),
        "train_plus_missing_image_refs": missing_image_refs,
        "promoted_image_path_violations": promoted_image_path_violations,
        "base_train_image_path_violations": base_train_image_path_violations,
        "bad_path_fragments": bad_path_fragments,
    }


def _sanitize_payload_images(payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    images = payload.get("images", [])
    changed = 0
    new_images = []
    for image in images:
        new_image = dict(image)
        orig = str(image.get("file_name", ""))
        sanitized = _sanitize_file_name(orig)
        if sanitized != orig:
            changed += 1
        new_image["file_name"] = sanitized
        new_images.append(new_image)
    payload = dict(payload)
    payload["images"] = new_images
    return payload, changed




def _copy_images(
    *,
    items: list[tuple[Path, Path]],
    workers: int,
) -> tuple[int, int]:
    copied = 0
    skipped = 0
    if workers <= 1:
        for src, dst in items:
            if dst.exists():
                skipped += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
        return copied, skipped

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for src, dst in items:
            if dst.exists():
                skipped += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            futures[pool.submit(shutil.copy2, src, dst)] = dst
        for fut in as_completed(futures):
            fut.result()
            copied += 1
    return copied, skipped


@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    data_root = Path(cfg.data.path).expanduser()
    repo_root = Path(get_original_cwd())
    report_dir = repo_root / "analysis_plots"
    report_dir.mkdir(parents=True, exist_ok=True)

    train_split_name = str(getattr(cfg, "train_name", "train"))
    val_split_name = str(getattr(cfg, "val_name", "valid"))
    test_split_name = str(getattr(cfg, "test_name", "test"))

    no300_cfg = getattr(cfg, "no300", None)
    copy_images = bool(getattr(no300_cfg, "copy_images", False)) if no300_cfg else False
    images_dir_name = (
        str(getattr(no300_cfg, "images_dir", "images")) if no300_cfg else "images"
    )
    copy_workers = int(getattr(no300_cfg, "copy_workers", 8)) if no300_cfg else 8
    repair_only = bool(getattr(no300_cfg, "repair_only", False)) if no300_cfg else False

    train_src_path = data_root / f"{train_split_name}_annotations.json"
    val_src_path = data_root / f"{val_split_name}_annotations.json"
    test_src_path = data_root / f"{test_split_name}_annotations.json"

    missing_paths = [
        path
        for path in (train_src_path, val_src_path, test_src_path)
        if not path.exists()
    ]
    if missing_paths:
        missing_msg = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(f"Missing source annotation files:\n{missing_msg}")

    print(f"Dataset root: {data_root}")
    print(
        f"Using splits: train={train_split_name}, val={val_split_name}, test={test_split_name}"
    )
    print(f"Threshold: >{THRESHOLD}")
    print(f"Copy images: {copy_images}")
    print(f"Repair only: {repair_only}")


    if repair_only:
        valid_no300_path = data_root / f"{VAL_OUTPUT_NAME}_annotations.json"
        test_no300_path = data_root / f"{TEST_OUTPUT_NAME}_annotations.json"
        train_plus_path = data_root / f"{TRAIN_PROMOTED_OUTPUT_NAME}_annotations.json"
        for path in (valid_no300_path, test_no300_path, train_plus_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing output annotation file for repair: {path}")

        valid_payload, valid_changed = _sanitize_payload_images(load_json(valid_no300_path))
        test_payload, test_changed = _sanitize_payload_images(load_json(test_no300_path))
        train_payload, train_changed = _sanitize_payload_images(load_json(train_plus_path))

        write_json(valid_no300_path, valid_payload, pretty=False)
        write_json(test_no300_path, test_payload, pretty=False)
        write_json(train_plus_path, train_payload, pretty=False)
        print(
            f"Repaired file_name values: valid_no300={valid_changed}, "
            f"test_no300={test_changed}, train_plus_valgt300={train_changed}"
        )

        if copy_images:
            images_root = data_root / images_dir_name
            val_src_root = images_root / val_split_name
            test_src_root = images_root / test_split_name
            train_src_root = images_root / train_split_name
            val_dst_root = images_root / VAL_OUTPUT_NAME
            test_dst_root = images_root / TEST_OUTPUT_NAME
            train_plus_dst_root = images_root / TRAIN_PROMOTED_OUTPUT_NAME

            print("\nCopying filtered validation images (repair)...")
            val_items: list[tuple[Path, Path]] = []
            for image in valid_payload.get("images", []):
                rel = str(image.get("file_name", ""))
                val_items.append((val_src_root / rel, val_dst_root / rel))
            copied, skipped = _copy_images(items=val_items, workers=copy_workers)
            print(f"  valid_no300: copied={copied}, skipped={skipped}")

            print("\nCopying filtered test images (repair)...")
            test_items: list[tuple[Path, Path]] = []
            for image in test_payload.get("images", []):
                rel = str(image.get("file_name", ""))
                test_items.append((test_src_root / rel, test_dst_root / rel))
            copied, skipped = _copy_images(items=test_items, workers=copy_workers)
            print(f"  test_no300: copied={copied}, skipped={skipped}")

            print("\nCopying train_plus_valgt300 images (repair)...")
            train_items: list[tuple[Path, Path]] = []
            missing = 0
            for image in train_payload.get("images", []):
                rel = str(image.get("file_name", ""))
                src = train_src_root / rel
                if not src.exists():
                    alt = val_src_root / rel
                    if alt.exists():
                        src = alt
                    else:
                        missing += 1
                        continue
                train_items.append((src, train_plus_dst_root / rel))
            copied, skipped = _copy_images(items=train_items, workers=copy_workers)
            print(f"  train_plus_valgt300: copied={copied}, skipped={skipped}, missing={missing}")

        print("\nRepair completed.")
        return

    print("\nLoading and filtering validation annotations...")
    val_payload = load_json(val_src_path)
    val_stats = analyze_and_filter_split(
        val_payload,
        val_split_name,
        THRESHOLD,
        normalize_file_names=copy_images,
    )
    valid_no300_path = data_root / f"{VAL_OUTPUT_NAME}_annotations.json"
    write_json(valid_no300_path, val_stats["filtered_payload"], pretty=False)
    print(
        f"  Wrote {valid_no300_path.name}: images={val_stats['kept_images_count']}, "
        f"annotations={val_stats['kept_annotations_count']}, excluded_images={val_stats['excluded_images_count']}"
    )

    dropped_val_images = val_stats["dropped_images"]
    dropped_val_annotations = val_stats["dropped_annotations"]

    print("\nLoading and filtering test annotations...")
    test_payload = load_json(test_src_path)
    test_stats = analyze_and_filter_split(
        test_payload,
        test_split_name,
        THRESHOLD,
        normalize_file_names=copy_images,
    )
    test_no300_path = data_root / f"{TEST_OUTPUT_NAME}_annotations.json"
    write_json(test_no300_path, test_stats["filtered_payload"], pretty=False)
    print(
        f"  Wrote {test_no300_path.name}: images={test_stats['kept_images_count']}, "
        f"annotations={test_stats['kept_annotations_count']}, excluded_images={test_stats['excluded_images_count']}"
    )

    print("\nLoading train annotations and creating promoted-train variant...")
    train_payload = load_json(train_src_path)
    train_plus = build_train_plus_val_promoted(
        train_payload=train_payload,
        dropped_val_images=dropped_val_images,
        dropped_val_annotations=dropped_val_annotations,
        val_split_name=val_split_name,
        train_split_name=train_split_name,
        copy_images=copy_images,
    )
    train_plus_path = data_root / f"{TRAIN_PROMOTED_OUTPUT_NAME}_annotations.json"
    write_json(train_plus_path, train_plus["payload"], pretty=False)
    print(
        f"  Wrote {train_plus_path.name}: images={len(train_plus['payload']['images'])}, "
        f"annotations={len(train_plus['payload']['annotations'])}, "
        f"promoted_images={train_plus['promoted_images_count']}"
    )

    report_path = report_dir / "no300_excluded_image_label_pairs.md"
    report_text = render_exclusion_markdown(
        dataset_path=data_root,
        threshold=THRESHOLD,
        val_stats=val_stats,
        test_stats=test_stats,
        promoted_val_source_split=val_split_name,
        output_paths={
            "val": valid_no300_path,
            "test": test_no300_path,
            "train_plus": train_plus_path,
        },
    )
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\nWrote report: {report_path}")

    if copy_images:
        images_root = data_root / images_dir_name
        val_src_root = images_root / val_split_name
        test_src_root = images_root / test_split_name
        train_src_root = images_root / train_split_name
        val_dst_root = images_root / VAL_OUTPUT_NAME
        test_dst_root = images_root / TEST_OUTPUT_NAME
        train_plus_dst_root = images_root / TRAIN_PROMOTED_OUTPUT_NAME

        print("\nCopying filtered validation images...")
        val_items: list[tuple[Path, Path]] = []
        for image in val_stats["filtered_payload"].get("images", []):
            rel = str(image.get("file_name", ""))
            val_items.append((val_src_root / rel, val_dst_root / rel))
        copied, skipped = _copy_images(items=val_items, workers=copy_workers)
        print(f"  valid_no300: copied={copied}, skipped={skipped}")

        print("\nCopying filtered test images...")
        test_items: list[tuple[Path, Path]] = []
        for image in test_stats["filtered_payload"].get("images", []):
            rel = str(image.get("file_name", ""))
            test_items.append((test_src_root / rel, test_dst_root / rel))
        copied, skipped = _copy_images(items=test_items, workers=copy_workers)
        print(f"  test_no300: copied={copied}, skipped={skipped}")

        print("\nCopying train_plus_valgt300 images...")
        train_items: list[tuple[Path, Path]] = []
        promoted_id_set = set(train_plus["promoted_new_image_ids"])
        for image in train_plus["payload"].get("images", []):
            rel = str(image.get("file_name", ""))
            image_id = int(image.get("id"))
            if image_id in promoted_id_set:
                src = val_src_root / rel
            else:
                src = train_src_root / rel
            train_items.append((src, train_plus_dst_root / rel))
        copied, skipped = _copy_images(items=train_items, workers=copy_workers)
        print(f"  train_plus_valgt300: copied={copied}, skipped={skipped}")

    validation_results = validate_outputs(
        val_path=valid_no300_path,
        test_path=test_no300_path,
        train_plus_payload=train_plus["payload"],
        promoted_new_image_ids=train_plus["promoted_new_image_ids"],
        threshold=THRESHOLD,
        promoted_val_split_name=val_split_name,
        train_split_name=train_split_name,
        copy_images=copy_images,
    )

    if not validation_results["val_filter_rule_satisfied"]:
        raise RuntimeError(
            f"Validation filter check failed: max bbox/image={validation_results['max_val_bbox_per_image']}"
        )
    if not validation_results["test_filter_rule_satisfied"]:
        raise RuntimeError(
            f"Test filter check failed: max bbox/image={validation_results['max_test_bbox_per_image']}"
        )
    if not validation_results["unique_train_plus_image_ids"]:
        raise RuntimeError("train_plus_valgt300 has duplicate image IDs.")
    if not validation_results["unique_train_plus_annotation_ids"]:
        raise RuntimeError("train_plus_valgt300 has duplicate annotation IDs.")
    if validation_results["train_plus_missing_image_refs"]:
        raise RuntimeError(
            "train_plus_valgt300 has annotations referencing missing images: "
            f"{validation_results['train_plus_missing_image_refs'][:10]}"
        )
    if validation_results["promoted_image_path_violations"]:
        raise RuntimeError(
            "Found promoted image file_name values not under expected split prefix: "
            f"{validation_results['promoted_image_path_violations'][:10]}"
        )
    if validation_results["base_train_image_path_violations"]:
        raise RuntimeError(
            "Found base train images missing the expected path prefix: "
            f"{validation_results['base_train_image_path_violations'][:10]}"
        )

    manifest_path = data_root / "no300_variants_manifest.json"
    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_path": str(data_root),
        "threshold": THRESHOLD,
        "source_splits": {
            "train": train_split_name,
            "val": val_split_name,
            "test": test_split_name,
        },
        "output_split_names": {
            "val_filtered": VAL_OUTPUT_NAME,
            "test_filtered": TEST_OUTPUT_NAME,
            "train_promoted": TRAIN_PROMOTED_OUTPUT_NAME,
        },
        "copy_images": copy_images,
        "images_dir": images_dir_name,
        "copy_workers": copy_workers,
        "source_files": {
            "train": _file_signature(train_src_path),
            "val": _file_signature(val_src_path),
            "test": _file_signature(test_src_path),
        },
        "generated_files": {
            "val_filtered": _file_signature(valid_no300_path),
            "test_filtered": _file_signature(test_no300_path),
            "train_promoted": _file_signature(train_plus_path),
            "report": _file_signature(report_path),
        },
        "stats": {
            "val": {
                "source_split": val_split_name,
                "excluded_images_count": val_stats["excluded_images_count"],
                "excluded_annotations_count": val_stats["excluded_annotations_count"],
                "kept_images_count": val_stats["kept_images_count"],
                "kept_annotations_count": val_stats["kept_annotations_count"],
                "excluded_image_ids": val_stats["dropped_image_ids"],
                "aggregate_excluded_label_counts": val_stats["aggregate_label_counts"],
                "promoted_to_train": True,
            },
            "test": {
                "source_split": test_split_name,
                "excluded_images_count": test_stats["excluded_images_count"],
                "excluded_annotations_count": test_stats["excluded_annotations_count"],
                "kept_images_count": test_stats["kept_images_count"],
                "kept_annotations_count": test_stats["kept_annotations_count"],
                "excluded_image_ids": test_stats["dropped_image_ids"],
                "aggregate_excluded_label_counts": test_stats["aggregate_label_counts"],
                "promoted_to_train": False,
            },
            "train_promoted": {
                "source_split": train_split_name,
                "promoted_from_split": val_split_name,
                "base_images_count": len(train_payload.get("images", [])),
                "base_annotations_count": len(train_payload.get("annotations", [])),
                "promoted_images_count": train_plus["promoted_images_count"],
                "promoted_annotations_count": train_plus["promoted_annotations_count"],
                "total_images_count": len(train_plus["payload"].get("images", [])),
                "total_annotations_count": len(
                    train_plus["payload"].get("annotations", [])
                ),
                "promoted_source_image_ids": train_plus["promoted_source_image_ids"],
                "new_image_id_range": train_plus["new_image_id_range"],
                "new_annotation_id_range": train_plus["new_annotation_id_range"],
            },
        },
        "validations": validation_results,
    }
    write_json(manifest_path, manifest, pretty=True)
    print(f"Wrote manifest: {manifest_path}")

    print("\nAll variants generated successfully.")
    print(
        f"Expected quick check -> valid_no300 images={val_stats['kept_images_count']}, "
        f"test_no300 images={test_stats['kept_images_count']}, "
        f"train_plus images={len(train_plus['payload']['images'])}"
    )


if __name__ == "__main__":
    main()
