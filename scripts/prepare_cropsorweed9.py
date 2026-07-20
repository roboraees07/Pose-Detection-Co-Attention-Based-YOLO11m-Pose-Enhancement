#!/usr/bin/env python3
"""Build CropsOrWeed9 YOLO-Pose dataset from CropAndWeed raw annotations.

Requires the CropAndWeed dataset (images + bboxes/CropsOrWeed9 CSV files).
Default paths assume sibling folders under TUBITAK; override with CLI flags.

Output layout:
  data/CropsOrWeed9/
    images/{train,val,test}/
    labels/{train,val,test}/
    all_labels/          # intermediate flat label dir
    dataset_info.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_ROOT / "data" / "CropsOrWeed9"
DEFAULT_CNW_DATA = PROJECT_ROOT.parent / "Datasets" / "cropandweed-dataset" / "data"
DEFAULT_CNW_REPO = PROJECT_ROOT.parent / "Datasets" / "cropandweed-dataset"

VARIANT = "CropsOrWeed9"
RANDOM_SEED = 42
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15
IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# CropsOrWeed9 class names (WACV 2023 CropAndWeed variant)
CLASS_NAMES = [
    "Maize",
    "Sugar beet",
    "Soy",
    "Sunflower",
    "Potato",
    "Pea",
    "Bean",
    "Pumpkin",
    "Weed",
]


def load_variant_mapping(cnw_repo: Path) -> dict[int, int]:
    """Load source label ID -> target class ID mapping from CropAndWeed utilities."""
    sys.path.insert(0, str(cnw_repo / "cnw"))
    from utilities.datasets import DATASETS  # noqa: WPS433

    ds = DATASETS[VARIANT]
    return {int(s): int(t) for s, t in ds.mapping.items()}


def find_image(images_root: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = images_root / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def parse_bbox_csv(csv_path: Path, source_to_target: dict[int, int], n_classes: int) -> list[tuple]:
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 7:
                continue
            try:
                left, top, right, bottom = map(float, row[:4])
                raw_id = int(float(row[4]))
                stem_x, stem_y = float(row[5]), float(row[6])
            except (ValueError, IndexError):
                continue
            if 0 <= raw_id < n_classes:
                cls_id = raw_id
            else:
                cls_id = source_to_target.get(raw_id)
                if cls_id is None:
                    continue
            rows.append((left, top, right, bottom, int(cls_id), stem_x, stem_y))
    return rows


def convert_labels(bbox_root: Path, images_root: Path, out_labels: Path, source_to_target: dict[int, int]) -> tuple[dict, list[str]]:
    n_classes = len(CLASS_NAMES)
    out_labels.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(bbox_root.glob("*.csv"))
    valid_stems: list[str] = []
    stats = {"images_with_labels": 0, "instances": 0, "skipped_missing_image": 0, "skipped_invalid": 0}

    for csv_path in tqdm(csv_files, desc=f"{VARIANT} -> YOLO-Pose labels"):
        stem = csv_path.stem
        img_path = find_image(images_root, stem)
        if img_path is None:
            stats["skipped_missing_image"] += 1
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            stats["skipped_missing_image"] += 1
            continue
        h, w = img.shape[:2]
        lines = []
        for left, top, right, bottom, cls_id, stem_x, stem_y in parse_bbox_csv(csv_path, source_to_target, n_classes):
            left = float(np.clip(left, 0, w - 1))
            right = float(np.clip(right, 0, w - 1))
            top = float(np.clip(top, 0, h - 1))
            bottom = float(np.clip(bottom, 0, h - 1))
            bw, bh = right - left, bottom - top
            if bw <= 1 or bh <= 1 or not (0 <= cls_id < n_classes):
                stats["skipped_invalid"] += 1
                continue
            cx, cy = (left + right) / 2 / w, (top + bottom) / 2 / h
            nw, nh = bw / w, bh / h
            inside = 0 <= stem_x < w and 0 <= stem_y < h
            kx, ky, v = (stem_x / w, stem_y / h, 2) if inside else (0.0, 0.0, 0)
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} {kx:.6f} {ky:.6f} {v}")
        if not lines:
            continue
        (out_labels / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        valid_stems.append(stem)
        stats["images_with_labels"] += 1
        stats["instances"] += len(lines)
    return stats, sorted(valid_stems)


def split_stems(stems: list[str], seed: int = RANDOM_SEED) -> dict[str, list[str]]:
    stems = sorted(set(stems))
    rng = random.Random(seed)
    rng.shuffle(stems)
    n = len(stems)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    return {
        "train": stems[:n_train],
        "val": stems[n_train : n_train + n_val],
        "test": stems[n_train + n_val :],
    }


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_splits(out_root: Path, images_root: Path, labels_all: Path, splits: dict[str, list[str]]) -> None:
    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    for split, stems in splits.items():
        for stem in stems:
            img = find_image(images_root, stem)
            lab = labels_all / f"{stem}.txt"
            if img is None or not lab.is_file():
                continue
            link_or_copy(img, out_root / "images" / split / f"{stem}{img.suffix}")
            link_or_copy(lab, out_root / "labels" / split / f"{stem}.txt")


def write_dataset_info(out_root: Path, stats: dict, splits: dict[str, list[str]]) -> None:
    info = {
        "variant": VARIANT,
        "source": "CropAndWeed dataset (WACV 2023)",
        "format": "YOLO-Pose (class cx cy w h kx ky v)",
        "classes": {i: n for i, n in enumerate(CLASS_NAMES)},
        "weed_class_id": 8,
        "split_seed": RANDOM_SEED,
        "split_ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": 1 - TRAIN_RATIO - VAL_RATIO},
        "split_counts": {k: len(v) for k, v in splits.items()},
        "preprocessing_stats": stats,
        "citation": "Steininger et al., WACV 2023 — The CropAndWeed Dataset",
    }
    (out_root / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare CropsOrWeed9 YOLO-Pose dataset")
    ap.add_argument("--cnw-data", type=Path, default=DEFAULT_CNW_DATA, help="CropAndWeed data/ folder (images + bboxes)")
    ap.add_argument("--cnw-repo", type=Path, default=DEFAULT_CNW_REPO, help="CropAndWeed repo root (for cnw/utilities)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output dataset root")
    ap.add_argument("--copy-images", action="store_true", help="Copy images instead of hardlink (for packaging)")
    ap.add_argument("--clean", action="store_true", help="Remove existing output before build")
    args = ap.parse_args()

    bbox_root = args.cnw_data / "bboxes" / VARIANT
    images_root = args.cnw_data / "images"
    if not bbox_root.is_dir():
        raise FileNotFoundError(f"Missing {bbox_root} — run CropAndWeed map_dataset.py --dataset_target CropsOrWeed9 first")

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)

    source_to_target = load_variant_mapping(args.cnw_repo)
    labels_all = args.out / "all_labels"
    stats, stems = convert_labels(bbox_root, images_root, labels_all, source_to_target)
    splits = split_stems(stems)

    global link_or_copy
    if args.copy_images:
        def link_or_copy(src: Path, dst: Path) -> None:  # noqa: ANN001
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    build_splits(args.out, images_root, labels_all, splits)
    write_dataset_info(args.out, stats, splits)

    # Update project data YAML path
    yaml_path = PROJECT_ROOT / "configs" / "cropsorweed9_yolopose.yaml"
    yaml_obj = {
        "path": str(args.out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
        "kpt_shape": [1, 3],
        "flip_idx": [0],
    }
    yaml_path.write_text(yaml.safe_dump(yaml_obj, sort_keys=False), encoding="utf-8")

    print(f"\nDone. Dataset: {args.out}")
    print(f"  train/val/test: {splits['train'].__len__()}/{splits['val'].__len__()}/{splits['test'].__len__()} images")
    print(f"  instances: {stats['instances']}")
    print(f"  YAML: {yaml_path}")


if __name__ == "__main__":
    main()
