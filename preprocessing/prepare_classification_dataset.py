"""
preprocessing/prepare_classification_dataset.py

Builds the 6-class (Healthy + 5 diseases) classification dataset consumed by
classification/train_classifier.py, from the split assignment in
dataset/processed/split_manifest.csv (run preprocessing/make_split_manifest.py
first).

Architecture decision (project brief section 5 -- "carefully consider
whether the CNN should operate on the complete image, YOLO crops, or both"):
this project trains the CNN on BOTH, for the disease classes:

  - the FULL original image (every class, including Healthy -- Healthy has
    no boxes so a full image is the *only* option for it)
  - every valid YOLO-box CROP (disease classes only), expanded by a context
    margin (utils.dataset_common.expand_and_clip_box)

This directly mirrors the two situations the CNN is actually asked to
classify at inference time (pipeline/inference_pipeline.py):
  1. YOLO finds nothing  -> CNN classifies the FULL frame (Healthy vs disease)
  2. YOLO finds a region -> CNN classifies the CROPPED region (verification)
A classifier trained only on crops would never have seen a full leaf frame
during training and vice versa, so training on both is the more scientifically
defensible choice, not an arbitrary complication (config toggles exist -- see
configs/config.yaml -> classification_dataset -- if you want to ablate this).

Data-leakage prevention (project brief section 9):
  - split assignment comes from the single master manifest, at the SOURCE
    IMAGE level -- every crop generated from one photo inherits that photo's
    split, so no two crops (or a crop + the full image) of the same source
    photo can ever land in different splits.
  - no augmentation happens here (only resize-free cropping from ground-truth
    boxes) -- augmentation belongs exclusively in the training dataloader.

Exact-duplicate images (identified by preprocessing/dataset_audit.py via
SHA-256, see outputs/dataset_audit.json -> integrity_issues) are de-duplicated
here at the SOURCE-IMAGE level before splitting, keeping only the first file
in each duplicate group, and logging what was dropped
(outputs/classification_dataset_prep_report.json -> duplicates_dropped) --
this prevents a byte-identical photo from appearing on both sides of a split.

Nothing under dataset/original/ is ever modified.

Usage:
    python preprocessing/prepare_classification_dataset.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import load_config, Config  # noqa: E402
from utils.dataset_common import (  # noqa: E402
    ALL_CLASSES,
    DISEASE_CLASSES,
    HEALTHY,
    discover_layout,
    expand_and_clip_box,
    find_exact_duplicate_keep_map,
    labels_dir_of,
    link_or_copy,
    parse_yolo_label_file,
)

SPLITS = ["train", "val", "test"]


def load_manifest(config: Config) -> pd.DataFrame:
    manifest_path = config.path("paths", "dataset_processed") / "split_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. Run preprocessing/make_split_manifest.py first."
        )
    return pd.read_csv(manifest_path)


def find_exact_duplicates(config: Config) -> dict:
    """Thin wrapper around utils.dataset_common.find_exact_duplicate_keep_map.

    NOTE: preprocessing/make_split_manifest.py now applies this SAME check
    before the split is even drawn, so by the time this script runs, exact
    duplicates should already be absent from split_manifest.csv entirely.
    This second pass is kept as a defensive no-op safety net (e.g. if
    someone runs this script against a stale/hand-edited manifest) -- it
    should find nothing left to drop in the normal pipeline.
    """
    return find_exact_duplicate_keep_map(config.path("paths", "outputs"))


def prepare(config: Config, margin: float, min_crop_px: int,
            include_full_disease: bool, include_crops_disease: bool,
            max_crops_per_image: Optional[int]) -> dict:
    layout = discover_layout(config.path("paths", "dataset_original"))
    manifest = load_manifest(config)
    dup_info = find_exact_duplicates(config)
    dup_keep = dup_info.get("keep", {})

    out_root = config.path("paths", "classification_dataset")
    images_root = out_root / "images"

    rows = []
    dropped_small_crop = 0
    dropped_duplicate = 0
    skipped_missing_file = 0

    # ---- Healthy: full image only ----
    healthy_rows = manifest[manifest["class"] == HEALTHY]
    for _, r in healthy_rows.iterrows():
        src = Path(r["abs_path"])
        if dup_keep and not dup_keep.get(str(src), True):
            dropped_duplicate += 1
            continue
        if not src.exists():
            skipped_missing_file += 1
            continue
        split = r["split"]
        dst = images_root / split / HEALTHY / src.name
        link_or_copy(src, dst)
        rows.append({
            "sample_id": f"{HEALTHY}__{src.stem}__full",
            "split": split, "label": HEALTHY, "sample_type": "full",
            "source_stem": src.stem, "image_path": str(dst),
            "box_cx": None, "box_cy": None, "box_w": None, "box_h": None,
        })

    # ---- Disease classes: full image (optional) + crops (optional) ----
    for cls in DISEASE_CLASSES:
        original_dir = layout.original_class_dirs.get(cls)
        ann_dir = layout.annotated_class_dirs.get(cls)
        labels_dir = labels_dir_of(ann_dir) if ann_dir else None
        class_rows = manifest[manifest["class"] == cls]

        for _, r in class_rows.iterrows():
            src = Path(r["abs_path"])
            stem, split = r["stem"], r["split"]

            if dup_keep and not dup_keep.get(str(src), True):
                dropped_duplicate += 1
                continue
            if not src.exists():
                skipped_missing_file += 1
                continue

            if include_full_disease:
                dst_full = images_root / split / cls / src.name
                link_or_copy(src, dst_full)
                rows.append({
                    "sample_id": f"{cls}__{stem}__full",
                    "split": split, "label": cls, "sample_type": "full",
                    "source_stem": stem, "image_path": str(dst_full),
                    "box_cx": None, "box_cy": None, "box_w": None, "box_h": None,
                })

            if include_crops_disease and labels_dir is not None:
                label_path = labels_dir / f"{stem}.txt"
                if not label_path.exists():
                    continue
                boxes = [b for b in parse_yolo_label_file(label_path) if b.valid]
                if max_crops_per_image is not None:
                    boxes = boxes[:max_crops_per_image]
                if not boxes:
                    continue

                try:
                    with Image.open(src) as img:
                        img = img.convert("RGB")
                        img_w, img_h = img.size
                        for idx, b in enumerate(boxes):
                            x0, y0, x1, y1 = expand_and_clip_box(b.cx, b.cy, b.w, b.h, img_w, img_h, margin)
                            if (x1 - x0) < min_crop_px or (y1 - y0) < min_crop_px:
                                dropped_small_crop += 1
                                continue
                            crop = img.crop((x0, y0, x1, y1))
                            crop_name = f"{stem}__crop{idx}.jpg"
                            dst_crop = images_root / split / cls / crop_name
                            dst_crop.parent.mkdir(parents=True, exist_ok=True)
                            crop.save(dst_crop, "JPEG", quality=95)
                            rows.append({
                                "sample_id": f"{cls}__{stem}__crop{idx}",
                                "split": split, "label": cls, "sample_type": "crop",
                                "source_stem": stem, "image_path": str(dst_crop),
                                "box_cx": b.cx, "box_cy": b.cy, "box_w": b.w, "box_h": b.h,
                            })
                except Exception as exc:  # noqa: BLE001
                    print(f"  WARNING: failed to crop {src} ({exc})")

    manifest_df = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(out_root / "manifest.csv", index=False)

    return {
        "manifest_df": manifest_df,
        "dropped_small_crop": dropped_small_crop,
        "dropped_duplicate": dropped_duplicate,
        "skipped_missing_file": skipped_missing_file,
        "duplicates_dropped_detail": dup_info.get("dropped", [])[:200],
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare the 6-class classification dataset.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--margin", type=float, default=None, help="Override crop_context_margin from config.yaml")
    parser.add_argument("--min-crop-px", type=int, default=None, help="Override min_crop_size_px from config.yaml")
    parser.add_argument("--no-full-images", action="store_true", help="Disable full-image samples for disease classes")
    parser.add_argument("--no-crops", action="store_true", help="Disable crop samples for disease classes")
    parser.add_argument("--max-crops-per-image", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config) if args.config else load_config()

    cds_cfg = config["classification_dataset"]
    margin = args.margin if args.margin is not None else cds_cfg["crop_context_margin"]
    min_crop_px = args.min_crop_px if args.min_crop_px is not None else cds_cfg["min_crop_size_px"]
    include_full = not args.no_full_images and cds_cfg["include_full_images_for_disease_classes"]
    include_crops = not args.no_crops and cds_cfg["include_crops_for_disease_classes"]

    print(f"margin={margin}  min_crop_px={min_crop_px}  "
          f"include_full_disease={include_full}  include_crops_disease={include_crops}")

    result = prepare(config, margin, min_crop_px, include_full, include_crops, args.max_crops_per_image)
    df = result["manifest_df"]

    print(f"\nTotal classification samples: {len(df)}")
    print(df.groupby(["label", "sample_type"], observed=True).size().unstack(fill_value=0))
    print("\nPer-split sample counts:")
    print(df.groupby(["split", "label"], observed=True).size().unstack(fill_value=0))

    print(f"\nDropped (crop below {min_crop_px}px after clipping): {result['dropped_small_crop']}")
    print(f"Dropped (exact-duplicate source image): {result['dropped_duplicate']}")
    if result["skipped_missing_file"]:
        print(f"Skipped (source file missing on disk): {result['skipped_missing_file']}")

    class_totals = df.groupby("label", observed=True).size()
    imbalance_ratio = round(float(class_totals.max() / max(class_totals.min(), 1)), 2)
    print(f"\nClass imbalance in final classification dataset: max={class_totals.idxmax()} "
          f"({class_totals.max()}), min={class_totals.idxmin()} ({class_totals.min()}), "
          f"ratio={imbalance_ratio}x -- train_classifier.py analyzes this and applies class-weighted "
          f"loss (see configs/config.yaml).")

    report_path = config.path("paths", "outputs") / "classification_dataset_prep_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "margin": margin, "min_crop_px": min_crop_px,
            "include_full_images_for_disease_classes": include_full,
            "include_crops_for_disease_classes": include_crops,
            "total_samples": int(len(df)),
            "counts_by_label_and_type": json.loads(
                df.groupby(["label", "sample_type"], observed=True).size().unstack(fill_value=0).to_json()
            ),
            "counts_by_split_and_label": json.loads(
                df.groupby(["split", "label"], observed=True).size().unstack(fill_value=0).to_json()
            ),
            "dropped_small_crop": result["dropped_small_crop"],
            "dropped_duplicate": result["dropped_duplicate"],
            "duplicates_dropped_detail": result["duplicates_dropped_detail"],
            "skipped_missing_file": result["skipped_missing_file"],
            "class_imbalance_ratio": imbalance_ratio,
        }, f, indent=2)
    print(f"\nWrote {report_path}")
    print(f"Wrote {config.path('paths', 'classification_dataset') / 'manifest.csv'}")


if __name__ == "__main__":
    main()
