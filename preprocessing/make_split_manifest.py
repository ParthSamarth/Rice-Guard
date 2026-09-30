"""
preprocessing/make_split_manifest.py

Creates ONE master, image-level train/val/test split and writes it to
dataset/processed/split_manifest.csv. This is the single source of truth
consumed by BOTH preprocessing/prepare_yolo_dataset.py and
preprocessing/prepare_classification_dataset.py, so that:

  - the same physical source image is never used for training by one model
    and evaluation by the other -- which would make the "held-out test set"
    meaningless for the integrated YOLO+CNN pipeline evaluation
    (evaluation/evaluate_pipeline.py) requested in the project brief
  - crops derived from a disease image (see prepare_classification_dataset.py)
    always inherit that source image's split, so no two crops of the same
    source photo can straddle train/val/test (a realistic leakage risk for
    Brown Spot / Leaf Smut images, which carry ~10-12 boxes each)

Split: 70/15/15 by default (configs/config.yaml -> split), stratified
independently within each of the 6 classes, fixed seed 42
(configs/config.yaml -> seed). No augmentation happens here -- this script
only decides which SOURCE IMAGES go where.

Usage:
    python preprocessing/make_split_manifest.py
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import load_config, Config  # noqa: E402
from utils.dataset_common import (  # noqa: E402
    ALL_CLASSES, discover_layout, find_exact_duplicate_keep_map, list_images,
)


def build_manifest(config: Config) -> pd.DataFrame:
    seed = config["seed"]
    ratios = config["split"]
    assert abs(sum(ratios.values()) - 1.0) < 1e-6, f"split ratios must sum to 1.0, got {ratios}"

    layout = discover_layout(config.path("paths", "dataset_original"))

    # Exclude exact byte-for-byte duplicate images (including cross-class
    # duplicates -- see outputs/dataset_audit.json) BEFORE splitting. Doing
    # this here, not just in preprocessing/prepare_classification_dataset.py,
    # is what keeps a duplicate pair from landing in different splits for
    # BOTH the YOLO dataset and the classification dataset, since both are
    # built from this one manifest.
    dup_info = find_exact_duplicate_keep_map(config.path("paths", "outputs"))
    dup_keep = dup_info.get("keep", {})
    if not dup_keep:
        print("  (no outputs/audit_details/per_image_scan.csv found -- run preprocessing/dataset_audit.py "
              "first for exact-duplicate exclusion; proceeding WITHOUT it for now)")
    n_excluded = 0

    rows = []
    for cls in ALL_CLASSES:
        directory = layout.original_class_dirs.get(cls)
        images = list_images(directory) if directory else []
        if dup_keep:
            kept_images = [p for p in images if dup_keep.get(str(p), True)]
            n_excluded += len(images) - len(kept_images)
            images = kept_images

        rng = random.Random(seed)
        shuffled = images[:]
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(round(n * ratios["train"]))
        n_val = int(round(n * ratios["val"]))
        n_test = n - n_train - n_val  # remainder, so rounding never drops/duplicates an image

        split_labels = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
        for path, split in zip(shuffled, split_labels):
            rows.append({
                "class": cls,
                "filename": path.name,
                "stem": path.stem,
                "abs_path": str(path),
                "split": split,
            })

    df = pd.DataFrame(rows).sort_values(["class", "split", "filename"]).reset_index(drop=True)
    print(f"Excluded {n_excluded} exact-duplicate images before splitting "
          f"(kept 1 representative per duplicate group).")
    return df


def main():
    parser = argparse.ArgumentParser(description="Build the master image-level train/val/test split manifest.")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    config = load_config(args.config) if args.config else load_config()

    df = build_manifest(config)

    # Sanity check: every image appears exactly once, no duplicates across splits.
    dup = df["abs_path"].duplicated().sum()
    if dup:
        raise RuntimeError(f"{dup} images appear more than once in the manifest -- this must never happen.")

    out_dir = config.path("paths", "dataset_processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "split_manifest.csv"
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} rows to {out_path}")
    print(df.groupby(["class", "split"], observed=True).size().unstack(fill_value=0))
    print("\nOverall split sizes:")
    print(df["split"].value_counts())


if __name__ == "__main__":
    main()
