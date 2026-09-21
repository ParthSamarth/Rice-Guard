"""
preprocessing/prepare_yolo_dataset.py

Builds a standard Ultralytics-layout YOLO dataset under
dataset/processed/yolo/{images,labels}/{train,val,test}/ from the raw
extracted archive, using the split assignment from
dataset/processed/split_manifest.csv (run preprocessing/make_split_manifest.py
first).

Architecture decision (project brief section 4): Healthy images have NO
bounding-box annotations, so YOLOv8 is trained as a 5-class DISEASE detector
only (Blast, Brown Spot, Leaf Smut, Rice Tungro, Sheath Blight). Healthy
images are included as background/negative images -- present in images/,
with an EMPTY label file -- which is how Ultralytics represents "no objects
in this image" (see https://docs.ultralytics.com -> "Background images").
No fake Healthy bounding boxes are ever created.

Class-ID remapping: the source label files use the dataset's own YOLO ids
(0=Brown Spot, 1=Blast, 2=Healthy, 3=Leaf Smut, 4=Rice Tungro, 5=Sheath
Blight -- see README.md). Since Healthy never has boxes, id 2 never appears
in any label file, so training on the original ids directly would leave an
unused class slot and break Ultralytics' contiguous-id assumption. This
script therefore assigns each box a NEW id purely from the folder it came
from (utils.dataset_common.YOLO_CLASS_TO_ID), ignoring whatever integer is
written in the source file -- this is more robust than trusting the source
integer, because folder placement is exactly what preprocessing/
dataset_audit.py cross-checks for internal consistency
(bounding_boxes.mapping_conflicts must be empty). If a source folder is
ever found to mix class ids, this script raises rather than silently
mislabeling boxes.

Invalid boxes (non-positive width/height, malformed lines, out-of-range
coordinates -- see dataset_audit.py) are dropped, never written into the
training set; every drop is counted and reported in
outputs/yolo_dataset_prep_report.json so nothing disappears silently.

Nothing under dataset/original/ is ever modified. Images are hardlinked
(falling back to a real copy) into dataset/processed/yolo/ -- hardlinking is
safe here because training only ever reads these files.

Usage:
    python preprocessing/prepare_yolo_dataset.py
    python preprocessing/prepare_yolo_dataset.py --healthy-background-fraction 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import load_config, Config  # noqa: E402
from utils.dataset_common import (  # noqa: E402
    DISEASE_CLASSES,
    HEALTHY,
    YOLO_CLASS_TO_ID,
    YOLO_ID_TO_CLASS,
    YOLO_TRAINING_CLASSES,
    discover_layout,
    labels_dir_of,
    link_or_copy,
    parse_yolo_label_file,
)

SPLITS = ["train", "val", "test"]


def load_manifest(config: Config) -> pd.DataFrame:
    manifest_path = config.path("paths", "dataset_processed") / "split_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. Run preprocessing/make_split_manifest.py first "
            "so YOLO and classification datasets share one consistent split."
        )
    return pd.read_csv(manifest_path)


def prepare_disease_classes(config: Config, layout, manifest: pd.DataFrame, yolo_root: Path) -> dict:
    report = {
        "per_class_per_split_images": defaultdict(lambda: defaultdict(int)),
        "per_class_dropped_boxes": defaultdict(int),
        "per_class_written_boxes": defaultdict(int),
        "images_without_any_valid_box": [],
    }

    for cls in DISEASE_CLASSES:
        ann_dir = layout.annotated_class_dirs.get(cls)
        labels_dir = labels_dir_of(ann_dir) if ann_dir else None
        original_dir = layout.original_class_dirs.get(cls)
        if labels_dir is None or original_dir is None:
            raise FileNotFoundError(f"Missing original/annotated folder for class '{cls}'")

        # Defensive consistency check: every label file in this folder must
        # resolve to a single observed class id (see module docstring).
        observed_ids = set()

        class_rows = manifest[manifest["class"] == cls]
        for _, row in class_rows.iterrows():
            stem, split = row["stem"], row["split"]
            src_image = Path(row["abs_path"])
            src_label = labels_dir / f"{stem}.txt"
            if not src_image.exists() or not src_label.exists():
                # Already surfaced by dataset_audit.py's missing_images/missing_labels checks.
                continue

            boxes = parse_yolo_label_file(src_label)
            valid_boxes = [b for b in boxes if b.valid]
            observed_ids.update(b.class_id for b in boxes if b.class_id != -1)

            dropped = len(boxes) - len(valid_boxes)
            report["per_class_dropped_boxes"][cls] += dropped
            if not valid_boxes:
                report["images_without_any_valid_box"].append(f"{cls}/{stem}")
                continue

            dst_image = yolo_root / "images" / split / src_image.name
            dst_label = yolo_root / "labels" / split / f"{stem}.txt"
            link_or_copy(src_image, dst_image)

            new_id = YOLO_CLASS_TO_ID[cls]
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for b in valid_boxes:
                cx = min(max(b.cx, 0.0), 1.0)
                cy = min(max(b.cy, 0.0), 1.0)
                w = min(max(b.w, 0.0), 1.0)
                h = min(max(b.h, 0.0), 1.0)
                lines.append(f"{new_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            dst_label.write_text("\n".join(lines) + "\n", encoding="utf-8")

            report["per_class_per_split_images"][cls][split] += 1
            report["per_class_written_boxes"][cls] += len(valid_boxes)

        if len(observed_ids) > 1:
            raise RuntimeError(
                f"Class folder '{cls}' contains label files with inconsistent original class ids "
                f"{observed_ids} -- refusing to remap silently. Investigate via dataset_audit.py output "
                f"(bounding_boxes.mapping_conflicts) before rerunning."
            )

    return report


def prepare_healthy_background(config: Config, manifest: pd.DataFrame, yolo_root: Path, fraction: float) -> dict:
    healthy_rows = manifest[manifest["class"] == HEALTHY]
    per_split_counts = defaultdict(lambda: {"available": 0, "included": 0})

    for split in SPLITS:
        split_rows = healthy_rows[healthy_rows["split"] == split]
        n_include = int(round(len(split_rows) * fraction))
        included = split_rows.iloc[:n_include]
        per_split_counts[split]["available"] = len(split_rows)
        per_split_counts[split]["included"] = len(included)

        for _, row in included.iterrows():
            src_image = Path(row["abs_path"])
            if not src_image.exists():
                continue
            dst_image = yolo_root / "images" / split / src_image.name
            link_or_copy(src_image, dst_image)
            # Empty label file == "no objects in this image" (Ultralytics
            # background-image convention). Never a fake box.
            dst_label = yolo_root / "labels" / split / f"{src_image.stem}.txt"
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            dst_label.write_text("", encoding="utf-8")

    return per_split_counts


def write_data_yaml(config: Config, yolo_root: Path) -> Path:
    yaml_path = Path(__file__).resolve().parent.parent / "configs" / "yolo_data.yaml"
    names_block = "\n".join(f"  {i}: {name}" for i, name in sorted(YOLO_ID_TO_CLASS.items()))
    content = f"""\
# AUTO-GENERATED by preprocessing/prepare_yolo_dataset.py -- do not hand-edit;
# rerun the script instead so this stays in sync with dataset/processed/yolo/.
#
# YOLOv8 is trained as a 5-class DISEASE detector only. Healthy images are
# present in images/{{train,val,test}}/ as background/negative examples with
# an empty label file -- Healthy is NOT a detection class (see
# preprocessing/prepare_yolo_dataset.py module docstring). Whole-image
# Healthy-vs-disease classification is handled by the CNN
# (classification/train_classifier.py), not by YOLO.

path: {yolo_root.as_posix()}
train: images/train
val: images/val
test: images/test

nc: {len(YOLO_TRAINING_CLASSES)}
names:
{names_block}
"""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Prepare the YOLOv8 disease-detection dataset.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--healthy-background-fraction", type=float, default=1.0,
        help="Fraction of each split's Healthy images to include as YOLO background/negative images "
             "(default 1.0 = all). Ultralytics' own guidance suggests background images be a modest "
             "share of the dataset (roughly 0-10%%) to avoid biasing the detector toward under-predicting; "
             "Healthy is ~16%% of this dataset, so lower this if validation shows recall degradation."
    )
    args = parser.parse_args()
    config = load_config(args.config) if args.config else load_config()

    layout = discover_layout(config.path("paths", "dataset_original"))
    manifest = load_manifest(config)

    yolo_root = config.path("paths", "yolo_dataset")
    for split in SPLITS:
        (yolo_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    print("Preparing disease-class images + remapped labels...")
    disease_report = prepare_disease_classes(config, layout, manifest, yolo_root)

    print(f"Preparing Healthy background images (fraction={args.healthy_background_fraction})...")
    healthy_report = prepare_healthy_background(config, manifest, yolo_root, args.healthy_background_fraction)

    yaml_path = write_data_yaml(config, yolo_root)

    # ---- summary ----
    total_images = 0
    print("\nImages per class per split:")
    for cls in DISEASE_CLASSES:
        row = disease_report["per_class_per_split_images"][cls]
        print(f"  {cls:15s} train={row.get('train',0):5d} val={row.get('val',0):5d} test={row.get('test',0):5d}")
        total_images += sum(row.values())
    print(f"  {'Healthy (bg)':15s} " + " ".join(
        f"{s}={healthy_report[s]['included']:5d}" for s in SPLITS
    ))
    total_images += sum(v["included"] for v in healthy_report.values())
    print(f"\nTotal images written to {yolo_root}: {total_images}")

    total_dropped = sum(disease_report["per_class_dropped_boxes"].values())
    total_written = sum(disease_report["per_class_written_boxes"].values())
    print(f"Boxes written: {total_written}  |  Boxes dropped (invalid): {total_dropped}")
    if disease_report["images_without_any_valid_box"]:
        print(f"WARNING: {len(disease_report['images_without_any_valid_box'])} disease images had "
              f"zero valid boxes after filtering and were excluded entirely -- see report JSON.")

    print(f"\nWrote {yaml_path}")

    report_path = config.path("paths", "outputs") / "yolo_dataset_prep_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "healthy_background_fraction": args.healthy_background_fraction,
            "per_class_per_split_images": {c: dict(v) for c, v in disease_report["per_class_per_split_images"].items()},
            "healthy_per_split": dict(healthy_report),
            "per_class_dropped_boxes": dict(disease_report["per_class_dropped_boxes"]),
            "per_class_written_boxes": dict(disease_report["per_class_written_boxes"]),
            "images_without_any_valid_box": disease_report["images_without_any_valid_box"],
            "total_images": total_images,
            "total_boxes_written": total_written,
            "total_boxes_dropped": total_dropped,
            "class_id_mapping": YOLO_CLASS_TO_ID,
            "yolo_data_yaml": str(yaml_path),
        }, f, indent=2)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
