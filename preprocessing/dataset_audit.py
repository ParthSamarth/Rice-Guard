"""
preprocessing/dataset_audit.py

Full integrity + statistics audit of the RiceLeafDiseaseBD dataset, run
against the ACTUAL extracted archive (dataset/original/RiceLeafDiseaseBD/).
Must be run before any training -- nothing downstream (YOLO dataset prep,
classification dataset prep, training) should assume anything about the
dataset that this script hasn't verified.

What it checks (see project brief section 3):
    - total images / images per class
    - annotated vs unannotated images
    - total label files, total bounding boxes, bounding boxes per class
    - missing images (label exists, image doesn't) / missing labels (image
      exists, label doesn't) for the 5 disease classes
    - duplicate filenames (within a class, and across the whole dataset)
    - duplicate image content (exact sha256 hash) and near-duplicate
      candidates (perceptual hash collisions -- see the "limitations" field
      in the output for what this does and doesn't catch)
    - corrupted images (fails to fully decode)
    - invalid YOLO annotation lines, out-of-range coordinates, zero/negative
      box dimensions
    - image pixel dimensions
    - class imbalance
    - boxes-per-image distribution (min/max/mean/median/std, overall and
      per class)
    - the ACTUAL class-ID -> class-name mapping, read directly from the
      label files and cross-checked against the documented one in README.md

Cross-checks everything against `Dataset metadata.xlsx` (which ships beside
the zip, not inside it -- see DATASET_ANALYSIS.md section 7 discrepancy #3).

Outputs:
    outputs/dataset_audit.json      -- full structured report
    outputs/dataset_statistics.csv  -- one row per class, key stats
    outputs/audit_details/*.csv     -- full lists of any issues found (only
                                        written for non-empty issue lists)
    outputs/plots/audit_*.png       -- distribution plots

Usage:
    python preprocessing/dataset_audit.py
    python preprocessing/dataset_audit.py --config configs/config.yaml
    python preprocessing/dataset_audit.py --skip-image-scan   # fast rerun,
        text-only checks (labels/metadata), skips the corruption/hash/
        dimension pass over all 9,769 images
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import load_config, Config  # noqa: E402
from utils.dataset_common import (  # noqa: E402
    ALL_CLASSES,
    DISEASE_CLASSES,
    DOCUMENTED_YOLO_ID_TO_CLASS,
    HEALTHY,
    discover_layout,
    labels_dir_of,
    list_images,
    normalize_class_name,
    parse_yolo_label_file,
    visuals_dir_of,
)

try:
    import imagehash

    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

MAX_INLINE_ISSUES = 200  # cap on items listed inline in the JSON; full lists go to CSV


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------
# 1. Original-image inventory (per class)
# --------------------------------------------------------------------------

def scan_original_images(layout) -> Dict[str, List[Path]]:
    per_class = {}
    for cls in ALL_CLASSES:
        directory = layout.original_class_dirs.get(cls)
        per_class[cls] = list_images(directory) if directory else []
    return per_class


# --------------------------------------------------------------------------
# 2. Annotation inventory + parsing (per disease class)
# --------------------------------------------------------------------------

def scan_annotations(layout):
    """Returns:
        labels_by_class: {class: {stem: Path}}
        visuals_by_class: {class: {stem: Path}}
        parsed_by_class: {class: {stem: [ParsedBox, ...]}}
    """
    labels_by_class, visuals_by_class, parsed_by_class = {}, {}, {}
    for cls in DISEASE_CLASSES:
        ann_dir = layout.annotated_class_dirs.get(cls)
        labels_by_class[cls] = {}
        visuals_by_class[cls] = {}
        parsed_by_class[cls] = {}
        if ann_dir is None:
            continue
        labels_dir = labels_dir_of(ann_dir)
        visuals_dir = visuals_dir_of(ann_dir)
        if labels_dir:
            for p in sorted(labels_dir.glob("*.txt")):
                labels_by_class[cls][p.stem] = p
                parsed_by_class[cls][p.stem] = parse_yolo_label_file(p)
        if visuals_dir:
            for p in list_images(visuals_dir):
                visuals_by_class[cls][p.stem] = p
    return labels_by_class, visuals_by_class, parsed_by_class


# --------------------------------------------------------------------------
# 3. Per-image pass: corruption check, dimensions, exact hash, perceptual hash
# --------------------------------------------------------------------------

def scan_image_contents(per_class_images: Dict[str, List[Path]]):
    dim_counter: Counter = Counter()
    width_list, height_list = [], []
    corrupted: List[dict] = []
    sha_to_files: Dict[str, List[str]] = defaultdict(list)
    phash_to_files: Dict[str, List[str]] = defaultdict(list)
    per_image_record: List[dict] = []

    total = sum(len(v) for v in per_class_images.values())
    done = 0
    t0 = time.time()

    for cls, paths in per_class_images.items():
        for p in paths:
            done += 1
            if done % 1000 == 0:
                elapsed = time.time() - t0
                log(f"  scanned {done}/{total} images ({elapsed:.0f}s elapsed)")
            rel = str(p)
            try:
                data = p.read_bytes()
            except OSError as exc:
                corrupted.append({"path": rel, "class": cls, "error": f"unreadable: {exc}"})
                continue

            sha = sha256_bytes(data)
            sha_to_files[sha].append(rel)

            try:
                img = Image.open(io.BytesIO(data))
                img.load()
                w, h = img.size
                mode = img.mode
            except Exception as exc:  # noqa: BLE001 - genuinely want to catch any decode failure
                corrupted.append({"path": rel, "class": cls, "error": str(exc)})
                continue

            dim_counter[(w, h)] += 1
            width_list.append(w)
            height_list.append(h)

            phash_val = None
            if HAS_IMAGEHASH:
                try:
                    phash_val = str(imagehash.phash(img))
                    phash_to_files[phash_val].append(rel)
                except Exception:  # noqa: BLE001
                    pass

            per_image_record.append({
                "path": rel, "class": cls, "width": w, "height": h,
                "mode": mode, "sha256": sha, "phash": phash_val,
                "file_size_bytes": len(data),
            })

    log(f"  finished scanning {total} images in {time.time() - t0:.0f}s")

    exact_dupes = {h: fs for h, fs in sha_to_files.items() if len(fs) > 1}
    perceptual_dupe_candidates = {h: fs for h, fs in phash_to_files.items() if len(fs) > 1}

    return {
        "dim_counter": dim_counter,
        "widths": width_list,
        "heights": height_list,
        "corrupted": corrupted,
        "exact_duplicate_groups": exact_dupes,
        "perceptual_duplicate_candidate_groups": perceptual_dupe_candidates,
        "per_image_record": per_image_record,
    }


# --------------------------------------------------------------------------
# 4. Filename-collision checks
# --------------------------------------------------------------------------

def check_duplicate_filenames(per_class_images: Dict[str, List[Path]]):
    within_class = {}
    for cls, paths in per_class_images.items():
        names = [p.name.lower() for p in paths]
        dupes = [n for n, c in Counter(names).items() if c > 1]
        if dupes:
            within_class[cls] = dupes

    global_name_map: Dict[str, List[str]] = defaultdict(list)
    for cls, paths in per_class_images.items():
        for p in paths:
            global_name_map[p.name.lower()].append(f"{cls}/{p.name}")
    global_dupes = {n: v for n, v in global_name_map.items() if len(v) > 1}

    return within_class, global_dupes


# --------------------------------------------------------------------------
# 5. Missing image / missing label cross-checks (disease classes only)
# --------------------------------------------------------------------------

def check_missing_pairs(per_class_images, labels_by_class, visuals_by_class):
    missing_labels = {}   # image exists, no label
    missing_images = {}   # label exists, no image
    label_visual_mismatch = {}

    for cls in DISEASE_CLASSES:
        image_stems = {p.stem for p in per_class_images.get(cls, [])}
        label_stems = set(labels_by_class.get(cls, {}).keys())
        visual_stems = set(visuals_by_class.get(cls, {}).keys())

        no_label = sorted(image_stems - label_stems)
        no_image = sorted(label_stems - image_stems)
        if no_label:
            missing_labels[cls] = no_label
        if no_image:
            missing_images[cls] = no_image
        if label_stems != visual_stems:
            label_visual_mismatch[cls] = {
                "labels_without_visual": sorted(label_stems - visual_stems),
                "visuals_without_label": sorted(visual_stems - label_stems),
            }

    return missing_labels, missing_images, label_visual_mismatch


# --------------------------------------------------------------------------
# 6. Bounding-box statistics + class-ID mapping verification
# --------------------------------------------------------------------------

def analyze_boxes(parsed_by_class):
    per_class_box_count = {}
    per_class_valid_box_count = {}
    per_class_invalid_lines = {}
    per_class_boxes_per_image = {}
    invalid_line_details = []
    out_of_range_details = []
    zero_neg_details = []
    observed_class_ids: Dict[str, Counter] = defaultdict(Counter)

    for cls, stem_map in parsed_by_class.items():
        total_boxes, total_valid = 0, 0
        boxes_per_image = []
        invalid_count = 0
        for stem, boxes in stem_map.items():
            boxes_per_image.append(len(boxes))
            total_boxes += len(boxes)
            for b in boxes:
                if b.class_id != -1:
                    observed_class_ids[cls][b.class_id] += 1
                if b.valid:
                    total_valid += 1
                else:
                    invalid_count += 1
                    detail = {"class": cls, "file": stem, "line": b.raw_line, "issues": b.issues}
                    invalid_line_details.append(detail)
                    if any("outside" in i for i in b.issues):
                        out_of_range_details.append(detail)
                    if any("non-positive" in i for i in b.issues):
                        zero_neg_details.append(detail)

        per_class_box_count[cls] = total_boxes
        per_class_valid_box_count[cls] = total_valid
        per_class_invalid_lines[cls] = invalid_count
        per_class_boxes_per_image[cls] = boxes_per_image

    all_boxes_per_image = [n for lst in per_class_boxes_per_image.values() for n in lst]

    def stats(lst):
        if not lst:
            return {"min": None, "max": None, "mean": None, "median": None, "std": None}
        arr = np.array(lst, dtype=float)
        return {
            "min": int(arr.min()), "max": int(arr.max()),
            "mean": round(float(arr.mean()), 3), "median": float(np.median(arr)),
            "std": round(float(arr.std()), 3),
        }

    boxes_per_image_stats_overall = stats(all_boxes_per_image)
    boxes_per_image_stats_per_class = {c: stats(v) for c, v in per_class_boxes_per_image.items()}

    # Verify class-ID mapping: does every label file under class folder X use
    # a single, consistent YOLO class_id?
    verified_mapping = {}
    mapping_conflicts = {}
    for cls, id_counter in observed_class_ids.items():
        if len(id_counter) == 1:
            verified_mapping[next(iter(id_counter))] = cls
        else:
            mapping_conflicts[cls] = dict(id_counter)

    matches_documented = all(
        verified_mapping.get(k) == v for k, v in DOCUMENTED_YOLO_ID_TO_CLASS.items()
        if v != HEALTHY  # Healthy never appears in disease label files by construction
    ) and len(mapping_conflicts) == 0 and len(verified_mapping) == len(DISEASE_CLASSES)

    return {
        "total_bounding_box_lines": sum(per_class_box_count.values()),
        "total_valid_boxes": sum(per_class_valid_box_count.values()),
        "total_invalid_lines": sum(per_class_invalid_lines.values()),
        "per_class_box_count": per_class_box_count,
        "per_class_valid_box_count": per_class_valid_box_count,
        "per_class_invalid_line_count": per_class_invalid_lines,
        "boxes_per_image_overall": boxes_per_image_stats_overall,
        "boxes_per_image_per_class": boxes_per_image_stats_per_class,
        "invalid_line_details": invalid_line_details,
        "out_of_range_details": out_of_range_details,
        "zero_or_negative_details": zero_neg_details,
        "observed_class_ids_per_folder": {c: dict(v) for c, v in observed_class_ids.items()},
        "verified_class_id_to_name": verified_mapping,
        "documented_class_id_to_name": DOCUMENTED_YOLO_ID_TO_CLASS,
        "mapping_conflicts": mapping_conflicts,
        "matches_documented": matches_documented,
    }


# --------------------------------------------------------------------------
# 7. Metadata spreadsheet cross-check
# --------------------------------------------------------------------------

def cross_check_metadata(config: Config, per_class_images: Dict[str, List[Path]]):
    xlsx_path = config.path("paths", "dataset_metadata_xlsx")
    if not xlsx_path.exists():
        return {"available": False, "reason": f"{xlsx_path} not found"}

    try:
        img_meta = pd.read_excel(xlsx_path, sheet_name="Image_Metadata")
        class_summary = pd.read_excel(xlsx_path, sheet_name="Class_Summary_Statistics")
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"failed to parse workbook: {exc}"}

    img_meta["_canonical_class"] = img_meta["Class"].astype(str).map(normalize_class_name)
    unmapped = img_meta[img_meta["_canonical_class"].isna()]["Class"].unique().tolist()

    counts_metadata = img_meta["_canonical_class"].value_counts().to_dict()

    # Build actual-file name sets per class for coverage check.
    actual_names = {cls: {p.name for p in paths} for cls, paths in per_class_images.items()}

    rows_without_file = []
    for _, row in img_meta.iterrows():
        cls = row["_canonical_class"]
        fname = row.get("New_Filename")
        if cls is None or not isinstance(fname, str):
            continue
        if fname not in actual_names.get(cls, set()):
            rows_without_file.append({"class": cls, "expected_filename": fname})

    files_without_metadata_row = {}
    meta_names_by_class = defaultdict(set)
    for _, row in img_meta.iterrows():
        cls = row["_canonical_class"]
        fname = row.get("New_Filename")
        if cls is not None and isinstance(fname, str):
            meta_names_by_class[cls].add(fname)
    for cls, names in actual_names.items():
        missing = sorted(names - meta_names_by_class.get(cls, set()))
        if missing:
            files_without_metadata_row[cls] = missing

    return {
        "available": True,
        "total_rows": int(len(img_meta)),
        "unmapped_class_labels_found": unmapped,
        "row_count_per_class": counts_metadata,
        "class_summary_sheet": class_summary.to_dict(orient="records"),
        "metadata_rows_with_no_matching_file": rows_without_file[:MAX_INLINE_ISSUES],
        "metadata_rows_with_no_matching_file_count": len(rows_without_file),
        "files_with_no_metadata_row": {k: v[:MAX_INLINE_ISSUES] for k, v in files_without_metadata_row.items()},
        "files_with_no_metadata_row_count": {k: len(v) for k, v in files_without_metadata_row.items()},
    }


# --------------------------------------------------------------------------
# 8. Recommended split preview (image-level, stratified per class)
# --------------------------------------------------------------------------

def recommended_split_preview(images_per_class: Dict[str, int], ratios: dict):
    preview = {}
    for cls, n in images_per_class.items():
        train_n = int(round(n * ratios["train"]))
        val_n = int(round(n * ratios["val"]))
        test_n = n - train_n - val_n
        preview[cls] = {"total": n, "train": train_n, "val": val_n, "test": test_n}
    return preview


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def make_plots(report: dict, plots_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    classes = ALL_CLASSES

    # 1. Images per class
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = [report["per_class"][c]["images_original"] for c in classes]
    ax.bar(classes, counts, color="#4C72B0")
    ax.set_title("Original images per class")
    ax.set_ylabel("Image count")
    plt.xticks(rotation=30, ha="right")
    for i, v in enumerate(counts):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(plots_dir / "audit_images_per_class.png", dpi=150)
    plt.close(fig)

    # 2. Bounding boxes per class
    fig, ax = plt.subplots(figsize=(8, 5))
    disease = DISEASE_CLASSES
    box_counts = [report["bounding_boxes"]["per_class_box_count"].get(c, 0) for c in disease]
    ax.bar(disease, box_counts, color="#C44E52")
    ax.set_title("Bounding boxes per class")
    ax.set_ylabel("Box count")
    plt.xticks(rotation=30, ha="right")
    for i, v in enumerate(box_counts):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(plots_dir / "audit_boxes_per_class.png", dpi=150)
    plt.close(fig)

    # 3. Boxes-per-image histogram (overall)
    all_counts = []
    for c in disease:
        all_counts.extend(report["_raw"]["per_class_boxes_per_image"].get(c, []))
    if all_counts:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(all_counts, bins=range(0, max(all_counts) + 2), color="#55A868", edgecolor="black")
        ax.set_title("Boxes per image (all disease images)")
        ax.set_xlabel("Number of boxes in image")
        ax.set_ylabel("Number of images")
        plt.tight_layout()
        fig.savefig(plots_dir / "audit_boxes_per_image_hist.png", dpi=150)
        plt.close(fig)

    # 4. Boxes-per-image boxplot per class
    data = [report["_raw"]["per_class_boxes_per_image"].get(c, []) for c in disease]
    if any(data):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(data, tick_labels=disease, showfliers=True)
        ax.set_title("Boxes-per-image distribution by class")
        ax.set_ylabel("Boxes per image")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        fig.savefig(plots_dir / "audit_boxes_per_image_boxplot.png", dpi=150)
        plt.close(fig)

    # 5. Image dimension scatter (width vs height); degenerate if all 1024x1024
    dims = report["_raw"].get("dim_counter")
    if dims:
        fig, ax = plt.subplots(figsize=(6, 6))
        ws = [w for (w, h) in dims for _ in range(dims[(w, h)])]
        hs = [h for (w, h) in dims for _ in range(dims[(w, h)])]
        ax.scatter(ws, hs, alpha=0.3, s=10)
        ax.set_title("Image dimensions (width vs height)")
        ax.set_xlabel("Width (px)")
        ax.set_ylabel("Height (px)")
        plt.tight_layout()
        fig.savefig(plots_dir / "audit_image_dimensions.png", dpi=150)
        plt.close(fig)

    # 6. Healthy vs Disease proportion
    healthy_n = report["per_class"][HEALTHY]["images_original"]
    disease_n = sum(report["per_class"][c]["images_original"] for c in DISEASE_CLASSES)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie([healthy_n, disease_n], labels=[f"Healthy ({healthy_n})", f"Disease ({disease_n})"],
           autopct="%1.1f%%", colors=["#55A868", "#C44E52"])
    ax.set_title("Healthy vs. Disease image proportion")
    plt.tight_layout()
    fig.savefig(plots_dir / "audit_healthy_vs_disease.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def run_audit(config: Config, skip_image_scan: bool = False) -> dict:
    t_start = time.time()
    log("Locating dataset root...")
    layout = discover_layout(config.path("paths", "dataset_original"))
    log(f"Dataset root: {layout.root}")
    log(f"Original images dir: {layout.original_dir.name}")
    log(f"Annotated images dir: {layout.annotated_dir.name}")

    log("Scanning original image folders...")
    per_class_images = scan_original_images(layout)
    images_per_class = {c: len(v) for c, v in per_class_images.items()}
    log(f"  images per class: {images_per_class}")

    log("Scanning annotation (label/visual) folders and parsing YOLO labels...")
    labels_by_class, visuals_by_class, parsed_by_class = scan_annotations(layout)
    label_file_counts = {c: len(v) for c, v in labels_by_class.items()}
    visual_file_counts = {c: len(v) for c, v in visuals_by_class.items()}

    log("Checking duplicate filenames...")
    dup_within_class, dup_global = check_duplicate_filenames(per_class_images)

    log("Checking missing image/label pairs...")
    missing_labels, missing_images, label_visual_mismatch = check_missing_pairs(
        per_class_images, labels_by_class, visuals_by_class
    )

    log("Analyzing bounding boxes and verifying class-ID mapping...")
    box_analysis = analyze_boxes(parsed_by_class)

    log("Cross-checking against Dataset metadata.xlsx...")
    metadata_check = cross_check_metadata(config, per_class_images)

    if skip_image_scan:
        log("Skipping full image-content scan (--skip-image-scan).")
        image_scan = {
            "dim_counter": Counter(), "widths": [], "heights": [], "corrupted": [],
            "exact_duplicate_groups": {}, "perceptual_duplicate_candidate_groups": {},
            "per_image_record": [],
        }
    else:
        log(f"Scanning pixel content of {sum(images_per_class.values())} images "
            f"(corruption / dimensions / exact + perceptual hash)...")
        image_scan = scan_image_contents(per_class_images)

    annotated_images_total = sum(label_file_counts.values())
    total_images = sum(images_per_class.values())

    per_class_report = {}
    for cls in ALL_CLASSES:
        is_disease = cls in DISEASE_CLASSES
        per_class_report[cls] = {
            "images_original": images_per_class.get(cls, 0),
            "images_metadata_xlsx": metadata_check.get("row_count_per_class", {}).get(cls) if metadata_check.get("available") else None,
            "annotated": is_disease,
            "label_files": label_file_counts.get(cls, 0) if is_disease else 0,
            "visual_files": visual_file_counts.get(cls, 0) if is_disease else 0,
            "bounding_boxes": box_analysis["per_class_box_count"].get(cls, 0) if is_disease else 0,
            "valid_bounding_boxes": box_analysis["per_class_valid_box_count"].get(cls, 0) if is_disease else 0,
        }

    max_cls = max(images_per_class, key=images_per_class.get)
    min_cls = min(images_per_class, key=images_per_class.get)
    class_imbalance = {
        "max_class": max_cls, "max_count": images_per_class[max_cls],
        "min_class": min_cls, "min_count": images_per_class[min_cls],
        "imbalance_ratio": round(images_per_class[max_cls] / max(images_per_class[min_cls], 1), 3),
        "mean_count": round(float(np.mean(list(images_per_class.values()))), 2),
        "std_count": round(float(np.std(list(images_per_class.values()))), 2),
    }

    widths, heights = image_scan["widths"], image_scan["heights"]
    dim_report = {
        "distinct_sizes": {f"{w}x{h}": n for (w, h), n in image_scan["dim_counter"].most_common()},
        "num_distinct_sizes": len(image_scan["dim_counter"]),
        "width_stats": {
            "min": int(np.min(widths)) if widths else None,
            "max": int(np.max(widths)) if widths else None,
            "mean": round(float(np.mean(widths)), 2) if widths else None,
        },
        "height_stats": {
            "min": int(np.min(heights)) if heights else None,
            "max": int(np.max(heights)) if heights else None,
            "mean": round(float(np.mean(heights)), 2) if heights else None,
        },
        "non_1024_count": sum(n for (w, h), n in image_scan["dim_counter"].items() if (w, h) != (1024, 1024)),
    }

    limitations = [
        "Duplicate-content detection uses exact SHA-256 file hashing, which only catches byte-identical "
        "files (e.g. the same photo saved/copied twice). It will NOT catch near-duplicates such as "
        "the same leaf photographed twice in quick succession, at a slightly different angle, or under "
        "slightly different lighting -- which is a realistic risk for a field-collected smartphone dataset.",
        "As a cheap proxy for near-duplicates, a perceptual hash (pHash) is computed per image and grouped "
        "by EXACT pHash equality only. This catches near-identical images that hash to the same 64-bit "
        "value but will miss visually similar images whose hashes differ by even one bit. A full pairwise "
        "comparison (e.g. Hamming distance <= k between every pair) was not performed because it is "
        "O(n^2) over ~9,769 images (~47.7 million pairs) and was judged not worth the compute cost for an "
        "audit script; if stricter leakage control is needed later, use a proper ANN/LSH index (e.g. "
        "`imagehash` + a BK-tree, or a vector index over CNN embeddings) instead of brute-force pairs."
        if HAS_IMAGEHASH else
        "The `imagehash` package was not installed when this audit ran, so NO perceptual/near-duplicate "
        "check was performed at all -- only exact byte-identical duplicates (SHA-256) were checked. "
        "Install `imagehash` and rerun for near-duplicate candidate detection.",
        "Corruption checking opens and fully decodes every image with Pillow; it will catch truncated "
        "files and decode errors but cannot catch subtler issues (e.g. wrong-but-valid-looking pixel data).",
    ]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generation_time_seconds": round(time.time() - t_start, 1),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "imagehash_available": HAS_IMAGEHASH,
        },
        "dataset_root": str(layout.root),
        "original_images_dir": layout.original_dir.name,
        "annotated_images_dir": layout.annotated_dir.name,
        "summary": {
            "total_images_original": total_images,
            "num_classes": len(ALL_CLASSES),
            "healthy_images": images_per_class.get(HEALTHY, 0),
            "disease_images": total_images - images_per_class.get(HEALTHY, 0),
            "annotated_images": annotated_images_total,
            "unannotated_images": total_images - annotated_images_total,
            "total_label_files": annotated_images_total,
            "total_bounding_box_lines": box_analysis["total_bounding_box_lines"],
            "total_valid_bounding_boxes": box_analysis["total_valid_boxes"],
            "total_invalid_annotation_lines": box_analysis["total_invalid_lines"],
            "corrupted_images": len(image_scan["corrupted"]),
            "exact_duplicate_image_groups": len(image_scan["exact_duplicate_groups"]),
            "perceptual_duplicate_candidate_groups": len(image_scan["perceptual_duplicate_candidate_groups"]),
        },
        "per_class": per_class_report,
        "class_imbalance": class_imbalance,
        "image_dimensions": dim_report,
        "bounding_boxes": {k: v for k, v in box_analysis.items()
                            if k not in ("invalid_line_details", "out_of_range_details", "zero_or_negative_details")},
        "integrity_issues": {
            "duplicate_filenames_within_class": dup_within_class,
            "duplicate_filenames_across_classes": dup_global,
            "missing_labels_image_has_no_label": {c: v[:MAX_INLINE_ISSUES] for c, v in missing_labels.items()},
            "missing_labels_count": {c: len(v) for c, v in missing_labels.items()},
            "missing_images_label_has_no_image": {c: v[:MAX_INLINE_ISSUES] for c, v in missing_images.items()},
            "missing_images_count": {c: len(v) for c, v in missing_images.items()},
            "label_visual_count_mismatch": label_visual_mismatch,
            "corrupted_images": image_scan["corrupted"][:MAX_INLINE_ISSUES],
            "corrupted_images_count": len(image_scan["corrupted"]),
            "invalid_annotation_lines_sample": box_analysis["invalid_line_details"][:MAX_INLINE_ISSUES],
            "invalid_annotation_lines_count": len(box_analysis["invalid_line_details"]),
            "out_of_range_coordinate_boxes_sample": box_analysis["out_of_range_details"][:MAX_INLINE_ISSUES],
            "out_of_range_coordinate_boxes_count": len(box_analysis["out_of_range_details"]),
            "zero_or_negative_dimension_boxes_sample": box_analysis["zero_or_negative_details"][:MAX_INLINE_ISSUES],
            "zero_or_negative_dimension_boxes_count": len(box_analysis["zero_or_negative_details"]),
            "exact_duplicate_image_groups_sample": dict(list(image_scan["exact_duplicate_groups"].items())[:MAX_INLINE_ISSUES]),
            "exact_duplicate_image_groups_count": len(image_scan["exact_duplicate_groups"]),
            "perceptual_duplicate_candidate_groups_sample": dict(list(image_scan["perceptual_duplicate_candidate_groups"].items())[:MAX_INLINE_ISSUES]),
            "perceptual_duplicate_candidate_groups_count": len(image_scan["perceptual_duplicate_candidate_groups"]),
        },
        "metadata_cross_check": metadata_check,
        "recommended_split_preview": recommended_split_preview(images_per_class, config["split"]),
        "limitations": limitations,
        "_raw": {
            "per_class_boxes_per_image": {
                c: [len(b) for b in parsed_by_class.get(c, {}).values()] for c in DISEASE_CLASSES
            },
            "dim_counter": image_scan["dim_counter"],
            "per_image_record": image_scan["per_image_record"],
        },
    }
    return report


def write_outputs(report: dict, config: Config):
    outputs_dir = config.path("paths", "outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = outputs_dir / "plots"
    details_dir = outputs_dir / "audit_details"

    # Full per-image record -> its own CSV (useful for prepare_*.py scripts too)
    if report["_raw"]["per_image_record"]:
        details_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(report["_raw"]["per_image_record"]).to_csv(
            details_dir / "per_image_scan.csv", index=False
        )

    # Any non-empty issue list also gets its own full CSV (JSON keeps a capped sample).
    issue_lists = {
        "duplicate_filenames_within_class": report["integrity_issues"]["duplicate_filenames_within_class"],
        "duplicate_filenames_across_classes": report["integrity_issues"]["duplicate_filenames_across_classes"],
        "missing_labels": report["integrity_issues"]["missing_labels_image_has_no_label"],
        "missing_images": report["integrity_issues"]["missing_images_label_has_no_image"],
    }
    details_written = []
    for name, obj in issue_lists.items():
        if obj:
            details_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for k, v in obj.items():
                for item in (v if isinstance(v, list) else [v]):
                    rows.append({"class": k, "item": item})
            if rows:
                pd.DataFrame(rows).to_csv(details_dir / f"{name}.csv", index=False)
                details_written.append(name)

    if report["integrity_issues"]["invalid_annotation_lines_count"] > 0:
        details_dir.mkdir(parents=True, exist_ok=True)
        # re-derive full list isn't stored past the sample cap by design (kept in JSON sample);
        # note: for a full un-capped export, rerun with MAX_INLINE_ISSUES raised.

    # dataset_statistics.csv -- one row per class
    rows = []
    for cls in ALL_CLASSES:
        c = report["per_class"][cls]
        bpi = report["bounding_boxes"]["boxes_per_image_per_class"].get(cls, {})
        rows.append({
            "class": cls,
            "images_original": c["images_original"],
            "images_metadata_xlsx": c["images_metadata_xlsx"],
            "annotated": c["annotated"],
            "label_files": c["label_files"],
            "bounding_boxes": c["bounding_boxes"],
            "valid_bounding_boxes": c["valid_bounding_boxes"],
            "min_boxes_per_image": bpi.get("min"),
            "max_boxes_per_image": bpi.get("max"),
            "mean_boxes_per_image": bpi.get("mean"),
        })
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(outputs_dir / "dataset_statistics.csv", index=False)

    # dataset_audit.json -- strip the bulky _raw section before writing
    json_report = {k: v for k, v in report.items() if k != "_raw"}

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, Counter):
            return dict(o)
        raise TypeError(f"Not JSON serializable: {type(o)}")

    with open(outputs_dir / "dataset_audit.json", "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2, default=default)

    log("Generating plots...")
    make_plots(report, plots_dir)

    log(f"Wrote {outputs_dir / 'dataset_audit.json'}")
    log(f"Wrote {outputs_dir / 'dataset_statistics.csv'}")
    if details_written:
        log(f"Wrote detail CSVs for: {details_written} -> {details_dir}")
    log(f"Wrote plots -> {plots_dir}")


def main():
    parser = argparse.ArgumentParser(description="Audit the RiceLeafDiseaseBD dataset.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--skip-image-scan", action="store_true",
                         help="Skip the per-image corruption/hash/dimension pass (fast rerun of label-only checks).")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    report = run_audit(config, skip_image_scan=args.skip_image_scan)
    write_outputs(report, config)

    log("=== AUDIT SUMMARY ===")
    log(json.dumps(report["summary"], indent=2))
    log(f"Class mapping matches documented README mapping: {report['bounding_boxes']['matches_documented']}")


if __name__ == "__main__":
    main()
