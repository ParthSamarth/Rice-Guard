"""
Shared dataset-discovery, class-normalization, and YOLO-label-parsing helpers
for the RiceLeafDiseaseBD project.

The archive on disk does NOT match the illustrative structure/names described
in the dataset's own README (see DATASET_ANALYSIS.md, section 7, for the full
list of discrepancies). Concretely, on disk we have:

    dataset/original/RiceLeafDiseaseBD/
        Original images/
            Blast/                 blast_1.jpg, blast_2.jpg, ...
            Brown spot/            brown spot_1.jpg, ...
            Healthy/                Healthy_1.jpg, ...
            Leaf smut/              Leaf smut_1.jpg, ...
            Rice Tungro/            Rice Tungro_1.jpg, ...
            Sheath blight/          Sheath blight_1.jpg, ...
        Annotated images ( visual with labels)/
            Blast/labels/*.txt, Blast/visuals/*.jpg
            Brown spot/labels/*.txt, Brown spot/visuals/*.jpg
            Leaf smut/...
            Rice Tungro/...
            Sheath blight/...
            (no Healthy/ subfolder here -- Healthy has no bounding boxes)

Every function here discovers folders by normalized name rather than by exact
string match, so stray spacing/casing in the archive does not silently break
downstream scripts.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Canonical class names used EVERYWHERE in this project's code, plots and
# JSON outputs. All other spellings encountered on disk / in the metadata
# spreadsheet are normalized to one of these six strings.
HEALTHY = "Healthy"
BLAST = "Blast"
BROWN_SPOT = "Brown Spot"
LEAF_SMUT = "Leaf Smut"
RICE_TUNGRO = "Rice Tungro"
SHEATH_BLIGHT = "Sheath Blight"

ALL_CLASSES = [HEALTHY, BLAST, BROWN_SPOT, LEAF_SMUT, RICE_TUNGRO, SHEATH_BLIGHT]
DISEASE_CLASSES = [BLAST, BROWN_SPOT, LEAF_SMUT, RICE_TUNGRO, SHEATH_BLIGHT]

# YOLOv8 detects disease regions ONLY (Healthy has no bounding boxes -- see
# project brief section 4). These are NEW, contiguous 0..4 ids used for
# training/inference, alphabetically ordered for determinism. They are
# DELIBERATELY different from the original dataset's 0/1/3/4/5 YOLO ids
# (which reserve slot 2 for Healthy) -- Ultralytics requires contiguous
# class ids matching exactly the classes present in the label files, and
# Healthy is architecturally excluded from detection. This remapping is
# applied explicitly and only inside dataset/processed/yolo/ -- the original
# label files under dataset/original/ are never modified.
YOLO_TRAINING_CLASSES = list(DISEASE_CLASSES)
YOLO_CLASS_TO_ID = {name: i for i, name in enumerate(YOLO_TRAINING_CLASSES)}
YOLO_ID_TO_CLASS = {i: name for name, i in YOLO_CLASS_TO_ID.items()}

# The class-ID -> class-name mapping as DOCUMENTED in README.md /
# Annotation Protocol.pdf. dataset_audit.py independently VERIFIES this by
# reading the actual label files; do not trust this constant without that
# verification (see outputs/dataset_audit.json -> class_mapping).
DOCUMENTED_YOLO_ID_TO_CLASS = {
    0: BROWN_SPOT,
    1: BLAST,
    2: HEALTHY,
    3: LEAF_SMUT,
    4: RICE_TUNGRO,
    5: SHEATH_BLIGHT,
}

# Lookup table from every raw spelling we've observed (folder names, metadata
# xlsx `Class` column, filename prefixes) to the canonical class name.
# Keys are matched case-insensitively after stripping whitespace.
_RAW_TO_CANONICAL = {
    "healthy": HEALTHY,
    "blast": BLAST,
    "brown spot": BROWN_SPOT,
    "brown_spot": BROWN_SPOT,
    "brownspot": BROWN_SPOT,
    "leaf smut": LEAF_SMUT,
    "leaf_smut": LEAF_SMUT,
    "leafsmut": LEAF_SMUT,
    "rice tungro": RICE_TUNGRO,
    "rice_tungro": RICE_TUNGRO,
    "tungro": RICE_TUNGRO,
    "sheath blight": SHEATH_BLIGHT,
    "sheath_blight": SHEATH_BLIGHT,
    "sheathblight": SHEATH_BLIGHT,
}


def normalize_class_name(raw: str) -> Optional[str]:
    """Map any observed spelling/casing of a class name to its canonical form.

    Returns None if `raw` does not match any known class (caller decides
    whether that is an error worth raising or just logging).
    """
    key = re.sub(r"\s+", " ", raw.strip()).lower()
    return _RAW_TO_CANONICAL.get(key)


def find_dataset_root(search_base: Path) -> Path:
    """Locate the extracted `RiceLeafDiseaseBD/` directory under `search_base`.

    Searches for a directory that itself contains a subfolder starting with
    "original images" (case-insensitive) -- this is robust to the archive
    being extracted directly, or nested one level deeper (e.g. because the
    zip's top-level folder was itself named RiceLeafDiseaseBD).
    """
    search_base = Path(search_base)
    candidates = [search_base] + [p for p in search_base.rglob("*") if p.is_dir()]
    for cand in candidates:
        try:
            children = [c.name.lower() for c in cand.iterdir() if c.is_dir()]
        except (FileNotFoundError, PermissionError):
            continue
        has_original = any(c.startswith("original images") for c in children)
        has_annotated = any(c.startswith("annotated images") for c in children)
        if has_original and has_annotated:
            return cand
    raise FileNotFoundError(
        f"Could not locate an extracted RiceLeafDiseaseBD/ folder under {search_base}. "
        "Run the zip extraction step first."
    )


def _find_child_starting_with(parent: Path, prefix: str) -> Optional[Path]:
    prefix = prefix.lower()
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower().startswith(prefix):
            return child
    return None


@dataclass
class DatasetLayout:
    root: Path
    original_dir: Path
    annotated_dir: Path
    # canonical class name -> directory holding that class's original images
    original_class_dirs: Dict[str, Path] = field(default_factory=dict)
    # canonical disease class name -> directory holding labels/ + visuals/
    annotated_class_dirs: Dict[str, Path] = field(default_factory=dict)


def discover_layout(search_base: Path) -> DatasetLayout:
    root = find_dataset_root(search_base)
    original_dir = _find_child_starting_with(root, "original images")
    annotated_dir = _find_child_starting_with(root, "annotated images")
    if original_dir is None or annotated_dir is None:
        raise FileNotFoundError(f"Expected 'Original images' and 'Annotated images...' under {root}")

    original_class_dirs: Dict[str, Path] = {}
    for child in sorted(original_dir.iterdir()):
        if not child.is_dir():
            continue
        canon = normalize_class_name(child.name)
        if canon is None:
            continue
        original_class_dirs[canon] = child

    annotated_class_dirs: Dict[str, Path] = {}
    for child in sorted(annotated_dir.iterdir()):
        if not child.is_dir():
            continue
        canon = normalize_class_name(child.name)
        if canon is None:
            continue
        annotated_class_dirs[canon] = child

    return DatasetLayout(
        root=root,
        original_dir=original_dir,
        annotated_dir=annotated_dir,
        original_class_dirs=original_class_dirs,
        annotated_class_dirs=annotated_class_dirs,
    )


def expand_and_clip_box(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int,
                         margin: float = 0.15):
    """Convert a normalized YOLO box (cx, cy, w, h in [0,1]) to pixel
    coordinates, expand it by `margin` (as a fraction of the box's own size)
    on each side for surrounding context, then clip to the image bounds.

    Returns (x0, y0, x1, y1) integer pixel coordinates.

    This is used identically at training-crop-generation time
    (preprocessing/prepare_classification_dataset.py) and at inference time
    (pipeline/inference_pipeline.py) so the CNN verifier always sees the
    same *kind* of crop it was trained on -- if these two call sites ever
    drift apart, the classifier will see a train/inference distribution
    shift on every YOLO-triggered prediction.
    """
    x_min = (cx - w / 2) * img_w
    x_max = (cx + w / 2) * img_w
    y_min = (cy - h / 2) * img_h
    y_max = (cy + h / 2) * img_h
    pad_w = margin * (x_max - x_min)
    pad_h = margin * (y_max - y_min)
    x0 = max(0, int(round(x_min - pad_w)))
    y0 = max(0, int(round(y_min - pad_h)))
    x1 = min(img_w, int(round(x_max + pad_w)))
    y1 = min(img_h, int(round(y_max + pad_h)))
    return x0, y0, x1, y1


def find_exact_duplicate_keep_map(outputs_dir: Path) -> dict:
    """Reuses preprocessing/dataset_audit.py's per-image SHA-256 scan
    (outputs/audit_details/per_image_scan.csv) rather than re-hashing 9,769
    images again. For every group of byte-identical files, keeps the first
    (by whatever order the scan produced) and marks the rest for exclusion.

    Returns {"keep": {abs_path: bool}, "dropped": [{"kept":..., "dropped":..., "sha256":...}]}
    or {} if the audit hasn't been run yet.

    IMPORTANT: this must be applied in preprocessing/make_split_manifest.py,
    BEFORE the train/val/test split is drawn -- not only in the
    classification dataset builder. Deduplicating only at the classification
    stage still leaves the YOLO dataset (which is built straight from the
    split manifest) exposed to duplicate-group members landing in different
    splits, including the cross-class duplicates -- exactly the leakage this
    function exists to prevent.
    """
    import pandas as pd

    per_image_csv = outputs_dir / "audit_details" / "per_image_scan.csv"
    if not per_image_csv.exists():
        return {}

    df = pd.read_csv(per_image_csv)
    keep: Dict[str, bool] = {}
    dropped = []
    for sha, group in df.groupby("sha256"):
        paths = group["path"].tolist()
        keep[paths[0]] = True
        for p in paths[1:]:
            keep[p] = False
            dropped.append({"kept": paths[0], "dropped": p, "sha256": sha})
    return {"keep": keep, "dropped": dropped}


def link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink `src` to `dst` (instant, zero extra disk space -- safe here
    because nothing in this project ever writes into dataset/processed
    images after creation) falling back to a real copy if hardlinking isn't
    possible (e.g. crossing filesystems). Never touches `src`.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def list_images(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def labels_dir_of(annotated_class_dir: Path) -> Optional[Path]:
    return _find_child_starting_with(annotated_class_dir, "label")


def visuals_dir_of(annotated_class_dir: Path) -> Optional[Path]:
    return _find_child_starting_with(annotated_class_dir, "visual")


@dataclass
class ParsedBox:
    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    raw_line: str
    valid: bool
    issues: List[str] = field(default_factory=list)


def parse_yolo_label_file(path: Path) -> List[ParsedBox]:
    """Parse one YOLO-format .txt annotation file, flagging malformed lines
    rather than raising, so a single bad line doesn't hide the rest of the
    file's boxes from the audit.
    """
    boxes: List[ParsedBox] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        boxes.append(ParsedBox(-1, 0, 0, 0, 0, "", False, [f"unreadable file: {exc}"]))
        return boxes

    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        issues: List[str] = []
        if len(tokens) != 5:
            issues.append(f"line {line_no}: expected 5 tokens, got {len(tokens)}")
            boxes.append(ParsedBox(-1, 0, 0, 0, 0, stripped, False, issues))
            continue
        try:
            class_id = int(tokens[0])
        except ValueError:
            issues.append(f"line {line_no}: non-integer class id {tokens[0]!r}")
            boxes.append(ParsedBox(-1, 0, 0, 0, 0, stripped, False, issues))
            continue
        try:
            cx, cy, w, h = (float(t) for t in tokens[1:])
        except ValueError:
            issues.append(f"line {line_no}: non-numeric coordinate in {tokens[1:]!r}")
            boxes.append(ParsedBox(class_id, 0, 0, 0, 0, stripped, False, issues))
            continue

        valid = True
        if w <= 0 or h <= 0:
            issues.append(f"line {line_no}: non-positive width/height (w={w}, h={h})")
            valid = False
        eps = 1e-6
        for name, val in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
            if val < -eps or val > 1 + eps:
                issues.append(f"line {line_no}: {name}={val} outside [0,1]")
                valid = False
        x_min, x_max = cx - w / 2, cx + w / 2
        y_min, y_max = cy - h / 2, cy + h / 2
        if x_min < -eps or x_max > 1 + eps or y_min < -eps or y_max > 1 + eps:
            issues.append(
                f"line {line_no}: box extends outside image bounds "
                f"(x=[{x_min:.4f},{x_max:.4f}], y=[{y_min:.4f},{y_max:.4f}])"
            )
            valid = False

        boxes.append(ParsedBox(class_id, cx, cy, w, h, stripped, valid, issues))

    return boxes
