"""
detection/evaluate_yolo.py

Formal evaluation of a trained YOLO checkpoint via Ultralytics' own
validator: mAP@50, mAP@50-95, precision, recall, per-class AP/P/R, and the
confusion matrix Ultralytics computes internally. Written defensively
against Ultralytics API drift (its metrics-object attributes have changed
across versions) -- every extraction is try/except'd so a missing attribute
degrades to "unavailable in this Ultralytics version" rather than crashing
the whole evaluation.

Usage:
    python detection/evaluate_yolo.py --weights models/yolo_yolov8n/best.pt --split test
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import load_config  # noqa: E402
from utils.hardware import detect_hardware  # noqa: E402


def safe_get(obj, attr, default=None):
    try:
        val = getattr(obj, attr)
        return val() if callable(val) else val
    except Exception:  # noqa: BLE001
        return default


def to_jsonable(x):
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
    except ImportError:
        pass
    return x


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO checkpoint.")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    from ultralytics import YOLO

    config = load_config(args.config) if args.config else load_config()
    profile = detect_hardware()
    data_yaml = Path(__file__).resolve().parent.parent / "configs" / "yolo_data.yaml"

    model = YOLO(args.weights)
    run_tag = Path(args.weights).parent.name.replace("yolo_", "")
    project_dir = config.path("paths", "outputs") / "yolo" / "runs"

    metrics = model.val(
        data=str(data_yaml),
        split=args.split,
        device=0 if profile.cuda_available else "cpu",
        project=str(project_dir),
        name=f"{run_tag}_eval_{args.split}",
        exist_ok=True,
        plots=True,
        save_json=True,
    )

    results_dict = {k: to_jsonable(v) for k, v in (safe_get(metrics, "results_dict", {}) or {}).items()}

    box = safe_get(metrics, "box")
    overall = {
        "map50": to_jsonable(safe_get(box, "map50")),
        "map50_95": to_jsonable(safe_get(box, "map")),
        "map75": to_jsonable(safe_get(box, "map75")),
        "mean_precision": to_jsonable(safe_get(box, "mp")),
        "mean_recall": to_jsonable(safe_get(box, "mr")),
    }

    names = safe_get(metrics, "names", {}) or {}
    per_class = {}
    ap_class_idx = safe_get(box, "ap_class_index")
    ap50_95_per_class = safe_get(box, "maps")  # per-class mAP50-95, len == nc
    p_per_class = safe_get(box, "p")
    r_per_class = safe_get(box, "r")
    ap50_per_class = None
    try:
        # `box.all_ap` is [n_classes, n_iou_thresholds] in most Ultralytics
        # versions; column 0 is the IoU=0.50 slice.
        all_ap = safe_get(box, "all_ap")
        if all_ap is not None:
            ap50_per_class = [row[0] for row in all_ap]
    except Exception:  # noqa: BLE001
        ap50_per_class = None

    try:
        for i, cls_idx in enumerate(ap_class_idx if ap_class_idx is not None else []):
            cname = names.get(int(cls_idx), str(cls_idx)) if isinstance(names, dict) else str(cls_idx)
            entry = {"mAP50_95": to_jsonable(ap50_95_per_class[i]) if ap50_95_per_class is not None else None}
            if ap50_per_class is not None and i < len(ap50_per_class):
                entry["mAP50"] = to_jsonable(ap50_per_class[i])
            if p_per_class is not None and i < len(p_per_class):
                entry["precision"] = to_jsonable(p_per_class[i])
            if r_per_class is not None and i < len(r_per_class):
                entry["recall"] = to_jsonable(r_per_class[i])
            per_class[cname] = entry
    except Exception as exc:  # noqa: BLE001
        per_class = {"error": f"per-class metric extraction failed on this Ultralytics version: {exc}"}

    confusion_matrix = None
    try:
        cm_obj = safe_get(metrics, "confusion_matrix")
        cm_matrix = safe_get(cm_obj, "matrix")
        if cm_matrix is not None:
            confusion_matrix = to_jsonable(cm_matrix)
    except Exception:  # noqa: BLE001
        pass

    speed = {k: to_jsonable(v) for k, v in (safe_get(metrics, "speed", {}) or {}).items()}

    report = {
        "weights": args.weights,
        "split": args.split,
        "class_names": names if isinstance(names, dict) else {i: n for i, n in enumerate(names)},
        "overall": overall,
        "results_dict": results_dict,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix,
        "speed_ms_per_image": speed,
    }

    out_dir = config.path("paths", "outputs") / "yolo" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_tag}_eval_{args.split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({"overall": overall, "per_class": per_class}, indent=2))
    print(f"\nWrote {out_path}")

    # Stage Ultralytics' own plots (PR curve, F1 curve, confusion matrix png)
    # next to the rest of this project's outputs for easy inclusion in the
    # final report.
    save_dir = safe_get(metrics, "save_dir")
    if save_dir:
        plots_dir = config.path("paths", "outputs") / "yolo" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        for png in Path(save_dir).glob("*.png"):
            try:
                shutil.copy2(png, plots_dir / f"{run_tag}_{args.split}_{png.name}")
            except OSError:
                pass


if __name__ == "__main__":
    main()
