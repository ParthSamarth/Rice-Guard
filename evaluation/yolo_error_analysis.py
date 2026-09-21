"""
evaluation/yolo_error_analysis.py

Diagnostic error analysis for the trained YOLOv8n detector, requested to decide
whether a YOLOv8s experiment is scientifically justified -- NOT a training script,
NOT a replacement for outputs/yolo/metrics/yolov8n_eval_test.json (the baseline
result this script reads but never modifies).

Why this doesn't just reuse Ultralytics' own confusion matrix from model.val():
that matrix is built at Ultralytics' internal validation confidence floor (used
to sweep the full precision-recall curve for mAP), which does NOT match this
project's actual deployment threshold (configs/config.yaml thresholds.yolo_confidence
= 0.35, applied in detection/predict_yolo.py). Reusing it here would silently mix
two different operating points. Instead this script:

  1. Runs the model ONCE per image at a very low confidence floor (0.01) to
     capture every candidate box Ultralytics would consider, together with its
     score -- this is cheap (one inference pass) and lets every confidence
     threshold below be swept analytically afterward without re-running the
     model.
  2. Matches predictions to ground truth with an explicit, disclosed IoU>=0.5
     greedy matching procedure, separately at the deployment threshold (0.35)
     and swept across a threshold grid (to distinguish "model never proposes a
     box here" from "model proposes it but below the operating threshold").
  3. Ties every ground-truth box to its pixel area (at both native 1024x1024
     resolution and at the model's actual training/inference resolution,
     imgsz=640) so recall can be broken down by box-size bucket.
  4. Saves annotated example images (GT vs prediction) for true positives,
     false negatives, false positives, and poor-localization cases.

Usage:
    python evaluation/yolo_error_analysis.py --weights models/yolo_yolov8n/best.pt --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import load_config  # noqa: E402
from utils.dataset_common import YOLO_ID_TO_CLASS  # noqa: E402
from utils.hardware import detect_hardware  # noqa: E402

DEPLOY_CONF = 0.35          # configs/config.yaml thresholds.yolo_confidence
NMS_IOU = 0.7                # Ultralytics default, matches detection/predict_yolo.py (no override there)
MATCH_IOU = 0.5               # standard "is this a correct detection" threshold
LOW_CONF_FLOOR = 0.01        # capture floor for the single inference pass
CONF_SWEEP = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.65]
TRAIN_IMGSZ = 640            # detection/train_yolo.py imgsz

# COCO-style area buckets, evaluated at the model's actual training/inference
# resolution (640x640, image area = 409600 px^2) rather than the native
# 1024x1024 file resolution, since that is the spatial scale the network
# itself operates at. "very_small" further splits COCO's own "small" bucket
# because this project's own findings (Section 3 of PROJECT_README.md) flag
# very small lesion boxes as a specific concern.
AREA_BUCKETS = [
    ("very_small", 0, 16 ** 2),
    ("small", 16 ** 2, 32 ** 2),
    ("medium", 32 ** 2, 96 ** 2),
    ("large", 96 ** 2, float("inf")),
]


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def area_bucket(area_640):
    for name, lo, hi in AREA_BUCKETS:
        if lo <= area_640 < hi:
            return name
    return "large"


def load_gt_boxes(label_path: Path, img_w: int, img_h: int):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append({"class_id": cls_id, "xyxy": (x1, y1, x2, y2), "img_w": img_w, "img_h": img_h})
    return boxes


def match_at_threshold(gt_boxes, preds, conf_thresh):
    """preds: list of dicts {class_id, conf, xyxy}. Greedy, highest-confidence-first,
    same-class-only IoU>=MATCH_IOU matching. Returns per-GT and per-pred outcome lists."""
    active_preds = sorted((p for p in preds if p["conf"] >= conf_thresh), key=lambda p: -p["conf"])
    gt_matched = [False] * len(gt_boxes)
    pred_outcomes = []  # ("tp" | "fp_wrong_class_localized" | "fp_hallucination" | "fp_poor_localization", pred)
    for p in active_preds:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gt_boxes):
            if gt_matched[j] or g["class_id"] != p["class_id"]:
                continue
            iou = iou_xyxy(p["xyxy"], g["xyxy"])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= MATCH_IOU:
            gt_matched[best_j] = True
            pred_outcomes.append(("tp", p, best_iou, best_j))
        else:
            # does it spatially overlap ANY gt (any class) at all?
            any_iou, any_j, any_cls_match = 0.0, -1, False
            for j, g in enumerate(gt_boxes):
                iou = iou_xyxy(p["xyxy"], g["xyxy"])
                if iou > any_iou:
                    any_iou, any_j = iou, j
                    any_cls_match = (g["class_id"] == p["class_id"])
            if any_j >= 0 and any_iou >= MATCH_IOU and not any_cls_match:
                pred_outcomes.append(("fp_wrong_class_localized", p, any_iou, any_j))
            elif any_j >= 0 and any_iou > 0.1:
                pred_outcomes.append(("fp_poor_localization", p, any_iou, any_j))
            else:
                pred_outcomes.append(("fp_hallucination", p, any_iou, any_j))
    gt_outcomes = []  # ("tp" | "fn", gt)
    for j, g in enumerate(gt_boxes):
        gt_outcomes.append(("tp" if gt_matched[j] else "fn", g))
    return gt_outcomes, pred_outcomes


def main():
    parser = argparse.ArgumentParser(description="Diagnostic error analysis for a trained YOLO checkpoint (read-only, no training).")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--max-examples-per-bucket", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N images (smoke test)")
    parser.add_argument("--out-dir", type=str, default=None,
                         help="Override output directory (default: outputs/yolo_error_analysis/). Filenames "
                              "here are NOT run-tag-namespaced (test_error_analysis.json, examples/*.png), so "
                              "a second checkpoint evaluated against the same --split would silently overwrite "
                              "a prior run's saved analysis at the default path. Pass a distinct --out-dir for "
                              "any run (e.g. an ablation) whose output must not clobber an existing baseline's.")
    args = parser.parse_args()

    from PIL import Image
    from ultralytics import YOLO

    config = load_config(args.config) if args.config else load_config()
    profile = detect_hardware()
    dataset_root = config.path("paths", "dataset_processed") / "yolo"
    img_dir = dataset_root / "images" / args.split
    lbl_dir = dataset_root / "labels" / args.split

    out_dir = Path(args.out_dir) if args.out_dir else config.path("paths", "outputs") / "yolo_error_analysis"
    examples_dir = out_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    device = 0 if profile.cuda_available else "cpu"

    image_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpeg"))
    if args.limit:
        image_paths = image_paths[:args.limit]
    print(f"Running diagnostic inference on {len(image_paths)} '{args.split}' images "
          f"(conf floor={LOW_CONF_FLOOR}, nms_iou={NMS_IOU}, weights={args.weights})")

    per_image_records = []  # for later matching/bucketing/visualization
    for i, img_path in enumerate(image_paths):
        with Image.open(img_path) as im:
            img_w, img_h = im.size
        gt_boxes = load_gt_boxes(lbl_dir / f"{img_path.stem}.txt", img_w, img_h)

        results = model.predict(source=str(img_path), conf=LOW_CONF_FLOOR, iou=NMS_IOU,
                                 device=device, verbose=False)
        preds = []
        r = results[0]
        if r.boxes is not None:
            for b in r.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                preds.append({"class_id": int(b.cls.item()), "conf": float(b.conf.item()), "xyxy": (x1, y1, x2, y2)})

        per_image_records.append({"path": str(img_path), "img_w": img_w, "img_h": img_h,
                                   "gt": gt_boxes, "preds": preds})
        if (i + 1) % 200 == 0:
            print(f"  processed {i + 1}/{len(image_paths)}")

    print("Inference pass complete. Computing matches at deployment threshold "
          f"(conf={DEPLOY_CONF}) and sweeping {CONF_SWEEP} ...")

    # ---- 1) Deployment-threshold TP/FP/FN per class, + example selection ----
    class_names = YOLO_ID_TO_CLASS
    n_classes = len(class_names)
    per_class_counts = {c: {"tp": 0, "fn": 0, "fp_hallucination": 0,
                             "fp_wrong_class_localized": 0, "fp_poor_localization": 0,
                             "gt_total": 0} for c in class_names.values()}

    # bucketed recall accumulators: {class: {bucket: [tp_count, total_count]}}
    bucket_stats = {c: {b[0]: [0, 0] for b in AREA_BUCKETS} for c in class_names.values()}

    example_pool = defaultdict(list)  # key -> list of (image_path, gt, preds, extra)

    for rec in per_image_records:
        gt_outcomes, pred_outcomes = match_at_threshold(rec["gt"], rec["preds"], DEPLOY_CONF)
        scale = (TRAIN_IMGSZ / rec["img_w"]) * (TRAIN_IMGSZ / rec["img_h"])  # area scale factor to 640-equivalent

        for outcome, g in gt_outcomes:
            cname = class_names[g["class_id"]]
            per_class_counts[cname]["gt_total"] += 1
            per_class_counts[cname][outcome if outcome == "tp" else "fn"] += 1
            x1, y1, x2, y2 = g["xyxy"]
            area_640 = (x2 - x1) * (y2 - y1) * scale
            b = area_bucket(area_640)
            bucket_stats[cname][b][1] += 1
            if outcome == "tp":
                bucket_stats[cname][b][0] += 1
            if outcome == "fn":
                example_pool[f"fn_{cname}"].append((rec, g))

        for outcome, p, iou, j in pred_outcomes:
            cname = class_names[p["class_id"]]
            if outcome == "tp":
                example_pool[f"tp_{cname}"].append((rec, p))
            else:
                per_class_counts[cname][outcome] += 1
                if outcome == "fp_hallucination":
                    example_pool[f"fp_{cname}"].append((rec, p))
                elif outcome == "fp_poor_localization":
                    example_pool[f"poorloc_{cname}"].append((rec, p))

    # TP count comes from gt_outcomes (one row per GT box), not from counting
    # "tp" pred_outcomes separately -- under this greedy 1:1 matching the two
    # are equivalent, but deriving it from the GT side is unambiguous by
    # construction and avoids any double-counting risk.
    for c in per_class_counts:
        per_class_counts[c]["tp"] = per_class_counts[c]["gt_total"] - per_class_counts[c]["fn"]

    for c, d in per_class_counts.items():
        d["precision"] = d["tp"] / (d["tp"] + d["fp_hallucination"] + d["fp_wrong_class_localized"] + d["fp_poor_localization"]) \
            if (d["tp"] + d["fp_hallucination"] + d["fp_wrong_class_localized"] + d["fp_poor_localization"]) > 0 else None
        d["recall"] = d["tp"] / d["gt_total"] if d["gt_total"] > 0 else None

    # ---- 2) Recall-vs-confidence-threshold sweep per class ----
    sweep_results = {c: {} for c in class_names.values()}
    for t in CONF_SWEEP:
        tp_count = defaultdict(int)
        gt_count = defaultdict(int)
        for rec in per_image_records:
            gt_outcomes, _ = match_at_threshold(rec["gt"], rec["preds"], t)
            for outcome, g in gt_outcomes:
                cname = class_names[g["class_id"]]
                gt_count[cname] += 1
                if outcome == "tp":
                    tp_count[cname] += 1
        for cname in class_names.values():
            sweep_results[cname][str(t)] = (tp_count[cname] / gt_count[cname]) if gt_count[cname] else None

    # ---- 3) Box size distribution (independent of detection outcome) ----
    size_distribution = {c: {b[0]: 0 for b in AREA_BUCKETS} for c in class_names.values()}
    size_distribution_pct = {c: {} for c in class_names.values()}
    for rec in per_image_records:
        scale = (TRAIN_IMGSZ / rec["img_w"]) * (TRAIN_IMGSZ / rec["img_h"])
        for g in rec["gt"]:
            cname = class_names[g["class_id"]]
            x1, y1, x2, y2 = g["xyxy"]
            area_640 = (x2 - x1) * (y2 - y1) * scale
            size_distribution[cname][area_bucket(area_640)] += 1
    for cname, buckets in size_distribution.items():
        total = sum(buckets.values())
        size_distribution_pct[cname] = {b: (buckets[b] / total if total else None) for b in buckets}

    # ---- write JSON ----
    report = {
        "weights": args.weights,
        "split": args.split,
        "methodology": {
            "deploy_conf": DEPLOY_CONF,
            "nms_iou": NMS_IOU,
            "match_iou_threshold": MATCH_IOU,
            "conf_sweep": CONF_SWEEP,
            "area_buckets_px2_at_640_scale": AREA_BUCKETS,
            "note": "All FP/FN/recall numbers here use the DEPLOYMENT confidence threshold "
                     f"(conf={DEPLOY_CONF}), matching detection/predict_yolo.py + configs/config.yaml -- "
                     "these will differ from Ultralytics' own confusion matrix in "
                     "outputs/yolo/metrics/yolov8n_eval_test.json, which is built at Ultralytics' internal "
                     "validation confidence floor, not this project's deployment threshold. Both are valid; "
                     "they answer different questions, and this file documents which one it uses.",
        },
        "per_class_deploy_threshold": per_class_counts,
        "recall_vs_confidence_threshold": sweep_results,
        "box_size_distribution_counts": size_distribution,
        "box_size_distribution_fraction": size_distribution_pct,
        "recall_by_box_size_at_deploy_threshold": {
            c: {b: (bucket_stats[c][b][0] / bucket_stats[c][b][1] if bucket_stats[c][b][1] else None)
                for b in bucket_stats[c]}
            for c in bucket_stats
        },
        "recall_by_box_size_counts": bucket_stats,
    }
    with open(out_dir / f"{args.split}_error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "recall_by_box_size_counts"}, indent=2))
    print(f"\nWrote {out_dir / f'{args.split}_error_analysis.json'}")

    # ---- 4) Render example images ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    def render(rec, highlight_box, highlight_kind, out_path, extra_label=""):
        with Image.open(rec["path"]) as im:
            fig, ax = plt.subplots(1, figsize=(8, 8))
            ax.imshow(im)
            for g in rec["gt"]:
                x1, y1, x2, y2 = g["xyxy"]
                ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                                edgecolor="lime", facecolor="none"))
                ax.text(x1, max(0, y1 - 6), f"GT:{class_names[g['class_id']]}", color="lime", fontsize=8,
                        bbox=dict(facecolor="black", alpha=0.5, pad=0))
            for p in rec["preds"]:
                if p["conf"] < DEPLOY_CONF:
                    continue
                x1, y1, x2, y2 = p["xyxy"]
                ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                                edgecolor="red", linestyle="--", facecolor="none"))
                ax.text(x1, min(rec["img_h"] - 4, y2 + 12),
                        f"pred:{class_names[p['class_id']]} {p['conf']:.2f}", color="red", fontsize=8,
                        bbox=dict(facecolor="black", alpha=0.5, pad=0))
            ax.set_title(f"{highlight_kind} | {Path(rec['path']).name} {extra_label}", fontsize=9)
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(out_path, dpi=110)
            plt.close(fig)

    saved = []
    for key, items in example_pool.items():
        kind = key.split("_", 1)[0]
        items_sorted = items
        if kind == "fp":
            items_sorted = sorted(items, key=lambda t: -t[1]["conf"])  # highest-confidence hallucinations first
        for idx, (rec, obj) in enumerate(items_sorted[:args.max_examples_per_bucket]):
            out_path = examples_dir / f"{key}_{idx}_{Path(rec['path']).stem}.png"
            render(rec, obj, key, out_path)
            saved.append(str(out_path))

    print(f"Saved {len(saved)} annotated example images to {examples_dir}")


if __name__ == "__main__":
    main()
