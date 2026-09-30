"""
evaluation/generate_yolo_plots.py

Generates the full YOLO visualization suite for the FINAL detector
(models/yolo_yolov8n/best.pt, imgsz=640) from already-saved, already-verified
artifacts -- never from hand-typed numbers:

  - Confusion matrix (raw + normalized)      <- outputs/yolo/metrics/yolov8n_eval_test.json
  - Per-class mAP50 / mAP50-95 / P / R       <- outputs/yolo/metrics/yolov8n_eval_test.json
  - Training curves (10 metrics vs step)     <- outputs/yolo/runs/yolov8n/results.csv
  - PR curve, F1/P/R-vs-confidence           <- a fresh model.val() forward pass on the
                                                 SAME test split (read-only inference, not
                                                 training), whose scalar outputs are asserted
                                                 to match yolov8n_eval_test.json exactly before
                                                 any curve is drawn from it -- Ultralytics never
                                                 persists the raw sweep arrays behind its own
                                                 curve PNGs, so this is the only way to get them
                                                 without re-deriving/guessing the numbers.

"mAP50 vs confidence" was requested but is deliberately NOT produced: mAP is defined as an
integral over the full precision-recall curve (already swept across every confidence
threshold internally), so "mAP at one confidence value" is not a standard or meaningful
quantity -- Ultralytics does not expose it, and inventing a computation for it here would
violate the "never fabricate a metric" rule this project runs under. The PR curve and the
F1-vs-confidence curve (which IS how Ultralytics picks its own best-confidence operating
point) are generated instead and serve the same diagnostic purpose.

Never overwrites outputs/yolo/{runs,plots,metrics}/ or outputs/yolo_error_analysis/ -- every
file this script writes lives under outputs/visualizations/yolo/.

Usage:
    python evaluation/generate_yolo_plots.py
    python main.py visualize-yolo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import load_config  # noqa: E402
from evaluation.plot_style import (  # noqa: E402
    YOLO_CLASSES, CLASS_COLORS, ACCENT_FINAL,
    plot_confusion_matrix, bar_chart, line_chart, new_figure, save_fig, annotate_source,
)

DEFAULT_WEIGHTS = "models/yolo_yolov8n/best.pt"
DEFAULT_EVAL_JSON = "outputs/yolo/metrics/yolov8n_eval_test.json"
DEFAULT_RESULTS_CSV = "outputs/yolo/runs/yolov8n/results.csv"


def _class_colors(names):
    return [CLASS_COLORS[n] for n in names]


def plot_confusion_matrices(eval_json: dict, out_dir: Path, source: str):
    names = eval_json["class_names"]  # {"0": "Blast", ...} -- 5 real classes
    ordered = [names[str(i)] for i in range(len(names))]
    labels = ordered + ["background"]
    matrix = eval_json["confusion_matrix"]
    if len(matrix) != len(labels):
        raise ValueError(f"confusion matrix is {len(matrix)}x{len(matrix[0])} but "
                          f"{len(labels)} labels (incl. background) were expected -- "
                          f"class list may have changed; refusing to mislabel axes.")
    plot_confusion_matrix(
        matrix, labels, labels, out_dir / "confusion_matrix_raw",
        title="YOLOv8n @ 640 -- confusion matrix (test split, counts)",
        source=source, normalize=False,
    )
    plot_confusion_matrix(
        matrix, labels, labels, out_dir / "confusion_matrix_normalized",
        title="YOLOv8n @ 640 -- confusion matrix (test split, % of true class)",
        source=source, normalize=True,
    )


def plot_per_class_bars(eval_json: dict, out_dir: Path, source: str):
    per_class = eval_json["per_class"]
    classes = [c for c in YOLO_CLASSES if c in per_class]
    colors = _class_colors(classes)
    specs = [
        ("mAP50", "mAP50", "mAP50 per class -- YOLOv8n @ 640 (test split)"),
        ("mAP50_95", "mAP50-95", "mAP50-95 per class -- YOLOv8n @ 640 (test split)"),
        ("precision", "Precision", "Precision per class -- YOLOv8n @ 640 (test split)"),
        ("recall", "Recall", "Recall per class -- YOLOv8n @ 640 (test split)"),
    ]
    for key, ylabel, title in specs:
        values = [per_class[c][key] for c in classes]
        bar_chart(classes, values, out_dir / f"per_class_{key.lower()}", title, ylabel, source,
                  colors=colors, ylim=(0, max(0.5, max(values) * 1.25)), rotate_xticks=20)


def plot_training_curves(results_csv: Path, out_dir: Path, source: str):
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    step = np.arange(1, len(df) + 1)
    duplicated_epochs = df["epoch"].duplicated().sum() > 0
    subtitle = (
        "x-axis is the logged training step, not the raw epoch number -- this run's epoch\n"
        "column contains real duplicates from mid-training resumes; see PROJECT_README.md §7.1"
    ) if duplicated_epochs else None

    curve_specs = [
        ("train/box_loss", "train_box_loss", "Training box loss vs. step", "Box loss"),
        ("train/cls_loss", "train_cls_loss", "Training classification loss vs. step", "Cls loss"),
        ("train/dfl_loss", "train_dfl_loss", "Training DFL loss vs. step", "DFL loss"),
        ("val/box_loss", "val_box_loss", "Validation box loss vs. step", "Box loss"),
        ("val/cls_loss", "val_cls_loss", "Validation classification loss vs. step", "Cls loss"),
        ("val/dfl_loss", "val_dfl_loss", "Validation DFL loss vs. step", "DFL loss"),
        ("metrics/mAP50(B)", "map50", "mAP50 vs. step (validation split)", "mAP50"),
        ("metrics/mAP50-95(B)", "map50_95", "mAP50-95 vs. step (validation split)", "mAP50-95"),
        ("metrics/precision(B)", "precision", "Precision vs. step (validation split)", "Precision"),
        ("metrics/recall(B)", "recall", "Recall vs. step (validation split)", "Recall"),
    ]
    xlabel = "Logged training step (1..%d)" % len(df) if duplicated_epochs else "Epoch"
    for col, fname, title, ylabel in curve_specs:
        if col not in df.columns:
            print(f"  WARNING: column '{col}' not found in {results_csv}, skipping {fname}")
            continue
        line_chart(step, {ylabel: df[col].values}, out_dir / fname, title, xlabel, ylabel, source,
                   colors={ylabel: ACCENT_FINAL}, subtitle=subtitle)


def plot_confidence_curves(weights: str, data_yaml: Path, out_dir: Path, eval_json: dict, source: str):
    from ultralytics import YOLO
    from utils.hardware import detect_hardware

    profile = detect_hardware()
    print("  Re-running model.val() on the test split to obtain curve arrays "
          "(read-only inference -- not training; Ultralytics does not persist these "
          "arrays anywhere else). Cross-checking against the saved eval JSON before plotting...")
    model = YOLO(weights)
    metrics = model.val(data=str(data_yaml), split="test",
                         device=0 if profile.cuda_available else "cpu",
                         plots=False, save_json=False, verbose=False)
    box = metrics.box

    saved = eval_json["overall"]
    live = {"map50": box.map50, "map50_95": box.map, "map75": box.map75,
            "mean_precision": box.mp, "mean_recall": box.mr}
    mismatches = [k for k in live if not np.isclose(live[k], saved[k], rtol=1e-6, atol=1e-9)]
    if mismatches:
        raise RuntimeError(
            f"Live model.val() re-run does not match the saved evaluation JSON on {mismatches} "
            f"(live={ {k: live[k] for k in mismatches} }, saved={ {k: saved[k] for k in mismatches} }). "
            f"This means the checkpoint, dataset, or evaluation settings have drifted since "
            f"{DEFAULT_EVAL_JSON} was written -- refusing to plot confidence curves from a run "
            f"that doesn't reproduce the authoritative numbers. Re-run detection/evaluate_yolo.py "
            f"and investigate before re-trying this script."
        )
    print(f"  OK -- live re-run reproduces the saved test-set metrics exactly (mAP50={live['map50']:.6f}).")

    names = metrics.names  # {0: 'Blast', ...}
    class_order = [names[i] for i in sorted(names)]
    colors = _class_colors(class_order)
    px = np.asarray(box.px)

    curve_map = {(x[2], x[3]): x for x in box.curves_results}  # (xlabel, ylabel) -> (x, y, xlabel, ylabel)

    def _plot_one(xlabel, ylabel, fname, title, x_is_recall=False):
        key = (xlabel, ylabel)
        if key not in curve_map:
            print(f"  WARNING: curve '{xlabel} vs {ylabel}' not found in this Ultralytics version, skipping.")
            return
        x, y, _, _ = curve_map[key]
        x = np.asarray(x)
        y = np.asarray(y)  # (n_classes, n_points)
        fig, ax = new_figure()
        for i, cname in enumerate(class_order):
            ax.plot(x, y[i], label=cname, color=colors[i], linewidth=1.6)
        mean_y = y.mean(axis=0)
        ax.plot(x, mean_y, label="all classes (mean)", color="#1A2420", linewidth=2.4, linestyle="--")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1)
        if not x_is_recall:
            ax.set_xlim(0, 1)
        ax.set_title(title, loc="left")
        ax.legend(loc="best", fontsize=8)
        annotate_source(ax, source + " (curve arrays from a verified-identical model.val() re-run)")
        fig.tight_layout()
        save_fig(fig, out_dir / fname)

    _plot_one("Recall", "Precision", "precision_recall_curve",
               "Precision-Recall curve -- YOLOv8n @ 640 (test split)", x_is_recall=True)
    _plot_one("Confidence", "F1", "f1_vs_confidence",
               "F1 vs. confidence threshold -- YOLOv8n @ 640 (test split)")
    _plot_one("Confidence", "Precision", "precision_vs_confidence",
               "Precision vs. confidence threshold -- YOLOv8n @ 640 (test split)")
    _plot_one("Confidence", "Recall", "recall_vs_confidence",
               "Recall vs. confidence threshold -- YOLOv8n @ 640 (test split)")


def main():
    parser = argparse.ArgumentParser(description="Generate the full YOLO visualization suite for the final 640 detector.")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS,
                         help=f"Default: {DEFAULT_WEIGHTS} (the final selected detector). Only override for "
                              f"a deliberate one-off comparison -- this does not change which model is 'final'.")
    parser.add_argument("--eval-json", type=str, default=DEFAULT_EVAL_JSON)
    parser.add_argument("--results-csv", type=str, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--skip-confidence-curves", action="store_true",
                         help="Skip the model.val() re-run (confusion matrix / per-class / training curves "
                              "still run, all from already-saved files).")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"{weights_path} not found -- this must be the final 640 detector checkpoint.")

    eval_json_path = Path(args.eval_json)
    if not eval_json_path.exists():
        raise FileNotFoundError(f"{eval_json_path} not found -- run detection/evaluate_yolo.py first.")
    with open(eval_json_path, "r", encoding="utf-8") as f:
        eval_json = json.load(f)
    if eval_json.get("weights") and Path(eval_json["weights"]).name != weights_path.name:
        print(f"  NOTE: {eval_json_path} was generated from {eval_json['weights']!r}, "
              f"plotting against --weights {weights_path} -- verify these are meant to match.")

    out_dir = config.path("paths", "outputs") / "visualizations" / "yolo"
    out_dir.mkdir(parents=True, exist_ok=True)
    source = f"source: {eval_json_path.as_posix()}"

    print(f"Weights: {weights_path}")
    print(f"Writing YOLO visualizations to {out_dir}")

    print("Confusion matrices (raw, normalized)...")
    plot_confusion_matrices(eval_json, out_dir, source)

    print("Per-class bar charts...")
    plot_per_class_bars(eval_json, out_dir, source)

    results_csv_path = Path(args.results_csv)
    if results_csv_path.exists():
        print(f"Training curves from {results_csv_path}...")
        plot_training_curves(results_csv_path, out_dir, f"source: {results_csv_path.as_posix()}")
    else:
        print(f"  WARNING: {results_csv_path} not found, skipping training curves.")

    if not args.skip_confidence_curves:
        print("Confidence-sweep curves (PR / F1 / P / R vs. confidence)...")
        data_yaml = Path(__file__).resolve().parent.parent / "configs" / "yolo_data.yaml"
        plot_confidence_curves(str(weights_path), data_yaml, out_dir, eval_json, source)

    print(f"\nDone. YOLO visualizations written to {out_dir}")
    print("mAP50-vs-confidence intentionally NOT generated -- see this script's module docstring for why.")


if __name__ == "__main__":
    main()
