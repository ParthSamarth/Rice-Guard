"""
evaluation/generate_classifier_plots.py

Generates the full ResNet50 classifier visualization suite for the FINAL
checkpoint (models/classifier_resnet50/best.pt) from already-saved,
already-verified artifacts only:

  - Confusion matrix (raw + normalized), 6 classes  <- outputs/classification/resnet50/evaluation/metrics_test.json
  - Training curves (5 metrics vs epoch)            <- outputs/classification/resnet50/train_summary.json:history
  - Precision / recall / F1 / support per class     <- metrics_test.json:per_class
  - Full-image vs. YOLO-crop comparison              <- metrics_test.json:by_sample_type

Class order is fixed everywhere in this script to
Healthy, Blast, Brown Spot, Leaf Smut, Rice Tungro, Sheath Blight (matches
classification/dataset.py:CLASS_TO_IDX and the order metrics_test.json's own
confusion_matrix_labels already uses).

Never overwrites outputs/classification/resnet50/evaluation/ -- every file
this script writes lives under outputs/visualizations/classification/.

Usage:
    python evaluation/generate_classifier_plots.py
    python main.py visualize-classifier
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import load_config  # noqa: E402
from evaluation.plot_style import (  # noqa: E402
    CNN_CLASSES, CLASS_COLORS, ACCENT_FINAL, ACCENT_PARTIAL,
    plot_confusion_matrix, bar_chart, line_chart, new_figure, save_fig, annotate_source,
)

DEFAULT_METRICS_JSON = "outputs/classification/resnet50/evaluation/metrics_test.json"
DEFAULT_TRAIN_SUMMARY = "outputs/classification/resnet50/train_summary.json"


def _colors(classes):
    return [CLASS_COLORS[c] for c in classes]


def plot_confusion_matrices(metrics: dict, out_dir: Path, source: str):
    labels = metrics["confusion_matrix_labels"]
    if labels != CNN_CLASSES:
        print(f"  NOTE: confusion_matrix_labels order in the JSON ({labels}) differs from this "
              f"script's expected order ({CNN_CLASSES}) -- plotting in the JSON's own order, "
              f"unchanged, rather than silently reindexing.")
    matrix = metrics["confusion_matrix"]
    plot_confusion_matrix(
        matrix, labels, labels, out_dir / "confusion_matrix_raw",
        title="ResNet50 -- confusion matrix (test split, all samples, counts)",
        source=source, normalize=False, row_axis_label="Predicted", col_axis_label="True label",
    )
    plot_confusion_matrix(
        matrix, labels, labels, out_dir / "confusion_matrix_normalized",
        title="ResNet50 -- confusion matrix (test split, all samples, % of true class)",
        source=source, normalize=True, row_axis_label="Predicted", col_axis_label="True label",
    )


def plot_training_curves(history: list, out_dir: Path, source: str):
    epochs = [h["epoch"] for h in history]
    specs = [
        ("train_loss", "train_loss", "Training loss vs. epoch -- ResNet50", "Loss", ACCENT_FINAL),
        ("val_loss", "val_loss", "Validation loss vs. epoch -- ResNet50", "Loss", ACCENT_PARTIAL),
        ("train_acc", "train_acc", "Training accuracy vs. epoch -- ResNet50", "Accuracy", ACCENT_FINAL),
        ("val_acc", "val_acc", "Validation accuracy vs. epoch -- ResNet50", "Accuracy", ACCENT_PARTIAL),
        ("val_macro_f1", "val_macro_f1", "Validation macro-F1 vs. epoch -- ResNet50", "Macro-F1", ACCENT_FINAL),
    ]
    for key, fname, title, ylabel, color in specs:
        values = [h[key] for h in history]
        line_chart(epochs, {ylabel: values}, out_dir / fname, title, "Epoch", ylabel, source,
                   colors={ylabel: color}, markers=True)


def plot_per_class_metrics(metrics: dict, out_dir: Path, source: str):
    per_class = metrics["per_class"]
    classes = [c for c in CNN_CLASSES if c in per_class]
    colors = _colors(classes)
    for key, ylabel, title, fmt in [
        ("precision", "Precision", "Precision per class -- ResNet50 (test split)", "{:.3f}"),
        ("recall", "Recall", "Recall per class -- ResNet50 (test split)", "{:.3f}"),
        ("f1", "F1", "F1 per class -- ResNet50 (test split)", "{:.3f}"),
    ]:
        values = [per_class[c][key] for c in classes]
        bar_chart(classes, values, out_dir / f"per_class_{key}", title, ylabel, source,
                  colors=colors, ylim=(0, 1.05), rotate_xticks=20, value_fmt=fmt)
    supports = [per_class[c]["support"] for c in classes]
    bar_chart(classes, supports, out_dir / "per_class_support", "Support (test-set sample count) per class",
               "# samples", source, colors=colors, rotate_xticks=20, value_fmt="{:.0f}")


def plot_full_vs_crop(metrics: dict, out_dir: Path, source: str):
    by_type = metrics["by_sample_type"]
    full, crop = by_type["full"], by_type["crop"]

    fig, ax = new_figure(figsize=(7.5, 5.5))
    modes = ["Full image", "YOLO crop"]
    x = [0, 1]
    width = 0.32
    acc_vals = [full["accuracy"], crop["accuracy"]]
    f1_vals = [full["macro_f1"], crop["macro_f1"]]
    ax.bar([xi - width / 2 for xi in x], acc_vals, width, label="Accuracy", color=ACCENT_FINAL)
    ax.bar([xi + width / 2 for xi in x], f1_vals, width, label="Macro-F1", color=ACCENT_PARTIAL)
    for xi, v in zip(x, acc_vals):
        ax.text(xi - width / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9.5)
    for xi, v in zip(x, f1_vals):
        ax.text(xi + width / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n(n={by_type[k]['num_samples']:,})" for m, k in zip(modes, ["full", "crop"])])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Full-image vs. YOLO-crop input -- ResNet50 (test split)", loc="left")
    ax.legend(loc="upper right")
    annotate_source(ax, source)
    fig.tight_layout()
    save_fig(fig, out_dir / "full_vs_crop_accuracy_macrof1")

    classes = [c for c in CNN_CLASSES if c in full["per_class"] and c in crop["per_class"]]
    fig, ax = new_figure(figsize=(9, 5.5))
    xi = range(len(classes))
    width = 0.36
    full_recall = [full["per_class"][c]["recall"] for c in classes]
    crop_recall = [crop["per_class"][c]["recall"] for c in classes]
    ax.bar([x - width / 2 for x in xi], full_recall, width, label="Full image", color=ACCENT_FINAL)
    ax.bar([x + width / 2 for x in xi], crop_recall, width, label="YOLO crop", color=ACCENT_PARTIAL)
    ax.set_xticks(list(xi))
    ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall")
    ax.set_title("Per-class recall: full-image vs. YOLO-crop -- ResNet50 (test split)", loc="left")
    ax.legend(loc="upper right")
    annotate_source(ax, source)
    fig.tight_layout()
    save_fig(fig, out_dir / "full_vs_crop_per_class_recall")


def main():
    parser = argparse.ArgumentParser(description="Generate the full ResNet50 classifier visualization suite.")
    parser.add_argument("--metrics-json", type=str, default=DEFAULT_METRICS_JSON)
    parser.add_argument("--train-summary", type=str, default=DEFAULT_TRAIN_SUMMARY)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()

    metrics_path = Path(args.metrics_json)
    if not metrics_path.exists():
        raise FileNotFoundError(f"{metrics_path} not found -- run classification/evaluate_classifier.py first.")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    summary_path = Path(args.train_summary)
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} not found.")
    with open(summary_path, "r", encoding="utf-8") as f:
        train_summary = json.load(f)

    checkpoint = train_summary.get("checkpoint_best", "")
    if checkpoint and "classifier_resnet50" not in str(checkpoint):
        print(f"  NOTE: {summary_path} points to checkpoint {checkpoint!r}, which is not "
              f"models/classifier_resnet50/ -- verify this is meant to be the final classifier.")

    out_dir = config.path("paths", "outputs") / "visualizations" / "classification"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_source = f"source: {metrics_path.as_posix()}"
    history_source = f"source: {summary_path.as_posix()}"

    print(f"Checkpoint: {checkpoint or metrics.get('checkpoint', DEFAULT_METRICS_JSON)}")
    print(f"Writing classifier visualizations to {out_dir}")

    print("Confusion matrices (raw, normalized)...")
    plot_confusion_matrices(metrics, out_dir, metrics_source)

    print("Training curves...")
    history = train_summary.get("history")
    if history:
        plot_training_curves(history, out_dir, history_source)
    else:
        print(f"  WARNING: no 'history' list in {summary_path}, skipping training curves.")

    print("Per-class classification metrics (precision/recall/F1/support)...")
    plot_per_class_metrics(metrics, out_dir, metrics_source)

    print("Full-image vs. YOLO-crop comparison...")
    if "by_sample_type" in metrics:
        plot_full_vs_crop(metrics, out_dir, metrics_source)
    else:
        print(f"  WARNING: no 'by_sample_type' in {metrics_path}, skipping full-vs-crop comparison.")

    print(f"\nDone. Classifier visualizations written to {out_dir}")


if __name__ == "__main__":
    main()
