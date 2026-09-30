"""
evaluation/generate_pipeline_plots.py

Generates the integrated-pipeline visualization suite from the ALREADY-SAVED
full test-set evaluation -- does not rerun the 1,458-image pipeline (only
reads outputs/pipeline_evaluation/pipeline_eval_test.csv +
pipeline_eval_test_summary.json, both already verified in PROJECT_README.md §7.5).

  - Confusion matrix (raw + normalized)   <- built from the per-image CSV's
                                              (ground_truth, predicted) columns,
                                              restricted to status == "ok" rows
                                              (the 1,339 cases where the system
                                              actually committed to an answer --
                                              see the module docstring note below
                                              on why abstained rows are excluded
                                              here rather than forced into a class).
  - Per-class accuracy/precision/recall/F1 <- same CSV subset, via sklearn.
  - Decision-status chart                  <- summary JSON: status_breakdown.
  - Detection-path chart                   <- CSV: decision_path counts.
  - Accuracy comparison (5 conditions)     <- each number read from its OWN
                                              source file (classifier metrics_test.json
                                              for the 3 ResNet50-standalone numbers,
                                              pipeline summary JSON for the 2 integrated
                                              numbers) -- never hand-typed.

Confusion matrix note: `predicted` is null for model_disagreement/low_confidence
rows by construction (the pipeline declined to answer). A confusion matrix has
no cell for "no prediction was made," so this script builds it only over the
1,339 confident (status=="ok") rows -- exactly the same subset
accuracy_when_confident (0.909) is computed over -- and reports the abstained
8.2% separately via the decision-status chart, rather than either dropping them
silently or forcing them into a specific wrong class that was never predicted.

Never overwrites outputs/pipeline_evaluation/ -- every file this script writes
lives under outputs/visualizations/pipeline/.

Usage:
    python evaluation/generate_pipeline_plots.py
    python main.py visualize-pipeline
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
    CNN_CLASSES, CLASS_COLORS, STATUS_COLORS, ACCENT_FINAL, ACCENT_COND, ACCENT_PARTIAL,
    plot_confusion_matrix, bar_chart, new_figure, save_fig, annotate_source,
)

DEFAULT_CSV = "outputs/pipeline_evaluation/pipeline_eval_test.csv"
DEFAULT_SUMMARY = "outputs/pipeline_evaluation/pipeline_eval_test_summary.json"
DEFAULT_CLASSIFIER_METRICS = "outputs/classification/resnet50/evaluation/metrics_test.json"


def build_confident_subset(df: pd.DataFrame) -> pd.DataFrame:
    confident = df[df["status"] == "ok"].copy()
    confident = confident[confident["predicted"].notna()]
    return confident


def plot_confusion_matrices(confident: pd.DataFrame, out_dir: Path, source: str):
    from sklearn.metrics import confusion_matrix as sk_cm

    labels = CNN_CLASSES
    cm = sk_cm(confident["ground_truth"], confident["predicted"], labels=labels)
    # sklearn's confusion_matrix is [true_row, pred_col]; this project's shared
    # plot_confusion_matrix helper expects [pred_row, true_col] (the Ultralytics
    # convention already used for the YOLO/classifier matrices) -- transpose so
    # every confusion matrix in this visualization suite reads the same way.
    cm = cm.T
    n = len(confident)
    plot_confusion_matrix(
        cm.tolist(), labels, labels, out_dir / "confusion_matrix_raw",
        title=f"Integrated pipeline -- confusion matrix (confident predictions only, n={n:,})",
        source=source, normalize=False, row_axis_label="Predicted", col_axis_label="True label",
    )
    plot_confusion_matrix(
        cm.tolist(), labels, labels, out_dir / "confusion_matrix_normalized",
        title=f"Integrated pipeline -- confusion matrix (confident predictions only, % of true class)",
        source=source, normalize=True, row_axis_label="Predicted", col_axis_label="True label",
    )


def plot_per_class_metrics(confident: pd.DataFrame, out_dir: Path, source: str):
    from sklearn.metrics import classification_report

    labels = CNN_CLASSES
    report = classification_report(confident["ground_truth"], confident["predicted"],
                                    labels=labels, output_dict=True, zero_division=0)
    colors = [CLASS_COLORS[c] for c in labels]

    accuracy = [report[c]["recall"] for c in labels]  # per-class accuracy == recall restricted to that true class
    precision = [report[c]["precision"] for c in labels]
    recall = [report[c]["recall"] for c in labels]
    f1 = [report[c]["f1-score"] for c in labels]

    bar_chart(labels, accuracy, out_dir / "per_class_accuracy",
              "Per-class accuracy (confident predictions only) -- integrated pipeline", "Accuracy",
              source, colors=colors, ylim=(0, 1.05), rotate_xticks=20)
    bar_chart(labels, precision, out_dir / "per_class_precision",
              "Per-class precision (confident predictions only) -- integrated pipeline", "Precision",
              source, colors=colors, ylim=(0, 1.05), rotate_xticks=20)
    bar_chart(labels, recall, out_dir / "per_class_recall",
              "Per-class recall (confident predictions only) -- integrated pipeline", "Recall",
              source, colors=colors, ylim=(0, 1.05), rotate_xticks=20)
    bar_chart(labels, f1, out_dir / "per_class_f1",
              "Per-class F1 (confident predictions only) -- integrated pipeline", "F1",
              source, colors=colors, ylim=(0, 1.05), rotate_xticks=20)


def plot_decision_status(summary: dict, out_dir: Path, source: str):
    status = summary["status_breakdown"]
    order = ["ok", "model_disagreement", "low_confidence", "error"]
    labels = [k for k in order if k in status]
    values = [status[k] for k in labels]
    colors = [STATUS_COLORS[k] for k in labels]
    total = sum(values)

    fig, ax = new_figure(figsize=(7, 5.5))
    wedges, _ = ax.pie(values, colors=colors, startangle=90, counterclock=False,
                        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2))
    ax.legend(wedges, [f"{l.replace('_', ' ')}  --  {v:,} ({v/total:.1%})" for l, v in zip(labels, values)],
              loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9.5)
    ax.set_title(f"Decision status -- integrated pipeline (test split, n={total:,})", loc="left")
    ax.text(0, 0, f"{total:,}\nimages", ha="center", va="center", fontsize=12, color="#1A2420")
    annotate_source(ax, source)
    fig.tight_layout()
    save_fig(fig, out_dir / "decision_status")

    bar_chart(labels, values, out_dir / "decision_status_bar",
              f"Decision status -- integrated pipeline (test split, n={total:,})", "# images", source,
              colors=colors, rotate_xticks=15, value_fmt="{:.0f}")


def plot_detection_path(df: pd.DataFrame, out_dir: Path, source: str):
    counts = df["decision_path"].value_counts()
    label_map = {
        "cnn_only_full_image": "No YOLO detection\n→ full-image CNN",
        "yolo_cnn_verification": "YOLO detection → crop\n→ CNN verification",
    }
    labels = [label_map.get(k, k) for k in counts.index]
    values = counts.values.tolist()
    total = sum(values)
    colors = [ACCENT_PARTIAL, ACCENT_FINAL][:len(labels)]

    fig, ax = new_figure(figsize=(7.5, 5.5))
    wedges, _ = ax.pie(values, colors=colors, startangle=90, counterclock=False,
                        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2))
    ax.legend(wedges, [f"{l}  --  {v:,} ({v/total:.1%})" for l, v in zip(labels, values)],
              loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9.5)
    ax.set_title(f"Detection path taken -- integrated pipeline (test split, n={total:,})", loc="left")
    annotate_source(ax, source)
    fig.tight_layout()
    save_fig(fig, out_dir / "detection_path")


def plot_accuracy_comparison(classifier_metrics: dict, pipeline_summary: dict, out_dir: Path,
                              classifier_source: str, pipeline_source: str):
    bars = [
        ("ResNet50\noverall", classifier_metrics["accuracy"], "ResNet50 standalone", ACCENT_PARTIAL),
        ("ResNet50\nfull-image", classifier_metrics["by_sample_type"]["full"]["accuracy"], "ResNet50 standalone", ACCENT_PARTIAL),
        ("ResNet50\ncrop", classifier_metrics["by_sample_type"]["crop"]["accuracy"], "ResNet50 standalone", ACCENT_PARTIAL),
        ("Integrated\noverall", pipeline_summary["accuracy_overall"], "Integrated system", ACCENT_FINAL),
        ("Integrated\nconfident-only", pipeline_summary["accuracy_when_confident"], "Integrated system", ACCENT_COND),
    ]
    labels = [b[0] for b in bars]
    values = [b[1] for b in bars]
    colors = [b[3] for b in bars]

    fig, ax = new_figure(figsize=(9.5, 6))
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, width=0.6)
    for xi, v in zip(x, values):
        ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy across evaluation conditions -- NOT directly equivalent metrics", loc="left")
    ax.axvline(2.5, color="#D9DED8", linewidth=1.4, linestyle="--")
    ax.text(1.0, 1.0, "ResNet50 evaluated standalone\n(full test set / full-image-only / crop-only)",
            ha="center", va="bottom", fontsize=8.5, color="#55625B")
    ax.text(3.5, 1.0, "Integrated YOLO+ResNet50 system\n(unconditional vs. confident-predictions-only)",
            ha="center", va="bottom", fontsize=8.5, color="#55625B")
    annotate_source(ax, f"sources: {classifier_source.removeprefix('source: ')} + "
                        f"{pipeline_source.removeprefix('source: ')}")
    fig.tight_layout()
    save_fig(fig, out_dir / "accuracy_comparison_labeled_conditions")


def main():
    parser = argparse.ArgumentParser(description="Generate the integrated-pipeline visualization suite "
                                                   "from the already-saved full test-set evaluation.")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV)
    parser.add_argument("--summary", type=str, default=DEFAULT_SUMMARY)
    parser.add_argument("--classifier-metrics", type=str, default=DEFAULT_CLASSIFIER_METRICS)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()

    csv_path = Path(args.csv)
    summary_path = Path(args.summary)
    clf_path = Path(args.classifier_metrics)
    for p, label in [(csv_path, "pipeline per-image CSV"), (summary_path, "pipeline summary JSON"),
                      (clf_path, "classifier metrics JSON")]:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found ({label}) -- this script only reads already-computed "
                                     f"results and will not rerun the pipeline evaluation to produce it.")

    df = pd.read_csv(csv_path)
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    with open(clf_path, "r", encoding="utf-8") as f:
        classifier_metrics = json.load(f)

    out_dir = config.path("paths", "outputs") / "visualizations" / "pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_source = f"source: {csv_path.as_posix()} (status==ok rows only)"
    summary_source = f"source: {summary_path.as_posix()}"
    clf_source = f"source: {clf_path.as_posix()}"

    print(f"Reading {len(df):,} rows from {csv_path} (rerun NOT performed).")
    confident = build_confident_subset(df)
    print(f"  {len(confident):,} confident (status==ok) rows used for confusion matrix / per-class metrics.")
    recomputed_acc = (confident["ground_truth"] == confident["predicted"]).mean()
    if not np.isclose(recomputed_acc, summary["accuracy_when_confident"], atol=1e-9):
        raise RuntimeError(
            f"Recomputed confident-only accuracy from {csv_path} ({recomputed_acc}) does not match "
            f"accuracy_when_confident in {summary_path} ({summary['accuracy_when_confident']}) -- "
            f"the CSV and summary JSON appear to be out of sync. Refusing to plot until this is resolved."
        )
    print(f"  OK -- recomputed accuracy matches summary JSON exactly ({recomputed_acc:.6f}).")

    print(f"Writing pipeline visualizations to {out_dir}")

    print("Confusion matrices (raw, normalized; confident predictions only)...")
    plot_confusion_matrices(confident, out_dir, csv_source)

    print("Per-class accuracy/precision/recall/F1...")
    plot_per_class_metrics(confident, out_dir, csv_source)

    print("Decision-status chart...")
    plot_decision_status(summary, out_dir, summary_source)

    print("Detection-path chart...")
    plot_detection_path(df, out_dir, csv_source)

    print("Accuracy comparison across evaluation conditions...")
    plot_accuracy_comparison(classifier_metrics, summary, out_dir, clf_source, summary_source)

    print(f"\nDone. Pipeline visualizations written to {out_dir}")


if __name__ == "__main__":
    main()
