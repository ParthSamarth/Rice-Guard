"""
evaluation/visualize_results.py

Aggregates results ALREADY WRITTEN by the training/evaluation scripts into
comparison plots + one JSON summary table for the final report:
  - detection/evaluate_yolo.py       -> outputs/yolo/metrics/<tag>_eval_test.json
  - classification/evaluate_classifier.py -> outputs/classification/<tag>/evaluation/metrics_test.json
  - evaluation/evaluate_pipeline.py  -> outputs/pipeline_evaluation/pipeline_eval_test_summary.json

Never invents a number: if an experiment's output file doesn't exist yet
(e.g. that model hasn't been trained/evaluated), it is skipped and reported
as missing in the console output and in outputs/results_comparison.json,
never silently filled in with a placeholder.

Usage:
    python evaluation/visualize_results.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import load_config  # noqa: E402


def collect_yolo_results(config, include_flagged: bool = False) -> tuple[dict, list]:
    """Skips any run whose *_train_summary.json declares a `status` field
    (e.g. "PARTIAL_INCOMPLETE_ABLATION") unless include_flagged is set --
    this is the one signal, set by detection/train_yolo.py:_finalize() for a
    normal run (never) vs. a manually-staged partial one (always), that
    distinguishes a finished baseline from an ablation that was stopped
    early. Without this, an ablation like the 1024 experiment would appear
    as an equal, undifferentiated bar next to the converged 640 baseline the
    moment its eval JSON exists on disk -- silently implying a head-to-head
    comparison this project's own documentation explicitly says not to make.
    Returns (results, skipped) so callers can disclose what was left out
    rather than silently dropping it."""
    metrics_dir = config.path("paths", "outputs") / "yolo" / "metrics"
    results, skipped = {}, []
    if metrics_dir.exists():
        for f in sorted(metrics_dir.glob("*_eval_test.json")):
            run_tag = f.name.replace("_eval_test.json", "")
            train_summary_f = metrics_dir / f"{run_tag}_train_summary.json"
            if train_summary_f.exists() and not include_flagged:
                with open(train_summary_f, "r", encoding="utf-8") as fh:
                    status = json.load(fh).get("status")
                if status:
                    skipped.append((run_tag, status))
                    continue
            with open(f, "r", encoding="utf-8") as fh:
                results[run_tag] = json.load(fh)
    return results, skipped


def collect_classifier_results(config) -> dict:
    class_dir = config.path("paths", "outputs") / "classification"
    results = {}
    if class_dir.exists():
        for run_dir in sorted(class_dir.iterdir()):
            f = run_dir / "evaluation" / "metrics_test.json"
            if f.exists():
                with open(f, "r", encoding="utf-8") as fh:
                    results[run_dir.name] = json.load(fh)
    return results


def collect_training_summaries(config) -> dict:
    """For params/training-time/GPU-memory comparison rows (project brief
    section 18's experiment comparison table)."""
    summaries = {}
    yolo_metrics_dir = config.path("paths", "outputs") / "yolo" / "metrics"
    if yolo_metrics_dir.exists():
        for f in sorted(yolo_metrics_dir.glob("*_train_summary.json")):
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            summaries[f"yolo_{data.get('run_tag', f.stem)}"] = data
    class_dir = config.path("paths", "outputs") / "classification"
    if class_dir.exists():
        for run_dir in sorted(class_dir.iterdir()):
            f = run_dir / "train_summary.json"
            if f.exists():
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                summaries[f"classifier_{run_dir.name}"] = data
    return summaries


def plot_yolo_comparison(results: dict, out_dir: Path):
    tags = list(results.keys())
    metrics = ["map50", "map50_95", "mean_precision", "mean_recall"]
    labels = ["mAP@50", "mAP@50-95", "Mean Precision", "Mean Recall"]

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(tags) + 2), 5))
    x = np.arange(len(tags))
    width = 0.8 / len(metrics)
    for i, (m, lbl) in enumerate(zip(metrics, labels)):
        vals = [results[t]["overall"].get(m) or 0 for t in tags]
        ax.bar(x + i * width, vals, width, label=lbl)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(tags)
    ax.set_ylabel("Score")
    ax.set_title("YOLO experiment comparison (test split)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "yolo_comparison.png", dpi=150)
    plt.close(fig)


def plot_classifier_comparison(results: dict, out_dir: Path):
    tags = list(results.keys())
    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    labels = ["Accuracy", "Macro F1", "Weighted F1"]

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(tags) + 2), 5))
    x = np.arange(len(tags))
    width = 0.8 / len(metrics)
    for i, (m, lbl) in enumerate(zip(metrics, labels)):
        vals = [results[t].get(m) or 0 for t in tags]
        ax.bar(x + i * width, vals, width, label=lbl)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(tags)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("CNN classifier experiment comparison (test split)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "classifier_comparison.png", dpi=150)
    plt.close(fig)


def plot_pipeline_summary(config, out_dir: Path):
    summary_path = config.path("paths", "outputs") / "pipeline_evaluation" / "pipeline_eval_test_summary.json"
    if not summary_path.exists():
        print("No pipeline evaluation summary found yet -- run evaluation/evaluate_pipeline.py first.")
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    status = summary["status_breakdown"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = list(status.keys())
    values = [status[k] for k in labels]
    axes[0].pie(values, labels=[f"{k} ({v})" for k, v in zip(labels, values)], autopct="%1.1f%%")
    axes[0].set_title("Integrated pipeline: status breakdown (test split)")

    acc_labels = ["Overall\n(disagreement/low-conf = wrong)", "When confident\n(status == ok only)"]
    acc_values = [summary.get("accuracy_overall") or 0, summary.get("accuracy_when_confident") or 0]
    axes[1].bar(acc_labels, acc_values, color=["#C44E52", "#55A868"])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Integrated pipeline accuracy")
    for i, v in enumerate(acc_values):
        axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    fig.savefig(out_dir / "pipeline_summary.png", dpi=150)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Build final comparison plots + outputs/results_comparison.json.")
    parser.add_argument("--include-flagged", action="store_true",
                         help="Include YOLO runs whose train_summary.json declares a `status` (e.g. a "
                              "partial/incomplete ablation) in the aggregate comparison chart. Default "
                              "(off) keeps this chart to finished, final experiments only -- see "
                              "collect_yolo_results().")
    args = parser.parse_args()

    config = load_config()
    out_dir = config.path("paths", "outputs") / "comparison_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    yolo_results, yolo_skipped = collect_yolo_results(config, include_flagged=args.include_flagged)
    clf_results = collect_classifier_results(config)
    train_summaries = collect_training_summaries(config)

    comparison = {"yolo": {}, "classifier": {}, "pipeline": None, "missing": [],
                  "yolo_excluded_flagged_runs": [{"run_tag": t, "status": s} for t, s in yolo_skipped]}

    if yolo_results:
        plot_yolo_comparison(yolo_results, out_dir)
        for tag, r in yolo_results.items():
            ts = train_summaries.get(f"yolo_{tag}", {})
            comparison["yolo"][tag] = {
                "overall": r["overall"],
                "training_time_sec": ts.get("total_training_time_sec"),
                "hardware": ts.get("hardware", {}).get("gpu_name"),
                "settings": ts.get("settings"),
            }
        print(f"Wrote {out_dir / 'yolo_comparison.png'}")
    else:
        comparison["missing"].append("yolo evaluation results (run detection/evaluate_yolo.py)")
        print("No YOLO evaluation results found yet.")
    if yolo_skipped:
        print("Excluded from the YOLO comparison chart (flagged, non-final -- pass --include-flagged to add back):")
        for tag, status in yolo_skipped:
            print(f"  - {tag}: {status}")

    if clf_results:
        plot_classifier_comparison(clf_results, out_dir)
        for tag, r in clf_results.items():
            ts = train_summaries.get(f"classifier_{tag}", {})
            comparison["classifier"][tag] = {
                "accuracy": r.get("accuracy"), "macro_f1": r.get("macro_f1"), "weighted_f1": r.get("weighted_f1"),
                "training_time_sec": ts.get("total_training_time_sec"),
                "param_info": ts.get("param_info"),
                "hardware": ts.get("hardware", {}).get("gpu_name"),
                "settings": ts.get("settings"),
            }
        print(f"Wrote {out_dir / 'classifier_comparison.png'}")
    else:
        comparison["missing"].append("classifier evaluation results (run classification/evaluate_classifier.py)")
        print("No classifier evaluation results found yet.")

    pipeline_summary = plot_pipeline_summary(config, out_dir)
    if pipeline_summary:
        comparison["pipeline"] = pipeline_summary
        print(f"Wrote {out_dir / 'pipeline_summary.png'}")
    else:
        comparison["missing"].append("pipeline evaluation results (run evaluation/evaluate_pipeline.py)")

    out_json = config.path("paths", "outputs") / "results_comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"Wrote {out_json}")
    if comparison["missing"]:
        print("\nStill missing (not fabricated, simply not yet run):")
        for m in comparison["missing"]:
            print(f"  - {m}")


if __name__ == "__main__":
    main()
