"""
evaluation/generate_final_dashboard.py

One final, PPT/report-ready summary figure covering the FINAL selected
models only (YOLOv8n @ 640 + ResNet50 + the integrated system), and a
SEPARATE, clearly-labeled figure for the YOLOv8n @ 1024 partial ablation --
the two are never combined into one comparison, per this project's own
"do not present 1024 as a fair replacement for 640" rule.

Every number plotted is read live from its authoritative JSON at generation
time (never hand-typed):

  outputs/yolo/metrics/yolov8n_eval_test.json               (640 detector)
  outputs/classification/resnet50/evaluation/metrics_test.json  (ResNet50 + full/crop split)
  outputs/pipeline_evaluation/pipeline_eval_test_summary.json   (integrated system)
  outputs/yolo/metrics/yolov8n_1024_eval_test.json           (1024 ablation, separate figure only)
  outputs/yolo/metrics/yolov8n_1024_train_summary.json       (1024 ablation status/epoch context)

Usage:
    python evaluation/generate_final_dashboard.py
    python main.py visualize-dashboard
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import load_config  # noqa: E402
from evaluation.plot_style import (  # noqa: E402
    ACCENT_FINAL, ACCENT_PARTIAL, ACCENT_COND, INK, MUTED, FAINT, GRID, SURFACE,
    save_fig, annotate_source,
)


def _panel_source(ax, text: str):
    """Per-panel source caption, axes-anchored (NOT the shared figure-anchored
    annotate_source() in plot_style.py -- that helper is for single-panel
    exports and would stack every panel's caption on top of each other in one
    corner of a multi-panel figure like this dashboard). Placed bottom-left of
    each panel's own axes, in the gridspec hspace gap below it."""
    ax.text(0, -0.16, text, transform=ax.transAxes, ha="left", va="top",
             fontsize=7, color=FAINT)

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_YOLO_JSON = "outputs/yolo/metrics/yolov8n_eval_test.json"
DEFAULT_CLASSIFIER_JSON = "outputs/classification/resnet50/evaluation/metrics_test.json"
DEFAULT_PIPELINE_JSON = "outputs/pipeline_evaluation/pipeline_eval_test_summary.json"
DEFAULT_ABLATION_JSON = "outputs/yolo/metrics/yolov8n_1024_eval_test.json"
DEFAULT_ABLATION_SUMMARY = "outputs/yolo/metrics/yolov8n_1024_train_summary.json"


def _panel_bars(ax, labels, values, colors, title, ylim=(0, 1.0), value_fmt="{:.3f}"):
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, width=0.6)
    for xi, v in zip(x, values):
        ax.text(xi, v, value_fmt.format(v), ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(*ylim)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def build_final_dashboard(yolo, clf, pipe, out_dir: Path, sources: dict):
    fig = plt.figure(figsize=(15, 11))
    fig.suptitle("RiceLeafDiseaseBD -- Intelligence Layer, final results", fontsize=17,
                 fontweight="bold", x=0.02, y=0.985, ha="left", va="top", color=INK)
    fig.text(0.02, 0.935, "YOLOv8n @ 640 detection -> ResNet50 verification -> Grad-CAM -> recommendation "
                          "engine  |  all figures below are FINAL, verified numbers, held-out test split",
             fontsize=10.5, color=MUTED, ha="left", va="top")

    gs = fig.add_gridspec(2, 2, hspace=0.65, wspace=0.28, top=0.84, bottom=0.08, left=0.06, right=0.97)

    ax1 = fig.add_subplot(gs[0, 0])
    _panel_bars(
        ax1,
        ["mAP50", "mAP50-95", "Precision", "Recall"],
        [yolo["overall"]["map50"], yolo["overall"]["map50_95"],
         yolo["overall"]["mean_precision"], yolo["overall"]["mean_recall"]],
        [ACCENT_FINAL] * 4,
        "Detection -- YOLOv8n @ 640 (test split)",
        ylim=(0, 0.5),
    )

    ax2 = fig.add_subplot(gs[0, 1])
    _panel_bars(
        ax2,
        ["Accuracy", "Macro-F1"],
        [clf["accuracy"], clf["macro_f1"]],
        [ACCENT_FINAL, ACCENT_COND],
        "Verification -- ResNet50, overall (test split)",
        ylim=(0, 1.0),
    )

    ax3 = fig.add_subplot(gs[1, 0])
    full, crop = clf["by_sample_type"]["full"], clf["by_sample_type"]["crop"]
    x = np.arange(2)
    width = 0.32
    ax3.bar(x - width / 2, [full["accuracy"], crop["accuracy"]], width, label="Accuracy", color=ACCENT_FINAL)
    ax3.bar(x + width / 2, [full["macro_f1"], crop["macro_f1"]], width, label="Macro-F1", color=ACCENT_PARTIAL)
    for xi, v in zip(x - width / 2, [full["accuracy"], crop["accuracy"]]):
        ax3.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    for xi, v in zip(x + width / 2, [full["macro_f1"], crop["macro_f1"]]):
        ax3.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"Full image\n(n={full['num_samples']:,})", f"YOLO crop\n(n={crop['num_samples']:,})"], fontsize=9)
    ax3.set_ylim(0, 1.0)
    ax3.set_title("ResNet50 -- input mode matters", loc="left", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=8.5)
    ax3.grid(axis="y", color=GRID, linewidth=0.8)
    ax3.set_axisbelow(True)
    for spine in ("top", "right"):
        ax3.spines[spine].set_visible(False)

    ax4 = fig.add_subplot(gs[1, 1])
    status = pipe["status_breakdown"]
    total = sum(status.values())
    abstention = (status.get("model_disagreement", 0) + status.get("low_confidence", 0)) / total
    _panel_bars(
        ax4,
        ["Overall\n(unconditional)", "Confident\nonly", "Macro-F1", "Weighted-F1"],
        [pipe["accuracy_overall"], pipe["accuracy_when_confident"], pipe["macro_f1"], pipe["weighted_f1"]],
        [ACCENT_FINAL, ACCENT_COND, ACCENT_FINAL, ACCENT_FINAL],
        f"Integrated system (test split, n={total:,}, abstention={abstention:.1%})",
        ylim=(0, 1.0),
    )

    for ax, key in [(ax1, "yolo"), (ax2, "clf"), (ax3, "clf"), (ax4, "pipe")]:
        _panel_source(ax, f"source: {sources[key]}")

    return save_fig(fig, out_dir / "final_dashboard")


def build_ablation_panel(ablation, ablation_summary, out_dir: Path, sources: dict):
    fig = plt.figure(figsize=(11, 7.5))
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle("YOLOv8n @ 1024 -- PARTIAL / INCOMPLETE ABLATION -- NOT SELECTED", fontsize=15,
                 fontweight="700", x=0.02, ha="left", color=ACCENT_PARTIAL)
    epochs_done = ablation_summary.get("epochs_completed", "?")
    epochs_cap = ablation_summary.get("epochs_requested_cap", "?")
    best_epoch = ablation_summary.get("best_epoch_completed", epochs_done)
    fig.text(0.02, 0.90,
              f"Stopped at epoch {epochs_done}/{epochs_cap} after repeated environment-level interruptions "
              f"(not a modeling decision, not a metric-driven stop). Shown here only for transparency -- "
              f"excluded from the final model comparison. See PROJECT_README.md §7.1b.",
              fontsize=10, color=MUTED, ha="left", wrap=True)

    gs = fig.add_gridspec(1, 1, top=0.72, bottom=0.14, left=0.09, right=0.95)
    ax = fig.add_subplot(gs[0, 0])
    labels = ["mAP50", "mAP50-95", "Precision", "Recall"]
    values = [ablation["overall"]["map50"], ablation["overall"]["map50_95"],
              ablation["overall"]["mean_precision"], ablation["overall"]["mean_recall"]]
    _panel_bars(ax, labels, values, [ACCENT_PARTIAL] * 4,
                f"Test-split metrics -- best-fitness checkpoint (epoch {best_epoch}/{epochs_cap}; "
                f"run stopped after epoch {epochs_done}/{epochs_cap})",
                ylim=(0, 0.5))
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(ACCENT_PARTIAL)
        spine.set_linewidth(1.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    annotate_source(ax, f"sources: {sources['ablation']} + {sources['ablation_summary']}")

    return save_fig(fig, out_dir / "ablation_1024_not_selected")


def main():
    parser = argparse.ArgumentParser(description="Generate the final PPT-ready dashboard (640/ResNet50/pipeline "
                                                   "only) and a separate, clearly-labeled 1024-ablation figure.")
    parser.add_argument("--yolo-json", type=str, default=DEFAULT_YOLO_JSON)
    parser.add_argument("--classifier-json", type=str, default=DEFAULT_CLASSIFIER_JSON)
    parser.add_argument("--pipeline-json", type=str, default=DEFAULT_PIPELINE_JSON)
    parser.add_argument("--ablation-json", type=str, default=DEFAULT_ABLATION_JSON)
    parser.add_argument("--ablation-summary", type=str, default=DEFAULT_ABLATION_SUMMARY)
    parser.add_argument("--skip-ablation-panel", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()

    def _load(path_str, label):
        p = Path(path_str)
        if not p.exists():
            raise FileNotFoundError(f"{p} not found ({label}).")
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f), p

    yolo, yolo_p = _load(args.yolo_json, "final YOLO eval")
    clf, clf_p = _load(args.classifier_json, "final classifier eval")
    pipe, pipe_p = _load(args.pipeline_json, "pipeline eval summary")

    out_dir = config.path("paths", "outputs") / "visualizations" / "dashboard"
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = {"yolo": yolo_p.as_posix(), "clf": clf_p.as_posix(), "pipe": pipe_p.as_posix()}

    print(f"Writing final dashboard (640 + ResNet50 + integrated, NO 1024) to {out_dir}")
    build_final_dashboard(yolo, clf, pipe, out_dir, sources)

    if not args.skip_ablation_panel:
        ablation_path = Path(args.ablation_json)
        ablation_summary_path = Path(args.ablation_summary)
        if ablation_path.exists() and ablation_summary_path.exists():
            with open(ablation_path, "r", encoding="utf-8") as f:
                ablation = json.load(f)
            with open(ablation_summary_path, "r", encoding="utf-8") as f:
                ablation_summary = json.load(f)
            sources["ablation"] = ablation_path.as_posix()
            sources["ablation_summary"] = ablation_summary_path.as_posix()
            print(f"Writing separate 1024-ablation figure (clearly labeled, not part of final comparison)...")
            build_ablation_panel(ablation, ablation_summary, out_dir, sources)
        else:
            print(f"  1024 ablation eval/summary not found at {ablation_path} / {ablation_summary_path} "
                  f"-- skipping the separate ablation figure (this is fine; it is not part of the final dashboard).")

    print(f"\nDone. Dashboard visualizations written to {out_dir}")


if __name__ == "__main__":
    main()
