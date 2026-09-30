"""
evaluation/build_experiment_summary.py

Assembles outputs/experiment_summary.json (project brief section 22,
Reproducibility): environment (Python/PyTorch/CUDA/GPU), every experiment's
hyperparameters and training duration, dataset split sizes, and the fixed
seed -- all pulled from the JSON files each script already wrote. Nothing
here is re-derived or guessed; sections are simply omitted (not
fabricated) if the corresponding step hasn't been run yet.

Usage:
    python evaluation/build_experiment_summary.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import load_config  # noqa: E402
from utils.hardware import detect_hardware  # noqa: E402


def main():
    config = load_config()
    profile = detect_hardware()
    outputs = config.path("paths", "outputs")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": config["seed"],
        "environment": profile.to_dict(),
        "dataset": {},
        "yolo_experiments": {},
        "classifier_experiments": {},
        "pipeline_evaluation": None,
        "notes": [],
    }

    audit_path = outputs / "dataset_audit.json"
    if audit_path.exists():
        with open(audit_path, "r", encoding="utf-8") as f:
            audit = json.load(f)
        summary["dataset"]["audit_summary"] = audit.get("summary")
        summary["dataset"]["class_mapping_matches_documented_readme"] = audit.get("bounding_boxes", {}).get("matches_documented")
    else:
        summary["notes"].append("dataset_audit.json not found -- run preprocessing/dataset_audit.py")

    manifest_path = config.path("paths", "dataset_processed") / "split_manifest.csv"
    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        summary["dataset"]["split_sizes_by_class"] = json.loads(
            df.groupby(["class", "split"], observed=True).size().unstack(fill_value=0).to_json()
        )
    else:
        summary["notes"].append("split_manifest.csv not found -- run preprocessing/make_split_manifest.py")

    cls_manifest_path = config.path("paths", "classification_dataset") / "manifest.csv"
    if cls_manifest_path.exists():
        cdf = pd.read_csv(cls_manifest_path)
        summary["dataset"]["classification_samples_by_label_and_type"] = json.loads(
            cdf.groupby(["label", "sample_type"], observed=True).size().unstack(fill_value=0).to_json()
        )

    yolo_metrics_dir = outputs / "yolo" / "metrics"
    if yolo_metrics_dir.exists():
        for f in sorted(yolo_metrics_dir.glob("*_train_summary.json")):
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            eval_f = yolo_metrics_dir / f"{data.get('run_tag')}_eval_test.json"
            if eval_f.exists():
                with open(eval_f, "r", encoding="utf-8") as fh:
                    data["test_eval"] = json.load(fh)
            summary["yolo_experiments"][data.get("run_tag", f.stem)] = data
    if not summary["yolo_experiments"]:
        summary["notes"].append("No YOLO training summaries found -- run detection/train_yolo.py")

    class_dir = outputs / "classification"
    if class_dir.exists():
        for run_dir in sorted(class_dir.iterdir()):
            f = run_dir / "train_summary.json"
            if f.exists():
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                eval_f = run_dir / "evaluation" / "metrics_test.json"
                if eval_f.exists():
                    with open(eval_f, "r", encoding="utf-8") as fh:
                        data["test_eval"] = json.load(fh)
                summary["classifier_experiments"][run_dir.name] = data
    if not summary["classifier_experiments"]:
        summary["notes"].append("No classifier training summaries found -- run classification/train_classifier.py")

    pipeline_summary_path = outputs / "pipeline_evaluation" / "pipeline_eval_test_summary.json"
    if pipeline_summary_path.exists():
        with open(pipeline_summary_path, "r", encoding="utf-8") as f:
            summary["pipeline_evaluation"] = json.load(f)
    else:
        summary["notes"].append("No integrated-pipeline evaluation found -- run evaluation/evaluate_pipeline.py")

    out_path = outputs / "experiment_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Wrote {out_path}")
    if summary["notes"]:
        print("Notes (steps not yet run):")
        for n in summary["notes"]:
            print(f"  - {n}")


if __name__ == "__main__":
    main()
