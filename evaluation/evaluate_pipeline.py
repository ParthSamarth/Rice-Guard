"""
evaluation/evaluate_pipeline.py

Evaluates the INTEGRATED YOLO -> CNN pipeline (not each model in isolation
-- that's detection/evaluate_yolo.py and classification/evaluate_classifier.py)
on the held-out TEST split's ORIGINAL, full, un-cropped images -- the same
kind of input the system would actually receive in the field.

Reports (project brief section 17 "Integrated pipeline"):
  - YOLO/CNN agreement rate and disagreement rate
  - final classification accuracy (multiple breakdowns -- see below)
  - failure cases, false positives, false negatives

Ground truth for every test image is its known source class (from
dataset/processed/split_manifest.csv), which is what makes this evaluation
possible at all -- and note that BOTH the YOLO and the CNN checkpoints
passed in here must have been trained with this same split_manifest.csv
(preprocessing/make_split_manifest.py) so that none of these test images
were used to train either model.

Because `status` can be "model_disagreement" or "low_confidence" (the
pipeline deliberately declines to output a final_disease in those cases --
see pipeline/inference_pipeline.py), a single "accuracy" number would hide
that. This script reports:
  - `accuracy_overall`: correct / all test images (disagreement/low-confidence
    count as incorrect -- the honest, conservative number)
  - `accuracy_when_confident`: correct / images where status == "ok" only
    (how good the system is when it commits to an answer)
  - `disagreement_rate`, `low_confidence_rate`, `error_rate` as their own
    first-class metrics, never folded silently into "accuracy"
  - `macro_f1`/`weighted_f1` (and precision/recall) for the integrated
    system, scored over every evaluated image -- unresolved rows count as a
    miss, they are never dropped from the metric just because the pipeline
    declined to answer
  - `num_no_detection_cases` / `no_detection_rate`: how often YOLO found
    nothing at all (the CNN-full-image path), tracked explicitly rather than
    folded into the other counts

Usage:
    python evaluation/evaluate_pipeline.py --yolo-weights models/yolo_yolov8n/best.pt --classifier-checkpoint models/classifier_resnet50/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.inference_pipeline import RiceDiseasePipeline  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.dataset_common import ALL_CLASSES, HEALTHY, normalize_class_name  # noqa: E402

UNRESOLVED = "__UNRESOLVED__"  # sentinel for model_disagreement/low_confidence/error rows in the P/R/F1 computation


def main():
    parser = argparse.ArgumentParser(description="Evaluate the integrated YOLO+CNN pipeline on the test split.")
    parser.add_argument("--yolo-weights", type=str, required=True)
    parser.add_argument("--classifier-checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--no-gradcam", action="store_true", help="Skip Grad-CAM generation (faster evaluation)")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N images (smoke test)")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    manifest_path = config.path("paths", "dataset_processed") / "split_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    split_df = manifest[manifest["split"] == args.split].reset_index(drop=True)
    if args.limit:
        split_df = split_df.iloc[:args.limit]

    print(f"Evaluating integrated pipeline on {len(split_df)} '{args.split}'-split images "
          f"(ground truth from {manifest_path})")

    pipeline = RiceDiseasePipeline(
        args.yolo_weights, args.classifier_checkpoint, config=config, save_gradcam=not args.no_gradcam
    )

    rows = []
    for i, row in split_df.iterrows():
        gt_class = row["class"]
        image_path = row["abs_path"]
        if not Path(image_path).exists():
            continue
        try:
            result = pipeline.run(image_path)
        except Exception as exc:  # noqa: BLE001
            rows.append({"image": image_path, "ground_truth": gt_class, "status": "error",
                         "predicted": None, "final_confidence": None, "decision_path": "error",
                         "agreement": None, "correct": False, "num_yolo_detections": 0, "error": str(exc)})
            continue

        predicted = result["final_disease"]
        correct = (predicted is not None) and (normalize_class_name(predicted) == normalize_class_name(gt_class))
        rows.append({
            "image": image_path,
            "ground_truth": gt_class,
            "predicted": predicted,
            "final_confidence": result["final_confidence"],
            "status": result["status"],
            "decision_path": result["decision_path"],
            "agreement": result["agreement"],
            "correct": correct,
            "num_yolo_detections": len(result["yolo"]["detections"]),
        })

        if (i + 1) % 200 == 0:
            print(f"  processed {i + 1}/{len(split_df)}")

    df = pd.DataFrame(rows)
    out_dir = config.path("paths", "outputs") / "pipeline_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"pipeline_eval_{args.split}.csv", index=False)

    n = len(df)
    n_ok = int((df["status"] == "ok").sum())
    n_disagree = int((df["status"] == "model_disagreement").sum())
    n_low_conf = int((df["status"] == "low_confidence").sum())
    n_error = int((df["status"] == "error").sum())
    n_correct_overall = int(df["correct"].sum())
    ok_df = df[df["status"] == "ok"]
    n_correct_ok = int(ok_df["correct"].sum()) if len(ok_df) else 0
    n_no_detection = int((df["num_yolo_detections"] == 0).sum()) if "num_yolo_detections" in df else None

    # Integrated-system macro/weighted precision/recall/F1 (project brief
    # section 22). Every row gets a prediction -- unresolved rows
    # (model_disagreement/low_confidence/error, where final_disease is None)
    # are scored against a sentinel label that can never match any true
    # class, so they correctly count as a miss for whatever the true class
    # was rather than being dropped from the metric entirely.
    y_true = df["ground_truth"].map(lambda c: normalize_class_name(c) or c).tolist()
    y_pred = df["predicted"].map(
        lambda p: (normalize_class_name(p) or p) if isinstance(p, str) and p else UNRESOLVED
    ).tolist()
    class_labels = [normalize_class_name(c) or c for c in ALL_CLASSES]
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, average="weighted", zero_division=0
    )

    # Failure-case taxonomy.
    healthy_norm = normalize_class_name(HEALTHY)
    false_positives = df[(df["ground_truth"].map(normalize_class_name) == healthy_norm) &
                          (df["correct"] == False) & (df["status"] == "ok")]  # noqa: E712  -- said Healthy is actually diseased
    false_negatives = df[(df["ground_truth"].map(normalize_class_name) != healthy_norm) &
                          (df["predicted"].map(lambda p: normalize_class_name(p) == healthy_norm if p else False))]  # missed a real disease as Healthy
    disease_confusions = df[(df["correct"] == False) & (df["status"] == "ok") &
                             (df["ground_truth"].map(normalize_class_name) != healthy_norm) &
                             (df["predicted"].map(lambda p: p is not None and normalize_class_name(p) != healthy_norm))]

    summary = {
        "split": args.split,
        "yolo_weights": args.yolo_weights,
        "classifier_checkpoint": args.classifier_checkpoint,
        "num_images_evaluated": n,
        "status_breakdown": {"ok": n_ok, "model_disagreement": n_disagree, "low_confidence": n_low_conf, "error": n_error},
        "agreement_rate": float((df["agreement"] == True).sum() / n) if n else None,  # noqa: E712
        "disagreement_rate": float(n_disagree / n) if n else None,
        "low_confidence_rate": float(n_low_conf / n) if n else None,
        "error_rate": float(n_error / n) if n else None,
        "accuracy_overall": float(n_correct_overall / n) if n else None,
        "accuracy_when_confident": float(n_correct_ok / len(ok_df)) if len(ok_df) else None,
        "macro_precision": float(macro_p), "macro_recall": float(macro_r), "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p), "weighted_recall": float(weighted_r), "weighted_f1": float(weighted_f1),
        "note_on_macro_metrics": "Computed over ALL evaluated images, including model_disagreement/low_confidence/"
                                  "error rows -- those score as a miss against whatever the true class was (see "
                                  "UNRESOLVED sentinel in this script), not dropped from the metric.",
        "num_no_detection_cases": n_no_detection,
        "no_detection_rate": float(n_no_detection / n) if (n and n_no_detection is not None) else None,
        "num_false_positive_healthy_missed": int(len(false_positives)),
        "num_false_negative_disease_missed_as_healthy": int(len(false_negatives)),
        "num_disease_to_disease_confusions": int(len(disease_confusions)),
        "false_positive_examples": false_positives["image"].tolist()[:20],
        "false_negative_examples": false_negatives["image"].tolist()[:20],
        "disease_confusion_examples": disease_confusions[["image", "ground_truth", "predicted"]].head(20).to_dict("records"),
        "per_class_accuracy_when_confident": {
            cls: (float(ok_df[ok_df["ground_truth"].map(normalize_class_name) == normalize_class_name(cls)]["correct"].mean())
                  if len(ok_df[ok_df["ground_truth"].map(normalize_class_name) == normalize_class_name(cls)]) else None)
            for cls in ALL_CLASSES
        },
    }

    with open(out_dir / f"pipeline_eval_{args.split}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_dir / f'pipeline_eval_{args.split}.csv'}")
    print(f"Wrote {out_dir / f'pipeline_eval_{args.split}_summary.json'}")


if __name__ == "__main__":
    main()
