"""
explainability/gradcam_quality_check.py

Generates Grad-CAM examples for every one of the 6 classes (from the held-
out test split, both correctly- and incorrectly-classified samples) so a
human can visually inspect whether the CNN is focusing on disease lesions /
discoloration / relevant leaf regions -- or on background, camera artifacts,
image borders, or other irrelevant content (project brief section 14).

This script does NOT claim to automatically prove or disprove model
correctness. It saves per-class example panels for human review, and
additionally computes one cheap automatic heuristic ("border energy
fraction" -- see `limitations` in the output JSON) purely to help a
reviewer triage which images to look at first.

Usage:
    python explainability/gradcam_quality_check.py --checkpoint models/classifier_resnet50/best.pt
    python explainability/gradcam_quality_check.py --checkpoint models/classifier_resnet50/best.pt --per-class 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from classification.dataset import ALL_CLASSES  # noqa: E402
from classification.predict_classifier import ClassifierPredictor  # noqa: E402
from explainability.gradcam import explain_image  # noqa: E402
from utils.config import load_config  # noqa: E402


def border_energy_fraction(cam: np.ndarray, border_frac: float = 0.15) -> float:
    """Share of the Grad-CAM activation mass that falls inside the outer
    `border_frac` ring of the map. A crude triage heuristic -- see
    `limitations` in the written summary for what this can and can't tell you.
    """
    h, w = cam.shape
    bh, bw = max(1, int(h * border_frac)), max(1, int(w * border_frac))
    border_mask = np.ones_like(cam, dtype=bool)
    border_mask[bh:h - bh, bw:w - bw] = False
    total = cam.sum()
    if total <= 1e-8:
        return 0.0
    return float(cam[border_mask].sum() / total)


def main():
    parser = argparse.ArgumentParser(description="Generate per-class Grad-CAM examples for human review.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--per-class", type=int, default=6, help="Correctly-classified examples to save per class")
    parser.add_argument("--misclassified-per-class", type=int, default=3,
                         help="Additionally save up to this many MISclassified examples per true class")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    predictor = ClassifierPredictor(args.checkpoint)

    manifest_csv = config.path("paths", "classification_dataset") / "manifest.csv"
    df = pd.read_csv(manifest_csv)
    df = df[df["split"] == args.split].reset_index(drop=True)

    out_dir = config.path("paths", "outputs") / "gradcam_quality_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    seed = config["seed"]

    for cls in ALL_CLASSES:
        cls_df = df[df["label"] == cls]
        if len(cls_df) == 0:
            print(f"  (no {args.split} samples for class '{cls}', skipping)")
            continue

        # First pass: classify everything in this class's sample pool so we
        # can deliberately save both correct AND incorrect examples, rather
        # than only whichever the random sample happens to contain.
        pool = cls_df.sample(n=min(len(cls_df), max(args.per_class, args.misclassified_per_class) * 8),
                              random_state=seed)
        classified = []
        for _, row in pool.iterrows():
            pred = predictor.predict(row["image_path"])
            classified.append((row, pred))

        correct = [(r, p) for r, p in classified if p["class"] == cls][:args.per_class]
        incorrect = [(r, p) for r, p in classified if p["class"] != cls][:args.misclassified_per_class]

        for tag, subset in (("correct", correct), ("misclassified", incorrect)):
            for row, _pred in subset:
                image_path = row["image_path"]
                safe_cls = cls.replace(" ", "_")
                save_path = out_dir / safe_cls / f"{tag}_{Path(image_path).stem}_gradcam.png"
                result = explain_image(predictor, image_path, save_path=save_path)
                bef = border_energy_fraction(result["cam"])
                rows.append({
                    "true_class": cls, "tag": tag, "sample_type": row.get("sample_type"),
                    "image_path": image_path, "predicted_class": result["target_class_name"],
                    "confidence": result["confidence"], "correct": result["target_class_name"] == cls,
                    "border_energy_fraction": round(bef, 4),
                    "gradcam_path": str(save_path),
                })
        print(f"  {cls}: saved {len(correct)} correct + {len(incorrect)} misclassified examples")

    report_df = pd.DataFrame(rows)
    report_df.to_csv(out_dir / "gradcam_quality_check.csv", index=False)

    high_border = report_df[report_df["border_energy_fraction"] > 0.5] if len(report_df) else report_df
    summary = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "total_examples_saved": int(len(report_df)),
        "per_class_counts": report_df.groupby(["true_class", "tag"]).size().unstack(fill_value=0).to_dict() if len(report_df) else {},
        "accuracy_on_sampled_pool": float(report_df["correct"].mean()) if len(report_df) else None,
        "mean_border_energy_fraction": float(report_df["border_energy_fraction"].mean()) if len(report_df) else None,
        "high_border_energy_fraction_count": int(len(high_border)),
        "high_border_energy_fraction_examples": high_border["gradcam_path"].tolist()[:20],
        "output_directory": str(out_dir),
        "limitations": [
            "This script does not itself judge whether Grad-CAM activations are 'meaningful' -- that "
            "requires a human to open the saved *_gradcam.png panels in outputs/gradcam_quality_check/ "
            "and visually check whether the highlighted region covers the lesion/discoloration rather "
            "than background, camera artifacts, or image borders. No such visual judgment has been made "
            "automatically here.",
            "'border_energy_fraction' (share of Grad-CAM mass within the outer 15% border ring) is a "
            "cheap, unvalidated proxy for 'possibly looking at the edge/background instead of the leaf'. "
            "It will over-flag legitimate cases (e.g. a lesion that happens to sit near the leaf edge, or "
            "Rice Tungro's genuinely diffuse whole-leaf symptoms) and under-flag failures that are not "
            "border-located (e.g. focusing on a shadow or a smartphone UI artifact in the middle of the "
            "frame). Use it only to help pick which images to look at first, not as a pass/fail metric.",
            "Misclassified examples are sampled from whatever the fixed random pool for each class "
            "happened to contain; if a class has very few test-set errors, its 'misclassified' folder may "
            "be smaller than --misclassified-per-class or empty.",
        ],
        "next_step": "A human (the project author) should now open outputs/gradcam_quality_check/<class>/ "
                      "for each of the 6 classes, look at the correct AND misclassified examples, and "
                      "record actual qualitative observations + failure cases in the final report -- this "
                      "is the step the project brief (section 14) explicitly asks for and that no script "
                      "can substitute for.",
    }
    with open(out_dir / "gradcam_quality_check_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved {len(report_df)} Grad-CAM examples to {out_dir}")
    print(f"Wrote {out_dir / 'gradcam_quality_check_summary.json'}")
    print(f"Wrote {out_dir / 'gradcam_quality_check.csv'}")


if __name__ == "__main__":
    main()
