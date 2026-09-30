"""
main.py

Single CLI front door for the whole RiceLeafDiseaseBD system. Every
subcommand just forwards to the corresponding module's own `main()` --
this file adds no new logic, only a convenient entrypoint so the project's
step-by-step workflow can be run as, in order:

    python main.py audit                                          # Step 1-2
    python main.py make-split
    python main.py prepare-yolo                                   # Step 3
    python main.py prepare-classification                         # Step 4
    python main.py train-yolo --model yolov8n.pt                  # Step 5
    python main.py evaluate-yolo --weights models/yolo_yolov8n/best.pt
    python main.py train-classifier --backbone resnet50           # Step 6
    python main.py evaluate-classifier --checkpoint models/classifier_resnet50/best.pt
    python main.py gradcam-check --checkpoint models/classifier_resnet50/best.pt   # Step 7
    python main.py predict --image path/to/leaf.jpg               # Step 8-9
    python main.py evaluate-pipeline --yolo-weights ... --classifier-checkpoint ...  # Step 10
    python main.py visualize                 # 640/classifier/pipeline comparison plots (final runs only)
    python main.py visualize-yolo             # full YOLO suite: confusion matrix, curves, training curves
    python main.py visualize-classifier       # full ResNet50 suite: confusion matrix, curves, full-vs-crop
    python main.py visualize-pipeline         # integrated-pipeline suite, from the existing saved evaluation
    python main.py visualize-dashboard        # one final PPT-ready figure + separate 1024-ablation figure
    python main.py summary

Run `python main.py <command> --help` to see that module's own options.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

COMMANDS = {
    "audit": ("preprocessing.dataset_audit", "Audit the raw dataset (Step 1-2)"),
    "make-split": ("preprocessing.make_split_manifest", "Build the master train/val/test split manifest"),
    "prepare-yolo": ("preprocessing.prepare_yolo_dataset", "Build the YOLO detection dataset (Step 3)"),
    "prepare-classification": ("preprocessing.prepare_classification_dataset", "Build the CNN classification dataset (Step 4)"),
    "train-yolo": ("detection.train_yolo", "Train YOLOv8 (Step 5)"),
    "evaluate-yolo": ("detection.evaluate_yolo", "Evaluate a trained YOLO checkpoint"),
    "predict-yolo": ("detection.predict_yolo", "Run YOLO alone on one image"),
    "train-classifier": ("classification.train_classifier", "Train the CNN classifier (Step 6)"),
    "evaluate-classifier": ("classification.evaluate_classifier", "Evaluate a trained classifier checkpoint"),
    "predict-classifier": ("classification.predict_classifier", "Run the CNN alone on one image"),
    "gradcam": ("explainability.gradcam", "Generate a Grad-CAM explanation for one image (Step 7)"),
    "gradcam-check": ("explainability.gradcam_quality_check", "Generate Grad-CAM examples for every class (Step 7)"),
    "recommend": ("recommendation.recommendation_engine", "Look up the treatment/management recommendation for one disease name"),
    "predict": ("pipeline.inference_pipeline", "Run the full YOLO->CNN->Grad-CAM->recommendation pipeline (Step 8-9)"),
    "evaluate-pipeline": ("evaluation.evaluate_pipeline", "Evaluate the integrated pipeline on the test split (Step 10)"),
    "visualize": ("evaluation.visualize_results", "Aggregate all results into comparison plots"),
    "visualize-yolo": ("evaluation.generate_yolo_plots", "Full YOLO visualization suite for the final 640 detector"),
    "visualize-classifier": ("evaluation.generate_classifier_plots", "Full ResNet50 visualization suite (confusion matrix, curves, full-vs-crop)"),
    "visualize-pipeline": ("evaluation.generate_pipeline_plots", "Integrated-pipeline visualization suite (from the existing saved evaluation, no rerun)"),
    "visualize-dashboard": ("evaluation.generate_final_dashboard", "One final PPT-ready dashboard figure, plus a separate labeled 1024-ablation figure"),
    "summary": ("evaluation.build_experiment_summary", "Build outputs/experiment_summary.json"),
}


def main():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="RiceLeafDiseaseBD: explainable rice-leaf disease detection + decision-support system.",
    )
    parser.add_argument("command", choices=list(COMMANDS.keys()))
    parser.add_argument("module_args", nargs=argparse.REMAINDER,
                         help="Arguments forwarded to the chosen subcommand's own parser")
    parsed = parser.parse_args()

    module_name, _ = COMMANDS[parsed.command]
    module = importlib.import_module(module_name)
    sys.argv = [module_name] + parsed.module_args
    module.main()


if __name__ == "__main__":
    main()
