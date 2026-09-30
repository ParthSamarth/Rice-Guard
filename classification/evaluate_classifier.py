"""
classification/evaluate_classifier.py

Full evaluation of a trained checkpoint on a chosen split (default: test).
Reports accuracy, precision/recall/F1 (per-class, macro, weighted),
confusion matrix, and the full sklearn classification report -- only ever
computed from actual model predictions, never fabricated.

Also reports the SAME metrics separately for full-image samples vs.
YOLO-crop samples (never mixed silently) -- the CNN is trained on both
(preprocessing/prepare_classification_dataset.py) because it faces both at
inference time (no-detection path vs. YOLO-triggered verification path;
see pipeline/inference_pipeline.py), and those two input distributions can
have genuinely different difficulty, so collapsing them into one number
would hide that.

Usage:
    python classification/evaluate_classifier.py --checkpoint models/classifier_resnet50/best.pt
    python classification/evaluate_classifier.py --checkpoint models/classifier_resnet18/best.pt --split test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from classification.dataset import ALL_CLASSES, RiceLeafClassificationDataset, build_transforms  # noqa: E402
from classification.model import build_model  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.hardware import detect_hardware  # noqa: E402


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    t0 = time.time()
    n = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.cpu().tolist())
        n += images.size(0)
    elapsed = time.time() - t0
    return all_preds, all_labels, all_probs, elapsed, n


def compute_metrics_block(labels, preds) -> dict:
    """One self-contained metrics block (accuracy/macro/weighted/per-class/
    confusion matrix) -- reused for the full split and for the full-image-only
    and crop-only subsets so all three are computed identically.
    """
    if len(labels) == 0:
        return {"num_samples": 0}
    labels_arr, preds_arr = np.array(labels), np.array(preds)
    acc = float(np.mean(preds_arr == labels_arr))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(ALL_CLASSES))), zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=list(range(len(ALL_CLASSES))))
    per_class = {
        ALL_CLASSES[i]: {"precision": float(precision[i]), "recall": float(recall[i]),
                          "f1": float(f1[i]), "support": int(support[i])}
        for i in range(len(ALL_CLASSES))
    }
    return {
        "num_samples": int(len(labels)), "accuracy": acc,
        "macro_precision": float(macro_p), "macro_recall": float(macro_r), "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p), "weighted_recall": float(weighted_r), "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(), "confusion_matrix_labels": ALL_CLASSES,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    profile = detect_hardware()
    device = torch.device("cuda:0" if profile.cuda_available else "cpu")

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone = ckpt["backbone"]
    image_size = ckpt.get("image_size", 224)

    model = build_model(backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded {backbone} checkpoint from {ckpt_path} (epoch {ckpt.get('epoch')})")

    manifest_csv = config.path("paths", "classification_dataset") / "manifest.csv"
    ds = RiceLeafClassificationDataset(manifest_csv, args.split, build_transforms(image_size, train=False))
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    print(f"Evaluating on '{args.split}' split: {len(ds)} samples")

    preds, labels, probs, elapsed, n = run_inference(model, loader, device)
    sample_types = ds.df["sample_type"].tolist()  # loader has shuffle=False, so this order matches preds/labels

    overall = compute_metrics_block(labels, preds)
    report_txt = classification_report(labels, preds, target_names=ALL_CLASSES, zero_division=0)
    report_dict = classification_report(labels, preds, target_names=ALL_CLASSES, zero_division=0, output_dict=True)

    print(f"\nAccuracy: {overall['accuracy']:.4f} | Macro F1: {overall['macro_f1']:.4f} | "
          f"Weighted F1: {overall['weighted_f1']:.4f}")
    print(f"Throughput: {n / elapsed:.1f} img/s ({elapsed:.1f}s for {n} images)\n")
    print(report_txt)

    # Full-image vs. YOLO-crop breakdown -- never mixed silently (project brief
    # section 16): these are two different input distributions the CNN faces
    # at inference time and can have genuinely different difficulty.
    by_type = {}
    for sample_type in ("full", "crop"):
        idx = [i for i, st in enumerate(sample_types) if st == sample_type]
        sub_labels = [labels[i] for i in idx]
        sub_preds = [preds[i] for i in idx]
        by_type[sample_type] = compute_metrics_block(sub_labels, sub_preds)
        block = by_type[sample_type]
        if block["num_samples"]:
            print(f"[{sample_type} samples only] n={block['num_samples']} accuracy={block['accuracy']:.4f} "
                  f"macro_f1={block['macro_f1']:.4f} weighted_f1={block['weighted_f1']:.4f}")
        else:
            print(f"[{sample_type} samples only] n=0 (e.g. Healthy has no crops)")

    # ---- outputs ----
    run_tag = ckpt_path.parent.name.replace("classifier_", "")
    out_dir = config.path("paths", "outputs") / "classification" / run_tag / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    cm = np.array(overall["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=ALL_CLASSES, yticklabels=ALL_CLASSES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix -- {backbone} ({args.split})")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(out_dir / f"confusion_matrix_{args.split}.png", dpi=150)
    plt.close(fig)

    # Row-normalized version -- easier to read per-class recall visually when classes are imbalanced.
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=ALL_CLASSES, yticklabels=ALL_CLASSES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (row-normalized) -- {backbone} ({args.split})")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(out_dir / f"confusion_matrix_{args.split}_normalized.png", dpi=150)
    plt.close(fig)

    metrics = {
        "checkpoint": str(ckpt_path), "backbone": backbone, "split": args.split,
        **overall,
        "throughput_img_per_sec": round(n / elapsed, 2),
        "classification_report": report_dict,
        "by_sample_type": by_type,
    }
    with open(out_dir / f"metrics_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nWrote {out_dir / f'metrics_{args.split}.json'}")
    print(f"Wrote {out_dir / f'confusion_matrix_{args.split}.png'}")


if __name__ == "__main__":
    main()
