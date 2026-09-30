"""
torch Dataset + transforms for the RiceLeafDiseaseBD 6-class classifier.

Reads dataset/processed/classification/manifest.csv (built by
preprocessing/prepare_classification_dataset.py) rather than scanning
directories, because samples mix two different provenances (full images and
YOLO-box crops) that a plain torchvision.datasets.ImageFolder can't
distinguish -- and evaluate_classifier.py / evaluate_pipeline.py need that
`sample_type` column to break results down by input type.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.dataset_common import ALL_CLASSES  # noqa: E402

# Fixed canonical class <-> index mapping used EVERYWHERE the classifier is
# trained, evaluated, or served (training, Grad-CAM, inference pipeline).
CLASS_TO_IDX = {c: i for i, c in enumerate(ALL_CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 224, train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((int(image_size * 1.14), int(image_size * 1.14))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            # Field photos of leaves have no fixed canonical orientation the
            # way e.g. faces or digits do, so a mild vertical flip is a
            # legitimate augmentation here (kept low-probability since most
            # captures were still taken roughly leaf-up).
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            # Kept modest: lesion color (e.g. Brown Spot's brown center vs.
            # Blast's gray-white center) is a diagnostic signal, not noise.
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class RiceLeafClassificationDataset(Dataset):
    def __init__(self, manifest_csv: Path, split: str, transform: Optional[transforms.Compose] = None):
        df = pd.read_csv(manifest_csv)
        if split is not None:
            df = df[df["split"] == split]
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = CLASS_TO_IDX[row["label"]]
        return image, label

    def class_counts(self) -> dict:
        return self.df["label"].value_counts().to_dict()


def compute_class_weights(class_counts: dict, imbalance_threshold: float = 2.0):
    """Inverse-frequency class weights, normalized to mean 1.0 -- but only
    worth applying if the distribution is actually imbalanced (project brief
    section 7: "do not blindly use class weights, first analyze the
    distribution"). Returns (weights_tensor_or_None, diagnostics_dict).
    """
    counts = torch.tensor([max(class_counts.get(c, 0), 1) for c in ALL_CLASSES], dtype=torch.float)
    ratio = float(counts.max() / counts.min())
    diagnostics = {
        "counts_per_class": {c: int(class_counts.get(c, 0)) for c in ALL_CLASSES},
        "imbalance_ratio": round(ratio, 3),
        "threshold": imbalance_threshold,
        "weighting_applied": ratio > imbalance_threshold,
    }
    if ratio <= imbalance_threshold:
        return None, diagnostics
    weights = counts.sum() / (len(counts) * counts)
    diagnostics["weights_per_class"] = {c: round(float(w), 4) for c, w in zip(ALL_CLASSES, weights)}
    return weights, diagnostics
