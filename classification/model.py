"""
Factory for the CNN classifiers used in this project: ImageNet-pretrained
ResNet18 / ResNet50 with a fresh linear head for the 6 rice-leaf classes.

ResNet50 is the primary/recommended backbone (project brief section 7);
ResNet18 is trained alongside it as the lighter comparison point (project
brief section 18, Experiments 3 & 4).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.dataset_common import ALL_CLASSES  # noqa: E402

NUM_CLASSES = len(ALL_CLASSES)

_BACKBONES = {
    "resnet18": (models.resnet18, "ResNet18_Weights", "IMAGENET1K_V1"),
    "resnet50": (models.resnet50, "ResNet50_Weights", "IMAGENET1K_V2"),
}

# Grad-CAM (explainability/gradcam.py) hooks the last conv block by name.
TARGET_LAYER_NAME = {"resnet18": "layer4", "resnet50": "layer4"}


def build_model(backbone: str = "resnet50", num_classes: int = NUM_CLASSES,
                 pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    backbone = backbone.lower()
    if backbone not in _BACKBONES:
        raise ValueError(f"Unsupported backbone '{backbone}'. Choose one of {list(_BACKBONES)}.")

    ctor, weights_enum_name, weights_tag = _BACKBONES[backbone]
    weights = None
    if pretrained:
        weights_enum = getattr(models, weights_enum_name)
        weights = getattr(weights_enum, weights_tag)
    model = ctor(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)  # always trainable, even when the backbone is frozen
    model._backbone_name = backbone  # read by explainability/gradcam.py to pick the right target layer
    return model


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}
