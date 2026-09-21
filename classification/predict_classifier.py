"""
classification/predict_classifier.py

Single-image / small-batch CNN prediction. Exposes `ClassifierPredictor`,
which pipeline/inference_pipeline.py and explainability/gradcam.py both
import directly, so there is exactly one place that knows how to load a
classifier checkpoint and preprocess an image for it.

CLI usage:
    python classification/predict_classifier.py --checkpoint models/classifier_resnet50/best.pt --image path/to/leaf.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Union

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from classification.dataset import ALL_CLASSES, IDX_TO_CLASS, build_transforms  # noqa: E402
from classification.model import build_model  # noqa: E402
from utils.hardware import detect_hardware  # noqa: E402


class ClassifierPredictor:
    def __init__(self, checkpoint_path: Union[str, Path], device: torch.device | None = None):
        self.checkpoint_path = Path(checkpoint_path)
        if device is None:
            profile = detect_hardware()
            device = torch.device("cuda:0" if profile.cuda_available else "cpu")
        self.device = device

        ckpt = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
        self.backbone = ckpt["backbone"]
        self.image_size = ckpt.get("image_size", 224)
        self.model = build_model(backbone=self.backbone, pretrained=False).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.transform = build_transforms(self.image_size, train=False)

    @torch.no_grad()
    def predict(self, image: Union[str, Path, Image.Image]) -> dict:
        """Predict on one PIL image (or a path to one). Returns the
        predicted class, its confidence, and the full class -> probability
        distribution -- callers decide what to do with low-confidence or
        disagreeing predictions (see pipeline/inference_pipeline.py), this
        function never hides uncertainty.
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
        pred_idx = int(probs.argmax())

        return {
            "class": IDX_TO_CLASS[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {ALL_CLASSES[i]: float(probs[i]) for i in range(len(ALL_CLASSES))},
        }

    def predict_batch(self, images: List[Union[str, Path, Image.Image]]) -> List[dict]:
        return [self.predict(img) for img in images]


def main():
    parser = argparse.ArgumentParser(description="Run the CNN classifier on one image.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    predictor = ClassifierPredictor(args.checkpoint)
    result = predictor.predict(args.image)
    print(f"Predicted: {result['class']}  (confidence={result['confidence']:.4f})")
    for cls, p in sorted(result["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"  {cls:15s} {p:.4f}")


if __name__ == "__main__":
    main()
