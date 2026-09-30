"""
explainability/gradcam.py

Reusable Grad-CAM (Selvaraju et al., 2017) for the trained CNN classifier.

IMPORTANT (project brief section 13): Grad-CAM explains the CNN's
classification decision. It is NOT a direct explanation of YOLOv8's
detections -- YOLOv8 and the CNN are two separate models in this pipeline
(see pipeline/inference_pipeline.py); Grad-CAM only opens up the CNN half.

Works on both crops (YOLO-triggered verification) and full images (Healthy
/ no-detection path) -- the CNN was trained on both (see
preprocessing/prepare_classification_dataset.py), and this module makes no
assumption about which kind of image it's given.

Usage as a library (used by pipeline/inference_pipeline.py and
explainability/gradcam_quality_check.py):

    predictor = ClassifierPredictor(checkpoint_path)
    result = explain_image(predictor, "leaf.jpg", save_path="outputs/gradcam/leaf.png")
    # result: {"cam": np.ndarray[H,W] in [0,1], "target_class_name": str,
    #          "confidence": float, "all_probabilities": {...}, "saved_to": str}

CLI usage:
    python explainability/gradcam.py --checkpoint models/classifier_resnet50/best.pt --image leaf.jpg --out outputs/gradcam/leaf.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from classification.dataset import ALL_CLASSES, IDX_TO_CLASS  # noqa: E402
from classification.model import TARGET_LAYER_NAME  # noqa: E402


class GradCAM:
    """Hooks a target conv layer, runs one forward+backward pass, and turns
    the resulting activation/gradient pair into a class-activation heatmap.
    Always `close()` (or use as a context manager) to remove the hooks once
    done -- they otherwise silently accumulate on every call.
    """

    def __init__(self, model: torch.nn.Module, target_layer_name: Optional[str] = None):
        self.model = model
        self.model.eval()
        backbone_name = getattr(model, "_backbone_name", None)
        layer_name = target_layer_name or TARGET_LAYER_NAME.get(backbone_name, "layer4")
        modules = dict(model.named_modules())
        if layer_name not in modules:
            raise ValueError(f"Layer '{layer_name}' not found on this model. "
                              f"Available top-level modules: {list(dict(model.named_children()))}")
        self.target_layer = modules[layer_name]
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> dict:
        # Gradients must flow back to `target_layer` even if the backbone is
        # frozen (freeze_backbone=True), so force the input to require grad
        # rather than relying on parameter.requires_grad.
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        self.model.zero_grad(set_to_none=True)

        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1)
        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        score = output[0, target_class]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not fire -- check target_layer_name.")

        gradients = self.gradients[0]      # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        weights = gradients.mean(dim=(1, 2))  # global-average-pooled gradients, [C]
        cam = torch.einsum("c,chw->hw", weights, activations)
        cam = torch.relu(cam)

        cam_min, cam_max = cam.min(), cam.max()
        if (cam_max - cam_min) > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return {
            "cam": cam.detach().cpu().numpy(),
            "target_class": target_class,
            "target_class_name": IDX_TO_CLASS[target_class],
            "confidence": float(probs[0, target_class].item()),
            "all_probabilities": {ALL_CLASSES[i]: float(probs[0, i].item()) for i in range(len(ALL_CLASSES))},
        }

    def close(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def overlay_heatmap(pil_image: Image.Image, cam: np.ndarray, alpha: float = 0.45):
    """Resizes `cam` ([H,W] in [0,1]) to the image's resolution and blends a
    JET colormap over the original. Returns (heatmap_rgb, overlay_rgb) as
    uint8 HxWx3 numpy arrays.
    """
    img = np.array(pil_image.convert("RGB"))
    h, w = img.shape[:2]
    cam_resized = cv2.resize(cam.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap_rgb.astype(np.float32) + (1 - alpha) * img.astype(np.float32)).astype(np.uint8)
    return heatmap_rgb, overlay


def explain_image(predictor, image_path: Union[str, Path, Image.Image],
                   target_class: Optional[int] = None,
                   save_path: Optional[Union[str, Path]] = None) -> dict:
    """High-level convenience used by the inference pipeline and the quality
    check script: run the classifier's own preprocessing + Grad-CAM on one
    image, and optionally save a 3-panel (original / heatmap / overlay)
    figure. `predictor` is a classification.predict_classifier.ClassifierPredictor.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pil_image = Image.open(image_path).convert("RGB") if isinstance(image_path, (str, Path)) else image_path.convert("RGB")
    tensor = predictor.transform(pil_image).unsqueeze(0).to(predictor.device)

    cam_computer = GradCAM(predictor.model)
    try:
        result = cam_computer.generate(tensor, target_class=target_class)
    finally:
        cam_computer.close()

    heatmap, overlay = overlay_heatmap(pil_image, result["cam"])
    result["heatmap_rgb"] = heatmap
    result["overlay_rgb"] = overlay

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
        axes[0].imshow(pil_image)
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(heatmap)
        axes[1].set_title("Grad-CAM heatmap")
        axes[1].axis("off")
        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay: {result['target_class_name']} ({result['confidence']:.2f})")
        axes[2].axis("off")
        plt.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        result["saved_to"] = str(save_path)

    return result


def main():
    parser = argparse.ArgumentParser(description="Generate a Grad-CAM explanation for one image.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--target-class", type=str, default=None, help="Force a class name instead of the predicted one")
    parser.add_argument("--out", type=str, default="outputs/gradcam/example.png")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from classification.predict_classifier import ClassifierPredictor  # noqa: E402
    from classification.dataset import CLASS_TO_IDX  # noqa: E402

    predictor = ClassifierPredictor(args.checkpoint)
    target_idx = CLASS_TO_IDX[args.target_class] if args.target_class else None
    result = explain_image(predictor, args.image, target_class=target_idx, save_path=args.out)

    print(f"Predicted: {result['target_class_name']} (confidence={result['confidence']:.4f})")
    print(f"Saved: {result['saved_to']}")


if __name__ == "__main__":
    main()
