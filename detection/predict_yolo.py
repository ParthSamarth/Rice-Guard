"""
detection/predict_yolo.py

Runs a trained YOLO checkpoint on one image and returns structured
detections. Exposes `YoloDetector`, which pipeline/inference_pipeline.py
imports directly -- one place that knows how to load YOLO weights and
normalize Ultralytics' `Results` object into plain dicts the rest of the
project can serialize to JSON.

CLI usage:
    python detection/predict_yolo.py --weights models/yolo_yolov8n/best.pt --image path/to/leaf.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import load_config  # noqa: E402
from utils.hardware import detect_hardware  # noqa: E402


class YoloDetector:
    def __init__(self, weights_path: Union[str, Path], confidence_threshold: float = 0.25, device=None):
        from ultralytics import YOLO

        self.weights_path = Path(weights_path)
        self.model = YOLO(str(self.weights_path))
        self.confidence_threshold = confidence_threshold
        if device is None:
            profile = detect_hardware()
            device = 0 if profile.cuda_available else "cpu"
        self.device = device
        self.class_names = self.model.names  # {id: name}, taken from the checkpoint itself

    def detect(self, image: Union[str, Path]) -> dict:
        """Returns:
            {
              "image": str(image),
              "detected": bool,
              "detections": [
                 {"class": str, "class_id": int, "confidence": float,
                  "bbox_xyxy": [x0,y0,x1,y1], "bbox_normalized_xywh": [cx,cy,w,h]}
              ]
            }
        """
        results = self.model.predict(
            source=str(image), conf=self.confidence_threshold, device=self.device, verbose=False
        )
        result = results[0]

        detections = []
        img_h, img_w = result.orig_shape
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x0, y0, x1, y1 = [float(v) for v in box.xyxy[0].tolist()]
                cx = ((x0 + x1) / 2) / img_w
                cy = ((y0 + y1) / 2) / img_h
                w = (x1 - x0) / img_w
                h = (y1 - y0) / img_h
                detections.append({
                    "class": self.class_names.get(cls_id, str(cls_id)),
                    "class_id": cls_id,
                    "confidence": conf,
                    "bbox_xyxy": [x0, y0, x1, y1],
                    "bbox_normalized_xywh": [cx, cy, w, h],
                })

        detections.sort(key=lambda d: -d["confidence"])
        return {
            "image": str(image),
            "image_size": {"width": img_w, "height": img_h},
            "detected": len(detections) > 0,
            "detections": detections,
        }


def main():
    parser = argparse.ArgumentParser(description="Run YOLO disease detection on one image.")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    detector = YoloDetector(args.weights, confidence_threshold=args.conf)
    result = detector.detect(args.image)

    if not result["detected"]:
        print("No disease regions detected.")
    for det in result["detections"]:
        print(f"  {det['class']:15s} conf={det['confidence']:.4f}  bbox_xyxy={[round(v,1) for v in det['bbox_xyxy']]}")


if __name__ == "__main__":
    main()
