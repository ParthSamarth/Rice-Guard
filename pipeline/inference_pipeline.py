"""
pipeline/inference_pipeline.py

The end-to-end system:

    INPUT IMAGE -> domain gate (is this plausibly a rice leaf at all?)
                     |
          recognized?  --no--> STOP. status="not_recognized", no disease
             |yes                name/confidence/regions/Grad-CAM are ever
             v                   produced for this image (see domain_gate.py)
                   YOLOv8 (disease regions only)
                     |
        found regions?  --no--> CNN classifies the FULL image (only path
             |yes                that can output "Healthy" -- YOLO never
             v                   does, by design; see README.md)
    crop each region (same margin as training-time crops) -> CNN verifies
    each crop -> per-region YOLO/CNN agreement check
                     |
                     v
            Grad-CAM on whatever the CNN looked at (crop or full frame --
            explains the CNN's decision only, never YOLO's)
                     |
                     v
            treatment recommendation lookup (independent rule-based engine,
            keyed ONLY on the final disease name string)
                     |
                     v
              structured FINAL RESULT (see `run()` docstring for the schema)

Decision logic:
  1. Run the domain gate (pipeline/domain_gate.py) on the full image. This
     is independent of, and runs before, the disease classifier -- it is
     NOT the same thing as "low confidence" (see that module's docstring for
     why cnn_confidence alone cannot tell "a hard rice leaf" apart from "not
     a rice leaf": genuine correctly-classified test-set leaves score as low
     as ~33%). If the gate rejects the image, `status` becomes
     "not_recognized", YOLO/CNN/Grad-CAM/recommendation never run for this
     request, and no disease name is ever produced.
  2. Otherwise, run YOLOv8 on the full image (5-class disease detector).
  3. If YOLO finds >=1 region above `thresholds.yolo_confidence`: crop and
     CNN-classify EVERY region (project brief section 10 -- never just the
     top one), and check YOLO/CNN agreement per region. The image-level
     "final" result is taken from the single highest-CNN-confidence region;
     every other region's full result is still returned in `regions`, never
     hidden. If that top region's YOLO and CNN classes disagree, `status`
     becomes "model_disagreement" and `final_disease` is left null --this
     system does not silently pick a winner (project brief section 12).
  4. If YOLO finds nothing, the CNN classifies the FULL image. If its
     confidence is below `thresholds.cnn_confidence`, `status` becomes
     "low_confidence" rather than reporting a shaky answer as certain.
  5. The recommendation engine is looked up from the final disease NAME
     string only -- it never sees model internals (project brief section 15).

Usage:
    python pipeline/inference_pipeline.py --image path/to/leaf.jpg
    python pipeline/inference_pipeline.py --image-dir path/to/folder --out outputs/pipeline_predictions
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Union

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from classification.predict_classifier import ClassifierPredictor  # noqa: E402
from detection.predict_yolo import YoloDetector  # noqa: E402
from explainability.gradcam import explain_image  # noqa: E402
from pipeline.domain_gate import DomainGate  # noqa: E402
from recommendation.recommendation_engine import RecommendationEngine  # noqa: E402
from utils.config import load_config, Config  # noqa: E402
from utils.dataset_common import HEALTHY, expand_and_clip_box, normalize_class_name  # noqa: E402


class RiceDiseasePipeline:
    def __init__(self, yolo_weights: Union[str, Path], classifier_checkpoint: Union[str, Path],
                 config: Optional[Config] = None, kb_path: Optional[Union[str, Path]] = None,
                 yolo_conf: Optional[float] = None, cnn_conf: Optional[float] = None,
                 gradcam_dir: Optional[Union[str, Path]] = None, save_gradcam: bool = True):
        self.config = config or load_config()
        thresholds = self.config["thresholds"]
        self.yolo_conf_threshold = yolo_conf if yolo_conf is not None else thresholds["yolo_confidence"]
        self.cnn_conf_threshold = cnn_conf if cnn_conf is not None else thresholds["cnn_confidence"]
        self.crop_margin = self.config["classification_dataset"]["crop_context_margin"]
        self.save_gradcam = save_gradcam

        print(f"Loading YOLO from {yolo_weights} ...")
        self.yolo = YoloDetector(yolo_weights, confidence_threshold=self.yolo_conf_threshold)
        print(f"Loading CNN from {classifier_checkpoint} ...")
        self.classifier = ClassifierPredictor(classifier_checkpoint)
        print("Loading domain gate (generic ImageNet classifier, rice-leaf validation) ...")
        self.domain_gate = DomainGate()
        self.recommender = RecommendationEngine(Path(kb_path)) if kb_path else RecommendationEngine()

        self.gradcam_dir = Path(gradcam_dir) if gradcam_dir else self.config.path("paths", "outputs") / "gradcam"
        if self.save_gradcam:
            self.gradcam_dir.mkdir(parents=True, exist_ok=True)

    def _gradcam_for(self, image: Image.Image, tag: str):
        if not self.save_gradcam:
            return None
        save_path = self.gradcam_dir / f"{tag}.png"
        try:
            explain_image(self.classifier, image, save_path=save_path)
            return str(save_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: Grad-CAM generation failed for {tag}: {exc}")
            return None

    def run(self, image_path: Union[str, Path]) -> dict:
        """Returns a dict matching (a superset of) the project brief's output
        schema:

            image, final_disease, final_confidence, status, decision_path,
            yolo_prediction, yolo_confidence, cnn_prediction, cnn_confidence
            (the flat disagreement-schema fields -- always populated when a
            YOLO/CNN comparison was made, i.e. whenever a region was
            detected; yolo_prediction/yolo_confidence are None on the
            no-detection path, since there is no YOLO prediction to compare),
            yolo {detected, detections} / detections (alias),
            cnn {class, confidence}, regions / cnn_verification (alias, full
            per-region detail -- project brief section 10/section 21),
            agreement, agreement_detail (populated with this same
            yolo/cnn_prediction+confidence shape only when
            status == "model_disagreement" -- project brief section 18),
            gradcam / gradcam_path (alias), recommendation, thresholds_used,
            processing_time_sec, domain_gate (the gate's own check result --
            see domain_gate.py; present on every response, not just rejected
            ones, so a caller can see how close a borderline "ok" result
            came to being rejected)

        status == "not_recognized" is a fifth possible status value (beyond
        ok/low_confidence/model_disagreement/error): the domain gate rejected
        the image before disease classification ever ran. final_disease,
        final_confidence, yolo_prediction/cnn_prediction, regions, gradcam,
        and recommendation are all None/empty on this path -- never a
        disease name for an image the gate didn't recognize as a rice leaf.
        """
        image_path = Path(image_path)
        t0 = time.time()
        full_image = Image.open(image_path).convert("RGB")
        img_w, img_h = full_image.size

        domain_check = self.domain_gate.check(full_image)
        if not domain_check["is_recognized"]:
            return {
                "image": str(image_path),
                "final_disease": None,
                "final_confidence": None,
                "status": "not_recognized",
                "decision_path": "not_recognized",
                "yolo_prediction": None,
                "yolo_confidence": None,
                "cnn_prediction": None,
                "cnn_confidence": None,
                "yolo": {"detected": False, "detections": []},
                "detections": [],
                "cnn": {"class": None, "confidence": None},
                "cnn_verification": [],
                "regions": [],
                "agreement": True,  # nothing to disagree with -- disease classification never ran
                "agreement_detail": None,
                "gradcam": None,
                "gradcam_path": None,
                "recommendation": None,
                "thresholds_used": {"yolo_confidence": self.yolo_conf_threshold, "cnn_confidence": self.cnn_conf_threshold},
                "processing_time_sec": round(time.time() - t0, 3),
                "domain_gate": domain_check,
            }

        yolo_result = self.yolo.detect(image_path)
        regions = []

        if yolo_result["detected"]:
            decision_path = "yolo_cnn_verification"
            for i, det in enumerate(yolo_result["detections"]):
                cx, cy, w, h = det["bbox_normalized_xywh"]
                x0, y0, x1, y1 = expand_and_clip_box(cx, cy, w, h, img_w, img_h, self.crop_margin)
                crop = full_image.crop((x0, y0, x1, y1))
                cnn_pred = self.classifier.predict(crop)
                agrees = normalize_class_name(det["class"]) == normalize_class_name(cnn_pred["class"])
                gradcam_path = self._gradcam_for(crop, f"{image_path.stem}_region{i}")

                regions.append({
                    "region_index": i,
                    "yolo_class": det["class"],
                    "yolo_confidence": det["confidence"],
                    "bbox_xyxy": det["bbox_xyxy"],
                    "cnn_class": cnn_pred["class"],
                    "cnn_confidence": cnn_pred["confidence"],
                    "cnn_probabilities": cnn_pred["probabilities"],
                    "agreement": agrees,
                    "gradcam": gradcam_path,
                })

            # Image-level decision: the region the CNN is most confident
            # about (documented aggregation rule; every region's own result
            # is still available in `regions` above -- nothing is discarded).
            best_region = max(regions, key=lambda r: r["cnn_confidence"])

            if best_region["agreement"]:
                final_disease = best_region["cnn_class"]
                final_confidence = best_region["cnn_confidence"]
                status = "ok" if final_confidence >= self.cnn_conf_threshold else "low_confidence"
                agreement_detail = None
            else:
                final_disease = None
                final_confidence = None
                status = "model_disagreement"
                agreement_detail = {
                    "status": "model_disagreement",
                    "yolo_prediction": best_region["yolo_class"],
                    "cnn_prediction": best_region["cnn_class"],
                    "yolo_confidence": best_region["yolo_confidence"],
                    "cnn_confidence": best_region["cnn_confidence"],
                }

            cnn_summary = {"class": best_region["cnn_class"], "confidence": best_region["cnn_confidence"]}
            gradcam_summary = best_region["gradcam"]
            agreement_bool = bool(best_region["agreement"])
            yolo_prediction, yolo_confidence = best_region["yolo_class"], best_region["yolo_confidence"]
            cnn_prediction, cnn_confidence = best_region["cnn_class"], best_region["cnn_confidence"]

        else:
            decision_path = "cnn_only_full_image"
            cnn_pred = self.classifier.predict(full_image)
            gradcam_path = self._gradcam_for(full_image, f"{image_path.stem}_full")
            final_disease = cnn_pred["class"]
            final_confidence = cnn_pred["confidence"]
            status = "ok" if final_confidence >= self.cnn_conf_threshold else "low_confidence"
            agreement_detail = None
            regions = [{
                "region_index": None, "yolo_class": None, "yolo_confidence": None, "bbox_xyxy": None,
                "cnn_class": cnn_pred["class"], "cnn_confidence": cnn_pred["confidence"],
                "cnn_probabilities": cnn_pred["probabilities"], "agreement": None,
                "gradcam": gradcam_path,
            }]
            cnn_summary = {"class": cnn_pred["class"], "confidence": cnn_pred["confidence"]}
            gradcam_summary = gradcam_path
            agreement_bool = True  # nothing to disagree with -- single model, single opinion
            # No YOLO region exists to compare against on this path -- this is
            # NOT itself a "disagreement" (YOLO made no competing prediction,
            # it simply found nothing), but the CNN's evidence is still
            # reported in full rather than silently accepted (project brief
            # section 19: "preserve the CNN evidence and confidence, make the
            # result transparent" even when the final label is a disease that
            # YOLO never localized).
            yolo_prediction, yolo_confidence = None, None
            cnn_prediction, cnn_confidence = cnn_pred["class"], cnn_pred["confidence"]

        recommendation = None
        if final_disease is not None:
            recommendation = self.recommender.get_recommendation(final_disease)

        return {
            "image": str(image_path),
            "final_disease": final_disease,
            "final_confidence": final_confidence,
            "status": status,
            "decision_path": decision_path,
            # Top-level yolo_prediction/cnn_prediction pair: always the
            # "best region" (or full-image) comparison, populated whether or
            # not the two models agreed -- see project brief section 18 for
            # the exact disagreement schema these satisfy.
            "yolo_prediction": yolo_prediction,
            "yolo_confidence": yolo_confidence,
            "cnn_prediction": cnn_prediction,
            "cnn_confidence": cnn_confidence,
            "yolo": {"detected": yolo_result["detected"], "detections": yolo_result["detections"]},
            "detections": yolo_result["detections"],  # alias of yolo.detections
            "cnn": cnn_summary,
            "cnn_verification": regions,  # alias of `regions`
            "regions": regions,
            "agreement": agreement_bool,
            "agreement_detail": agreement_detail,
            "gradcam": gradcam_summary,
            "gradcam_path": gradcam_summary,  # alias of `gradcam`
            "recommendation": recommendation,
            "thresholds_used": {"yolo_confidence": self.yolo_conf_threshold, "cnn_confidence": self.cnn_conf_threshold},
            "processing_time_sec": round(time.time() - t0, 3),
            "domain_gate": domain_check,
        }

    def run_batch(self, image_paths, out_dir: Optional[Union[str, Path]] = None) -> list:
        results = []
        out_dir = Path(out_dir) if out_dir else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
        for p in image_paths:
            try:
                result = self.run(p)
            except Exception as exc:  # noqa: BLE001
                result = {"image": str(p), "status": "error", "error": str(exc)}
            results.append(result)
            if out_dir:
                with open(out_dir / f"{Path(p).stem}.json", "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
        return results


def main():
    parser = argparse.ArgumentParser(description="Run the full rice-disease detection + decision support pipeline.")
    parser.add_argument("--yolo-weights", type=str, default=None)
    parser.add_argument("--classifier-checkpoint", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--out", type=str, default=None, help="Output directory for --image-dir batch runs")
    parser.add_argument("--no-gradcam", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Provide --image or --image-dir")

    config = load_config(args.config) if args.config else load_config()
    yolo_weights = args.yolo_weights or str(config.path("paths", "models") / "yolo_yolov8n" / "best.pt")
    classifier_checkpoint = args.classifier_checkpoint or str(config.path("paths", "models") / "classifier_resnet50" / "best.pt")

    pipeline = RiceDiseasePipeline(
        yolo_weights, classifier_checkpoint, config=config, save_gradcam=not args.no_gradcam
    )

    if args.image:
        result = pipeline.run(args.image)
        print(json.dumps(result, indent=2))
    else:
        image_dir = Path(args.image_dir)
        image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        out_dir = args.out or (config.path("paths", "outputs") / "pipeline_predictions")
        print(f"Running pipeline on {len(image_paths)} images from {image_dir} ...")
        results = pipeline.run_batch(image_paths, out_dir=out_dir)
        n_ok = sum(1 for r in results if r.get("status") == "ok")
        n_disagree = sum(1 for r in results if r.get("status") == "model_disagreement")
        n_low_conf = sum(1 for r in results if r.get("status") == "low_confidence")
        n_error = sum(1 for r in results if r.get("status") == "error")
        print(f"Done: ok={n_ok} model_disagreement={n_disagree} low_confidence={n_low_conf} error={n_error}")
        print(f"Wrote per-image JSON to {out_dir}")


if __name__ == "__main__":
    main()
