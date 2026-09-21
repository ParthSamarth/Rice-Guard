"""
recommendation/recommendation_engine.py

A rule-based treatment/management recommendation engine, deliberately kept
INDEPENDENT of the ML models (project brief section 15: "The ML model must
NOT directly generate pesticide advice. The recommendation engine should be
independent of the model."). It does exactly one thing: given a confirmed
disease NAME (a plain string -- it has no idea whether that name came from
YOLO, the CNN, or a human typing it in), look it up in
recommendation/knowledge_base.json and return the stored fields verbatim.

It never calls a model, never computes a confidence score, and never
invents content beyond what's in the knowledge base -- if the knowledge
base marks a field as needing authoritative validation, this engine passes
that flag straight through rather than hiding it.

Usage:
    from recommendation.recommendation_engine import RecommendationEngine
    engine = RecommendationEngine()
    engine.get_recommendation("Brown Spot")
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.dataset_common import ALL_CLASSES, normalize_class_name  # noqa: E402

DEFAULT_KB_PATH = Path(__file__).resolve().parent / "knowledge_base.json"


class RecommendationEngine:
    def __init__(self, kb_path: Path = DEFAULT_KB_PATH):
        self.kb_path = Path(kb_path)
        with open(self.kb_path, "r", encoding="utf-8") as f:
            kb = json.load(f)
        self.disclaimer = kb.get("_disclaimer", "")
        self.classes = kb["classes"]

        missing = [c for c in ALL_CLASSES if c not in self.classes]
        if missing:
            raise ValueError(f"knowledge_base.json is missing entries for: {missing}")

    def get_recommendation(self, disease_name: Optional[str]) -> dict:
        """Returns the full knowledge-base entry for `disease_name`, plus the
        global disclaimer. If `disease_name` doesn't normalize to one of the
        6 known classes (e.g. None, or a genuinely unrecognized string --
        which should not happen downstream of the classifier, but this
        function makes no assumption about its caller), `available` is False
        and no content is fabricated.
        """
        canonical = normalize_class_name(disease_name) if disease_name else None
        if canonical is None or canonical not in self.classes:
            return {
                "available": False,
                "requested": disease_name,
                "reason": f"'{disease_name}' does not match any of the {len(ALL_CLASSES)} known classes "
                          f"{ALL_CLASSES}.",
            }

        entry = self.classes[canonical]
        return {
            "available": True,
            "disease": canonical,
            "causal_agent": entry.get("causal_agent"),
            "description": entry.get("description"),
            "symptoms": entry.get("symptoms", []),
            "management": entry.get("management", []),
            "treatment": entry.get("treatment", []),
            "prevention": entry.get("prevention", []),
            "requires_expert_validation": entry.get("requires_expert_validation", True),
            "disclaimer": self.disclaimer,
        }


def main():
    parser = argparse.ArgumentParser(description="Look up the treatment/management recommendation for a disease.")
    parser.add_argument("--disease", type=str, required=True)
    parser.add_argument("--kb", type=str, default=str(DEFAULT_KB_PATH))
    args = parser.parse_args()

    engine = RecommendationEngine(Path(args.kb))
    result = engine.get_recommendation(args.disease)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
