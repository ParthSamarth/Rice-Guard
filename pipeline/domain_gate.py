"""
pipeline/domain_gate.py -- rejects images that aren't plausibly a rice leaf
BEFORE disease classification runs, so the 6-way disease classifier (which
always has to pick one of its 6 classes, by construction -- softmax always
sums to 1) never gets to hand out a disease name for e.g. a laptop photo.

Why this can't just be "raise the cnn_confidence threshold": calibrated
against this project's own held-out test split
(outputs/pipeline_evaluation/pipeline_eval_test.csv), *genuine, correctly
classified* rice leaf photos go as low as ~33% confidence. A wrong-looking
low confidence number does not mean "not a rice leaf" -- it can just mean
"a hard rice leaf photo." The disease classifier's own confidence is not a
reliable signal for this and is deliberately NOT used for the reject
decision below.

Approach (no new training -- this repo is in a documented finalization
freeze; see PROJECT_README.md -- so this reuses an off-the-shelf, already
zero-shot-available pretrained model rather than training a new one):

  A generic ImageNet-pretrained ResNet50 (torchvision, downloaded once and
  cached -- NOT the fine-tuned 6-class disease classifier) looks at the same
  image. If its top-1 prediction both (a) falls in a curated range of
  ImageNet's clearly man-made/object/indoor/vehicle/furniture classes and
  (b) is confident about it, that is strong independent evidence the photo
  isn't a rice leaf at all -- a laptop photo confidently looks like a
  "laptop" to a generic classifier; a rice leaf photo does not confidently
  look like anything in that range (empirically, on this project's own test
  images, it scatters low-confidence guesses across insect/macro-photography
  classes like "damselfly"/"ladybug"/"leafhopper", or lands on the
  agriculture-adjacent "ear" (of corn) class -- see calibration notes below).

Calibration (2026-08-20, against outputs/pipeline_evaluation/pipeline_eval_test.csv,
this project's own held-out test split -- no new images were collected):
  - 400 real rice-leaf test images sampled: only 4 had their ImageNet top-1
    land in the man-made-object reject range AT ALL, and the highest
    confidence among those four was 0.366 ("paddle", for a genuine Rice
    Tungro leaf -- a real, if rare, visual confusion). REJECT_CONFIDENCE=0.40
    sits just above that empirical ceiling, so it should not false-reject
    real leaves like that one.
  - A first, separate 120-image sample found max overall top-1 confidence
    0.569 ("ear", i.e. corn ear -- an agricultural class, correctly excluded
    from the reject range).
  - Known false-negative case found during manual testing: a full desktop
    screenshot (an unusual, UI-chrome-heavy image, not a typical "wrong
    object" photo) landed at exactly "web site" 0.357 -- just BELOW
    REJECT_CONFIDENCE, so it was NOT rejected. A more typical confident
    wrong-object photo (e.g. the same kind of screenshot cropped tighter)
    hit "web site" at 0.708 and WAS correctly rejected. This is the honest
    shape of the tradeoff: this heuristic reliably catches confident,
    canonical "wrong object" photos (laptops, keyboards, furniture, rooms --
    the failure mode that was actually reported), but an ambiguous or
    atypical non-leaf image sitting near the 0.35-0.40 boundary can still
    slip through. It is not a guarantee for every possible input, and
    should be re-calibrated (raise/lower REJECT_CONFIDENCE, or sample a
    larger/more diverse negative set) if false accepts/rejects show up in
    real use -- see this file's `check()` and the smoke-test pattern used
    to derive these numbers for how to re-run calibration.

Second signal -- REJECT_MASS_THRESHOLD (added 2026-08-20 after the top-1-only
check above was found to still miss the originally-reported laptop photo on
live re-test): a busy indoor scene (a laptop/keyboard/desk) does NOT always
produce one dominant confident top-1 guess -- probability can split across
several correlated man-made classes ("cellular telephone" 0.14, "computer
keyboard" 0.09, "notebook" 0.03, "space bar" 0.03, "modem" 0.02 for the
reported photo), each individually under REJECT_CONFIDENCE, so the top-1-only
check above let it through with `is_recognized=True`. The fix is a second,
independent check on the SAME softmax output: the TOTAL probability mass
across the whole man-made-object range (398-919, same exceptions), not just
the top-1 slice of it. Re-running this file's calibration set
(outputs/pipeline_evaluation/pipeline_eval_test.csv, all 1458 held-out
images, not just the CNN's own confidence) for this second metric found a
clean separation: max reject-range mass across every real leaf in that
split was 0.664 (a Rice Tungro leaf; 99.5th percentile 0.524), while the
reported laptop photo scored 0.749, a second down-sized copy of the same
photo scored 0.750, and other known non-leaf images (a person -- 0.844; two
UI-diagram screenshots already caught by the top-1 check anyway -- 0.771 and
0.786) all scored well above the real-leaf ceiling. REJECT_MASS_THRESHOLD is
set to 0.70 -- above every real leaf seen in the full held-out split, with
comfortable margin below the confirmed non-leaf scores. An image is now
rejected if EITHER the top-1 check OR this aggregate-mass check fires; the
top-1 check still catches its own cases (e.g. a person: top-1 "suit" 0.40)
independently. Still not a guarantee for every input: a UI-heavy screenshot
found during this same re-test scored 0.613 -- inside the real-leaf range on
this metric too -- so screenshot-style images remain the documented gap
above, unchanged by this addition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

# ImageNet-1k class indices 398-919 are, with one exception, exclusively
# man-made objects/scenes/vehicles/furniture/clothing/tools/instruments (see
# torchvision.models.ResNet50_Weights.IMAGENET1K_V2.meta["categories"] for
# the full ordered list) -- 0-397 is almost entirely animals, and 920-999 is
# food/fungi/plants/nature scenes. 408 ("amphibian") is the one outlier in
# the object block and is excluded since it isn't man-made.
_REJECT_RANGE = range(398, 920)
_REJECT_RANGE_EXCEPTIONS = {408}  # "amphibian"

REJECT_CONFIDENCE = 0.40
REJECT_MASS_THRESHOLD = 0.70  # total probability mass across _REJECT_RANGE; see "Second signal" above


class DomainGate:
    """Loads its own generic ImageNet-pretrained backbone -- deliberately
    separate from ClassifierPredictor's fine-tuned 6-class model, which
    cannot answer "is this a rice leaf at all" since it never saw non-leaf
    images during training and always has to pick one of its 6 classes."""

    def __init__(self, device: torch.device | None = None):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        weights = ResNet50_Weights.IMAGENET1K_V2
        self.model = resnet50(weights=weights).to(device).eval()
        self.categories = weights.meta["categories"]
        self.preprocess = weights.transforms()
        # Boolean mask over the 1000 ImageNet classes, precomputed once, so
        # `check()` can sum the reject-range probability mass with a single
        # masked sum instead of a 1000-iteration Python loop per image.
        reject_mask = torch.zeros(len(self.categories), dtype=torch.bool)
        for idx in _REJECT_RANGE:
            if idx not in _REJECT_RANGE_EXCEPTIONS:
                reject_mask[idx] = True
        self._reject_mask = reject_mask.to(device)

    @torch.no_grad()
    def check(self, image: Union[str, Path, Image.Image]) -> dict:
        """Returns {"is_recognized": bool, "reason": str | None,
        "imagenet_top1_class": str, "imagenet_top1_confidence": float,
        "imagenet_reject_mass": float}. `reason` is a short, non-technical
        explanation suitable for logging (never shown to the end user as-is
        -- the client owns that copy)."""
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        image = image.convert("RGB")

        batch = self.preprocess(image).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(batch), dim=1).squeeze(0)
        top1_conf, top1_idx = torch.max(probs, dim=0)
        top1_idx = int(top1_idx)
        top1_conf = float(top1_conf)
        top1_class = self.categories[top1_idx]
        reject_mass = float(probs[self._reject_mask].sum())

        is_confident_nonplant_object = (
            top1_idx in _REJECT_RANGE
            and top1_idx not in _REJECT_RANGE_EXCEPTIONS
            and top1_conf >= REJECT_CONFIDENCE
        )
        # Second, independent signal -- see this module's docstring
        # ("Second signal -- REJECT_MASS_THRESHOLD"): catches busy
        # man-made-object scenes (e.g. a laptop/keyboard/desk) whose
        # probability splits across several correlated object classes with
        # no single confident top-1, which the check above alone misses.
        is_dominated_by_nonplant_mass = reject_mass >= REJECT_MASS_THRESHOLD
        is_rejected = is_confident_nonplant_object or is_dominated_by_nonplant_mass

        if is_confident_nonplant_object:
            reason = (
                f"Confidently classified as '{top1_class}' ({top1_conf:.2f}) by a generic "
                f"image classifier -- not plausibly a rice leaf."
            )
        elif is_dominated_by_nonplant_mass:
            reason = (
                f"A generic image classifier's guesses were dominated by man-made-object "
                f"classes (top guess '{top1_class}' {top1_conf:.2f}, but {reject_mass:.2f} total "
                f"probability across that whole class range) -- not plausibly a rice leaf."
            )
        else:
            reason = None

        return {
            "is_recognized": not is_rejected,
            "reason": reason,
            "imagenet_top1_class": top1_class,
            "imagenet_top1_confidence": top1_conf,
            "imagenet_reject_mass": reject_mass,
        }
