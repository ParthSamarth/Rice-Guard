# An Explainable Real-Time Rice Leaf Disease Detection and Decision Support System Using CNN, YOLOv8, and Intelligent Treatment Recommendation

*Developed by: Parth Samarth*

This is the project's own documentation (kept separate from `README.md`, which is the
**RiceLeafDiseaseBD dataset's own README** and must not be edited). This file documents the
*system* built on top of that dataset: what it actually is, what the verified dataset audit found,
and how the implementation maps back to the project's proposed architecture.

> Numbers in this document are only ever copied from files under `outputs/` after the corresponding
> script actually ran. A section that says "not yet run" means exactly that.

---

## 1. How this maps to the proposed 4-layer architecture

```
CLIENT LAYER            camera/upload, result display, Grad-CAM view, treatment view, history
        |                (mobile app -- NOT built in this milestone)
        v
APPLICATION LAYER        auth, upload handling, routing, result formatting, history storage
        |                (backend service -- NOT built in this milestone)
        v
INTELLIGENCE LAYER       YOLOv8n -> ResNet50 -> Grad-CAM -> Recommendation Engine
        |                (THIS MILESTONE -- fully implemented, see below)
        v
DATA LAYER               disease images, detection annotations, classification labels,
                          treatment knowledge base
```

This milestone implements the **Intelligence Layer** only, as scoped. Every component is exposed as
a plain, importable Python class with a narrow interface, specifically so a future backend can call
it without depending on any training/CLI code:

- `detection/predict_yolo.py:YoloDetector` -- `.detect(image) -> dict`
- `classification/predict_classifier.py:ClassifierPredictor` -- `.predict(image) -> dict`
- `explainability/gradcam.py:explain_image(...)` -- Grad-CAM for one image
- `recommendation/recommendation_engine.py:RecommendationEngine` -- `.get_recommendation(name) -> dict`
- `pipeline/inference_pipeline.py:RiceDiseasePipeline` -- `.run(image) -> dict` (the whole intelligence
  layer end to end, in one call -- this is the function an Application Layer would call)

The Client and Application layers, and the future extensions (ViT verification, on-device/edge
inference, multilingual assistant, weather/advisory integration, severity estimation) are explicitly
**out of scope for this milestone** and have not been built, per instruction.

---

## 2. The actual dataset overrides an earlier PPT slide

An earlier project document described a larger disease-class scope. **The actual RiceLeafDiseaseBD
dataset is authoritative and is what this system is built and evaluated against.** It contains:

- **9,769 images total**, **6 classes**: Healthy + Blast, Brown Spot, Leaf Smut, Rice Tungro, Sheath
  Blight (**not** 12/13 classes)
- 1,575 Healthy images (no bounding boxes -- classification-only, by construction)
- 8,194 disease images with YOLO-format bounding-box annotations
- 1024x1024 released resolution

No disease classes were fabricated to match any earlier slide's number.

---

## 3. Verified dataset audit findings (`preprocessing/dataset_audit.py` -> `outputs/dataset_audit.json`)

These ten points are the dataset-quality findings this project's documentation is required to carry
forward, in full, rather than hide:

1. **Documentation vs. archive structure differ.** The dataset's own README describes an illustrative
   layout (`Original_Images/`, `Annotated_Data/.../images/`, zero-padded filenames). The real archive
   uses `Original images/` (space), `Annotated images ( visual with labels)/` (stray space), `labels/`
   + `visuals/` subfolders, and unpadded, inconsistently-cased filenames. All loader code
   (`utils/dataset_common.py`) discovers folders by normalized name, never by hard-coded exact string.
2. **Raw class IDs are inconsistent with the documentation.** README.md documents
   `0=Brown Spot, 1=Blast, 2=Healthy, 3=Leaf Smut, 4=Rice Tungro, 5=Sheath Blight`. Reading every one
   of the 70,067 raw label lines shows Brown Spot(0)/Blast(1)/Leaf Smut(3) match, but **Rice Tungro's
   boxes are actually written with id 3** (colliding with Leaf Smut's documented id) and **Sheath
   Blight's with id 4** (colliding with Rice Tungro's documented id); id 5 never appears anywhere.
3. **Class identity was therefore re-derived from folder provenance, not the embedded integer.** Every
   label folder is internally self-consistent (exactly one raw id per folder -- verified, zero
   conflicts), so which folder a `.txt` file lives in is a fully reliable signal even though the number
   written inside some of those files is not. `preprocessing/prepare_yolo_dataset.py` assigns the final
   training class purely from folder location for all 5 classes (not just the 2 that collide), and
   writes a clean, contiguous, newly-assigned mapping **only** into `dataset/processed/yolo/`:
   `0=Blast, 1=Brown Spot, 2=Leaf Smut, 3=Rice Tungro, 4=Sheath Blight`. The original label files under
   `dataset/original/` are never modified -- confirmed by re-reading them live (they still carry the
   original colliding ids).
4. **The actual bounding-box count differs from both published figures.** README.md claims 76,150;
   the Annotation Protocol PDF claims 74,836 (internally consistent with its own per-class table).
   Summing every line of all 8,194 raw label files gives **70,067** -- the number this project trains
   against. Four of five classes match the PDF's table exactly; the entire 4,769-box gap traces to
   Leaf Smut alone (3,885 actual vs. 8,654 claimed), which this project cannot explain from the files
   themselves.
5. **583 disease images have an empty label file** (mostly Rice Tungro -- ~19% of that class), i.e.
   they are correctly classed as diseased but contain zero boxes to crop. This is consistent with the
   Annotation Protocol's own description of Rice Tungro's diffuse, whole-leaf symptoms (hardest to box
   exhaustively), and is the concrete reason the CNN is trained on full images as well as crops for
   every disease class -- a crops-only classifier would never see these images at all.
6. **Healthy has no bounding boxes**, by construction -- Healthy is excluded from YOLO detection
   entirely (Section 5) rather than given fabricated boxes.
7. **39 exact-duplicate image groups were found** (SHA-256), 8 of them cross-class (the same photo
   filed under two different disease labels). All 39 are excluded -- keeping exactly one representative
   per group -- **before** the train/val/test split is drawn
   (`preprocessing/make_split_manifest.py`), so no duplicate (same-class or cross-class) can ever
   straddle two splits. This was verified twice: once against the split manifest, and once by scanning
   the materialized `dataset/processed/yolo/images/` directory for any of the 39 dropped filenames
   (found: 0). **9,730 unique images** remain after exclusion; the clean YOLO dataset built from them
   contains **9,147 images and 69,761 boxes**.
8. **Geographic/temporal scope**: Gazipur, Bangladesh, Summer 2025 -- results should not be assumed to
   generalize to other regions, cultivars, cameras, or seasons without validation.
9. **Bounding boxes are region-level, not pixel-level segmentation** -- lesion shape/area is not
   directly measurable from this annotation format.
10. **Annotations are expert-validated visual symptom labels, not laboratory-confirmed diagnoses**
    (per the Annotation Protocol: inter-annotator Cohen's kappa = 0.82, IoU = 0.78, F1 = 0.85 on a
    200-image QC sample -- a genuinely useful but not infallible data-quality signal).

See `outputs/dataset_audit.json` / `outputs/dataset_statistics.csv` / `outputs/plots/` for the full
machine-generated audit this summary is drawn from.

---

## 4. Split integrity (leakage prevention)

- **One master image-level split** (`dataset/processed/split_manifest.csv`, 70/15/15, seed 42,
  stratified per class) is the single source of truth for both the YOLO dataset and the classification
  dataset -- built once, consumed by both, so the same source photo can never train one model and test
  the other.
- Exact-duplicate exclusion happens **before** this split is drawn (Section 3, point 7) -- an earlier
  version of this pipeline only deduplicated inside the classification dataset builder, which left the
  YOLO dataset's split leaky (18 of 39 duplicate groups, including all 8 cross-class ones, landed in
  different splits). This was caught, fixed at the source, and re-verified before any YOLO training run
  whose results are reported here.
- Every crop generated from a source image inherits that image's split assignment
  (`preprocessing/prepare_classification_dataset.py`) -- verified: 0 of 9,730 source images have
  samples spanning more than one split.
- Ground-truth boxes are **never** fed into the integrated pipeline during evaluation
  (`evaluation/evaluate_pipeline.py`) -- YOLO's own predictions are what the CNN sees, exactly as in
  deployment.

---

## 5. Architecture decisions

- **YOLOv8n is a 5-class disease detector, not 6-class.** Healthy has no bounding-box annotations;
  training a 6th class would require fabricating boxes, which is not done. Healthy images are included
  as background/negative images (present in `images/`, empty label file -- Ultralytics' own convention
  for "no objects here"), never as a fake Healthy detection.
- **ResNet50 is a 6-class verifier**, trained on **both** full images and YOLO-box crops for every
  disease class (Healthy is necessarily full-image-only). This mirrors the two situations it is
  actually asked to handle at inference: YOLO finds nothing -> classify the full frame; YOLO finds a
  region -> verify the crop. The exact same crop-with-context-margin function
  (`utils/dataset_common.py:expand_and_clip_box`, 15% margin) is used at training-crop-generation time
  and at inference time, so the CNN never sees a systematically different kind of crop than it trained on.
- **Class-ID ordering for the CNN** is fixed as `0=Healthy, 1=Blast, 2=Brown Spot, 3=Leaf Smut,
  4=Rice Tungro, 5=Sheath Blight` (`classification/dataset.py:CLASS_TO_IDX`), used consistently across
  training, evaluation, Grad-CAM, and the inference pipeline.
- **YOLO/CNN disagreement is never silently resolved.** When the top-confidence detected region's YOLO
  class and the CNN's crop-level prediction disagree, the pipeline returns
  `"status": "model_disagreement"` with `final_disease: null` and both models' predictions +
  confidences exposed at the top level of the result (`yolo_prediction`, `yolo_confidence`,
  `cnn_prediction`, `cnn_confidence`) -- it does not pick a side.
- **The recommendation engine never imports or calls a model.** It is a pure lookup from a disease-name
  string into `recommendation/knowledge_base.json`. Its `treatment` fields are explicitly marked
  `NEEDS_AUTHORITATIVE_VALIDATION` -- no pesticide/dosage/concentration/frequency data was supplied
  with the project materials, and none was invented. `management`/`prevention` fields contain only
  general, non-quantitative agronomic practice categories.
- **Grad-CAM explains the CNN's decision only.** It is never presented as an explanation of YOLOv8's
  own region-proposal mechanism, and is never presented as proof the model is correct --
  `explainability/gradcam_quality_check.py` generates correct- and incorrect-prediction examples for
  every class specifically so a human can look for failure cases (background/border/artifact focus),
  not just confirmation cases.

---

## 6. Experiment scope for this milestone

Per instruction: **YOLOv8n only** (no YOLOv8s comparison) and **ResNet50 only** (no ResNet18 or ViT
comparison) unless explicitly requested later. Both use ImageNet/COCO-pretrained transfer learning,
seed 42, AMP, and hardware-aware settings for the training machine's 6 GB GPU
(`utils/hardware.py`) -- see `outputs/experiment_summary.json` for the exact detected environment.

## 7. Results

### 7.1 YOLOv8n detection (test split)

**Training-budget disclosure -- read before the numbers.** An explicit instruction capped this run's
epoch budget at 70 (after repeated external process interruptions -- see Section 6/`outputs/yolo/metrics/yolov8n_train_summary.json`
for the interruption pattern) and required that the run **not** be allowed to continue past 70 or be
misreported as a clean, uninterrupted run of any length. Auditing the saved artifacts for this report
(`outputs/yolo/runs/yolov8n/results.csv`, `args.yaml`, and file timestamps) after the fact found that
this was **not honored**: the log shows a clean, monotonic sequence of epochs 1-70, but is then followed
by a **resume from an epoch-67 checkpoint that continues to epoch 100** -- i.e. epochs 68, 69 and 70 each
appear twice in the log, and training continued 30 epochs past the directed cap. `args.yaml`'s final
state reads `epochs: 100`, not 70. The saved `best.pt`/`last.pt` (timestamp 2026-08-18 20:06) reflect
this epoch-100 end state, not the epoch-70 state that was actually intended to be final.

Separately, the test-set evaluation JSON that existed on disk (`outputs/yolo/metrics/yolov8n_eval_test.json`,
timestamp 18:08) predated that final checkpoint save (20:06) by nearly two hours -- it had been computed
against an earlier checkpoint state and no longer matched the weights actually saved as `best.pt`. This
mismatch was caught while assembling this report (by comparing file timestamps) and corrected by
re-running `evaluate_yolo.py` against the current `models/yolo_yolov8n/best.pt` immediately before
writing these numbers, so the figures below are verified to match the weights actually on disk. The
numbers below are **not** the honestly-70-epoch-capped result that was directed; they describe an
epoch-100 model reached through an uncontrolled additional resume, which is disclosed here rather than
presented as either a clean 70-epoch run or a clean uninterrupted 100-epoch run. mAP50-95 was still
rising slowly and monotonically at the point the log was cut off (0.157 at epoch 70 -> 0.170 at epoch
100), so the extra epochs did not visibly harm the model -- but the process deviation itself is the point
being disclosed, independent of whether the outcome happened to be benign.

Verified test-set numbers (`outputs/yolo/metrics/yolov8n_eval_test.json`, re-generated
2026-08-18, weights = `models/yolo_yolov8n/best.pt`, 1,378 test images / 11,345 ground-truth boxes):

| | mAP50 | mAP50-95 | mAP75 | Precision | Recall |
|---|---|---|---|---|---|
| **Overall (5 classes)** | 0.170 | 0.070 | 0.045 | 0.381 | 0.181 |

| Class | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| Blast | 0.255 | 0.097 | 0.446 | 0.262 |
| Brown Spot | 0.356 | 0.157 | 0.473 | 0.370 |
| Leaf Smut | 0.118 | 0.045 | 0.387 | 0.144 |
| Rice Tungro | 0.099 | 0.040 | 0.310 | 0.110 |
| Sheath Blight | 0.022 | 0.010 | 0.292 | 0.019 |

Recall is low across every class, and Sheath Blight in particular is very weak on detection (recall
0.019) despite being one of the better-performing classes downstream in the CNN. This detector should be
described in the final report as a genuinely weak first-stage detector, not a strong one -- most of the
system's effective accuracy comes from ResNet50 (Section 7.2), which is trained on full images
specifically so a missed/weak detection is not fatal to the overall pipeline (Section 5). The per-class
confusion matrix (`outputs/yolo/metrics/yolov8n_eval_test.json`) shows the dominant error mode is
missed detections (predicted background) rather than cross-class confusion.

### 7.1b YOLOv8n @ imgsz=1024 resolution ablation -- PARTIAL / INCOMPLETE (stopped at epoch 17/40)

**This is not a converged model and must never be presented as a fair replacement for, or compared
head-to-head with, the fully trained 640 baseline above as if both were finished runs.** It is disclosed
here in full because hiding a stopped experiment would misrepresent this project's reproducibility record,
not because it changes the detector this milestone ships.

**Why this was run.** `outputs/yolo_error_analysis/ERROR_ANALYSIS_REPORT.md` (diagnostic error analysis of
the 640 baseline, Section 7) identified a small-object/resolution problem as 640's dominant recall-limiting
failure mode -- over half of every class's ground-truth boxes, and over three-quarters of Sheath Blight's
and Rice Tungro's, fall in a size bucket where 640's recall is near-zero regardless of class -- and
recommended testing input resolution (not a YOLOv8s backbone swap) as the next, more targeted experiment.
`outputs/yolo_ablation_1024/pre_training_record.md` records the pre-flight verification (dataset/split
byte-identical, baseline untouched, AutoBatch check) done before this run started.

**Why it was stopped.** Launched 2026-08-18 22:19:41 (`train_batch0.jpg` timestamp) with an explicit
40-epoch ablation budget (patience=10) and Windows workers=0 as an already-proven crash-recovery setting.
Over the following ~8h51m, the run needed 88 total resume attempts, the great majority of which died before
completing another epoch with no new error text, no CUDA OOM (GPU repeatedly confirmed idle/0 MiB used at
every stall check), and no dataset/checkpoint corruption -- the same clean external-kill signature
throughout. It reached epoch 17 and then could not advance past it through **15 consecutive** no-progress
resumes. Per the overnight directive's own escalation threshold, this was reported to the user rather than
resumed indefinitely (`pre_training_record.md`, "Escalation checkpoint," 2026-08-19 07:10:22 -- an elapsed
time of 8h 50m 41s, independently confirmed by comparing these two file timestamps directly, not just by
trusting the log's own arithmetic). The user's decision: accept epoch 17 as the practical stopping point,
evaluate the best available checkpoint honestly as a partial/incomplete ablation, and do not resume again.

Of the 8h51m elapsed, only **~3h04m (34.7%)** was actual GPU-active training compute (summed directly from
the per-epoch time deltas in `outputs/yolo/runs/yolov8n_1024/results.csv`); the remaining **~5h47m (65.3%)**
was interruption/restart overhead. This is the concrete cost of the environment instability, kept separate
from anything the resolution hypothesis itself is being judged on.

**Checkpoint verification.** Both `last.pt` (17 completed epochs) and `best.pt` (16 completed epochs,
selected by Ultralytics' own best-fitness tracking, best_fitness=0.02723) were loaded directly and
confirmed: `train_args` consistent with the planned configuration in every field (imgsz=1024, batch=3,
epochs=40, patience=10, workers=0, seed=42), optimizer state and EMA weights present in both, no
corruption. No training process was found running at the time of evaluation. All formal metrics below use
`best.pt`, matching the same convention used for the 640 baseline throughout this document; `last.pt` was
verified but not separately re-scored, since it trails `best.pt` by one epoch and by fitness.
`models/yolo_yolov8n_1024/best.pt` and `last.pt` were staged from the raw Ultralytics run directory as a
manual step (mirroring what `detection/train_yolo.py:_finalize()` does for a run that returns normally --
this run never did, since every resume died inside `model.train()`), specifically so the existing
`detection/evaluate_yolo.py` run-tag convention could be reused unmodified for an apples-to-apples
evaluation. This does not touch, and is fully namespaced away from, `models/yolo_yolov8n/` (640).

Verified test-set numbers (`outputs/yolo/metrics/yolov8n_1024_eval_test.json`, generated 2026-08-19,
weights = `models/yolo_yolov8n_1024/best.pt`, same 1,378 test images / 11,345 ground-truth boxes as 7.1),
next to the 640 baseline for reference -- **not as an equivalent comparison** (640 is a converged ~100-epoch
run; 1024 is 16 epochs into a 40-epoch ablation budget that was itself smaller than 640's):

| | mAP50 | mAP50-95 | mAP75 | Precision | Recall |
|---|---|---|---|---|---|
| **640 -- converged baseline** | 0.170 | 0.070 | 0.045 | 0.381 | 0.181 |
| **1024 -- partial, epoch 16/40** | 0.063 | 0.026 | 0.015 | 0.240 | 0.076 |

| Class | 640 mAP50 / 1024 mAP50 | 640 recall / 1024 recall |
|---|---|---|
| Blast | 0.255 / 0.081 | 0.262 / 0.076 |
| Brown Spot | 0.356 / 0.179 | 0.370 / 0.241 |
| Leaf Smut | 0.118 / 0.035 | 0.144 / 0.051 |
| Rice Tungro | 0.099 / 0.015 | 0.110 / 0.006 |
| Sheath Blight | 0.022 / 0.005 | 0.019 / 0.004 |

Every overall and per-class metric is lower for the partial 1024 checkpoint than for the converged 640
baseline -- expected given the training-amount gap, and on its own not informative about the resolution
hypothesis specifically. Two further checks control for that confound:

1. **Matched-epoch check.** 640's own training-time validation-split numbers at epochs 16 and 17 (read from
   `outputs/yolo/runs/yolov8n/results.csv`, a clean unduplicated range that predates 640's own later
   interruption -- see 7.1 above) already lead 1024's numbers at the identical epoch count, on all 4 tracked
   metrics (precision, recall, mAP50, mAP50-95) at both epochs. At equal training amount, 1024 is not
   ahead on anything measured yet.
2. **Box-size-bucketed recall at the deployment threshold** (conf=0.35, `configs/config.yaml`; methodology
   and area buckets identical to `ERROR_ANALYSIS_REPORT.md`, computed via `evaluation/yolo_error_analysis.py
   --out-dir outputs/yolo_ablation_1024/error_analysis` -- a `--out-dir` override was added to that script
   for this run, default behavior/output path unchanged, specifically so this run could not silently
   overwrite the 640 baseline's saved analysis at its default path). This is the direct test of the
   hypothesis that motivated the ablation:

   | Class | 640 very-small recall | 1024 very-small recall |
   |---|---|---|
   | Blast | 1.1% | 0.0% |
   | Brown Spot | 0.7% | 0.0% |
   | Leaf Smut | 0.0% | 0.0% |
   | Rice Tungro | 0.8% | 0.0% |
   | Sheath Blight | 0.0% | 0.0% |

   1024's very-small-bucket recall is 0.0% for every class -- behind or tied with 640's already-weak
   0.0-1.1%. Sweeping the full confidence range (0.05-0.65, not just the 0.35 deploy point) shows the same
   ordering at every threshold for every class, with no crossover point. At the deployment threshold
   specifically, Leaf Smut, Rice Tungro, and Sheath Blight all reach **exactly zero** true positives out of
   662, 2,184, and 2,867 ground-truth boxes respectively.

**Answering the questions this ablation was meant to answer:**

- *Did 1024 show early evidence of improving small/diffuse-object recall?* No -- see the box-size-bucket
  and confidence-sweep results immediately above.
- *Did Leaf Smut, Rice Tungro, or Sheath Blight recall improve relative to 640?* No, all three fell:
  Leaf Smut 0.144->0.051, Rice Tungro 0.110->0.006, Sheath Blight 0.019->0.004.
- *Is the observed improvement, if any, large enough to justify a future full 1024 experiment?* There is no
  observed improvement -- every metric moved the wrong way. This does not refute the resolution hypothesis
  (17 of 40 epochs is nowhere near convergence, and the stop was environment-driven, not metric-driven), but
  it provides no positive evidence to justify further overnight compute on this hardware before the
  resume-stability problem itself is understood.
- *Is the current partial result sufficient to select 1024 as the final detector?* No -- it trails the
  converged baseline on every axis checked, including a matched-epoch comparison that controls for the
  training-amount gap.

No claim is made anywhere in this section about what a full 40-epoch (or further) 1024 run would have
scored. That outcome is unknown and is not estimated, interpolated, or implied.

**Qualitative examples.** Representative annotated detections (ground truth vs. epoch-17-checkpoint
predictions at the deploy threshold) were generated for all 5 classes across true positives, false
negatives, false positives, and poor-localization cases -- 55 images saved to
`outputs/yolo_ablation_1024/error_analysis/examples/`. Three are discussed in the accompanying evaluation
report (link below): a Sheath Blight leaf with 9 tiny ground-truth lesions and 0 detections, a Rice Tungro
leaf with one large diffuse region and one embedded tiny lesion with 0 detections of either, and a Brown
Spot leaf with 8 lesions where the one medium-sized box was caught -- chosen as the first qualifying example
per category, not cherry-picked for best or worst case.

**Recommendation: A -- keep 640 as the final detector for this milestone.** Across every comparison run
for this ablation -- overall/per-class test-set metrics, the matched-epoch check, box-size-bucketed recall,
and the full confidence-threshold sweep -- the partial 1024 checkpoint did not lead 640 on a single measured
axis. This is consistent with severe undertraining, not evidence that higher input resolution categorically
fails; the resolution hypothesis from `ERROR_ANALYSIS_REPORT.md` remains scientifically **untested** to
convergence, not refuted, because the run never got far enough to test it. But untested is not the same as
promising, and nothing in this partial result is actionable evidence in 1024's favor. **The 640 baseline in
Section 7.1 is unchanged and remains this milestone's detector**; ResNet50 (7.2) and the integrated pipeline
(7.5) were not rerun as part of this evaluation.

A full interactive version of this evaluation (side-by-side tables, box-size-bucket bar comparison,
confidence-sweep charts, and the three annotated examples) was generated as an artifact alongside this
update; the numbers in it match this section exactly.

### 7.2 ResNet50 classification (test split)

Trained for a **directed, disclosed 15-epoch budget** (`outputs/classification/resnet50/train_summary.json`),
completed as a single clean run with **0 OOM events** and best checkpoint selected at **epoch 11** by
validation macro-F1 (0.675). This run's log is fully sequential (epochs 1-15, no duplicate/rolled-back
epochs), so no equivalent caveat applies here.

Verified test-set numbers (`outputs/classification/resnet50/evaluation/metrics_test.json`, 6,798 samples):

| | Accuracy | Macro-P | Macro-R | Macro-F1 | Weighted-F1 |
|---|---|---|---|---|---|
| **Overall** | 0.691 | 0.682 | 0.714 | 0.692 | 0.697 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Healthy | 0.896 | 0.945 | 0.920 | 236 |
| Blast | 0.656 | 0.585 | 0.618 | 1265 |
| Brown Spot | 0.826 | 0.691 | 0.752 | 2433 |
| Leaf Smut | 0.367 | 0.571 | 0.447 | 487 |
| Rice Tungro | 0.616 | 0.706 | 0.658 | 1110 |
| Sheath Blight | 0.733 | 0.785 | 0.758 | 1267 |

**Full-image vs. crop performance -- reported separately, not mixed** (`by_sample_type` in the same
JSON, per instruction):

| Sample type | N | Accuracy | Macro-F1 |
|---|---|---|---|
| Full images | 1,458 | 0.883 | 0.867 |
| YOLO-box crops | 5,340 | 0.639 | 0.591 |

The classifier is substantially stronger on full images than on crops (+24 points accuracy). This is
consistent with Section 3's finding that crop quality varies a great deal (many crops are small,
low-resolution, or come from a weak detector per 7.1) -- it is also visible directly in the Grad-CAM
review below, where the worst failure cases are on visibly low-information crops. Leaf Smut is the
weakest class overall (F1 0.447) and is the class most often confused with Brown Spot in both the overall
and crop-only confusion matrices.

### 7.3 Grad-CAM qualitative review (human-reviewed)

`explainability/gradcam_quality_check.py` only computes a crude, explicitly-unvalidated
`border_energy_fraction` triage heuristic -- it does not itself judge whether an activation map is
meaningful. The following is the required human visual review of its 53 saved panels
(`outputs/gradcam_quality_check/<class>/{correct,misclassified}_*_gradcam.png`), covering all 6 classes
with both correct and misclassified examples, done specifically to surface failure cases rather than to
confirm the model is correct.

**Well-localized, plausible attention (4 of 9 reviewed examples):**
- *Healthy*, correct @0.99 -- heatmap tightly centered on the leaf blade, correctly ignoring background
  weeds and a person's leg/boot visible in frame.
- *Leaf Smut*, correct @0.92 -- attention tightly on the thin dark streak lesion.
- *Rice Tungro*, correct @0.81 (`Rice Tungro_1167`) -- a tight hotspot sits precisely on the localized
  brown leaf-margin lesion. Notable because Rice Tungro symptoms are documented as diffuse/whole-leaf
  (Section 3, point 5) -- this particular case had a localized visible symptom, and the model localized
  it correctly rather than defaulting to a diffuse whole-leaf pattern.
- *Blast*, correct @0.48 (`blast_991__crop30`) -- a tight hotspot sits exactly on the dark diamond-shaped
  lesion. Worth flagging: 0.48 is *below* this project's configured `cnn_confidence` threshold (0.50,
  `configs/config.yaml`), so despite being correct and well-localized, this exact case would surface as
  a low-confidence result in the deployed pipeline rather than a confident correct answer -- correctness,
  good localization, and confidence do not always move together.

**A specific, non-obvious pattern worth reporting precisely (not just "it focuses on the lesion"):**
- *Brown Spot*, correct @0.73 (`brown spot_1257__crop17`) -- activation concentrates on the lesion's
  margin/halo (hotspots above and below), while the darker lesion **core** is comparatively cold. This is
  plausible botanically (classic Brown Spot lesions have a necrotic center with a lighter halo), but it
  means the model may be keying on the halo/transition zone as the discriminating feature rather than the
  lesion interior -- a specific hypothesis worth stating rather than a generic "attends to the lesion"
  claim.

**Genuine failure cases (attention on border/artifact/adjacent-to-lesion, not the lesion itself):**
- *Leaf Smut* -> predicted Brown Spot @0.61 (`Leaf smut_659__crop0`) -- the actual thin dark streak lesion
  (top-left of the crop) receives almost no activation. Instead, attention concentrates on a dark, blurry
  region at the crop's right edge (shadow/out-of-focus background). A clean example of border/artifact-
  driven misclassification, exactly the failure mode the `border_energy_fraction` heuristic is designed
  to flag, and it correctly flagged this one.
- *Sheath Blight* -> misclassified @0.68 (`Sheath blight_977__crop4`) -- the crop itself is almost
  featureless (a blurry gradient with no visible lesion structure at all); Grad-CAM attention sits
  entirely on a vertical strip at the crop's right border. This looks like a crop that simply lacks
  diagnostic content -- consistent with 7.1's finding that this detector's boxes are often low-quality --
  rather than a model attention failure per se.
- *Sheath Blight*, **correct** @0.88 (`Sheath blight_833__crop1`) -- correct label, high confidence, but
  the Grad-CAM hotspot sits at the bottom-right/border of the crop, **not** on the dark lesion blob
  clearly visible in the upper-left of the original image. A "right answer, unverified reason" case --
  high confidence and a correct label do not guarantee the model actually looked at the lesion. This is
  the kind of case that only a human visual review (not accuracy/F1 numbers, and not even a
  misclassified-only review) can surface.
- *Rice Tungro* -> misclassified @0.56 (`Rice Tungro_526__crop2`) -- a dark diagonal lesion streak is
  visible in the crop; the hotspot sits just below/beside the streak rather than on it. A partial/
  adjacent-attention case, milder than the two pure border cases above but still not lesion-aligned.

**Synthesis for the final report:** of the 9 examples reviewed across all 6 classes, roughly half show
attention plausibly aligned with the visible lesion, one shows a specific halo-vs-core pattern worth
further study, and the rest show attention that is partially or wholly disconnected from the visible
lesion -- including one high-confidence *correct* prediction whose attention was not actually on the
lesion. The classes with the clearest documented Grad-CAM failures (Leaf Smut, Sheath Blight) are also
the classes with the weakest or most-confused quantitative metrics in 7.2, which is corroborating rather
than coincidental. **Grad-CAM is not used here as evidence the model is correct** -- it is used as
required, to find and document real failure cases; the full 53-panel set remains available under
`outputs/gradcam_quality_check/` for further review beyond these 9.

### 7.4 End-to-end pipeline smoke test

10 held-out test-split images run through `pipeline.RiceDiseasePipeline.run()` end to end (raw image in,
full JSON out -- no pre-cropping, no ground-truth boxes given to the pipeline). 6 were picked one per
class; 4 more were specifically selected because YOLO fires a real detection on them (found by scanning
40 random non-Healthy test images with the detector directly), so both the no-detection->full-image-CNN
path and the detection->crop->CNN-verification path are actually exercised, not just assumed to work.

| Image | True class | YOLO says | CNN says | Final | Status | Correct? |
|---|---|---|---|---|---|---|
| Blast_10 (full) | Blast | -- (no detection) | Blast @0.74 | Blast | ok | Yes |
| brown spot_100 (full) | Brown Spot | -- (no detection) | Leaf Smut @0.45 | Leaf Smut | **low_confidence** | No -- correctly flagged |
| Healthy_10 (full) | Healthy | -- (no detection) | Healthy @0.54 | Healthy | ok | Yes |
| Leaf smut_104 (full) | Leaf Smut | -- (no detection) | Leaf Smut @0.92 | Leaf Smut | ok | Yes |
| Rice Tungro_100 (full) | Rice Tungro | -- (no detection) | Rice Tungro @0.98 | Rice Tungro | ok | Yes |
| Sheath blight_100 (full) | Sheath Blight | -- (no detection) | Sheath Blight @0.93 | Sheath Blight | ok | Yes |
| blast_864 | Blast | Blast @0.75 | Blast @0.84 | Blast | ok | Yes |
| Leaf smut_435 | Leaf Smut | Leaf Smut @0.37 | Leaf Smut @0.95 | Leaf Smut | ok | Yes |
| Leaf smut_599 | Leaf Smut | Leaf Smut @0.73 | Leaf Smut @0.97 | Leaf Smut | ok | Yes |
| Rice Tungro_1881 | Rice Tungro | Brown Spot @0.51 | Brown Spot @0.88 | Brown Spot | ok | **No -- not flagged** |

Result: no crashes; every run produced the full documented schema (`status`, `final_disease`,
`final_confidence`, `yolo_prediction`/`yolo_confidence`, `cnn_prediction`/`cnn_confidence`, `agreement`,
a saved Grad-CAM PNG, and a populated recommendation with `NEEDS_AUTHORITATIVE_VALIDATION` markers
intact). 6 of 6 sampled-one-per-class images had no YOLO detection above threshold and correctly fell
back to full-image CNN classification -- expected given 7.1's measured recall of 0.18, and this is
exactly why the architecture (Section 5) does not require a detection to produce a result.

Two results are worth calling out explicitly rather than only counting the 8/10 correct:
- **Row 2 is wrong but the system caught it**: Brown Spot misclassified as Leaf Smut at 0.45 confidence,
  correctly surfaced as `status: "low_confidence"` rather than presented as a confident answer -- the
  threshold mechanism worked as designed on a real example, not just in unit tests.
- **Row 10 is wrong and the system did *not* catch it**: YOLO (0.51) and the CNN (0.88) independently
  agree with each other on "Brown Spot" for a true Rice Tungro image -- high combined confidence, full
  `agreement: true`, `status: "ok"`. `model_disagreement` only fires when YOLO's and the CNN's
  predictions differ from **each other**; it has no mechanism to catch the case where both models are
  confidently wrong **together**. This is a genuine, previously-undocumented limitation surfaced by this
  smoke test, and it is consistent with the Grad-CAM finding in 7.3 that agreement/confidence is not
  proof of correctness -- it should be stated plainly in the final report rather than only reporting the
  cases that happened to work.

Raw JSON for all 10 runs is not persisted under `outputs/` (single-image `predict` runs print to stdout
rather than writing a file -- only `--image-dir` batch runs do); the commands and images used are listed
above for reproducibility.

### 7.5 Integrated pipeline evaluation (full test set)

All **1,458** original test-split images (never ground-truth boxes -- YOLO's own predictions are what the
CNN sees, per Section 4), run through `RiceDiseasePipeline.run()` end to end.

**A real bug was found and fixed while running this.** The first full run completed all 1,458
pipeline calls successfully (confirmed by the saved per-image CSV) but then crashed *after* the loop,
while computing aggregate metrics: `model_disagreement`/`low_confidence` rows have `predicted = None`,
and building a pandas DataFrame from those rows turned `None` into `NaN` (a float) in the `predicted`
column; the metrics code's `if p else UNRESOLVED` check treats `NaN` as truthy (`bool(float('nan'))` is
`True` in Python), so it called `.strip()` on a float and crashed. Fixed in
`evaluation/evaluate_pipeline.py` by checking `isinstance(p, str) and p` instead of `if p`, and by giving
error-branch rows the same dict keys as normal rows for consistency. Because the expensive part (all
1,458 pipeline runs) had already completed and been saved to
`outputs/pipeline_evaluation/pipeline_eval_test.csv` before the crash, the summary was recomputed directly
from that CSV with the fixed logic rather than re-running inference -- verified to use the exact same
computation as the (now-fixed) script would produce on a fresh run.

Verified results (`outputs/pipeline_evaluation/pipeline_eval_test_summary.json`):

| | Value |
|---|---|
| Images evaluated | 1,458 |
| Status: ok / model_disagreement / low_confidence / error | 1339 / 61 / 58 / 0 |
| YOLO/CNN agreement rate (when both fire) | 0.958 |
| **Accuracy, overall** (disagreement/low-confidence counted as wrong -- the conservative number) | **0.853** |
| **Accuracy, when confident** (status == "ok" only) | **0.909** |
| Macro-F1 / Weighted-F1 (unresolved rows scored as a miss, never dropped) | 0.857 / 0.871 |
| No-detection rate (YOLO found nothing -> full-image CNN path) | 0.602 |

Two numbers need to be read together, not separately: **0.853** is the honest, unconditional accuracy
across every test image, including the 8.2% the system explicitly declined to answer (61
model-disagreement + 58 low-confidence cases). **0.909** is accuracy restricted to the 91.8% of cases
where the system committed to an answer. The gap between them (5.6 points) is the abstention mechanism
actually working -- it is declining disproportionately on its hardest cases rather than guessing, which
is the entire point of Section 5's `model_disagreement`/low-confidence design.

This is also **higher** than the standalone ResNet50 test accuracy in 7.2 (0.691), which looks like a
contradiction until the reason is stated: 60% of these original test images get **no YOLO detection at
all** (consistent with 7.1's measured recall of 0.18) and fall back to the full-image CNN path, which 7.2
already showed is the classifier's strongest mode (0.883 accuracy) -- so the integrated pipeline's
confident-case accuracy is dominated by the easier full-image path, plus it gets to skip its hardest cases
via abstention. It is not evidence the classifier got better; it is a selection effect from how the two
numbers are conditioned, and both are reported here rather than picking the more flattering one.

Failure-case taxonomy: 16 Healthy images misclassified as diseased (false positives), 19 real disease
images misclassified as Healthy (false negatives -- the more dangerous error class for a farmer-facing
tool, since it means "no treatment needed" when treatment is), and 91 disease-to-disease confusions.
Per-class accuracy-when-confident ranges from 0.839 (Leaf Smut) to 0.947 (Sheath Blight) -- Leaf Smut
remains the weakest class end to end, consistent with 7.2 and 7.3.

### 7.6 Reproducibility record

Generated: `outputs/experiment_summary.json` (environment, dataset audit summary, both models' training
settings/hardware/history, both models' test metrics, and the full pipeline evaluation above, all in one
file). Seed 42 throughout. `outputs/comparison_plots/` holds the accompanying YOLO/classifier/pipeline
comparison charts (`python main.py visualize`).

Regenerated 2026-08-19 as part of finalization: `experiment_summary.json` now also carries the 1024
ablation (7.1b) under `yolo_experiments.yolov8n_1024`, explicitly tagged `"status":
"PARTIAL_INCOMPLETE_ABLATION"` -- disclosed in the machine-readable record, not just in prose. The
`yolov8n_1024_train_summary.json` it reads from was assembled manually (the run never called
`detection/train_yolo.py:_finalize()` -- see 7.1b), mirroring exactly what that function writes for a
normal run. `evaluation/visualize_results.py` gained a matching `--include-flagged` option: by default it
now skips any YOLO run whose train summary declares a `status` (i.e. a flagged/non-final run) when building
`outputs/comparison_plots/yolo_comparison.png`, so that chart continues to show the 640 baseline alone
rather than silently plotting it next to an incomplete ablation as if they were equivalent, finished
experiments. The exclusion itself is disclosed, not silent: both the console output and
`outputs/results_comparison.json`'s `yolo_excluded_flagged_runs` field name what was left out and why.

### 7.7 Final results summary (finalization pass, 2026-08-19)

This milestone's finalization pass verified every checkpoint, dataset artifact, and evaluation file listed
in this document still matches what is on disk (checkpoint contents read directly, not assumed; dataset
split re-hashed and confirmed byte-identical; `outputs/gradcam_quality_check/` confirmed to hold all 6
classes, 53 panels; the integrated pipeline's default weights path confirmed by reading
`pipeline/inference_pipeline.py` directly, and empirically by rerunning the exact two images from 7.4's
smoke test -- `blast_864.jpg` and `Healthy_10.jpg` -- through it unchanged, reproducing the identical
predictions/confidences already reported there). No integrity problems were found, and consequently no
retraining was performed -- this section only consolidates already-verified numbers into one authoritative
table.

| Component | Model | Key result | Status |
|---|---|---:|---|
| Detection | YOLOv8n @ 640 | mAP50 = 0.170 | **FINAL** |
| Verification | ResNet50 | Accuracy = 0.691 | **FINAL** |
| -- full-image path | ResNet50 | Accuracy = 0.883 | **FINAL** |
| -- YOLO-crop path | ResNet50 | Accuracy = 0.639 | **FINAL** |
| Integrated system (unconditional) | YOLOv8n + ResNet50 | Accuracy = 0.853 | **FINAL** |
| Integrated system (confident predictions only) | YOLOv8n + ResNet50 | Accuracy = 0.909 | CONDITIONAL -- 91.8% of cases; see 7.5 |
| Resolution ablation | YOLOv8n @ 1024 | epoch 17/40, no positive evidence over 640 (7.1b) | **INCOMPLETE -- not a candidate for final selection** |

The full-image vs. YOLO-crop split and the unconditional-vs-confident split are both load-bearing, not
decorative -- collapsing either into a single headline number would hide exactly the selection effects 7.2
and 7.5 document.

**Final project claim**, worded to match what was actually measured and tested (not aspirational):

> The developed prototype integrates disease-region detection, CNN-based verification, Grad-CAM
> explainability, and rule-based decision support into a unified rice-leaf disease diagnosis pipeline. The
> validated YOLOv8n detector provides the localization stage, while ResNet50 provides disease
> verification. The integrated system achieved 85.3% accuracy on the held-out test set, with an explicit
> abstention mechanism for uncertain cases.

This claim does **not** assert clinical/agricultural diagnostic certainty, field-deployment readiness, a
strong standalone detector (7.1 already documents YOLOv8n's recall as weak), or validated treatment
guidance (14/`recommendation/knowledge_base.json`'s `NEEDS_AUTHORITATIVE_VALIDATION` markers remain
un-replaced by design). The project's contribution is the integration itself -- localization, verification,
interpretability, uncertainty handling, and recommendation in one architecture with disclosed, measured
numbers at every stage -- not a single high-accuracy classifier claim.

A viva/slide-ready version of this table plus the full 640-vs-1024 evidence trail is also published as an
interactive artifact (linked in the conversation this finalization pass was performed in).

## 8. Explicitly not built in this milestone

Client Layer (mobile app), Application Layer (backend/auth/history), ViT verification comparison,
on-device/edge export, multilingual assistant, weather/advisory integration, severity estimation. The
Intelligence Layer's Python interfaces (Section 1) are written so these can be added later without
modifying the trained-model code.

## 9. Final visualizations

Generated 2026-08-19 by four reusable scripts under `evaluation/` (`generate_yolo_plots.py`,
`generate_classifier_plots.py`, `generate_pipeline_plots.py`, `generate_final_dashboard.py`, all sharing
one style/color module, `evaluation/plot_style.py`), wired into `main.py` as `visualize-yolo`,
`visualize-classifier`, `visualize-pipeline`, `visualize-dashboard` alongside the pre-existing `visualize`.
Every number plotted is read at generation time from an already-saved, already-verified JSON/CSV file --
none are hand-typed -- and every script prints (or asserts) a cross-check against those source files before
drawing anything from a live re-computation. **45 distinct charts, saved as both PNG (150 dpi, PPT-ready)
and SVG (publication quality) -- 90 files total**, none of which overwrite any file under
`outputs/yolo/{runs,plots,metrics}/`, `outputs/yolo_error_analysis/`,
`outputs/classification/resnet50/evaluation/`, or `outputs/pipeline_evaluation/` (all pre-existing baseline
artifacts, confirmed byte-identical/timestamp-unchanged after every run of every script below).

Class order and color are fixed once, in `plot_style.py`, and reused by every chart in the whole suite:
Healthy `#2E8B57`, Blast `#2A6FB0`, Brown Spot `#A6631B`, Leaf Smut `#7B5AA6`, Rice Tungro `#C0472B`, Sheath
Blight `#0E9AA0` (YOLO charts use the same 5 disease colors, omitting Healthy, since YOLO has no Healthy
class -- Section 4). This 6-color set passed the colorblind-safe/contrast/chroma validator
(`scripts/validate_palette.js`, light surface) before use.

### 9.1 `outputs/visualizations/yolo/` -- YOLOv8n @ 640 (the final detector)

Model/checkpoint: `models/yolo_yolov8n/best.pt`. Split: **test** (1,378 images / 11,345 boxes) for the
confusion matrix, per-class bars, and confidence curves; the **training/validation log** for the 10
step-vs-metric curves (labeled accordingly on each chart -- never mixed).

| File | What it shows | Source |
|---|---|---|
| `confusion_matrix_raw` / `_normalized` | 6x6 (5 classes + background row/col), Ultralytics' own predicted-vs-true convention, verified against this project's per-class recall/precision before axes were labeled | `outputs/yolo/metrics/yolov8n_eval_test.json` |
| `per_class_map50`, `per_class_map50_95`, `per_class_precision`, `per_class_recall` | one bar per disease class | same JSON |
| `precision_recall_curve`, `f1_vs_confidence`, `precision_vs_confidence`, `recall_vs_confidence` | full per-class + mean curves, swept across all confidence thresholds | a fresh `model.val()` forward pass on the test split (read-only inference, not training) -- its scalar outputs (mAP50/mAP50-95/P/R) are asserted to match the saved JSON exactly before any curve is drawn; Ultralytics does not persist these sweep arrays anywhere else |
| `train_box_loss`, `train_cls_loss`, `train_dfl_loss`, `val_box_loss`, `val_cls_loss`, `val_dfl_loss`, `map50`, `map50_95`, `precision`, `recall` | 10 curves vs. logged training step | `outputs/yolo/runs/yolov8n/results.csv` |

Training-curve x-axis is the **logged step index**, not the raw `epoch` column -- that column contains
real duplicates from this run's mid-training resumes (Section 7.1), so plotting it directly would draw a
misleading back-and-forth line; every affected chart carries this as an on-chart caption, not just a
footnote here. **"mAP50 vs. confidence" was requested but deliberately not produced** -- mAP is already an
integral over the full precision-recall curve, so "mAP at one confidence value" is not a standard or
meaningful quantity; the PR curve and F1-vs-confidence curve serve the same diagnostic purpose without
inventing a non-standard metric.

### 9.2 `outputs/visualizations/classification/` -- ResNet50 (the final verifier)

Model/checkpoint: `models/classifier_resnet50/best.pt`. Split: **test** (6,798 samples: 1,458 full images +
5,340 YOLO-crops) for everything except the 5 training curves, which use the **training/validation** history.

| File | What it shows | Source |
|---|---|---|
| `confusion_matrix_raw` / `_normalized` | 6x6, all test samples (full images + crops together) | `outputs/classification/resnet50/evaluation/metrics_test.json` |
| `per_class_precision`, `per_class_recall`, `per_class_f1`, `per_class_support` | one bar per class (6 classes) | same JSON |
| `train_loss`, `val_loss`, `train_acc`, `val_acc`, `val_macro_f1` | vs. epoch (1-15, a single clean run, no duplicates) | `outputs/classification/resnet50/train_summary.json:history` |
| `full_vs_crop_accuracy_macrof1` | full-image vs. YOLO-crop, accuracy + macro-F1 side by side | `metrics_test.json:by_sample_type` |
| `full_vs_crop_per_class_recall` | same split, per-class recall | same |

### 9.3 `outputs/visualizations/pipeline/` -- integrated YOLOv8n + ResNet50 system

Reads the **already-saved** full test-set evaluation only (`outputs/pipeline_evaluation/pipeline_eval_test.csv`
+ `..._summary.json`, 1,458 images) -- **the 1,458-image pipeline run was not repeated** to build any chart
in this section.

| File | What it shows | Source |
|---|---|---|
| `confusion_matrix_raw` / `_normalized` | 6x6, built **only from the 1,339 confident (`status=="ok"`) rows** -- a confusion matrix has no cell for "declined to answer," so the abstained 8.2% is reported separately (below) rather than dropped silently or forced into a class that was never predicted. Recomputed accuracy from this subset was asserted to match `accuracy_when_confident` (0.909) exactly before plotting. | `pipeline_eval_test.csv` |
| `per_class_accuracy`, `per_class_precision`, `per_class_recall`, `per_class_f1` | same confident-only subset, via sklearn | same |
| `decision_status`, `decision_status_bar` | ok / model_disagreement / low_confidence / error counts | `pipeline_eval_test_summary.json:status_breakdown` |
| `detection_path` | "no detection -> full-image CNN" vs. "YOLO detection -> crop -> CNN verification" | `pipeline_eval_test.csv:decision_path` |
| `accuracy_comparison_labeled_conditions` | all 5 accuracy figures from Sections 7.2/7.5 side by side, explicitly grouped/labeled as ResNet50-standalone vs. integrated-system conditions -- never implied to be equivalent | `metrics_test.json` + `pipeline_eval_test_summary.json` |

### 9.4 `outputs/visualizations/dashboard/` -- final summary + the 1024 ablation, kept separate

- **`final_dashboard`**: one figure, four panels (detection / verification / input-mode / integrated),
  covering **only the final selected models** (640 + ResNet50 + the integrated system) -- the 1024 ablation
  does not appear anywhere on this figure.
- **`ablation_1024_not_selected`**: a separate file, titled "PARTIAL / INCOMPLETE ABLATION -- NOT SELECTED"
  in the same ochre used for "partial" throughout this project's reporting, showing the epoch-16/40
  best-fitness checkpoint's test-split metrics with the epoch-17/40 stop point stated explicitly. Reads
  `outputs/yolo/metrics/yolov8n_1024_eval_test.json` + `yolov8n_1024_train_summary.json` (Section 7.1b).

### 9.5 Regenerating

```
python main.py visualize-yolo
python main.py visualize-classifier
python main.py visualize-pipeline
python main.py visualize-dashboard
```

Each script is independent and re-runnable on its own; `generate_yolo_plots.py` is the only one that
touches the GPU (one read-only `model.val()` pass, ~30s on this hardware) -- pass `--skip-confidence-curves`
to skip it if only the confusion matrix / per-class / training-curve charts are needed.

## 10. RiceGuard AI -- Android client + local API server

Built on top of, and never modifying, the frozen Intelligence Layer above: `models/yolo_yolov8n/best.pt`
and `models/classifier_resnet50/best.pt` remain exactly the checkpoints verified in Sections 7.1/7.2, and
`server/main.py` calls `pipeline.inference_pipeline.RiceDiseasePipeline.run()` directly rather than
duplicating any detection/classification/Grad-CAM/recommendation logic. Full setup, the phone-to-PC data
flow, and privacy/deletion behavior are documented in `android/README.md`; this section records what was
built, what was actually verified, and what was not.

**Two new components, two different verification stories, disclosed honestly:**

- **`server/` (FastAPI + Uvicorn, the Application Layer).** Installed, started, and exercised live during
  development: `GET /health` and `POST /predict` were both called against real test images covering the
  YOLO-detection-with-agreement path and the no-detection/full-image path, plus deliberate error cases
  (corrupt upload, wrong MIME type, a path-traversal attempt against `/results/{id}/{filename}`, a missing
  result). All returned the expected status codes and structured JSON. One real concurrency bug was found
  and fixed during this testing -- `RiceDiseasePipeline`'s `gradcam_dir` is fixed once at server startup
  (it cannot be threaded per-request through an already-constructed pipeline instance), so every request's
  Grad-CAM files were initially landing in one shared, unnamespaced directory: a second request would
  silently overwrite the first's saved images before its client ever fetched them. Fixed by relocating each
  request's Grad-CAM files into its own `server/tmp/<request_id>/` folder *before* releasing the single-GPU
  lock for the next queued request (`server/pipeline_service.py:_relocate_gradcam_files`), then re-verified
  with two sequential requests confirming both remained independently fetchable.
- **`android/` (Kotlin, Jetpack Compose, CameraX, Room -- the Client Layer).** This machine has no JDK,
  Android SDK, or Gradle installed (checked before writing any code, not assumed) -- so unlike the server,
  **the Android app has not been compiled or run**. Every file was written to compile cleanly by manual
  review: every screen's function signature was cross-checked against its call site in
  `RiceGuardNavGraph.kt`, every Kotlin file's `package` declaration was verified to match its directory
  (51/51 files), and every internal cross-file reference was checked for a matching declaration. Several
  real mistakes were caught this way before being left in the tree -- among them, a stray import
  (`DynamicBaseUrlInterceptor.let`) that would have been a hard compile error, and two screens
  (`HomeScreen`'s recent-scans list and `HistoryScreen`'s history rows) where a card's `onClick` was defined
  but never actually attached to a `clickable` modifier, so tapping a row would silently have done nothing.
  This disclosure is deliberate: **the first real build must happen in Android Studio** on a machine with
  the SDK installed (see `android/README.md` "First build" for the specific, narrow things most likely to
  need a version bump), and any remaining issue found there should be treated as expected for
  never-compiled code, not as a sign the review step above was skipped.

**What was explicitly preserved, not touched:** `models/yolo_yolov8n/best.pt`, `models/classifier_resnet50/best.pt`,
`recommendation/knowledge_base.json`'s `NEEDS_AUTHORITATIVE_VALIDATION` markers, and every number in Sections
7.1/7.2/7.5 above -- confirmed unchanged (byte-identical timestamps) after this work, the same way every
other addition to this project has been.

**Scope boundary:** this section covers the client/server wrapper only. It does not re-run, re-evaluate, or
change the verdict of the 640-vs-1024 detector comparison (Section 7.1b) or any other already-decided
question in this document.

**Implementation frozen 2026-08-19** after a final source-level-only verification pass (18 explicit checks:
package declarations, cross-file references, manifest, Gradle catalog, Room, Retrofit-vs-Pydantic schema
parity, navigation routes, click-handler wiring, the capture/confirm/upload flow, history deletion, IP/port
configurability, endpoint paths, result-URL construction, uncertainty-state mapping, filesystem-path
exposure, upload cleanup, and on-device-inference absence -- see `android/README.md` "FIRST REAL BUILD /
DEVICE VERIFICATION" for the full list and what each one found). One additional real gap was caught and
fixed in this pass: `RegionDto` (Android) was missing the `cnn_probabilities` field present in the server's
`RegionOut` schema -- Gson would have silently dropped it rather than failing to build, so it was only
found by an explicit field-by-field diff. No ML model, dataset, or already-validated result was touched.
The remaining verification step -- an actual Gradle build and on-device run -- requires Android Studio with
a JDK/SDK installed, which this environment does not have; it is the one item this project cannot
self-certify further.
