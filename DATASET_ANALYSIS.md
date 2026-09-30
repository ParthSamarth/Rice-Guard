# RiceLeafDiseaseBD — Folder Analysis & Understanding

This document summarizes everything found in this folder after reading the README, the annotation
protocol PDF, both diagram images, the metadata spreadsheet, and inspecting the contents of the
5.6 GB `RiceLeafDiseaseBD.zip` archive directly (file listing, without full extraction). It is meant as a
single reference for what this dataset is, how it's organized, and where the shipped documentation
and the actual archive contents disagree.

---

## 1. What this folder is

This folder is a local copy of the **Mendeley Data record** for:

> **RiceLeafDiseaseBD: A Field-Based Annotated Smartphone Image Dataset of Healthy and Diseased
> Rice Leaves from Bangladesh**
> Munna, M.M.H. et al. (2026), Mendeley Data, V1, doi: `10.17632/86s4jzj2m4.1`, license **CC BY 4.0**.

It is an agricultural computer-vision dataset: smartphone photos of rice (*Oryza sativa*) leaves —
healthy and five diseased categories — with bounding-box (YOLO-format) annotations marking diseased
regions, collected in Gazipur, Bangladesh during Summer 2025.

---

## 2. Files present in this folder

| File | Size | What it is |
|---|---|---|
| `README.md` | ~8 KB | The dataset's own README: overview, stats, directory-structure diagram, class list, annotation format spec, licensing, and citation info. Primary documentation file. |
| `Annotation Protocol.pdf` | ~134 KB | A short (5-page) formal protocol document describing how annotators labeled diseased regions: per-disease annotation rules, the QC/validation pipeline, and inter-annotator agreement statistics. Details below in §6. |
| `Data Collection Pipeline.png` | 2752×1536 px | An infographic (4 panels: a–d) illustrating the full pipeline from smartphone capture → cleaning/preprocessing → annotation/QC → final dataset outputs. Useful as a one-glance summary of the methodology. |
| `Dataset folder Structure.png` | 2752×1536 px | An infographic showing how the Mendeley record is packaged: the standalone preview files (PDF/PNGs/README) sit outside the zip, while the zip itself contains `RiceLeafDiseaseBD/` with `Original_Images/`, `Annotated images (visual with labels)/`, and a metadata CSV. |
| `Dataset metadata.xlsx` | ~468 KB | A 2-sheet Excel workbook with **per-image** metadata (not included inside the zip itself — see discrepancy in §7). Sheet details in §5. |
| `RiceLeafDiseaseBD.zip` | **5.6 GB** | The actual image + annotation archive. Not fully extracted (too large); inspected via its internal file listing. Real structure documented in §4. |

---

## 3. Dataset summary (as documented)

- **Domain:** Agricultural imaging / computer vision / plant pathology
- **Total images:** 9,769 (verified against both the metadata spreadsheet and the zip's file listing)
- **Classes:** 6 — Healthy + 5 diseases
- **Image resolution (released):** standardized to 1024×1024 px (originals were captured at various
  phone-native resolutions, e.g. 1200×1600, 3472×4624, then resized)
- **Total bounding boxes:** README states **76,150**; the Annotation Protocol PDF states **74,836**
  (these two source documents disagree — see §7)
- **Annotation format:** YOLO-compatible `.txt` (`class_id center_x center_y width height`, normalized 0–1)
- **Geography:** Gazipur, Dhaka, Bangladesh
- **Collection period:** Summer 2025 (monsoon season, per the pipeline diagram)
- **Rice varieties photographed:** BRRI dhan28, BRRI dhan29
- **Capture devices:** CMF Phone 2 Pro (50 MP) and Google Pixel 5
- **License:** CC BY 4.0

### Class distribution (verified — matches metadata spreadsheet, zip listing, and protocol PDF)

| Class ID (YOLO) | Class | Images | Annotated? | Avg. boxes/image | Total boxes (per PDF) |
|---|---|---|---|---|---|
| 2 | Healthy | 1,575 | No (empty/no annotation folder) | – | – |
| 1 | Blast | 1,326 | Yes | ~9 | 12,279 |
| 0 | Brown Spot | 2,178 | Yes | ~10 | 21,885 |
| 3 | Leaf Smut | 724 | Yes | ~12 | 8,654 |
| 4 | Rice Tungro | 2,244 | Yes | ~7 | 14,762 |
| 5 | Sheath Blight | 1,722 | Yes | ~10 | 17,256 |
| — | **Total** | **9,769** | | | **74,836** |

Note the YOLO class-ID order is *not* alphabetical/class-list order — Brown Spot is class 0, Blast is
class 1, Healthy is class 2, etc. This exact mapping was cross-checked between the README and the PDF
and both agree.

---

## 4. Actual structure inside `RiceLeafDiseaseBD.zip` (verified via file listing)

The real internal structure differs slightly from what the README's ASCII diagram describes. Verified
directly from the zip's central directory listing (26,187 entries):

```
RiceLeafDiseaseBD/
├── Original images/                                   (note: singular "images", lowercase i — not "Original_Images")
│   ├── Blast/                 1,326 files   (blast_1.jpg, blast_10.jpg, ... no zero-padding)
│   ├── Brown spot/             2,178 files  ("brown spot_1.jpg" — space, not underscore)
│   ├── Healthy/                1,575 files  (Healthy_1.jpg, capitalized)
│   ├── Leaf smut/               724 files   (Leaf smut_1.jpg)
│   ├── Rice Tungro/            2,244 files  (Rice Tungro_1.jpg)
│   └── Sheath blight/          1,722 files  (Sheath blight_1.jpg)
└── Annotated images ( visual with labels)/             (note stray space after opening paren)
    ├── Blast/
    │   ├── labels/             1,326 .txt files (blast_1.txt, ...)
    │   └── visuals/            1,326 .jpg files (bounding boxes drawn on image)
    ├── Brown spot/
    │   ├── labels/             2,178 files
    │   └── visuals/            2,178 files
    ├── Leaf smut/
    │   ├── labels/               724 files
    │   └── visuals/               724 files
    ├── Rice Tungro/
    │   ├── labels/             2,244 files
    │   └── visuals/             2,244 files
    └── Sheath blight/
        ├── labels/             1,722 files
        └── visuals/             1,722 files
```

No `Healthy` folder exists under "Annotated images" (consistent with Healthy having no diseased
regions to box). No metadata CSV/spreadsheet was found at the root of the zip — the metadata instead
ships separately as `Dataset metadata.xlsx` in this folder (see §7, discrepancy #3).

File counts per class match exactly across: the metadata spreadsheet, the zip's `Original images/`
folders, and the zip's `Annotated images/*/labels` and `*/visuals` folder pairs — total 9,769 original
images and (1326+2178+724+2244+1722 =) 8,194 annotated image/label pairs.

---

## 5. `Dataset metadata.xlsx` contents

Two sheets, confirmed by parsing the underlying XML directly (Excel-compatible tool wasn't available
in this environment, so the workbook was unzipped and its `sheetN.xml` / `sharedStrings.xml` parsed
manually):

### Sheet 1 — `Image_Metadata` (9,769 data rows, one per image)

Columns:

| Column | Meaning |
|---|---|
| `Class` | Disease class name (e.g. "Healthy", "Leaf smut", "Rice Tungro", "Sheath blight", "blast", "brown spot" — casing is inconsistent across classes) |
| `Original_Filename` | The raw filename as captured by the phone (e.g. `1743593818130.jpg`, or camera-style names like `20250409_134626.jpg`) — this is a timestamp/epoch-style name for CMF Phone 2 Pro captures and a date_time name for Pixel 5 captures |
| `New_Filename` | The renamed/standardized filename used in the released dataset (e.g. `Healthy_1.jpg`) |
| `Original_Width_px` / `Original_Height_px` | Native capture resolution before resizing (varies: e.g. 1200×1600, 3472×4624) |
| `Original_FileSize_KB` | File size before resizing |
| `Resized_Width_px` / `Resized_Height_px` | Always 1024×1024 in the sample checked — the standardized release resolution |
| `Resized_FileSize_KB` | File size after resizing to 1024×1024 |

This sheet effectively provides full **provenance tracing** from original phone capture to the final
released filename for every one of the 9,769 images.

### Sheet 2 — `Class_Summary_Statistics` (6 rows, one per class)

Aggregated per-class stats: `Class`, `Num_Images`, `Mean_Orig_Width_px`, `Mean_Orig_Height_px`,
`Mean_Orig_FileSize_KB`, `Mean_Resized_FileSize_KB`, `Min_Orig_FileSize_KB`, `Max_Orig_FileSize_KB`.

Confirmed image counts here (Healthy 1,575; Leaf smut 724; Rice Tungro 2,244; Sheath blight 1,722;
blast 1,326; brown spot 2,178) sum to exactly 9,769 and match both the README and the zip listing.

---

## 6. Annotation Protocol (from `Annotation Protocol.pdf`)

A 5-page methods document covering:

1. **Purpose & objectives** — standardized disease identification, consistent bounding-box placement,
   expert validation, YOLO-compatible output. Explicitly states **no automated detection models
   (YOLO/CNNs) were used to generate the annotations** — all boxes were drawn manually.
2. **Scope** — annotations cover the 5 disease classes only; Healthy images have no bounding boxes.
3. **Annotation challenges & solutions** — irregular lesion boundaries, overlapping symptoms,
   similar early-stage symptoms, and natural variation were addressed via trained annotators, expert
   supervision, standardized per-disease guidelines, and multi-stage validation.
4. **Per-disease annotation rules** (Table 3 in the PDF):
   - **Blast:** spindle-shaped lesions, gray-white centers, brown margins, each lesion boxed
     individually, typical size 1–2 cm.
   - **Brown Spot:** circular/oval spots, brown centers, each spot boxed individually; can be very
     high density (explains the highest avg. boxes/image, ~10, and highest total box count, 21,885).
   - **Leaf Smut:** small black spots/streaks following leaf veins, all visible pustules included,
     typical size 2–5 mm (smallest lesions, hence highest boxes/image despite fewest images).
   - **Rice Tungro:** yellowing/discoloration and mottled chlorotic patterns rather than discrete
     lesions — diffuse symptoms, lowest avg. boxes/image (~7).
   - **Sheath Blight:** large irregular lesions, greenish-gray to brown, water-soaked borders,
     typically 1–3 cm or larger.
5. **YOLO conversion pipeline:** visual (hand-drawn) annotations → red-box detection → coordinate
   normalization → YOLO `.txt` format.
6. **Quality control pipeline:** first-pass annotation → peer review → expert validation → automated
   checks (on a 10% sample) → visual verification, with a correction loop feeding back to first-pass.
7. **Inter-annotator agreement** (measured on n=200 images): Cohen's κ = 0.82 ("substantial
   agreement"), IoU = 0.78 ("good overlap"), F1 = 0.85 ("high detection consistency"). This is a
   genuinely useful data-quality signal for anyone deciding whether to trust the box placements.
8. **Ethical considerations:** farm owner permissions obtained, local farmer expertise acknowledged,
   dataset released publicly for agricultural research.

---

## 7. Discrepancies noticed between documentation and the actual archive

Worth knowing before writing any code against this dataset:

1. **Total bounding box count disagrees between sources.** README says 76,150; the Annotation
   Protocol PDF says 74,836 (and its own per-class breakdown in Table 4 sums to exactly 74,836:
   21,885+12,279+8,654+14,762+17,256 = 74,836). The PDF's number is internally self-consistent; the
   README's is not backed by a shown breakdown. **Treat 74,836 as the more trustworthy figure**, but
   verify by summing bounding-box lines in the actual `labels/*.txt` files if exact precision matters.
2. **Folder/file naming does not match the README's example.** The README's directory diagram and
   naming-convention section describe snake_case, zero-padded names like `Original_Images/`,
   `Annotated_Data/Blast/images/`, `brown_spot_001.jpg`. The real zip instead uses `Original images/`
   (space, no underscore), `Annotated images ( visual with labels)/` (note the stray space after the
   parenthesis), sub-folders named `labels/` and `visuals/` (not `images/`), and filenames with no
   zero-padding and inconsistent casing/spacing per class (e.g. `blast_1.jpg`, `brown spot_1.jpg`,
   `Healthy_1.jpg`, `Rice Tungro_1.jpg`, `Sheath blight_1.jpg`, `Leaf smut_1.jpg`). Any loader script
   should be written against the **real** names (verified above), not the README's illustrative ones.
3. **No metadata CSV inside the zip.** Both the README and the `Dataset folder Structure.png`
   infographic describe a `Dataset_Metadata.csv` living inside the zip alongside the image folders.
   The actual zip's root contains only the two top-level folders (`Original images/` and
   `Annotated images ( visual with labels)/`) — no CSV at any level was found in the listing. The
   metadata instead exists as the separate `Dataset metadata.xlsx` file sitting alongside the zip in
   this folder, in `.xlsx` format rather than `.csv`.
4. **README's "Limitations" section has a season inconsistency.** It says images were collected
   "during a particular season (Summer 2024)" while every other part of the documentation (dataset
   summary, pipeline diagram) says **Summer 2025**. This looks like a leftover/copy-paste error in
   the README — Summer 2025 is corroborated everywhere else, including original-file EXIF-style
   filenames from the metadata sheet (e.g. `1743593818130` epoch-ms ≈ April 2025, `20250409_134626`
   ≈ April 9, 2025).
5. **Class casing is inconsistent throughout, including in the shipped metadata itself.** The
   metadata spreadsheet's `Class` column mixes cases: "Healthy", "Leaf smut", "Rice Tungro",
   "Sheath blight" (capitalized) vs. "blast", "brown spot" (lowercase). Code consuming this column
   should normalize case before matching against class names.

---

## 8. Practical notes for anyone building on this dataset

- **9,769 total images**, only **8,194** of which (the 5 disease classes, excluding Healthy) have
  bounding-box annotations; Healthy images are classification-only.
- Every annotated image has a matching pair: one entry in `labels/` (YOLO `.txt`) and one in
  `visuals/` (same image with boxes drawn on it for human inspection) — counts match exactly, so a
  simple loop pairing by filename stem is reliable.
- All released images are 1024×1024 regardless of original phone resolution — any pixel-space
  bounding box math should assume this canvas size (or better, use the pre-normalized YOLO
  coordinates directly).
- No official train/val/test split is provided — splitting is left to the user.
- The `Dataset metadata.xlsx` file gives full original→released filename provenance per image if
  traceability back to the raw phone capture is ever needed.
- Given the 5.6 GB zip size, this analysis was done via directory-listing inspection
  (`unzip -l`) rather than full extraction — no image pixel content or actual label-file box
  coordinates beyond the examples shown in the PDF were inspected.

---

*Generated by reading `README.md`, `Annotation Protocol.pdf`, `Data Collection Pipeline.png`,
`Dataset folder Structure.png`, `Dataset metadata.xlsx` (parsed via its internal XML), and the file
listing of `RiceLeafDiseaseBD.zip` (26,187 entries) — no files in this folder were modified.*
