#   RiceLeafDiseaseBD: A Field-Based Annotated Smartphone Image Dataset of Healthy and Diseased Rice Leaves from Bangladesh

## Overview

RiceLeafDiseaseBD is a curated image dataset of rice leaves (*Oryza sativa*) collected from major rice-producing regions of Bangladesh. The dataset is designed to support research in computer vision, plant disease analysis, and precision agriculture. It provides both original rice leaf images and region-level annotated data for common rice leaf diseases.

The dataset emphasizes consistency, reproducibility, and transparency and is released without predefined training, validation, or test splits to allow flexible experimental design by downstream users.

---

## Dataset Summary

- **Domain:** Agricultural imaging, computer vision, plant pathology
- **Total Images:** 9,769
- **Number of Classes:** 6
- **Image Resolution:** 1024 × 1024 pixels
- **Total Annotations:** 76,150 bounding boxes
- **Annotation Type:** Bounding boxes (region-level lesion localization)
- **Annotation Format:** YOLO-compatible text format
- **Geographic Region:** Gazipur, Dhaka, Bangladesh
- **Data Collection Period:** Summer 2025
- **Rice Varieties:** BRRI dhan28, BRRI dhan29

---

## Dataset Contents

The dataset is divided into the following components:

### 1. Original Images
- Contains unannotated rice leaf images
- Organized into class-wise folders
- Suitable for image-level classification, visualization, and general analysis

### 2. Annotated Data
- Contains region-level annotations for diseased images
- Includes bounding boxes indicating diseased regions
- Annotation files are provided in YOLO-compatible plain-text format
- Visualization images with bounding box overlays are included

---

## Directory Structure

```
RiceLeafDiseaseBD/
├── README.md
├── Data_Collection_Pipeline.png
├── Annotation_Protocol.pdf
├── Dataset_Metadata.csv
├── Original_Images/
│   ├── Healthy/
│   │   ├── healthy_001.jpg
│   │   ├── healthy_002.jpg
│   │   └── ...
│   ├── Blast/
│   ├── Brown_Spot/
│   ├── Leaf_Smut/
│   ├── Rice_Tungro/
│   └── Sheath_Blight/
└── Annotated_Data/
    ├── Blast/
    │   ├── images/
    │   │   ├── blast_001.jpg
    │   │   └── ...
    │   └── labels/
    │       ├── blast_001.txt
    │       └── ...
    ├── Brown_Spot/
    ├── Leaf_Smut/
    ├── Rice_Tungro/
    └── Sheath_Blight/
```

---

## Image Acquisition

- Images were captured using two smartphone models: CMF Phone 2 Pro (50 MP) and Google Pixel 5.
- Rice leaves were photographed under natural field conditions with original backgrounds.
- Data were collected during the critical Summer 2025 growing season from multiple rice fields.
- Images capture natural variations in lighting, leaf dimensions, plant growth stages, and disease progression.
- All images were standardized to a resolution of 1024 × 1024 pixels prior to release.

---

## Disease Classes

The dataset includes six classes:

1. Healthy
2. Blast (caused by *Magnaporthe oryzae*)
3. Brown Spot (caused by *Bipolaris oryzae*)
4. Leaf Smut (caused by *Entyloma oryzae*)
5. Rice Tungro (viral disease transmitted by leafhoppers)
6. Sheath Blight (caused by *Rhizoctonia solani*)

All image filenames begin with their corresponding class name as a prefix to ensure traceability. The dataset maintains a balanced distribution with an average of 1,628 images per class.

---

## Annotation Method

Region-level annotations were generated manually by trained annotators under guidance from agricultural experts, experienced rice farmers. The annotation process includes:

- Visual identification of disease lesions and symptomatic areas
- Rectangular bounding box placement around each diseased region
- Standardized protocols to ensure consistency across disease categories
- Validation by agricultural experts

Final annotations are provided as bounding boxes enclosing diseased regions. All annotated images were visually inspected to ensure accuracy and consistency across disease classes.

**No automated detection models (e.g., YOLO, CNNs) were used during annotation.**

A detailed description of the annotation rules and validation steps is provided in `Annotation_Protocol.pdf`.

---

## Annotation Format

Annotations are stored in YOLO-compatible plain-text format.
Each annotation file corresponds to one image with the same base filename and follows the format:

```
<class_id> <center_x> <center_y> <width> <height>
```

Example:
```
0 0.512695 0.651855 0.005859 0.006836
0 0.602051 0.639648 0.004883 0.005859
1 0.473145 0.839355 0.051758 0.083008
```

All coordinates are normalized with respect to image width and height.

**Class ID Mapping:**
- 0: Brown Spot
- 1: Blast
- 2: Healthy
- 3: Leaf Smut
- 4: Rice Tungro
- 5: Sheath Blight

---

## File Naming Convention

- All filenames begin with the class name as a prefix.
- Image files and annotation files follow a one-to-one naming correspondence, differing only by file extension.

Example:
```
brown_spot_001.jpg
brown_spot_001.txt
```

---

## Dataset Splits

The dataset is released without predefined training, validation, or test splits.
Users are encouraged to create task-specific splits according to their experimental requirements.

---

## Pipeline Documentation

The file `Data_Collection_Pipeline.png`, located in the root directory, provides a visual summary of the complete data collection, preprocessing, annotation, and dataset organization workflow. This figure is included to improve transparency and reproducibility.

---

## Intended Use

The dataset can be used for:
- Rice leaf disease classification
- Lesion localization and detection
- Disease severity assessment and lesion counting
- Benchmarking object detection algorithms
- Multi-task learning combining classification and localization
- Transfer learning studies
- Self-supervised learning approaches
- Development of lightweight mobile-based diagnostic applications for edge devices
- Precision agriculture research

---

## Limitations

- Annotations are bounding boxes and do not provide pixel-level segmentation.
- Images were collected from a specific geographic region (Gazipur, Bangladesh) during a particular season (Summer 2024).
- Annotation density varies by disease characteristics—Blast and Brown Spot have numerous small lesions, while Sheath Blight has larger symptomatic areas.
- Annotations represent visually observable symptoms validated by experts but are not laboratory-confirmed diagnoses.

---

## Technical Specifications

- **Framework Compatibility:** TensorFlow, PyTorch, Keras
- **Computer Vision Libraries:** OpenCV, scikit-image, Albumentations
- **Object Detection Models:** YOLOv5, YOLOv8, Faster R-CNN
- **Supported Tasks:** Classification, object detection, instance segmentation, severity estimation

---

## Citation

If you use this dataset in your research, please cite the corresponding data paper.

Munna, Mohammad Mehedi Hasan; Islam, Toriqul ; Saha, Smita; Safy, Iqbal Hossain; Bari, Saiful ; Akter, Jasmin; Rahman, Md Momtajur; Akter, Riya; Antu, Nahid Hossain; Kabir, Acramul Haque; Tulona, Reefka Fabliha; Islam, Ashraful; Amin, M Ashraful (2026), " RiceLeafDiseaseBD: A Field-Based Annotated Smartphone Image Dataset of Healthy and Diseased Rice Leaves from Bangladesh", Mendeley Data, V1, doi: 10.17632/86s4jzj2m4.1

---

## License

This dataset is released under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.

---

## Acknowledgments

The dataset was created with support from:
- Bangladesh Rice Research Institute (BRRI)
- Agricultural extension officers from Gazipur region
- Experienced rice farmers who provided expert validation

---

## Contact

For questions or issues related to the dataset, please contact the dataset authors.

mohammadmehedi.munna@gmail.com


---

## Version

- **Version:** v1.0
- **Release Date:** 2026
