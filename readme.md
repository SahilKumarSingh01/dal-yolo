# DAL-YOLO11n: UAV Road Damage Detection

Evaluation repository comparing a baseline YOLO11n object detection model against a custom DAL-YOLO11n variant utilizing deformable attention mechanisms (`C3k2_DAttention`) for UAV-based road damage detection.

## Directory Structure

```text
.
├── documents/
│   ├── presentation.pptx    # Project presentation slides
│   ├── report.pdf           # Technical report
│   └── research_paper.pdf   # Research paper draft
├── models/
│   ├── yolo_best.pt         # Fine-tuned baseline YOLO11n weights
│   └── dal_best.pt          # Fine-tuned custom DAL-YOLO11n weights
├── main.ipynb               # Data handling, architecture, training, and testing
├── requirements.txt         # Dependencies
└── README.md
```

## Overview

Road damage identification from aerial imagery involves handling arbitrary object orientations, low contrast, and high scale variation. This implementation modifies the YOLO11n architecture by replacing a default bottleneck block with a shape-preserving deformable attention module (`C3k2_DAttention`) implemented via `torchvision.ops.deform_conv2d`.

Both models are evaluated on a 14-class dataset derived from UAV aerial images.

## Setup

### Requirements
* Python 3.10 or higher
* PyTorch with CUDA support

### Installation
```bash
git clone https://github.com/username/repository-name.git
cd repository-name
pip install -r requirements.txt
```

## Usage

### Training and Evaluation
All dataset splitting, model definitions, training routines, and benchmark evaluations are executed through `main.ipynb`.

To run the notebook:
```bash
jupyter notebook main.ipynb
```

### Inference
Load saved weights from the `models/` directory using the `ultralytics` API:

```python
from ultralytics import YOLO

# Load custom model
model = YOLO('models/dal_best.pt')

# Run inference
results = model.predict(source='path/to/image.jpg', imgsz=640)
```

## Dataset Specifications

* **Source:** Kaggle (`sweetodd/uavroad`)
* **Total Samples:** 5,002 image and XML pairs
* **Splits:**
  * Train: 4,001 images
  * Validation: 500 images
  * Test: 501 images (945 target instances)
* **Target Classes (14):** `alligator crack`, `damaged curb`, `damaged drainage`, `damaged guardrail`, `damaged marking`, `damaged sign`, `greenbelt encroachment`, `longitudinal crack`, `oblique crack`, `pothole`, `repair`, `transverse crack`, `trash`, `water`

## Experimental Results

Evaluations conducted on the 501 held-out test images (`imgsz=640`, `batch=16`).

### Aggregate Metrics

| Model | Precision | Recall | mAP50 | mAP50-95 | Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline YOLO11n | 0.6353 | 0.5419 | 0.5559 | 0.3395 | 2,608,042 |
| DAL-YOLO11n (Custom) | **0.6841** | **0.5726** | **0.5931** | **0.3537** | **2,592,570** |

### Per-Class AP50:95 Comparison

| Class Name | Baseline YOLO11n | DAL-YOLO11n | Difference |
| :--- | :---: | :---: | :---: |
| alligator crack | 0.3902 | 0.3784 | -0.0118 |
| damaged curb | 0.1396 | 0.1437 | +0.0041 |
| damaged drainage | 0.7784 | 0.7485 | -0.0299 |
| damaged guardrail | 0.3297 | 0.4115 | +0.0818 |
| damaged marking | 0.1816 | 0.2088 | +0.0272 |
| damaged sign | 0.3395 | 0.3537 | +0.0142 |
| greenbelt encroachment | 0.0810 | 0.0960 | +0.0150 |
| longitudinal crack | 0.3365 | 0.3540 | +0.0175 |
| oblique crack | 0.3417 | 0.3197 | -0.0220 |
| pothole | 0.3767 | 0.4544 | +0.0777 |
| repair | 0.6924 | 0.6325 | -0.0599 |
| transverse crack | 0.4209 | 0.4176 | -0.0033 |
| trash | 0.1697 | 0.1597 | -0.0100 |
| water | 0.1751 | 0.2735 | +0.0984 |

## Documentation

Further details regarding architectural design, loss functions, and detailed qualitative outputs can be found under the `documents/` directory.