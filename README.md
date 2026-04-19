# Thermal Image-Based Anomaly Detection

Object detection pipeline for identifying anomalies in thermal images using YOLOv8x, with systematic evaluation across multiple random seeds for robustness analysis.

Developed as a course project for ITE4052 Computer Vision, Hanyang University ERICA (Sep. 2025 – Dec. 2025).

---

## Overview

- **Task**: Binary object detection on thermal images (normal vs. anomaly)
- **Model**: YOLOv8x (Ultralytics), fine-tuned on custom thermal dataset
- **Input format**: COCO JSON annotations → converted to YOLO format
- **Classes**: `normal` (0), `anomaly` (1)

---

## Architecture & Pipeline

```
COCO JSON + Thermal Images
        │
        ▼
  Data Preprocessing
  (stratified train/val split 9:1, COCO→YOLO conversion)
        │
        ▼
  YOLOv8x Fine-tuning
  (SGD, lr=0.0008, conservative augmentation for thermal domain)
        │
        ▼
  Evaluation
  (mAP50-95, Precision, Recall, FinalScore)
```

**Augmentation strategy**: Conservative settings (HSV jitter, small translate/scale, low flip probability) chosen to preserve thermal image characteristics and avoid distribution shift.

---

## Results

Evaluated across 3 random seeds for robustness. Validation set (300 images, 150 per class):

### Seed 0
| Class   | Images | mAP50 | mAP50-95 |
|---------|--------|-------|----------|
| normal  | 150    | 0.995 | 0.968    |
| anomaly | 150    | 0.995 | 0.974    |
| **all** | **300**| **0.995** | **0.971** |

```
mAP50-95  : 0.7942
Precision : 0.9687
Recall    : 0.6809
FinalScore: 0.8003   (mAP×0.8 + Precision×0.1 + Recall×0.1)
```

### Seed 1
| Class   | Images | mAP50 | mAP50-95 |
|---------|--------|-------|----------|
| normal  | 150    | 0.986 | 0.963    |
| anomaly | 150    | 0.995 | 0.979    |
| **all** | **300**| **0.991** | **0.971** |

```
mAP50-95  : 0.7929
Precision : 0.9616
Recall    : 0.6776
FinalScore: 0.7982
```

### Seed 2
| Class   | Images | mAP50 | mAP50-95 |
|---------|--------|-------|----------|
| normal  | 150    | 0.995 | 0.966    |
| anomaly | 150    | 0.995 | 0.977    |
| **all** | **300**| **0.995** | **0.972** |

### Average across seeds
| Metric       | Mean   |
|--------------|--------|
| mAP50-95     | 0.7936 |
| Precision    | 0.9663 |
| FinalScore   | ~0.800 |

---

## Tech Stack

- Python 3.x
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV, NumPy, pandas
- scikit-learn (stratified split)
- PyTorch

---

## Project Structure

```
.
├── main.py             # Full pipeline: data conversion, training, evaluation
├── requirements.txt
└── README.md
```

---

## Usage

```bash
pip install -r requirements.txt

# Train
python main.py --mode train --model yolov8x.pt --epochs 30 --seed 0

# Evaluate
python main.py --mode test --weights runs/detect/train/weights/best.pt --seed 0
```

> **Note**: Update `WORKSPACE` path in `main.py` to point to your local dataset directory.

---

## Dataset

- Thermal images with COCO-format JSON annotations
- Binary classification: normal vs. anomaly regions
- Train/val split: 90/10 (stratified by image-level label)
- Raw data not included in this repository.
