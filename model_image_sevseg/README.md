# SevSeg-YOLO — Industrial Defect Detection + Severity Scoring

A custom extension of YOLO26 that performs **three tasks simultaneously**:
1. **Defect Detection** — bounding box localization
2. **Severity Scoring** — continuous severity score [0, 10] per defect
3. **Zero-Annotation Segmentation** — approximate defect masks without pixel-level labels

## Architecture

SevSeg-YOLO extends YOLO26 with a `ScoreHead` that predicts a severity score alongside standard detection outputs. The `MaskGenerator` produces segmentation masks from intermediate feature maps using traditional image processing (no extra training needed).

```
Input Image
    → YOLO26 Backbone + Neck (P3/P4/P5 features)
    → ScoreDetect Head
        ├── Detection: [x1, y1, x2, y2, conf, class_id]
        └── Score: severity ∈ [0, 1] (×10 → [0, 10])
    → MaskGenerator (optional)
        └── Binary mask per detection (from P3/P4/P5 features)
```

## Supported Models

| Scale | Model | Params | GFLOPs | mAP@50 | Score MAE |
|-------|-------|--------|--------|--------|-----------|
| N | yolo26n-score | 2.57M | 5.3 | 51.3% | 1.317 |
| S | yolo26s-score | 10.19M | 20.8 | 57.3% | 1.306 |
| M (rec.) | yolo26m-score | 22.19M | 68.5 | 60.8% | 1.316 |
| L | yolo26l-score | 26.59M | 86.8 | 62.6% | 1.297 |
| X | yolo26x-score | 56.08M | 194.8 | 62.3% | 1.224 |

## Key Training Configuration

- **Score Loss**: Gaussian NLL (λ=0.05, σ=0.10)
- **MixUp**: Forced OFF (destroys score semantics)
- **Optimizer**: SGD (or configurable)
- **Epochs**: 105 (recommended)

## MaskGenerator Versions

| Version | Channel Selection | Weighting | Upsampling | Edge Alignment |
|---------|------------------|-----------|------------|----------------|
| V1 | variance | L2 | bilinear | — |
| V2 (default) | variance | L2 | Canny-guided | — |
| V3 (configurable) | variance/bimodal | equal/contrast | Canny/Sobel | gradient_snap |

## Export & Deployment

SevSeg uses its own export pipeline (not the generic model_conversion):

```bash
# Via GUI: "Model Conversion" tab in app.py
# Supports: ONNX, TensorRT (FP16/FP32)
```

**ONNX output nodes**:
- `det_output` — shape `[B, K, 7]`: `[x1, y1, x2, y2, conf, cls_id, severity_01]`
- `feat_p3/p4/p5` (optional) — PCA-compressed feature maps for MaskGenerator

**Important**: Raw severity output is in [0, 1]. Multiply by 10 to get [0, 10] scale.

The exported model is packaged as `.dlhub` via DLHubPackager with a SevSeg-specific `deploy_config.json`.

## Quick Start

```bash
python app.py
```

## Data Format

YOLO format with 6-column labels (extra column for severity score):

```
# label file: image_001.txt
# class_id  cx  cy  w  h  severity_score
0  0.5  0.3  0.1  0.2  7.5
1  0.8  0.6  0.05  0.1  3.2
```

## Dependencies

```bash
pip install torch torchvision ultralytics opencv-contrib-python gradio
```

## License

This module contains modified [Ultralytics](https://github.com/ultralytics/ultralytics) source code in the `ultralytics/` subdirectory, licensed under **AGPL-3.0**. The SevSeg-specific extensions (`sevseg_yolo/`, `app.py`, etc.) follow the project's Apache-2.0 license, but using them together with the bundled Ultralytics code triggers AGPL-3.0 obligations for the combined work.
