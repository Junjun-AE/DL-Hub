# YOLO Object Detection Module

Training, conversion, and inference tool for YOLO object detection models.

## Supported Models

| Series | Models | Params | mAP@50 |
|--------|--------|--------|--------|
| YOLOv5 | n / s / m / l / x | 1.9M–86.7M | 45.7–68.9% |
| YOLOv8 | n / s / m / l / x | 3.2M–68.2M | 52.6–71.0% |
| YOLOv11 | n / s / m / l / x | 2.6M–56.9M | 54.4–72.0% |
| YOLO26 | n / s / m / l / x | 2.4M–55.5M | 55.4–72.6% |

## Features

- **Data Conversion**: LabelMe JSON → YOLO format with auto train/val split
- **Training**: Ultralytics engine with real-time loss/mAP monitoring, early stopping
- **Model Conversion**: ONNX → TensorRT / OpenVINO / ONNX Runtime (via model_conversion pipeline)
- **Batch Inference**: Multi-image inference with visualization

## Quick Start

```bash
# Standalone
python app.py

# Under DL-Hub
python app.py --task-dir /path/to/task --port 7861
```

## Data Format

Place LabelMe-annotated data in a folder:
```
dataset/
├── image_001.jpg
├── image_001.json   # LabelMe annotation
├── image_002.jpg
├── image_002.json
└── ...
```

The tool automatically converts to YOLO format during training.

## Dependencies

```bash
pip install torch torchvision ultralytics gradio
```
