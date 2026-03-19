# SegFormer Semantic Segmentation Module

Training, conversion, and inference tool for SegFormer semantic segmentation models.

## Supported Models

| Scale | Model | Backbone | Params |
|-------|-------|----------|--------|
| XS | segformer_b0 | MiT-B0 | 3.7M |
| S | segformer_b1 | MiT-B1 | 13.7M |
| M (recommended) | segformer_b2 | MiT-B2 | 24.7M |
| L | segformer_b3 | MiT-B3 | 44.6M |
| XL | segformer_b4 | MiT-B4 | 61.4M |
| XXL | segformer_b5 | MiT-B5 | 82.0M |

## Features

- **Data Conversion**: LabelMe polygon annotations → segmentation masks
- **Training**: MMSegmentation-based training with real-time mIoU monitoring
- **Model Conversion**: ONNX → TensorRT / OpenVINO / ONNX Runtime
- **Batch Inference**: Sliding-window inference for large images, color-coded visualization

## Preprocessing

SegFormer uses **pixel-level ImageNet normalization** (not [0,1]-based):
- Mean: `[123.675, 116.28, 103.53]` (applied to raw [0,255] pixel values)
- Std: `[58.395, 57.12, 57.375]`
- Letterbox padding color: `[0, 0, 0]` (not 114 like detection)

## Quick Start

```bash
python app.py
```

## Dependencies

```bash
pip install torch torchvision
pip install -U openmim
mim install mmengine 'mmcv>=2.0.0' mmsegmentation
pip install gradio
```
