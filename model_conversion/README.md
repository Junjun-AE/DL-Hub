# Model Conversion Pipeline

8-stage industrial-grade model conversion tool. Converts PyTorch models to deployment-ready formats (ONNX, TensorRT, OpenVINO).

## Pipeline Stages

```
Stage 1: Import     → Load .pth/.pt (timm / Ultralytics / MMSeg)
Stage 2: Analyze    → Compatibility check, performance analysis
Stage 3: Optimize   → Conv+BN fusion, redundant layer removal
Stage 4: Export     → ONNX export (dynamic axes, simplification)
Stage 5: Convert    → TensorRT / OpenVINO / ONNX Runtime quantization
Stage 6: Validate   → Accuracy & performance validation
Stage 8: Config     → Generate model_config.yaml
Stage 9: Package    → Create .dlhub deployment package
```

## Supported Tasks & Frameworks

| Task | Framework | Models |
|------|-----------|--------|
| Classification (cls) | timm | EfficientNet, MobileNetV3, ResNet, ViT |
| Detection (det) | Ultralytics | YOLOv5, v8, v11, YOLO26 |
| Segmentation (seg) | MMSegmentation | SegFormer B0–B5 |

## Precision Modes

| Mode | Description |
|------|-------------|
| FP32 | Full precision, no quantization |
| FP16 | Half precision, simple and effective |
| INT8 | Post-training static quantization (requires calibration data) |
| MIXED | Sensitive layers keep FP16, others INT8 |

## Usage

```bash
# Generate config template
python main.py init -t det

# Run full pipeline
python main.py run -c config_det.yaml

# Analyze only
python main.py analyze -m model.pth -t cls

# Direct CLI export
python main.py pipeline -m model.pth -t det -o ./output --target tensorrt --precision fp16
```

## Output: .dlhub Package

The pipeline produces a `.dlhub` file (ZIP) containing the converted model, `deploy_config.json`, and `model_config.yaml`. See the project root's deployment interface specification for full format details.
