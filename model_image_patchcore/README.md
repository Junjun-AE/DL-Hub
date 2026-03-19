# PatchCore Anomaly Detection Module

Industrial-grade unsupervised anomaly detection using PatchCore. Only **good (defect-free) samples** are required for training.

## How It Works

PatchCore extracts deep features from a pretrained backbone, builds a memory bank of "normal" feature patches via coreset sampling, and detects anomalies by measuring the distance between test image features and the nearest normal patches.

```
Training: Good images → Backbone → Feature extraction → Coreset sampling → Memory Bank + Faiss Index
Inference: Test image → Backbone → Features → KNN search → Anomaly score + Heatmap
```

## Features

- **Unsupervised**: No defect labels needed — train with only good samples
- **Backbone Options**: WideResNet-50, ResNet-18/50, EfficientNet (via timm)
- **PCA Compression**: Optional dimensionality reduction for faster inference
- **Faiss Acceleration**: GPU-accelerated nearest neighbor search
- **Threshold Tuning**: Automatic optimal threshold search (F1, percentile)
- **Export**: `.pkg` package with ONNX backbone + Memory Bank + Faiss index + TensorRT engine (optional)
- **Full GUI**: Config panel, training panel, evaluation panel, inference panel

## Export Format

PatchCore uses its own `.pkg` format (ZIP archive), **not** the `.dlhub` format:

```
patchcore_<timestamp>.pkg (ZIP)
├── config.json              # Full configuration
├── manifest.json            # File checksums (MD5)
├── backbone/
│   ├── backbone.onnx        # Backbone ONNX (opset 14)
│   └── backbone_fp16.engine # Backbone TensorRT (optional)
├── memory_bank/
│   ├── features.npy         # Memory Bank [N, D] float16
│   ├── faiss_index.bin      # Faiss index
│   └── pca_model.npz        # PCA: components + mean (optional)
├── normalization/
│   └── params.json          # Score normalization parameters
├── threshold/
│   └── config.json          # Threshold configuration
└── metadata/
    └── info.json            # Training metadata
```

## Quick Start

```bash
python app.py
```

## Data Format

```
dataset/
├── good/          # Good (normal) samples — required for training
│   ├── 001.png
│   ├── 002.png
│   └── ...
└── defect/        # Defect samples — optional, for evaluation only
    ├── 001.png
    └── ...
```

## Dependencies

```bash
pip install torch torchvision timm scikit-learn faiss-cpu gradio
# For GPU-accelerated Faiss: pip install faiss-gpu
```

## Loading a Trained Model

```python
from export.exporter import load_patchcore_model
from inference.predictor import PatchCorePredictor

# Method 1: Load components
components = load_patchcore_model("patchcore_20260318.pkg")

# Method 2: Direct prediction
predictor = PatchCorePredictor.from_package("model.pkg")
result = predictor.predict(image)
print(f"Score: {result.score}, Anomaly: {result.is_anomaly}")
# result.anomaly_map — pixel-level heatmap [H, W] (0–100)
# result.binary_mask — binary mask [H, W] (0/1)
```
