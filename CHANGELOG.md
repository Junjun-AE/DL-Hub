# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-03-19

### Added
- **DL-Hub Platform**: Unified web management for all 6 tasks (React frontend + FastAPI backend)
- **Image Classification**: timm-based training with 15 models (EfficientNet/MobileNetV3/ResNet), CutMix/MixUp augmentation, AMP training
- **Object Detection**: YOLO training with 20 models (v5/v8/v11/v26), LabelMe→YOLO data conversion
- **Semantic Segmentation**: SegFormer B0–B5 training via MMSegmentation, batch inference with sliding window
- **Anomaly Detection**: PatchCore unsupervised detection, coreset sampling, Faiss-accelerated KNN, ONNX/TensorRT export
- **OCR**: PaddleOCR integration with detection + recognition, ONNX/TensorRT/OpenVINO export
- **Defect Scoring (SevSeg-YOLO)**: Custom YOLO26+ScoreHead for detection + severity scoring + zero-annotation mask generation (MaskGenerator V1/V2/V3)
- **Model Conversion Pipeline**: 8-stage conversion (import→analyze→optimize→ONNX→TRT/OV/ORT→validate→config→package)
- **Parameter Persistence**: Auto-save/restore UI parameters, training history, and logs via DLHubParams singleton
- **User Authentication**: Login system with password hashing, session management, avatar support
- **.dlhub Packaging**: Unified model packaging format with deploy_config.json for cls/det/seg/sevseg tasks
