#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置模板生成器与加载器

根据任务类型 (cls/det/seg) 生成 YAML 配置模板，
支持通过配置文件运行完整的模型转换流程。

使用方法:
---------
# 生成配置模板
python config_templates.py --generate cls  # 生成分类任务模板
python config_templates.py --generate det  # 生成检测任务模板
python config_templates.py --generate seg  # 生成分割任务模板
python config_templates.py --generate all  # 生成所有模板

# 使用配置文件运行
python main.py run -c config_det.yaml

Author: Model Converter Team
Version: 1.1.0 (修复 opset 版本说明和兼容性警告，统一模板格式)
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from unified_logger import Logger, console, Timer


# YAML 支持
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# ============================================================================
# 配置模板定义 - 通用头部
# ============================================================================

# 通用的 Stage 1-3 模板部分
_STAGE_1_2_3_TEMPLATE = '''# ============================================================================
# Stage 1: 模型导入 (Model Import)
# ============================================================================
stage1_import:
  # [必填] 模型文件路径
  # 支持格式: .pth, .pt, .ckpt, .bin
  model_path: "{model_path}"
  
  # [必填] 任务类型
  # 可选值: cls (分类), det (检测), seg (分割)
  task_type: "{task_type}"
  
  # [可选] 输入形状 (B, C, H, W)
  # 如果不设置，将自动推断
  input_shape: null
  
  # [可选] 推理设备
  # 可选值: "cuda", "cuda:0", "cuda:1", "cpu"
  # 默认: 自动选择 (优先 GPU)
  device: null
  
  # [可选] 类别数量
  # 如果模型无法自动推断，需要手动指定
  num_classes: null

# ============================================================================
# Stage 2: 模型分析 (Model Analysis)
# ============================================================================
stage2_analyze:
  # [可选] 目标后端列表
  # 用于兼容性检测，检查模型是否支持目标部署平台
  # 可选值: onnxruntime, tensorrt, openvino
  target_backends:
    - "onnxruntime"
    - "tensorrt"
    - "openvino"
  
  # [可选] 生成分析报告
  # 可选值: null (不生成), "console", "json", "html"
  report_format: "console"
  
  # [可选] 报告输出路径
  # 仅当 report_format 不为 null 时有效
  report_output: null

# ============================================================================
# Stage 3: 图优化 (Graph Optimization)
# ============================================================================
stage3_optimize:
  # [可选] 是否启用优化
  # 强烈建议开启，可显著提升推理性能
  enabled: true
  
  # [可选] Conv + BatchNorm 融合
  # 将相邻的卷积层和 BatchNorm 层融合为单个卷积层
  # 可减少约 10-20% 的推理时间
  # 推荐: true
  enable_fusion: true
  
  # [可选] 冗余层消除
  # 移除 Dropout、恒等映射等推理时无用的层
  # 推荐: true
  enable_elimination: true
  
  # [可选] 输出验证
  # 优化后验证输出是否与原模型一致
  # 推荐: true (生产环境)
  verify_output: true
'''

# 通用的 Stage 8 和 Global 模板部分
_STAGE_8_GLOBAL_TEMPLATE = '''
# ============================================================================
# Stage 8: 配置文件生成 (Config Generation)
# ============================================================================
stage8_config:
  # [可选] 是否生成部署配置
  # 生成 YAML 格式的推理配置文件
  enabled: true
  
  # [可选] 配置文件输出路径
  # 如果不设置，自动生成为 {model_name}_deploy.yaml
  output_path: null

# ============================================================================
# 全局配置
# ============================================================================
global:
  # [必填] 输出目录
  # 所有生成的文件将保存到此目录
  output_dir: "./output"
  
  # [可选] 详细日志
  # 输出更多调试信息
  verbose: false
  
  # [可选] 跳过的阶段
  # 可以跳过某些阶段，例如已有 ONNX 可跳过 Stage 1-4
  # 可选值: ["stage3"], ["stage2", "stage3"], 等
  skip_stages: []
'''


# ============================================================================
# 分类任务 (Classification) 配置模板
# ============================================================================
TEMPLATE_CLS = '''# ============================================================================
# 🎯 模型转换配置文件 - 图像分类任务 (Classification)
# ============================================================================
# 
# 使用方法:
#   python main.py run -c config_cls.yaml
#
# 提示:
#   - 带 [必填] 的参数必须设置
#   - 带 [可选] 的参数可以删除或注释掉，使用默认值
#   - 带 [条件] 的参数只在特定情况下需要
#
# ============================================================================

# ============================================================================
# Stage 1: 模型导入 (Model Import)
# ============================================================================
stage1_import:
  # [必填] 模型文件路径
  # 支持格式: .pth, .pt, .ckpt, .bin
  # 支持框架: timm, torchvision, 自定义 PyTorch 模型
  model_path: "./models/resnet50.pth"
  
  # [必填] 任务类型
  # 可选值: cls (分类), det (检测), seg (分割)
  task_type: "cls"
  
  # [可选] 输入形状 (B, C, H, W)
  # 分类任务推荐: [1, 3, 224, 224] 或 [1, 3, 384, 384]
  # 如果不设置，将自动推断
  input_shape: null
  
  # [可选] 推理设备
  # 可选值: "cuda", "cuda:0", "cuda:1", "cpu"
  # 默认: 自动选择 (优先 GPU)
  device: null
  
  # [可选] 类别数量
  # 如果模型无法自动推断，需要手动指定
  # ImageNet: 1000, CIFAR-10: 10, CIFAR-100: 100
  num_classes: null

# ============================================================================
# Stage 2: 模型分析 (Model Analysis)
# ============================================================================
stage2_analyze:
  # [可选] 目标后端列表
  # 用于兼容性检测，检查模型是否支持目标部署平台
  # 可选值: onnxruntime, tensorrt, openvino
  target_backends:
    - "onnxruntime"
    - "tensorrt"
    - "openvino"
  
  # [可选] 生成分析报告
  # 可选值: null (不生成), "console", "json", "html"
  report_format: "console"
  
  # [可选] 报告输出路径
  # 仅当 report_format 不为 null 时有效
  report_output: null

# ============================================================================
# Stage 3: 图优化 (Graph Optimization)
# ============================================================================
stage3_optimize:
  # [可选] 是否启用优化
  # 强烈建议开启，可显著提升推理性能
  enabled: true
  
  # [可选] Conv + BatchNorm 融合
  # 将相邻的卷积层和 BatchNorm 层融合为单个卷积层
  # 可减少约 10-20% 的推理时间
  # 推荐: true
  enable_fusion: true
  
  # [可选] 冗余层消除
  # 移除 Dropout、恒等映射等推理时无用的层
  # 推荐: true
  enable_elimination: true
  
  # [可选] 输出验证
  # 优化后验证输出是否与原模型一致
  # 推荐: true (生产环境)
  verify_output: true

# ============================================================================
# Stage 4: ONNX 导出 (ONNX Export)
# ============================================================================
stage4_export:
  # [可选] ONNX 输出路径
  # 如果不设置，自动生成为 {model_name}.onnx
  output_path: null
  
  # [重要] ONNX opset 版本
  # ============================================================
  # ⚠️ 兼容性警告:
  #   - EfficientNet、MobileNetV3 等使用深度可分离卷积的模型
  #     在 opset >= 18 时可能导致 TensorRT 解析失败
  #   - 建议这类模型使用 opset 17 或更低版本
  # ============================================================
  # 推荐值:
  #   - opset 17: 推荐！兼容性最佳 (TensorRT 8.5+)
  #   - opset 13: 广泛兼容 (TensorRT 8.0+)
  #   - opset 11: 最大兼容性 (TensorRT 7.0+)
  #   - null: 自动选择 (默认使用 17)
  # 
  # 如果遇到 TensorRT "Kernel weight dimension" 错误:
  #   1. 尝试降低 opset 版本到 13 或 11
  #   2. 或设置 dynamic_batch: false
  opset_version: 17
  
  # [可选] 启用 ONNX 简化
  # 使用 onnxsim 简化模型图结构
  # 推荐: true
  enable_simplify: true
  
  # [可选] 启用导出后验证
  # 对比 PyTorch 和 ONNX Runtime 输出
  # 推荐: true
  enable_validation: true
  
  # [重要] 动态 batch
  # ============================================================
  # ⚠️ 兼容性警告:
  #   - EfficientNet、MobileNetV3、MobileNetV2 等模型
  #     使用深度可分离卷积 (Depthwise Separable Convolution)
  #   - 当 dynamic_batch=true 时，可能导致 TensorRT 解析失败
  #   - 错误信息: "Kernel weight dimension failed to broadcast"
  # ============================================================
  # 推荐设置:
  #   - ResNet、VGG、DenseNet 等标准卷积模型: true (可以启用)
  #   - EfficientNet、MobileNet 系列: false (推荐禁用)
  #   - 如需批量推理，请在 Stage 5 使用 TensorRT 动态形状配置
  dynamic_batch: false
  
  # [可选] 动态 H/W
  # 允许运行时改变输入分辨率
  # 分类任务推荐: false (通常固定分辨率)
  dynamic_hw: false

# ============================================================================
# Stage 5-6: 量化转换与验证 (Quantization & Validation)
# ============================================================================
stage5_convert:
  # [必填] 目标后端
  # 可选值:
  #   - tensorrt: NVIDIA GPU 高性能推理 (推荐)
  #   - openvino: Intel CPU/GPU/VPU 推理
  #   - ort: ONNX Runtime 跨平台推理
  target_backend: "tensorrt"
  
  # [必填] 精度模式
  # 可选值:
  #   - fp32: 全精度，无量化，精度最高但最慢
  #   - fp16: 半精度，推荐大多数场景，速度提升 2x
  #   - int8: 8位整数量化，速度提升 4x，需要校准数据
  #   - mixed: 混合精度，敏感层保持高精度
  # ============================================================
  # 分类任务推荐:
  #   - ResNet、VGG 等: fp16 或 int8 均可
  #   - EfficientNet、MobileNet: fp16 或 mixed (int8 精度损失较大)
  # ============================================================
  precision: "fp16"
  
  # [条件] 校准数据路径 (INT8/MIXED 量化必需)
  # 应包含代表性的训练/验证图像
  # 推荐: 300-1000 张图像
  calib_data_path: null
  
  # [条件] 校准数据格式
  # 可选值:
  #   - imagefolder: 文件夹结构 (images/*.jpg)
  #   - coco: COCO 格式 (images/ + annotations.json)
  calib_data_format: "imagefolder"
  
  # [条件] 校准样本数量
  # 推荐: 300 (平衡精度和速度)
  # 更多样本可能提升 INT8 精度，但会增加校准时间
  calib_num_samples: 300
  
  # [条件] 校准方法
  # 可选值:
  #   - entropy: 熵校准，精度更高 (推荐)
  #   - minmax: 最小最大值校准，速度更快
  #   - percentile: 百分位校准
  calib_method: "entropy"
  
  # [可选] 启用转换后验证
  # 对比 ONNX 和量化后模型输出
  # 推荐: true
  enable_validation: true
  
  # [可选] 使用真实数据验证
  # 使用校准数据进行验证，而非随机数据
  # 推荐: true (更准确的验证结果)
  use_real_validation_data: true
  
  # [可选] 验证样本数量
  validation_num_samples: 10
  
  # [可选] 启用性能测试
  # 测量推理延迟和吞吐量
  # 推荐: true (了解实际性能)
  enable_perf_test: false
  
  # ---------- TensorRT 专用配置 ----------
  tensorrt:
    # [可选] 工作空间大小 (GB)
    # 更大的空间可能找到更优的算法
    # 推荐: 4 (普通模型), 8+ (大模型)
    workspace_gb: 4
    
    # [可选] Timing Cache 路径
    # 缓存 TensorRT 优化结果，加速后续构建
    # 推荐: 设置一个固定路径
    timing_cache_path: null
    
    # [可选] DLA 核心 ID (Jetson 设备专用)
    # 可选值: 0, 1, null (不使用 DLA)
    dla_core: null
    
    # ---------- 动态形状配置 ----------
    # ============================================================
    # 💡 如需批量推理，推荐使用此配置而非 Stage 4 的 dynamic_batch
    #    这种方式对 EfficientNet/MobileNet 等模型更稳定
    # ============================================================
    
    # [可选] 动态 batch size
    # 设置后可在推理时使用不同的 batch size
    # 对于需要批量处理的场景，强烈推荐启用
    dynamic_batch:
      enabled: false      # 是否启用动态 batch
      min: 1              # 最小 batch size
      opt: 1              # 最优 batch size (TensorRT 优化目标)
      max: 1              # 最大 batch size
    
    # [可选] 动态输入尺寸 (H, W)
    # 设置后可在推理时使用不同的图像尺寸
    # 注意: 分类任务通常不需要动态尺寸，保持 enabled: false
    dynamic_shapes:
      enabled: false      # 是否启用动态尺寸
      min: [224, 224]     # 最小尺寸 [H, W]
      opt: [224, 224]     # 最优尺寸 [H, W] (TensorRT 优化目标)
      max: [224, 224]     # 最大尺寸 [H, W]
  
  # ---------- OpenVINO 专用配置 ----------
  openvino:
    # [可选] 推理设备
    # 可选值: "CPU", "GPU", "AUTO"
    device: "CPU"
    
    # [可选] 推理流数
    # 用于多核并行推理
    num_streams: null
    
    # ---------- 动态形状配置 (新增) ----------
    # [可选] 动态 batch size
    # 设置后可在推理时使用不同的 batch size
    dynamic_batch:
      enabled: false      # 是否启用动态 batch
      min: 1              # 最小 batch size
      max: 16             # 最大 batch size
  
  # ---------- ONNX Runtime 专用配置 ----------
  ort:
    # [可选] 跳过 ORT 预处理 (quant_pre_process)
    # 对于复杂模型 (如 Wide ResNet)，预处理可能非常慢
    # 如果预处理卡住，设为 true
    skip_preprocess: false
    
    # [可选] 预处理超时时间 (秒)
    # 超时后自动跳过预处理，使用原始模型
    # 推荐: 120 (2分钟)
    preprocess_timeout: 120
    
    # ---------- 动态形状配置 (新增) ----------
    # [可选] 动态 batch size
    # 设置后可在推理时使用不同的 batch size
    # 注意: ONNX Runtime 会自动保留 ONNX 模型的动态轴
    dynamic_batch:
      enabled: false      # 是否启用动态 batch
      min: 1              # 最小 batch size
      max: 16             # 最大 batch size

# ============================================================================
# Stage 8: 配置文件生成 (Config Generation)
# ============================================================================
stage8_config:
  # [可选] 是否生成部署配置
  # 生成 YAML 格式的推理配置文件
  enabled: true
  
  # [可选] 配置文件输出路径
  # 如果不设置，自动生成为 {model_name}_deploy.yaml
  output_path: null

# ============================================================================
# 全局配置
# ============================================================================
global:
  # [必填] 输出目录
  # 所有生成的文件将保存到此目录
  output_dir: "./output"
  
  # [可选] 详细日志
  # 输出更多调试信息
  verbose: false
  
  # [可选] 跳过的阶段
  # 可以跳过某些阶段，例如已有 ONNX 可跳过 Stage 1-4
  # 可选值: ["stage3"], ["stage2", "stage3"], 等
  skip_stages: []

# ============================================================================
# 常见问题排查 (FAQ)
# ============================================================================
#
# Q1: TensorRT 转换失败，提示 "Kernel weight dimension failed to broadcast"
# A1: 这通常是 EfficientNet/MobileNet + 动态 batch 导致的
#     解决方案:
#     1. 设置 stage4_export.dynamic_batch: false
#     2. 如需批量推理，启用 stage5_convert.tensorrt.dynamic_batch.enabled: true
#
# Q2: TensorRT 转换失败，提示 ONNX parse error
# A2: 可能是 opset 版本兼容性问题
#     解决方案: 尝试降低 stage4_export.opset_version 到 13 或 11
#
# Q3: INT8 量化后精度损失很大
# A3: 某些模型（如 EfficientNet）对 INT8 量化敏感
#     解决方案:
#     1. 使用 precision: "mixed" 混合精度
#     2. 或使用 precision: "fp16"
#     3. 增加校准样本数量
#
# Q4: 转换速度很慢
# A4: 可能原因:
#     1. 校准样本过多 - 减少 calib_num_samples
#     2. 工作空间过大 - 减小 tensorrt.workspace_gb
#     3. 使用 timing_cache_path 缓存优化结果
#
'''

# ============================================================================
# 检测任务 (Detection) 配置模板
# ============================================================================
TEMPLATE_DET = '''# ============================================================================
# 🎯 模型转换配置文件 - 目标检测任务 (Detection)
# ============================================================================
# 
# 使用方法:
#   python main.py run -c config_det.yaml
#
# 提示:
#   - 带 [必填] 的参数必须设置
#   - 带 [可选] 的参数可以删除或注释掉，使用默认值
#   - 带 [条件] 的参数只在特定情况下需要
#
# ============================================================================

# ============================================================================
# Stage 1: 模型导入 (Model Import)
# ============================================================================
stage1_import:
  # [必填] 模型文件路径
  # 支持格式: .pth, .pt (YOLO 原生格式)
  # 支持框架: ultralytics (YOLOv5/v8/v11)
  model_path: "./models/yolov8n.pt"
  
  # [必填] 任务类型
  # 可选值: cls (分类), det (检测), seg (分割)
  task_type: "det"
  
  # [可选] 输入形状 (B, C, H, W)
  # YOLO 推荐: [1, 3, 640, 640] 或 [1, 3, 1280, 1280]
  # 如果不设置，将自动推断
  input_shape: null
  
  # [可选] 推理设备
  # 可选值: "cuda", "cuda:0", "cuda:1", "cpu"
  # 默认: 自动选择 (优先 GPU)
  device: null
  
  # [可选] 类别数量
  # 如果模型无法自动推断，需要手动指定
  # COCO: 80, VOC: 20
  num_classes: null

# ============================================================================
# Stage 2: 模型分析 (Model Analysis)
# ============================================================================
stage2_analyze:
  # [可选] 目标后端列表
  # 用于兼容性检测，检查模型是否支持目标部署平台
  # 可选值: onnxruntime, tensorrt, openvino
  target_backends:
    - "onnxruntime"
    - "tensorrt"
    - "openvino"
  
  # [可选] 生成分析报告
  # 可选值: null (不生成), "console", "json", "html"
  report_format: "console"
  
  # [可选] 报告输出路径
  # 仅当 report_format 不为 null 时有效
  report_output: null

# ============================================================================
# Stage 3: 图优化 (Graph Optimization)
# ============================================================================
stage3_optimize:
  # [可选] 是否启用优化
  # 强烈建议开启，可显著提升推理性能
  enabled: true
  
  # [可选] Conv + BatchNorm 融合
  # 将相邻的卷积层和 BatchNorm 层融合为单个卷积层
  # 可减少约 10-20% 的推理时间
  # 推荐: true
  enable_fusion: true
  
  # [可选] 冗余层消除
  # 移除 Dropout、恒等映射等推理时无用的层
  # 推荐: true
  enable_elimination: true
  
  # [可选] 输出验证
  # 优化后验证输出是否与原模型一致
  # 推荐: true (生产环境)
  verify_output: true

# ============================================================================
# Stage 4: ONNX 导出 (ONNX Export)
# ============================================================================
stage4_export:
  # [可选] ONNX 输出路径
  # 如果不设置，自动生成为 {model_name}.onnx
  output_path: null
  
  # [重要] ONNX opset 版本
  # ============================================================
  # YOLO 模型推荐:
  #   - opset 17: 推荐！与 ultralytics 官方一致
  #   - opset 13: 广泛兼容
  #   - opset 11: 最大兼容性
  # ============================================================
  opset_version: 17
  
  # [可选] 启用 ONNX 简化
  # 使用 onnxsim 简化模型图结构
  # 推荐: true
  enable_simplify: true
  
  # [可选] 启用导出后验证
  # 对比 PyTorch 和 ONNX Runtime 输出
  # 推荐: true
  enable_validation: true
  
  # [可选] 动态 batch
  # 允许运行时改变 batch size
  # YOLO 模型支持动态 batch，推荐: true
  dynamic_batch: true
  
  # [可选] 动态 H/W
  # 允许运行时改变输入分辨率
  # 检测任务通常需要支持不同分辨率输入
  # 推荐: true
  dynamic_hw: true

# ============================================================================
# Stage 5-6: 量化转换与验证 (Quantization & Validation)
# ============================================================================
stage5_convert:
  # [必填] 目标后端
  # 可选值:
  #   - tensorrt: NVIDIA GPU 高性能推理 (推荐)
  #   - openvino: Intel CPU/GPU/VPU 推理
  #   - ort: ONNX Runtime 跨平台推理
  target_backend: "tensorrt"
  
  # [必填] 精度模式
  # 可选值:
  #   - fp32: 全精度，无量化，精度最高但最慢
  #   - fp16: 半精度，推荐大多数场景，速度提升 2x
  #   - int8: 8位整数量化，速度提升 4x，需要校准数据
  #   - mixed: 混合精度，敏感层保持高精度
  # ============================================================
  # YOLO 推荐:
  #   - fp16: 速度快，精度损失极小 (推荐)
  #   - int8: 速度最快，可能有一定精度损失
  # ============================================================
  precision: "fp16"
  
  # [条件] 校准数据路径 (INT8/MIXED 量化必需)
  # 应包含代表性的训练/验证图像
  # 推荐: 300-1000 张图像
  calib_data_path: null
  
  # [条件] 校准数据格式
  # 可选值:
  #   - imagefolder: 文件夹结构 (images/*.jpg)
  #   - coco: COCO 格式 (images/ + annotations.json)
  calib_data_format: "imagefolder"
  
  # [条件] 校准样本数量
  # 推荐: 300 (平衡精度和速度)
  # 更多样本可能提升 INT8 精度，但会增加校准时间
  calib_num_samples: 300
  
  # [条件] 校准方法
  # 可选值:
  #   - entropy: 熵校准，精度更高 (推荐)
  #   - minmax: 最小最大值校准，速度更快
  #   - percentile: 百分位校准
  calib_method: "entropy"
  
  # [可选] 启用转换后验证
  # 对比 ONNX 和量化后模型输出
  # 推荐: true
  enable_validation: true
  
  # [可选] 使用真实数据验证
  # 使用校准数据进行验证，而非随机数据
  # 推荐: true (更准确的验证结果)
  use_real_validation_data: true
  
  # [可选] 验证样本数量
  validation_num_samples: 10
  
  # [可选] 启用性能测试
  # 测量推理延迟和吞吐量
  # 推荐: true (了解实际性能)
  enable_perf_test: false
  
  # ---------- TensorRT 专用配置 ----------
  tensorrt:
    # [可选] 工作空间大小 (GB)
    # 更大的空间可能找到更优的算法
    # 推荐: 4 (普通模型), 8+ (大模型)
    workspace_gb: 4
    
    # [可选] Timing Cache 路径
    # 缓存 TensorRT 优化结果，加速后续构建
    # 推荐: 设置一个固定路径
    timing_cache_path: null
    
    # [可选] DLA 核心 ID (Jetson 设备专用)
    # 可选值: 0, 1, null (不使用 DLA)
    dla_core: null
    
    # ---------- 动态形状配置 ----------
    # ============================================================
    # 💡 YOLO 模型支持动态 batch 和动态分辨率
    #    推荐启用以支持不同输入尺寸
    # ============================================================
    
    # [可选] 动态 batch size
    # 设置后可在推理时使用不同的 batch size
    # 对于需要批量处理的场景，强烈推荐启用
    dynamic_batch:
      enabled: true       # 是否启用动态 batch
      min: 1              # 最小 batch size
      opt: 4              # 最优 batch size (TensorRT 优化目标)
      max: 16             # 最大 batch size
    
    # [可选] 动态输入尺寸 (H, W)
    # 设置后可在推理时使用不同的图像尺寸
    # 检测任务推荐启用，支持多种输入分辨率
    dynamic_shapes:
      enabled: true       # 是否启用动态尺寸
      min: [320, 320]     # 最小尺寸 [H, W]
      opt: [640, 640]     # 最优尺寸 [H, W] (TensorRT 优化目标)
      max: [1280, 1280]   # 最大尺寸 [H, W]
  
  # ---------- OpenVINO 专用配置 ----------
  openvino:
    # [可选] 推理设备
    # 可选值: "CPU", "GPU", "AUTO"
    device: "CPU"
    
    # [可选] 推理流数
    # 用于多核并行推理
    num_streams: null
  
  # ---------- ONNX Runtime 专用配置 ----------
  ort:
    # [可选] 跳过 ORT 预处理 (quant_pre_process)
    # 对于复杂模型，预处理可能非常慢
    # 如果预处理卡住，设为 true
    skip_preprocess: false
    
    # [可选] 预处理超时时间 (秒)
    # 超时后自动跳过预处理，使用原始模型
    # 推荐: 120 (2分钟)
    preprocess_timeout: 120

# ============================================================================
# Stage 8: 配置文件生成 (Config Generation)
# ============================================================================
stage8_config:
  # [可选] 是否生成部署配置
  # 生成 YAML 格式的推理配置文件
  enabled: true
  
  # [可选] 配置文件输出路径
  # 如果不设置，自动生成为 {model_name}_deploy.yaml
  output_path: null

# ============================================================================
# 全局配置
# ============================================================================
global:
  # [必填] 输出目录
  # 所有生成的文件将保存到此目录
  output_dir: "./output"
  
  # [可选] 详细日志
  # 输出更多调试信息
  verbose: false
  
  # [可选] 跳过的阶段
  # 可以跳过某些阶段，例如已有 ONNX 可跳过 Stage 1-4
  # 可选值: ["stage3"], ["stage2", "stage3"], 等
  skip_stages: []

# ============================================================================
# 常见问题排查 (FAQ)
# ============================================================================
#
# Q1: YOLO 转换后检测结果异常
# A1: 可能原因:
#     1. 预处理不匹配 - 确保使用 letterbox 缩放
#     2. NMS 后处理未正确配置
#     3. 校准数据不足 - 增加 calib_num_samples
#
# Q2: TensorRT 转换失败
# A2: 可能原因:
#     1. opset 版本不兼容 - 尝试 opset 13 或 11
#     2. 模型包含不支持的算子 - 检查日志中的警告
#
# Q3: INT8 量化后精度损失很大
# A3: 解决方案:
#     1. 使用 precision: "mixed" 混合精度
#     2. 增加校准样本数量到 500+
#     3. 使用更多样化的校准数据
#
# Q4: 动态分辨率推理速度慢
# A4: TensorRT 会为 min/opt/max 分别优化
#     如果实际输入与 opt 差距大，性能会下降
#     建议将 opt 设为最常用的分辨率
#
'''

# ============================================================================
# 分割任务 (Segmentation) 配置模板 - SegFormer 专用
# ============================================================================
TEMPLATE_SEG = '''# ============================================================================
# 🎯 模型转换配置文件 - SegFormer 语义分割 (Semantic Segmentation)
# ============================================================================
# 
# 专为内部训练软件导出的 SegFormer 模型设计
# 
# 使用方法:
#   python main.py run -c config_seg.yaml
#
# 支持的 SegFormer 变体:
#   - segformer_b0: 3.7M 参数，轻量级
#   - segformer_b1: 13.7M 参数
#   - segformer_b2: 24.7M 参数 (推荐，平衡速度和精度)
#   - segformer_b3: 44.6M 参数
#   - segformer_b4: 61.4M 参数
#   - segformer_b5: 81.9M 参数，最高精度
#
# 预处理说明:
#   SegFormer 使用像素级 ImageNet 归一化:
#   - 输入格式: RGB, 0-255 范围
#   - 归一化: normalized = (pixel - mean) / std
#   - mean = [123.675, 116.28, 103.53]
#   - std = [58.395, 57.12, 57.375]
#
# ============================================================================

# ============================================================================
# Stage 1: 模型导入 (Model Import)
# ============================================================================
stage1_import:
  # [必填] 模型文件路径
  # 支持格式: .pth (由训练软件导出)
  # 框架: mmsegmentation (SegFormer)
  model_path: "./models/segformer_b2.pth"
  
  # [必填] 任务类型
  # 固定为 "seg"
  task_type: "seg"
  
  # [可选] 输入形状 (B, C, H, W)
  # ============================================================
  # SegFormer 常用输入尺寸:
  #   - [1, 3, 512, 512]: 推荐，适合大多数场景
  #   - [1, 3, 640, 640]: 更高精度
  #   - [1, 3, 768, 768]: 高精度场景
  #   - [1, 3, 1024, 1024]: 最高精度，但速度较慢
  # 
  # 如果不设置，将从模型元数据自动推断
  # ============================================================
  input_shape: [1, 3, 512, 512]
  
  # [可选] 推理设备
  # 可选值: "cuda", "cuda:0", "cuda:1", "cpu"
  # 默认: 自动选择 (优先 GPU)
  device: null
  
  # [可选] 类别数量
  # 通常由模型自动推断，无需手动指定
  # 只有在推断失败时才需要设置
  num_classes: null

# ============================================================================
# Stage 2: 模型分析 (Model Analysis)
# ============================================================================
stage2_analyze:
  # [可选] 目标后端列表
  # 用于兼容性检测，检查模型是否支持目标部署平台
  target_backends:
    - "onnxruntime"
    - "tensorrt"
    - "openvino"
  
  # [可选] 生成分析报告
  # 可选值: null (不生成), "console", "json", "html"
  report_format: "console"
  
  # [可选] 报告输出路径
  report_output: null

# ============================================================================
# Stage 3: 图优化 (Graph Optimization)
# ============================================================================
stage3_optimize:
  # [可选] 是否启用优化
  # SegFormer 模型强烈建议开启
  enabled: true
  
  # [可选] Conv + BatchNorm 融合
  # SegFormer 使用 LayerNorm，此选项对 Transformer 块无影响
  # 但仍建议开启，用于 decode_head 中的卷积层
  enable_fusion: true
  
  # [可选] 冗余层消除
  # 移除 Dropout 等推理时无用的层
  # 推荐: true
  enable_elimination: true
  
  # [可选] 输出验证
  # 优化后验证输出是否与原模型一致
  verify_output: true

# ============================================================================
# Stage 4: ONNX 导出 (ONNX Export)
# ============================================================================
stage4_export:
  # [可选] ONNX 输出路径
  # 如果不设置，自动生成为 {model_name}.onnx
  output_path: null
  
  # [重要] ONNX opset 版本
  # ============================================================
  # SegFormer 推荐:
  #   - opset 17: 推荐！最新功能，TensorRT 8.6+ 支持
  #   - opset 14: 广泛兼容 (包含 GeLU 支持)
  #   - opset 13: 最大兼容性
  # ============================================================
  opset_version: 17
  
  # [可选] 启用 ONNX 简化
  # 使用 onnxsim 简化模型图结构
  # 对 Transformer 模型效果明显
  enable_simplify: true
  
  # [可选] 启用导出后验证
  # 对比 PyTorch 和 ONNX Runtime 输出
  enable_validation: true
  
  # [可选] 动态 batch
  # 允许运行时改变 batch size
  # SegFormer 支持动态 batch，推荐: true
  dynamic_batch: true
  
  # [可选] 动态 H/W
  # 允许运行时改变输入分辨率
  # ============================================================
  # ⚠️ SegFormer 注意事项:
  #   动态 H/W 可能影响位置编码精度
  #   如果目标分辨率固定，建议设为 false
  #   如果需要支持多分辨率，设为 true
  # ============================================================
  dynamic_hw: false

# ============================================================================
# Stage 5-6: 量化转换与验证 (Quantization & Validation)
# ============================================================================
stage5_convert:
  # [必填] 目标后端
  # 可选值:
  #   - tensorrt: NVIDIA GPU 高性能推理 (推荐)
  #   - openvino: Intel CPU/GPU/VPU 推理
  #   - ort: ONNX Runtime 跨平台推理
  target_backend: "tensorrt"
  
  # [必填] 精度模式
  # ============================================================
  # SegFormer 推荐:
  #   - fp16: 强烈推荐！语义分割对量化敏感
  #   - fp32: 最高精度，速度较慢
  #   - int8: 可能导致分割边缘模糊，谨慎使用
  #   - mixed: Attention 层保持 FP16，其他 INT8
  # ============================================================
  precision: "fp16"
  
  # [条件] 校准数据路径 (INT8/MIXED 量化必需)
  # 应包含代表性的训练/验证图像
  # SegFormer 推荐: 500+ 张图像以保证精度
  calib_data_path: null
  
  # [条件] 校准数据格式
  # 可选值:
  #   - imagefolder: 文件夹结构 (images/*.jpg)
  #   - coco: COCO 格式 (images/ + annotations.json)
  calib_data_format: "imagefolder"
  
  # [条件] 校准样本数量
  # SegFormer 推荐: 500 (分割任务需要更多样本)
  calib_num_samples: 500
  
  # [条件] 校准方法
  # 可选值:
  #   - entropy: 熵校准，精度更高 (推荐)
  #   - minmax: 最小最大值校准，速度更快
  calib_method: "entropy"
  
  # [可选] 启用转换后验证
  enable_validation: true
  
  # [可选] 使用真实数据验证
  use_real_validation_data: true
  
  # [可选] 验证样本数量
  validation_num_samples: 10
  
  # [可选] 启用性能测试
  enable_perf_test: false
  
  # ---------- TensorRT 专用配置 ----------
  tensorrt:
    # [可选] 工作空间大小 (GB)
    # SegFormer Transformer 架构需要较大空间
    # 推荐: 8 (B0-B2), 12 (B3-B5)
    workspace_gb: 8
    
    # [可选] Timing Cache 路径
    timing_cache_path: null
    
    # [可选] DLA 核心 ID (Jetson 设备专用)
    dla_core: null
    
    # ---------- 动态形状配置 ----------
    # [可选] 动态 batch size
    dynamic_batch:
      enabled: true
      min: 1
      opt: 1            # SegFormer 推荐单 batch 推理
      max: 8
    
    # [可选] 动态输入尺寸 (H, W)
    # ============================================================
    # ⚠️ SegFormer 动态形状注意事项:
    #   - 建议 opt 尺寸与训练尺寸一致
    #   - 过大的尺寸范围会影响优化效果
    # ============================================================
    dynamic_shapes:
      enabled: false      # 默认禁用，固定分辨率性能更好
      min: [384, 384]
      opt: [512, 512]     # 与训练尺寸一致
      max: [768, 768]
  
  # ---------- OpenVINO 专用配置 ----------
  openvino:
    device: "CPU"
    num_streams: null
  
  # ---------- ONNX Runtime 专用配置 ----------
  ort:
    skip_preprocess: false
    preprocess_timeout: 120

# ============================================================================
# Stage 8: 配置文件生成 (Config Generation)
# ============================================================================
stage8_config:
  enabled: true
  output_path: null

# ============================================================================
# 全局配置
# ============================================================================
global:
  output_dir: "./output"
  verbose: false
  skip_stages: []

# ============================================================================
# SegFormer 预处理配置参考 (用于推理时)
# ============================================================================
# 以下配置由模型元数据自动提供，仅供参考:
#
# preprocessing:
#   input_format: "RGB"           # 输入通道顺序
#   pixel_range: [0, 255]         # 像素值范围
#   normalize_method: "imagenet_pixel"  # 像素级归一化
#   normalize_mean: [123.675, 116.28, 103.53]
#   normalize_std: [58.395, 57.12, 57.375]
#   resize_mode: "bilinear"       # 缩放插值方式
#   keep_aspect_ratio: false      # 不保持宽高比
#
# output:
#   format: "logits"              # 输出未经 softmax
#   shape: [B, num_classes, H, W] # 输出形状
#   post_process: "argmax"        # 后处理: argmax(dim=1) 得到类别图

# ============================================================================
# 常见问题排查 (FAQ)
# ============================================================================
#
# Q1: 分割结果边缘模糊或不连续
# A1: 可能原因:
#     1. 使用了 INT8 量化 - 改用 precision: "fp16"
#     2. 动态尺寸影响 - 禁用 dynamic_hw
#     3. 输入分辨率太低 - 增加到 512x512 或更高
#
# Q2: ONNX 导出失败，提示不支持的算子
# A2: SegFormer 使用 GeLU 激活函数
#     解决方案: 使用 opset_version: 14 或更高
#
# Q3: TensorRT 转换失败
# A3: 可能原因:
#     1. 工作空间不足 - 增加 workspace_gb 到 12+
#     2. TensorRT 版本过低 - 升级到 8.6+
#     3. 动态形状范围过大 - 缩小 min/max 差距
#
# Q4: 推理速度慢
# A4: 优化建议:
#     1. 使用 FP16 精度而非 FP32
#     2. 禁用 dynamic_hw，使用固定分辨率
#     3. 减小输入分辨率 (如 512 -> 384)
#     4. 使用更轻量的变体 (如 B2 -> B0)
#
# Q5: 内存溢出 (OOM)
# A5: 解决方案:
#     1. 减小 batch size
#     2. 使用更小的输入分辨率
#     3. 使用轻量变体 (B0/B1)
#     4. 启用 FP16 减少显存占用
#
'''


# ============================================================================
# 配置数据类
# ============================================================================

@dataclass
class PipelineConfig:
    """Pipeline 配置"""
    # Stage 1
    model_path: str = ""
    task_type: str = "cls"
    input_shape: tuple = None
    device: str = None
    num_classes: int = None
    
    # Stage 2
    target_backends: list = field(default_factory=lambda: ["onnxruntime", "tensorrt", "openvino"])
    report_format: str = "console"
    report_output: str = None
    
    # Stage 3
    optimize_enabled: bool = True
    enable_fusion: bool = True
    enable_elimination: bool = True
    verify_output: bool = True
    
    # Stage 4
    onnx_output_path: str = None
    opset_version: int = 17  # 默认使用 17，兼容性最佳
    enable_simplify: bool = True
    enable_export_validation: bool = True
    dynamic_batch: bool = False  # 默认 False，避免 EfficientNet 等模型出问题
    dynamic_hw: bool = False
    dynamic_axes: dict = None
    
    # Stage 5-6
    target_backend: str = "tensorrt"
    precision: str = "fp16"
    calib_data_path: str = None
    calib_data_format: str = "imagefolder"
    calib_num_samples: int = 300
    calib_method: str = "entropy"
    enable_convert_validation: bool = True
    enable_perf_test: bool = False
    
    # TensorRT
    trt_workspace_gb: int = 4
    trt_timing_cache_path: str = None
    trt_dla_core: int = None
    trt_dynamic_batch_enabled: bool = False
    trt_min_batch: int = 1
    trt_opt_batch: int = 1
    trt_max_batch: int = 1
    trt_dynamic_shapes_enabled: bool = False
    trt_min_shapes: tuple = None
    trt_opt_shapes: tuple = None
    trt_max_shapes: tuple = None
    
    # OpenVINO
    ov_device: str = "CPU"
    ov_num_streams: int = None
    ov_dynamic_batch_enabled: bool = False
    ov_min_batch: int = 1
    ov_max_batch: int = 16
    
    # ONNX Runtime
    ort_dynamic_batch_enabled: bool = False
    ort_min_batch: int = 1
    ort_max_batch: int = 16
    
    # Stage 8
    config_enabled: bool = True
    config_output_path: str = None
    
    # Preprocessing
    input_scale: float = 1.0 / 255.0
    normalize_mean: tuple = (0.485, 0.456, 0.406)
    normalize_std: tuple = (0.229, 0.224, 0.225)
    use_letterbox: bool = False
    letterbox_color: tuple = (114, 114, 114)
    letterbox_stride: int = 32
    
    # Global
    output_dir: str = "./output"
    verbose: bool = False
    skip_stages: list = field(default_factory=list)


# ============================================================================
# 配置加载器
# ============================================================================

def load_config(config_path: str) -> PipelineConfig:
    """
    从 YAML 文件加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        PipelineConfig 对象
        
    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 配置格式错误
    """
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        raw_config = yaml.safe_load(f)
    
    if raw_config is None:
        raise ValueError(f"Empty config file: {config_path}")
    
    # 解析配置
    config = PipelineConfig()
    
    # Stage 1
    if 'stage1_import' in raw_config:
        s1 = raw_config['stage1_import']
        config.model_path = s1.get('model_path', '')
        config.task_type = s1.get('task_type', 'cls')
        if s1.get('input_shape'):
            config.input_shape = tuple(s1['input_shape'])
        config.device = s1.get('device')
        config.num_classes = s1.get('num_classes')
    
    # Stage 2
    if 'stage2_analyze' in raw_config:
        s2 = raw_config['stage2_analyze']
        config.target_backends = s2.get('target_backends', config.target_backends)
        config.report_format = s2.get('report_format', 'console')
        config.report_output = s2.get('report_output')
    
    # Stage 3
    if 'stage3_optimize' in raw_config:
        s3 = raw_config['stage3_optimize']
        config.optimize_enabled = s3.get('enabled', True)
        config.enable_fusion = s3.get('enable_fusion', True)
        config.enable_elimination = s3.get('enable_elimination', True)
        config.verify_output = s3.get('verify_output', True)
    
    # Stage 4
    if 'stage4_export' in raw_config:
        s4 = raw_config['stage4_export']
        config.onnx_output_path = s4.get('output_path')
        # opset_version: 显式设置优先，否则使用默认值 17
        config.opset_version = s4.get('opset_version') if s4.get('opset_version') is not None else 17
        config.enable_simplify = s4.get('enable_simplify', True)
        config.enable_export_validation = s4.get('enable_validation', True)
        config.dynamic_batch = s4.get('dynamic_batch', False)  # 默认 False
        config.dynamic_hw = s4.get('dynamic_hw', False)
        config.dynamic_axes = s4.get('dynamic_axes')
    
    # Stage 5-6
    if 'stage5_convert' in raw_config:
        s5 = raw_config['stage5_convert']
        config.target_backend = s5.get('target_backend', 'tensorrt')
        config.precision = s5.get('precision', 'fp16')
        config.calib_data_path = s5.get('calib_data_path')
        config.calib_data_format = s5.get('calib_data_format', 'imagefolder')
        config.calib_num_samples = s5.get('calib_num_samples', 300)
        config.calib_method = s5.get('calib_method', 'entropy')
        config.enable_convert_validation = s5.get('enable_validation', True)
        config.enable_perf_test = s5.get('enable_perf_test', False)
        
        # TensorRT
        if 'tensorrt' in s5:
            trt = s5['tensorrt']
            config.trt_workspace_gb = trt.get('workspace_gb', 4)
            config.trt_timing_cache_path = trt.get('timing_cache_path')
            config.trt_dla_core = trt.get('dla_core')
            
            # 动态 batch 配置
            if 'dynamic_batch' in trt:
                db = trt['dynamic_batch']
                config.trt_dynamic_batch_enabled = db.get('enabled', False)
                config.trt_min_batch = db.get('min', 1)
                config.trt_opt_batch = db.get('opt', 1)
                config.trt_max_batch = db.get('max', 1)
            
            # 动态尺寸配置
            if 'dynamic_shapes' in trt:
                ds = trt['dynamic_shapes']
                config.trt_dynamic_shapes_enabled = ds.get('enabled', False)
                if ds.get('min'):
                    config.trt_min_shapes = tuple(ds['min'])
                if ds.get('opt'):
                    config.trt_opt_shapes = tuple(ds['opt'])
                if ds.get('max'):
                    config.trt_max_shapes = tuple(ds['max'])
        
        # OpenVINO
        if 'openvino' in s5:
            ov = s5['openvino']
            config.ov_device = ov.get('device', 'CPU')
            config.ov_num_streams = ov.get('num_streams')
            
            # OpenVINO 动态batch配置
            if 'dynamic_batch' in ov:
                db = ov['dynamic_batch']
                config.ov_dynamic_batch_enabled = db.get('enabled', False)
                config.ov_min_batch = db.get('min', 1)
                config.ov_max_batch = db.get('max', 16)
        
        # ONNX Runtime
        if 'ort' in s5:
            ort = s5['ort']
            # ONNX Runtime 动态batch配置
            if 'dynamic_batch' in ort:
                db = ort['dynamic_batch']
                config.ort_dynamic_batch_enabled = db.get('enabled', False)
                config.ort_min_batch = db.get('min', 1)
                config.ort_max_batch = db.get('max', 16)
    
    # Stage 8
    if 'stage8_config' in raw_config:
        s8 = raw_config['stage8_config']
        config.config_enabled = s8.get('enabled', True)
        config.config_output_path = s8.get('output_path')
    
    # Preprocessing
    if 'preprocessing' in raw_config:
        pre = raw_config['preprocessing']
        config.input_scale = pre.get('input_scale', 1.0/255.0)
        if pre.get('normalize_mean'):
            config.normalize_mean = tuple(pre['normalize_mean'])
        if pre.get('normalize_std'):
            config.normalize_std = tuple(pre['normalize_std'])
        config.use_letterbox = pre.get('use_letterbox', False)
        if pre.get('letterbox_color'):
            config.letterbox_color = tuple(pre['letterbox_color'])
        config.letterbox_stride = pre.get('letterbox_stride', 32)
    
    # Global
    if 'global' in raw_config:
        g = raw_config['global']
        config.output_dir = g.get('output_dir', './output')
        config.verbose = g.get('verbose', False)
        config.skip_stages = g.get('skip_stages', [])
    
    return config


def validate_config(config: PipelineConfig) -> list:
    """
    验证配置有效性
    
    Args:
        config: PipelineConfig 对象
        
    Returns:
        错误消息列表，空列表表示验证通过
    """
    errors = []
    warnings = []
    
    # 检查必填项
    if not config.model_path:
        errors.append("[stage1_import.model_path] 模型路径必须设置")
    elif not os.path.exists(config.model_path):
        errors.append(f"[stage1_import.model_path] 模型文件不存在: {config.model_path}")
    
    if config.task_type not in ('cls', 'det', 'seg'):
        errors.append(f"[stage1_import.task_type] 无效的任务类型: {config.task_type}")
    
    if config.target_backend not in ('ort', 'tensorrt', 'openvino'):
        errors.append(f"[stage5_convert.target_backend] 无效的后端: {config.target_backend}")
    
    if config.precision not in ('fp32', 'fp16', 'int8', 'mixed'):
        errors.append(f"[stage5_convert.precision] 无效的精度: {config.precision}")
    
    # INT8 和 MIXED 需要校准数据
    if config.precision in ('int8', 'mixed'):
        if not config.calib_data_path:
            errors.append(f"[stage5_convert.calib_data_path] {config.precision.upper()} 量化需要校准数据路径")
        elif not os.path.exists(config.calib_data_path):
            errors.append(f"[stage5_convert.calib_data_path] 校准数据路径不存在: {config.calib_data_path}")
    
    # 兼容性警告
    if config.task_type == 'cls' and config.dynamic_batch and config.target_backend == 'tensorrt':
        warnings.append(
            "[兼容性警告] 分类任务启用 dynamic_batch 可能导致 EfficientNet/MobileNet 转换失败\n"
            "  建议: 设置 stage4_export.dynamic_batch: false"
        )
    
    if config.opset_version and config.opset_version >= 18 and config.target_backend == 'tensorrt':
        warnings.append(
            f"[兼容性警告] opset {config.opset_version} 可能导致某些模型 TensorRT 转换失败\n"
            "  建议: 设置 stage4_export.opset_version: 17"
        )
    
    # 打印警告
    for w in warnings:
        print(f"⚠️  {w}")
    
    return errors


def generate_template(task_type: str, output_path: str = None) -> str:
    """
    生成配置模板文件
    
    Args:
        task_type: 任务类型 (cls, det, seg)
        output_path: 输出路径，None 则使用默认名称
        
    Returns:
        生成的文件路径
    """
    templates = {
        'cls': TEMPLATE_CLS,
        'det': TEMPLATE_DET,
        'seg': TEMPLATE_SEG,
    }
    
    if task_type not in templates:
        raise ValueError(f"Unknown task type: {task_type}. Valid options: cls, det, seg")
    
    if output_path is None:
        output_path = f"config_{task_type}.yaml"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(templates[task_type])
    
    return output_path


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="配置模板生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成分类任务模板
  python config_templates.py --generate cls
  
  # 生成检测任务模板
  python config_templates.py --generate det
  
  # 生成所有模板
  python config_templates.py --generate all
  
  # 指定输出路径
  python config_templates.py --generate det -o my_config.yaml
  
  # 验证配置文件
  python config_templates.py --validate config_det.yaml
        """
    )
    
    parser.add_argument('--generate', '-g', type=str,
                        choices=['cls', 'det', 'seg', 'all'],
                        help='生成配置模板')
    parser.add_argument('--output', '-o', type=str,
                        help='输出路径')
    parser.add_argument('--validate', '-v', type=str,
                        help='验证配置文件')
    
    args = parser.parse_args()
    
    if args.generate:
        if args.generate == 'all':
            for task in ['cls', 'det', 'seg']:
                path = generate_template(task)
                print(f"✅ 生成: {path}")
        else:
            path = generate_template(args.generate, args.output)
            print(f"✅ 生成: {path}")
        return 0
    
    if args.validate:
        try:
            config = load_config(args.validate)
            errors = validate_config(config)
            
            if errors:
                print("❌ 配置验证失败:")
                for err in errors:
                    print(f"  - {err}")
                return 1
            else:
                print("✅ 配置验证通过")
                print(f"  模型: {config.model_path}")
                print(f"  任务: {config.task_type}")
                print(f"  后端: {config.target_backend}")
                print(f"  精度: {config.precision}")
                print(f"  opset: {config.opset_version}")
                print(f"  动态batch: {config.dynamic_batch}")
                return 0
                
        except Exception as e:
            print(f"❌ 配置加载失败: {e}")
            return 1
    
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())