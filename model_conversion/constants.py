#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一常量配置模块

集中管理所有默认值、阈值和魔法数字，便于维护和调整。

Author: Model Converter Team
Version: 1.0.0
"""

from dataclasses import dataclass
from typing import Tuple


# ============================================================================
# ONNX 导出相关常量
# ============================================================================

@dataclass(frozen=True)
class ONNXDefaults:
    """ONNX 导出默认配置"""
    
    # ONNX opset 版本
    # opset 17: 最新功能，TensorRT 8.6+ 支持
    # opset 13: 广泛兼容
    # opset 11: 最大兼容性
    OPSET_VERSION: int = 17
    
    # 输入输出名称
    DEFAULT_INPUT_NAME: str = "input"
    DEFAULT_OUTPUT_NAME: str = "output"


# ============================================================================
# 校准相关常量
# ============================================================================

@dataclass(frozen=True)
class CalibrationDefaults:
    """校准默认配置"""
    
    # 校准样本数量
    # 300: 平衡精度和速度的推荐值
    # 更多样本可能提升 INT8 精度，但会增加校准时间
    NUM_SAMPLES: int = 300
    
    # 校准批次大小
    # 8: 适合大多数 GPU 显存配置
    BATCH_SIZE: int = 8
    
    # Letterbox 填充颜色 (BGR)
    # (114, 114, 114): YOLO 系列标准灰色填充
    LETTERBOX_COLOR: Tuple[int, int, int] = (114, 114, 114)
    
    # Letterbox 对齐步长
    # 32: 适用于大多数检测模型
    LETTERBOX_STRIDE: int = 32


# ============================================================================
# 验证相关常量
# ============================================================================

@dataclass(frozen=True)
class ValidationThresholds:
    """验证阈值配置"""
    
    # FP32 最大差异阈值
    # 复杂模型的浮点误差可达 1e-3 量级
    FP32_MAX_DIFF: float = 1e-3
    
    # FP16 余弦相似度阈值
    # > 0.999 表示高度相似
    FP16_COSINE_THRESHOLD: float = 0.999
    
    # INT8 余弦相似度阈值
    # INT8 量化会有更多精度损失，0.99 是可接受的阈值
    INT8_COSINE_THRESHOLD: float = 0.99
    
    # 验证测试样本数
    NUM_TEST_SAMPLES: int = 10
    
    # 验证随机种子 (确保可复现)
    RANDOM_SEED: int = 42


# ============================================================================
# TensorRT 相关常量
# ============================================================================

@dataclass(frozen=True)
class TensorRTDefaults:
    """TensorRT 默认配置"""
    
    # 工作空间大小 (GB)
    # 4: 适合普通模型
    # 8+: 大模型可能需要更多
    WORKSPACE_GB: int = 4
    
    # 最低版本要求
    MIN_VERSION: str = "8.6.0"


# ============================================================================
# OpenVINO 相关常量
# ============================================================================

@dataclass(frozen=True)
class OpenVINODefaults:
    """OpenVINO 默认配置"""
    
    # 默认设备
    DEFAULT_DEVICE: str = "CPU"
    
    # 最低版本要求
    MIN_OV_VERSION: str = "2023.2.0"
    MIN_NNCF_VERSION: str = "2.5.0"


# ============================================================================
# 输入形状相关常量
# ============================================================================

@dataclass(frozen=True)
class InputShapeDefaults:
    """输入形状默认配置"""
    
    # 分类任务默认输入
    CLASSIFICATION_SHAPE: Tuple[int, int, int, int] = (1, 3, 224, 224)
    
    # 检测任务默认输入
    DETECTION_SHAPE: Tuple[int, int, int, int] = (1, 3, 640, 640)
    
    # 分割任务默认输入
    SEGMENTATION_SHAPE: Tuple[int, int, int, int] = (1, 3, 512, 512)
    
    # 动态 batch 范围 (min, opt, max)
    BATCH_RANGE: Tuple[int, int, int] = (1, 4, 16)
    
    # 动态 height/width 范围 (min, opt, max)
    HW_RANGE: Tuple[int, int, int] = (320, 640, 1280)


# ============================================================================
# 归一化相关常量
# ============================================================================

@dataclass(frozen=True)
class NormalizationDefaults:
    """归一化默认配置"""
    
    # 默认缩放因子 (0-255 -> 0-1)
    INPUT_SCALE: float = 1.0 / 255.0
    
    # ImageNet 归一化均值 (0-1 范围，用于分类模型)
    IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    
    # ImageNet 归一化标准差 (0-1 范围，用于分类模型)
    IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # 无归一化 (YOLO 等)
    NO_NORMALIZE_MEAN: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    NO_NORMALIZE_STD: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    
    # ============================================================================
    # SegFormer / MMSegmentation 像素级归一化 (0-255 范围)
    # ============================================================================
    # SegFormer 使用像素级归一化，直接在 0-255 范围内进行归一化
    # 公式: normalized = (pixel - mean) / std
    # 其中 pixel 在 0-255 范围内
    SEGFORMER_MEAN: Tuple[float, float, float] = (123.675, 116.28, 103.53)
    SEGFORMER_STD: Tuple[float, float, float] = (58.395, 57.12, 57.375)
    
    # SegFormer 不使用 input_scale，直接使用原始像素值
    SEGFORMER_INPUT_SCALE: float = 1.0


# ============================================================================
# 日志相关常量
# ============================================================================

@dataclass(frozen=True)
class LoggingDefaults:
    """日志默认配置"""
    
    # 日志格式
    FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    
    # 默认日志级别
    DEFAULT_LEVEL: str = "INFO"
    
    # 日志文件名
    DEFAULT_LOG_FILE: str = "model_converter.log"


# ============================================================================
# 性能测试相关常量
# ============================================================================

@dataclass(frozen=True)
class PerformanceDefaults:
    """性能测试默认配置"""
    
    # 预热迭代次数
    WARMUP_ITERATIONS: int = 10
    
    # 测试迭代次数
    TEST_ITERATIONS: int = 100
    
    # 超时时间 (秒)
    TIMEOUT_SECONDS: int = 300


# ============================================================================
# 全局常量实例 (方便导入使用)
# ============================================================================

ONNX = ONNXDefaults()
CALIBRATION = CalibrationDefaults()
VALIDATION = ValidationThresholds()
TENSORRT = TensorRTDefaults()
OPENVINO = OpenVINODefaults()
INPUT_SHAPE = InputShapeDefaults()
NORMALIZATION = NormalizationDefaults()
LOGGING = LoggingDefaults()
PERFORMANCE = PerformanceDefaults()


# ============================================================================
# 便捷函数
# ============================================================================

def get_default_input_shape(task_type: str) -> Tuple[int, int, int, int]:
    """
    根据任务类型获取默认输入形状
    
    Args:
        task_type: 任务类型 (cls/det/seg)
        
    Returns:
        默认输入形状 (B, C, H, W)
    """
    task_shapes = {
        "cls": INPUT_SHAPE.CLASSIFICATION_SHAPE,
        "det": INPUT_SHAPE.DETECTION_SHAPE,
        "seg": INPUT_SHAPE.SEGMENTATION_SHAPE,
    }
    return task_shapes.get(task_type, INPUT_SHAPE.DETECTION_SHAPE)


def get_validation_threshold(precision: str) -> float:
    """
    根据精度模式获取验证阈值
    
    Args:
        precision: 精度模式 (fp32/fp16/int8)
        
    Returns:
        余弦相似度阈值
    """
    thresholds = {
        "fp32": 0.9999,
        "fp16": VALIDATION.FP16_COSINE_THRESHOLD,
        "int8": VALIDATION.INT8_COSINE_THRESHOLD,
    }
    return thresholds.get(precision.lower(), VALIDATION.FP16_COSINE_THRESHOLD)
