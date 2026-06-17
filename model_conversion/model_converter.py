#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 5: 模型量化转换与验证 (Model Quantization, Conversion & Validation)

本模块提供统一的模型量化转换接口，支持三种后端:
- ONNX Runtime: 通用部署，输出 .onnx
- TensorRT: NVIDIA GPU 高性能部署，输出 .engine
- OpenVINO: Intel 硬件优化部署，输出 .xml + .bin

支持的精度模式:
- FP32: 全精度，无量化
- FP16: 半精度，简单有效
- INT8: 训练后静态量化 (PTQ)，需要校准数据
- MIXED: 混合精度，敏感层保持高精度

作者: Industrial ML Team
版本: 1.0.0
"""

import os
import re
import json
import time
import shutil
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Union, Iterator, Any, Callable

import numpy as np
from unified_logger import Logger, console, Timer


# 条件导入 - 核心依赖
try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

# 图像处理
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# 进度条
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# ============================================================================
# 日志配置
# ============================================================================

def get_logger(name: str = "ModelConverter"):
    """获取 logger（兼容旧 API）"""
    return Logger.get(name)


logger = Logger.get("ModelConverter")


# ============================================================================
# 异常类
# ============================================================================

class ConversionError(Exception):
    """转换过程中的错误"""
    pass


class CalibrationError(Exception):
    """校准过程中的错误"""
    pass


class ValidationError(Exception):
    """验证过程中的错误"""
    pass


class DependencyError(Exception):
    """依赖缺失错误"""
    pass


# ============================================================================
# 枚举类
# ============================================================================

class TargetBackend(Enum):
    """目标后端"""
    ORT = "ort"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    
    @classmethod
    def from_string(cls, s: str) -> "TargetBackend":
        """从字符串创建枚举"""
        mapping = {
            "ort": cls.ORT,
            "onnxruntime": cls.ORT,
            "tensorrt": cls.TENSORRT,
            "trt": cls.TENSORRT,
            "openvino": cls.OPENVINO,
            "ov": cls.OPENVINO,
        }
        s_lower = s.lower().strip()
        if s_lower not in mapping:
            valid = list(mapping.keys())
            raise ValueError(f"Unknown backend '{s}'. Valid options: {valid}")
        return mapping[s_lower]


class PrecisionMode(Enum):
    """精度模式"""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    MIXED = "mixed"
    
    @classmethod
    def from_string(cls, s: str) -> "PrecisionMode":
        """从字符串创建枚举"""
        mapping = {
            "fp32": cls.FP32,
            "fp16": cls.FP16,
            "int8": cls.INT8,
            "int8_ptq": cls.INT8,
            "mixed": cls.MIXED,
        }
        s_lower = s.lower().strip()
        if s_lower not in mapping:
            valid = list(mapping.keys())
            raise ValueError(f"Unknown precision '{s}'. Valid options: {valid}")
        return mapping[s_lower]
    
    def requires_calibration(self) -> bool:
        """是否需要校准数据"""
        return self in (PrecisionMode.INT8, PrecisionMode.MIXED)


class CalibrationMethod(Enum):
    """校准方法"""
    ENTROPY = "entropy"
    MINMAX = "minmax"
    
    @classmethod
    def from_string(cls, s: str) -> "CalibrationMethod":
        """从字符串创建枚举"""
        mapping = {
            "entropy": cls.ENTROPY,
            "minmax": cls.MINMAX,
        }
        s_lower = s.lower().strip()
        if s_lower not in mapping:
            valid = list(mapping.keys())
            raise ValueError(f"Unknown calibration method '{s}'. Valid options: {valid}")
        return mapping[s_lower]


class CalibDataFormat(Enum):
    """校准数据格式"""
    IMAGEFOLDER = "imagefolder"
    COCO = "coco"
    
    @classmethod
    def from_string(cls, s: str) -> "CalibDataFormat":
        """从字符串创建枚举"""
        mapping = {
            "imagefolder": cls.IMAGEFOLDER,
            "folder": cls.IMAGEFOLDER,
            "coco": cls.COCO,
        }
        s_lower = s.lower().strip()
        if s_lower not in mapping:
            valid = list(mapping.keys())
            raise ValueError(f"Unknown data format '{s}'. Valid options: {valid}")
        return mapping[s_lower]


# ============================================================================
# 预处理配置工厂
# ============================================================================

class PreprocessingPreset(Enum):
    """
    预处理配置预设
    
    根据模型类型和框架自动选择正确的预处理配置。
    """
    # 分类模型预设
    IMAGENET_STANDARD = "imagenet_standard"      # 标准 ImageNet (timm, torchvision)
    IMAGENET_INCEPTION = "imagenet_inception"    # Inception 风格 (GoogLeNet, Inception)
    
    # 检测模型预设
    YOLO = "yolo"                                # YOLO 系列 (YOLOv5, YOLOv8 等)
    DETECTRON2 = "detectron2"                    # Detectron2 (Faster R-CNN 等) [保留兼容]
    
    # 分割模型预设
    SEGMENTATION_STANDARD = "segmentation_standard"  # 通用分割 (已弃用)
    SEGFORMER = "segformer"                          # SegFormer 像素级归一化 (推荐)
    
    # 自定义
    CUSTOM = "custom"


@dataclass
class PreprocessingConfig:
    """
    预处理配置
    
    定义了校准数据的预处理方式，包括归一化、缩放、填充等。
    不同的模型架构需要不同的预处理配置才能获得正确的量化结果。
    """
    # 像素值缩放
    input_scale: float = 1.0 / 255.0
    
    # 归一化 (在 input_scale 之后应用)
    normalize_mean: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    normalize_std: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    
    # Letterbox 填充 (主要用于检测模型)
    use_letterbox: bool = False
    letterbox_color: Tuple[int, int, int] = (114, 114, 114)
    letterbox_stride: int = 32
    
    # 颜色空间
    channel_order: str = "RGB"  # RGB 或 BGR
    
    # 预设名称 (用于日志和调试)
    preset_name: str = "custom"


class PreprocessingFactory:
    """
    预处理配置工厂
    
    根据任务类型、框架等信息自动生成正确的预处理配置。
    
    使用方法:
    ---------
    >>> config = PreprocessingFactory.from_task_and_framework(
    ...     task_type="det",
    ...     framework="ultralytics"
    ... )
    >>> print(config.input_scale)  # 0.00392156862745098 (1/255)
    
    >>> config = PreprocessingFactory.from_task_and_framework(
    ...     task_type="cls",
    ...     framework="timm"
    ... )
    >>> print(config.normalize_mean)  # (0.485, 0.456, 0.406)
    """
    
    # 预设配置字典
    PRESETS: Dict[PreprocessingPreset, PreprocessingConfig] = {
        PreprocessingPreset.IMAGENET_STANDARD: PreprocessingConfig(
            input_scale=1.0 / 255.0,
            normalize_mean=(0.485, 0.456, 0.406),
            normalize_std=(0.229, 0.224, 0.225),
            use_letterbox=False,
            channel_order="RGB",
            preset_name="imagenet_standard",
        ),
        PreprocessingPreset.IMAGENET_INCEPTION: PreprocessingConfig(
            input_scale=1.0 / 255.0,
            normalize_mean=(0.5, 0.5, 0.5),
            normalize_std=(0.5, 0.5, 0.5),
            use_letterbox=False,
            channel_order="RGB",
            preset_name="imagenet_inception",
        ),
        PreprocessingPreset.YOLO: PreprocessingConfig(
            input_scale=1.0 / 255.0,
            normalize_mean=(0.0, 0.0, 0.0),
            normalize_std=(1.0, 1.0, 1.0),
            use_letterbox=True,
            letterbox_color=(114, 114, 114),
            letterbox_stride=32,
            channel_order="RGB",
            preset_name="yolo",
        ),
        PreprocessingPreset.DETECTRON2: PreprocessingConfig(
            input_scale=1.0,  # Detectron2 使用 0-255 像素值
            normalize_mean=(103.530, 116.280, 123.675),  # BGR 均值
            normalize_std=(1.0, 1.0, 1.0),
            use_letterbox=False,
            channel_order="BGR",
            preset_name="detectron2",
        ),
        PreprocessingPreset.SEGMENTATION_STANDARD: PreprocessingConfig(
            input_scale=1.0 / 255.0,
            normalize_mean=(0.485, 0.456, 0.406),
            normalize_std=(0.229, 0.224, 0.225),
            use_letterbox=False,
            channel_order="RGB",
            preset_name="segmentation_standard",
        ),
        # ============================================================================
        # SegFormer 像素级归一化预设 (MMSegmentation)
        # ============================================================================
        # SegFormer 使用像素级归一化，直接在 0-255 范围内进行
        # 公式: normalized = (pixel - mean) / std
        # 其中 pixel 在 0-255 范围，mean/std 也是 0-255 范围的值
        # 
        # 与 IMAGENET_STANDARD 的区别:
        # - IMAGENET_STANDARD: 先 /255 缩放到 0-1，再用 0-1 范围的 mean/std 归一化
        # - SEGFORMER: 不缩放，直接用 0-255 范围的 mean/std 归一化
        # ============================================================================
        PreprocessingPreset.SEGFORMER: PreprocessingConfig(
            input_scale=1.0,  # 不缩放，保持 0-255 范围
            normalize_mean=(123.675, 116.28, 103.53),  # 像素级均值 (0-255)
            normalize_std=(58.395, 57.12, 57.375),     # 像素级标准差 (0-255)
            use_letterbox=False,
            channel_order="RGB",
            preset_name="segformer",
        ),
    }
    
    # 任务类型 + 框架 -> 预设映射
    TASK_FRAMEWORK_MAPPING: Dict[Tuple[str, str], PreprocessingPreset] = {
        # 分类任务
        ("cls", "timm"): PreprocessingPreset.IMAGENET_STANDARD,
        ("cls", "torchvision"): PreprocessingPreset.IMAGENET_STANDARD,
        ("cls", "unknown"): PreprocessingPreset.IMAGENET_STANDARD,
        
        # 检测任务
        ("det", "ultralytics"): PreprocessingPreset.YOLO,
        ("det", "yolov5"): PreprocessingPreset.YOLO,
        ("det", "detectron2"): PreprocessingPreset.DETECTRON2,
        ("det", "unknown"): PreprocessingPreset.YOLO,  # 默认使用 YOLO 预处理
        
        # 分割任务
        ("seg", "mmsegmentation"): PreprocessingPreset.SEGFORMER,  # SegFormer (推荐)
        ("seg", "segformer"): PreprocessingPreset.SEGFORMER,       # SegFormer 别名
        ("seg", "ultralytics"): PreprocessingPreset.YOLO,          # YOLO-Seg
        ("seg", "torchvision"): PreprocessingPreset.SEGMENTATION_STANDARD,
        ("seg", "unknown"): PreprocessingPreset.SEGFORMER,  # 默认使用 SegFormer 预处理
    }
    
    @classmethod
    def get_preset(cls, preset: PreprocessingPreset) -> PreprocessingConfig:
        """
        获取预设配置
        
        Args:
            preset: 预设枚举值
            
        Returns:
            预处理配置
        """
        if preset not in cls.PRESETS:
            raise ValueError(f"Unknown preset: {preset}")
        return cls.PRESETS[preset]
    
    @classmethod
    def from_task_and_framework(
        cls,
        task_type: str,
        framework: str = "unknown",
    ) -> PreprocessingConfig:
        """
        根据任务类型和框架自动选择预处理配置
        
        Args:
            task_type: 任务类型 (cls/det/seg)
            framework: 框架名称 (timm/ultralytics/mmsegmentation/segformer/torchvision/unknown)
            
        Returns:
            预处理配置
            
        Raises:
            ValueError: 无法确定预处理配置时
            
        Example:
            >>> config = PreprocessingFactory.from_task_and_framework("seg", "mmsegmentation")
            >>> print(config.preset_name)  # "segformer"
            >>> print(config.normalize_mean)  # (123.675, 116.28, 103.53)
        """
        task_type = task_type.lower().strip()
        framework = framework.lower().strip()
        
        # 查找映射
        key = (task_type, framework)
        if key in cls.TASK_FRAMEWORK_MAPPING:
            preset = cls.TASK_FRAMEWORK_MAPPING[key]
            config = cls.get_preset(preset)
            logger.info(f"自动选择预处理配置: {config.preset_name} (task={task_type}, framework={framework})")
            return config
        
        # 回退到 unknown 框架
        fallback_key = (task_type, "unknown")
        if fallback_key in cls.TASK_FRAMEWORK_MAPPING:
            preset = cls.TASK_FRAMEWORK_MAPPING[fallback_key]
            config = cls.get_preset(preset)
            logger.warning(
                f"未知框架 '{framework}'，使用默认预处理配置: {config.preset_name}"
            )
            return config
        
        # 无法确定配置
        raise ValueError(
            f"无法为任务类型 '{task_type}' 和框架 '{framework}' 确定预处理配置。\n"
            f"请手动指定 PreprocessingConfig 或使用支持的组合:\n"
            f"  - 分类 (cls): timm, torchvision\n"
            f"  - 检测 (det): ultralytics, yolov5, detectron2\n"
            f"  - 分割 (seg): mmsegmentation, segformer, ultralytics, torchvision"
        )
    
    @classmethod
    def custom(
        cls,
        input_scale: float = 1.0 / 255.0,
        normalize_mean: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        normalize_std: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        use_letterbox: bool = False,
        letterbox_color: Tuple[int, int, int] = (114, 114, 114),
        letterbox_stride: int = 32,
        channel_order: str = "RGB",
    ) -> PreprocessingConfig:
        """
        创建自定义预处理配置
        
        Args:
            input_scale: 像素值缩放因子
            normalize_mean: 归一化均值 (在 input_scale 之后应用)
            normalize_std: 归一化标准差
            use_letterbox: 是否使用 letterbox 填充
            letterbox_color: letterbox 填充颜色
            letterbox_stride: letterbox 对齐步长
            channel_order: 颜色通道顺序 (RGB 或 BGR)
            
        Returns:
            自定义预处理配置
        """
        return PreprocessingConfig(
            input_scale=input_scale,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            use_letterbox=use_letterbox,
            letterbox_color=letterbox_color,
            letterbox_stride=letterbox_stride,
            channel_order=channel_order,
            preset_name="custom",
        )


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class ConversionConfig:
    """
    转换配置
    
    统一的配置类，包含所有后端和精度模式的配置选项。
    """
    # 必需参数
    target_backend: TargetBackend
    precision_mode: PrecisionMode = PrecisionMode.FP16
    
    # 校准配置 (INT8/MIXED 需要)
    calib_data_path: Optional[str] = None
    calib_data_format: CalibDataFormat = CalibDataFormat.IMAGEFOLDER
    calib_num_samples: int = 300
    calib_batch_size: int = 8
    calib_method: CalibrationMethod = CalibrationMethod.ENTROPY
    calib_cache_path: Optional[str] = None  # 校准缓存路径
    
    # 混合精度配置
    sensitive_layers: Optional[List[str]] = None  # 手动指定敏感层
    use_default_sensitive: bool = True  # 使用默认敏感层策略
    
    # 验证配置
    enable_validation: bool = True
    enable_perf_test: bool = False
    validation_samples: int = 10
    cosine_threshold_int8: float = 0.99
    cosine_threshold_fp16: float = 0.999
    
    # 输出配置
    save_intermediate_onnx: bool = False  # 是否保存中间 ONNX
    generate_report: bool = True  # 生成 JSON 报告
    
    # TensorRT 特定配置
    trt_workspace_gb: int = 4
    trt_timing_cache_path: Optional[str] = None
    trt_dla_core: Optional[int] = None  # DLA 核心 ID (Jetson)
    
    # TensorRT 动态形状配置
    trt_dynamic_batch_enabled: bool = False
    trt_min_batch: int = 1
    trt_opt_batch: int = 1
    trt_max_batch: int = 1
    
    trt_dynamic_shapes_enabled: bool = False
    trt_min_shapes: Optional[Tuple[int, int]] = None  # (H, W)
    trt_opt_shapes: Optional[Tuple[int, int]] = None  # (H, W)
    trt_max_shapes: Optional[Tuple[int, int]] = None  # (H, W)
    
    # OpenVINO 特定配置
    openvino_device: str = "CPU"
    openvino_num_streams: Optional[int] = None
    
    # OpenVINO 动态形状配置（新增）
    ov_dynamic_batch_enabled: bool = False
    ov_min_batch: int = 1
    ov_max_batch: int = 16
    
    # ONNX Runtime 动态形状配置（新增）
    ort_dynamic_batch_enabled: bool = False
    ort_min_batch: int = 1
    ort_max_batch: int = 16
    
    # 输入配置 (从 Stage 4 传入)
    input_shape: Optional[Tuple[int, ...]] = None  # (B, C, H, W)
    input_names: Optional[List[str]] = None
    dynamic_axes_spec: Optional[Dict] = None  # min/opt/max shapes
    
    # InputSpec (从 Stage 1 传入，包含完整的输入规格信息)
    input_spec: Optional[Any] = None  # InputSpec 对象，包含 normalize_mean/std 等
    
    # 预处理配置 - 推荐使用 preprocessing_config 对象
    # 以下单独字段保留向后兼容，但会被 preprocessing_config 覆盖
    preprocessing_config: Optional[PreprocessingConfig] = None  # 推荐方式
    letterbox_color: Tuple[int, int, int] = (114, 114, 114)
    letterbox_stride: int = 32
    normalize_mean: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    normalize_std: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    input_scale: float = 1.0 / 255.0  # 像素值缩放
    
    # 任务和框架信息 (用于自动选择预处理配置)
    task_type: Optional[str] = None  # cls/det/seg
    framework: Optional[str] = None  # timm/ultralytics/detectron2
    
    def __post_init__(self):
        """验证配置"""
        # 检查校准数据
        if self.precision_mode.requires_calibration():
            if not self.calib_data_path:
                raise ValueError(
                    f"Calibration data path is required for {self.precision_mode.value} mode"
                )
            if not os.path.exists(self.calib_data_path):
                raise ValueError(
                    f"Calibration data path does not exist: {self.calib_data_path}"
                )
        
        # ============================================================
        # 预处理配置优先级：
        # 1. 显式提供的 preprocessing_config (最高优先级)
        # 2. 从 input_spec 提取的配置 (Stage 1 传入)
        # 3. 根据 task_type + framework 自动推断
        # 4. 默认配置 (最低优先级)
        # ============================================================
        
        if self.preprocessing_config is not None:
            # 优先级 1: 使用显式提供的预处理配置
            self._apply_preprocessing_config(self.preprocessing_config)
            logger.info(f"📋 使用显式指定的预处理配置: {self.preprocessing_config.preset_name}")
            
        elif self.input_spec is not None:
            # 优先级 2: 从 InputSpec 提取预处理配置
            self._apply_input_spec()
            logger.info(f"📋 从 InputSpec 提取预处理配置")
            
        elif self.task_type:
            # 优先级 3: 根据 task_type + framework 自动推断
            try:
                self.preprocessing_config = PreprocessingFactory.from_task_and_framework(
                    task_type=self.task_type,
                    framework=self.framework or "unknown",
                )
                self._apply_preprocessing_config(self.preprocessing_config)
                logger.info(f"📋 自动选择预处理配置: {self.preprocessing_config.preset_name}")
            except ValueError as e:
                logger.warning(f"无法自动选择预处理配置: {e}")
                logger.warning(f"将使用默认预处理: scale=1/255, mean=(0,0,0), std=(1,1,1)")
        
        # 输出最终的预处理配置
        logger.info(f"   input_scale: {self.input_scale}")
        logger.info(f"   normalize_mean: {self.normalize_mean}")
        logger.info(f"   normalize_std: {self.normalize_std}")
        if self.preprocessing_config:
            logger.info(f"   use_letterbox: {self.preprocessing_config.use_letterbox}")
        
        # ============================================================
        # 预处理配置验证 - 检查配置是否与模型框架匹配
        # ============================================================
        self._validate_preprocessing_config()
    
    def _apply_preprocessing_config(self, config: PreprocessingConfig):
        """应用预处理配置到旧字段（向后兼容）"""
        self.preprocessing_config = config
        self.input_scale = config.input_scale
        self.normalize_mean = config.normalize_mean
        self.normalize_std = config.normalize_std
        self.letterbox_color = config.letterbox_color
        self.letterbox_stride = config.letterbox_stride
    
    def _apply_input_spec(self):
        """从 InputSpec 提取预处理配置"""
        input_spec = self.input_spec
        
        # 提取 normalize_mean 和 normalize_std
        if hasattr(input_spec, 'normalize_mean') and input_spec.normalize_mean:
            self.normalize_mean = tuple(input_spec.normalize_mean)
        if hasattr(input_spec, 'normalize_std') and input_spec.normalize_std:
            self.normalize_std = tuple(input_spec.normalize_std)
        
        # 根据任务类型决定是否使用 letterbox
        # CLS 任务不使用 letterbox，DET/SEG 使用
        use_letterbox = self.task_type in ('det', 'seg') if self.task_type else False
        
        # 创建 preprocessing_config
        self.preprocessing_config = PreprocessingConfig(
            input_scale=self.input_scale,
            normalize_mean=self.normalize_mean,
            normalize_std=self.normalize_std,
            use_letterbox=use_letterbox,
            letterbox_color=self.letterbox_color,
            letterbox_stride=self.letterbox_stride,
            channel_order=getattr(input_spec, 'channel_order', 'RGB'),
            preset_name="from_input_spec",
        )
    
    def _validate_preprocessing_config(self):
        """
        验证预处理配置是否与模型框架匹配
        
        如果检测到配置不匹配，会发出警告但不会阻止执行。
        """
        if not self.task_type:
            return  # 没有任务类型信息，跳过验证
        
        try:
            # 获取预期的预处理配置
            expected = PreprocessingFactory.from_task_and_framework(
                task_type=self.task_type,
                framework=self.framework or "unknown",
            )
        except ValueError:
            return  # 无法获取预期配置，跳过验证
        
        warnings_list = []
        
        # 检查 normalize_mean
        if self.normalize_mean != expected.normalize_mean:
            warnings_list.append(
                f"normalize_mean 不匹配: 当前={self.normalize_mean}, "
                f"预期={expected.normalize_mean} (基于 {self.task_type}/{self.framework})"
            )
        
        # 检查 normalize_std
        if self.normalize_std != expected.normalize_std:
            warnings_list.append(
                f"normalize_std 不匹配: 当前={self.normalize_std}, "
                f"预期={expected.normalize_std} (基于 {self.task_type}/{self.framework})"
            )
        
        # 检查 use_letterbox
        if self.preprocessing_config:
            if self.preprocessing_config.use_letterbox != expected.use_letterbox:
                warnings_list.append(
                    f"use_letterbox 不匹配: 当前={self.preprocessing_config.use_letterbox}, "
                    f"预期={expected.use_letterbox} (基于 {self.task_type}/{self.framework})"
                )
        
        # 输出警告
        if warnings_list:
            logger.warning("⚠️ 预处理配置验证警告:")
            for w in warnings_list:
                logger.warning(f"   - {w}")
            logger.warning("   如果这是有意为之，可以忽略此警告。否则请检查配置是否正确。")
    
    def get_preprocessing_config(self) -> PreprocessingConfig:
        """
        获取预处理配置
        
        如果没有显式设置，则根据当前字段值创建一个配置对象。
        
        Returns:
            PreprocessingConfig 对象
        """
        if self.preprocessing_config is not None:
            return self.preprocessing_config
        
        # 根据任务类型决定是否使用 letterbox
        use_letterbox = self.task_type in ('det', 'seg') if self.task_type else True
        
        # 从旧字段创建配置
        return PreprocessingConfig(
            input_scale=self.input_scale,
            normalize_mean=self.normalize_mean,
            normalize_std=self.normalize_std,
            use_letterbox=use_letterbox,
            letterbox_color=self.letterbox_color,
            letterbox_stride=self.letterbox_stride,
            preset_name="legacy",
        )
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        result = {}
        for k, v in asdict(self).items():
            if isinstance(v, Enum):
                result[k] = v.value
            else:
                result[k] = v
        return result
    
    @classmethod
    def from_dict(cls, d: Dict) -> "ConversionConfig":
        """从字典创建"""
        # 转换枚举类型
        if "target_backend" in d and isinstance(d["target_backend"], str):
            d["target_backend"] = TargetBackend.from_string(d["target_backend"])
        if "precision_mode" in d and isinstance(d["precision_mode"], str):
            d["precision_mode"] = PrecisionMode.from_string(d["precision_mode"])
        if "calib_method" in d and isinstance(d["calib_method"], str):
            d["calib_method"] = CalibrationMethod.from_string(d["calib_method"])
        if "calib_data_format" in d and isinstance(d["calib_data_format"], str):
            d["calib_data_format"] = CalibDataFormat.from_string(d["calib_data_format"])
        return cls(**d)


@dataclass
class ValidationResult:
    """验证结果"""
    passed: bool
    cosine_sim: float
    max_diff: float
    mse: float
    original_size_mb: float
    output_size_mb: float
    compression_ratio: float
    latency_ms: Optional[float] = None
    throughput_fps: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """
        转换为字典
        
        确保所有 numpy 类型都转换为 Python 原生类型，以便 JSON 序列化。
        """
        def convert_value(v):
            """转换单个值为 JSON 可序列化类型"""
            if v is None:
                return None
            # numpy 类型转换
            if hasattr(v, 'item'):  # numpy scalar
                return v.item()
            if isinstance(v, (list, tuple)):
                return [convert_value(x) for x in v]
            if isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            # 确保 bool 类型正确
            if isinstance(v, (bool, np.bool_)):
                return bool(v)
            if isinstance(v, (int, np.integer)):
                return int(v)
            if isinstance(v, (float, np.floating)):
                return float(v)
            return v
        
        return {
            "passed": bool(self.passed),
            "cosine_sim": float(self.cosine_sim),
            "max_diff": float(self.max_diff),
            "mse": float(self.mse),
            "original_size_mb": float(self.original_size_mb),
            "output_size_mb": float(self.output_size_mb),
            "compression_ratio": float(self.compression_ratio),
            "latency_ms": float(self.latency_ms) if self.latency_ms is not None else None,
            "throughput_fps": float(self.throughput_fps) if self.throughput_fps is not None else None,
            "warnings": list(self.warnings),
        }


@dataclass
class ConversionResult:
    """转换结果"""
    success: bool
    target_backend: TargetBackend
    precision_mode: PrecisionMode
    output_files: List[str]
    validation: Optional[ValidationResult] = None
    build_time_seconds: float = 0.0
    message: str = ""
    stats: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """
        转换为字典
        
        确保所有值都是 JSON 可序列化的。
        """
        def convert_stats(stats: Dict) -> Dict:
            """递归转换 stats 字典中的 numpy 类型"""
            result = {}
            for k, v in stats.items():
                if v is None:
                    result[k] = None
                elif hasattr(v, 'item'):  # numpy scalar
                    result[k] = v.item()
                elif isinstance(v, (bool, np.bool_)):
                    result[k] = bool(v)
                elif isinstance(v, (int, np.integer)):
                    result[k] = int(v)
                elif isinstance(v, (float, np.floating)):
                    result[k] = float(v)
                elif isinstance(v, dict):
                    result[k] = convert_stats(v)
                elif isinstance(v, (list, tuple)):
                    result[k] = [
                        x.item() if hasattr(x, 'item') else x 
                        for x in v
                    ]
                else:
                    result[k] = v
            return result
        
        return {
            "success": bool(self.success),
            "target_backend": self.target_backend.value,
            "precision_mode": self.precision_mode.value,
            "output_files": list(self.output_files),
            "validation": self.validation.to_dict() if self.validation else None,
            "build_time_seconds": float(self.build_time_seconds),
            "message": str(self.message),
            "stats": convert_stats(self.stats),
        }
    
    def save_report(self, path: str):
        """保存 JSON 报告"""
        report = self.to_dict()
        report["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 确保目录存在
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Conversion report saved to: {path}")


# ============================================================================
# Letterbox 预处理
# ============================================================================

class LetterboxTransform:
    """
    Letterbox 变换 - YOLO 标准预处理
    
    保持长宽比的缩放 + 居中填充
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int],  # (H, W)
        color: Tuple[int, int, int] = (114, 114, 114),
        stride: int = 32,
        auto: bool = False,
        scale_up: bool = True,
    ):
        """
        初始化 Letterbox 变换
        
        Args:
            target_size: 目标尺寸 (H, W)
            color: 填充颜色 (B, G, R)
            stride: 对齐步长
            auto: 是否自动计算最小填充
            scale_up: 是否允许放大
        """
        self.target_size = target_size
        self.color = color
        self.stride = stride
        self.auto = auto
        self.scale_up = scale_up
    
    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        应用 Letterbox 变换
        
        Args:
            image: 输入图像 (H, W, C), BGR 格式
            
        Returns:
            变换后的图像和元信息
        """
        shape = image.shape[:2]  # (H, W)
        new_shape = self.target_size  # (H, W)
        
        # 计算缩放比例
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scale_up:
            r = min(r, 1.0)
        
        # 计算新尺寸
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (W, H)
        dw = new_shape[1] - new_unpad[0]  # width padding
        dh = new_shape[0] - new_unpad[1]  # height padding
        
        if self.auto:
            dw = dw % self.stride
            dh = dh % self.stride
        
        # 居中填充
        dw /= 2
        dh /= 2
        
        # 缩放
        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        # 填充
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=self.color
        )
        
        # 确保尺寸对齐到 stride
        h, w = image.shape[:2]
        if h % self.stride != 0 or w % self.stride != 0:
            new_h = ((h // self.stride) + 1) * self.stride
            new_w = ((w // self.stride) + 1) * self.stride
            pad_h = new_h - h
            pad_w = new_w - w
            image = cv2.copyMakeBorder(
                image, 0, pad_h, 0, pad_w,
                cv2.BORDER_CONSTANT, value=self.color
            )
        
        meta = {
            "original_shape": shape,
            "ratio": r,
            "padding": (dw, dh),
            "new_shape": image.shape[:2],
        }
        
        return image, meta


# ============================================================================
# 校准数据管理器
# ============================================================================

class CalibrationDataManager:
    """
    校准数据管理器
    
    支持 ImageFolder 和 COCO 格式。
    根据任务类型自动选择正确的预处理方式：
    - CLS (分类): 直接 Resize，使用 ImageNet 标准化
    - DET (检测): Letterbox 保持比例，无标准化
    - SEG (分割): Letterbox 保持比例，标准化
    
    提供多种数据迭代接口以适配不同后端。
    """
    
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    
    def __init__(
        self,
        config: ConversionConfig,
        input_shape: Tuple[int, ...],  # (B, C, H, W)
    ):
        """
        初始化校准数据管理器
        
        Args:
            config: 转换配置
            input_shape: 输入形状 (B, C, H, W)
        """
        if not CV2_AVAILABLE:
            raise DependencyError("OpenCV (cv2) is required for calibration data processing")
        
        self.config = config
        self.input_shape = input_shape
        self.batch_size = config.calib_batch_size
        self.num_samples = config.calib_num_samples
        
        # 解析输入尺寸
        if len(input_shape) == 4:
            _, self.channels, self.height, self.width = input_shape
        else:
            raise ValueError(f"Expected 4D input shape (B,C,H,W), got {input_shape}")
        
        # 获取预处理配置
        self.preprocess_config = config.get_preprocessing_config()
        
        # 初始化 Letterbox 变换 (仅在需要时使用)
        self.letterbox = LetterboxTransform(
            target_size=(self.height, self.width),
            color=config.letterbox_color,
            stride=config.letterbox_stride,
        )
        
        # 记录预处理方式
        self._log_preprocessing_config()
        
        # 收集图像路径
        self.image_paths = self._collect_image_paths()
        
        if len(self.image_paths) == 0:
            raise CalibrationError(
                f"No images found in {config.calib_data_path} "
                f"with format {config.calib_data_format.value}"
            )
        
        # 限制样本数
        if len(self.image_paths) > self.num_samples:
            self.image_paths = self.image_paths[:self.num_samples]
        
        logger.info(f"Calibration data: {len(self.image_paths)} images loaded")
    
    def _log_preprocessing_config(self):
        """记录预处理配置信息"""
        task_type = self.config.task_type or "unknown"
        use_letterbox = self.preprocess_config.use_letterbox
        
        logger.info(f"📋 校准数据预处理配置:")
        logger.info(f"   任务类型: {task_type}")
        logger.info(f"   缩放方式: {'Letterbox (保持比例)' if use_letterbox else '直接 Resize'}")
        logger.info(f"   通道顺序: {self.preprocess_config.channel_order}")
        logger.info(f"   input_scale: {self.config.input_scale}")
        logger.info(f"   normalize_mean: {self.config.normalize_mean}")
        logger.info(f"   normalize_std: {self.config.normalize_std}")
    
    def _collect_image_paths(self) -> List[str]:
        """收集图像路径"""
        data_path = Path(self.config.calib_data_path)
        
        if self.config.calib_data_format == CalibDataFormat.COCO:
            return self._collect_coco_images(data_path)
        else:
            return self._collect_folder_images(data_path)
    
    def _collect_folder_images(self, data_path: Path) -> List[str]:
        """从 ImageFolder 格式收集图像"""
        image_paths = []
        
        for ext in self.SUPPORTED_EXTENSIONS:
            image_paths.extend(data_path.rglob(f"*{ext}"))
            image_paths.extend(data_path.rglob(f"*{ext.upper()}"))
        
        return sorted([str(p) for p in image_paths])
    
    def _collect_coco_images(self, data_path: Path) -> List[str]:
        """从 COCO 格式收集图像"""
        # COCO 格式: data_path/images/ 或 data_path/
        images_dir = data_path / "images"
        if not images_dir.exists():
            images_dir = data_path
        
        return self._collect_folder_images(images_dir)
    
    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """
        预处理单张图像
        
        根据任务类型选择正确的预处理方式：
        - CLS (分类): 直接 Resize 到目标尺寸，使用 ImageNet 标准化
        - DET (检测): Letterbox 保持比例缩放，灰色填充，仅 /255 归一化
        - SEG (分割): Letterbox 保持比例缩放，标准化
        
        Returns:
            预处理后的图像 (C, H, W), float32, 归一化
        """
        # 读取图像 (BGR) - 支持中文路径
        # cv2.imread 在 Windows 上不支持非 ASCII 路径
        image = None
        
        # 方法1: 使用 Python 内置 open() + np.frombuffer (最可靠的中文路径支持)
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        except Exception as e:
            pass  # 继续尝试其他方法
        
        # 方法2: 使用 np.fromfile (备选方案)
        if image is None:
            try:
                image_array = np.fromfile(image_path, dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            except Exception:
                pass
        
        # 方法3: 回退到 cv2.imread (仅限 ASCII 路径)
        if image is None:
            try:
                image = cv2.imread(image_path)
            except Exception:
                pass
        
        if image is None:
            raise CalibrationError(f"Failed to load image: {image_path}")
        
        # ============================================================
        # Step 1: 缩放方式 - 根据 use_letterbox 配置决定
        # ============================================================
        if self.preprocess_config.use_letterbox:
            # Letterbox 变换 (检测/分割任务) - 保持比例，灰色填充
            image, _ = self.letterbox(image)
        else:
            # 直接 Resize (分类任务) - 可能会改变比例
            image = cv2.resize(
                image, 
                (self.width, self.height),
                interpolation=cv2.INTER_LINEAR
            )
        
        # ============================================================
        # Step 2: 颜色通道转换 - 根据 channel_order 配置决定
        # ============================================================
        if self.preprocess_config.channel_order == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 如果是 BGR，不需要转换 (cv2.imread 默认就是 BGR)
        
        # ============================================================
        # Step 3: 维度转换 HWC -> CHW
        # ============================================================
        image = image.transpose(2, 0, 1)
        
        # ============================================================
        # Step 4: 像素值缩放 (通常是 /255)
        # ============================================================
        image = image.astype(np.float32)
        image *= self.config.input_scale  # 默认 1/255
        
        # ============================================================
        # Step 5: 标准化 (减均值除标准差)
        # - CLS: 使用 ImageNet 均值 (0.485, 0.456, 0.406) 和标准差 (0.229, 0.224, 0.225)
        # - DET (YOLO): 均值 (0,0,0)，标准差 (1,1,1)，相当于不做标准化
        # ============================================================
        mean = np.array(self.config.normalize_mean, dtype=np.float32).reshape(3, 1, 1)
        std = np.array(self.config.normalize_std, dtype=np.float32).reshape(3, 1, 1)
        image = (image - mean) / std
        
        return image
    
    def get_batch_iterator(self) -> Iterator[np.ndarray]:
        """
        获取批量数据迭代器
        
        Yields:
            批量数据 (B, C, H, W), float32
        """
        batch = []
        
        iterator = tqdm(self.image_paths, desc="Loading calibration data") if TQDM_AVAILABLE else self.image_paths
        
        for image_path in iterator:
            try:
                image = self._preprocess_image(image_path)
                batch.append(image)
                
                if len(batch) >= self.batch_size:
                    yield np.stack(batch, axis=0)
                    batch = []
            except Exception as e:
                logger.warning(f"Failed to process {image_path}: {e}")
                continue
        
        # 处理剩余数据
        if batch:
            yield np.stack(batch, axis=0)
    
    def get_single_iterator(self) -> Iterator[np.ndarray]:
        """
        获取单张图像迭代器
        
        Yields:
            单张图像 (1, C, H, W), float32
        """
        for image_path in self.image_paths:
            try:
                image = self._preprocess_image(image_path)
                yield image[np.newaxis, ...]
            except Exception as e:
                logger.warning(f"Failed to process {image_path}: {e}")
                continue
    
    def get_numpy_data(self, max_samples: Optional[int] = None) -> np.ndarray:
        """
        获取全部数据为 numpy 数组
        
        Args:
            max_samples: 最大样本数
            
        Returns:
            (N, C, H, W), float32
        """
        samples = []
        count = 0
        max_count = max_samples or self.num_samples
        
        for batch in self.get_batch_iterator():
            for i in range(batch.shape[0]):
                if count >= max_count:
                    break
                samples.append(batch[i])
                count += 1
            if count >= max_count:
                break
        
        return np.stack(samples, axis=0)
    
    def __len__(self) -> int:
        return len(self.image_paths)


# ============================================================================
# 敏感层管理器
# ============================================================================

class SensitiveLayerManager:
    """
    敏感层管理器 - MIXED 模式使用
    
    管理需要保持高精度 (FP16) 的层，其他层使用 INT8。
    
    支持的框架命名格式:
    - ONNX: "/classifier/Gemm", "Conv_0"
    - OpenVINO: "classifier", "MatMul_123"
    - TensorRT: 类似 ONNX
    """
    
    # 默认敏感层模式 (正则表达式)
    # 设计为兼容多种模型和命名格式:
    # - EfficientNet: conv_stem, conv_head, blocks.X.se, classifier
    # - MobileNetV3: conv_stem, conv_head, blocks.X.se, classifier  
    # - WideResNet: conv1, fc (无 SE 模块)
    DEFAULT_SENSITIVE_PATTERNS = [
        # ============================================================
        # 第一层卷积 (输入分布变化大，对量化最敏感)
        # ============================================================
        r"(?:^|[/._])Conv_?0(?:[/._]|$)",           # Conv_0, Conv0, /Conv_0/
        r"^/?conv1[/._]",                           # /conv1/ (WideResNet 根路径，不匹配 /layer1.0/conv1/)
        r"(?:^|[/._])conv_stem(?:[/._])",           # /conv_stem/ (EfficientNet, MobileNetV3)
        r"(?:^|[/._])stem(?:[/._])",                # stem., /stem/
        r"(?:^|[/._])features[/._]0(?:[/._]|$)",    # features.0, features/0
        
        # ============================================================
        # 最后几层 (对输出影响大) - 分类头
        # ============================================================
        r"(?:^|[/._])conv_head(?:[/._])",           # /conv_head/ (EfficientNet, MobileNetV3)
        r"(?:^|[/._])head(?:[/._]|$)",              # head, /head/, .head.
        r"(?:^|[/._])fc(?:[/._]|$)",                # fc, .fc., /fc/ (WideResNet 分类器)
        r"(?:^|[/._])classifier(?:[/._]|$)",        # classifier, /classifier/ (EfficientNet)
        r"(?:^|[/._])pred(?:[/._]|$)",              # pred, /pred/, .pred.
        r"(?:^|[/._])output(?:[/._]|$)",            # output layer
        
        # ============================================================
        # SE 模块 (Squeeze-and-Excitation) - EfficientNet/MobileNetV3 特有
        # 对量化非常敏感: Sigmoid 输出 [0,1]，乘法累积误差
        # 节点名示例: /blocks.0/se/fc1/Conv, /blocks.0/se/Sigmoid
        # ============================================================
        r"[/._]se[/._]",                            # /se/, .se. (SE 模块内所有操作)
        r"[/._]se_module[/._]",                     # /se_module/ (备选命名)
        r"[/._]squeeze[/._]",                       # squeeze 操作
        r"[/._]excitation[/._]",                    # excitation 操作
        
        # ============================================================
        # 检测/分割头 (YOLO, Detectron2)
        # ============================================================
        r"(?:^|[/._])detect(?:[/._]|$)",            # detect head
        r"(?:^|[/._])segment(?:[/._]|$)",           # segment head
        
        # ============================================================
        # 精度敏感算子 (通用)
        # ============================================================
        r"Softmax",                                  # Softmax 操作
        r"Sigmoid",                                  # Sigmoid 操作 (SE 模块关键)
        r"LayerNorm",                               # LayerNormalization
        r"layer_norm",
        
        # ============================================================
        # 按操作类型匹配 (OpenVINO 格式)
        # ============================================================
        r"^Softmax:",                                # Softmax:xxx
        r"^LayerNorm",                               # LayerNormalization
        r"^MatMul:.*(?:classifier|fc|head|output)", # MatMul in classifier
        r"^Gemm:.*(?:classifier|fc|head|output)",   # Gemm in classifier
    ]
    
    def __init__(
        self,
        use_default: bool = True,
        custom_layers: Optional[List[str]] = None,
    ):
        """
        初始化敏感层管理器
        
        Args:
            use_default: 是否使用默认敏感层策略
            custom_layers: 自定义敏感层列表
        """
        self.patterns = []
        
        if use_default:
            self.patterns.extend([re.compile(p) for p in self.DEFAULT_SENSITIVE_PATTERNS])
        
        if custom_layers:
            # 自定义层名称作为精确匹配或包含匹配
            for layer in custom_layers:
                # 如果是正则表达式格式，直接编译
                if layer.startswith("^") or layer.endswith("$") or "*" in layer:
                    self.patterns.append(re.compile(layer.replace("*", ".*")))
                else:
                    # 否则作为包含匹配
                    self.patterns.append(re.compile(re.escape(layer)))
    
    def is_sensitive(self, layer_name: str) -> bool:
        """判断某层是否为敏感层"""
        for pattern in self.patterns:
            if pattern.search(layer_name):
                return True
        return False
    
    def get_sensitive_layers(self, model_or_path: Union[str, "onnx.ModelProto"]) -> List[str]:
        """
        从模型中获取所有敏感层名称
        
        Args:
            model_or_path: ONNX 模型或路径
            
        Returns:
            敏感层名称列表
        """
        if not ONNX_AVAILABLE:
            raise DependencyError("ONNX is required for sensitive layer analysis")
        
        if isinstance(model_or_path, str):
            model = onnx.load(model_or_path)
        else:
            model = model_or_path
        
        sensitive_layers = []
        
        for node in model.graph.node:
            if self.is_sensitive(node.name) or self.is_sensitive(node.op_type):
                sensitive_layers.append(node.name)
        
        return sensitive_layers
    
    def get_non_sensitive_layers(self, model_or_path: Union[str, "onnx.ModelProto"]) -> List[str]:
        """获取所有非敏感层（用于 INT8 量化）"""
        if not ONNX_AVAILABLE:
            raise DependencyError("ONNX is required for sensitive layer analysis")
        
        if isinstance(model_or_path, str):
            model = onnx.load(model_or_path)
        else:
            model = model_or_path
        
        sensitive_set = set(self.get_sensitive_layers(model))
        
        non_sensitive = []
        for node in model.graph.node:
            if node.name not in sensitive_set:
                non_sensitive.append(node.name)
        
        return non_sensitive


# ============================================================================
# 基础转换器抽象类
# ============================================================================

class BaseConverter(ABC):
    """
    转换器基类
    
    所有后端转换器必须继承此类并实现 convert 方法。
    """
    
    def __init__(self, config: ConversionConfig):
        """
        初始化转换器
        
        Args:
            config: 转换配置
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
    
    def _check_onnx_available(self) -> bool:
        """
        检查 ONNX 是否可用
        
        Returns:
            bool: ONNX 是否可用
        """
        if not ONNX_AVAILABLE:
            self.logger.error("ONNX is not installed. Please install with: pip install onnx")
            return False
        return True
    
    def _check_file_exists(self, path: str, file_type: str = "file") -> bool:
        """
        检查文件是否存在
        
        Args:
            path: 文件路径
            file_type: 文件类型描述 (用于错误信息)
            
        Returns:
            bool: 文件是否存在
        """
        if not os.path.exists(path):
            self.logger.error(f"{file_type} not found: {path}")
            return False
        return True
    
    def _ensure_output_dir(self, output_path: str) -> None:
        """
        确保输出目录存在
        
        Args:
            output_path: 输出文件路径
        """
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def check_dependencies(self) -> bool:
        """检查依赖是否满足"""
        pass
    
    @abstractmethod
    def convert(
        self,
        onnx_path: str,
        output_path: str,
        input_shape: Tuple[int, ...],
        dynamic_axes_spec: Optional[Dict] = None,
    ) -> ConversionResult:
        """
        执行转换
        
        Args:
            onnx_path: 输入 ONNX 模型路径
            output_path: 输出路径
            input_shape: 输入形状 (B, C, H, W)
            dynamic_axes_spec: 动态轴规格 (min/opt/max)
            
        Returns:
            转换结果
        """
        pass
    
    def get_file_size_mb(self, path: str) -> float:
        """获取文件大小 (MB)"""
        if os.path.isfile(path):
            return os.path.getsize(path) / (1024 * 1024)
        elif os.path.isdir(path):
            total = 0
            for f in Path(path).rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
            return total / (1024 * 1024)
        return 0.0


# ============================================================================
# 主转换器类
# ============================================================================

class ModelConverter:
    """
    模型转换器 - 统一入口
    
    根据配置自动路由到对应的后端转换器。
    """
    
    def __init__(self, config: ConversionConfig):
        """
        初始化模型转换器
        
        Args:
            config: 转换配置
        """
        self.config = config
        self.logger = Logger.get("ModelConverter")
        self._converter: Optional[BaseConverter] = None
        self._validator = None
        
        # 初始化对应的转换器
        self._init_converter()
    
    def _init_converter(self):
        """初始化对应后端的转换器"""
        backend = self.config.target_backend
        
        if backend == TargetBackend.ORT:
            from converter_ort import ORTConverter
            self._converter = ORTConverter(self.config)
            
        elif backend == TargetBackend.TENSORRT:
            from converter_tensorrt import TensorRTConverter
            self._converter = TensorRTConverter(self.config)
            
        elif backend == TargetBackend.OPENVINO:
            from converter_openvino import OpenVINOConverter
            self._converter = OpenVINOConverter(self.config)
            
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        # 检查依赖
        if not self._converter.check_dependencies():
            raise DependencyError(
                f"Dependencies for {backend.value} are not satisfied. "
                f"Please install the required packages."
            )
    
    def convert(
        self,
        onnx_path: str,
        output_path: str,
        input_shape: Optional[Tuple[int, ...]] = None,
        dynamic_axes_spec: Optional[Dict] = None,
    ) -> ConversionResult:
        """
        执行模型转换
        
        Args:
            onnx_path: 输入 ONNX 模型路径
            output_path: 输出路径
            input_shape: 输入形状 (B, C, H, W)，如果为 None 则从模型推断
            dynamic_axes_spec: 动态轴规格
            
        Returns:
            转换结果
        """
        # 检查输入
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        
        # 推断输入形状
        if input_shape is None:
            input_shape = self._infer_input_shape(onnx_path)
        
        # 使用配置中的 dynamic_axes_spec
        if dynamic_axes_spec is None:
            dynamic_axes_spec = self.config.dynamic_axes_spec
        
        # 记录开始时间
        start_time = time.time()
        
        # 创建校准数据管理器（用于 INT8 校准和后续验证）
        # 注意：这里创建一次，供转换和验证共同使用
        self._calib_data_manager = None
        if self.config.calib_data_path and os.path.exists(self.config.calib_data_path):
            try:
                self._calib_data_manager = CalibrationDataManager(
                    config=self.config,
                    input_shape=input_shape,
                )
                self.logger.info(f"Calibration data: {len(self._calib_data_manager)} images loaded")
            except Exception as e:
                self.logger.warning(f"Failed to create calibration data manager: {e}")
                self._calib_data_manager = None
        
        try:
            # 执行转换
            result = self._converter.convert(
                onnx_path=onnx_path,
                output_path=output_path,
                input_shape=input_shape,
                dynamic_axes_spec=dynamic_axes_spec,
            )
            
            # 记录构建时间
            result.build_time_seconds = time.time() - start_time
            
            # 验证（使用真实校准数据）
            if self.config.enable_validation and result.success:
                self.logger.info("Running validation...")
                result.validation = self._validate(
                    onnx_path=onnx_path,
                    converted_path=result.output_files[0] if result.output_files else output_path,
                    input_shape=input_shape,
                    calib_data_manager=self._calib_data_manager,
                )
                
                # 检查验证是否成功
                if result.validation and not result.validation.passed:
                    if result.validation.warnings:
                        for warning in result.validation.warnings:
                            self.logger.warning(f"Validation warning: {warning}")
            
            # 生成报告
            if self.config.generate_report and result.success:
                report_path = self._get_report_path(output_path)
                result.save_report(report_path)
            
            # 日志
            if result.success:
                self.logger.info(f"Conversion successful!")
                self.logger.info(f"Build time: {result.build_time_seconds:.2f}s")
                if result.validation:
                    if result.validation.passed:
                        self.logger.info(f"Cosine similarity: {result.validation.cosine_sim:.6f}")
                        self.logger.info(f"Compression ratio: {result.validation.compression_ratio:.2f}x")
                    else:
                        self.logger.warning(f"Validation FAILED - results may be inaccurate")
            else:
                self.logger.error(f"Conversion failed: {result.message}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Conversion error: {e}")
            import traceback
            traceback.print_exc()
            return ConversionResult(
                success=False,
                target_backend=self.config.target_backend,
                precision_mode=self.config.precision_mode,
                output_files=[],
                build_time_seconds=time.time() - start_time,
                message=str(e),
            )
    
    def _infer_input_shape(self, onnx_path: str) -> Tuple[int, ...]:
        """
        从 ONNX 模型推断输入形状
        
        遍历模型的输入节点，提取第一个 4D 输入的形状。
        对于动态维度，使用默认值 1 替代。
        
        Args:
            onnx_path: ONNX 模型文件路径
            
        Returns:
            输入形状元组 (B, C, H, W)
            
        Raises:
            DependencyError: ONNX 未安装
            ValueError: 无法从模型推断输入形状
        """
        if not ONNX_AVAILABLE:
            raise DependencyError("ONNX is required to infer input shape")
        
        model = onnx.load(onnx_path)
        
        for inp in model.graph.input:
            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    # 动态维度，使用默认值
                    shape.append(1)
            
            if len(shape) == 4:
                return tuple(shape)
        
        raise ValueError("Could not infer input shape from ONNX model")
    
    def _validate(
        self,
        onnx_path: str,
        converted_path: str,
        input_shape: Tuple[int, ...],
        calib_data_manager: Optional[CalibrationDataManager] = None,
    ) -> ValidationResult:
        """
        验证转换结果
        
        Args:
            onnx_path: 原始 ONNX 模型路径
            converted_path: 转换后模型路径
            input_shape: 输入形状
            calib_data_manager: 校准数据管理器（用于真实数据验证）
            
        Returns:
            验证结果
        """
        from conversion_validator import ConversionValidator
        
        validator = ConversionValidator(
            enable_perf_test=self.config.enable_perf_test,
        )
        
        return validator.validate(
            original_onnx_path=onnx_path,
            converted_path=converted_path,
            target_backend=self.config.target_backend,
            precision_mode=self.config.precision_mode,
            input_shape=input_shape,
            num_samples=self.config.validation_samples,
            calib_data_manager=calib_data_manager,
        )
    
    def _get_report_path(self, output_path: str) -> str:
        """获取报告路径"""
        output_path = Path(output_path)
        if output_path.is_dir():
            return str(output_path / "conversion_report.json")
        else:
            return str(output_path.parent / f"{output_path.stem}_report.json")


# ============================================================================
# 便捷函数
# ============================================================================

def convert_model(
    onnx_path: str,
    output_path: str,
    backend: Union[str, TargetBackend],
    precision: Union[str, PrecisionMode] = "fp16",
    calib_data_path: Optional[str] = None,
    calib_data_format: str = "imagefolder",
    calib_num_samples: int = 300,
    input_shape: Optional[Tuple[int, ...]] = None,
    dynamic_axes_spec: Optional[Dict] = None,
    **kwargs,
) -> ConversionResult:
    """
    便捷函数：转换模型
    
    Args:
        onnx_path: 输入 ONNX 模型路径
        output_path: 输出路径
        backend: 目标后端 ("ort", "tensorrt", "openvino")
        precision: 精度模式 ("fp32", "fp16", "int8", "mixed")
        calib_data_path: 校准数据路径 (INT8/MIXED 需要)
        calib_data_format: 校准数据格式 ("imagefolder", "coco")
        calib_num_samples: 校准样本数
        input_shape: 输入形状 (B, C, H, W)
        dynamic_axes_spec: 动态轴规格
        **kwargs: 其他配置参数
        
    Returns:
        转换结果
        
    Example:
        >>> result = convert_model(
        ...     "model.onnx",
        ...     "model.engine",
        ...     backend="tensorrt",
        ...     precision="int8",
        ...     calib_data_path="./coco_val/images",
        ...     calib_data_format="coco",
        ... )
    """
    # 转换枚举类型
    if isinstance(backend, str):
        backend = TargetBackend.from_string(backend)
    if isinstance(precision, str):
        precision = PrecisionMode.from_string(precision)
    
    # 创建配置
    config = ConversionConfig(
        target_backend=backend,
        precision_mode=precision,
        calib_data_path=calib_data_path,
        calib_data_format=CalibDataFormat.from_string(calib_data_format) if calib_data_path else CalibDataFormat.IMAGEFOLDER,
        calib_num_samples=calib_num_samples,
        input_shape=input_shape,
        dynamic_axes_spec=dynamic_axes_spec,
        **kwargs,
    )
    
    # 创建转换器并执行
    converter = ModelConverter(config)
    return converter.convert(
        onnx_path=onnx_path,
        output_path=output_path,
        input_shape=input_shape,
        dynamic_axes_spec=dynamic_axes_spec,
    )


# ============================================================================
# CLI 接口
# ============================================================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Model Quantization and Conversion Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # FP16 with ONNX Runtime
  python model_converter.py -m model.onnx -b ort -p fp16 -o model_fp16.onnx
  
  # INT8 with TensorRT
  python model_converter.py -m model.onnx -b tensorrt -p int8 \\
      --calib-data ./images --calib-format imagefolder -o model.engine
  
  # Mixed precision with OpenVINO
  python model_converter.py -m model.onnx -b openvino -p mixed \\
      --calib-data ./coco_val/images --calib-format coco -o model_dir/
        """
    )
    
    # 必需参数
    parser.add_argument("-m", "--model", required=True, help="Input ONNX model path")
    parser.add_argument("-b", "--backend", required=True,
                       choices=["ort", "tensorrt", "openvino"],
                       help="Target backend")
    parser.add_argument("-o", "--output", required=True, help="Output path")
    
    # 精度配置
    parser.add_argument("-p", "--precision", default="fp16",
                       choices=["fp32", "fp16", "int8", "mixed"],
                       help="Precision mode (default: fp16)")
    
    # 校准配置
    parser.add_argument("--calib-data", help="Calibration data path (required for INT8/MIXED)")
    parser.add_argument("--calib-format", default="imagefolder",
                       choices=["imagefolder", "coco"],
                       help="Calibration data format (default: imagefolder)")
    parser.add_argument("--calib-samples", type=int, default=300,
                       help="Number of calibration samples (default: 300)")
    parser.add_argument("--calib-batch-size", type=int, default=8,
                       help="Calibration batch size (default: 8)")
    parser.add_argument("--calib-method", default="entropy",
                       choices=["entropy", "minmax"],
                       help="Calibration method (default: entropy)")
    
    # 混合精度配置
    parser.add_argument("--sensitive-layers", nargs="+",
                       help="Sensitive layers to keep in FP16 (for MIXED mode)")
    parser.add_argument("--no-default-sensitive", action="store_true",
                       help="Disable default sensitive layer detection")
    
    # 验证配置
    parser.add_argument("--no-validation", action="store_true",
                       help="Disable validation after conversion")
    parser.add_argument("--perf-test", action="store_true",
                       help="Enable performance testing")
    
    # TensorRT 配置
    parser.add_argument("--trt-workspace", type=int, default=4,
                       help="TensorRT workspace size in GB (default: 4)")
    parser.add_argument("--trt-timing-cache",
                       help="TensorRT timing cache path")
    
    # OpenVINO 配置
    parser.add_argument("--ov-device", default="CPU",
                       help="OpenVINO device (default: CPU)")
    
    # 输入配置
    parser.add_argument("--input-shape", nargs=4, type=int,
                       metavar=("B", "C", "H", "W"),
                       help="Input shape (B C H W)")
    
    # 其他
    parser.add_argument("--save-intermediate", action="store_true",
                       help="Save intermediate ONNX model")
    parser.add_argument("--no-report", action="store_true",
                       help="Disable JSON report generation")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        Logger.get("root").setLevel(10)
    
    # 构建配置
    config_dict = {
        "target_backend": TargetBackend.from_string(args.backend),
        "precision_mode": PrecisionMode.from_string(args.precision),
        "calib_data_path": args.calib_data,
        "calib_data_format": CalibDataFormat.from_string(args.calib_format),
        "calib_num_samples": args.calib_samples,
        "calib_batch_size": args.calib_batch_size,
        "calib_method": CalibrationMethod.from_string(args.calib_method),
        "sensitive_layers": args.sensitive_layers,
        "use_default_sensitive": not args.no_default_sensitive,
        "enable_validation": not args.no_validation,
        "enable_perf_test": args.perf_test,
        "save_intermediate_onnx": args.save_intermediate,
        "generate_report": not args.no_report,
        "trt_workspace_gb": args.trt_workspace,
        "trt_timing_cache_path": args.trt_timing_cache,
        "openvino_device": args.ov_device,
    }
    
    if args.input_shape:
        config_dict["input_shape"] = tuple(args.input_shape)
    
    try:
        config = ConversionConfig(**config_dict)
    except ValueError as e:
        parser.error(str(e))
        return 1
    
    # 执行转换
    converter = ModelConverter(config)
    result = converter.convert(
        onnx_path=args.model,
        output_path=args.output,
        input_shape=config_dict.get("input_shape"),
    )
    
    # 返回状态码
    return 0 if result.success else 1


if __name__ == "__main__":
    exit(main())    