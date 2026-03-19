# -*- coding: utf-8 -*-
"""
PatchCore 默认配置

工业级异常检测系统配置
"""

from typing import List, Tuple

# ==================== 图像处理配置 ====================

# ImageNet 归一化参数
IMAGENET_MEAN: Tuple[float, ...] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, ...] = (0.229, 0.224, 0.225)

# 支持的图像格式
SUPPORTED_IMAGE_FORMATS: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

# 图像尺寸选项
IMAGE_SIZE_OPTIONS: List[int] = [224, 256, 384, 512]
DEFAULT_IMAGE_SIZE: int = 256

# ==================== Backbone 配置 ====================

BACKBONE_OPTIONS = {
    'wide_resnet50_2': {
        'name': 'WideResNet50-2',
        'description': '高精度骨干网络，适合精细缺陷检测',
        'feature_dims': {'layer2': 512, 'layer3': 1024},
        'total_dim': 1536,
        'speed': 'medium',
        'accuracy': 'high',
    },
    'resnet18': {
        'name': 'ResNet18',
        'description': '轻量级骨干网络，适合实时检测',
        'feature_dims': {'layer2': 128, 'layer3': 256},
        'total_dim': 384,
        'speed': 'fast',
        'accuracy': 'medium',
    },
    'resnet50': {
        'name': 'ResNet50',
        'description': '标准骨干网络，精度速度平衡',
        'feature_dims': {'layer2': 512, 'layer3': 1024},
        'total_dim': 1536,
        'speed': 'medium',
        'accuracy': 'high',
    },
}

DEFAULT_BACKBONE: str = 'wide_resnet50_2'
DEFAULT_LAYERS: List[str] = ['layer2', 'layer3']

# ==================== Memory Bank 配置 ====================

# CoreSet 采样
DEFAULT_CORESET_RATIO: float = 0.01  # 1% 采样率
MIN_CORESET_RATIO: float = 0.001
MAX_CORESET_RATIO: float = 0.1

# PCA 降维
DEFAULT_PCA_COMPONENTS: int = 256
PCA_VARIANCE_THRESHOLD: float = 0.995  # 保留99.5%方差
AUTO_PCA_THRESHOLD_DIM: int = 512  # 超过此维度自动启用PCA

# KNN 配置
DEFAULT_KNN_K: int = 9
MIN_KNN_K: int = 1
MAX_KNN_K: int = 50

# ==================== 阈值配置 ====================

# 归一化分数范围
SCORE_RANGE: Tuple[float, float] = (0.0, 100.0)

# 默认阈值
DEFAULT_THRESHOLD: float = 50.0
THRESHOLD_PRESETS = {
    'ultra_sensitive': 30.0,   # 高召回
    'sensitive': 40.0,
    'balanced': 50.0,          # 默认
    'strict': 65.0,
    'very_strict': 80.0,       # 高精度
}

# 阈值校准配置
CALIBRATION_PERCENTILES = {
    'p1': 1,
    'p50': 50,
    'p95': 95,
    'p99': 99,
    'p99_5': 99.5,
}

# ==================== 导出配置 ====================

EXPORT_FORMATS = ['pkg', 'onnx', 'tensorrt']
DEFAULT_EXPORT_FORMAT: str = 'pkg'

TENSORRT_PRECISIONS = ['fp32', 'fp16', 'int8']
DEFAULT_TENSORRT_PRECISION: str = 'fp16'

# Faiss 索引类型
FAISS_INDEX_TYPES = {
    'auto': '自动选择（推荐）',
    'Flat': '暴力搜索（小规模精确）',
    'IVFFlat': 'IVF索引（中规模）',
    'IVFPQ': 'IVF-PQ索引（大规模）',
}

# 自动选择阈值
FAISS_AUTO_THRESHOLDS = {
    'flat_max': 5000,      # Flat索引最大规模
    'ivf_max': 50000,      # IVFFlat索引最大规模
}

# ==================== 性能优化配置 ====================

# 内存优化
DEFAULT_USE_FP16_FEATURES: bool = True
DEFAULT_FEATURE_CHUNK_SIZE: int = 10000
DEFAULT_INCREMENTAL_PCA: bool = True
DEFAULT_PCA_BATCH_SIZE: int = 5000

# 采样优化
DEFAULT_RANDOM_PROJECTION_DIM: int = 128
DEFAULT_CORESET_MAX_ITER: int = 100

# 批处理
DEFAULT_FEATURE_BATCH_SIZE: int = 8
DEFAULT_INFERENCE_BATCH_SIZE: int = 4

# ==================== 输出目录结构 ====================

OUTPUT_DIRS = {
    'weights': 'weights',
    'exports': 'exports',
    'visualizations': 'visualizations',
    'logs': 'logs',
    'cache': 'cache',
}

OUTPUT_FILENAMES = {
    'model_config': 'config.json',
    'training_info': 'training_info.json',
    'memory_bank': 'memory_bank.npz',
    'faiss_index': 'faiss_index.bin',
    'pca_model': 'pca_model.npz',
    'normalization': 'normalization.json',
    'thresholds': 'thresholds.json',
    'backbone_onnx': 'backbone.onnx',
    'backbone_trt': 'backbone.engine',
    'package': 'model.pkg',
}

# ==================== 设备配置 ====================

DEVICE_OPTIONS = ['cuda:0', 'cuda:1', 'cpu', 'auto']
DEFAULT_DEVICE: str = 'auto'

# ==================== 后处理配置 ====================

# 高斯平滑
DEFAULT_GAUSSIAN_SIGMA: float = 4.0
DEFAULT_GAUSSIAN_KERNEL_SIZE: int = 33

# 上采样
DEFAULT_UPSAMPLE_MODE: str = 'bilinear'

# ==================== 可视化配置 ====================

HEATMAP_COLORMAP: str = 'jet'
CONTOUR_COLOR: Tuple[int, int, int] = (0, 255, 0)  # 绿色
CONTOUR_THICKNESS: int = 2

# ==================== 数据增强配置 (简化版) ====================

AUGMENTATION_CONFIG = {
    'brightness': {
        'enabled': False,
        'range': 0.1,
    },
    'contrast': {
        'enabled': False,
        'range': 0.1,
    },
    'rotation': {
        'enabled': False,
        'degrees': 5,
    },
    'hflip': {
        'enabled': False,
    },
    'vflip': {
        'enabled': False,
    },
}

# ==================== 场景模板 (PatchCore) ====================
# 采样率说明：工业推荐1-5%，高精度5-10%，快速0.5-1%
# 图像增强：训练时可选，推理时不使用（保持一致性）

SCENE_TEMPLATES = {
    'default': {
        'name': '🔧 通用检测',
        'description': '平衡配置，适合大多数场景',
        'backbone': 'wide_resnet50_2',
        'image_size': 256,
        'coreset_ratio': 0.01,  # 1%
        'pca_components': 256,
        'knn_k': 9,
        'augmentation': {
            'enabled': False,
            'hflip': False,
            'vflip': False,
            'rotation': 0,
            'brightness': 0,
            'contrast': 0,
        },
    },
    'high_precision': {
        'name': '🎯 高精度检测',
        'description': '适合精细缺陷检测，速度较慢',
        'backbone': 'wide_resnet50_2',
        'image_size': 384,
        'coreset_ratio': 0.05,  # 5%
        'pca_components': 384,
        'knn_k': 9,
        'augmentation': {
            'enabled': False,
            'hflip': False,
            'vflip': False,
            'rotation': 0,
            'brightness': 0,
            'contrast': 0,
        },
    },
    'ultra_precision': {
        'name': '🔬 极高精度',
        'description': '最高精度，适合微小缺陷',
        'backbone': 'wide_resnet50_2',
        'image_size': 512,
        'coreset_ratio': 0.10,  # 10%
        'pca_components': 512,
        'knn_k': 9,
        'augmentation': {
            'enabled': False,
            'hflip': False,
            'vflip': False,
            'rotation': 0,
            'brightness': 0,
            'contrast': 0,
        },
    },
    'fast': {
        'name': '⚡ 快速检测',
        'description': '适合实时检测，精度略降',
        'backbone': 'resnet18',
        'image_size': 224,
        'coreset_ratio': 0.005,  # 0.5%
        'pca_components': 128,
        'knn_k': 5,
        'augmentation': {
            'enabled': False,
            'hflip': False,
            'vflip': False,
            'rotation': 0,
            'brightness': 0,
            'contrast': 0,
        },
    },
    'texture': {
        'name': '🧵 纹理检测',
        'description': '布料、金属表面纹理缺陷，开启翻转增强',
        'backbone': 'wide_resnet50_2',
        'image_size': 256,
        'coreset_ratio': 0.03,  # 3%
        'pca_components': 256,
        'knn_k': 9,
        'augmentation': {
            'enabled': True,
            'hflip': True,
            'vflip': True,
            'rotation': 0,
            'brightness': 0.05,
            'contrast': 0.05,
        },
    },
    'structure': {
        'name': '🔌 结构检测',
        'description': 'PCB、组装件结构缺陷，不使用增强',
        'backbone': 'wide_resnet50_2',
        'image_size': 384,
        'coreset_ratio': 0.02,  # 2%
        'pca_components': 256,
        'knn_k': 9,
        'augmentation': {
            'enabled': False,
            'hflip': False,
            'vflip': False,
            'rotation': 0,
            'brightness': 0,
            'contrast': 0,
        },
    },
    'rotation_invariant': {
        'name': '🔄 旋转不变',
        'description': '物体可能任意角度放置，开启旋转增强',
        'backbone': 'wide_resnet50_2',
        'image_size': 256,
        'coreset_ratio': 0.03,  # 3%
        'pca_components': 256,
        'knn_k': 9,
        'augmentation': {
            'enabled': True,
            'hflip': True,
            'vflip': True,
            'rotation': 90,
            'brightness': 0,
            'contrast': 0,
        },
    },
}
