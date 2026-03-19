# -*- coding: utf-8 -*-
"""
工具函数模块

包含内存估算、日志、计时等通用工具
"""

import os
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
from contextlib import contextmanager

import numpy as np


# ==================== 日志工具 ====================

def setup_logger(
    name: str = 'patchcore',
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """设置日志器"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


# ==================== 计时工具 ====================

class Timer:
    """计时器"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed = 0
    
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """停止计时"""
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        return self.elapsed
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()
    
    @property
    def elapsed_str(self) -> str:
        """格式化的耗时字符串"""
        if self.elapsed < 60:
            return f"{self.elapsed:.1f}s"
        elif self.elapsed < 3600:
            return f"{self.elapsed/60:.1f}min"
        else:
            return f"{self.elapsed/3600:.1f}h"


@contextmanager
def timer(name: str = "操作"):
    """计时上下文管理器"""
    t = Timer()
    t.start()
    try:
        yield t
    finally:
        t.stop()
        print(f"{name} 耗时: {t.elapsed_str}")


# ==================== 内存估算 ====================

class MemoryEstimator:
    """
    PatchCore 训练内存估算器
    
    用于在训练前预估所需内存，避免OOM
    """
    
    # 每个特征的字节数 (FP16)
    BYTES_PER_FEATURE_FP16 = 2
    BYTES_PER_FEATURE_FP32 = 4
    
    # Backbone 输出特征图大小 (相对于输入尺寸)
    FEATURE_MAP_RATIO = 8  # 例如 256x256 -> 32x32
    
    def __init__(
        self,
        num_images: int,
        image_size: int,
        backbone_name: str = 'wide_resnet50_2',
        coreset_ratio: float = 0.01,
        pca_components: int = 256,
        use_fp16: bool = True,
    ):
        self.num_images = num_images
        self.image_size = image_size
        self.backbone_name = backbone_name
        self.coreset_ratio = coreset_ratio
        self.pca_components = pca_components
        self.use_fp16 = use_fp16
        
        # 获取特征维度
        self.feature_dim = self._get_feature_dim()
        
        # 计算patch数量
        self.feature_map_size = image_size // self.FEATURE_MAP_RATIO
        self.patches_per_image = self.feature_map_size ** 2
        self.total_patches = num_images * self.patches_per_image
    
    def _get_feature_dim(self) -> int:
        """获取特征维度"""
        from config import BACKBONE_OPTIONS
        if self.backbone_name in BACKBONE_OPTIONS:
            return BACKBONE_OPTIONS[self.backbone_name]['total_dim']
        return 1536  # 默认
    
    def estimate_feature_memory_gb(self) -> float:
        """估算特征提取阶段内存 (GB)"""
        bytes_per_feature = (self.BYTES_PER_FEATURE_FP16 if self.use_fp16 
                            else self.BYTES_PER_FEATURE_FP32)
        
        total_bytes = self.total_patches * self.feature_dim * bytes_per_feature
        
        return total_bytes / (1024 ** 3)
    
    def estimate_pca_memory_gb(self) -> float:
        """估算PCA阶段内存 (GB)"""
        if self.pca_components == 0:
            return 0
        
        # 协方差矩阵
        cov_matrix_bytes = self.feature_dim ** 2 * 4  # FP32
        
        # 变换后的特征
        transformed_bytes = self.total_patches * self.pca_components * 4
        
        return (cov_matrix_bytes + transformed_bytes) / (1024 ** 3)
    
    def estimate_coreset_memory_gb(self) -> float:
        """估算CoreSet采样阶段内存 (GB)"""
        # 距离矩阵 (最坏情况)
        # 使用优化算法后不需要完整距离矩阵
        sampled_patches = int(self.total_patches * self.coreset_ratio)
        
        return sampled_patches * self.pca_components * 4 / (1024 ** 3)
    
    def estimate_total_memory_gb(self) -> Tuple[float, float]:
        """
        估算总内存需求
        
        Returns:
            (peak_memory_gb, final_memory_gb)
        """
        feature_mem = self.estimate_feature_memory_gb()
        pca_mem = self.estimate_pca_memory_gb()
        coreset_mem = self.estimate_coreset_memory_gb()
        
        # 峰值内存 (特征提取 + PCA同时进行)
        peak_memory = feature_mem + pca_mem + 2.0  # 额外2GB用于模型和缓冲
        
        # 最终内存 (只保留Memory Bank)
        final_memory = coreset_mem + 1.0
        
        return peak_memory, final_memory
    
    def get_report(self) -> str:
        """生成内存估算报告"""
        feature_mem = self.estimate_feature_memory_gb()
        pca_mem = self.estimate_pca_memory_gb()
        coreset_mem = self.estimate_coreset_memory_gb()
        peak_mem, final_mem = self.estimate_total_memory_gb()
        
        sampled_patches = int(self.total_patches * self.coreset_ratio)
        
        report = f"""
## 内存估算报告

### 配置信息
- 图像数量: {self.num_images}
- 图像尺寸: {self.image_size}×{self.image_size}
- Backbone: {self.backbone_name}
- 特征维度: {self.feature_dim}-D
- 使用FP16: {self.use_fp16}

### 数据规模
- 每图Patch数: {self.patches_per_image:,}
- 总Patch数: {self.total_patches:,}
- 采样后: {sampled_patches:,} ({self.coreset_ratio:.1%})

### 内存需求
| 阶段 | 内存 (GB) |
|------|-----------|
| 特征提取 | {feature_mem:.2f} |
| PCA降维 | {pca_mem:.2f} |
| CoreSet | {coreset_mem:.2f} |

### 总结
- **峰值内存**: {peak_mem:.2f} GB
- **最终内存**: {final_mem:.2f} GB

### 建议
"""
        
        if peak_mem > 16:
            report += "⚠️ 内存需求较高，建议:\n"
            report += "- 减小图像尺寸\n"
            report += "- 降低CoreSet采样率\n"
            report += "- 启用增量PCA\n"
        elif peak_mem > 8:
            report += "✅ 内存需求适中，8GB以上GPU可运行\n"
        else:
            report += "✅ 内存需求较低，大多数GPU可运行\n"
        
        return report


def estimate_memory(
    num_images: int,
    image_size: int = 256,
    backbone: str = 'wide_resnet50_2',
    coreset_ratio: float = 0.01,
) -> Dict[str, float]:
    """
    快速估算内存需求
    
    Returns:
        {'peak_gb': float, 'final_gb': float}
    """
    estimator = MemoryEstimator(
        num_images=num_images,
        image_size=image_size,
        backbone_name=backbone,
        coreset_ratio=coreset_ratio,
    )
    
    peak, final = estimator.estimate_total_memory_gb()
    
    return {'peak_gb': peak, 'final_gb': final}


# ==================== GPU工具 ====================

def get_gpu_memory_info() -> Dict[str, float]:
    """获取GPU内存信息"""
    try:
        import torch
        
        if not torch.cuda.is_available():
            return {'available': False}
        
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory
        allocated = torch.cuda.memory_allocated(device)
        cached = torch.cuda.memory_reserved(device)
        free = total - cached
        
        return {
            'available': True,
            'total_gb': total / (1024**3),
            'allocated_gb': allocated / (1024**3),
            'cached_gb': cached / (1024**3),
            'free_gb': free / (1024**3),
        }
    except Exception:
        return {'available': False}


def clear_gpu_memory():
    """清理GPU内存"""
    try:
        import torch
        import gc
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ==================== 文件工具 ====================

def get_file_size_mb(filepath: str) -> float:
    """获取文件大小 (MB)"""
    return os.path.getsize(filepath) / (1024 * 1024)


def get_folder_size_mb(folder: str) -> float:
    """获取文件夹大小 (MB)"""
    total = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total / (1024 * 1024)


def ensure_dir(path: str) -> Path:
    """确保目录存在"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ==================== 数据验证 ====================

def validate_image(filepath: str) -> Tuple[bool, str]:
    """验证图像文件"""
    try:
        from PIL import Image
        
        with Image.open(filepath) as img:
            img.verify()
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def count_images_in_folder(folder: str) -> int:
    """统计文件夹中的图像数量"""
    from config import SUPPORTED_IMAGE_FORMATS
    
    count = 0
    folder = Path(folder)
    
    if not folder.exists():
        return 0
    
    for ext in SUPPORTED_IMAGE_FORMATS:
        count += len(list(folder.glob(f'*{ext}')))
        count += len(list(folder.glob(f'*{ext.upper()}')))
    
    return count
