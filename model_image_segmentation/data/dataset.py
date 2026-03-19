"""
数据集加载模块 - MMSegmentation格式数据集处理
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import random

import numpy as np
from PIL import Image


@dataclass
class DatasetInfo:
    """数据集信息"""
    is_valid: bool
    message: str
    dataset_dir: str = ""
    num_classes: int = 0
    class_names: List[str] = None
    train_images: int = 0
    val_images: int = 0
    ignore_index: int = 255
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = []


def count_files(directory: Path, extensions: List[str]) -> int:
    """统计目录中指定扩展名的文件数量"""
    count = 0
    if directory.exists():
        for ext in extensions:
            count += len(list(directory.glob(f'*{ext}')))
            count += len(list(directory.glob(f'*{ext.upper()}')))
    return count


def validate_mmseg_dataset(dataset_dir: str) -> DatasetInfo:
    """
    验证MMSegmentation格式数据集
    
    Args:
        dataset_dir: 数据集目录路径
    
    Returns:
        DatasetInfo对象
    """
    dataset_path = Path(dataset_dir)
    
    # 检查目录是否存在
    if not dataset_path.exists():
        return DatasetInfo(
            is_valid=False,
            message=f"❌ 数据集目录不存在: {dataset_dir}"
        )
    
    # 检查必要的子目录
    train_images_dir = dataset_path / 'images' / 'train'
    val_images_dir = dataset_path / 'images' / 'val'
    train_masks_dir = dataset_path / 'masks' / 'train'
    val_masks_dir = dataset_path / 'masks' / 'val'
    
    if not train_images_dir.exists():
        return DatasetInfo(
            is_valid=False,
            message=f"❌ 训练集图像目录不存在: {train_images_dir}"
        )
    
    if not val_images_dir.exists():
        return DatasetInfo(
            is_valid=False,
            message=f"❌ 验证集图像目录不存在: {val_images_dir}"
        )
    
    # 统计图像数量
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    train_images = count_files(train_images_dir, image_extensions)
    val_images = count_files(val_images_dir, image_extensions)
    
    if train_images == 0:
        return DatasetInfo(
            is_valid=False,
            message="❌ 训练集中没有图像文件"
        )
    
    if val_images == 0:
        return DatasetInfo(
            is_valid=False,
            message="❌ 验证集中没有图像文件"
        )
    
    # 统计mask数量
    train_masks = count_files(train_masks_dir, ['.png'])
    val_masks = count_files(val_masks_dir, ['.png'])
    
    # 读取类别信息
    class_info_file = dataset_path / 'class_info.txt'
    num_classes = 0
    class_names = []
    
    if class_info_file.exists():
        try:
            with open(class_info_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('# 总类别数'):
                        num_classes = int(line.split(':')[1].strip())
                        class_names = [str(i) for i in range(num_classes)]
                        break
        except Exception:
            pass
    
    # 如果没有class_info.txt，尝试从mask中推断类别数
    if num_classes == 0:
        try:
            # 读取一些mask来统计类别
            mask_files = list(train_masks_dir.glob('*.png'))[:10]
            all_classes = set()
            for mask_file in mask_files:
                mask = np.array(Image.open(mask_file))
                unique_values = np.unique(mask)
                # 排除背景255
                all_classes.update([v for v in unique_values if v != 255])
            
            if all_classes:
                num_classes = max(all_classes) + 1
                class_names = [str(i) for i in range(num_classes)]
        except Exception:
            pass
    
    message = (
        f"✅ 数据集验证通过\n"
        f"   📁 路径: {dataset_dir}\n"
        f"   📊 类别数: {num_classes}\n"
        f"   🏋️ 训练集: {train_images} 张图像, {train_masks} 个mask\n"
        f"   📝 验证集: {val_images} 张图像, {val_masks} 个mask"
    )
    
    return DatasetInfo(
        is_valid=True,
        message=message,
        dataset_dir=str(dataset_path),
        num_classes=num_classes,
        class_names=class_names,
        train_images=train_images,
        val_images=val_images,
    )


def get_val_images(dataset_dir: str, num_images: int = 4) -> List[str]:
    """
    获取验证集图像路径用于推理测试
    
    Args:
        dataset_dir: 数据集目录
        num_images: 返回的图像数量
    
    Returns:
        图像路径列表
    """
    dataset_path = Path(dataset_dir)
    val_images_dir = dataset_path / 'images' / 'val'
    
    if not val_images_dir.exists():
        return []
    
    # 收集所有图像
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    images = []
    for ext in image_extensions:
        images.extend(val_images_dir.glob(f'*{ext}'))
        images.extend(val_images_dir.glob(f'*{ext.upper()}'))
    
    if len(images) == 0:
        return []
    
    # 随机选择
    selected = random.sample(images, min(num_images, len(images)))
    return [str(img) for img in selected]


def get_val_image_mask_pairs(dataset_dir: str, num_samples: int = 4) -> List[Tuple[str, str]]:
    """
    获取验证集图像-mask配对
    
    Args:
        dataset_dir: 数据集目录
        num_samples: 返回的样本数量
    
    Returns:
        [(image_path, mask_path), ...]
    """
    dataset_path = Path(dataset_dir)
    val_images_dir = dataset_path / 'images' / 'val'
    val_masks_dir = dataset_path / 'masks' / 'val'
    
    if not val_images_dir.exists() or not val_masks_dir.exists():
        return []
    
    pairs = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    
    for ext in image_extensions:
        for img_file in val_images_dir.glob(f'*{ext}'):
            mask_file = val_masks_dir / f"{img_file.stem}.png"
            if mask_file.exists():
                pairs.append((str(img_file), str(mask_file)))
        for img_file in val_images_dir.glob(f'*{ext.upper()}'):
            mask_file = val_masks_dir / f"{img_file.stem}.png"
            if mask_file.exists():
                pairs.append((str(img_file), str(mask_file)))
    
    if len(pairs) == 0:
        return []
    
    selected = random.sample(pairs, min(num_samples, len(pairs)))
    return selected


def get_class_names(dataset_dir: str) -> Dict[int, str]:
    """
    从数据集获取类别名称
    
    Args:
        dataset_dir: 数据集目录
    
    Returns:
        类别ID到名称的映射
    """
    dataset_path = Path(dataset_dir)
    class_info_file = dataset_path / 'class_info.txt'
    
    if class_info_file.exists():
        try:
            num_classes = 0
            with open(class_info_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('# 总类别数'):
                        num_classes = int(line.split(':')[1].strip())
                        break
            
            if num_classes > 0:
                return {i: str(i) for i in range(num_classes)}
        except Exception:
            pass
    
    return {}


def analyze_class_distribution(dataset_dir: str, split: str = 'train') -> Dict[int, int]:
    """
    分析数据集类别分布（像素数统计）
    
    Args:
        dataset_dir: 数据集目录
        split: 'train' 或 'val'
    
    Returns:
        类别ID到像素数的映射
    """
    dataset_path = Path(dataset_dir)
    masks_dir = dataset_path / 'masks' / split
    
    if not masks_dir.exists():
        return {}
    
    class_pixels = {}
    
    for mask_file in masks_dir.glob('*.png'):
        try:
            mask = np.array(Image.open(mask_file))
            unique, counts = np.unique(mask, return_counts=True)
            
            for cls_id, count in zip(unique, counts):
                if cls_id == 255:  # 跳过背景
                    continue
                cls_id = int(cls_id)
                if cls_id not in class_pixels:
                    class_pixels[cls_id] = 0
                class_pixels[cls_id] += int(count)
        except Exception:
            continue
    
    return class_pixels


def compute_class_weights(dataset_dir: str, num_classes: int, split: str = 'train') -> List[float]:
    """
    计算类别权重（用于处理类别不平衡）
    
    Args:
        dataset_dir: 数据集目录
        num_classes: 类别数量
        split: 使用哪个split计算
    
    Returns:
        类别权重列表
    """
    class_pixels = analyze_class_distribution(dataset_dir, split)
    
    if not class_pixels:
        return [1.0] * num_classes
    
    total_pixels = sum(class_pixels.values())
    
    weights = []
    for i in range(num_classes):
        pixels = class_pixels.get(i, 0)
        if pixels > 0:
            # 反比例权重
            weight = total_pixels / (num_classes * pixels)
        else:
            weight = 1.0
        weights.append(weight)
    
    # 归一化，使平均权重为1
    avg_weight = sum(weights) / len(weights)
    weights = [w / avg_weight for w in weights]
    
    return weights
