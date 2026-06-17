"""
数据集加载模块 - YOLO格式数据集处理
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class DatasetInfo:
    """数据集信息"""
    is_valid: bool
    message: str
    data_yaml: str = ""
    num_classes: int = 0
    class_names: List[str] = None
    train_images: int = 0
    val_images: int = 0
    train_labels: int = 0
    val_labels: int = 0
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = []


def count_files(directory: Path, extensions: List[str]) -> int:
    """
    统计目录中指定扩展名的文件数量
    
    Args:
        directory: 目录路径
        extensions: 扩展名列表
    
    Returns:
        文件数量
    """
    count = 0
    if directory.exists():
        for ext in extensions:
            count += len(list(directory.glob(f'*{ext}')))
            count += len(list(directory.glob(f'*{ext.upper()}')))
    return count


def validate_yolo_dataset(data_yaml: str) -> DatasetInfo:
    """
    验证YOLO格式数据集
    
    Args:
        data_yaml: data.yaml文件路径
    
    Returns:
        DatasetInfo对象
    """
    yaml_path = Path(data_yaml)
    
    # 检查yaml文件是否存在
    if not yaml_path.exists():
        return DatasetInfo(
            is_valid=False,
            message=f"❌ data.yaml不存在: {data_yaml}"
        )
    
    # 读取yaml
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        return DatasetInfo(
            is_valid=False,
            message=f"❌ 读取data.yaml失败: {str(e)}"
        )
    
    # 获取数据集根目录
    dataset_path = Path(config.get('path', yaml_path.parent))
    if not dataset_path.is_absolute():
        dataset_path = yaml_path.parent / dataset_path
    
    # 获取训练集和验证集路径
    train_path = dataset_path / config.get('train', 'images/train')
    val_path = dataset_path / config.get('val', 'images/val')
    
    # 检查训练集
    if not train_path.exists():
        return DatasetInfo(
            is_valid=False,
            message=f"❌ 训练集路径不存在: {train_path}"
        )
    
    # 检查验证集
    if not val_path.exists():
        return DatasetInfo(
            is_valid=False,
            message=f"❌ 验证集路径不存在: {val_path}"
        )
    
    # 统计图像数量
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    train_images = count_files(train_path, image_extensions)
    val_images = count_files(val_path, image_extensions)
    
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
    
    # 统计标签数量
    train_labels_path = dataset_path / 'labels' / 'train'
    val_labels_path = dataset_path / 'labels' / 'val'
    
    train_labels = count_files(train_labels_path, ['.txt'])
    val_labels = count_files(val_labels_path, ['.txt'])
    
    # 获取类别信息
    num_classes = config.get('nc', 0)
    names = config.get('names', {})
    
    if isinstance(names, dict):
        class_names = [str(names.get(i, str(i))) for i in range(num_classes)]
    elif isinstance(names, list):
        class_names = [str(n) for n in names]
    else:
        class_names = [str(i) for i in range(num_classes)]
    
    # 构建成功消息
    message = (
        f"✅ 数据集验证通过\n"
        f"   📁 路径: {dataset_path}\n"
        f"   📊 类别数: {num_classes}\n"
        f"   🏋️ 训练集: {train_images} 张图像, {train_labels} 个标签\n"
        f"   📝 验证集: {val_images} 张图像, {val_labels} 个标签\n"
        f"   📋 类别: {', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''}"
    )
    
    return DatasetInfo(
        is_valid=True,
        message=message,
        data_yaml=str(yaml_path),
        num_classes=num_classes,
        class_names=class_names,
        train_images=train_images,
        val_images=val_images,
        train_labels=train_labels,
        val_labels=val_labels,
    )


def get_val_images(data_yaml: str, num_images: int = 4) -> List[str]:
    """
    获取验证集图像路径用于推理测试
    
    Args:
        data_yaml: data.yaml文件路径
        num_images: 返回的图像数量
    
    Returns:
        图像路径列表
    """
    import random
    
    yaml_path = Path(data_yaml)
    
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception:
        return []
    
    # 获取验证集路径
    dataset_path = Path(config.get('path', yaml_path.parent))
    if not dataset_path.is_absolute():
        dataset_path = yaml_path.parent / dataset_path
    
    val_path = dataset_path / config.get('val', 'images/val')
    
    if not val_path.exists():
        return []
    
    # 收集所有图像
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    images = []
    for ext in image_extensions:
        images.extend(val_path.glob(f'*{ext}'))
        images.extend(val_path.glob(f'*{ext.upper()}'))
    
    if len(images) == 0:
        return []
    
    # 随机选择
    selected = random.sample(images, min(num_images, len(images)))
    return [str(img) for img in selected]


def get_class_names(data_yaml: str) -> Dict[int, str]:
    """
    从data.yaml获取类别名称
    
    Args:
        data_yaml: data.yaml文件路径
    
    Returns:
        类别ID到名称的映射
    """
    yaml_path = Path(data_yaml)
    
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception:
        return {}
    
    names = config.get('names', {})
    num_classes = config.get('nc', 0)
    
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    elif isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    else:
        return {i: str(i) for i in range(num_classes)}
