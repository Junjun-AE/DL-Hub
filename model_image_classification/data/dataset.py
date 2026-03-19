"""
数据加载模块 - 使用timm官方预处理
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import timm
from timm.data import create_transform
from PIL import Image


class ImageFolderDataset(datasets.ImageFolder):
    """增强的ImageFolder数据集"""
    
    def __init__(self, root: str, transform=None, **kwargs):
        super().__init__(root, transform=transform, **kwargs)
        self.root_path = root
    
    def get_class_names(self) -> List[str]:
        """获取类别名称列表"""
        return self.classes
    
    def get_class_to_idx(self) -> Dict[str, int]:
        """获取类别到索引的映射"""
        return self.class_to_idx


def get_timm_transforms(
    model_name: str,
    img_size: int = 224,
    is_training: bool = True,
) -> Any:
    """
    获取timm官方预处理变换
    
    Args:
        model_name: timm模型名称
        img_size: 输入图像尺寸
        is_training: 是否用于训练
    
    Returns:
        torchvision transforms
    """
    # 获取模型默认配置
    try:
        data_config = timm.data.resolve_data_config({}, model=model_name)
    except Exception:
        # 如果获取失败，使用默认值
        data_config = {
            'input_size': (3, img_size, img_size),
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
            'interpolation': 'bilinear',
        }
    
    # 更新输入尺寸
    data_config['input_size'] = (3, img_size, img_size)
    
    if is_training:
        # 训练时使用数据增强
        transform = create_transform(
            input_size=data_config['input_size'],
            is_training=True,
            mean=data_config['mean'],
            std=data_config['std'],
            auto_augment='rand-m9-mstd0.5',  # RandAugment
            re_prob=0.25,  # Random Erasing
            re_mode='pixel',
        )
    else:
        # 验证/测试时只做基本预处理
        transform = create_transform(
            input_size=data_config['input_size'],
            is_training=False,
            mean=data_config['mean'],
            std=data_config['std'],
        )
    
    return transform


def create_dataloaders(
    data_path: str,
    model_name: str,
    img_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    创建训练和验证数据加载器
    
    Args:
        data_path: 数据集路径
        model_name: timm模型名称（用于获取预处理配置）
        img_size: 输入图像尺寸
        batch_size: 批次大小
        val_split: 验证集比例（仅当没有train/val分割时使用）
        num_workers: 数据加载线程数
        seed: 随机种子
    
    Returns:
        (train_loader, val_loader, data_info)
    """
    import platform
    
    # Windows上num_workers可能有问题，自动调整
    if platform.system() == 'Windows' and num_workers > 0:
        # Windows上建议使用较少的workers或0
        num_workers = min(num_workers, 2)
    
    path = Path(data_path)
    train_path = path / 'train'
    val_path = path / 'val'
    
    # 获取transforms
    train_transform = get_timm_transforms(model_name, img_size, is_training=True)
    val_transform = get_timm_transforms(model_name, img_size, is_training=False)
    
    # 检查是否已有train/val分割
    if train_path.exists() and val_path.exists():
        # 使用现有分割
        train_dataset = ImageFolderDataset(str(train_path), transform=train_transform)
        val_dataset = ImageFolderDataset(str(val_path), transform=val_transform)
        
        data_info = {
            'has_split': True,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'num_classes': len(train_dataset.classes),
            'class_names': train_dataset.classes,
            'class_to_idx': train_dataset.class_to_idx,
        }
    else:
        # 需要自动分割
        full_dataset = ImageFolderDataset(data_path, transform=None)
        
        # 计算分割大小
        total_size = len(full_dataset)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size
        
        # 设置随机种子并生成随机索引
        torch.manual_seed(seed)
        indices = torch.randperm(total_size).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # 创建分割后的数据集
        train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
        val_dataset = TransformSubset(full_dataset, val_indices, val_transform)
        
        data_info = {
            'has_split': False,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'num_classes': len(full_dataset.classes),
            'class_names': full_dataset.classes,
            'class_to_idx': full_dataset.class_to_idx,
            'val_split_ratio': val_split,
        }
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    return train_loader, val_loader, data_info


class TransformSubset(Dataset):
    """支持不同transform的Subset"""
    
    def __init__(self, dataset: Dataset, indices: List[int], transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    
    def __getitem__(self, idx):
        # 获取原始样本
        original_idx = self.indices[idx]
        img_path, label = self.dataset.samples[original_idx]
        
        # 加载图像
        img = Image.open(img_path).convert('RGB')
        
        # 应用变换
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label
    
    def __len__(self):
        return len(self.indices)
