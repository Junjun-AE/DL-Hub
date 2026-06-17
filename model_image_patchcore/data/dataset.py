# -*- coding: utf-8 -*-
"""
数据集模块

支持标准的异常检测数据集格式
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # 创建占位基类
    class Dataset:
        pass

try:
    import torchvision.transforms as T
except ImportError:
    T = None


class AnomalyDataset(Dataset):
    """
    异常检测数据集
    
    支持的目录结构:
    
    方式1 (MVTec格式):
    root_dir/
    ├── train/
    │   └── good/
    │       └── *.png
    └── test/
        ├── good/
        │   └── *.png
        └── defect_type/
            └── *.png
    
    方式2 (简单格式):
    root_dir/
    ├── good/
    │   └── *.png
    └── defect/  (可选)
        └── *.png
    """
    
    def __init__(
        self,
        root_dir: str,
        image_size: int = 256,
        split: str = 'train',  # 'train', 'test', 'all'
        transform: Optional[Callable] = None,
    ):
        """
        初始化数据集
        
        Args:
            root_dir: 数据集根目录
            image_size: 图像尺寸
            split: 数据划分 ('train', 'test', 'all')
            transform: 自定义转换
        """
        if not TORCH_AVAILABLE:
            raise ImportError("需要安装PyTorch: pip install torch torchvision")
        
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.split = split
        
        # 设置转换
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transform()
        
        # 加载样本
        self.samples = self._load_samples()
    
    def _get_default_transform(self):
        """获取默认转换"""
        from config import IMAGENET_MEAN, IMAGENET_STD
        
        return T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    
    def _load_samples(self) -> List[Dict]:
        """加载样本列表"""
        from config import SUPPORTED_IMAGE_FORMATS
        
        samples = []
        
        # 检测目录结构
        if (self.root_dir / 'train').exists():
            # MVTec格式
            samples = self._load_mvtec_format()
        else:
            # 简单格式
            samples = self._load_simple_format()
        
        return samples
    
    def _load_mvtec_format(self) -> List[Dict]:
        """加载MVTec格式数据"""
        from config import SUPPORTED_IMAGE_FORMATS
        samples = []
        
        if self.split in ['train', 'all']:
            train_good_dir = self.root_dir / 'train' / 'good'
            if train_good_dir.exists():
                for img_path in sorted(train_good_dir.iterdir()):
                    if img_path.suffix.lower() in SUPPORTED_IMAGE_FORMATS:
                        samples.append({
                            'image_path': str(img_path),
                            'label': 0,  # 良品
                            'label_name': 'good',
                            'split': 'train',
                        })
        
        if self.split in ['test', 'all']:
            test_dir = self.root_dir / 'test'
            if test_dir.exists():
                for defect_dir in sorted(test_dir.iterdir()):
                    if not defect_dir.is_dir():
                        continue
                    
                    label = 0 if defect_dir.name == 'good' else 1
                    
                    for img_path in sorted(defect_dir.iterdir()):
                        if img_path.suffix.lower() in SUPPORTED_IMAGE_FORMATS:
                            samples.append({
                                'image_path': str(img_path),
                                'label': label,
                                'label_name': defect_dir.name,
                                'split': 'test',
                            })
        
        return samples
    
    def _load_simple_format(self) -> List[Dict]:
        """加载简单格式数据"""
        from config import SUPPORTED_IMAGE_FORMATS
        samples = []
        
        # good 目录
        good_dir = self.root_dir / 'good'
        if good_dir.exists():
            for img_path in sorted(good_dir.iterdir()):
                if img_path.suffix.lower() in SUPPORTED_IMAGE_FORMATS:
                    samples.append({
                        'image_path': str(img_path),
                        'label': 0,
                        'label_name': 'good',
                        'split': 'train' if self.split == 'train' else 'all',
                    })
        
        # defect 目录 (可选)
        defect_dirs = ['defect', 'bad', 'anomaly', 'ng']
        for defect_name in defect_dirs:
            defect_dir = self.root_dir / defect_name
            if defect_dir.exists():
                for img_path in sorted(defect_dir.iterdir()):
                    if img_path.suffix.lower() in SUPPORTED_IMAGE_FORMATS:
                        samples.append({
                            'image_path': str(img_path),
                            'label': 1,
                            'label_name': defect_name,
                            'split': 'test',
                        })
        
        # 如果只要训练集，过滤掉异常样本
        if self.split == 'train':
            samples = [s for s in samples if s['label'] == 0]
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # 加载图像
        image = Image.open(sample['image_path']).convert('RGB')
        
        # 应用转换
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'label': sample['label'],
            'label_name': sample['label_name'],
            'image_path': sample['image_path'],
        }
    
    def get_statistics(self) -> Dict:
        """获取数据集统计信息"""
        good_count = sum(1 for s in self.samples if s['label'] == 0)
        defect_count = sum(1 for s in self.samples if s['label'] == 1)
        
        label_names = set(s['label_name'] for s in self.samples)
        
        return {
            'total': len(self.samples),
            'good': good_count,
            'defect': defect_count,
            'label_names': sorted(label_names),
        }


class InferenceDataset(Dataset):
    """推理数据集（不需要标签）"""
    
    def __init__(
        self,
        image_paths: List[str],
        image_size: int = 256,
        transform: Optional[Callable] = None,
    ):
        self.image_paths = image_paths
        self.image_size = image_size
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transform()
    
    def _get_default_transform(self):
        from config import IMAGENET_MEAN, IMAGENET_STD
        
        return T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'image_path': image_path,
        }


def scan_image_directory(directory: str) -> List[str]:
    """扫描目录下的所有图像文件"""
    from config import SUPPORTED_IMAGE_FORMATS
    
    directory = Path(directory)
    image_paths = []
    
    for ext in SUPPORTED_IMAGE_FORMATS:
        image_paths.extend(directory.glob(f'*{ext}'))
        image_paths.extend(directory.glob(f'*{ext.upper()}'))
    
    return sorted([str(p) for p in image_paths])


def validate_dataset_structure(root_dir: str) -> Tuple[bool, str, Dict]:
    """
    验证数据集结构
    
    Returns:
        (is_valid, message, statistics)
    """
    from config import SUPPORTED_IMAGE_FORMATS
    
    root_dir = Path(root_dir)
    
    if not root_dir.exists():
        return False, f"目录不存在: {root_dir}", {}
    
    # 检测格式
    if (root_dir / 'train').exists():
        format_type = 'mvtec'
        good_dir = root_dir / 'train' / 'good'
    elif (root_dir / 'good').exists():
        format_type = 'simple'
        good_dir = root_dir / 'good'
    else:
        return False, "未找到 'train/good' 或 'good' 目录", {}
    
    # 统计图像数量 (使用set避免Windows大小写不敏感导致的重复)
    good_files = set()
    for ext in SUPPORTED_IMAGE_FORMATS:
        # 只搜索小写扩展名，glob在Windows上大小写不敏感会自动匹配
        for p in good_dir.glob(f'*{ext}'):
            good_files.add(str(p).lower())  # 用小写路径去重
    
    good_count = len(good_files)
    
    if good_count == 0:
        return False, f"良品目录为空: {good_dir}", {}
    
    # 统计异常样本 (同样使用set去重)
    defect_files = set()
    defect_types = []
    
    if format_type == 'mvtec':
        test_dir = root_dir / 'test'
        if test_dir.exists():
            for defect_dir in test_dir.iterdir():
                if defect_dir.is_dir() and defect_dir.name != 'good':
                    dir_files = set()
                    for ext in SUPPORTED_IMAGE_FORMATS:
                        for p in defect_dir.glob(f'*{ext}'):
                            dir_files.add(str(p).lower())
                    if dir_files:
                        defect_files.update(dir_files)
                        defect_types.append(defect_dir.name)
    else:
        for defect_name in ['defect', 'bad', 'anomaly', 'ng']:
            defect_dir = root_dir / defect_name
            if defect_dir.exists():
                for ext in SUPPORTED_IMAGE_FORMATS:
                    for p in defect_dir.glob(f'*{ext}'):
                        defect_files.add(str(p).lower())
                if defect_dir.exists():
                    defect_types.append(defect_name)
    
    defect_count = len(defect_files)
    
    statistics = {
        'format': format_type,
        'good_count': good_count,
        'defect_count': defect_count,
        'defect_types': defect_types,
        'total': good_count + defect_count,
    }
    
    message = f"数据集有效 ({format_type}格式): {good_count}张良品"
    if defect_count > 0:
        message += f", {defect_count}张异常"
    
    return True, message, statistics
