"""
数据集验证器 - 验证ImageFolder格式数据集

[Bug-7修复] 添加中文路径支持
"""

import os
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass

# [Bug-7修复] Windows系统中文路径支持
if sys.platform == 'win32':
    try:
        import locale
        # 设置locale以支持中文
        locale.setlocale(locale.LC_ALL, '')
    except Exception:
        pass
    
    # 确保stdout/stderr支持Unicode
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass


# 支持的图像格式
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif'}


@dataclass
class DatasetInfo:
    """数据集信息"""
    is_valid: bool
    message: str
    path: str
    num_classes: int = 0
    class_names: List[str] = None
    total_images: int = 0
    images_per_class: Dict[str, int] = None
    has_train_val_split: bool = False
    train_images: int = 0
    val_images: int = 0
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = []
        if self.images_per_class is None:
            self.images_per_class = {}


def is_image_file(filename: str) -> bool:
    """检查文件是否是支持的图像格式"""
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


def count_images_in_folder(folder_path: str) -> Tuple[int, Dict[str, int]]:
    """
    统计文件夹中的图像数量
    
    [Bug-7修复] 增强中文路径处理
    
    Returns:
        (total_count, class_counts): 总数和每个类别的数量
    """
    class_counts = {}
    total = 0
    
    try:
        # [Bug-7修复] 确保路径编码正确
        if isinstance(folder_path, bytes):
            folder_path = folder_path.decode('utf-8', errors='replace')
        
        folder = Path(folder_path)
        if not folder.exists():
            return 0, {}
        
        for class_dir in sorted(folder.iterdir()):
            try:
                if class_dir.is_dir() and not class_dir.name.startswith('.'):
                    count = 0
                    for f in class_dir.iterdir():
                        try:
                            if f.is_file() and is_image_file(f.name):
                                count += 1
                        except (OSError, UnicodeDecodeError) as e:
                            # [Bug-7修复] 跳过无法处理的文件
                            print(f"[DataValidator] 跳过文件 {f}: {e}")
                            continue
                    
                    if count > 0:
                        class_counts[class_dir.name] = count
                        total += count
            except (OSError, UnicodeDecodeError) as e:
                # [Bug-7修复] 跳过无法处理的目录
                print(f"[DataValidator] 跳过目录 {class_dir}: {e}")
                continue
                
    except Exception as e:
        print(f"[DataValidator] 统计图像时出错: {e}")
    
    return total, class_counts


def validate_dataset(data_path: str) -> DatasetInfo:
    """
    验证数据集
    
    支持两种结构:
    1. 单文件夹 (需要用val_split划分):
       data_path/
       ├── class_a/
       │   ├── img1.jpg
       │   └── img2.jpg
       └── class_b/
           └── img3.jpg
    
    2. train/val分开的结构:
       data_path/
       ├── train/
       │   ├── class_a/
       │   └── class_b/
       └── val/
           ├── class_a/
           └── class_b/
    
    Args:
        data_path: 数据集路径
    
    Returns:
        DatasetInfo对象
    """
    path = Path(data_path)
    
    # 检查路径是否存在
    if not path.exists():
        return DatasetInfo(
            is_valid=False,
            message=f"❌ 路径不存在: {data_path}",
            path=data_path,
        )
    
    if not path.is_dir():
        return DatasetInfo(
            is_valid=False,
            message=f"❌ 路径不是文件夹: {data_path}",
            path=data_path,
        )
    
    # 检查是否是 train/val 分开的结构
    train_path = path / 'train'
    val_path = path / 'val'
    
    if train_path.exists() and val_path.exists():
        # train/val 分开的结构
        return _validate_split_dataset(data_path, train_path, val_path)
    else:
        # 单文件夹结构
        return _validate_single_folder(data_path)


def _validate_single_folder(data_path: str) -> DatasetInfo:
    """验证单文件夹结构的数据集"""
    path = Path(data_path)
    
    # 获取所有子文件夹（类别）
    subdirs = [d for d in path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if len(subdirs) == 0:
        return DatasetInfo(
            is_valid=False,
            message="❌ 未找到类别文件夹，请确保数据集为ImageFolder格式",
            path=data_path,
        )
    
    # 统计图像
    total, class_counts = count_images_in_folder(data_path)
    
    if total == 0:
        return DatasetInfo(
            is_valid=False,
            message="❌ 未找到任何图像文件",
            path=data_path,
        )
    
    # 检查类别数量
    class_names = sorted(class_counts.keys())
    num_classes = len(class_names)
    
    if num_classes < 2:
        return DatasetInfo(
            is_valid=False,
            message=f"❌ 至少需要2个类别，当前只有{num_classes}个",
            path=data_path,
            num_classes=num_classes,
            class_names=class_names,
        )
    
    # 检查是否有空类别
    empty_classes = [name for name, count in class_counts.items() if count == 0]
    if empty_classes:
        return DatasetInfo(
            is_valid=False,
            message=f"❌ 以下类别没有图像: {', '.join(empty_classes)}",
            path=data_path,
            num_classes=num_classes,
            class_names=class_names,
        )
    
    # 检查类别是否平衡（警告，不阻止）
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    warning = ""
    if imbalance_ratio > 10:
        warning = f"\n⚠️ 类别不平衡 (最大/最小 = {imbalance_ratio:.1f}倍)，建议数据增强"
    
    # 构建成功消息
    message = (
        f"✅ 数据集验证通过\n"
        f"   📁 路径: {data_path}\n"
        f"   📊 类别数: {num_classes}\n"
        f"   🖼️ 图像总数: {total}\n"
        f"   📋 类别: {', '.join(class_names[:5])}{'...' if num_classes > 5 else ''}\n"
        f"   ℹ️ 将使用 val_split 参数自动划分验证集"
        f"{warning}"
    )
    
    return DatasetInfo(
        is_valid=True,
        message=message,
        path=data_path,
        num_classes=num_classes,
        class_names=class_names,
        total_images=total,
        images_per_class=class_counts,
        has_train_val_split=False,
    )


def _validate_split_dataset(data_path: str, train_path: Path, val_path: Path) -> DatasetInfo:
    """验证 train/val 分开的数据集"""
    
    # 统计训练集
    train_total, train_counts = count_images_in_folder(str(train_path))
    
    # 统计验证集
    val_total, val_counts = count_images_in_folder(str(val_path))
    
    # 检查是否有数据
    if train_total == 0:
        return DatasetInfo(
            is_valid=False,
            message="❌ 训练集为空",
            path=data_path,
        )
    
    if val_total == 0:
        return DatasetInfo(
            is_valid=False,
            message="❌ 验证集为空",
            path=data_path,
        )
    
    # 检查类别是否一致
    train_classes = set(train_counts.keys())
    val_classes = set(val_counts.keys())
    
    if train_classes != val_classes:
        missing_in_val = train_classes - val_classes
        missing_in_train = val_classes - train_classes
        
        msg = "❌ 训练集和验证集的类别不一致"
        if missing_in_val:
            msg += f"\n   验证集缺少: {', '.join(missing_in_val)}"
        if missing_in_train:
            msg += f"\n   训练集缺少: {', '.join(missing_in_train)}"
        
        return DatasetInfo(
            is_valid=False,
            message=msg,
            path=data_path,
        )
    
    class_names = sorted(train_classes)
    num_classes = len(class_names)
    
    if num_classes < 2:
        return DatasetInfo(
            is_valid=False,
            message=f"❌ 至少需要2个类别，当前只有{num_classes}个",
            path=data_path,
        )
    
    # 构建成功消息
    message = (
        f"✅ 数据集验证通过\n"
        f"   📁 路径: {data_path}\n"
        f"   📊 类别数: {num_classes}\n"
        f"   🏋️ 训练集: {train_total} 张\n"
        f"   📝 验证集: {val_total} 张\n"
        f"   📋 类别: {', '.join(class_names[:5])}{'...' if num_classes > 5 else ''}"
    )
    
    # 合并统计
    total_counts = {}
    for name in class_names:
        total_counts[name] = train_counts.get(name, 0) + val_counts.get(name, 0)
    
    return DatasetInfo(
        is_valid=True,
        message=message,
        path=data_path,
        num_classes=num_classes,
        class_names=class_names,
        total_images=train_total + val_total,
        images_per_class=total_counts,
        has_train_val_split=True,
        train_images=train_total,
        val_images=val_total,
    )
