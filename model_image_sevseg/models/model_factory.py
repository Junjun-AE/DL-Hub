"""
SevSeg-YOLO 模型工厂
预训练权重管理 + 模型创建
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from config.model_registry import MODEL_REGISTRY, get_model_config


def get_pretrained_dir() -> Path:
    """
    获取预训练权重目录路径

    查找顺序:
    1. Deep_learning_tools4/pretrained_model/image_sevseg
    2. model_image_sevseg/models/pretrained
    """
    current_dir = Path(__file__).parent  # model_image_sevseg/models/

    possible_paths = [
        current_dir.parent.parent / 'pretrained_model' / 'image_sevseg',
        current_dir.parent.parent.parent / 'pretrained_model' / 'image_sevseg',
        Path.cwd() / 'pretrained_model' / 'image_sevseg',
        current_dir / 'pretrained',
    ]

    for path in possible_paths:
        if path.exists() and any(path.glob('*.pt')):
            print(f"📂 预训练模型目录: {path}")
            return path

    default_path = current_dir.parent.parent / 'pretrained_model' / 'image_sevseg'
    default_path.mkdir(parents=True, exist_ok=True)
    print(f"📂 预训练模型目录 (新建): {default_path}")
    print(f"   💡 可将 yolo26n/s/m/l/x.pt 放入此目录避免下载")
    return default_path


def get_pretrained_path(model_name: str, weights_name: str = None) -> Optional[Path]:
    """
    获取本地预训练权重路径

    Args:
        model_name: 模型名称 (如 'yolo26n-score')
        weights_name: 权重文件名 (如 'yolo26n.pt')

    Returns:
        权重文件路径，不存在返回None
    """
    pretrained_dir = get_pretrained_dir()

    # 可能的文件名
    possible_names = []
    if weights_name:
        possible_names.append(weights_name)
    possible_names.extend([
        f"{model_name}.pt",
        f"{model_name.replace('-score', '')}.pt",
    ])

    # 也检查不带score后缀的基础权重
    base_name = model_name.replace('-score', '')
    possible_names.append(f"{base_name}.pt")

    for name in possible_names:
        path = pretrained_dir / name
        if path.exists():
            return path

    return None


def cache_pretrained_weights(model_name: str, source_path: str) -> Path:
    """缓存预训练权重到本地"""
    pretrained_dir = get_pretrained_dir()
    cache_path = pretrained_dir / f"{model_name}.pt"

    if not cache_path.exists():
        shutil.copy(source_path, cache_path)
        print(f"💾 权重已缓存: {cache_path}")

    return cache_path


def get_model_yaml_path(scale: str) -> Optional[Path]:
    """
    获取模型YAML配置文件的绝对路径

    Args:
        scale: 模型规模

    Returns:
        YAML文件的绝对路径
    """
    config = get_model_config(scale)
    if config is None:
        return None

    # model_yaml 是相对于 model_image_sevseg/ 的路径
    sevseg_dir = Path(__file__).parent.parent  # model_image_sevseg/
    yaml_path = sevseg_dir / config['model_yaml']

    if yaml_path.exists():
        return yaml_path

    return None


def check_pretrained_status(scale: str) -> str:
    """
    检查预训练权重状态

    Args:
        scale: 模型规模

    Returns:
        状态描述字符串
    """
    config = get_model_config(scale)
    if config is None:
        return "❌ 未找到模型配置"

    model_name = config['name']
    weights_name = config['pretrained_weights']
    local_path = get_pretrained_path(model_name, weights_name)

    if local_path:
        return f"✅ 已缓存: {local_path.name}"
    else:
        pretrained_dir = get_pretrained_dir()
        return (
            f"📥 首次使用将从 ultralytics 自动下载\n"
            f"   💡 或手动放置 {weights_name} 到:\n"
            f"   {pretrained_dir}"
        )
