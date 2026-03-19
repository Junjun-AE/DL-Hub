"""
模型工厂 - 创建和配置YOLO模型
支持本地预训练权重缓存和ultralytics自动下载
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import torch

from config.model_registry import MODEL_REGISTRY, get_model_config


def get_pretrained_dir() -> Path:
    """
    获取预训练权重目录路径
    
    查找顺序:
    1. Deep_learning_tools3/pretrained_model/image_detection (如果从Deep_learning_tools3运行)
    2. Deep_learning_tools/pretrained_model/image_detection
    3. 当前目录下的 pretrained_model/image_detection
    4. model_image_detection/models/pretrained (作为备用)
    """
    current_dir = Path(__file__).parent  # model_image_detection/models/
    
    # 可能的预训练目录路径（按优先级排列）
    possible_paths = [
        # 向上3级 (如果是 Deep_learning_tools3/model_image_detection/models/)
        current_dir.parent.parent / 'pretrained_model' / 'image_detection',
        # 向上2级再找 pretrained_model
        current_dir.parent.parent.parent / 'pretrained_model' / 'image_detection',
        # 当前工作目录下
        Path.cwd() / 'pretrained_model' / 'image_detection',
        # model_image_detection 同级目录
        current_dir.parent / 'pretrained',
        # models目录下的pretrained
        current_dir / 'pretrained',
    ]
    
    # 查找存在的目录
    for path in possible_paths:
        if path.exists() and any(path.glob('*.pt')):
            print(f"📂 预训练模型目录: {path}")
            return path
    
    # 如果都不存在，使用默认路径并创建
    default_path = current_dir.parent.parent / 'pretrained_model' / 'image_detection'
    default_path.mkdir(parents=True, exist_ok=True)
    print(f"📂 预训练模型目录 (未找到已有模型，使用默认): {default_path}")
    print(f"   请将预训练模型 (.pt文件) 放入此目录")
    return default_path


def get_pretrained_path(model_name: str, ultralytics_name: str = None) -> Optional[Path]:
    """
    获取本地预训练权重路径
    
    Args:
        model_name: 模型名称，如 'yolov8n'
        ultralytics_name: ultralytics官方名称，如 'yolov8n.pt'
    
    Returns:
        权重文件路径，如果不存在返回None
    """
    pretrained_dir = get_pretrained_dir()
    
    # 检查不同的文件命名格式
    possible_names = [
        f"{model_name}.pt",
        f"{model_name}u.pt",  # ultralytics格式 (yolov5nu.pt)
        f"{model_name}_pretrained.pt",
        f"{model_name.lower()}.pt",  # 小写
        f"{model_name.upper()}.pt",  # 大写
    ]
    
    # 如果提供了 ultralytics_name，也加入查找列表
    if ultralytics_name:
        possible_names.insert(0, ultralytics_name)  # 优先检查
        # 去掉.pt后缀的变体
        base_name = ultralytics_name.replace('.pt', '')
        possible_names.append(f"{base_name}.pt")
    
    print(f"🔍 查找预训练模型: {model_name}")
    print(f"   目录: {pretrained_dir}")
    
    for name in possible_names:
        path = pretrained_dir / name
        if path.exists():
            print(f"   ✅ 找到: {path}")
            return path
    
    # 列出目录中的所有.pt文件
    existing_files = list(pretrained_dir.glob('*.pt'))
    if existing_files:
        print(f"   ⚠️ 未找到匹配的模型，目录中现有文件:")
        for f in existing_files:
            print(f"      - {f.name}")
        print(f"   💡 请确保文件名为以下之一: {possible_names[:3]}")
    else:
        print(f"   ⚠️ 目录中没有.pt文件，将从网络下载")
    
    return None


def cache_pretrained_weights(model_name: str, source_path: str) -> Path:
    """
    缓存预训练权重到本地
    
    Args:
        model_name: 模型名称
        source_path: 源权重文件路径
    
    Returns:
        缓存后的路径
    """
    pretrained_dir = get_pretrained_dir()
    cache_path = pretrained_dir / f"{model_name}.pt"
    
    if not cache_path.exists():
        shutil.copy(source_path, cache_path)
    
    return cache_path


def create_yolo_model(
    model_family: str,
    model_scale: str,
    num_classes: Optional[int] = None,
    pretrained: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """
    创建YOLO模型
    
    Args:
        model_family: 模型系列 ('YOLOv5', 'YOLOv8', 'YOLOv11')
        model_scale: 模型规模 ('超小', '小', '中', '大', '超大')
        num_classes: 分类数量（如果为None，则在训练时自动从data.yaml获取）
        pretrained: 是否使用预训练权重
    
    Returns:
        (yolo_model, model_info): YOLO模型对象和模型信息字典
    """
    from ultralytics import YOLO
    
    # 获取模型配置
    config = get_model_config(model_family, model_scale)
    if config is None:
        raise ValueError(f"未找到模型配置: {model_family} - {model_scale}")
    
    model_name = config['name']
    ultralytics_name = config['ultralytics_name']
    
    # 检查本地缓存（传入ultralytics_name以支持更多命名格式）
    local_path = get_pretrained_path(model_name, ultralytics_name)
    
    if pretrained and local_path is not None:
        # 使用本地缓存的权重
        print(f"📂 使用本地预训练权重: {local_path}")
        model = YOLO(str(local_path))
        weights_source = f"本地缓存: {local_path.name}"
    else:
        # 从ultralytics下载
        print(f"📥 加载预训练模型: {ultralytics_name}")
        model = YOLO(ultralytics_name)
        weights_source = f"ultralytics: {ultralytics_name}"
        
        # 尝试缓存到本地
        if pretrained:
            try:
                # ultralytics会自动下载到 ~/.cache/ultralytics 或当前目录
                # 我们将其复制到本地pretrained目录
                downloaded_path = Path(ultralytics_name)
                if downloaded_path.exists():
                    cache_pretrained_weights(model_name, str(downloaded_path))
                    print(f"💾 权重已缓存到本地")
            except Exception as e:
                print(f"⚠️ 缓存权重失败: {e}")
    
    # 构建模型信息
    model_info = {
        'model_name': model_name,
        'model_family': model_family,
        'model_scale': model_scale,
        'ultralytics_name': ultralytics_name,
        'num_classes': num_classes,
        'params_m': config['params'],
        'mAP50': config['mAP50'],
        'mAP50_95': config['mAP50_95'],
        'weights_source': weights_source,
        'input_size': config['input_size'],
    }
    
    return model, model_info


def get_model_summary(model) -> Dict[str, Any]:
    """
    获取模型摘要信息
    
    Args:
        model: YOLO模型对象
    
    Returns:
        包含参数量等信息的字典
    """
    try:
        # ultralytics模型
        if hasattr(model, 'model'):
            pytorch_model = model.model
        else:
            pytorch_model = model
        
        total_params = sum(p.numel() for p in pytorch_model.parameters())
        trainable_params = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'total_params_m': total_params / 1e6,
            'trainable_params_m': trainable_params / 1e6,
        }
    except Exception as e:
        return {
            'total_params': 0,
            'trainable_params': 0,
            'total_params_m': 0,
            'trainable_params_m': 0,
            'error': str(e),
        }


def check_pretrained_status(model_family: str, model_scale: str) -> str:
    """
    检查预训练权重状态
    
    Args:
        model_family: 模型系列
        model_scale: 模型规模
    
    Returns:
        状态描述字符串
    """
    config = get_model_config(model_family, model_scale)
    if config is None:
        return "❌ 未找到模型配置"
    
    model_name = config['name']
    ultralytics_name = config['ultralytics_name']
    local_path = get_pretrained_path(model_name, ultralytics_name)
    
    if local_path:
        return f"✅ 已缓存: {local_path.name}"
    else:
        pretrained_dir = get_pretrained_dir()
        return f"📥 首次使用将从ultralytics下载\n   💡 或手动放置到: {pretrained_dir}"