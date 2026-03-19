"""
模型工厂 - 创建和配置SegFormer模型
支持本地预训练权重缓存和自动下载
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch

from config.model_registry import MODEL_REGISTRY, get_model_config, IMAGENET_MEAN, IMAGENET_STD


def get_pretrained_dir() -> Path:
    """获取预训练权重目录路径
    
    路径优先级：
    1. Deep_learning_tools/pretrained_model/images_segmentation (推荐)
    2. 当前模块目录下的 pretrained 文件夹 (兼容旧版)
    """
    # 优先使用统一的预训练模型目录
    current_file = Path(__file__).resolve()
    # 向上查找 Deep_learning_tools 目录
    for parent in current_file.parents:
        unified_dir = parent / 'pretrained_model' / 'images_segmentation'
        if parent.name == 'Deep_learning_tools' or unified_dir.exists():
            unified_dir.mkdir(parents=True, exist_ok=True)
            return unified_dir
    
    # 回退到旧路径
    current_dir = Path(__file__).parent
    pretrained_dir = current_dir / 'pretrained'
    pretrained_dir.mkdir(exist_ok=True)
    return pretrained_dir


def get_pretrained_path(backbone_name: str) -> Optional[Path]:
    """
    获取本地预训练权重路径
    
    Args:
        backbone_name: backbone名称，如 'mit_b2'
    
    Returns:
        权重文件路径，如果不存在返回None
    """
    pretrained_dir = get_pretrained_dir()
    
    possible_names = [
        f"{backbone_name}.pth",
        f"{backbone_name}_imagenet.pth",
    ]
    
    for name in possible_names:
        path = pretrained_dir / name
        if path.exists():
            return path
    
    return None


def download_pretrained_weights(url: str, save_path: Path) -> bool:
    """
    下载预训练权重
    
    Args:
        url: 下载URL
        save_path: 保存路径
    
    Returns:
        是否成功
    """
    try:
        import urllib.request
        print(f"📥 下载预训练权重: {url}")
        urllib.request.urlretrieve(url, save_path)
        print(f"💾 已保存到: {save_path}")
        return True
    except Exception as e:
        print(f"⚠️ 下载失败: {e}")
        return False


def create_segformer_config(
    model_scale: str,
    num_classes: int,
    img_size: int = 512,
) -> Dict[str, Any]:
    """
    创建SegFormer配置字典
    
    Args:
        model_scale: 模型规模
        num_classes: 类别数量
        img_size: 输入图像尺寸
    
    Returns:
        配置字典
    """
    config = get_model_config(model_scale)
    if config is None:
        raise ValueError(f"未找到模型配置: {model_scale}")
    
    # 构建MMSegmentation配置
    cfg = {
        'model': {
            'type': 'EncoderDecoder',
            'backbone': {
                'type': 'MixVisionTransformer',
                'in_channels': 3,
                'embed_dims': config['embed_dims'][0],
                'num_stages': 4,
                'num_layers': config['depths'],
                'num_heads': config['num_heads'],
                'patch_sizes': [7, 3, 3, 3],
                'sr_ratios': [8, 4, 2, 1],
                'out_indices': (0, 1, 2, 3),
                'mlp_ratio': 4,
                'qkv_bias': True,
                'drop_rate': 0.0,
                'attn_drop_rate': 0.0,
                'drop_path_rate': 0.1,
            },
            'decode_head': {
                'type': 'SegformerHead',
                'in_channels': config['embed_dims'],
                'in_index': [0, 1, 2, 3],
                'channels': 256,
                'dropout_ratio': 0.1,
                'num_classes': num_classes,
                'norm_cfg': {'type': 'SyncBN', 'requires_grad': True},
                'align_corners': False,
                'loss_decode': {
                    'type': 'CrossEntropyLoss',
                    'use_sigmoid': False,
                    'loss_weight': 1.0,
                },
            },
            'train_cfg': {},
            'test_cfg': {'mode': 'whole'},
        },
        'img_size': img_size,
        'num_classes': num_classes,
        'backbone_name': config['backbone'],
        'model_name': config['name'],
        'checkpoint_url': config['checkpoint'],
    }
    
    return cfg


def build_segformer_model(
    model_scale: str,
    num_classes: int,
    img_size: int = 512,
    pretrained: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """
    构建SegFormer模型
    
    Args:
        model_scale: 模型规模 ('超小', '小', '中', '大', '超大', '超超大')
        num_classes: 类别数量
        img_size: 输入尺寸
        pretrained: 是否使用预训练权重
    
    Returns:
        (model, model_info): 模型对象和模型信息字典
    """
    try:
        from mmseg.models import build_segmentor
        from mmseg.registry import MODELS
        from mmengine.config import Config
    except ImportError:
        raise ImportError("请安装MMSegmentation: pip install mmsegmentation")
    
    config = get_model_config(model_scale)
    if config is None:
        raise ValueError(f"未找到模型配置: {model_scale}")
    
    # 获取或下载预训练权重
    backbone_name = config['backbone']
    local_path = get_pretrained_path(backbone_name)
    
    if pretrained and local_path is None:
        # 下载预训练权重
        pretrained_dir = get_pretrained_dir()
        save_path = pretrained_dir / f"{backbone_name}.pth"
        if download_pretrained_weights(config['checkpoint'], save_path):
            local_path = save_path
    
    weights_source = f"本地缓存: {local_path.name}" if local_path else "将从网络下载"
    
    # 构建模型信息
    model_info = {
        'model_name': config['name'],
        'backbone': backbone_name,
        'model_scale': model_scale,
        'num_classes': num_classes,
        'params_m': config['params'],
        'img_size': img_size,
        'weights_source': weights_source,
        'pretrained_path': str(local_path) if local_path else None,
        'checkpoint_url': config['checkpoint'],
        'input_spec': {
            'shape': (1, 3, img_size, img_size),
            'color_format': 'RGB',
            'pixel_range': (0, 255),
            'normalize_method': 'imagenet',
            'normalize_mean': IMAGENET_MEAN,
            'normalize_std': IMAGENET_STD,
        },
    }
    
    return None, model_info  # 实际模型由MMSeg训练器创建


def get_model_summary(model) -> Dict[str, Any]:
    """获取模型摘要信息"""
    try:
        if hasattr(model, 'parameters'):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            total_params = 0
            trainable_params = 0
        
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
            'error': str(e),
        }


def check_pretrained_status(model_scale: str) -> str:
    """检查预训练权重状态"""
    config = get_model_config(model_scale)
    if config is None:
        return "❌ 未找到模型配置"
    
    backbone_name = config['backbone']
    local_path = get_pretrained_path(backbone_name)
    
    if local_path:
        return f"✅ 已缓存: {local_path.name}"
    else:
        return "📥 首次使用将自动下载并缓存"