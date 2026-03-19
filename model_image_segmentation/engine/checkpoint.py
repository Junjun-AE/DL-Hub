"""
Checkpoint保存模块 - 保存兼容格式的SegFormer模型
包含完整的预处理元数据，便于推理时使用
"""

import torch
from typing import Dict, Any, List, Optional
from pathlib import Path


def save_segformer_checkpoint(
    checkpoint_path: str,
    save_path: str,
    model_name: str = 'segformer_b2',
    num_classes: int = 2,
    class_names: Optional[List[str]] = None,
    img_size: int = 512,
    best_mIoU: float = 0.0,
    best_epoch: int = 0,
    train_args: Optional[Dict] = None,
):
    """
    保存兼容格式的SegFormer checkpoint
    
    Args:
        checkpoint_path: 原始checkpoint路径（MMSeg格式）
        save_path: 保存路径
        model_name: 模型名称
        num_classes: 类别数量
        class_names: 类别名称列表
        img_size: 输入图像尺寸
        best_mIoU: 最佳mIoU
        best_epoch: 最佳epoch
        train_args: 训练参数字典
    """
    from config.model_registry import IMAGENET_MEAN, IMAGENET_STD
    
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    
    # 加载原始checkpoint
    original_ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # 提取state_dict
    if 'state_dict' in original_ckpt:
        state_dict = original_ckpt['state_dict']
    else:
        state_dict = original_ckpt
    
    # 创建names字典
    names_dict: Dict[int, str] = {i: name for i, name in enumerate(class_names)}
    
    # 默认训练参数
    default_train_args = {
        'img_size': img_size,
        'batch_size': 8,
        'epochs': 100,
        'num_classes': num_classes,
    }
    if train_args:
        default_train_args.update(train_args)
    
    # 构建兼容格式的checkpoint
    checkpoint = {
        # ============================================
        # 框架标识 - 用于识别模型类型
        # ============================================
        'framework': 'mmsegmentation',
        'model_type': 'segformer',
        'model_name': model_name,
        
        # ============================================
        # 模型权重
        # ============================================
        'state_dict': state_dict,
        
        # ============================================
        # 类别信息
        # ============================================
        'num_classes': num_classes,
        'names': names_dict,
        'class_names': class_names,
        
        # ============================================
        # 预处理元数据 (关键！)
        # ============================================
        'model_metadata': {
            'input_spec': {
                # 输入尺寸 (NCHW格式)
                'shape': (1, 3, img_size, img_size),
                
                # 颜色空间
                'color_format': 'RGB',
                
                # 原始像素范围
                'pixel_range': (0, 255),
                
                # 归一化方法: ImageNet标准归一化
                'normalize_method': 'imagenet',
                
                # 归一化参数 (ImageNet mean/std)
                'normalize_mean': IMAGENET_MEAN,
                'normalize_std': IMAGENET_STD,
                
                # 归一化后的值范围（大约）
                'value_range': (-2.5, 2.5),
            },
            'num_classes': num_classes,
            'class_names': class_names,
            'ignore_index': 255,  # 背景/忽略像素值
            'task': 'semantic_segmentation',
        },
        
        # ============================================
        # 训练信息
        # ============================================
        'train_args': default_train_args,
        'best_mIoU': best_mIoU,
        'best_epoch': best_epoch,
        
        # ============================================
        # 推理配置建议
        # ============================================
        'inference_config': {
            'sliding_window': {
                'enabled': True,
                'window_size': img_size,
                'stride': img_size // 2,
            },
            'threshold': {
                'default': 'argmax',
                'optional_conf_threshold': 0.5,
            },
        },
    }
    
    # 保存
    torch.save(checkpoint, save_path)
    return save_path


def load_segformer_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    加载SegFormer checkpoint
    
    Args:
        checkpoint_path: checkpoint文件路径
    
    Returns:
        checkpoint字典
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint


def get_checkpoint_info(checkpoint_path: str) -> Dict[str, Any]:
    """
    获取checkpoint信息（不加载完整模型）
    
    Args:
        checkpoint_path: checkpoint文件路径
    
    Returns:
        checkpoint信息字典
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        info = {
            'framework': checkpoint.get('framework', 'unknown'),
            'model_type': checkpoint.get('model_type', 'unknown'),
            'model_name': checkpoint.get('model_name', 'unknown'),
            'num_classes': checkpoint.get('num_classes', 0),
            'best_mIoU': checkpoint.get('best_mIoU', 0),
            'best_epoch': checkpoint.get('best_epoch', 0),
        }
        
        if 'model_metadata' in checkpoint:
            metadata = checkpoint['model_metadata']
            info['class_names'] = metadata.get('class_names', [])
            info['ignore_index'] = metadata.get('ignore_index', 255)
            if 'input_spec' in metadata:
                info['img_size'] = metadata['input_spec'].get('shape', [1, 3, 512, 512])[-1]
        
        return info
        
    except Exception as e:
        return {'error': str(e)}


def convert_mmseg_checkpoint(
    mmseg_ckpt_path: str,
    output_path: str,
    model_name: str,
    num_classes: int,
    class_names: List[str],
    img_size: int = 512,
) -> str:
    """
    将MMSegmentation原生checkpoint转换为兼容格式
    
    Args:
        mmseg_ckpt_path: MMSeg checkpoint路径
        output_path: 输出路径
        model_name: 模型名称
        num_classes: 类别数量
        class_names: 类别名称列表
        img_size: 输入尺寸
    
    Returns:
        输出文件路径
    """
    return save_segformer_checkpoint(
        checkpoint_path=mmseg_ckpt_path,
        save_path=output_path,
        model_name=model_name,
        num_classes=num_classes,
        class_names=class_names,
        img_size=img_size,
    )
