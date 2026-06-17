"""
Checkpoint保存模块 - 兼容 model_importer.py 格式和 ultralytics 标准格式
"""

import torch
from typing import Dict, Any, List, Optional, Union
from pathlib import Path


def save_yolo_checkpoint(
    model,
    save_path: str,
    model_name: str = 'yolov8n',
    num_classes: int = 80,
    class_names: Optional[List[str]] = None,
    input_size: int = 640,
    epoch: int = 0,
    best_fitness: float = 0.0,
    train_args: Optional[Dict] = None,
    ema_model=None,
):
    """
    保存兼容 model_importer.py 和 ultralytics 标准格式的 YOLO checkpoint
    
    保存格式包含:
    - ultralytics 标准字段: nc, names (用于 model_importer 推断类别数)
    - 扩展元数据字段: model_metadata (用于预处理配置)
    
    Args:
        model: 训练好的 YOLO 模型 (nn.Module 或 ultralytics模型)
        save_path: 保存路径
        model_name: 模型名称 ('yolov5n', 'yolov8n', 'yolo11n' 等)
        num_classes: 类别数
        class_names: 类别名称列表
        input_size: 输入图像尺寸
        epoch: 训练轮数
        best_fitness: 最佳指标 (mAP@0.5:0.95)
        train_args: 训练参数字典
        ema_model: EMA模型 (如果有)
    """
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    
    # 确保 class_names 长度匹配 num_classes
    if len(class_names) < num_classes:
        class_names = class_names + [str(i) for i in range(len(class_names), num_classes)]
    
    # 获取实际的PyTorch模型
    if hasattr(model, 'model'):
        pytorch_model = model.model
    else:
        pytorch_model = model
    
    # 创建 names 字典 (ultralytics 标准格式)
    names_dict: Dict[int, str] = {i: name for i, name in enumerate(class_names)}
    
    # 构建默认训练参数
    default_train_args = {
        'imgsz': input_size,
        'batch': 16,
        'epochs': 100,
        'nc': num_classes,  # 添加nc字段，model_importer可能需要
    }
    if train_args:
        default_train_args.update(train_args)
    # 确保nc始终存在
    default_train_args['nc'] = num_classes
    
    checkpoint = {
        # ============================================
        # ultralytics 标准字段 (model_importer 会优先检查这些)
        # ============================================
        'nc': num_classes,                 # 类别数 (ultralytics 标准字段)
        'names': names_dict,               # 类别名称字典 {0: 'class0', 1: 'class1', ...}
        
        # ============================================
        # 必需字段 - model_importer.py 依赖这些
        # ============================================
        'model': pytorch_model,            # 完整的模型对象 (必需!)
        
        # ============================================
        # 推荐字段 - 帮助 model_importer 推断信息
        # ============================================
        'model_name': model_name,          # 模型架构名称
        'task': 'detect',                  # 任务类型: 'detect', 'segment', 'classify'
        'framework': 'ultralytics',        # 框架标识
        
        # ============================================
        # 元数据字段 - 用于预处理配置
        # ============================================
        'model_metadata': {
            'input_spec': {
                # 输入尺寸 (NCHW格式)
                'shape': (1, 3, input_size, input_size),
                
                # 颜色空间
                'color_format': 'RGB',  # YOLO使用RGB，不是BGR
                
                # 原始像素范围
                'pixel_range': (0, 255),
                
                # 归一化方法: YOLO只做除以255，不做ImageNet mean/std归一化
                'normalize_method': 'divide_255',
                
                # 归一化参数 (假设输入已归一化到0-1范围)
                # YOLO不使用ImageNet的mean/std，这里是恒等变换
                'normalize_mean': (0.0, 0.0, 0.0),
                'normalize_std': (1.0, 1.0, 1.0),
                
                # 归一化后的值范围
                'value_range': (0.0, 1.0),
                
                # Letterbox填充颜色 (灰色)
                'letterbox_color': (114, 114, 114),
            },
            'num_classes': num_classes,
            'class_names': class_names,
        },
        
        # ============================================
        # 可选字段 - 训练信息
        # ============================================
        'ema': ema_model,                  # EMA模型 (如果有)
        'epoch': epoch,                    # 训练轮数
        'best_fitness': best_fitness,      # 最佳指标
        'train_args': default_train_args,  # 训练参数
        
        # ============================================
        # 量化建议
        # ============================================
        'quantization_hints': {
            'recommended_precision': 'fp16',
            'int8_friendly': True,
        },
    }
    
    # 如果模型本身有 nc 和 names 属性，也尝试从模型获取
    # 这样可以确保与 ultralytics 原生格式一致
    if hasattr(pytorch_model, 'nc'):
        checkpoint['nc'] = pytorch_model.nc
    if hasattr(pytorch_model, 'names'):
        if isinstance(pytorch_model.names, dict):
            checkpoint['names'] = pytorch_model.names
        elif isinstance(pytorch_model.names, (list, tuple)):
            checkpoint['names'] = {i: str(n) for i, n in enumerate(pytorch_model.names)}
    
    # 保存
    torch.save(checkpoint, save_path)
    return save_path


def load_yolo_checkpoint(
    checkpoint_path: str,
) -> Dict[str, Any]:
    """
    加载YOLO checkpoint
    
    Args:
        checkpoint_path: checkpoint文件路径
    
    Returns:
        checkpoint字典
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint


def convert_ultralytics_to_compatible(
    ultralytics_pt: str,
    output_path: str,
    model_name: str,
    class_names: List[str],
    input_size: int = 640,
) -> str:
    """
    将ultralytics原生checkpoint转换为兼容格式
    
    Args:
        ultralytics_pt: ultralytics保存的.pt文件路径
        output_path: 输出路径
        model_name: 模型名称
        class_names: 类别名称列表
        input_size: 输入尺寸
    
    Returns:
        输出文件路径
    """
    from ultralytics import YOLO
    
    # 加载ultralytics模型
    yolo = YOLO(ultralytics_pt)
    
    # 获取训练参数
    train_args = {}
    if hasattr(yolo, 'overrides'):
        train_args = dict(yolo.overrides)
    
    # 保存为兼容格式
    save_yolo_checkpoint(
        model=yolo.model,
        save_path=output_path,
        model_name=model_name,
        num_classes=len(class_names),
        class_names=class_names,
        input_size=input_size,
        train_args=train_args,
    )
    
    return output_path


def get_checkpoint_info(checkpoint_path: str) -> Dict[str, Any]:
    """
    获取checkpoint信息（不加载完整模型）
    
    Args:
        checkpoint_path: checkpoint文件路径
    
    Returns:
        checkpoint信息字典
    """
    try:
        # 使用weights_only尝试加载
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except Exception:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        info = {
            'model_name': checkpoint.get('model_name', 'unknown'),
            'task': checkpoint.get('task', 'unknown'),
            'framework': checkpoint.get('framework', 'unknown'),
            'epoch': checkpoint.get('epoch', 0),
            'best_fitness': checkpoint.get('best_fitness', 0),
        }
        
        # 获取元数据
        if 'model_metadata' in checkpoint:
            metadata = checkpoint['model_metadata']
            info['num_classes'] = metadata.get('num_classes', 0)
            info['class_names'] = metadata.get('class_names', [])
            if 'input_spec' in metadata:
                info['input_size'] = metadata['input_spec'].get('shape', [1, 3, 640, 640])[-1]
        
        return info
        
    except Exception as e:
        return {'error': str(e)}
