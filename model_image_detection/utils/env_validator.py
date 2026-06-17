"""
环境验证器 - 检查PyTorch、ultralytics、CUDA等环境
"""

import os
import sys
from typing import Dict, List, Tuple, Any


def validate_environment() -> Tuple[bool, str, Dict[str, Any]]:
    """
    验证运行环境
    
    Returns:
        (success, message, info): 是否通过, 消息, 环境信息字典
    """
    info = {
        'python_version': sys.version.split()[0],
        'torch_version': None,
        'ultralytics_version': None,
        'cuda_available': False,
        'cuda_version': None,
        'gpu_count': 0,
        'gpus': [],
    }
    
    errors = []
    warnings = []
    
    # 检查 PyTorch
    try:
        import torch
        info['torch_version'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
        
        if info['cuda_available']:
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()
            
            for i in range(info['gpu_count']):
                props = torch.cuda.get_device_properties(i)
                info['gpus'].append({
                    'id': i,
                    'name': props.name,
                    'memory_gb': props.total_memory / (1024**3),
                })
    except ImportError:
        errors.append("❌ PyTorch 未安装，请运行: pip install torch torchvision")
    
    # 检查 ultralytics
    try:
        import ultralytics
        info['ultralytics_version'] = ultralytics.__version__
    except ImportError:
        errors.append("❌ ultralytics 未安装，请运行: pip install ultralytics")
    
    # 检查其他依赖
    try:
        import PIL
        info['pillow_version'] = PIL.__version__
    except ImportError:
        errors.append("❌ Pillow 未安装，请运行: pip install Pillow")
    
    try:
        import numpy
        info['numpy_version'] = numpy.__version__
    except ImportError:
        errors.append("❌ NumPy 未安装，请运行: pip install numpy")
    
    try:
        import yaml
        info['pyyaml_installed'] = True
    except ImportError:
        errors.append("❌ PyYAML 未安装，请运行: pip install pyyaml")
    
    try:
        import gradio
        info['gradio_version'] = gradio.__version__
    except ImportError:
        warnings.append("⚠️ Gradio 未安装（UI界面需要），请运行: pip install gradio")
    
    # 构建消息
    if errors:
        message = "环境验证失败:\n" + "\n".join(errors)
        if warnings:
            message += "\n" + "\n".join(warnings)
        return False, message, info
    
    # 成功消息
    lines = [
        "✅ 环境验证通过",
        f"   Python: {info['python_version']}",
        f"   PyTorch: {info['torch_version']}",
        f"   ultralytics: {info['ultralytics_version']}",
    ]
    
    if info['cuda_available']:
        lines.append(f"   CUDA: {info['cuda_version']}")
        lines.append(f"   GPU数量: {info['gpu_count']}")
        for gpu in info['gpus']:
            lines.append(f"   - GPU{gpu['id']}: {gpu['name']} ({gpu['memory_gb']:.1f}GB)")
    else:
        lines.append("   ⚠️ CUDA不可用，将使用CPU训练")
    
    if warnings:
        lines.extend(warnings)
    
    return True, "\n".join(lines), info


def get_gpu_info() -> List[Dict[str, Any]]:
    """
    获取GPU信息列表
    
    Returns:
        GPU信息列表，每个元素包含id, name, memory_gb
    """
    gpus = []
    
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append({
                    'id': i,
                    'name': props.name,
                    'memory_gb': props.total_memory / (1024**3),
                })
    except ImportError:
        pass
    
    return gpus


def get_gpu_choices() -> List[str]:
    """
    获取GPU选择列表（用于UI下拉框）
    
    Returns:
        选项列表，如 ["CPU", "GPU0 - RTX 3090 (24GB)", "全部GPU (2张)"]
    """
    choices = ["CPU"]
    
    gpus = get_gpu_info()
    
    for gpu in gpus:
        choices.append(f"GPU{gpu['id']} - {gpu['name']} ({gpu['memory_gb']:.1f}GB)")
    
    if len(gpus) > 1:
        choices.append(f"全部GPU ({len(gpus)}张)")
    
    return choices


def parse_gpu_choice(choice: str) -> Tuple[str, List[int]]:
    """
    解析GPU选择
    
    Args:
        choice: UI选择的字符串
    
    Returns:
        (device_str, gpu_ids): 设备字符串和GPU ID列表
        device_str: "cpu" 或 "0" 或 "0,1,2,3"
    """
    if choice == "CPU":
        return "cpu", []
    
    if choice.startswith("全部GPU"):
        gpus = get_gpu_info()
        gpu_ids = [g['id'] for g in gpus]
        device_str = ','.join(str(i) for i in gpu_ids)
        return device_str, gpu_ids
    
    if choice.startswith("GPU"):
        # 解析 "GPU0 - RTX 3090 (24GB)" 格式
        gpu_id = int(choice.split()[0].replace("GPU", ""))
        return str(gpu_id), [gpu_id]
    
    return "cpu", []


def get_recommended_batch_size(gpu_memory_gb: float, img_size: int = 640, model_scale: str = "中") -> int:
    """
    根据GPU显存和模型大小推荐批次大小
    
    Args:
        gpu_memory_gb: GPU显存（GB）
        img_size: 输入图像尺寸
        model_scale: 模型规模
    
    Returns:
        推荐的批次大小
    """
    # 基础参考值 (640尺寸, 中型模型)
    base_memory_per_image = 0.5  # GB per image (大约估计)
    
    # 模型规模系数
    scale_factors = {
        "超小": 0.5,
        "小": 0.7,
        "中": 1.0,
        "大": 1.5,
        "超大": 2.0,
    }
    scale_factor = scale_factors.get(model_scale, 1.0)
    
    # 图像尺寸系数 (相对于640)
    size_factor = (img_size / 640) ** 2
    
    # 计算每张图像所需显存
    memory_per_image = base_memory_per_image * scale_factor * size_factor
    
    # 预留1GB给系统
    available_memory = gpu_memory_gb - 1.0
    
    # 计算推荐批次大小
    batch_size = int(available_memory / memory_per_image)
    
    # 限制在合理范围内
    batch_size = max(1, min(batch_size, 64))
    
    # 对齐到2的倍数
    batch_size = (batch_size // 2) * 2
    batch_size = max(2, batch_size)
    
    return batch_size
