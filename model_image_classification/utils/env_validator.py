"""
环境验证器 - 检查PyTorch、timm、CUDA等环境
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
        'timm_version': None,
        'cuda_available': False,
        'cuda_version': None,
        'gpu_count': 0,
        'gpus': [],
    }
    
    errors = []
    
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
    
    # 检查 timm
    try:
        import timm
        info['timm_version'] = timm.__version__
    except ImportError:
        errors.append("❌ timm 未安装，请运行: pip install timm")
    
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
    
    # 构建消息
    if errors:
        message = "环境验证失败:\n" + "\n".join(errors)
        return False, message, info
    
    # 成功消息
    lines = [
        "✅ 环境验证通过",
        f"   Python: {info['python_version']}",
        f"   PyTorch: {info['torch_version']}",
        f"   timm: {info['timm_version']}",
    ]
    
    if info['cuda_available']:
        lines.append(f"   CUDA: {info['cuda_version']}")
        lines.append(f"   GPU数量: {info['gpu_count']}")
        for gpu in info['gpus']:
            lines.append(f"   - GPU{gpu['id']}: {gpu['name']} ({gpu['memory_gb']:.1f}GB)")
    else:
        lines.append("   ⚠️ CUDA不可用，将使用CPU训练")
    
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
        (device_type, gpu_ids): 设备类型和GPU ID列表
        device_type: "cpu" 或 "cuda"
        gpu_ids: GPU ID列表，如 [0], [0,1,2,3]
    """
    if choice == "CPU":
        return "cpu", []
    
    if choice.startswith("全部GPU"):
        gpus = get_gpu_info()
        return "cuda", [g['id'] for g in gpus]
    
    if choice.startswith("GPU"):
        # 解析 "GPU0 - RTX 3090 (24GB)" 格式
        gpu_id = int(choice.split()[0].replace("GPU", ""))
        return "cuda", [gpu_id]
    
    return "cpu", []
