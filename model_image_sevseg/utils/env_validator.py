"""
SevSeg-YOLO 环境验证器
检查 GPU、PyTorch、ultralytics、opencv-contrib 等
与 detection/classification 任务保持一致的检测方式
"""
import os, sys
from typing import Tuple, List, Dict, Any


def validate_environment() -> Tuple[bool, str, Dict[str, Any]]:
    info = {'python': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'cuda_available': False, 'gpus': [], 'errors': [], 'warnings': []}
    lines = [f"✅ Python: {info['python']}"]
    errors = info['errors']

    # PyTorch
    try:
        import torch
        info['torch'] = torch.__version__
        lines.append(f"✅ PyTorch: {torch.__version__}")
        info['cuda_available'] = torch.cuda.is_available()
        if info['cuda_available']:
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()
            for i in range(info['gpu_count']):
                props = torch.cuda.get_device_properties(i)
                mem_gb = props.total_memory / (1024**3)
                info['gpus'].append({'id': i, 'name': props.name, 'memory_gb': mem_gb})
                lines.append(f"✅ GPU {i}: {props.name} ({mem_gb:.1f}GB)")
            lines.append(f"   CUDA: {info['cuda_version']}")
        else:
            lines.append("⚠️ CUDA 不可用，将使用CPU训练")
    except ImportError:
        errors.append("❌ PyTorch 未安装")

    # Ultralytics (内部修改版)
    try:
        sevseg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if sevseg_dir not in sys.path:
            sys.path.insert(0, sevseg_dir)
        from ultralytics import __version__ as uv
        lines.append(f"✅ Ultralytics (SevSeg修改版): {uv}")
    except ImportError:
        errors.append("❌ Ultralytics 未找到")

    # ScoreDetect
    try:
        from ultralytics.nn.modules.head import ScoreDetect
        lines.append("✅ ScoreDetect 头模块")
    except ImportError:
        errors.append("❌ ScoreDetect 未找到")

    # SevSeg-YOLO
    try:
        from sevseg_yolo import SevSegYOLO, MaskGeneratorV2
        lines.append("✅ SevSeg-YOLO 核心模块")
    except ImportError as e:
        errors.append(f"❌ SevSeg-YOLO: {e}")

    # OpenCV
    try:
        import cv2
        lines.append(f"✅ OpenCV: {cv2.__version__}")
    except ImportError:
        errors.append("❌ OpenCV 未安装")

    # NumPy / SciPy / Matplotlib
    try:
        import numpy; lines.append(f"✅ NumPy: {numpy.__version__}")
    except: errors.append("❌ NumPy 未安装")
    try:
        import scipy; lines.append(f"✅ SciPy: {scipy.__version__}")
    except: lines.append("⚠️ SciPy 未安装 (Spearman用备用算法)")
    try:
        import matplotlib; lines.append(f"✅ Matplotlib: {matplotlib.__version__}")
    except: lines.append("⚠️ Matplotlib 未安装")

    all_ok = len(errors) == 0
    return all_ok, "\n".join(errors + lines), info


def get_gpu_info() -> List[Dict[str, Any]]:
    """获取GPU信息列表"""
    gpus = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append({'id': i, 'name': props.name, 'memory_gb': props.total_memory / (1024**3)})
    except: pass
    return gpus


def get_gpu_choices() -> List[str]:
    """获取GPU选择列表（与detection任务一致）"""
    choices = ["CPU"]
    gpus = get_gpu_info()
    for g in gpus:
        choices.append(f"GPU{g['id']} - {g['name']} ({g['memory_gb']:.1f}GB)")
    if len(gpus) > 1:
        choices.append(f"全部GPU ({len(gpus)}张)")
    return choices


def parse_gpu_choice(choice: str) -> Tuple[str, List[int]]:
    """解析GPU选择 → (device_str, gpu_ids)"""
    if not choice or choice == "CPU":
        return "cpu", []
    if choice.startswith("全部GPU"):
        gpus = get_gpu_info()
        ids = [g['id'] for g in gpus]
        return ','.join(str(i) for i in ids), ids
    try:
        # "GPU0 - RTX 3090 (24GB)" → 提取 0
        gpu_id = int(choice.split("-")[0].replace("GPU", "").strip())
        return str(gpu_id), [gpu_id]
    except Exception:
        return "cpu", []
