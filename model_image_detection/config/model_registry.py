"""
YOLO模型注册表 - 20个预训练模型配置
包含 YOLOv5, YOLOv8, YOLOv11, YOLO26 四个系列
所有模型均使用ultralytics官方预训练权重

注意: YOLO26 需要 ultralytics >= 8.4.0
"""

from typing import Dict, Any, Optional, List

# 模型注册表
MODEL_REGISTRY: Dict[str, Dict[str, Dict[str, Any]]] = {
    "YOLOv5": {
        "超小": {
            "name": "yolov5n",
            "ultralytics_name": "yolov5nu.pt",
            "params": 1.9,
            "mAP50": 45.7,
            "mAP50_95": 28.0,
            "input_size": 640,
            "description": "最轻量级，适合边缘设备部署",
        },
        "小": {
            "name": "yolov5s",
            "ultralytics_name": "yolov5su.pt",
            "params": 7.2,
            "mAP50": 56.8,
            "mAP50_95": 37.4,
            "input_size": 640,
            "description": "轻量级，平衡速度与精度",
        },
        "中": {
            "name": "yolov5m",
            "ultralytics_name": "yolov5mu.pt",
            "params": 21.2,
            "mAP50": 63.9,
            "mAP50_95": 45.4,
            "input_size": 640,
            "description": "推荐：性价比最高",
        },
        "大": {
            "name": "yolov5l",
            "ultralytics_name": "yolov5lu.pt",
            "params": 46.5,
            "mAP50": 67.3,
            "mAP50_95": 49.0,
            "input_size": 640,
            "description": "高精度，适合服务器",
        },
        "超大": {
            "name": "yolov5x",
            "ultralytics_name": "yolov5xu.pt",
            "params": 86.7,
            "mAP50": 68.9,
            "mAP50_95": 50.7,
            "input_size": 640,
            "description": "最高精度，需要大显存",
        },
    },
    "YOLOv8": {
        "超小": {
            "name": "yolov8n",
            "ultralytics_name": "yolov8n.pt",
            "params": 3.2,
            "mAP50": 52.6,
            "mAP50_95": 37.3,
            "input_size": 640,
            "description": "v8轻量级，推理速度最快",
        },
        "小": {
            "name": "yolov8s",
            "ultralytics_name": "yolov8s.pt",
            "params": 11.2,
            "mAP50": 61.8,
            "mAP50_95": 44.9,
            "input_size": 640,
            "description": "v8小型，速度精度平衡",
        },
        "中": {
            "name": "yolov8m",
            "ultralytics_name": "yolov8m.pt",
            "params": 25.9,
            "mAP50": 67.2,
            "mAP50_95": 50.2,
            "input_size": 640,
            "description": "推荐：v8中型，综合表现优秀",
        },
        "大": {
            "name": "yolov8l",
            "ultralytics_name": "yolov8l.pt",
            "params": 43.7,
            "mAP50": 69.8,
            "mAP50_95": 52.9,
            "input_size": 640,
            "description": "v8大型，高精度",
        },
        "超大": {
            "name": "yolov8x",
            "ultralytics_name": "yolov8x.pt",
            "params": 68.2,
            "mAP50": 71.0,
            "mAP50_95": 53.9,
            "input_size": 640,
            "description": "v8最大，极致精度",
        },
    },
    "YOLOv11": {
        "超小": {
            "name": "yolo11n",
            "ultralytics_name": "yolo11n.pt",
            "params": 2.6,
            "mAP50": 54.4,
            "mAP50_95": 39.5,
            "input_size": 640,
            "description": "v11轻量级，最新架构",
        },
        "小": {
            "name": "yolo11s",
            "ultralytics_name": "yolo11s.pt",
            "params": 9.4,
            "mAP50": 62.5,
            "mAP50_95": 47.0,
            "input_size": 640,
            "description": "v11小型，效率提升",
        },
        "中": {
            "name": "yolo11m",
            "ultralytics_name": "yolo11m.pt",
            "params": 20.1,
            "mAP50": 68.0,
            "mAP50_95": 51.5,
            "input_size": 640,
            "description": "推荐：v11中型，最新技术",
        },
        "大": {
            "name": "yolo11l",
            "ultralytics_name": "yolo11l.pt",
            "params": 25.3,
            "mAP50": 69.5,
            "mAP50_95": 53.4,
            "input_size": 640,
            "description": "v11大型，精度优秀",
        },
        "超大": {
            "name": "yolo11x",
            "ultralytics_name": "yolo11x.pt",
            "params": 56.9,
            "mAP50": 72.0,
            "mAP50_95": 54.7,
            "input_size": 640,
            "description": "v11最大，SOTA性能",
        },
    },
    "YOLO26": {
        "超小": {
            "name": "yolo26n",
            "ultralytics_name": "yolo26n.pt",
            "params": 2.4,
            "mAP50": 55.4,
            "mAP50_95": 40.6,
            "input_size": 640,
            "description": "v26轻量级，NMS-free端到端推理，CPU推理速度最快",
        },
        "小": {
            "name": "yolo26s",
            "ultralytics_name": "yolo26s.pt",
            "params": 9.2,
            "mAP50": 63.6,
            "mAP50_95": 48.0,
            "input_size": 640,
            "description": "v26小型，端到端推理，边缘设备友好",
        },
        "中": {
            "name": "yolo26m",
            "ultralytics_name": "yolo26m.pt",
            "params": 19.1,
            "mAP50": 68.8,
            "mAP50_95": 52.5,
            "input_size": 640,
            "description": "推荐：v26中型，去DFL+NMS-free+MuSGD优化器",
        },
        "大": {
            "name": "yolo26l",
            "ultralytics_name": "yolo26l.pt",
            "params": 24.5,
            "mAP50": 70.3,
            "mAP50_95": 53.8,
            "input_size": 640,
            "description": "v26大型，小目标检测优化(STAL+ProgLoss)",
        },
        "超大": {
            "name": "yolo26x",
            "ultralytics_name": "yolo26x.pt",
            "params": 55.5,
            "mAP50": 72.6,
            "mAP50_95": 55.4,
            "input_size": 640,
            "description": "v26最大，最新SOTA，端到端NMS-free推理",
        },
    },
}


def get_model_config(family: str, scale: str) -> Optional[Dict[str, Any]]:
    """
    获取模型配置
    
    Args:
        family: 模型系列名称 ('YOLOv5', 'YOLOv8', 'YOLOv11')
        scale: 模型规模 ('超小', '小', '中', '大', '超大')
    
    Returns:
        模型配置字典，不存在返回None
    """
    if family in MODEL_REGISTRY and scale in MODEL_REGISTRY[family]:
        return MODEL_REGISTRY[family][scale]
    return None


def get_all_families() -> List[str]:
    """获取所有模型系列"""
    return list(MODEL_REGISTRY.keys())


def get_all_scales() -> List[str]:
    """获取所有模型规模"""
    return ["超小", "小", "中", "大", "超大"]


def get_model_display_info(family: str, scale: str) -> str:
    """
    获取模型展示信息
    
    Args:
        family: 模型系列
        scale: 模型规模
    
    Returns:
        格式化的展示字符串
    """
    config = get_model_config(family, scale)
    if config is None:
        return "未找到模型配置"
    
    return (
        f"📊 模型: {config['name']}\n"
        f"📦 参数量: {config['params']}M\n"
        f"🎯 mAP@50: {config['mAP50']}%\n"
        f"🎯 mAP@50:95: {config['mAP50_95']}%\n"
        f"📐 输入尺寸: {config['input_size']}×{config['input_size']}\n"
        f"💡 {config['description']}"
    )


def list_all_models() -> str:
    """列出所有可用模型"""
    lines = ["=" * 50, "可用YOLO模型列表", "=" * 50]
    
    for family, scales in MODEL_REGISTRY.items():
        lines.append(f"\n【{family}】")
        for scale, config in scales.items():
            lines.append(
                f"  {scale}: {config['name']} "
                f"({config['params']}M, mAP50={config['mAP50']}%)"
            )
    
    return "\n".join(lines)
