"""
YOLO-seg 实例分割模型注册表
===========================
为「图像分割」模块新增的实例分割引擎（区别于 SegFormer 语义分割）。

包含 YOLOv8-seg / YOLOv11-seg / YOLO26-seg 三个系列，每个系列 5 种规模。
规模命名与 SegFormer 保持统一中文风格：超小/小/中/大/超大 -> n/s/m/l/x。

所有模型均使用 Ultralytics 官方 COCO 预训练权重。
注意: YOLO26-seg 需要 ultralytics >= 8.4.0。

数据/标签说明:
- 实例分割使用 Ultralytics YOLO polygon 标签 (一图一 txt, 每行: <class> <x1> <y1> ... <xn> <yn>, 归一化)
- 与 SegFormer 共用同一份 LabelMe polygon 标注源, 仅转换目标不同
"""

from typing import Dict, Any, Optional, List


# 规模 -> YOLO 后缀字母
SCALE_TO_LETTER: Dict[str, str] = {
    "超小": "n",
    "小": "s",
    "中": "m",
    "大": "l",
    "超大": "x",
}

# 系列 -> 基础模型名前缀（用于拼接 {prefix}{letter}-seg）
FAMILY_TO_PREFIX: Dict[str, str] = {
    "YOLOv8": "yolov8",
    "YOLOv11": "yolo11",
    "YOLO26": "yolo26",
}


# 参考精度 (COCO val mask mAP50-95) 与参数量, 仅用于 UI 展示
# 数据来源: Ultralytics 官方 segment 文档
_FAMILY_STATS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "YOLOv8": {
        "超小": {"params": 3.4, "mAP": 30.5, "desc": "v8 最轻量分割，边缘设备首选"},
        "小":   {"params": 11.8, "mAP": 36.8, "desc": "v8 小型，速度精度平衡"},
        "中":   {"params": 27.3, "mAP": 40.8, "desc": "推荐：v8 中型，综合表现优秀"},
        "大":   {"params": 46.0, "mAP": 42.6, "desc": "v8 大型，高精度"},
        "超大": {"params": 71.8, "mAP": 43.4, "desc": "v8 最大，极致精度"},
    },
    "YOLOv11": {
        "超小": {"params": 2.9, "mAP": 32.0, "desc": "v11 轻量分割，最新架构"},
        "小":   {"params": 10.1, "mAP": 37.8, "desc": "v11 小型，效率提升"},
        "中":   {"params": 22.4, "mAP": 41.5, "desc": "推荐：v11 中型，最新技术"},
        "大":   {"params": 27.6, "mAP": 42.9, "desc": "v11 大型，精度优秀"},
        "超大": {"params": 62.1, "mAP": 43.8, "desc": "v11 最大，SOTA 性能"},
    },
    "YOLO26": {
        "超小": {"params": 2.8, "mAP": 33.4, "desc": "v26 轻量分割，多尺度 proto + 语义损失增强"},
        "小":   {"params": 10.0, "mAP": 39.2, "desc": "v26 小型，端到端 NMS-free"},
        "中":   {"params": 22.0, "mAP": 43.2, "desc": "推荐：v26 中型，mask 质量最佳(较 v11 +3.7 mask AP)"},
        "大":   {"params": 27.4, "mAP": 44.5, "desc": "v26 大型，小目标分割优化"},
        "超大": {"params": 62.0, "mAP": 45.6, "desc": "v26 最大，最新 SOTA"},
    },
}

# 默认输入尺寸
DEFAULT_IMG_SIZE = 640

# 支持的输入尺寸 (YOLO 要求 32 的倍数)
SUPPORTED_IMG_SIZES = [512, 640, 768, 1024, 1280]


def _build_registry() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """构建 family -> scale -> 配置 的注册表。"""
    registry: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for family, prefix in FAMILY_TO_PREFIX.items():
        registry[family] = {}
        for scale, letter in SCALE_TO_LETTER.items():
            base = f"{prefix}{letter}"            # e.g. yolo26m
            seg_name = f"{base}-seg"               # e.g. yolo26m-seg
            stats = _FAMILY_STATS[family][scale]
            registry[family][scale] = {
                "name": seg_name,                  # 展示名 / 内部标识
                "ultralytics_name": f"{seg_name}.pt",  # 预训练权重文件名
                "base_name": base,                 # 不含 -seg 的基础名
                "params": stats["params"],
                "mAP_mask": stats["mAP"],
                "input_size": DEFAULT_IMG_SIZE,
                "description": stats["desc"],
            }
    return registry


# 模型注册表: family -> scale -> config
MODEL_REGISTRY: Dict[str, Dict[str, Dict[str, Any]]] = _build_registry()


# 默认训练配置 (YOLO-seg)
DEFAULT_CONFIG = {
    "family": "YOLO26",
    "model_scale": "中",
    "img_size": DEFAULT_IMG_SIZE,
    "batch_size": 16,
    "epochs": 100,
    "learning_rate": 0.01,
    "optimizer": "SGD",
    "patience": 50,
    "val_split": 0.2,
    "save_period": 0,
    "weight_decay": 0.0005,
    "momentum": 0.937,
    "warmup_epochs": 3.0,
    "cos_lr": True,
    "mosaic": 1.0,
    "mixup": 0.0,
    "fliplr": 0.5,
}


def get_all_families() -> List[str]:
    """获取所有 YOLO-seg 系列。"""
    return list(MODEL_REGISTRY.keys())


def get_all_scales() -> List[str]:
    """获取所有规模 (统一顺序)。"""
    return list(SCALE_TO_LETTER.keys())


def get_model_scales(family: str) -> List[str]:
    """获取指定系列支持的规模。"""
    if family in MODEL_REGISTRY:
        return list(MODEL_REGISTRY[family].keys())
    return []


def get_model_config(family: str, scale: str) -> Optional[Dict[str, Any]]:
    """
    获取模型配置。

    Args:
        family: 系列名 ('YOLOv8'/'YOLOv11'/'YOLO26')
        scale: 规模 ('超小'/'小'/'中'/'大'/'超大')

    Returns:
        配置字典，不存在返回 None
    """
    if family in MODEL_REGISTRY and scale in MODEL_REGISTRY[family]:
        return MODEL_REGISTRY[family][scale]
    return None


def get_model_display_info(family: str, scale: str) -> str:
    """获取模型展示信息。"""
    config = get_model_config(family, scale)
    if config is None:
        return "未找到模型配置"

    return (
        f"📊 模型: {config['name']}\n"
        f"📦 参数量: {config['params']}M\n"
        f"🎯 参考 mask mAP@50-95: {config['mAP_mask']}%\n"
        f"📐 输入尺寸: {config['input_size']}×{config['input_size']}\n"
        f"💡 {config['description']}"
    )


def list_all_models() -> str:
    """列出所有可用 YOLO-seg 模型。"""
    lines = ["=" * 50, "可用 YOLO-seg 实例分割模型列表", "=" * 50]
    for family, scales in MODEL_REGISTRY.items():
        lines.append(f"\n【{family}】")
        for scale, config in scales.items():
            lines.append(
                f"  {scale}: {config['name']} "
                f"({config['params']}M, mask mAP={config['mAP_mask']}%)"
            )
    return "\n".join(lines)
