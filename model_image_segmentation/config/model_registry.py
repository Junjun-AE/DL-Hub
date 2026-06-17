"""
SegFormer模型注册表 - 6个预训练模型配置
包含 SegFormer B0-B5 六个规模
所有模型均使用ImageNet-1K预训练权重
"""

from typing import Dict, Any, Optional, List


# 模型注册表
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "超小": {
        "name": "segformer_b0",
        "backbone": "mit_b0",
        "checkpoint": "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth",
        "params": 3.7,
        "description": "最轻量级，适合边缘设备部署",
        "embed_dims": [32, 64, 160, 256],
        "num_heads": [1, 2, 5, 8],
        "depths": [2, 2, 2, 2],
    },
    "小": {
        "name": "segformer_b1",
        "backbone": "mit_b1",
        "checkpoint": "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b1_20220624-02e5a6a1.pth",
        "params": 13.7,
        "description": "轻量级，平衡速度与精度",
        "embed_dims": [64, 128, 320, 512],
        "num_heads": [1, 2, 5, 8],
        "depths": [2, 2, 2, 2],
    },
    "中": {
        "name": "segformer_b2",
        "backbone": "mit_b2",
        "checkpoint": "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth",
        "params": 24.7,
        "description": "推荐：性价比最高，工业首选",
        "embed_dims": [64, 128, 320, 512],
        "num_heads": [1, 2, 5, 8],
        "depths": [3, 4, 6, 3],
    },
    "大": {
        "name": "segformer_b3",
        "backbone": "mit_b3",
        "checkpoint": "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b3_20220624-13b1141c.pth",
        "params": 44.6,
        "description": "高精度，适合复杂场景",
        "embed_dims": [64, 128, 320, 512],
        "num_heads": [1, 2, 5, 8],
        "depths": [3, 4, 18, 3],
    },
    "超大": {
        "name": "segformer_b4",
        "backbone": "mit_b4",
        "checkpoint": "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b4_20220624-d588d980.pth",
        "params": 61.4,
        "description": "更高精度，需要较大显存",
        "embed_dims": [64, 128, 320, 512],
        "num_heads": [1, 2, 5, 8],
        "depths": [3, 8, 27, 3],
    },
    "超超大": {
        "name": "segformer_b5",
        "backbone": "mit_b5",
        "checkpoint": "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth",
        "params": 82.0,
        "description": "最高精度，需要大显存",
        "embed_dims": [64, 128, 320, 512],
        "num_heads": [1, 2, 5, 8],
        "depths": [3, 6, 40, 3],
    },
}


# 支持的输入尺寸
SUPPORTED_IMG_SIZES = [512, 640, 768, 1024]


# 默认配置
DEFAULT_CONFIG = {
    "model_scale": "中",
    "img_size": 512,
    "batch_size": 8,
    "epochs": 100,
    "learning_rate": 6e-5,
    "optimizer": "AdamW",
    "weight_decay": 0.01,
    "warmup_iters": 500,
    "patience": 20,
    "val_split": 0.2,
    "save_period": 10,
}


def get_model_config(scale: str) -> Optional[Dict[str, Any]]:
    """
    获取模型配置
    
    Args:
        scale: 模型规模 ('超小', '小', '中', '大', '超大', '超超大')
    
    Returns:
        模型配置字典，不存在返回None
    """
    return MODEL_REGISTRY.get(scale, None)


def get_all_scales() -> List[str]:
    """获取所有模型规模"""
    return list(MODEL_REGISTRY.keys())


def get_model_display_info(scale: str) -> str:
    """
    获取模型展示信息
    
    Args:
        scale: 模型规模
    
    Returns:
        格式化的展示字符串
    """
    config = get_model_config(scale)
    if config is None:
        return "未找到模型配置"
    
    return (
        f"📊 模型: {config['name']}\n"
        f"🔧 Backbone: {config['backbone']}\n"
        f"📦 参数量: {config['params']}M\n"
        f"💡 {config['description']}"
    )


def list_all_models() -> str:
    """列出所有可用模型"""
    lines = ["=" * 50, "可用SegFormer模型列表", "=" * 50]
    
    for scale, config in MODEL_REGISTRY.items():
        lines.append(
            f"  {scale}: {config['name']} "
            f"({config['params']}M 参数)"
        )
        lines.append(f"       {config['description']}")
    
    return "\n".join(lines)


# ImageNet归一化参数（MMSegmentation使用）
IMAGENET_MEAN = (123.675, 116.28, 103.53)
IMAGENET_STD = (58.395, 57.12, 57.375)


# 分割可视化颜色表（12种颜色循环使用）
SEGMENTATION_COLORS = [
    (255, 0, 0),      # 红
    (0, 255, 0),      # 绿
    (0, 0, 255),      # 蓝
    (255, 255, 0),    # 黄
    (255, 0, 255),    # 品红
    (0, 255, 255),    # 青
    (255, 128, 0),    # 橙
    (128, 0, 255),    # 紫
    (0, 255, 128),    # 春绿
    (255, 0, 128),    # 玫红
    (128, 255, 0),    # 黄绿
    (0, 128, 255),    # 天蓝
]


def get_color_for_class(class_id: int) -> tuple:
    """
    获取类别对应的颜色
    
    Args:
        class_id: 类别ID
    
    Returns:
        RGB颜色元组
    """
    return SEGMENTATION_COLORS[class_id % len(SEGMENTATION_COLORS)]
