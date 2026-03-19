"""
模型注册表 - 15个预训练模型配置
包含 EfficientNet, MobileNetV3, ResNet 三个系列
所有模型均在timm中有预训练权重
"""

from typing import Dict, Any, Optional, List

# 模型注册表
MODEL_REGISTRY: Dict[str, Dict[str, Dict[str, Any]]] = {
    "EfficientNet": {
        "超小": {
            "name": "efficientnet_b0",
            "params": 5.3,
            "acc": 77.1,
            "input_size": 224,
            "description": "轻量级，适合移动端部署",
        },
        "小": {
            "name": "efficientnet_b1",
            "params": 7.8,
            "acc": 79.1,
            "input_size": 240,
            "description": "平衡性能与效率",
        },
        "中": {
            "name": "efficientnet_b2",
            "params": 9.2,
            "acc": 80.1,
            "input_size": 260,
            "description": "推荐：性价比最高",
        },
        "大": {
            "name": "efficientnet_b3",
            "params": 12.0,
            "acc": 81.6,
            "input_size": 300,
            "description": "高精度，适合服务器",
        },
        "超大": {
            "name": "efficientnet_b4",
            "params": 19.0,
            "acc": 82.9,
            "input_size": 380,
            "description": "最高精度，需要较多显存",
        },
    },
    "MobileNetV3": {
        "超小": {
            "name": "mobilenetv3_small_050",
            "params": 1.6,
            "acc": 57.9,
            "input_size": 224,
            "description": "极轻量，适合嵌入式设备",
        },
        "小": {
            "name": "mobilenetv3_small_100",
            "params": 2.5,
            "acc": 67.4,
            "input_size": 224,
            "description": "轻量级移动端方案",
        },
        "中": {
            "name": "mobilenetv2_100",
            "params": 3.5,
            "acc": 72.0,
            "input_size": 224,
            "description": "移动端性价比之选",
        },
        "大": {
            "name": "mobilenetv3_large_100",
            "params": 5.4,
            "acc": 75.2,
            "input_size": 224,
            "description": "移动端最佳精度",
        },
        "超大": {
            "name": "tf_mobilenetv3_large_100",
            "params": 5.4,
            "acc": 75.5,
            "input_size": 224,
            "description": "TF版本，略高精度",
        },
    },
    "ResNet": {
        "超小": {
            "name": "resnet18",
            "params": 11.7,
            "acc": 69.8,
            "input_size": 224,
            "description": "轻量ResNet，快速训练",
        },
        "小": {
            "name": "resnet34",
            "params": 21.8,
            "acc": 73.3,
            "input_size": 224,
            "description": "经典ResNet34",
        },
        "中": {
            "name": "resnet50",
            "params": 25.6,
            "acc": 80.4,
            "input_size": 224,
            "description": "经典ResNet50，推荐",
        },
        "大": {
            "name": "wide_resnet50_2",
            "params": 68.9,
            "acc": 81.5,
            "input_size": 224,
            "description": "宽残差网络，高精度",
        },
        "超大": {
            "name": "wide_resnet101_2",
            "params": 126.9,
            "acc": 82.5,
            "input_size": 224,
            "description": "最强ResNet，需要大显存",
        },
    },
}


def get_model_config(family: str, scale: str) -> Optional[Dict[str, Any]]:
    """
    获取模型配置
    
    Args:
        family: 模型系列名称
        scale: 模型规模
    
    Returns:
        模型配置字典，不存在返回None
    """
    if family in MODEL_REGISTRY and scale in MODEL_REGISTRY[family]:
        return MODEL_REGISTRY[family][scale]
    return None


def get_all_families() -> List[str]:
    """获取所有模型系列"""
    return list(MODEL_REGISTRY.keys())


def get_model_families() -> List[str]:
    """获取所有模型系列（别名）"""
    return list(MODEL_REGISTRY.keys())


def get_all_scales() -> List[str]:
    """获取所有模型规模"""
    return ["超小", "小", "中", "大", "超大"]


def get_model_scales(family: str) -> List[str]:
    """
    获取指定系列的模型规模
    
    Args:
        family: 模型系列名称
    
    Returns:
        该系列支持的规模列表
    """
    if family in MODEL_REGISTRY:
        return list(MODEL_REGISTRY[family].keys())
    return []


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
        f"🎯 参考精度: {config['acc']}%\n"
        f"📐 输入尺寸: {config['input_size']}×{config['input_size']}\n"
        f"💡 {config['description']}"
    )


def list_all_models() -> str:
    """列出所有可用模型"""
    lines = ["=" * 50, "可用模型列表", "=" * 50]
    
    for family, scales in MODEL_REGISTRY.items():
        lines.append(f"\n【{family}】")
        for scale, config in scales.items():
            lines.append(
                f"  {scale}: {config['name']} "
                f"({config['params']}M, {config['acc']}%)"
            )
    
    return "\n".join(lines)
