"""
SevSeg-YOLO 模型注册表 - 5个模型规模配置
基于 YOLO26 + ScoreHead, 支持 n/s/m/l/x 五种规模
所有模型使用 YOLO26 基础预训练权重 + 随机初始化的 ScoreHead
"""

from typing import Dict, Any, Optional, List


# 模型注册表
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "超小": {
        "name": "yolo26n-score",
        "scale_letter": "n",
        "model_yaml": "ultralytics/cfg/models/26/yolo26n-score.yaml",
        "pretrained_weights": "yolo26n.pt",
        "params": 2.57,
        "gflops": 5.3,
        "mAP50": 51.3,
        "score_mae": 1.317,
        "spearman_rho": 0.742,
        "input_size": 640,
        "description": "最轻量级，适合嵌入式/边缘设备部署",
        "recommended_batch": 64,
        "recommended_epochs": 105,
    },
    "小": {
        "name": "yolo26s-score",
        "scale_letter": "s",
        "model_yaml": "ultralytics/cfg/models/26/yolo26s-score.yaml",
        "pretrained_weights": "yolo26s.pt",
        "params": 10.19,
        "gflops": 20.8,
        "mAP50": 57.3,
        "score_mae": 1.306,
        "spearman_rho": 0.720,
        "input_size": 640,
        "description": "轻量级，适合轻量服务器",
        "recommended_batch": 32,
        "recommended_epochs": 105,
    },
    "中": {
        "name": "yolo26m-score",
        "scale_letter": "m",
        "model_yaml": "ultralytics/cfg/models/26/yolo26m-score.yaml",
        "pretrained_weights": "yolo26m.pt",
        "params": 22.19,
        "gflops": 68.5,
        "mAP50": 60.8,
        "score_mae": 1.316,
        "spearman_rho": 0.715,
        "input_size": 640,
        "description": "推荐：通用场景，性价比最高",
        "recommended_batch": 32,
        "recommended_epochs": 105,
    },
    "大": {
        "name": "yolo26l-score",
        "scale_letter": "l",
        "model_yaml": "ultralytics/cfg/models/26/yolo26l-score.yaml",
        "pretrained_weights": "yolo26l.pt",
        "params": 26.59,
        "gflops": 86.8,
        "mAP50": 62.6,
        "score_mae": 1.297,
        "spearman_rho": 0.709,
        "input_size": 640,
        "description": "高精度，适合服务器部署",
        "recommended_batch": 16,
        "recommended_epochs": 150,
    },
    "超大": {
        "name": "yolo26x-score",
        "scale_letter": "x",
        "model_yaml": "ultralytics/cfg/models/26/yolo26x-score.yaml",
        "pretrained_weights": "yolo26x.pt",
        "params": 56.08,
        "gflops": 194.8,
        "mAP50": 62.3,
        "score_mae": 1.224,
        "spearman_rho": 0.744,
        "input_size": 640,
        "description": "最高精度，需要大显存 (>=16GB)",
        "recommended_batch": 8,
        "recommended_epochs": 150,
    },
}

# 固定的Score训练参数 (实验已确认最优，不暴露给用户)
SCORE_CONFIG = {
    "score_loss": "gaussian_nll",
    "lambda_score": 0.05,
    "gaussian_sigma": 0.10,
    "mixup": 0.0,  # 必须关闭，MixUp破坏score语义
}

# 支持的输入尺寸
SUPPORTED_IMG_SIZES = [320, 416, 512, 640, 768, 1024, 1280]


def get_model_config(scale: str) -> Optional[Dict[str, Any]]:
    """获取模型配置"""
    return MODEL_REGISTRY.get(scale, None)


def get_all_scales() -> List[str]:
    """获取所有模型规模"""
    return list(MODEL_REGISTRY.keys())


def get_model_display_info(scale: str) -> str:
    """获取模型展示信息"""
    config = get_model_config(scale)
    if config is None:
        return "未找到模型配置"

    return (
        f"📊 模型: {config['name']}\n"
        f"📦 参数量: {config['params']}M\n"
        f"⚡ GFLOPs: {config['gflops']}\n"
        f"🎯 mAP@50: {config['mAP50']}%\n"
        f"📏 Score MAE: {config['score_mae']}\n"
        f"📈 Spearman ρ: {config['spearman_rho']}\n"
        f"📐 输入尺寸: {config['input_size']}×{config['input_size']}\n"
        f"💡 {config['description']}"
    )


def list_all_models() -> str:
    """列出所有可用模型"""
    lines = ["=" * 55, "可用 SevSeg-YOLO 模型列表", "=" * 55]

    for scale, config in MODEL_REGISTRY.items():
        lines.append(
            f"  {scale} ({config['scale_letter']}): {config['name']} "
            f"({config['params']}M, mAP50={config['mAP50']}%, MAE={config['score_mae']})"
        )
        lines.append(f"       {config['description']}")

    return "\n".join(lines)
