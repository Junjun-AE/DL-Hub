"""
模型工厂 - 创建和配置timm模型
支持本地预训练权重和timm自动下载
下载的权重会保存到pretrained文件夹供下次使用
"""

import os
import torch
import timm
from typing import Dict, Any, Optional, Tuple
from config.model_registry import MODEL_REGISTRY, get_model_config


def get_pretrained_dir() -> str:
    """
    获取预训练权重目录路径
    
    路径: Deep_learning_tools/pretrained_model/image_classification
    """
    # 获取 Deep_learning_tools 根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))  # models/
    package_dir = os.path.dirname(current_dir)  # model_image_classification/
    root_dir = os.path.dirname(package_dir)  # Deep_learning_tools/
    
    # 预训练模型目录
    pretrained_dir = os.path.join(root_dir, 'pretrained_model', 'image_classification')
    os.makedirs(pretrained_dir, exist_ok=True)
    return pretrained_dir


def get_pretrained_path(model_name: str) -> Optional[str]:
    """
    获取本地预训练权重路径
    
    Args:
        model_name: 模型名称，如 'efficientnet_b2'
    
    Returns:
        权重文件路径，如果不存在返回None
    """
    pretrained_dir = get_pretrained_dir()
    
    # 检查不同的文件命名格式
    possible_names = [
        f"{model_name}.pth",
        f"{model_name}.pt",
        f"{model_name}_pretrained.pth",
        f"{model_name}_imagenet.pth",
    ]
    
    for name in possible_names:
        path = os.path.join(pretrained_dir, name)
        if os.path.exists(path):
            return path
    
    return None


def save_pretrained_weights(model: torch.nn.Module, model_name: str) -> str:
    """
    保存预训练权重到pretrained文件夹
    
    Args:
        model: timm模型（已加载预训练权重）
        model_name: 模型名称
    
    Returns:
        保存的路径
    """
    pretrained_dir = get_pretrained_dir()
    save_path = os.path.join(pretrained_dir, f"{model_name}.pth")
    
    # 保存模型权重
    torch.save(model.state_dict(), save_path)
    
    return save_path


def download_and_cache_pretrained(model_name: str, num_classes: int = 1000) -> Tuple[torch.nn.Module, str]:
    """
    从timm下载预训练权重并缓存到本地
    
    Args:
        model_name: 模型名称
        num_classes: 原始分类数（ImageNet为1000）
    
    Returns:
        (model, weights_source): 模型和权重来源描述
    """
    pretrained_dir = get_pretrained_dir()
    cache_path = os.path.join(pretrained_dir, f"{model_name}.pth")
    
    # 检查是否已缓存
    if os.path.exists(cache_path):
        # 从本地缓存加载
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        state_dict = torch.load(cache_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        return model, f"本地缓存: {model_name}.pth"
    
    # 从timm下载
    print(f"📥 正在从timm下载预训练权重: {model_name}...")
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    
    # 保存到本地缓存
    torch.save(model.state_dict(), cache_path)
    print(f"💾 预训练权重已保存到: {cache_path}")
    
    return model, f"timm下载并缓存: {model_name}.pth"


def load_pretrained_weights(model: torch.nn.Module, weights_path: str) -> torch.nn.Module:
    """
    加载本地预训练权重
    
    Args:
        model: timm创建的模型
        weights_path: 权重文件路径
    
    Returns:
        加载了权重的模型
    """
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    # 处理不同的checkpoint格式
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # 移除可能的 'module.' 前缀 (DataParallel保存的模型)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    # 加载权重（允许部分匹配）
    model.load_state_dict(new_state_dict, strict=False)
    
    return model


def create_model(
    model_family: str,
    model_scale: str,
    num_classes: int,
    pretrained: bool = True,
    drop_rate: float = 0.2,
    drop_path_rate: float = 0.2,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    创建timm模型
    
    Args:
        model_family: 模型系列 ('EfficientNet', 'MobileNetV3', 'ResNet')
        model_scale: 模型规模 ('超小', '小', '中', '大', '超大')
        num_classes: 分类数量
        pretrained: 是否使用预训练权重
        drop_rate: Dropout比率
        drop_path_rate: DropPath比率
    
    Returns:
        (model, model_info): 模型和模型信息字典
    """
    # 获取模型配置
    config = get_model_config(model_family, model_scale)
    if config is None:
        raise ValueError(f"未找到模型配置: {model_family} - {model_scale}")
    
    model_name = config['name']
    
    # 检查本地预训练权重
    local_weights_path = get_pretrained_path(model_name)
    
    if pretrained:
        if local_weights_path is not None:
            # 使用本地权重
            print(f"📂 使用本地预训练权重: {local_weights_path}")
            model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=1000,  # 先用ImageNet类别数加载
                drop_rate=drop_rate,
            )
            model = load_pretrained_weights(model, local_weights_path)
            weights_source = f"本地权重: {os.path.basename(local_weights_path)}"
            
            # 替换分类头
            model = replace_classifier(model, num_classes)
        else:
            # 从timm下载并缓存
            model, weights_source = download_and_cache_pretrained(model_name, num_classes=1000)
            
            # 替换分类头
            model = replace_classifier(model, num_classes)
    else:
        # 不使用预训练权重
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
        weights_source = "随机初始化"
    
    # 获取模型的数据配置
    data_config = timm.data.resolve_model_data_config(model)
    
    # 构建模型信息
    model_info = {
        'model_name': model_name,
        'model_family': model_family,
        'model_scale': model_scale,
        'num_classes': num_classes,
        'params_m': config['params'],
        'reference_acc': config['acc'],
        'weights_source': weights_source,
        'input_size': data_config['input_size'],
        'mean': data_config['mean'],
        'std': data_config['std'],
        'interpolation': data_config['interpolation'],
    }
    
    return model, model_info


def replace_classifier(model: torch.nn.Module, num_classes: int) -> torch.nn.Module:
    """
    替换模型的分类头
    
    Args:
        model: timm模型
        num_classes: 新的分类数量
    
    Returns:
        替换了分类头的模型
    """
    # 获取当前分类器
    if hasattr(model, 'classifier'):
        # EfficientNet, MobileNetV3
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features, num_classes)
    elif hasattr(model, 'fc'):
        # ResNet, WideResNet
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    elif hasattr(model, 'head'):
        # 其他模型
        if hasattr(model.head, 'fc'):
            in_features = model.head.fc.in_features
            model.head.fc = torch.nn.Linear(in_features, num_classes)
        elif hasattr(model.head, 'in_features'):
            in_features = model.head.in_features
            model.head = torch.nn.Linear(in_features, num_classes)
    
    return model


def setup_model_for_training(
    model: torch.nn.Module,
    device: torch.device,
    gpu_ids: list = None,
    sync_bn: bool = True,
) -> torch.nn.Module:
    """
    配置模型用于训练（多GPU支持）
    
    Args:
        model: 模型
        device: 主设备
        gpu_ids: GPU ID列表，如 [0, 1, 2, 3]
        sync_bn: 是否同步BatchNorm
    
    Returns:
        配置好的模型
    """
    # 多GPU设置
    if gpu_ids is not None and len(gpu_ids) > 1:
        # 同步BatchNorm
        if sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        # DataParallel
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    
    # 移动到设备
    model = model.to(device)
    
    return model


def get_model_summary(model: torch.nn.Module) -> Dict[str, Any]:
    """
    获取模型摘要信息
    
    Args:
        model: 模型
    
    Returns:
        包含参数量等信息的字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_params_m': total_params / 1e6,
        'trainable_params_m': trainable_params / 1e6,
    }


def freeze_backbone(model: torch.nn.Module, freeze: bool = True) -> torch.nn.Module:
    """
    冻结/解冻主干网络
    
    Args:
        model: 模型
        freeze: True冻结，False解冻
    
    Returns:
        处理后的模型
    """
    # 获取实际模型（处理DataParallel包装）
    actual_model = model.module if hasattr(model, 'module') else model
    
    # 找到分类器层的名称
    classifier_names = ['classifier', 'fc', 'head']
    
    for name, param in actual_model.named_parameters():
        is_classifier = any(cn in name for cn in classifier_names)
        if not is_classifier:
            param.requires_grad = not freeze
    
    return model


def save_model_checkpoint(
    model: torch.nn.Module,
    model_info: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_acc: float,
    save_path: str,
    is_best: bool = False,
    training_config: Dict[str, Any] = None,
):
    """
    保存模型检查点（兼容 model_importer.py 格式）
    
    保存格式完全兼容 model_importer.py 中的 save_timm_model() 格式:
    {
        'model': state_dict,        # 模型权重
        'framework': 'timm',        # 框架标识
        'task': 'cls',              # 任务类型
        'model_name': 'xxx',        # 模型架构名称
        'num_classes': N,           # 类别数
        'input_size': 224,          # 输入尺寸
        'class_to_idx': {...},      # 类别映射
        ...
    }
    
    Args:
        model: 模型
        model_info: 模型信息（包含 class_to_idx）
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前epoch
        best_acc: 最佳准确率
        save_path: 保存路径
        is_best: 是否是最佳模型
        training_config: 训练配置
    """
    import time
    from datetime import datetime
    
    # 获取实际模型状态（处理DataParallel）
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    # 获取输入尺寸
    input_size = model_info.get('input_size', (3, 224, 224))
    if isinstance(input_size, (list, tuple)) and len(input_size) == 3:
        input_size_hw = input_size[1]  # 取 H 或 W
    else:
        input_size_hw = input_size
    
    # 获取类别映射
    class_to_idx = model_info.get('class_to_idx', {})
    idx_to_class = {v: k for k, v in class_to_idx.items()} if class_to_idx else {}
    
    # ===== 兼容 model_importer.py 的格式 =====
    checkpoint = {
        # ============ 核心字段（model_importer.py 必需）============
        'model': model_state,                       # 模型权重 (注意是 'model' 不是 'state_dict')
        'framework': 'timm',                        # 框架标识
        'task': 'cls',                              # 任务类型
        'model_name': model_info['model_name'],     # 模型架构名称
        'num_classes': model_info['num_classes'],   # 类别数
        'input_size': input_size_hw,                # 输入尺寸 (H/W)
        
        # ============ 类别映射（推理必需）============
        'class_to_idx': class_to_idx,               # 类别名 -> 索引
        'idx_to_class': idx_to_class,               # 索引 -> 类别名
        'class_names': list(class_to_idx.keys()) if class_to_idx else [],  # 类别名列表
        
        # ============ 预处理配置（推理必需）============
        'normalize_mean': model_info.get('mean', (0.485, 0.456, 0.406)),
        'normalize_std': model_info.get('std', (0.229, 0.224, 0.225)),
        'interpolation': model_info.get('interpolation', 'bilinear'),
        
        # ============ 额外兼容字段 ============
        'arch': model_info['model_name'],           # 架构名称（备用）
        'state_dict': model_state,                  # 兼容旧格式
        
        # ============ 训练状态（用于续训）============
        'training_state': {
            'epoch': epoch,
            'best_acc': best_acc,
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        },
        
        # ============ 模型元数据 ============
        'model_metadata': {
            'model_name': model_info['model_name'],
            'model_family': model_info.get('model_family', ''),
            'model_scale': model_info.get('model_scale', ''),
            'num_classes': model_info['num_classes'],
            'params_m': model_info.get('params_m', 0),
            'reference_acc': model_info.get('reference_acc', 0),
            'input_size': input_size,
            'mean': model_info.get('mean', (0.485, 0.456, 0.406)),
            'std': model_info.get('std', (0.229, 0.224, 0.225)),
        },
        
        # ============ 训练配置 ============
        'training_config': training_config or {},
        
        # ============ 保存信息 ============
        'save_info': {
            'is_best': is_best,
            'save_time': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
        },
    }
    
    torch.save(checkpoint, save_path)


def load_checkpoint_for_resume(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler = None,
) -> Tuple[torch.nn.Module, int, float]:
    """
    加载检查点用于续训
    
    Args:
        checkpoint_path: 检查点路径
        model: 模型
        optimizer: 优化器
        scheduler: 调度器
    
    Returns:
        (model, start_epoch, best_acc)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 加载模型权重
    model_state = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(model_state, strict=False)
    
    # 加载优化器状态
    if optimizer and 'training_state' in checkpoint:
        optimizer_state = checkpoint['training_state'].get('optimizer_state_dict')
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
    
    # 加载调度器状态
    if scheduler and 'training_state' in checkpoint:
        scheduler_state = checkpoint['training_state'].get('scheduler_state_dict')
        if scheduler_state:
            scheduler.load_state_dict(scheduler_state)
    
    # 获取训练状态
    start_epoch = 0
    best_acc = 0.0
    if 'training_state' in checkpoint:
        start_epoch = checkpoint['training_state'].get('epoch', 0) + 1
        best_acc = checkpoint['training_state'].get('best_acc', 0.0)
    
    return model, start_epoch, best_acc
