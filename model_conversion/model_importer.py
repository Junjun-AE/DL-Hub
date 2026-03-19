"""
🚀 Stage 1: 模型导入器 (Model Importer)
========================================

将 PyTorch 模型文件 (.pth/.pt) 导入并重建为可用的 nn.Module。

支持的框架和任务:
- 图像分类 (cls): timm (WideResNet, MobileNetV3, EfficientNet, ViT)
- 目标检测 (det): ultralytics (YOLO v5/v8/v11/v26)
- 语义分割 (seg): mmsegmentation (SegFormer B0-B5)

支持的分类模型:
- WideResNet: wide_resnet50_2, wide_resnet101_2, wide_resnet50_4, wide_resnet101_4, wide_resnet200_2
- MobileNetV3: mobilenetv3_small_050, mobilenetv3_small_100, mobilenetv3_large_075, mobilenetv3_large_100, tf_mobilenetv3_large_100
- EfficientNet: efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4
- VisionTransformer: vit_tiny_patch16_224, vit_small_patch16_224, vit_base_patch16_224, vit_large_patch16_224, vit_huge_patch14_224

支持的分割模型:
- SegFormer: segformer_b0, segformer_b1, segformer_b2, segformer_b3, segformer_b4, segformer_b5

使用方法:
---------
>>> from model_importer import import_model
>>> info = import_model('model.pth', task_type='cls')
>>> model = info.model
>>> print(info.summary())

模块结构:
---------
1. 数据结构与异常 (DataStructures)
2. 模型处理器基类 (BaseHandler)
3. 分类模型处理器 (ClassificationHandlers) - timm 限定模型
4. 检测模型处理器 (DetectionHandlers) - 仅 YOLO
5. 分割模型处理器 (SegmentationHandlers) - SegFormer (MMSegmentation)
6. 主导入器 (ModelImporter)
7. 便捷函数与CLI
8. 简化测试接口

作者: Model Converter Team
版本: 2.1.0 (SegFormer 专用版)
"""

import os
import sys
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import torch
import torch.nn as nn
from unified_logger import Logger, console, Timer


# ==================== 日志配置 ====================

logger = Logger.get("model_importer")


# ==============================================================================
# 第1部分: 数据结构与异常 (DataStructures & Exceptions)
# ==============================================================================

class TaskType(Enum):
    """任务类型"""
    CLASSIFICATION = "cls"
    DETECTION = "det"
    SEGMENTATION = "seg"


class Framework(Enum):
    """深度学习框架"""
    TIMM = "timm"
    MMSEGMENTATION = "mmsegmentation"  # SegFormer 语义分割
    ULTRALYTICS = "ultralytics"
    TORCHVISION = "torchvision"
    UNKNOWN = "unknown"


@dataclass
class InputSpec:
    """输入规格"""
    shape: Tuple[int, ...]                              # (B, C, H, W)
    dtype: torch.dtype = torch.float32
    channel_order: str = "RGB"
    normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    value_range: Tuple[float, float] = (0.0, 1.0)


@dataclass
class OutputSpec:
    """输出规格"""
    structure: str                                       # 输出结构描述
    num_classes: Optional[int] = None
    output_type: str = "tensor"                          # tensor / list / dict
    extra_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInfo:
    """模型信息"""
    model: nn.Module
    task_type: TaskType
    framework: Framework
    device: torch.device
    input_spec: InputSpec
    output_spec: OutputSpec
    architecture: str
    model_path: str
    config: Optional[str] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> str:
        """打印模型摘要"""
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return f"""
╔══════════════════════════════════════════════════════════════════════╗
║                       📊 Model Info Summary                          ║
╠══════════════════════════════════════════════════════════════════════╣
  📁 Model Path    : {self.model_path}
  🏷️  Task Type     : {self.task_type.value} ({self.task_type.name})
  🔧 Architecture  : {self.architecture}
  📚 Framework     : {self.framework.value}
  💻 Device        : {self.device}
  
  📥 Input Spec:
     - Shape       : {self.input_spec.shape}
     - Normalize   : mean={self.input_spec.normalize_mean}, std={self.input_spec.normalize_std}
  
  📤 Output Spec:
     - Structure   : {self.output_spec.structure}
     - Num Classes : {self.output_spec.num_classes}
  
  📊 Parameters    : {param_count:,} total ({trainable:,} trainable)
  ✅ ONNX Ready    : {self.extra_metadata.get('onnx_ready', False)}
╚══════════════════════════════════════════════════════════════════════╝
"""


# ==================== 异常定义 ====================

class ModelImportError(Exception):
    """模型导入错误基类"""
    pass


class UnsupportedModelError(ModelImportError):
    """不支持的模型架构"""
    def __init__(self, message: str, task_type: str = None, supported: List[str] = None):
        full_msg = f"❌ 不支持的模型: {message}"
        if supported:
            full_msg += f"\n支持的模型: {', '.join(supported)}"
        super().__init__(full_msg)


class ModelRebuildError(ModelImportError):
    """模型重建失败"""
    def __init__(self, architecture: str, reason: str):
        super().__init__(f"❌ 模型重建失败 [{architecture}]: {reason}")


class NumClassesInferenceError(ModelImportError):
    """无法推断类别数"""
    def __init__(self, message: str, tried_methods: List[str] = None):
        full_msg = f"❌ {message}"
        if tried_methods:
            full_msg += "\n已尝试:\n" + "\n".join(f"  - {m}" for m in tried_methods)
        full_msg += "\n\n💡 解决方案: 请使用 num_classes 参数手动指定"
        super().__init__(full_msg)


class ForwardVerificationError(ModelImportError):
    """前向传播验证失败"""
    def __init__(self, architecture: str, reason: str):
        super().__init__(f"❌ 前向验证失败 [{architecture}]: {reason}")


class FrameworkNotInstalledError(ModelImportError):
    """框架未安装"""
    def __init__(self, framework: str, install_cmd: str):
        super().__init__(
            f"❌ 框架未安装: {framework}\n"
            f"💡 安装命令: {install_cmd}"
        )


# ==============================================================================
# 第2部分: 模型处理器基类 (BaseHandler)
# ==============================================================================

class ModelHandler(ABC):
    """
    模型处理器抽象基类
    ==================
    所有模型处理器必须继承此类并实现所有抽象方法。
    """
    
    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        """返回任务类型"""
        pass
    
    @property
    @abstractmethod
    def framework(self) -> Framework:
        """返回框架类型"""
        pass
    
    @property
    @abstractmethod
    def supported_architectures(self) -> List[str]:
        """返回支持的架构列表"""
        pass
    
    @abstractmethod
    def can_handle(self, checkpoint: Dict[str, Any]) -> bool:
        """判断是否能处理这个 checkpoint"""
        pass
    
    @abstractmethod
    def get_model_name(self, checkpoint: Dict[str, Any]) -> str:
        """获取模型名称/架构"""
        pass
    
    @abstractmethod
    def rebuild(self, checkpoint: Dict[str, Any], 
                num_classes: Optional[int] = None,
                **kwargs) -> nn.Module:
        """重建模型"""
        pass
    
    @abstractmethod
    def get_num_classes(self, checkpoint: Dict[str, Any]) -> int:
        """获取类别数"""
        pass
    
    @abstractmethod
    def get_input_spec(self, checkpoint: Dict[str, Any]) -> InputSpec:
        """获取输入规格"""
        pass
    
    @abstractmethod
    def get_output_spec(self, checkpoint: Dict[str, Any], 
                        num_classes: int) -> OutputSpec:
        """获取输出规格"""
        pass
    
    def verify_forward(
        self,
        model: nn.Module,
        input_spec: InputSpec,
        device: torch.device,
        sample_input: Optional[torch.Tensor] = None,
    ) -> bool:
        """
        验证前向传播
        """
        try:
            model.eval()
            
            if sample_input is not None:
                x = sample_input.to(device)
            else:
                x = torch.rand(input_spec.shape, dtype=input_spec.dtype).to(device)
            
            with torch.no_grad():
                _ = model(x)
            return True
        except Exception as e:
            raise ForwardVerificationError(
                architecture=type(model).__name__,
                reason=str(e)
            )


# ==============================================================================
# 第3部分: 分类模型处理器 (ClassificationHandlers)
# ==============================================================================

class TimmClassificationHandler(ModelHandler):
    """
    Timm 分类模型处理器 (限定版)
    ============================
    
    仅支持以下 timm 库中的分类模型:
    
    WideResNet 系列:
    - wide_resnet50_2   (68.9M, 78.5%)
    - wide_resnet101_2  (126.9M, 79.3%)
    - wide_resnet50_4   (267.5M, 79.8%)  [需要 timm 支持]
    - wide_resnet101_4  (498.2M, 80.1%)  [需要 timm 支持]
    - wide_resnet200_2  (126.4M, 80.4%)  [需要 timm 支持]
    
    MobileNetV3 系列:
    - mobilenetv3_small_050  (1.6M, 56.6%)
    - mobilenetv3_small_100  (2.9M, 67.4%)
    - mobilenetv3_large_075  (4.0M, 73.3%)
    - mobilenetv3_large_100  (5.4M, 75.2%)
    - tf_mobilenetv3_large_100 (5.4M, 75.2%)
    
    EfficientNet 系列:
    - efficientnet_b0  (5.3M, 77.1%)
    - efficientnet_b1  (7.8M, 79.1%)
    - efficientnet_b2  (9.2M, 80.1%)
    - efficientnet_b3  (12.0M, 81.6%)
    - efficientnet_b4  (19.0M, 82.9%)
    
    Vision Transformer 系列:
    - vit_tiny_patch16_224   (5.5M, 72.2%)
    - vit_small_patch16_224  (22.0M, 79.8%)
    - vit_base_patch16_224   (86.0M, 81.8%)
    - vit_large_patch16_224  (307.0M, 82.6%)
    - vit_huge_patch14_224   (632.0M, 83.2%)
    
    Checkpoint 格式要求:
    {
        'model': state_dict,
        'framework': 'timm',
        'task': 'cls',
        'model_name': 'efficientnet_b0',
        'num_classes': 10,
        'input_size': 224,
    }
    """
    
    # 支持的模型列表（限定范围）
    SUPPORTED_MODELS = {
        # WideResNet 系列
        'wide_resnet50_2': {'params': 68.9, 'input_size': 224, 'acc': 78.5, 'family': 'WideResNet', 'size': '超小'},
        'wide_resnet101_2': {'params': 126.9, 'input_size': 224, 'acc': 79.3, 'family': 'WideResNet', 'size': '小'},
        'wide_resnet50_4': {'params': 267.5, 'input_size': 224, 'acc': 79.8, 'family': 'WideResNet', 'size': '中'},
        'wide_resnet101_4': {'params': 498.2, 'input_size': 224, 'acc': 80.1, 'family': 'WideResNet', 'size': '大'},
        'wide_resnet200_2': {'params': 126.4, 'input_size': 224, 'acc': 80.4, 'family': 'WideResNet', 'size': '超大'},
        
        # MobileNetV3 系列
        'mobilenetv3_small_050': {'params': 1.6, 'input_size': 224, 'acc': 56.6, 'family': 'MobileNetV3', 'size': '超小'},
        'mobilenetv3_small_100': {'params': 2.9, 'input_size': 224, 'acc': 67.4, 'family': 'MobileNetV3', 'size': '小'},
        'mobilenetv3_large_075': {'params': 4.0, 'input_size': 224, 'acc': 73.3, 'family': 'MobileNetV3', 'size': '中'},
        'mobilenetv3_large_100': {'params': 5.4, 'input_size': 224, 'acc': 75.2, 'family': 'MobileNetV3', 'size': '大'},
        'tf_mobilenetv3_large_100': {'params': 5.4, 'input_size': 224, 'acc': 75.2, 'family': 'MobileNetV3', 'size': '超大'},
        
        # EfficientNet 系列
        'efficientnet_b0': {'params': 5.3, 'input_size': 224, 'acc': 77.1, 'family': 'EfficientNet', 'size': '超小'},
        'efficientnet_b1': {'params': 7.8, 'input_size': 240, 'acc': 79.1, 'family': 'EfficientNet', 'size': '小'},
        'efficientnet_b2': {'params': 9.2, 'input_size': 260, 'acc': 80.1, 'family': 'EfficientNet', 'size': '中'},
        'efficientnet_b3': {'params': 12.0, 'input_size': 300, 'acc': 81.6, 'family': 'EfficientNet', 'size': '大'},
        'efficientnet_b4': {'params': 19.0, 'input_size': 380, 'acc': 82.9, 'family': 'EfficientNet', 'size': '超大'},
        
        # Vision Transformer 系列
        'vit_tiny_patch16_224': {'params': 5.5, 'input_size': 224, 'acc': 72.2, 'family': 'VisionTransformer', 'size': '超小'},
        'vit_small_patch16_224': {'params': 22.0, 'input_size': 224, 'acc': 79.8, 'family': 'VisionTransformer', 'size': '小'},
        'vit_base_patch16_224': {'params': 86.0, 'input_size': 224, 'acc': 81.8, 'family': 'VisionTransformer', 'size': '中'},
        'vit_large_patch16_224': {'params': 307.0, 'input_size': 224, 'acc': 82.6, 'family': 'VisionTransformer', 'size': '大'},
        'vit_huge_patch14_224': {'params': 632.0, 'input_size': 224, 'acc': 83.2, 'family': 'VisionTransformer', 'size': '超大'},
    }
    
    # 模型族别名映射
    FAMILY_ALIASES = {
        # WideResNet
        'wideresnet': 'wide_resnet',
        'wide-resnet': 'wide_resnet',
        'wrn': 'wide_resnet',
        # MobileNetV3
        'mobilenet': 'mobilenetv3',
        'mobilenet_v3': 'mobilenetv3',
        'mnv3': 'mobilenetv3',
        # EfficientNet
        'effnet': 'efficientnet',
        'efficient_net': 'efficientnet',
        # ViT
        'vit': 'vit',
        'vision_transformer': 'vit',
        'transformer': 'vit',
    }
    
    def __init__(self):
        self._timm = None
    
    def reset(self):
        """重置handler状态，清理缓存"""
        self._timm = None
        import gc
        gc.collect()
    
    @property
    def timm(self):
        """延迟加载 timm"""
        if self._timm is None:
            try:
                import timm
                self._timm = timm
            except ImportError:
                raise FrameworkNotInstalledError("timm", "pip install timm")
        return self._timm
    
    @property
    def task_type(self) -> TaskType:
        return TaskType.CLASSIFICATION
    
    @property
    def framework(self) -> Framework:
        return Framework.TIMM
    
    @property
    def supported_architectures(self) -> List[str]:
        """返回支持的模型列表"""
        return list(self.SUPPORTED_MODELS.keys())
    
    @classmethod
    def get_supported_models_info(cls) -> Dict[str, Dict]:
        """获取支持的模型详细信息"""
        return cls.SUPPORTED_MODELS.copy()
    
    @classmethod
    def get_models_by_family(cls, family: str) -> List[str]:
        """根据模型族获取模型列表"""
        family_lower = family.lower()
        # 处理别名
        for alias, canonical in cls.FAMILY_ALIASES.items():
            if alias in family_lower:
                family_lower = canonical
                break
        
        return [
            name for name, info in cls.SUPPORTED_MODELS.items()
            if info['family'].lower().replace(' ', '_') == family_lower or
               info['family'].lower() == family_lower
        ]
    
    @classmethod
    def is_model_supported(cls, model_name: str) -> bool:
        """检查模型是否在支持列表中"""
        return model_name.lower() in [m.lower() for m in cls.SUPPORTED_MODELS.keys()]
    
    def can_handle(self, checkpoint: Dict[str, Any]) -> bool:
        """检查是否是支持的 timm 模型"""
        # 显式标记
        if checkpoint.get('framework') == 'timm':
            model_name = checkpoint.get('model_name', '')
            if model_name and self.is_model_supported(model_name):
                return True
            # 如果没有 model_name，尝试推断
            if not model_name:
                return True  # 让后续推断处理
        
        if checkpoint.get('task') == 'cls' and 'model_name' in checkpoint:
            if self.is_model_supported(checkpoint['model_name']):
                return True
        
        # 检查 state_dict 结构
        state_dict = checkpoint.get('model', checkpoint)
        if isinstance(state_dict, dict):
            keys = set(state_dict.keys())
            
            # EfficientNet (timm 格式)
            if 'conv_stem.weight' in keys and 'classifier.weight' in keys:
                return True
            
            # Wide ResNet / ResNet
            if 'conv1.weight' in keys and 'fc.weight' in keys:
                if any('layer4' in k for k in keys):
                    return True
            
            # ViT
            if 'cls_token' in keys and 'patch_embed.proj.weight' in keys:
                return True
            
            # MobileNetV3
            if 'conv_stem.weight' in keys or 'blocks.0.0.conv.weight' in keys:
                return True
        
        return False
    
    def get_model_name(self, checkpoint: Dict[str, Any], model_path: str = None) -> str:
        """
        获取模型名称
        
        优先级:
        1. checkpoint 中的 model_name (最可靠，训练时保存)
        2. checkpoint['model_metadata']['model_name'] (备用位置)
        3. checkpoint['arch'] (兼容字段)
        4. 从文件名推断
        5. 从 state_dict 推断 (最后手段，可能不准确)
        
        重要: 如果 checkpoint 中有 model_name，应该始终优先使用！
        因为这是训练时保存的准确信息，比推断更可靠。
        """
        # ============================================================
        # 调试: 打印 checkpoint 中的关键字段
        # ============================================================
        logger.debug("=" * 60)
        logger.debug("🔍 检查 checkpoint 中的模型标识字段:")
        
        # 检查所有可能包含 model_name 的位置
        possible_model_names = []
        
        # 位置1: 顶层 model_name
        if 'model_name' in checkpoint:
            name = checkpoint['model_name']
            logger.debug(f"   📌 checkpoint['model_name'] = '{name}'")
            possible_model_names.append(('model_name', name))
        
        # 位置2: model_metadata.model_name
        if 'model_metadata' in checkpoint and isinstance(checkpoint['model_metadata'], dict):
            if 'model_name' in checkpoint['model_metadata']:
                name = checkpoint['model_metadata']['model_name']
                logger.debug(f"   📌 checkpoint['model_metadata']['model_name'] = '{name}'")
                possible_model_names.append(('model_metadata.model_name', name))
        
        # 位置3: arch 字段 (兼容)
        if 'arch' in checkpoint:
            name = checkpoint['arch']
            logger.debug(f"   📌 checkpoint['arch'] = '{name}'")
            possible_model_names.append(('arch', name))
        
        # 位置4: framework 和 task 信息
        if 'framework' in checkpoint:
            logger.debug(f"   📌 checkpoint['framework'] = '{checkpoint['framework']}'")
        if 'task' in checkpoint:
            logger.debug(f"   📌 checkpoint['task'] = '{checkpoint['task']}'")
        
        logger.debug("=" * 60)
        
        # ============================================================
        # Step 1: 优先使用 checkpoint 中保存的 model_name
        # ============================================================
        for source, model_name in possible_model_names:
            if model_name and isinstance(model_name, str):
                # 标准化模型名称
                model_name_normalized = model_name.strip().lower()
                
                # 检查是否在支持列表中
                if self.is_model_supported(model_name):
                    logger.info(f"✅ 使用 checkpoint 中保存的模型名称: {model_name} (来源: {source})")
                    return model_name
                
                # 尝试模糊匹配（处理大小写和下划线差异）
                for supported_name in self.SUPPORTED_MODELS.keys():
                    if supported_name.lower() == model_name_normalized:
                        logger.info(f"✅ 模糊匹配成功: {model_name} -> {supported_name} (来源: {source})")
                        return supported_name
                
                logger.warning(f"⚠️ checkpoint 中的模型 '{model_name}' (来源: {source}) 不在支持列表中")
        
        # ============================================================
        # Step 2: 从文件名推断
        # ============================================================
        if model_path:
            inferred = self._infer_from_filename(model_path)
            if inferred and self.is_model_supported(inferred):
                logger.info(f"📝 从文件名推断模型架构: {inferred}")
                logger.warning("⚠️ 建议: 请确保模型保存时包含正确的 model_name 字段")
                return inferred
        
        # ============================================================
        # Step 3: 从 state_dict 推断 (最后手段)
        # ============================================================
        state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        if isinstance(state_dict, nn.Module):
            state_dict = state_dict.state_dict()
        
        if isinstance(state_dict, dict):
            inferred = self._infer_model_name(state_dict)
            if inferred and self.is_model_supported(inferred):
                logger.info(f"🔬 从 state_dict 结构推断模型架构: {inferred}")
                logger.warning("⚠️ 警告: 从权重结构推断模型名称可能不准确！")
                logger.warning("⚠️ 强烈建议: 请重新训练模型并确保保存时包含正确的 model_name 字段")
                return inferred
        
        # ============================================================
        # 无法推断，返回默认值并给出详细警告
        # ============================================================
        logger.error("❌ 无法识别模型架构！")
        logger.error("   可能原因:")
        logger.error("   1. checkpoint 中没有保存 model_name 字段")
        logger.error("   2. 模型架构不在支持列表中")
        logger.error("   3. 权重结构与已知模型不匹配")
        logger.error("   ")
        logger.error("   解决方案:")
        logger.error("   1. 检查训练代码，确保保存模型时包含 model_name 字段")
        logger.error("   2. 使用正确的模型名称重新保存 checkpoint")
        logger.warning("⚠️ 将尝试使用默认配置 (efficientnet_b0)，这可能导致转换失败！")
        return 'efficientnet_b0'
    
    def _infer_from_filename(self, model_path: str) -> Optional[str]:
        """
        从文件名推断模型名称
        
        重要设计原则：
        1. 更具体的模式优先匹配（如 mobilenetv3_large_100 优先于 mobilenetv3）
        2. 模型族之间不能互相干扰（MobileNetV3 不能被识别为 EfficientNet）
        3. 支持用户的常见命名习惯（如 MobileNetV3_b1.pth）
        """
        filename = os.path.basename(model_path).lower()
        filename_no_ext = os.path.splitext(filename)[0]  # 去除扩展名
        
        # ============================================================
        # Step 1: 精确匹配 SUPPORTED_MODELS 中的模型名
        # ============================================================
        for model_name in self.SUPPORTED_MODELS.keys():
            normalized = model_name.lower().replace('_', '')
            filename_normalized = filename_no_ext.replace('_', '').replace('-', '')
            # 完全匹配或作为前缀匹配
            if filename_normalized == normalized or filename_normalized.startswith(normalized):
                return model_name
        
        # ============================================================
        # Step 2: 模式匹配 - 按模型族分组，确保不会跨族误识别
        # 
        # 关键：先检测模型族，再在族内匹配具体型号
        # ============================================================
        
        # 检测模型族
        family = None
        if 'mobilenetv3' in filename or 'mobilenet_v3' in filename or 'mnv3' in filename:
            family = 'mobilenetv3'
        elif 'efficientnet' in filename or 'effnet' in filename:
            family = 'efficientnet'
        elif 'wide_resnet' in filename or 'wideresnet' in filename or 'wrn' in filename:
            family = 'wideresnet'
        elif 'vit' in filename or 'vision_transformer' in filename:
            family = 'vit'
        
        # ============================================================
        # Step 3: 在确定的模型族内进行具体型号匹配
        # ============================================================
        
        if family == 'mobilenetv3':
            # MobileNetV3 模式匹配（从大到小）
            mobilenetv3_patterns = [
                ('tf_mobilenetv3_large_100', ['tf_mobilenetv3_large', 'tfmobilenetv3_large', 'tf_mnv3_large']),
                ('mobilenetv3_large_100', ['mobilenetv3_large_100', 'mobilenetv3large100', 'mnv3_large_100', 
                                           'mobilenetv3_large', 'mobilenetv3large', 'mnv3_large']),
                ('mobilenetv3_large_075', ['mobilenetv3_large_075', 'mobilenetv3large075', 'mnv3_large_075']),
                ('mobilenetv3_small_100', ['mobilenetv3_small_100', 'mobilenetv3small100', 'mnv3_small_100',
                                           'mobilenetv3_small', 'mobilenetv3small', 'mnv3_small']),
                ('mobilenetv3_small_050', ['mobilenetv3_small_050', 'mobilenetv3small050', 'mnv3_small_050']),
            ]
            for model_name, patterns in mobilenetv3_patterns:
                for pattern in patterns:
                    if pattern in filename:
                        return model_name
            # 默认返回 large_100（最常用）
            return 'mobilenetv3_large_100'
        
        elif family == 'efficientnet':
            # EfficientNet 模式匹配（从大到小）
            efficientnet_patterns = [
                ('efficientnet_b4', ['efficientnet_b4', 'efficientnetb4', 'effnet_b4', 'effnetb4']),
                ('efficientnet_b3', ['efficientnet_b3', 'efficientnetb3', 'effnet_b3', 'effnetb3']),
                ('efficientnet_b2', ['efficientnet_b2', 'efficientnetb2', 'effnet_b2', 'effnetb2']),
                ('efficientnet_b1', ['efficientnet_b1', 'efficientnetb1', 'effnet_b1', 'effnetb1']),
                ('efficientnet_b0', ['efficientnet_b0', 'efficientnetb0', 'effnet_b0', 'effnetb0']),
            ]
            for model_name, patterns in efficientnet_patterns:
                for pattern in patterns:
                    if pattern in filename:
                        return model_name
            # 默认返回 b0（最常用）
            return 'efficientnet_b0'
        
        elif family == 'wideresnet':
            # WideResNet 模式匹配
            wideresnet_patterns = [
                ('wide_resnet200_2', ['wide_resnet200_2', 'wideresnet200_2', 'wideresnet200', 'wrn200']),
                ('wide_resnet101_4', ['wide_resnet101_4', 'wideresnet101_4']),
                ('wide_resnet101_2', ['wide_resnet101_2', 'wideresnet101_2', 'wideresnet101', 'wrn101']),
                ('wide_resnet50_4', ['wide_resnet50_4', 'wideresnet50_4']),
                ('wide_resnet50_2', ['wide_resnet50_2', 'wideresnet50_2', 'wideresnet50', 'wrn50', 
                                     'wide_resnet50', 'wideresnet_50']),
            ]
            for model_name, patterns in wideresnet_patterns:
                for pattern in patterns:
                    if pattern in filename:
                        return model_name
            # 默认返回 50_2（最常用）
            return 'wide_resnet50_2'
        
        elif family == 'vit':
            # ViT 模式匹配
            vit_patterns = [
                ('vit_huge_patch14_224', ['vit_huge', 'vith_', 'vit_h_', 'vithuge']),
                ('vit_large_patch16_224', ['vit_large', 'vitl_', 'vit_l_', 'vitlarge']),
                ('vit_base_patch16_224', ['vit_base', 'vitb_', 'vit_b_', 'vitbase', 'vit_base_patch16']),
                ('vit_small_patch16_224', ['vit_small', 'vits_', 'vit_s_', 'vitsmall']),
                ('vit_tiny_patch16_224', ['vit_tiny', 'vitt_', 'vit_t_', 'vittiny']),
            ]
            for model_name, patterns in vit_patterns:
                for pattern in patterns:
                    if pattern in filename:
                        return model_name
            # 默认返回 base（最常用）
            return 'vit_base_patch16_224'
        
        return None
    
    def _infer_model_name(self, state_dict: Dict[str, torch.Tensor]) -> str:
        """
        从 state_dict 推断模型名称
        
        重要设计原则：
        1. 使用模型的独特结构特征进行区分
        2. MobileNetV3 和 EfficientNet 结构相似，需要精确区分
        3. 通过通道数和层结构来确定具体型号
        
        模型结构差异：
        - MobileNetV3: blocks.X.Y 结构，有 conv_pw/conv_dw/conv_pwl，最后一层是 conv_head
        - EfficientNet: blocks.X.Y 结构，有 conv_pw/conv_dw/conv_pwl，最后一层是 conv_head
        - WideResNet: layer1-4 结构，标准 ResNet 风格
        - ViT: cls_token + patch_embed + blocks 结构
        
        区分 MobileNetV3 和 EfficientNet 的关键：
        1. 检查 blocks.0.0 的结构（MobileNetV3 的第一个 block 是 ConvBnAct，没有 SE）
        2. 检查 conv_head 的通道数
        3. 检查 blocks 的深度
        """
        keys = set(state_dict.keys())
        keys_str = ' '.join(keys)
        
        # ============================================================
        # 1. ViT 检测（最独特的结构）
        # ============================================================
        if 'cls_token' in keys and 'patch_embed.proj.weight' in keys:
            if 'blocks.0.attn.qkv.weight' in keys:
                qkv_shape = state_dict['blocks.0.attn.qkv.weight'].shape
                embed_dim = qkv_shape[1]
                
                dim_to_model = {
                    192: 'vit_tiny_patch16_224',
                    384: 'vit_small_patch16_224',
                    768: 'vit_base_patch16_224',
                    1024: 'vit_large_patch16_224',
                    1280: 'vit_huge_patch14_224',
                }
                model = dim_to_model.get(embed_dim, 'vit_base_patch16_224')
                if self.is_model_supported(model):
                    return model
            return 'vit_base_patch16_224'
        
        # ============================================================
        # 2. WideResNet 检测（ResNet 风格结构）
        # ============================================================
        # WideResNet 各变体特征:
        # | 变体            | layer1.0.conv1 ch | layer3 深度 | 宽度倍数 |
        # |-----------------|-------------------|-------------|----------|
        # | wide_resnet50_2 | 128               | 6 blocks    | 2x       |
        # | wide_resnet101_2| 128               | 23 blocks   | 2x       |
        # | wide_resnet200_2| 128               | 63 blocks   | 2x       |
        # | wide_resnet50_4 | 256               | 6 blocks    | 4x       |
        # | wide_resnet101_4| 256               | 23 blocks   | 4x       |
        # ============================================================
        if 'conv1.weight' in keys and 'layer4' in keys_str and 'fc.weight' in keys:
            if 'layer1.0.conv1.weight' in keys:
                conv_shape = state_dict['layer1.0.conv1.weight'].shape
                layer1_ch = conv_shape[0]
                is_wide = layer1_ch >= 128
                
                if is_wide:
                    # 计算 layer3 的最大 block 索引来确定深度
                    max_layer3_idx = 0
                    for k in keys:
                        if 'layer3.' in k:
                            parts = k.split('.')
                            for i, p in enumerate(parts):
                                if p == 'layer3' and i + 1 < len(parts):
                                    try:
                                        idx = int(parts[i + 1])
                                        max_layer3_idx = max(max_layer3_idx, idx)
                                    except ValueError:
                                        pass
                    
                    # 判断宽度倍数 (2x vs 4x)
                    is_4x_wide = layer1_ch >= 256
                    
                    logger.debug(f"WideResNet 推断: layer1_ch={layer1_ch}, "
                                f"max_layer3_idx={max_layer3_idx}, is_4x_wide={is_4x_wide}")
                    
                    # 根据宽度和深度确定变体
                    if is_4x_wide:
                        # 4倍宽度系列
                        if max_layer3_idx >= 22:  # 101层: layer3 有 23 个 blocks (0-22)
                            return 'wide_resnet101_4'
                        else:  # 50层: layer3 有 6 个 blocks (0-5)
                            return 'wide_resnet50_4'
                    else:
                        # 2倍宽度系列
                        if max_layer3_idx >= 50:  # 200层: layer3 非常深
                            return 'wide_resnet200_2'
                        elif max_layer3_idx >= 22:  # 101层
                            return 'wide_resnet101_2'
                        else:  # 50层
                            return 'wide_resnet50_2'
        
        # ============================================================
        # 3. MobileNetV3 和 EfficientNet 区分（关键改进）
        #
        # 两者都有 conv_stem.weight 和 classifier.weight
        # 区分方法：
        # - MobileNetV3: blocks.0.0.conv.weight 存在（第一个 block 是简单的 ConvBnAct）
        # - EfficientNet: blocks.0.0.conv_pw.weight 存在（第一个 block 是 InvertedResidual）
        # - 检查 conv_head 的输出通道数
        # ============================================================
        if 'conv_stem.weight' in keys and 'classifier.weight' in keys:
            # 检查第一个 block 的结构
            has_simple_first_block = 'blocks.0.0.conv.weight' in keys
            has_inverted_first_block = 'blocks.0.0.conv_pw.weight' in keys or 'blocks.0.0.conv_dw.weight' in keys
            
            # 检查 conv_head（MobileNetV3 特有）
            has_conv_head = 'conv_head.weight' in keys
            
            # 检查 bn2（MobileNetV3 最后的 bn 层命名）
            has_bn2 = 'bn2.weight' in keys
            
            # MobileNetV3 判断条件：
            # 1. 第一个 block 是简单卷积 (blocks.0.0.conv.weight)
            # 2. 有 conv_head 层
            # 3. 有 bn2 层
            is_mobilenetv3 = has_simple_first_block or (has_conv_head and has_bn2)
            
            if is_mobilenetv3:
                # ============================================================
                # MobileNetV3 变体识别 - 关键修复
                # 
                # 各变体 conv_head.weight 的形状 [output_ch, input_ch, 1, 1]:
                # | 变体              | 输出通道(shape[0]) | 输入通道(shape[1]) |
                # |-------------------|-------------------|-------------------|
                # | large_100         | 1280              | 960               |
                # | large_075         | 960               | 720               |
                # | small_100         | 1024              | 576               |
                # | small_050         | 1024              | 288               |
                #
                # 注意: small_100 和 small_050 的输出通道相同(1024)，
                # 必须通过输入通道(shape[1])来区分！
                # ============================================================
                if 'conv_head.weight' in keys:
                    head_weight = state_dict['conv_head.weight']
                    head_out_ch = head_weight.shape[0]  # 输出通道
                    head_in_ch = head_weight.shape[1]   # 输入通道
                    
                    logger.debug(f"MobileNetV3 推断: conv_head shape={head_weight.shape}, "
                                f"out_ch={head_out_ch}, in_ch={head_in_ch}")
                    
                    # 1. 首先通过输出通道区分 large 和 small 系列
                    if head_out_ch >= 1280:
                        # Large-100: 输出 1280
                        return 'mobilenetv3_large_100'
                    elif head_out_ch >= 960 and head_out_ch < 1024:
                        # Large-075: 输出 960
                        return 'mobilenetv3_large_075'
                    elif head_out_ch >= 1024:
                        # Small 系列: 输出都是 1024，需要通过输入通道区分
                        # small_100: 输入 576
                        # small_050: 输入 288
                        if head_in_ch >= 400:  # 576 附近
                            return 'mobilenetv3_small_100'
                        else:  # 288 附近
                            return 'mobilenetv3_small_050'
                    else:
                        # 未知的输出通道，尝试通过输入通道推断
                        if head_in_ch >= 800:
                            return 'mobilenetv3_large_100'
                        elif head_in_ch >= 600:
                            return 'mobilenetv3_large_075'
                        elif head_in_ch >= 400:
                            return 'mobilenetv3_small_100'
                        else:
                            return 'mobilenetv3_small_050'
                
                # 备用：通过 blocks.5.0 (最后一个 stage) 的通道数判断
                # MobileNetV3 的 blocks.5.0.conv.weight 或 blocks.5.0.bn1.weight
                for key in ['blocks.5.0.conv.weight', 'blocks.5.0.bn1.weight']:
                    if key in keys:
                        ch = state_dict[key].shape[0]
                        logger.debug(f"MobileNetV3 备用推断: {key} shape[0]={ch}")
                        if ch >= 900:
                            return 'mobilenetv3_large_100'
                        elif ch >= 700:
                            return 'mobilenetv3_large_075'
                        elif ch >= 500:
                            return 'mobilenetv3_small_100'
                        else:
                            return 'mobilenetv3_small_050'
                
                # 备用：通过 blocks.0.0.conv 的通道数判断
                if 'blocks.0.0.conv.weight' in keys:
                    ch = state_dict['blocks.0.0.conv.weight'].shape[0]
                    if ch >= 16:
                        return 'mobilenetv3_large_100'
                    else:
                        return 'mobilenetv3_small_100'
                
                # 默认返回 large_100
                return 'mobilenetv3_large_100'
            
            # EfficientNet 判断
            else:
                # ============================================================
                # EfficientNet 版本推断 - 使用多维度特征综合判断
                # 
                # 各版本特征对比:
                # | 版本 | stem_ch | classifier_in | blocks 深度 |
                # |------|---------|---------------|-------------|
                # | B0   | 32      | 1280          | 6 (0-6)     |
                # | B1   | 32      | 1280          | 7 (0-7)     |
                # | B2   | 32      | 1408          | 7 (0-7)     |
                # | B3   | 40      | 1536          | 7 (0-7)     |
                # | B4   | 48      | 1792          | 7 (0-7)     |
                #
                # 主要判断依据: classifier_in (最可靠)
                # 辅助判断依据: stem_ch + blocks 深度 (区分 B0/B1)
                # ============================================================
                
                stem_ch = state_dict['conv_stem.weight'].shape[0]
                
                # 计算 blocks 最大深度
                max_block_idx = 0
                for k in keys:
                    if k.startswith('blocks.'):
                        parts = k.split('.')
                        if len(parts) >= 2:
                            try:
                                idx = int(parts[1])
                                max_block_idx = max(max_block_idx, idx)
                            except ValueError:
                                pass
                
                # 使用 classifier_in 作为主要判断依据
                if 'classifier.weight' in keys:
                    classifier_in = state_dict['classifier.weight'].shape[1]
                    
                    # 按 classifier_in 精确匹配
                    if classifier_in >= 1792:
                        result = 'efficientnet_b4'
                    elif classifier_in >= 1536:
                        result = 'efficientnet_b3'
                    elif classifier_in >= 1408:
                        result = 'efficientnet_b2'
                    elif classifier_in == 1280:
                        # B0 和 B1 的 classifier_in 都是 1280
                        # 需要使用 blocks 深度区分
                        # B0: max_block_idx = 6
                        # B1: max_block_idx = 7
                        if max_block_idx >= 7:
                            result = 'efficientnet_b1'
                        else:
                            result = 'efficientnet_b0'
                    else:
                        # 未知的 classifier_in，使用 stem_ch 辅助判断
                        if stem_ch >= 48:
                            result = 'efficientnet_b4'
                        elif stem_ch >= 40:
                            result = 'efficientnet_b3'
                        elif max_block_idx >= 7:
                            result = 'efficientnet_b1'
                        else:
                            result = 'efficientnet_b0'
                    
                    logger.debug(
                        f"EfficientNet 推断: classifier_in={classifier_in}, "
                        f"stem_ch={stem_ch}, max_block_idx={max_block_idx} => {result}"
                    )
                    return result
                
                # 没有 classifier.weight，使用 stem_ch + blocks 深度
                # 注意: B2 的 stem_ch=32 但 classifier_in=1408
                # 在没有 classifier 的情况下，B2 无法与 B0/B1 精确区分
                # 只能做近似判断
                if stem_ch >= 48:
                    return 'efficientnet_b4'
                elif stem_ch >= 40:
                    return 'efficientnet_b3'
                elif stem_ch >= 32:
                    # stem_ch=32 可能是 B0, B1, 或 B2
                    # 使用 blocks 深度来区分
                    if max_block_idx >= 7:
                        # 可能是 B1 或 B2，默认返回 B1
                        # 因为 B2 的 stem_ch=32 和 B1 相同，无法精确区分
                        logger.warning("⚠️ 无法精确区分 EfficientNet B1/B2，默认使用 B1")
                        return 'efficientnet_b1'
                    else:
                        return 'efficientnet_b0'
                
                return 'efficientnet_b0'
        
        # ============================================================
        # 4. 其他 MobileNetV3 变体（没有标准 conv_stem，备用路径）
        # ============================================================
        # 这个路径处理一些非标准格式的checkpoint
        # MobileNetV3 blocks.0.0.conv.weight 的输出通道:
        # | 变体            | blocks.0.0.conv.weight 输出通道 |
        # |-----------------|-------------------------------|
        # | small_050       | 8                             |
        # | small_100       | 16                            |
        # | large_075       | 16                            |
        # | large_100       | 16                            |
        # 
        # 注意: small_100 和 large 系列的 blocks.0.0.conv 通道数相同
        # 需要结合其他特征来区分
        # ============================================================
        if 'blocks.0.0' in keys_str or 'blocks.0.0.conv.weight' in keys:
            if 'blocks.0.0.conv.weight' in keys:
                ch = state_dict['blocks.0.0.conv.weight'].shape[0]
                logger.debug(f"MobileNetV3 备用路径: blocks.0.0.conv.weight 输出通道={ch}")
                
                # 检查是否有更深层的特征来区分
                # blocks.5.0 是 MobileNetV3 的最后一个 stage
                if 'blocks.5.0.conv.weight' in keys:
                    last_stage_ch = state_dict['blocks.5.0.conv.weight'].shape[0]
                    logger.debug(f"MobileNetV3 备用路径: blocks.5.0.conv.weight 输出通道={last_stage_ch}")
                    
                    # 根据最后一个 stage 的通道数区分
                    if last_stage_ch >= 900:
                        return 'mobilenetv3_large_100'
                    elif last_stage_ch >= 700:
                        return 'mobilenetv3_large_075'
                    elif last_stage_ch >= 500:
                        return 'mobilenetv3_small_100'
                    else:
                        return 'mobilenetv3_small_050'
                
                # 备用: 检查 blocks.4 的通道数
                if 'blocks.4.0.conv_pw.weight' in keys:
                    b4_ch = state_dict['blocks.4.0.conv_pw.weight'].shape[0]
                    logger.debug(f"MobileNetV3 备用路径: blocks.4.0.conv_pw.weight 输出通道={b4_ch}")
                    
                    if b4_ch >= 240:
                        return 'mobilenetv3_large_100'
                    elif b4_ch >= 180:
                        return 'mobilenetv3_large_075'
                    elif b4_ch >= 120:
                        return 'mobilenetv3_small_100'
                    else:
                        return 'mobilenetv3_small_050'
                
                # 最后的备用: 只能通过 blocks.0.0.conv 判断 small vs large
                if ch <= 8:
                    return 'mobilenetv3_small_050'
                elif ch <= 16:
                    # 无法区分 small_100 和 large，默认 small_100
                    logger.warning("⚠️ 无法精确区分 MobileNetV3 变体，默认使用 small_100")
                    return 'mobilenetv3_small_100'
                else:
                    return 'mobilenetv3_large_100'
        
        # 默认返回
        logger.warning("⚠️ 无法推断模型架构，使用默认值 efficientnet_b0")
        return 'efficientnet_b0'
    
    def rebuild(self, checkpoint: Dict[str, Any],
                num_classes: Optional[int] = None,
                model_path: str = None,
                **kwargs) -> nn.Module:
        """
        重建分类模型
        
        支持的框架：
        1. timm（优先）
        2. torchvision（备选）
        
        智能匹配策略：
        1. 首先尝试推断的模型名称
        2. 如果失败，尝试同一模型族的所有变体
        3. 如果还是失败，尝试 torchvision 模型
        4. 提供详细的诊断信息
        """
        model_name = self.get_model_name(checkpoint, model_path=model_path)
        
        # 验证模型是否支持
        if not self.is_model_supported(model_name):
            supported_str = ', '.join(self.supported_architectures[:10]) + '...'
            raise UnsupportedModelError(
                f"模型 '{model_name}' 不在支持列表中",
                task_type='cls',
                supported=self.supported_architectures
            )
        
        if num_classes is None:
            num_classes = self.get_num_classes(checkpoint)
        
        model_info = self.SUPPORTED_MODELS.get(model_name, {})
        logger.info(f"🔨 重建 timm 模型: {model_name}")
        logger.info(f"   模型族: {model_info.get('family', 'Unknown')}")
        logger.info(f"   规模: {model_info.get('size', 'Unknown')}")
        logger.info(f"   参数量: {model_info.get('params', 'Unknown')}M")
        logger.info(f"   类别数: {num_classes}")
        
        state_dict = checkpoint.get('model', checkpoint)
        if isinstance(state_dict, nn.Module):
            state_dict = state_dict.state_dict()
        
        # ============================================================
        # Step 1: 尝试推断的模型名称
        # ============================================================
        try:
            model = self.timm.create_model(
                model_name,
                pretrained=False,
                num_classes=num_classes
            )
            model.load_state_dict(state_dict, strict=True)
            logger.info(f"✅ timm 模型重建成功: {model_name}")
            return model
        except Exception as first_error:
            logger.warning(f"⚠️ 首选模型 {model_name} 加载失败，尝试其他变体...")
        
        # ============================================================
        # Step 2: 尝试同一模型族的所有变体
        # ============================================================
        family = model_info.get('family', '')
        family_models = self.get_models_by_family(family) if family else []
        
        # 排除已尝试的模型
        family_models = [m for m in family_models if m != model_name]
        
        best_match = None
        best_match_rate = 0
        best_model_name = None
        
        for variant in family_models:
            try:
                variant_model = self.timm.create_model(
                    variant,
                    pretrained=False,
                    num_classes=num_classes
                )
                
                # 计算匹配率
                variant_keys = set(variant_model.state_dict().keys())
                checkpoint_keys = set(state_dict.keys())
                
                matched = len(variant_keys & checkpoint_keys)
                total = len(variant_keys | checkpoint_keys)
                match_rate = matched / total if total > 0 else 0
                
                logger.debug(f"   尝试 {variant}: 匹配率 {match_rate:.2%}")
                
                if match_rate > best_match_rate:
                    best_match_rate = match_rate
                    best_match = variant_model
                    best_model_name = variant
                
                # 如果完全匹配，直接使用
                if match_rate > 0.95:
                    try:
                        variant_model.load_state_dict(state_dict, strict=True)
                        logger.info(f"✅ 找到匹配的变体: {variant} (匹配率: {match_rate:.2%})")
                        return variant_model
                    except Exception:
                        pass
                        
            except Exception:
                continue
        
        # 如果有高匹配率的变体（>80%），尝试 strict=False 加载
        if best_match is not None and best_match_rate > 0.8:
            try:
                missing, unexpected = best_match.load_state_dict(state_dict, strict=False)
                if len(missing) < len(best_match.state_dict()) * 0.1:  # missing < 10%
                    logger.warning(f"⚠️ 使用 {best_model_name} (匹配率: {best_match_rate:.2%}, strict=False)")
                    logger.warning(f"   Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
                    return best_match
            except Exception:
                pass
        
        # ============================================================
        # Step 3: 尝试 torchvision 模型
        # ============================================================
        torchvision_model = self._try_torchvision_model(state_dict, num_classes, model_name)
        if torchvision_model is not None:
            return torchvision_model
        
        # ============================================================
        # Step 4: 最后尝试 - strict=False 加载原始模型
        # ============================================================
        try:
            model = self.timm.create_model(
                model_name,
                pretrained=False,
                num_classes=num_classes
            )
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            
            # 计算实际加载了多少参数
            loaded_params = sum(p.numel() for n, p in model.named_parameters() 
                               if n not in missing)
            total_params = sum(p.numel() for p in model.parameters())
            load_rate = loaded_params / total_params if total_params > 0 else 0
            
            if load_rate > 0.5:  # 至少加载了 50% 的参数
                logger.warning(f"⚠️ 使用 {model_name} (参数加载率: {load_rate:.2%}, strict=False)")
                logger.warning(f"   Missing keys: {len(missing)}")
                logger.warning(f"   Unexpected keys: {len(unexpected)}")
                logger.warning(f"   ⚠️ 警告: 模型可能不完整，输出精度可能受影响")
                return model
        except Exception:
            pass
        
        # ============================================================
        # 诊断信息
        # ============================================================
        self._print_diagnostic_info(state_dict, model_name, family_models)
        raise ModelRebuildError(model_name, str(first_error))
    
    def _try_torchvision_model(
        self, 
        state_dict: Dict[str, torch.Tensor], 
        num_classes: int,
        original_model_name: str
    ) -> Optional[nn.Module]:
        """
        尝试使用 torchvision 加载模型
        
        支持的 torchvision 模型：
        - mobilenet_v3_large
        - mobilenet_v3_small
        - efficientnet_b0 ~ b7
        - wide_resnet50_2, wide_resnet101_2
        - vit_b_16, vit_b_32, vit_l_16, vit_l_32
        """
        try:
            import torchvision.models as tv_models
        except ImportError:
            return None
        
        keys = set(state_dict.keys())
        
        # 检测是否是 torchvision 格式
        is_torchvision = (
            'features.0.0.weight' in keys or  # MobileNet/EfficientNet
            'classifier.0.weight' in keys or   # MobileNet
            'conv1.weight' in keys and 'fc.weight' in keys  # ResNet
        )
        
        if not is_torchvision:
            # 也可能是 timm 格式但结构不同，仍然尝试
            pass
        
        # 根据模型名称尝试对应的 torchvision 模型
        torchvision_mapping = {
            # MobileNetV3
            'mobilenetv3_large_100': ('mobilenet_v3_large', {}),
            'mobilenetv3_large_075': ('mobilenet_v3_large', {}),
            'mobilenetv3_small_100': ('mobilenet_v3_small', {}),
            'mobilenetv3_small_050': ('mobilenet_v3_small', {}),
            'tf_mobilenetv3_large_100': ('mobilenet_v3_large', {}),
            # EfficientNet
            'efficientnet_b0': ('efficientnet_b0', {}),
            'efficientnet_b1': ('efficientnet_b1', {}),
            'efficientnet_b2': ('efficientnet_b2', {}),
            'efficientnet_b3': ('efficientnet_b3', {}),
            'efficientnet_b4': ('efficientnet_b4', {}),
            # WideResNet
            'wide_resnet50_2': ('wide_resnet50_2', {}),
            'wide_resnet101_2': ('wide_resnet101_2', {}),
            # ViT
            'vit_base_patch16_224': ('vit_b_16', {}),
            'vit_large_patch16_224': ('vit_l_16', {}),
        }
        
        tv_model_name, tv_kwargs = torchvision_mapping.get(
            original_model_name, (None, {})
        )
        
        if tv_model_name is None:
            return None
        
        try:
            # 创建 torchvision 模型
            if hasattr(tv_models, tv_model_name):
                model_fn = getattr(tv_models, tv_model_name)
                model = model_fn(weights=None, num_classes=num_classes, **tv_kwargs)
                
                # 尝试加载
                try:
                    model.load_state_dict(state_dict, strict=True)
                    logger.info(f"✅ torchvision 模型加载成功: {tv_model_name}")
                    return model
                except Exception:
                    # 尝试 strict=False
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                    
                    model_keys = set(model.state_dict().keys())
                    match_rate = len(model_keys - set(missing)) / len(model_keys)
                    
                    if match_rate > 0.8:
                        logger.info(f"✅ torchvision 模型加载成功: {tv_model_name} (匹配率: {match_rate:.2%})")
                        return model
                        
        except Exception as e:
            logger.debug(f"torchvision {tv_model_name} 加载失败: {e}")
        
        return None
    
    def _print_diagnostic_info(
        self, 
        state_dict: Dict[str, torch.Tensor],
        model_name: str,
        tried_variants: List[str]
    ):
        """打印诊断信息，帮助用户解决问题"""
        logger.error("=" * 60)
        logger.error("模型重建失败 - 诊断信息")
        logger.error("=" * 60)
        
        # 分析 state_dict 结构
        keys = list(state_dict.keys())
        
        logger.error(f"目标模型: {model_name}")
        logger.error(f"尝试的变体: {', '.join(tried_variants[:5])}...")
        logger.error(f"Checkpoint keys 数量: {len(keys)}")
        logger.error(f"前 10 个 keys: {keys[:10]}")
        
        # 检测可能的来源
        if 'features.0.0.weight' in keys:
            logger.error("🔍 检测到 torchvision 格式的 MobileNet/EfficientNet")
            logger.error("   建议: 确保使用 torchvision 保存的模型")
        elif 'conv_stem.weight' in keys:
            logger.error("🔍 检测到 timm 格式")
            
            # 检查 SE 模块分布
            se_keys = [k for k in keys if '.se.' in k]
            if se_keys:
                logger.error(f"   SE 模块: {len(se_keys)} 个")
                logger.error(f"   示例: {se_keys[:3]}")
        
        logger.error("")
        logger.error("解决方案:")
        logger.error("1. 在保存模型时，在 checkpoint 中记录 model_name")
        logger.error("   示例: torch.save({'model': model.state_dict(), 'model_name': 'mobilenetv3_large_100'}, 'model.pth')")
        logger.error("2. 使用与训练时相同的框架（timm 或 torchvision）")
        logger.error("3. 如果是自定义结构，考虑直接保存整个模型对象")
        logger.error("=" * 60)
    
    def get_num_classes(self, checkpoint: Dict[str, Any]) -> int:
        """获取类别数"""
        if 'num_classes' in checkpoint:
            return checkpoint['num_classes']
        
        state_dict = checkpoint.get('model', checkpoint)
        if isinstance(state_dict, dict):
            for key in ['classifier.weight', 'fc.weight', 'head.weight']:
                if key in state_dict:
                    return state_dict[key].shape[0]
        
        raise NumClassesInferenceError(
            "无法推断 timm 模型类别数",
            tried_methods=['checkpoint.num_classes', 'classifier.weight', 'fc.weight', 'head.weight']
        )
    
    def get_input_spec(self, checkpoint: Dict[str, Any]) -> InputSpec:
        """获取输入规格"""
        model_name = self.get_model_name(checkpoint)
        model_info = self.SUPPORTED_MODELS.get(model_name, {})
        input_size = model_info.get('input_size', 224)
        
        # 也检查 checkpoint 中是否有指定
        if 'input_size' in checkpoint:
            input_size = checkpoint['input_size']
        
        return InputSpec(
            shape=(1, 3, input_size, input_size),
            normalize_mean=(0.485, 0.456, 0.406),
            normalize_std=(0.229, 0.224, 0.225)
        )
    
    def get_output_spec(self, checkpoint: Dict[str, Any], 
                        num_classes: int) -> OutputSpec:
        """获取输出规格"""
        return OutputSpec(
            structure=f"Tensor[B, {num_classes}]",
            num_classes=num_classes,
            output_type="tensor"
        )


# ==============================================================================
# 第4部分: 检测模型处理器 (DetectionHandlers) - 仅 YOLO
# ==============================================================================

class YOLOLoader:
    """YOLO 模型加载器 - 处理 YOLOv5 特殊的 pickle 格式"""
    
    _yolov5_path: Optional[str] = os.environ.get("YOLOV5_PATH", None)
    
    @classmethod
    def set_yolov5_path(cls, path: str):
        """设置 YOLOv5 源码路径"""
        cls._yolov5_path = path
        logger.info(f"📁 YOLOv5 路径设置为: {path}")
    
    @classmethod
    def get_yolov5_path(cls) -> Optional[str]:
        """获取 YOLOv5 路径"""
        if cls._yolov5_path:
            return cls._yolov5_path
        
        env_path = os.environ.get('YOLOV5_PATH')
        if env_path and os.path.exists(env_path):
            return env_path
        
        common_paths = [
            './yolov5',
            '../yolov5',
            os.path.expanduser('~/yolov5'),
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    @classmethod
    def load(cls, model_path: str) -> Dict[str, Any]:
        """加载 YOLO 模型"""
        yolov5_path = cls.get_yolov5_path()
        
        if yolov5_path and yolov5_path not in sys.path:
            sys.path.insert(0, yolov5_path)
            logger.info(f"📁 添加 YOLOv5 到 sys.path: {yolov5_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                model = checkpoint['model']
                if hasattr(model, 'state_dict'):
                    return {
                        'model': model.state_dict() if callable(model.state_dict) else model.state_dict,
                        'framework': 'ultralytics',
                        'task': 'det',
                        '_original_model': model,
                        'yaml': getattr(model, 'yaml', None),
                        'names': getattr(model, 'names', None),
                        'nc': getattr(model, 'nc', None),
                    }
            
            return checkpoint
            
        except ModuleNotFoundError as e:
            if 'models' in str(e):
                raise FrameworkNotInstalledError(
                    "YOLOv5",
                    f"设置 YOLOv5 路径: YOLOLoader.set_yolov5_path('/path/to/yolov5')"
                )
            raise


class YOLODetectionHandler(ModelHandler):
    """
    YOLO 检测模型处理器
    ===================
    支持 YOLOv5, YOLOv8, YOLOv11, YOLO26
    
    这是目标检测任务唯一支持的处理器。
    注意: YOLO26 需要 ultralytics >= 8.4.0
    """
    
    def __init__(self):
        self._ultralytics = None
    
    @property
    def ultralytics(self):
        """延迟加载 ultralytics"""
        if self._ultralytics is None:
            try:
                import ultralytics
                self._ultralytics = ultralytics
            except ImportError:
                raise FrameworkNotInstalledError(
                    "ultralytics",
                    "pip install ultralytics"
                )
        return self._ultralytics
    
    @property
    def task_type(self) -> TaskType:
        return TaskType.DETECTION
    
    @property
    def framework(self) -> Framework:
        return Framework.ULTRALYTICS
    
    @property
    def supported_architectures(self) -> List[str]:
        return [
            # YOLOv5
            'yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x',
            # YOLOv8
            'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
            # YOLOv11
            'yolov11n', 'yolov11s', 'yolov11m', 'yolov11l', 'yolov11x',
            # YOLO26 (NMS-free, no DFL)
            'yolo26n', 'yolo26s', 'yolo26m', 'yolo26l', 'yolo26x',
        ]
    
    def can_handle(self, checkpoint: Dict[str, Any]) -> bool:
        """检查是否是 YOLO 模型"""
        if checkpoint.get('framework') == 'ultralytics':
            return True
        
        if '_original_model' in checkpoint:
            model = checkpoint['_original_model']
            class_name = type(model).__name__.lower()
            if 'yolo' in class_name or 'detection' in class_name:
                return True
        
        state_dict = checkpoint.get('model', checkpoint)
        if isinstance(state_dict, dict):
            keys_str = ' '.join(list(state_dict.keys())[:50])
            if 'model.0.' in keys_str or 'backbone' in keys_str:
                return True
        
        return False
    
    def get_model_name(self, checkpoint: Dict[str, Any], model_path: str = None) -> str:
        """获取 YOLO 版本
        
        Args:
            checkpoint: 模型检查点
            model_path: 模型文件路径（可选，用于从文件名推断版本）
        
        优先级:
        1. checkpoint 中保存的 model_name (训练时保存，最可靠)
        2. 从文件名推断
        3. 从 yaml 字段推断
        4. 从原始模型类名推断
        """
        # ============================================================
        # 1. 优先使用 checkpoint 中保存的 model_name
        # ============================================================
        if 'model_name' in checkpoint and checkpoint['model_name']:
            model_name = checkpoint['model_name'].lower()
            logger.info(f"✅ 使用 checkpoint 中保存的 YOLO 模型名称: {checkpoint['model_name']}")
            # 标准化版本名称
            if 'yolo26' in model_name or 'yolov26' in model_name:
                return 'yolo26'
            if 'yolo11' in model_name or 'yolov11' in model_name:
                return 'yolov11'
            elif 'yolov8' in model_name:
                return 'yolov8'
            elif 'yolov5' in model_name:
                return 'yolov5'
            # 返回完整的模型名称（如 yolov8n, yolov8s 等）
            return checkpoint['model_name']
        
        # ============================================================
        # 2. 从文件名推断
        # ============================================================
        if model_path:
            filename = os.path.basename(model_path).lower()
            for v in ['yolo26', 'yolov5', 'yolov8', 'yolov11', 'yolo11']:
                if v in filename:
                    logger.info(f"📝 从文件名推断 YOLO 版本: {v}")
                    return v.replace('yolo11', 'yolov11')
        
        # ============================================================
        # 3. 从 checkpoint 的 yaml 字段推断
        # ============================================================
        if 'yaml' in checkpoint and checkpoint['yaml']:
            yaml_str = str(checkpoint['yaml']).lower()
            for v in ['yolo26', 'yolov5', 'yolov8', 'yolov11', 'yolo11']:
                if v in yaml_str:
                    logger.info(f"📝 从 yaml 字段推断 YOLO 版本: {v}")
                    return v.replace('yolo11', 'yolov11')
        
        # ============================================================
        # 4. 从原始模型类名推断
        # ============================================================
        if '_original_model' in checkpoint:
            model = checkpoint['_original_model']
            class_name = type(model).__name__.lower()
            if 'yolo26' in class_name:
                return 'yolo26'
            if 'yolov5' in class_name:
                return 'yolov5'
            if 'yolov8' in class_name or 'yolo' in class_name:
                return 'yolov8'
            if 'yolov11' in class_name:
                return 'yolov11'
        
        logger.warning("⚠️ 无法识别 YOLO 版本，使用默认 'yolo'")
        return 'yolo'
    
    def rebuild(self, checkpoint: Dict[str, Any],
                num_classes: Optional[int] = None,
                **kwargs) -> nn.Module:
        """重建 YOLO 模型"""
        model_name = self.get_model_name(checkpoint)
        
        if '_original_model' in checkpoint:
            model = checkpoint['_original_model']
            if hasattr(model, 'float'):
                model = model.float()
            model.eval()
            logger.info(f"✅ YOLO 模型重建成功 (使用原始模型)")
            return model
        
        logger.warning(f"⚠️ YOLO 需要原始模型对象进行重建")
        
        try:
            from ultralytics import YOLO
            
            model_path = kwargs.get('model_path', 'yolov8n.pt')
            yolo = YOLO(model_path)
            model = yolo.model
            
            state_dict = checkpoint.get('model', checkpoint)
            if isinstance(state_dict, dict):
                model.load_state_dict(state_dict, strict=False)
            
            model.eval()
            logger.info(f"✅ YOLO 模型重建成功")
            return model
            
        except Exception as e:
            raise ModelRebuildError(model_name, str(e))
    
    def get_num_classes(self, checkpoint: Dict[str, Any]) -> int:
        """获取类别数"""
        tried_methods = []
        
        # 方法1: 直接从 checkpoint 获取
        if 'nc' in checkpoint and checkpoint['nc'] is not None:
            return checkpoint['nc']
        tried_methods.append('checkpoint.nc')
        
        # 方法2: 从 names 推断
        if 'names' in checkpoint and checkpoint['names']:
            names = checkpoint['names']
            if isinstance(names, dict):
                return len(names)
            if isinstance(names, (list, tuple)):
                return len(names)
        tried_methods.append('len(checkpoint.names)')
        
        # 方法3: 从原始模型获取
        if '_original_model' in checkpoint:
            model = checkpoint['_original_model']
            if hasattr(model, 'nc'):
                return model.nc
            if hasattr(model, 'names'):
                return len(model.names)
        tried_methods.append('model.nc / len(model.names)')
        
        # 方法4: 从 yaml 配置获取
        if 'yaml' in checkpoint and checkpoint['yaml']:
            yaml_cfg = checkpoint['yaml']
            if isinstance(yaml_cfg, dict) and 'nc' in yaml_cfg:
                return yaml_cfg['nc']
        tried_methods.append('yaml.nc')
        
        raise NumClassesInferenceError(
            "无法推断 YOLO 类别数",
            tried_methods=tried_methods
        )
    
    def get_input_spec(self, checkpoint: Dict[str, Any]) -> InputSpec:
        """获取输入规格"""
        return InputSpec(
            shape=(1, 3, 640, 640),
            channel_order="RGB",
            normalize_mean=(0.0, 0.0, 0.0),
            normalize_std=(1.0, 1.0, 1.0),
            value_range=(0.0, 1.0)
        )
    
    def get_output_spec(self, checkpoint: Dict[str, Any], 
                        num_classes: int) -> OutputSpec:
        """获取输出规格"""
        return OutputSpec(
            structure=f"List[Tensor] - detections",
            num_classes=num_classes,
            output_type="list",
            extra_info={'format': 'xyxy, conf, cls'}
        )


# ==============================================================================
# 第5部分: 分割模型处理器 (SegmentationHandlers) - SegFormer (MMSegmentation)
# ==============================================================================

class SegFormerSegmentationHandler(ModelHandler):
    """
    SegFormer 语义分割模型处理器
    ==============================
    
    专为内部训练软件导出的 SegFormer 模型设计。
    基于 MMSegmentation 框架，使用 mmseg.apis.init_model 重建模型。
    
    支持的 SegFormer 变体:
    - segformer_b0: MiT-B0 backbone (3.7M params)
    - segformer_b1: MiT-B1 backbone (13.7M params)
    - segformer_b2: MiT-B2 backbone (24.7M params)
    - segformer_b3: MiT-B3 backbone (44.6M params)
    - segformer_b4: MiT-B4 backbone (61.4M params)
    - segformer_b5: MiT-B5 backbone (81.9M params)
    
    Checkpoint 格式要求 (由训练软件生成):
    {
        'framework': 'mmsegmentation',
        'model_type': 'segformer',
        'model_name': 'segformer_b2',
        'state_dict': {...},
        'meta': {
            'dataset_meta': {'classes': (...), 'palette': [...]},
            'CLASSES': (...),
            'PALETTE': [...],
        },
        'num_classes': 10,
        'class_names': ['class0', 'class1', ...],
        'model_metadata': {
            'input_size': 512,
            'input_spec': {
                'normalize_mean': [123.675, 116.28, 103.53],
                'normalize_std': [58.395, 57.12, 57.375],
            },
        },
    }
    
    预处理说明:
    - 输入格式: RGB, 0-255 范围
    - 归一化: 像素级 ImageNet 归一化
      normalized = (pixel - mean) / std
      mean = [123.675, 116.28, 103.53]
      std = [58.395, 57.12, 57.375]
    - 输出格式: (B, num_classes, H, W) 语义分割 logits
    """
    
    # =========================================================================
    # ONNX 导出包装器
    # =========================================================================
    # MMSeg 模型的 forward 方法接受复杂输入格式，不适合直接导出 ONNX
    # 使用此包装器创建一个简化的前向传播路径
    # =========================================================================
    
    class SegFormerONNXWrapper(nn.Module):
        """
        SegFormer ONNX 导出包装器
        
        将 MMSeg 的 EncoderDecoder 模型包装为 ONNX 友好格式:
        - 输入: Tensor[B, 3, H, W] (已归一化)
        - 输出: Tensor[B, num_classes, H, W] (logits)
        
        归一化在包装器内部完成，导出后的 ONNX 模型接受 0-255 范围的 RGB 图像。
        """
        
        def __init__(
            self, 
            backbone: nn.Module, 
            decode_head: nn.Module,
            mean: Tuple[float, ...] = (123.675, 116.28, 103.53),
            std: Tuple[float, ...] = (58.395, 57.12, 57.375),
            align_corners: bool = False,
        ):
            super().__init__()
            self.backbone = backbone
            self.decode_head = decode_head
            
            # 注册归一化参数为 buffer (会被保存到 ONNX)
            self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
            self.align_corners = align_corners
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            前向传播
            
            Args:
                x: 输入图像 Tensor[B, 3, H, W]，值范围 0-255，RGB 格式
                
            Returns:
                分割 logits Tensor[B, num_classes, H, W]
            """
            # 归一化: (x - mean) / std
            x = (x - self.mean) / self.std
            
            # 提取多尺度特征
            features = self.backbone(x)
            
            # 解码得到分割 logits
            output = self.decode_head(features)
            
            # 上采样到输入尺寸 (如果需要)
            if output.shape[2:] != x.shape[2:]:
                output = nn.functional.interpolate(
                    output, 
                    size=x.shape[2:], 
                    mode='bilinear', 
                    align_corners=self.align_corners
                )
            
            return output
    
    # =========================================================================
    # SegFormer 变体配置
    # =========================================================================
    # SegFormer 变体配置
    SEGFORMER_VARIANTS = {
        'segformer_b0': {
            'embed_dims': [32, 64, 160, 256],
            'depths': [2, 2, 2, 2],
            'num_heads': [1, 2, 5, 8],
            'params': 3.7,
        },
        'segformer_b1': {
            'embed_dims': [64, 128, 320, 512],
            'depths': [2, 2, 2, 2],
            'num_heads': [1, 2, 5, 8],
            'params': 13.7,
        },
        'segformer_b2': {
            'embed_dims': [64, 128, 320, 512],
            'depths': [3, 4, 6, 3],
            'num_heads': [1, 2, 5, 8],
            'params': 24.7,
        },
        'segformer_b3': {
            'embed_dims': [64, 128, 320, 512],
            'depths': [3, 4, 18, 3],
            'num_heads': [1, 2, 5, 8],
            'params': 44.6,
        },
        'segformer_b4': {
            'embed_dims': [64, 128, 320, 512],
            'depths': [3, 8, 27, 3],
            'num_heads': [1, 2, 5, 8],
            'params': 61.4,
        },
        'segformer_b5': {
            'embed_dims': [64, 128, 320, 512],
            'depths': [3, 6, 40, 3],
            'num_heads': [1, 2, 5, 8],
            'params': 81.9,
        },
    }
    
    # SegFormer 像素级 ImageNet 归一化 (0-255 范围)
    SEGFORMER_MEAN = (123.675, 116.28, 103.53)
    SEGFORMER_STD = (58.395, 57.12, 57.375)
    
    def __init__(self):
        self._mmseg = None
        self._mmengine = None
    
    @property
    def mmseg(self):
        """延迟加载 mmsegmentation"""
        if self._mmseg is None:
            try:
                import mmseg
                self._mmseg = mmseg
            except ImportError:
                raise FrameworkNotInstalledError(
                    "mmsegmentation",
                    "pip install mmsegmentation mmengine mmcv"
                )
        return self._mmseg
    
    @property
    def mmengine(self):
        """延迟加载 mmengine"""
        if self._mmengine is None:
            try:
                import mmengine
                self._mmengine = mmengine
            except ImportError:
                raise FrameworkNotInstalledError(
                    "mmengine",
                    "pip install mmengine"
                )
        return self._mmengine
    
    @property
    def task_type(self) -> TaskType:
        return TaskType.SEGMENTATION
    
    @property
    def framework(self) -> Framework:
        return Framework.MMSEGMENTATION
    
    @property
    def supported_architectures(self) -> List[str]:
        return list(self.SEGFORMER_VARIANTS.keys())
    
    def can_handle(self, checkpoint: Dict[str, Any]) -> bool:
        """
        检查是否是 SegFormer 分割模型
        
        识别条件 (必须同时满足):
        1. framework == 'mmsegmentation'
        2. model_type == 'segformer'
        """
        # 主要识别方式: 显式标记
        framework = checkpoint.get('framework', '').lower()
        model_type = checkpoint.get('model_type', '').lower()
        
        if framework == 'mmsegmentation' and model_type == 'segformer':
            return True
        
        # 备用识别: 检查 model_name 是否包含 segformer
        model_name = checkpoint.get('model_name', '').lower()
        if 'segformer' in model_name and framework == 'mmsegmentation':
            return True
        
        # 备用识别: 检查 state_dict 结构 (SegFormer 特有的 key)
        state_dict = checkpoint.get('state_dict', checkpoint.get('model', {}))
        if isinstance(state_dict, dict):
            keys = set(state_dict.keys())
            # SegFormer 的 backbone 使用 MixVisionTransformer
            # 检查特征性 key
            has_patch_embed = any('patch_embed' in k for k in keys)
            has_decode_head = any('decode_head' in k for k in keys)
            has_linear_fuse = any('linear_fuse' in k for k in keys)
            
            if has_patch_embed and has_decode_head and has_linear_fuse:
                return True
        
        return False
    
    def get_model_name(self, checkpoint: Dict[str, Any], model_path: str = None) -> str:
        """
        获取模型名称
        
        优先级:
        1. checkpoint['model_name'] (训练时保存，最可靠)
        2. checkpoint['model_metadata']['model_name']
        3. 从 state_dict 推断变体
        4. 从文件名推断
        5. 默认 'segformer_b2'
        """
        logger.debug("=" * 60)
        logger.debug("🔍 SegFormer 模型名称识别:")
        
        # 方法1: 直接从 checkpoint 获取
        if 'model_name' in checkpoint:
            name = checkpoint['model_name']
            logger.debug(f"   📌 checkpoint['model_name'] = '{name}'")
            if name and isinstance(name, str):
                logger.info(f"✅ 使用 checkpoint 中保存的模型名称: {name}")
                return name.lower()
        
        # 方法2: 从 model_metadata 获取
        metadata = checkpoint.get('model_metadata', {})
        if 'model_name' in metadata:
            name = metadata['model_name']
            logger.debug(f"   📌 checkpoint['model_metadata']['model_name'] = '{name}'")
            if name and isinstance(name, str):
                logger.info(f"✅ 使用 model_metadata 中保存的模型名称: {name}")
                return name.lower()
        
        # 方法3: 从 state_dict 推断变体
        state_dict = checkpoint.get('state_dict', checkpoint.get('model', {}))
        if isinstance(state_dict, dict):
            variant = self._infer_variant_from_state_dict(state_dict)
            if variant:
                logger.info(f"🔬 从 state_dict 结构推断 SegFormer 变体: {variant}")
                logger.warning("⚠️ 建议: 请确保模型保存时包含 model_name 字段")
                return variant
        
        # 方法4: 从文件名推断
        if model_path:
            filename = os.path.basename(model_path).lower()
            for variant in self.SEGFORMER_VARIANTS.keys():
                # 匹配 segformer_b0, segformer-b0, segformerb0 等
                variant_patterns = [
                    variant,  # segformer_b0
                    variant.replace('_', '-'),  # segformer-b0
                    variant.replace('_', ''),  # segformerb0
                    variant.split('_')[1] if '_' in variant else '',  # b0
                ]
                for pattern in variant_patterns:
                    if pattern and pattern in filename:
                        logger.info(f"📝 从文件名推断 SegFormer 变体: {variant}")
                        return variant
        
        # 默认返回 b2 (最常用的变体)
        logger.warning("⚠️ 无法推断 SegFormer 变体，默认使用 segformer_b2")
        logger.warning("⚠️ 建议: 请确保训练时正确保存 model_name 字段")
        return 'segformer_b2'
    
    def _infer_variant_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Optional[str]:
        """
        从 state_dict 推断 SegFormer 变体
        
        通过分析 backbone 的 embed_dims 和 depths 来确定变体:
        - B0: embed_dims[0]=32, depths=[2,2,2,2]
        - B1-B5: embed_dims[0]=64, 通过 depths 区分
        """
        keys = list(state_dict.keys())
        
        # 检查 backbone.patch_embed1.proj.weight 的形状来确定 embed_dims[0]
        embed_key = None
        for k in keys:
            if 'backbone.patch_embed1.proj.weight' in k or 'patch_embed1.proj.weight' in k:
                embed_key = k
                break
        
        if embed_key is None:
            return None
        
        try:
            # embed_dims[0] 是 patch_embed1 的输出通道数
            embed_dim_0 = state_dict[embed_key].shape[0]
            
            # B0 的 embed_dims[0] = 32
            if embed_dim_0 == 32:
                return 'segformer_b0'
            
            # B1-B5 的 embed_dims[0] = 64，需要通过 depths 区分
            if embed_dim_0 == 64:
                # 统计 backbone.block3 的层数来确定 depths[2]
                block3_count = 0
                for k in keys:
                    if 'backbone.block3.' in k or 'block3.' in k:
                        # 提取层索引
                        parts = k.split('.')
                        for i, p in enumerate(parts):
                            if p == 'block3' and i + 1 < len(parts):
                                try:
                                    layer_idx = int(parts[i + 1])
                                    block3_count = max(block3_count, layer_idx + 1)
                                except ValueError:
                                    pass
                
                # 根据 block3 的层数推断变体
                # B1: depths[2]=2, B2: depths[2]=6, B3: depths[2]=18
                # B4: depths[2]=27, B5: depths[2]=40
                if block3_count <= 2:
                    return 'segformer_b1'
                elif block3_count <= 6:
                    return 'segformer_b2'
                elif block3_count <= 18:
                    return 'segformer_b3'
                elif block3_count <= 27:
                    return 'segformer_b4'
                else:
                    return 'segformer_b5'
        except Exception as e:
            logger.debug(f"推断 SegFormer 变体失败: {e}")
        
        return None
    
    def rebuild(self, checkpoint: Dict[str, Any],
                num_classes: Optional[int] = None,
                **kwargs) -> nn.Module:
        """
        重建 SegFormer 模型
        
        策略: 分别构建 backbone 和 decode_head，绕过 build_segmentor
        这样可以避免 SegDataPreProcessor 注册问题，并且更适合 ONNX 导出
        
        步骤:
        1. 从 checkpoint 获取 num_classes 和 model_metadata
        2. 分别构建 backbone (MixVisionTransformer) 和 decode_head (SegformerHead)
        3. 加载 state_dict
        4. 包装为 ONNX 友好格式
        """
        # =====================================================================
        # 导入必要的组件
        # =====================================================================
        try:
            # 导入 backbone 和 decode_head 的构建器
            from mmseg.models.backbones import MixVisionTransformer
            from mmseg.models.decode_heads import SegformerHead
        except ImportError as e:
            raise FrameworkNotInstalledError(
                "mmsegmentation",
                f"pip install mmsegmentation mmengine mmcv\n导入错误: {e}"
            )
        
        # 获取模型名称
        model_name = self.get_model_name(checkpoint, kwargs.get('model_path'))
        logger.info(f"🔧 重建 SegFormer 模型: {model_name}")
        
        # 获取类别数
        if num_classes is None:
            num_classes = self.get_num_classes(checkpoint)
        
        # 获取输入尺寸
        input_size = self._get_input_size(checkpoint)
        
        # 获取变体配置
        variant_config = self.SEGFORMER_VARIANTS.get(
            model_name, 
            self.SEGFORMER_VARIANTS['segformer_b2']
        )
        
        # 获取归一化参数
        norm_mean, norm_std = self._get_normalize_params(checkpoint)
        
        try:
            # =================================================================
            # 构建 Backbone: MixVisionTransformer
            # =================================================================
            embed_dims = variant_config['embed_dims']
            depths = variant_config['depths']
            num_heads = variant_config['num_heads']
            
            backbone = MixVisionTransformer(
                in_channels=3,
                embed_dims=embed_dims[0],  # 第一阶段的 embed_dim
                num_stages=4,
                num_layers=depths,
                num_heads=num_heads,
                patch_sizes=[7, 3, 3, 3],
                sr_ratios=[8, 4, 2, 1],
                out_indices=(0, 1, 2, 3),
                mlp_ratio=4,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.1,
            )
            
            # =================================================================
            # 构建 Decode Head: SegformerHead
            # =================================================================
            # SegformerHead 的 in_channels 是各阶段的输出通道数
            # 添加 loss_decode 参数以确保与 BaseDecodeHead 兼容
            decode_head = SegformerHead(
                in_channels=embed_dims,  # [32, 64, 160, 256] for B0 or [64, 128, 320, 512] for B1-B5
                in_index=[0, 1, 2, 3],
                channels=256,
                dropout_ratio=0.1,
                num_classes=num_classes,
                norm_cfg=dict(type='BN', requires_grad=True),  # 使用 BN 而非 SyncBN
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0,
                ),
            )
            
            # =================================================================
            # 加载 state_dict
            # =================================================================
            state_dict = checkpoint.get('state_dict', checkpoint.get('model', {}))
            if state_dict:
                # 分别提取 backbone 和 decode_head 的权重
                backbone_state = {}
                decode_head_state = {}
                
                for key, value in state_dict.items():
                    # 移除可能的前缀
                    clean_key = key
                    for prefix in ['module.', '_orig_mod.', 'model.']:
                        if clean_key.startswith(prefix):
                            clean_key = clean_key[len(prefix):]
                    
                    # 分配到对应模块
                    if clean_key.startswith('backbone.'):
                        new_key = clean_key[len('backbone.'):]
                        backbone_state[new_key] = value
                    elif clean_key.startswith('decode_head.'):
                        new_key = clean_key[len('decode_head.'):]
                        decode_head_state[new_key] = value
                
                # 加载 backbone 权重
                if backbone_state:
                    missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
                    if missing:
                        logger.debug(f"Backbone 缺失的 keys ({len(missing)}): {missing[:3]}...")
                    if unexpected:
                        logger.debug(f"Backbone 多余的 keys ({len(unexpected)}): {unexpected[:3]}...")
                    logger.info(f"   ✅ Backbone 权重加载完成")
                
                # 加载 decode_head 权重
                if decode_head_state:
                    missing, unexpected = decode_head.load_state_dict(decode_head_state, strict=False)
                    if missing:
                        logger.debug(f"DecodeHead 缺失的 keys ({len(missing)}): {missing[:3]}...")
                    if unexpected:
                        logger.debug(f"DecodeHead 多余的 keys ({len(unexpected)}): {unexpected[:3]}...")
                    logger.info(f"   ✅ Decode Head 权重加载完成")
            
            # 设置为评估模式
            backbone.eval()
            decode_head.eval()
            
            # 获取 decode_head 的 align_corners 设置
            align_corners = getattr(decode_head, 'align_corners', False)
            
            # =================================================================
            # 创建 ONNX 导出友好的包装器
            # =================================================================
            wrapped_model = self.SegFormerONNXWrapper(
                backbone=backbone,
                decode_head=decode_head,
                mean=norm_mean,
                std=norm_std,
                align_corners=align_corners,
            )
            wrapped_model.eval()
            
            # 存储元数据到包装模型
            wrapped_model._segformer_metadata = {
                'model_name': model_name,
                'num_classes': num_classes,
                'input_size': input_size,
                'normalize_mean': norm_mean,
                'normalize_std': norm_std,
                'class_names': checkpoint.get('class_names', []),
                'embed_dims': embed_dims,
                'depths': depths,
            }
            
            logger.info(f"✅ SegFormer 模型重建成功: {model_name}")
            logger.info(f"   类别数: {num_classes}, 输入尺寸: {input_size}x{input_size}")
            logger.info(f"   已包装为 ONNX 导出友好格式")
            return wrapped_model
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise ModelRebuildError(
                f"segformer_{model_name}",
                f"模型重建失败: {str(e)}\n"
                f"请确保已安装: pip install mmsegmentation mmengine mmcv"
            )
    
    def _get_input_size(self, checkpoint: Dict[str, Any]) -> int:
        """
        获取输入尺寸
        
        优先级:
        1. model_metadata.input_size
        2. model_metadata.input_spec.shape
        3. 默认 512
        """
        metadata = checkpoint.get('model_metadata', {})
        
        # 方法1: 直接获取 input_size
        if 'input_size' in metadata:
            size = metadata['input_size']
            if isinstance(size, (int, float)):
                return int(size)
            if isinstance(size, (list, tuple)) and len(size) >= 1:
                return int(size[0])
        
        # 方法2: 从 input_spec.shape 获取
        input_spec = metadata.get('input_spec', {})
        if 'shape' in input_spec:
            shape = input_spec['shape']
            if isinstance(shape, (list, tuple)) and len(shape) >= 4:
                # shape = (B, C, H, W)
                return int(shape[2])
        
        # 默认 512
        return 512
    
    def _get_normalize_params(
        self, 
        checkpoint: Dict[str, Any]
    ) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """
        获取归一化参数
        
        优先级:
        1. model_metadata.input_spec.normalize_mean/std
        2. 默认 SegFormer 像素级归一化
        """
        metadata = checkpoint.get('model_metadata', {})
        input_spec = metadata.get('input_spec', {})
        
        mean = input_spec.get('normalize_mean', list(self.SEGFORMER_MEAN))
        std = input_spec.get('normalize_std', list(self.SEGFORMER_STD))
        
        # 确保是 tuple
        if isinstance(mean, list):
            mean = tuple(mean)
        if isinstance(std, list):
            std = tuple(std)
        
        return mean, std
    
    def get_num_classes(self, checkpoint: Dict[str, Any]) -> int:
        """
        获取类别数
        
        优先级:
        1. checkpoint['num_classes']
        2. checkpoint['nc']
        3. checkpoint['model_metadata']['num_classes']
        4. len(checkpoint['class_names'])
        5. 从 meta.dataset_meta.classes 推断
        """
        tried_methods = []
        
        # 方法1: 直接获取 num_classes
        if 'num_classes' in checkpoint:
            nc = checkpoint['num_classes']
            if nc is not None and nc > 0:
                return int(nc)
        tried_methods.append('checkpoint.num_classes')
        
        # 方法2: 获取 nc
        if 'nc' in checkpoint:
            nc = checkpoint['nc']
            if nc is not None and nc > 0:
                return int(nc)
        tried_methods.append('checkpoint.nc')
        
        # 方法3: 从 model_metadata 获取
        metadata = checkpoint.get('model_metadata', {})
        if 'num_classes' in metadata:
            nc = metadata['num_classes']
            if nc is not None and nc > 0:
                return int(nc)
        tried_methods.append('checkpoint.model_metadata.num_classes')
        
        # 方法4: 从 class_names 推断
        if 'class_names' in checkpoint:
            names = checkpoint['class_names']
            if isinstance(names, (list, tuple)) and len(names) > 0:
                return len(names)
        tried_methods.append('len(checkpoint.class_names)')
        
        # 方法5: 从 meta.dataset_meta.classes 推断
        meta = checkpoint.get('meta', {})
        dataset_meta = meta.get('dataset_meta', {})
        if 'classes' in dataset_meta:
            classes = dataset_meta['classes']
            if isinstance(classes, (list, tuple)) and len(classes) > 0:
                return len(classes)
        tried_methods.append('len(meta.dataset_meta.classes)')
        
        # 方法6: 从 state_dict 推断 (decode_head 的输出通道数)
        state_dict = checkpoint.get('state_dict', checkpoint.get('model', {}))
        if isinstance(state_dict, dict):
            # 查找 decode_head.linear_pred.weight 或 decode_head.conv_seg.weight
            for key in state_dict.keys():
                if 'decode_head' in key and 'linear_pred.weight' in key:
                    nc = state_dict[key].shape[0]
                    if nc > 0:
                        return int(nc)
            tried_methods.append('state_dict.decode_head.linear_pred.weight.shape[0]')
            
            # 方法6b: 查找 conv_seg.weight (MMSegmentation 标准命名)
            for key in state_dict.keys():
                if 'decode_head' in key and 'conv_seg.weight' in key:
                    # conv_seg.weight 形状为 [num_classes, channels, 1, 1]
                    nc = state_dict[key].shape[0]
                    if nc > 0:
                        return int(nc)
        tried_methods.append('state_dict.decode_head.conv_seg.weight.shape[0]')
        
        raise NumClassesInferenceError(
            "无法推断 SegFormer 类别数",
            tried_methods=tried_methods
        )
    
    def get_input_spec(self, checkpoint: Dict[str, Any]) -> InputSpec:
        """
        获取输入规格
        
        SegFormer 使用像素级 ImageNet 归一化:
        - 输入范围: 0-255
        - 归一化: (pixel - mean) / std
        - mean = [123.675, 116.28, 103.53]
        - std = [58.395, 57.12, 57.375]
        """
        input_size = self._get_input_size(checkpoint)
        norm_mean, norm_std = self._get_normalize_params(checkpoint)
        
        return InputSpec(
            shape=(1, 3, input_size, input_size),
            dtype=torch.float32,
            channel_order="RGB",
            normalize_mean=norm_mean,
            normalize_std=norm_std,
            value_range=(0.0, 255.0),  # SegFormer 使用 0-255 范围
        )
    
    def get_output_spec(self, checkpoint: Dict[str, Any], 
                        num_classes: int) -> OutputSpec:
        """
        获取输出规格
        
        SegFormer 语义分割输出:
        - 形状: (B, num_classes, H, W)
        - 类型: logits (未经 softmax)
        """
        input_size = self._get_input_size(checkpoint)
        
        # 获取类别名称
        class_names = checkpoint.get('class_names', [])
        if not class_names:
            meta = checkpoint.get('meta', {})
            dataset_meta = meta.get('dataset_meta', {})
            class_names = list(dataset_meta.get('classes', []))
        
        return OutputSpec(
            structure=f"Tensor[B, {num_classes}, H, W] - semantic segmentation logits",
            num_classes=num_classes,
            output_type="tensor",
            extra_info={
                'task': 'semantic_segmentation',
                'output_stride': 4,  # SegFormer 默认输出步长
                'input_size': input_size,
                'class_names': class_names,
            }
        )
    
    def verify_forward(
        self,
        model: nn.Module,
        input_spec: InputSpec,
        device: torch.device,
        sample_input: Optional[torch.Tensor] = None,
    ) -> bool:
        """
        验证前向传播
        
        对于 SegFormerONNXWrapper:
        - 直接接受图像张量输入 (0-255 范围)
        - 输出分割 logits
        """
        try:
            model.eval()
            model.to(device)
            
            if sample_input is not None:
                x = sample_input.to(device)
            else:
                # 创建符合 SegFormer 输入要求的测试输入
                # 值范围 0-255，包装器内部会进行归一化
                x = torch.randint(
                    0, 256, 
                    input_spec.shape, 
                    dtype=torch.float32
                ).to(device)
            
            with torch.no_grad():
                # 包装后的模型可以直接调用 forward
                output = model(x)
                
                # 验证输出
                if isinstance(output, torch.Tensor):
                    B, C, H, W = output.shape
                    logger.info(f"✅ 前向传播验证成功")
                    logger.info(f"   输入形状: {x.shape}")
                    logger.info(f"   输出形状: {output.shape} (classes={C})")
                else:
                    logger.info(f"✅ 前向传播验证成功，输出类型: {type(output)}")
            
            return True
            
        except Exception as e:
            raise ForwardVerificationError(
                architecture="SegFormer",
                reason=str(e)
            )


# ==============================================================================
# 第6部分: 处理器注册与获取
# ==============================================================================

# 全局处理器列表 - 改为函数以避免状态污染
def _create_handlers() -> List[ModelHandler]:
    """创建新的处理器实例列表，避免状态污染"""
    return [
        # 分类 - timm (限定模型)
        TimmClassificationHandler(),
        # 检测 - 仅 YOLO
        YOLODetectionHandler(),
        # 分割 - SegFormer (MMSegmentation)
        SegFormerSegmentationHandler(),
    ]


def get_all_handlers() -> List[ModelHandler]:
    """获取所有已注册的处理器 - 每次返回新实例"""
    return _create_handlers()


def get_handlers_by_task(task_type: TaskType) -> List[ModelHandler]:
    """根据任务类型获取处理器 - 每次返回新实例"""
    return [h for h in _create_handlers() if h.task_type == task_type]


def get_handlers_by_framework(framework: Framework) -> List[ModelHandler]:
    """根据框架获取处理器 - 每次返回新实例"""
    return [h for h in _create_handlers() if h.framework == framework]


def get_supported_models() -> Dict[str, Dict[str, List[str]]]:
    """获取所有支持的模型列表"""
    result = {'cls': {}, 'det': {}, 'seg': {}}
    
    for handler in _create_handlers():
        task_key = handler.task_type.value
        framework_key = handler.framework.value
        
        if framework_key not in result[task_key]:
            result[task_key][framework_key] = []
        
        result[task_key][framework_key].extend(handler.supported_architectures)
    
    return result


def print_supported_models():
    """打印支持的模型列表（详细版）"""
    print("\n" + "=" * 70)
    print("📋 支持的模型列表")
    print("=" * 70)
    
    # 分类模型
    print("\n🏷️ 图像分类 (cls) - timm")
    print("-" * 70)
    
    cls_handler = TimmClassificationHandler()
    models_info = cls_handler.get_supported_models_info()
    
    # 按家族分组
    families = {}
    for name, info in models_info.items():
        family = info['family']
        if family not in families:
            families[family] = []
        families[family].append((name, info))
    
    for family, models in families.items():
        print(f"\n  [{family}]")
        for name, info in models:
            print(f"    • {name:<30} ({info['size']}, {info['params']}M, {info['acc']}%)")
    
    # 检测模型
    print("\n\n🏷️ 目标检测 (det) - ultralytics YOLO")
    print("-" * 70)
    det_handler = YOLODetectionHandler()
    for arch in det_handler.supported_architectures:
        print(f"    • {arch}")
    
    # 分割模型
    print("\n\n🏷️ 语义分割 (seg) - SegFormer (MMSegmentation)")
    print("-" * 70)
    seg_handler = SegFormerSegmentationHandler()
    for arch in seg_handler.supported_architectures:
        variant_info = seg_handler.SEGFORMER_VARIANTS.get(arch, {})
        params = variant_info.get('params', '?')
        print(f"    • {arch:<20} ({params}M params)")
    
    print("\n" + "=" * 70)


# ==============================================================================
# 第7部分: 主导入器 (ModelImporter)
# ==============================================================================

class DeviceManager:
    """设备管理器"""
    
    @staticmethod
    def get_device(device: Optional[str] = None) -> torch.device:
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")


class ModelImporter:
    """
    🚀 模型导入器 - 框架驱动版 (SegFormer 专用)
    
    支持的任务和框架:
    - 分类 (cls): timm - WideResNet, MobileNetV3, EfficientNet, ViT
    - 检测 (det): ultralytics - YOLO v5/v8/v11
    - 分割 (seg): mmsegmentation - SegFormer B0-B5
    """
    
    def __init__(self):
        self.handlers = get_all_handlers()
        logger.info(f"📋 已注册 {len(self.handlers)} 个模型处理器")
    
    def import_model(
        self,
        model_path: str,
        task_type: Union[str, TaskType],
        device: Optional[str] = None,
        num_classes: Optional[int] = None,
        verify_forward: bool = True,
        **kwargs,
    ) -> ModelInfo:
        """
        导入模型
        
        Args:
            model_path: 模型文件路径
            task_type: 任务类型 ('cls' / 'det' / 'seg')
            device: 设备
            num_classes: 类别数 (可选，自动推断)
            verify_forward: 是否验证前向传播
            **kwargs: 额外参数
            
        Returns:
            ModelInfo
        """
        # 1. 解析任务类型
        task = self._parse_task_type(task_type)
        logger.info(f"  任务类型: {task.value} ({task.name})")
        
        # 2. 加载 checkpoint
        checkpoint = self._load_checkpoint(model_path, task)
        
        # 3. 找到合适的处理器
        handler = self._find_handler(task, checkpoint)
        logger.info(f"  处理器: {type(handler).__name__} ({handler.framework.value})")
        
        # 4. 获取模型名称（传入 model_path 以支持从文件名推断）
        model_name = handler.get_model_name(checkpoint, model_path=model_path)
        logger.info(f"  模型架构: {model_name}")
        
        # 5. 获取类别数
        if num_classes is None:
            num_classes = handler.get_num_classes(checkpoint)
        logger.info(f"  类别数: {num_classes}")
        
        # 6. 重建模型
        logger.debug("  重建模型...")
        model = handler.rebuild(checkpoint, num_classes=num_classes, model_path=model_path, **kwargs)
        
        # 7. 移动到设备
        dev = DeviceManager.get_device(device)
        model.to(dev)
        model.eval()
        
        # 8. 获取规格
        input_spec = handler.get_input_spec(checkpoint)
        output_spec = handler.get_output_spec(checkpoint, num_classes)
        
        # 9. 前向验证
        onnx_ready = False
        if verify_forward:
            logger.debug("  前向验证...")
            try:
                handler.verify_forward(model, input_spec, dev)
                onnx_ready = True
            except ForwardVerificationError as e:
                logger.warning(f"  ⚠️ 前向验证失败: {e}")
        
        # 10. 构建结果
        config_str = None
        if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
            config_str = checkpoint['config'].get('yaml', '')
        
        info = ModelInfo(
            model=model,
            task_type=task,
            framework=handler.framework,
            device=dev,
            input_spec=input_spec,
            output_spec=output_spec,
            architecture=model_name,
            model_path=model_path,
            config=config_str,
            extra_metadata={
                'onnx_ready': onnx_ready,
                'handler': type(handler).__name__,
            }
        )
        
        return info
    
    def _parse_task_type(self, task_type: Union[str, TaskType]) -> TaskType:
        """解析任务类型"""
        if isinstance(task_type, TaskType):
            return task_type
        
        mapping = {
            'cls': TaskType.CLASSIFICATION,
            'classification': TaskType.CLASSIFICATION,
            'det': TaskType.DETECTION,
            'detection': TaskType.DETECTION,
            'seg': TaskType.SEGMENTATION,
            'segmentation': TaskType.SEGMENTATION,
        }
        
        task = mapping.get(task_type.lower())
        if task is None:
            raise ValueError(f"无效的任务类型: {task_type}")
        return task
    
    def _load_checkpoint(self, model_path: str, 
                         task: TaskType) -> Dict[str, Any]:
        """加载 checkpoint"""
        path = Path(model_path)
        
        if not path.exists():
            raise FileNotFoundError(f"❌ 文件不存在: {model_path}")
        
        file_size = path.stat().st_size / (1024 * 1024)
        logger.info(f"📁 加载: {model_path} ({file_size:.2f} MB)")
        
        # YOLO 特殊处理
        if task == TaskType.DETECTION:
            name_lower = path.stem.lower()
            if any(v in name_lower for v in ['yolo', 'yolov5', 'yolov8', 'yolov11']):
                logger.info("🔧 检测到 YOLO 模型，使用特殊加载...")
                return YOLOLoader.load(model_path)
        
        # 标准加载
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        except ModuleNotFoundError as e:
            if 'models' in str(e):
                logger.info("🔧 检测到可能是 YOLOv5，尝试特殊加载...")
                return YOLOLoader.load(model_path)
            raise
        
        # 确保是字典格式
        if isinstance(checkpoint, nn.Module):
            checkpoint = {'model': checkpoint.state_dict()}
        elif not isinstance(checkpoint, dict):
            checkpoint = {'model': checkpoint}
        
        return checkpoint
    
    def _find_handler(self, task: TaskType, 
                      checkpoint: Dict[str, Any]) -> ModelHandler:
        """找到合适的处理器"""
        handlers = get_handlers_by_task(task)
        
        if not handlers:
            raise UnsupportedModelError(f"没有 {task.value} 任务的处理器")
        
        for handler in handlers:
            if handler.can_handle(checkpoint):
                return handler
        
        # 调试信息
        logger.info("🔍 无法自动识别模型，checkpoint 信息:")
        for key in list(checkpoint.keys())[:10]:
            logger.info(f"    {key}: {type(checkpoint[key])}")
        
        all_supported = []
        for h in handlers:
            all_supported.extend(h.supported_architectures[:5])
        
        raise UnsupportedModelError(
            f"无法识别的 {task.value} 模型",
            task_type=task.value,
            supported=all_supported + ['...']
        )


# ==============================================================================
# 第8部分: 便捷函数与CLI
# ==============================================================================

def import_model(
    model_path: str,
    task_type: str,
    device: Optional[str] = None,
    num_classes: Optional[int] = None,
    verify_forward: bool = True,
    **kwargs,
) -> ModelInfo:
    """
    导入模型 (便捷函数)
    
    Args:
        model_path: 模型文件路径
        task_type: 任务类型 ('cls' / 'det' / 'seg')
        device: 设备
        num_classes: 类别数 (可选)
        verify_forward: 是否验证前向传播
        
    Returns:
        ModelInfo
    """
    return ModelImporter().import_model(
        model_path=model_path,
        task_type=task_type,
        device=device,
        num_classes=num_classes,
        verify_forward=verify_forward,
        **kwargs,
    )


def list_supported_models() -> None:
    """打印支持的模型列表"""
    print_supported_models()


# ==================== 模型保存工具 ====================

def save_timm_model(
    model: nn.Module,
    save_path: str,
    model_name: str,
    num_classes: int,
    input_size: int = 224,
    extra_info: Dict[str, Any] = None,
) -> None:
    """
    保存 timm 模型 (推荐格式)
    
    Args:
        model: timm 模型
        save_path: 保存路径
        model_name: timm 模型名称 (必须是支持的模型)
        num_classes: 类别数
        input_size: 输入尺寸
        extra_info: 额外信息
    """
    # 验证模型名称
    if not TimmClassificationHandler.is_model_supported(model_name):
        logger.warning(f"⚠️ 模型 '{model_name}' 不在支持列表中，保存可能无法正确加载")
    
    checkpoint = {
        'model': model.state_dict(),
        'framework': 'timm',
        'task': 'cls',
        'model_name': model_name,
        'num_classes': num_classes,
        'input_size': input_size,
    }
    
    if extra_info:
        checkpoint.update(extra_info)
    
    torch.save(checkpoint, save_path)
    logger.info(f"✅ timm 模型已保存: {save_path}")


def save_segformer_model(
    model: nn.Module,
    save_path: str,
    model_name: str,
    num_classes: int,
    class_names: List[str] = None,
    input_size: int = 512,
    extra_info: Dict[str, Any] = None,
) -> None:
    """
    保存 SegFormer 模型 (推荐格式)
    
    保存格式与训练软件完全兼容，可直接用于模型转换。
    
    Args:
        model: SegFormer 模型 (nn.Module)
        save_path: 保存路径 (.pth)
        model_name: 模型名称 (如 'segformer_b2')
        num_classes: 类别数
        class_names: 类别名称列表 (可选)
        input_size: 输入尺寸 (默认 512)
        extra_info: 额外信息 (可选)
        
    Example:
        >>> save_segformer_model(
        ...     model=model,
        ...     save_path='segformer_b2.pth',
        ...     model_name='segformer_b2',
        ...     num_classes=10,
        ...     class_names=['class0', 'class1', ...],
        ... )
    """
    # SegFormer 像素级归一化参数
    SEGFORMER_MEAN = [123.675, 116.28, 103.53]
    SEGFORMER_STD = [58.395, 57.12, 57.375]
    
    # 生成类别名称
    if class_names is None:
        class_names = [f'class_{i}' for i in range(num_classes)]
    
    # 生成调色板
    palette = []
    predefined_colors = [
        [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
        [255, 255, 0], [255, 0, 255], [0, 255, 255], [128, 0, 0],
        [0, 128, 0], [0, 0, 128], [128, 128, 0], [128, 0, 128],
    ]
    for i in range(num_classes):
        if i < len(predefined_colors):
            palette.append(predefined_colors[i])
        else:
            import random
            random.seed(i)
            palette.append([random.randint(0, 255) for _ in range(3)])
    
    # 构建 checkpoint
    checkpoint = {
        # 框架识别字段
        'framework': 'mmsegmentation',
        'model_type': 'segformer',
        'model_name': model_name,
        
        # 模型权重
        'state_dict': model.state_dict(),
        
        # MMSegmentation 需要的 meta 字段
        'meta': {
            'dataset_meta': {
                'classes': tuple(class_names),
                'palette': palette,
            },
            'CLASSES': tuple(class_names),
            'PALETTE': palette,
        },
        
        # 类别信息
        'nc': num_classes,
        'num_classes': num_classes,
        'names': {i: name for i, name in enumerate(class_names)},
        'class_names': class_names,
        
        # 预处理信息
        'model_metadata': {
            'model_name': model_name,
            'num_classes': num_classes,
            'class_names': class_names,
            'input_size': input_size,
            'input_spec': {
                'shape': (1, 3, input_size, input_size),
                'color_format': 'RGB',
                'pixel_range': (0, 255),
                'normalize_method': 'imagenet_pixel',
                'normalize_mean': SEGFORMER_MEAN,
                'normalize_std': SEGFORMER_STD,
            },
            'ignore_index': 255,
            'task': 'semantic_segmentation',
        },
    }
    
    # 添加额外信息
    if extra_info:
        checkpoint.update(extra_info)
    
    # 保存
    torch.save(checkpoint, save_path)
    logger.info(f"✅ SegFormer 模型已保存: {save_path}")


# 保留旧函数名以保持兼容性 (已弃用)
def save_detectron2_model(*args, **kwargs):
    """
    [已弃用] 请使用 save_segformer_model()
    
    此函数已弃用，仅保留用于向后兼容。
    """
    logger.warning("⚠️ save_detectron2_model() 已弃用，请使用 save_segformer_model()")
    raise NotImplementedError(
        "Detectron2 支持已移除。请使用 save_segformer_model() 保存 SegFormer 模型。"
    )


# ==============================================================================
# CLI 入口
# ==============================================================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="🚀 Stage 1: 模型导入器 (SegFormer 版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 导入 timm 分类模型
  python model_importer.py -m efficientnet.pth -t cls
  
  # 导入 YOLO 检测模型
  python model_importer.py -m yolov8.pt -t det
  
  # 导入 SegFormer 分割模型
  python model_importer.py -m segformer_b2.pth -t seg
  
  # 列出支持的模型
  python model_importer.py --list

支持的分类模型:
  WideResNet: wide_resnet50_2, wide_resnet101_2, wide_resnet50_4, wide_resnet101_4, wide_resnet200_2
  MobileNetV3: mobilenetv3_small_050, mobilenetv3_small_100, mobilenetv3_large_075, mobilenetv3_large_100, tf_mobilenetv3_large_100
  EfficientNet: efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4
  ViT: vit_tiny_patch16_224, vit_small_patch16_224, vit_base_patch16_224, vit_large_patch16_224, vit_huge_patch14_224

支持的分割模型:
  SegFormer: segformer_b0, segformer_b1, segformer_b2, segformer_b3, segformer_b4, segformer_b5
        """
    )
    
    parser.add_argument("-m", "--model", help="模型文件路径")
    parser.add_argument("-t", "--task", choices=["cls", "det", "seg"], help="任务类型")
    parser.add_argument("-d", "--device", default=None, help="设备")
    parser.add_argument("--num-classes", type=int, default=None, help="类别数")
    parser.add_argument("--no-verify", action="store_true", help="跳过前向验证")
    parser.add_argument("--yolov5-path", type=str, default=None, help="YOLOv5 源码路径")
    parser.add_argument("--list", action="store_true", help="列出支持的模型")
    
    args = parser.parse_args()
    
    # 列出支持的模型
    if args.list:
        console.header("📦 支持的模型列表")
        
        handlers = [
            TimmClassificationHandler(),
            YOLODetectionHandler(),
            SegFormerSegmentationHandler(),
        ]
        
        for handler in handlers:
            console.tree([
                (f"{handler.framework.value} ({handler.task_type.value})", handler.supported_architectures[:10]),
            ])
        return 0
    
    # 模型导入
    if args.model is None or args.task is None:
        parser.print_help()
        return 0
    
    try:
        info = import_model(
            model_path=args.model,
            task_type=args.task,
            device=args.device,
            num_classes=args.num_classes,
            verify_forward=not args.no_verify,
        )
        print(info.summary())
        return 0
        
    except (FileNotFoundError, UnsupportedModelError, ModelRebuildError,
            NumClassesInferenceError, ForwardVerificationError,
            FrameworkNotInstalledError) as e:
        logger.error(str(e))
        return 1
    
    except Exception as e:
        logger.error(f"❌ 未知错误: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())