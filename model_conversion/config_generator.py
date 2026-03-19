#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📦 Stage 8: 配置文件生成器 (Config Generator)
==============================================

生成模型部署所需的配置文件 (YAML 格式)。

配置文件包含:
- 基本信息: 模型名称、任务类型、架构、框架
- 输入规格: shape、dtype、动态轴
- 预处理配置: 归一化、letterbox (检测/分割)
- 输出规格: 输出名称、类别数
- 转换信息: 后端、精度、文件路径
- 验证结果: 精度指标、性能指标 (可选)

使用方法:
---------
>>> from config_generator import ConfigGenerator, generate_config
>>> 
>>> # 方式1: 便捷函数
>>> config_path = generate_config(
...     model_name="yolov8n",
...     task_type="det",
...     input_shape=(1, 3, 640, 640),
...     output_dir="./output",
...     backend="tensorrt",
...     precision="fp16",
... )
>>>
>>> # 方式2: 从 Pipeline 上下文生成
>>> generator = ConfigGenerator()
>>> config_path = generator.generate_from_context(ctx, output_dir)

作者: Model Converter Team
版本: 1.0.0
"""

import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any, Union
from unified_logger import Logger, console, Timer


# YAML 支持
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = Logger.get("config_generator")


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class MetaConfig:
    """基本信息配置"""
    model_name: str
    task_type: str                           # cls/det/seg
    architecture: str = ""
    framework: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))


@dataclass
class DynamicAxisRange:
    """动态轴范围"""
    min: int
    opt: int
    max: int


@dataclass
class InputConfig:
    """输入配置"""
    name: str = "images"
    shape: List[int] = field(default_factory=lambda: [1, 3, 640, 640])
    dtype: str = "float32"
    dynamic_axes: Optional[Dict[str, Dict[str, int]]] = None  # {"batch": {"min": 1, "opt": 4, "max": 16}}


@dataclass
class NormalizeConfig:
    """归一化配置"""
    mean: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    std: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])


@dataclass
class LetterboxConfig:
    """Letterbox 配置 (检测/分割任务)"""
    enabled: bool = True
    color: List[int] = field(default_factory=lambda: [114, 114, 114])
    stride: int = 32


@dataclass
class PreprocessingConfig:
    """预处理配置"""
    channel_order: str = "RGB"
    value_range: List[float] = field(default_factory=lambda: [0.0, 1.0])
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)
    letterbox: Optional[LetterboxConfig] = None  # 仅 det/seg


@dataclass
class OutputConfig:
    """输出配置"""
    names: List[str] = field(default_factory=lambda: ["output0"])
    num_classes: Optional[int] = None


@dataclass
class ConversionInfo:
    """转换信息配置"""
    backend: str = "tensorrt"
    precision: str = "fp16"
    opset: int = 17
    model_file: str = ""
    model_size_mb: Optional[float] = None


@dataclass 
class ValidationInfo:
    """验证结果配置"""
    cosine_similarity: Optional[float] = None
    max_diff: Optional[float] = None
    latency_ms: Optional[float] = None
    throughput_fps: Optional[float] = None


@dataclass
class ModelConfig:
    """完整模型配置"""
    meta: MetaConfig
    input: InputConfig
    preprocessing: PreprocessingConfig
    output: OutputConfig
    conversion: ConversionInfo
    validation: Optional[ValidationInfo] = None
    
    def to_dict(self) -> Dict:
        """转换为字典 (用于 YAML 序列化)"""
        result = {}
        
        # meta
        result['meta'] = asdict(self.meta)
        
        # input
        input_dict = asdict(self.input)
        # 移除 None 的 dynamic_axes
        if input_dict.get('dynamic_axes') is None:
            del input_dict['dynamic_axes']
        result['input'] = input_dict
        
        # preprocessing
        preproc_dict = {
            'channel_order': self.preprocessing.channel_order,
            'value_range': self.preprocessing.value_range,
            'normalize': asdict(self.preprocessing.normalize),
        }
        # 仅在 letterbox 存在时添加
        if self.preprocessing.letterbox is not None:
            preproc_dict['letterbox'] = asdict(self.preprocessing.letterbox)
        result['preprocessing'] = preproc_dict
        
        # output
        output_dict = {'names': self.output.names}
        if self.output.num_classes is not None:
            output_dict['num_classes'] = self.output.num_classes
        result['output'] = output_dict
        
        # conversion
        conv_dict = {
            'backend': self.conversion.backend,
            'precision': self.conversion.precision,
            'opset': self.conversion.opset,
            'model_file': self.conversion.model_file,
        }
        if self.conversion.model_size_mb is not None:
            conv_dict['model_size_mb'] = round(self.conversion.model_size_mb, 2)
        result['conversion'] = conv_dict
        
        # validation (仅在有数据时添加)
        if self.validation is not None:
            val_dict = {}
            if self.validation.cosine_similarity is not None:
                val_dict['cosine_similarity'] = round(self.validation.cosine_similarity, 6)
            if self.validation.max_diff is not None:
                val_dict['max_diff'] = round(self.validation.max_diff, 6)
            if self.validation.latency_ms is not None:
                val_dict['latency_ms'] = round(self.validation.latency_ms, 2)
            if self.validation.throughput_fps is not None:
                val_dict['throughput_fps'] = round(self.validation.throughput_fps, 2)
            if val_dict:
                result['validation'] = val_dict
        
        return result


# ============================================================================
# 配置生成器
# ============================================================================

class ConfigGenerator:
    """
    配置文件生成器
    
    从 Pipeline 上下文或手动参数生成部署配置文件
    """
    
    DEFAULT_CONFIG_NAME = "model_config.yaml"
    
    def __init__(self):
        """初始化生成器"""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required. Install with: pip install pyyaml")
    
    def generate_from_context(
        self,
        ctx: Any,
        output_dir: str,
        conversion_result: Any = None,
        config_name: str = None,
    ) -> str:
        """
        从 Pipeline 上下文生成配置文件
        
        Args:
            ctx: PipelineContext 对象
            output_dir: 输出目录
            conversion_result: ConversionResult 对象 (可选)
            config_name: 配置文件名 (可选)
            
        Returns:
            配置文件路径
        """
        # 提取 meta 信息
        meta = MetaConfig(
            model_name=getattr(ctx, 'model_name', 'unknown'),
            task_type=getattr(ctx, 'task_type', 'cls'),
            architecture=getattr(ctx, 'model_name', ''),
            framework=getattr(ctx, 'framework', ''),
        )
        
        # 提取 input 信息
        input_shape = list(getattr(ctx, 'input_shape', [1, 3, 640, 640]))
        input_name = "images"
        
        # 从 input_spec 获取更多信息
        input_spec = getattr(ctx, 'input_spec', None)
        if input_spec:
            if hasattr(input_spec, 'shape') and input_spec.shape:
                input_shape = list(input_spec.shape)
        
        # 构建动态轴配置
        dynamic_axes = None
        dynamic_axes_spec = getattr(ctx, 'dynamic_axes_spec', None)
        if dynamic_axes_spec is not None:
            dynamic_axes = self._extract_dynamic_axes(dynamic_axes_spec)
        
        input_config = InputConfig(
            name=input_name,
            shape=input_shape,
            dtype="float32",
            dynamic_axes=dynamic_axes,
        )
        
        # 提取预处理信息
        normalize_mean = [0.0, 0.0, 0.0]
        normalize_std = [1.0, 1.0, 1.0]
        channel_order = "RGB"
        value_range = [0.0, 1.0]
        
        if input_spec:
            if hasattr(input_spec, 'normalize_mean') and input_spec.normalize_mean:
                normalize_mean = list(input_spec.normalize_mean)
            if hasattr(input_spec, 'normalize_std') and input_spec.normalize_std:
                normalize_std = list(input_spec.normalize_std)
            if hasattr(input_spec, 'channel_order'):
                channel_order = input_spec.channel_order
            if hasattr(input_spec, 'value_range') and input_spec.value_range:
                value_range = list(input_spec.value_range)
        
        # Letterbox 配置 (仅 det/seg)
        letterbox = None
        task_type = getattr(ctx, 'task_type', 'cls')
        if task_type in ('det', 'seg'):
            letterbox = LetterboxConfig(
                enabled=True,
                color=[114, 114, 114],
                stride=32,
            )
        
        preprocessing = PreprocessingConfig(
            channel_order=channel_order,
            value_range=value_range,
            normalize=NormalizeConfig(mean=normalize_mean, std=normalize_std),
            letterbox=letterbox,
        )
        
        # 提取输出信息
        output_names = ["output0"]
        num_classes = None
        
        output_spec = getattr(ctx, 'output_spec', None)
        if output_spec:
            if hasattr(output_spec, 'num_classes'):
                num_classes = output_spec.num_classes
        
        # 从 export_result 获取输出名称
        export_result = getattr(ctx, 'export_result', None)
        if export_result and hasattr(export_result, 'onnx_model') and export_result.onnx_model:
            try:
                onnx_model = export_result.onnx_model
                output_names = [o.name for o in onnx_model.graph.output]
            except (AttributeError, TypeError, Exception) as e:
                logger.debug(f"Failed to extract output names from ONNX model: {e}")
        
        output_config = OutputConfig(
            names=output_names,
            num_classes=num_classes,
        )
        
        # 提取转换信息
        backend = getattr(ctx, 'target_backend', 'onnxruntime')
        precision = getattr(ctx, 'precision', 'fp16')
        opset = getattr(ctx, 'opset', 17)
        model_file = ""
        model_size_mb = None
        
        if conversion_result:
            if hasattr(conversion_result, 'target_backend'):
                backend_val = conversion_result.target_backend
                backend = backend_val.value if hasattr(backend_val, 'value') else str(backend_val)
            if hasattr(conversion_result, 'precision_mode'):
                precision_val = conversion_result.precision_mode
                precision = precision_val.value if hasattr(precision_val, 'value') else str(precision_val)
            if hasattr(conversion_result, 'output_files') and conversion_result.output_files:
                model_file = os.path.basename(conversion_result.output_files[0])
            if hasattr(conversion_result, 'validation') and conversion_result.validation:
                if hasattr(conversion_result.validation, 'output_size_mb'):
                    model_size_mb = conversion_result.validation.output_size_mb
        else:
            # 从 ctx 获取 ONNX 路径
            onnx_path = getattr(ctx, 'onnx_path', '')
            if onnx_path:
                model_file = os.path.basename(onnx_path)
                if os.path.exists(onnx_path):
                    model_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        
        conversion = ConversionInfo(
            backend=backend,
            precision=precision,
            opset=opset,
            model_file=model_file,
            model_size_mb=model_size_mb,
        )
        
        # 提取验证信息
        validation = None
        if conversion_result and hasattr(conversion_result, 'validation') and conversion_result.validation:
            val = conversion_result.validation
            validation = ValidationInfo(
                cosine_similarity=getattr(val, 'cosine_sim', None),
                max_diff=getattr(val, 'max_diff', None),
                latency_ms=getattr(val, 'latency_ms', None),
                throughput_fps=getattr(val, 'throughput_fps', None),
            )
        
        # 构建完整配置
        model_config = ModelConfig(
            meta=meta,
            input=input_config,
            preprocessing=preprocessing,
            output=output_config,
            conversion=conversion,
            validation=validation,
        )
        
        # 保存配置文件
        return self.save_config(model_config, output_dir, config_name)
    
    def generate(
        self,
        model_name: str,
        task_type: str,
        input_shape: Tuple[int, ...],
        output_dir: str,
        backend: str = "tensorrt",
        precision: str = "fp16",
        opset: int = 17,
        model_file: str = "",
        architecture: str = "",
        framework: str = "",
        num_classes: Optional[int] = None,
        normalize_mean: Tuple[float, ...] = (0.0, 0.0, 0.0),
        normalize_std: Tuple[float, ...] = (1.0, 1.0, 1.0),
        dynamic_axes: Optional[Dict] = None,
        validation_result: Any = None,
        config_name: str = None,
    ) -> str:
        """
        手动生成配置文件
        
        Args:
            model_name: 模型名称
            task_type: 任务类型 (cls/det/seg)
            input_shape: 输入形状 (B, C, H, W)
            output_dir: 输出目录
            backend: 目标后端
            precision: 精度模式
            opset: ONNX opset 版本
            model_file: 模型文件名
            architecture: 模型架构
            framework: 框架名称
            num_classes: 类别数
            normalize_mean: 归一化均值
            normalize_std: 归一化标准差
            dynamic_axes: 动态轴配置
            validation_result: 验证结果
            config_name: 配置文件名
            
        Returns:
            配置文件路径
        """
        # meta
        meta = MetaConfig(
            model_name=model_name,
            task_type=task_type,
            architecture=architecture or model_name,
            framework=framework,
        )
        
        # input
        input_config = InputConfig(
            name="images",
            shape=list(input_shape),
            dtype="float32",
            dynamic_axes=dynamic_axes,
        )
        
        # preprocessing
        letterbox = None
        if task_type in ('det', 'seg'):
            letterbox = LetterboxConfig()
        
        preprocessing = PreprocessingConfig(
            channel_order="RGB",
            value_range=[0.0, 1.0],
            normalize=NormalizeConfig(
                mean=list(normalize_mean),
                std=list(normalize_std),
            ),
            letterbox=letterbox,
        )
        
        # output
        output_config = OutputConfig(
            names=["output0"],
            num_classes=num_classes,
        )
        
        # conversion
        model_size_mb = None
        if model_file and os.path.exists(os.path.join(output_dir, model_file)):
            model_size_mb = os.path.getsize(os.path.join(output_dir, model_file)) / (1024 * 1024)
        
        conversion = ConversionInfo(
            backend=backend,
            precision=precision,
            opset=opset,
            model_file=model_file,
            model_size_mb=model_size_mb,
        )
        
        # validation
        validation = None
        if validation_result:
            validation = ValidationInfo(
                cosine_similarity=getattr(validation_result, 'cosine_sim', None),
                max_diff=getattr(validation_result, 'max_diff', None),
                latency_ms=getattr(validation_result, 'latency_ms', None),
                throughput_fps=getattr(validation_result, 'throughput_fps', None),
            )
        
        # 构建完整配置
        model_config = ModelConfig(
            meta=meta,
            input=input_config,
            preprocessing=preprocessing,
            output=output_config,
            conversion=conversion,
            validation=validation,
        )
        
        return self.save_config(model_config, output_dir, config_name)
    
    def save_config(
        self,
        config: ModelConfig,
        output_dir: str,
        config_name: str = None,
    ) -> str:
        """
        保存配置文件
        
        Args:
            config: ModelConfig 对象
            output_dir: 输出目录
            config_name: 配置文件名
            
        Returns:
            配置文件路径
        """
        # 确保输出目录存在
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 确定配置文件名
        if config_name is None:
            config_name = self.DEFAULT_CONFIG_NAME
        
        config_path = output_dir / config_name
        
        # 转换为字典
        config_dict = config.to_dict()
        
        # 保存为 YAML
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                config_dict,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                indent=2,
            )
        
        logger.info(f"配置文件已保存: {config_path}")
        return str(config_path)
    
    def _extract_dynamic_axes(self, dynamic_axes_spec: Any) -> Optional[Dict]:
        """
        从 DynamicAxesSpec 提取动态轴配置
        
        Args:
            dynamic_axes_spec: DynamicAxesSpec 对象
            
        Returns:
            动态轴配置字典
        """
        if dynamic_axes_spec is None:
            return None
        
        result = {}
        
        # 尝试从 to_tensorrt_profile 方法获取
        if hasattr(dynamic_axes_spec, 'to_tensorrt_profile'):
            try:
                profile = dynamic_axes_spec.to_tensorrt_profile()
                for input_name, axes_info in profile.items():
                    for axis_name, (min_val, opt_val, max_val) in axes_info.items():
                        result[axis_name] = {
                            'min': min_val,
                            'opt': opt_val,
                            'max': max_val,
                        }
                if result:
                    return result
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug(f"Failed to get TensorRT profile: {e}")
        
        # 尝试从 input_axes 属性获取
        if hasattr(dynamic_axes_spec, 'input_axes'):
            try:
                for input_name, axes_list in dynamic_axes_spec.input_axes.items():
                    for axis in axes_list:
                        if hasattr(axis, 'name'):
                            result[axis.name] = {
                                'min': getattr(axis, 'min_value', 1),
                                'opt': getattr(axis, 'opt_value', 1),
                                'max': getattr(axis, 'max_value', 1),
                            }
                if result:
                    return result
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug(f"Failed to extract input axes: {e}")
        
        return None if not result else result


# ============================================================================
# 便捷函数
# ============================================================================

def generate_config(
    model_name: str,
    task_type: str,
    input_shape: Tuple[int, ...],
    output_dir: str,
    backend: str = "tensorrt",
    precision: str = "fp16",
    **kwargs,
) -> str:
    """
    便捷函数：生成配置文件
    
    Args:
        model_name: 模型名称
        task_type: 任务类型 (cls/det/seg)
        input_shape: 输入形状 (B, C, H, W)
        output_dir: 输出目录
        backend: 目标后端
        precision: 精度模式
        **kwargs: 其他参数
        
        
    Returns:
        配置文件路径
        
    Example:
        >>> config_path = generate_config(
        ...     model_name="yolov8n",
        ...     task_type="det",
        ...     input_shape=(1, 3, 640, 640),
        ...     output_dir="./output",
        ...     backend="tensorrt",
        ...     precision="fp16",
        ... )
    """
    generator = ConfigGenerator()
    return generator.generate(
        model_name=model_name,
        task_type=task_type,
        input_shape=input_shape,
        output_dir=output_dir,
        backend=backend,
        precision=precision,
        **kwargs,
    )


def generate_config_from_context(
    ctx: Any,
    output_dir: str,
    conversion_result: Any = None,
) -> str:
    """
    便捷函数：从 Pipeline 上下文生成配置文件
    
    Args:
        ctx: PipelineContext 对象
        output_dir: 输出目录
        conversion_result: ConversionResult 对象 (可选)
        
    Returns:
        配置文件路径
    """
    generator = ConfigGenerator()
    return generator.generate_from_context(
        ctx=ctx,
        output_dir=output_dir,
        conversion_result=conversion_result,
    )


# ============================================================================
# 配置加载器 (部署时使用)
# ============================================================================

def load_config(config_path: str) -> Dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ============================================================================
# CLI 接口
# ============================================================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Model Config Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("-n", "--name", required=True, help="Model name")
    parser.add_argument("-t", "--task", required=True, 
                       choices=["cls", "det", "seg"], help="Task type")
    parser.add_argument("-s", "--shape", required=True, 
                       help="Input shape (B,C,H,W)")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("-b", "--backend", default="tensorrt",
                       choices=["ort", "tensorrt", "openvino"],
                       help="Target backend")
    parser.add_argument("-p", "--precision", default="fp16",
                       choices=["fp32", "fp16", "int8"],
                       help="Precision mode")
    parser.add_argument("--model-file", default="", help="Model file name")
    parser.add_argument("--num-classes", type=int, help="Number of classes")
    
    args = parser.parse_args()
    
    # 解析输入形状
    input_shape = tuple(int(x) for x in args.shape.split(','))
    
    # 生成配置
    config_path = generate_config(
        model_name=args.name,
        task_type=args.task,
        input_shape=input_shape,
        output_dir=args.output,
        backend=args.backend,
        precision=args.precision,
        model_file=args.model_file,
        num_classes=args.num_classes,
    )
    
    print(f"✅ 配置文件已生成: {config_path}")
    return 0


if __name__ == "__main__":
    exit(main())
