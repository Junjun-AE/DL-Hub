#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenVINO 量化转换器

支持的精度模式:
- FP32: 直接转换
- FP16: 半精度压缩
- INT8: NNCF 量化
- MIXED: IgnoredScope 混合精度

输出格式: .xml + .bin

版本要求: OpenVINO >= 2023.2

作者: Industrial ML Team
版本: 1.0.0
"""

import os
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any, Iterator

import numpy as np
from unified_logger import Logger, console, Timer

from model_converter import (
    BaseConverter,
    ConversionConfig,
    ConversionResult,
    PrecisionMode,
    CalibrationDataManager,
    SensitiveLayerManager,
    CalibrationError,
    DependencyError,
    get_logger,
    TQDM_AVAILABLE,
)

if TQDM_AVAILABLE:
    from tqdm import tqdm

# 条件导入
try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# OpenVINO
OV_AVAILABLE = False
OV_VERSION = None

try:
    import openvino as ov
    from openvino.runtime import Core, Type, serialize
    OV_AVAILABLE = True
    OV_VERSION = ov.__version__
except ImportError:
    pass

# NNCF (Neural Network Compression Framework)
NNCF_AVAILABLE = False
NNCF_VERSION = None

try:
    import nncf
    from nncf import Dataset
    from nncf.quantization import quantize
    from nncf.parameters import TargetDevice
    NNCF_AVAILABLE = True
    NNCF_VERSION = nncf.__version__
except ImportError:
    pass


logger = Logger.get("OpenVINOConverter")


# ============================================================================
# NNCF 数据集适配器
# ============================================================================

class NNCFCalibrationDataset:
    """
    NNCF 校准数据集
    
    包装 CalibrationDataManager 以适配 NNCF Dataset 接口。
    """
    
    def __init__(
        self,
        calib_data_manager: CalibrationDataManager,
        num_samples: int = 300,
    ):
        """
        初始化 NNCF 数据集
        
        Args:
            calib_data_manager: 校准数据管理器
            num_samples: 校准样本数
        """
        self.calib_data_manager = calib_data_manager
        self.num_samples = min(num_samples, len(calib_data_manager))
        
        # 预加载数据
        self.data_list = []
        count = 0
        for batch in calib_data_manager.get_single_iterator():
            if count >= self.num_samples:
                break
            self.data_list.append(batch)
            count += 1
        
        logger.info(f"NNCF dataset: {len(self.data_list)} samples loaded")
    
    def __iter__(self) -> Iterator[np.ndarray]:
        """迭代返回校准数据"""
        for data in self.data_list:
            yield data
    
    def __len__(self) -> int:
        return len(self.data_list)


def create_nncf_dataset(
    calib_data_manager: CalibrationDataManager,
    num_samples: int = 300,
) -> Any:
    """
    创建 NNCF Dataset
    
    Args:
        calib_data_manager: 校准数据管理器
        num_samples: 校准样本数
        
    Returns:
        NNCF Dataset 对象
    """
    if not NNCF_AVAILABLE:
        raise DependencyError("NNCF is required for INT8 quantization")
    
    nncf_dataset = NNCFCalibrationDataset(calib_data_manager, num_samples)
    
    # 创建 NNCF Dataset
    def transform_fn(data):
        return data
    
    return Dataset(nncf_dataset, transform_fn)


# ============================================================================
# OpenVINO 转换器
# ============================================================================

class OpenVINOConverter(BaseConverter):
    """
    OpenVINO 量化转换器
    
    支持 FP32/FP16/INT8/MIXED 精度模式。
    使用 NNCF 进行 INT8 量化。
    """
    
    # 最低版本要求
    MIN_OV_VERSION = "2023.2.0"
    MIN_NNCF_VERSION = "2.5.0"
    
    def __init__(self, config: ConversionConfig):
        """
        初始化 OpenVINO 转换器
        
        Args:
            config: 转换配置
        """
        super().__init__(config)
        
        self.core = None
        if OV_AVAILABLE:
            self.core = Core()
    
    def check_dependencies(self) -> bool:
        """检查依赖"""
        # 使用公共方法检查 ONNX
        if not self._check_onnx_available():
            return False
        
        if not OV_AVAILABLE:
            self.logger.error(
                "OpenVINO is not installed. "
                "Run: pip install openvino>=2023.2"
            )
            return False
        
        # 检查版本
        if OV_VERSION:
            self.logger.info(f"OpenVINO version: {OV_VERSION}")
        
        # INT8/MIXED 需要 NNCF
        if self.config.precision_mode in (PrecisionMode.INT8, PrecisionMode.MIXED):
            if not NNCF_AVAILABLE:
                self.logger.error(
                    "NNCF is not installed. Required for INT8 quantization. "
                    "Run: pip install nncf>=2.5.0"
                )
                return False
            
            self.logger.info(f"NNCF version: {NNCF_VERSION}")
        
        return True
    
    def convert(
        self,
        onnx_path: str,
        output_path: str,
        input_shape: Tuple[int, ...],
        dynamic_axes_spec: Optional[Dict] = None,
    ) -> ConversionResult:
        """
        执行 OpenVINO 转换
        
        Args:
            onnx_path: 输入 ONNX 模型路径
            output_path: 输出路径
            input_shape: 输入形状 (B, C, H, W)
            dynamic_axes_spec: 动态轴规格
            
        Returns:
            转换结果
        """
        precision = self.config.precision_mode
        
        self.logger.info(f"OpenVINO Conversion: {precision.value}")
        self.logger.info(f"OpenVINO version: {OV_VERSION}")
        
        try:
            if precision == PrecisionMode.FP32:
                return self._convert_fp32(onnx_path, output_path)
            
            elif precision == PrecisionMode.FP16:
                return self._convert_fp16(onnx_path, output_path)
            
            elif precision == PrecisionMode.INT8:
                return self._convert_int8(onnx_path, output_path, input_shape)
            
            elif precision == PrecisionMode.MIXED:
                return self._convert_mixed(onnx_path, output_path, input_shape)
            
            else:
                raise ValueError(f"Unsupported precision mode: {precision}")
                
        except Exception as e:
            self.logger.error(f"OpenVINO conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return ConversionResult(
                success=False,
                target_backend=self.config.target_backend,
                precision_mode=precision,
                output_files=[],
                message=str(e),
            )
    
    def _convert_fp32(self, onnx_path: str, output_path: str) -> ConversionResult:
        """FP32 转换"""
        self.logger.info("Converting to OpenVINO FP32...")
        
        start_time = time.time()
        
        # 读取 ONNX 模型
        ov_model = self.core.read_model(onnx_path)
        
        # 准备输出路径
        output_path = self._prepare_output_path(output_path)
        xml_path = str(output_path / "model.xml")
        bin_path = str(output_path / "model.bin")
        
        # 保存模型
        serialize(ov_model, xml_path)
        
        build_time = time.time() - start_time
        
        self.logger.info(f"FP32 model saved to: {xml_path}")
        
        return ConversionResult(
            success=True,
            target_backend=self.config.target_backend,
            precision_mode=PrecisionMode.FP32,
            output_files=[xml_path, bin_path],
            build_time_seconds=build_time,
            message="OpenVINO FP32 conversion successful",
            stats={
                "original_size_mb": self.get_file_size_mb(onnx_path),
                "output_size_mb": self.get_file_size_mb(str(output_path)),
                "openvino_version": OV_VERSION,
            },
        )
    
    def _convert_fp16(self, onnx_path: str, output_path: str) -> ConversionResult:
        """FP16 转换"""
        self.logger.info("Converting to OpenVINO FP16...")
        
        start_time = time.time()
        
        # 检查原始ONNX模型的动态轴信息
        dynamic_info = self._get_onnx_dynamic_axes_info(onnx_path)
        if dynamic_info['has_dynamic']:
            self.logger.info(f"检测到动态轴: {dynamic_info['details']}")
        
        # 读取 ONNX 模型
        ov_model = self.core.read_model(onnx_path)
        
        # 设置动态batch（如果配置启用或原始模型有动态轴）
        if getattr(self.config, 'ov_dynamic_batch_enabled', False) or \
           getattr(self.config, 'trt_dynamic_batch_enabled', False) or \
           dynamic_info['has_dynamic']:
            ov_model = self._set_dynamic_batch(ov_model)
            self.logger.info("已为 OpenVINO 设置动态 batch")
        
        # FP16 压缩
        from openvino.runtime import save_model
        from openvino.runtime.passes import Manager, Serialize
        
        # 准备输出路径
        output_path = self._prepare_output_path(output_path)
        xml_path = str(output_path / "model.xml")
        bin_path = str(output_path / "model.bin")
        
        # 使用 compress_to_fp16 参数保存
        # OpenVINO 2023.2+ 支持
        try:
            # 新版本 API
            ov.save_model(ov_model, xml_path, compress_to_fp16=True)
        except TypeError:
            # 旧版本 API fallback
            serialize(ov_model, xml_path)
            self.logger.warning("FP16 compression not available, using FP32")
        
        build_time = time.time() - start_time
        
        self.logger.info(f"FP16 model saved to: {xml_path}")
        
        return ConversionResult(
            success=True,
            target_backend=self.config.target_backend,
            precision_mode=PrecisionMode.FP16,
            output_files=[xml_path, bin_path],
            build_time_seconds=build_time,
            message="OpenVINO FP16 conversion successful",
            stats={
                "original_size_mb": self.get_file_size_mb(onnx_path),
                "output_size_mb": self.get_file_size_mb(str(output_path)),
                "openvino_version": OV_VERSION,
                "dynamic_batch_enabled": dynamic_info['has_dynamic'] or \
                                         getattr(self.config, 'ov_dynamic_batch_enabled', False),
            },
        )
    
    def _get_onnx_dynamic_axes_info(self, onnx_path: str) -> dict:
        """获取ONNX模型的动态轴信息"""
        info = {'has_dynamic': False, 'details': []}
        
        try:
            model = onnx.load(onnx_path)
            
            for inp in model.graph.input:
                input_name = inp.name
                shape_info = []
                has_dynamic = False
                
                for i, dim in enumerate(inp.type.tensor_type.shape.dim):
                    if dim.HasField('dim_param') and dim.dim_param:
                        shape_info.append(f"dim[{i}]={dim.dim_param}")
                        has_dynamic = True
                    elif dim.HasField('dim_value'):
                        shape_info.append(str(dim.dim_value))
                    else:
                        shape_info.append("?")
                        has_dynamic = True
                
                if has_dynamic:
                    info['has_dynamic'] = True
                    info['details'].append(f"{input_name}: [{', '.join(shape_info)}]")
        except Exception as e:
            self.logger.debug(f"获取ONNX动态轴信息失败: {e}")
        
        return info
    
    def _set_dynamic_batch(self, ov_model):
        """为OpenVINO模型设置动态batch维度"""
        try:
            from openvino.runtime import PartialShape, Dimension
            
            # 获取所有输入
            for i in range(len(ov_model.inputs)):
                input_node = ov_model.inputs[i]
                shape = input_node.get_partial_shape()
                
                # 检查是否是4D输入 (NCHW)
                if len(shape) == 4:
                    # 将batch维度设置为动态 (-1 表示任意值)
                    new_shape = PartialShape([
                        Dimension(-1),  # batch: 动态
                        shape[1],       # channels: 保持不变
                        shape[2],       # height: 保持不变
                        shape[3],       # width: 保持不变
                    ])
                    ov_model.reshape({input_node.get_any_name(): new_shape})
                    self.logger.info(f"输入 '{input_node.get_any_name()}' 已设置为动态 batch")
            
        except Exception as e:
            self.logger.warning(f"设置动态batch失败: {e}")
            self.logger.info("将使用静态batch")
        
        return ov_model
    
    def _convert_int8(
        self,
        onnx_path: str,
        output_path: str,
        input_shape: Tuple[int, ...],
    ) -> ConversionResult:
        """INT8 量化"""
        self.logger.info("Converting to OpenVINO INT8 with NNCF...")
        
        start_time = time.time()
        
        # 读取 ONNX 模型
        ov_model = self.core.read_model(onnx_path)
        
        # 创建校准数据管理器
        calib_manager = CalibrationDataManager(
            config=self.config,
            input_shape=input_shape,
        )
        
        # 创建 NNCF 数据集
        calibration_dataset = create_nncf_dataset(
            calib_manager,
            num_samples=self.config.calib_num_samples,
        )
        
        # 配置量化参数
        self.logger.info(f"Calibration samples: {len(calib_manager)}")
        self.logger.info(f"Target device: CPU")
        
        # 执行量化
        quantized_model = quantize(
            model=ov_model,
            calibration_dataset=calibration_dataset,
            target_device=TargetDevice.CPU,
            subset_size=min(self.config.calib_num_samples, len(calib_manager)),
            fast_bias_correction=True,
        )
        
        # 准备输出路径
        output_path = self._prepare_output_path(output_path)
        xml_path = str(output_path / "model.xml")
        bin_path = str(output_path / "model.bin")
        
        # 保存量化模型
        serialize(quantized_model, xml_path)
        
        build_time = time.time() - start_time
        
        self.logger.info(f"INT8 model saved to: {xml_path}")
        
        return ConversionResult(
            success=True,
            target_backend=self.config.target_backend,
            precision_mode=PrecisionMode.INT8,
            output_files=[xml_path, bin_path],
            build_time_seconds=build_time,
            message="OpenVINO INT8 quantization successful",
            stats={
                "original_size_mb": self.get_file_size_mb(onnx_path),
                "output_size_mb": self.get_file_size_mb(str(output_path)),
                "openvino_version": OV_VERSION,
                "nncf_version": NNCF_VERSION,
                "calibration_samples": len(calib_manager),
            },
        )
    
    def _get_sensitive_patterns(self, sensitive_manager: SensitiveLayerManager) -> List[str]:
        """
        获取敏感层的正则表达式模式（用于 NNCF IgnoredScope）
        
        注意：NNCF 使用 ONNX 风格的节点名称，格式如：
        - /conv_stem/Conv
        - /blocks.0/se/fc1/Conv  
        - /classifier/Gemm
        
        设计原则：
        - 使用 .* 进行通配匹配
        - 使用 / 作为路径分隔符 (ONNX 风格)
        - 兼容 EfficientNet, MobileNetV3, WideResNet
        
        Args:
            sensitive_manager: 敏感层管理器
            
        Returns:
            正则表达式模式列表
        """
        # ============================================================
        # ONNX/OpenVINO 风格的节点名称模式
        # 节点名通常是: /module_path/OperationType
        # ============================================================
        base_patterns = [
            # ============================================================
            # 第一层卷积 (对量化最敏感)
            # ============================================================
            r".*/conv_stem/.*",           # EfficientNet, MobileNetV3: /conv_stem/Conv
            r"^/?conv1/.*",               # WideResNet: /conv1/Conv (只匹配根路径，不匹配 /layer1.0/conv1/)
            r".*/stem/.*",                # 通用: /stem/...
            r".*features\.0/.*",          # 一些模型: /features.0/...
            r".*features/0/.*",           # 一些模型: /features/0/...
            
            # ============================================================
            # 最后几层 (分类头)
            # ============================================================
            r".*/conv_head/.*",           # EfficientNet, MobileNetV3: /conv_head/Conv
            r".*/classifier/.*",          # EfficientNet: /classifier/Gemm
            r"^/?fc/.*",                  # WideResNet: /fc/Gemm (只匹配根路径的 fc)
            r".*/head/.*",                # 通用: /head/...
            
            # ============================================================
            # SE 模块 (EfficientNet/MobileNetV3 特有，对量化非常敏感)
            # 
            # ONNX 节点名示例:
            #   /blocks.0/se/fc1/Conv
            #   /blocks.0/se/fc2/Conv
            #   /blocks.0/se/Sigmoid
            #   /blocks.0/se/Mul
            # ============================================================
            r".*/se/.*",                  # 所有 SE 模块内的操作 (关键!)
            r".*/se_module/.*",           # 备选命名
            
            # ============================================================
            # Sigmoid 操作 (SE 模块的核心，输出 [0,1] 对量化敏感)
            # ============================================================
            r".*/Sigmoid.*",              # Sigmoid 操作
            r".*Sigmoid$",                # 以 Sigmoid 结尾的节点
        ]
        
        return base_patterns
    
    def _get_openvino_node_names(self, ov_model) -> List[str]:
        """
        获取 OpenVINO 模型中所有操作节点的名称
        
        Args:
            ov_model: OpenVINO Model 对象
            
        Returns:
            节点名称列表
        """
        node_names = []
        try:
            # 遍历 OpenVINO 模型的操作
            for op in ov_model.get_ordered_ops():
                # 获取节点名称
                friendly_name = op.get_friendly_name()
                if friendly_name:
                    node_names.append(friendly_name)
                
                # 也尝试获取 type_name
                type_name = op.get_type_name()
                if type_name:
                    # 创建 "type:name" 格式的标识符，方便匹配
                    node_names.append(f"{type_name}:{friendly_name}")
                    
        except Exception as e:
            self.logger.debug(f"获取 OpenVINO 节点名称失败: {e}")
            # Fallback: 尝试使用其他方法
            try:
                for node in ov_model.get_ops():
                    name = str(node.get_friendly_name())
                    if name:
                        node_names.append(name)
            except Exception:
                pass
        
        return node_names
    
    def _convert_mixed(
        self,
        onnx_path: str,
        output_path: str,
        input_shape: Tuple[int, ...],
    ) -> ConversionResult:
        """混合精度量化"""
        self.logger.info("Converting to OpenVINO MIXED precision with NNCF...")
        
        start_time = time.time()
        
        # 读取 ONNX 模型
        ov_model = self.core.read_model(onnx_path)
        
        # 初始化敏感层管理器
        sensitive_manager = SensitiveLayerManager(
            use_default=self.config.use_default_sensitive,
            custom_layers=self.config.sensitive_layers,
        )
        
        # 获取敏感层的正则表达式模式（NNCF 支持 types 和 patterns）
        sensitive_patterns = self._get_sensitive_patterns(sensitive_manager)
        
        self.logger.info(f"Sensitive layer patterns: {len(sensitive_patterns)}")
        for pattern in sensitive_patterns[:5]:
            self.logger.info(f"  - {pattern}")
        if len(sensitive_patterns) > 5:
            self.logger.info(f"  ... and {len(sensitive_patterns) - 5} more")
        
        # 创建校准数据管理器
        calib_manager = CalibrationDataManager(
            config=self.config,
            input_shape=input_shape,
        )
        
        # 创建 NNCF 数据集
        calibration_dataset = create_nncf_dataset(
            calib_manager,
            num_samples=self.config.calib_num_samples,
        )
        
        # 创建 IgnoredScope（使用正则表达式模式）
        ignored_scope = None
        if sensitive_patterns:
            try:
                from nncf import IgnoredScope
                # 首先尝试使用 patterns 参数（fnmatch 风格）
                try:
                    ignored_scope = IgnoredScope(patterns=sensitive_patterns)
                    self.logger.info(f"Created IgnoredScope with {len(sensitive_patterns)} patterns")
                except Exception as pattern_error:
                    # 如果 patterns 失败，尝试使用 types（按操作类型忽略）
                    self.logger.warning(f"IgnoredScope patterns failed: {pattern_error}")
                    self.logger.info("Trying type-based IgnoredScope (Convolution layers in head)...")
                    try:
                        # 忽略特定类型的操作（最后的全连接层）
                        ignored_scope = IgnoredScope(
                            types=['MatMul', 'FullyConnected']
                        )
                        self.logger.info("Created type-based IgnoredScope")
                    except Exception as type_error:
                        self.logger.warning(f"Type-based IgnoredScope also failed: {type_error}")
                        ignored_scope = None
            except ImportError:
                self.logger.warning("IgnoredScope not available in this NNCF version")
            except Exception as e:
                self.logger.warning(f"Failed to create IgnoredScope: {e}")
        
        # 执行量化
        try:
            if ignored_scope:
                quantized_model = quantize(
                    model=ov_model,
                    calibration_dataset=calibration_dataset,
                    target_device=TargetDevice.CPU,
                    subset_size=min(self.config.calib_num_samples, len(calib_manager)),
                    ignored_scope=ignored_scope,
                    fast_bias_correction=True,
                )
            else:
                # Fallback: 不使用 ignored_scope
                self.logger.warning("No sensitive layers configured, using full INT8 quantization")
                quantized_model = quantize(
                    model=ov_model,
                    calibration_dataset=calibration_dataset,
                    target_device=TargetDevice.CPU,
                    subset_size=min(self.config.calib_num_samples, len(calib_manager)),
                    fast_bias_correction=True,
                )
        except Exception as e:
            # 如果敏感层匹配失败，尝试不使用 ignored_scope
            error_msg = str(e)
            if "Ignored nodes" in error_msg and "were not found" in error_msg:
                self.logger.warning(f"Sensitive layer matching failed: {e}")
                self.logger.warning("Retrying without ignored_scope (full INT8)...")
                quantized_model = quantize(
                    model=ov_model,
                    calibration_dataset=calibration_dataset,
                    target_device=TargetDevice.CPU,
                    subset_size=min(self.config.calib_num_samples, len(calib_manager)),
                    fast_bias_correction=True,
                )
            else:
                raise
        
        # 准备输出路径
        output_path = self._prepare_output_path(output_path)
        xml_path = str(output_path / "model.xml")
        bin_path = str(output_path / "model.bin")
        
        # 保存量化模型
        serialize(quantized_model, xml_path)
        
        build_time = time.time() - start_time
        
        self.logger.info(f"MIXED model saved to: {xml_path}")
        
        return ConversionResult(
            success=True,
            target_backend=self.config.target_backend,
            precision_mode=PrecisionMode.MIXED,
            output_files=[xml_path, bin_path],
            build_time_seconds=build_time,
            message="OpenVINO MIXED quantization successful",
            stats={
                "original_size_mb": self.get_file_size_mb(onnx_path),
                "output_size_mb": self.get_file_size_mb(str(output_path)),
                "openvino_version": OV_VERSION,
                "nncf_version": NNCF_VERSION,
                "calibration_samples": len(calib_manager),
                "sensitive_patterns_count": len(sensitive_patterns),
            },
        )
    
    def _prepare_output_path(self, output_path: str) -> Path:
        """
        准备输出路径（确保是目录）
        
        OpenVINO 模型由两个文件组成 (.xml + .bin)，所以输出路径应该是目录。
        此方法会处理各种输入格式并返回正确的目录路径。
        
        Args:
            output_path: 输出路径（可以是目录或文件路径）
            
        Returns:
            准备好的目录路径
            
        Examples:
            "/path/to/model.xml" -> "/path/to/"
            "/path/to/output/" -> "/path/to/output/"
            "/path/to/model.onnx" -> "/path/to/"
        """
        output_path = Path(output_path)
        
        # 判断是否是文件路径（有扩展名）
        known_extensions = {".xml", ".bin", ".onnx", ".engine", ".trt"}
        
        if output_path.suffix.lower() in known_extensions:
            # 这是一个文件路径，使用其父目录
            output_dir = output_path.parent
            self.logger.debug(f"输出路径 '{output_path}' 是文件路径，使用父目录 '{output_dir}'")
        elif output_path.suffix:
            # 有其他扩展名，可能是用户误输入，发出警告
            self.logger.warning(
                f"输出路径 '{output_path}' 有未知扩展名 '{output_path.suffix}'，"
                f"将作为目录处理"
            )
            output_dir = output_path
        else:
            # 没有扩展名，假定是目录
            output_dir = output_path
        
        # 确保目录存在
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return output_dir