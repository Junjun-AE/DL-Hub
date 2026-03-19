#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorRT 量化转换器

支持的精度模式:
- FP32: 默认精度构建
- FP16: 半精度优化
- INT8: 静态量化 (需要校准)
- MIXED: 层级精度控制

输出格式: .engine

版本要求: TensorRT >= 8.6

作者: Industrial ML Team
版本: 1.0.0
"""

import os
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any

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

# TensorRT
TRT_AVAILABLE = False
TRT_VERSION = None

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
    TRT_VERSION = trt.__version__
except ImportError:
    pass

# PyCUDA - 使用统一的 cuda_utils 模块
# 注意：不能直接导入 cuda 和 PYCUDA_AVAILABLE 变量，因为它们是延迟初始化的
# 需要通过模块访问或在初始化后获取
import cuda_utils
from cuda_utils import (
    init_pycuda as _init_pycuda,
    cleanup_cuda_context as _cleanup_cuda_context,
)

# cuda 和 PYCUDA_AVAILABLE 需要在 init_pycuda() 调用后才能使用
# 通过函数获取以确保获得正确的引用
def _get_cuda():
    """获取 PyCUDA cuda 模块（在初始化后）"""
    return cuda_utils.cuda

def _is_pycuda_available():
    """检查 PyCUDA 是否可用（在初始化后）"""
    return cuda_utils.PYCUDA_AVAILABLE


logger = Logger.get("TensorRTConverter")


# ============================================================================
# TensorRT 日志器
# ============================================================================

class TRTLogger(trt.ILogger if TRT_AVAILABLE else object):
    """TensorRT 日志器"""
    
    def __init__(self, verbose: bool = False):
        if TRT_AVAILABLE:
            super().__init__()
        self.verbose = verbose
        self.logger = Logger.get("TensorRT")
    
    def log(self, severity, msg):
        if severity == trt.Logger.ERROR:
            self.logger.error(msg)
        elif severity == trt.Logger.WARNING:
            self.logger.warning(msg)
        elif severity == trt.Logger.INFO:
            if self.verbose:
                self.logger.info(msg)
        elif self.verbose:
            self.logger.debug(msg)


# ============================================================================
# TensorRT INT8 校准器
# ============================================================================

class TensorRTCalibrator(trt.IInt8EntropyCalibrator2 if TRT_AVAILABLE else object):
    """
    TensorRT INT8 校准器
    
    继承 IInt8EntropyCalibrator2，实现 Entropy 校准。
    支持校准缓存以加速重复构建。
    """
    
    def __init__(
        self,
        calib_data_manager: CalibrationDataManager,
        input_name: str,
        cache_path: Optional[str] = None,
        algorithm: str = "entropy",
    ):
        """
        初始化校准器
        
        Args:
            calib_data_manager: 校准数据管理器
            input_name: 模型输入名称
            cache_path: 校准缓存路径
            algorithm: 校准算法 ("entropy" 或 "minmax")
        """
        if TRT_AVAILABLE:
            super().__init__()
        
        self.calib_data_manager = calib_data_manager
        self.input_name = input_name
        self.cache_path = cache_path
        self.algorithm = algorithm
        
        self.batch_size = calib_data_manager.batch_size
        self.current_index = 0
        
        # 预加载所有数据
        self.data_batches = list(calib_data_manager.get_batch_iterator())
        self.num_batches = len(self.data_batches)
        
        # GPU 内存分配
        # 重要：需要先获取 CUDA 上下文才能分配内存
        self.device_input = None
        self._cuda_context = None
        
        if self.num_batches > 0:
            try:
                # 使用全局 CUDA 上下文管理器（确保上下文已创建并激活）
                from cuda_utils import get_global_cuda_manager
                cuda_manager = get_global_cuda_manager()
                
                if cuda_manager is not None:
                    self._cuda_context = cuda_manager
                    # 确保上下文在当前线程中处于活动状态
                    cuda_manager.ensure_context_active()
                    
                    first_batch = self.data_batches[0]
                    self.device_input = _get_cuda().mem_alloc(first_batch.nbytes)
                    
                    if self.device_input is None:
                        raise RuntimeError("cuda.mem_alloc returned None")
                        
                    logger.info(f"Calibrator initialized with {self.num_batches} batches")
                else:
                    raise RuntimeError("Failed to get CUDA context manager")
                    
            except Exception as e:
                logger.error(f"Failed to allocate GPU memory for calibration: {e}")
                raise RuntimeError(
                    f"CUDA memory allocation failed: {e}\n"
                    f"Please ensure CUDA is properly installed and GPU is available."
                )
        else:
            logger.warning("No calibration data batches available")
    
    def get_batch_size(self) -> int:
        """返回批量大小"""
        return self.batch_size
    
    def get_batch(self, names: List[str]) -> Optional[List[int]]:
        """
        获取下一批校准数据
        
        Args:
            names: 输入名称列表
            
        Returns:
            GPU 内存指针列表，或 None 表示结束
        """
        if self.current_index >= self.num_batches:
            return None
        
        batch = self.data_batches[self.current_index]
        self.current_index += 1
        
        # 确保数据连续
        batch = np.ascontiguousarray(batch)
        
        # 复制到 GPU
        if _is_pycuda_available():
            _get_cuda().memcpy_htod(self.device_input, batch)
            return [int(self.device_input)]
        else:
            # 如果没有 PyCUDA，返回 None（将导致校准失败）
            logger.warning("PyCUDA not available, calibration may fail")
            return None
    
    def read_calibration_cache(self) -> Optional[bytes]:
        """读取校准缓存"""
        if self.cache_path and os.path.exists(self.cache_path):
            logger.info(f"Loading calibration cache from: {self.cache_path}")
            with open(self.cache_path, "rb") as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache: bytes):
        """写入校准缓存"""
        if self.cache_path:
            logger.info(f"Saving calibration cache to: {self.cache_path}")
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            with open(self.cache_path, "wb") as f:
                f.write(cache)
    
    def cleanup(self) -> None:
        """
        清理 GPU 内存
        
        应在校准完成后调用以释放资源
        """
        if self.device_input is not None:
            try:
                self.device_input.free()
                self.device_input = None
                logger.debug("TensorRT calibrator GPU memory freed")
            except Exception as e:
                logger.warning(f"Failed to free calibrator GPU memory: {e}")
        
        # 清理数据缓存以释放内存
        self.data_batches = []
    
    def __del__(self):
        """析构函数 - 确保 GPU 内存被释放"""
        self.cleanup()
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口 - 自动清理"""
        self.cleanup()
        return False


class TensorRTMinMaxCalibrator(trt.IInt8MinMaxCalibrator if TRT_AVAILABLE else object):
    """
    TensorRT MinMax 校准器
    
    继承 IInt8MinMaxCalibrator，使用 MinMax 校准算法。
    """
    
    def __init__(
        self,
        calib_data_manager: CalibrationDataManager,
        input_name: str,
        cache_path: Optional[str] = None,
    ):
        if TRT_AVAILABLE:
            super().__init__()
        
        self.calib_data_manager = calib_data_manager
        self.input_name = input_name
        self.cache_path = cache_path
        
        self.batch_size = calib_data_manager.batch_size
        self.current_index = 0
        
        # 预加载所有数据
        self.data_batches = list(calib_data_manager.get_batch_iterator())
        self.num_batches = len(self.data_batches)
        
        # GPU 内存分配
        self.device_input = None
        self._cuda_context = None
        
        if self.num_batches > 0:
            try:
                from cuda_utils import get_global_cuda_manager
                cuda_manager = get_global_cuda_manager()
                
                if cuda_manager is not None:
                    self._cuda_context = cuda_manager
                    cuda_manager.ensure_context_active()
                    
                    first_batch = self.data_batches[0]
                    self.device_input = _get_cuda().mem_alloc(first_batch.nbytes)
                    
                    if self.device_input is None:
                        raise RuntimeError("cuda.mem_alloc returned None")
                else:
                    raise RuntimeError("Failed to get CUDA context manager")
                    
            except Exception as e:
                logger.error(f"Failed to allocate GPU memory for MinMax calibration: {e}")
                raise RuntimeError(f"CUDA memory allocation failed: {e}")
    
    def get_batch_size(self) -> int:
        return self.batch_size
    
    def get_batch(self, names: List[str]) -> Optional[List[int]]:
        if self.current_index >= self.num_batches:
            return None
        
        batch = self.data_batches[self.current_index]
        self.current_index += 1
        
        batch = np.ascontiguousarray(batch)
        
        if _is_pycuda_available():  # 这里可以直接检查，因为上面已经初始化过了
            _get_cuda().memcpy_htod(self.device_input, batch)
            return [int(self.device_input)]
        return None
    
    def read_calibration_cache(self) -> Optional[bytes]:
        if self.cache_path and os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache: bytes):
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            with open(self.cache_path, "wb") as f:
                f.write(cache)
    
    def cleanup(self) -> None:
        """清理 GPU 内存"""
        if self.device_input is not None:
            try:
                self.device_input.free()
                self.device_input = None
            except Exception as e:
                logger.warning(f"Failed to free MinMax calibrator GPU memory: {e}")
        self.data_batches = []
    
    def __del__(self):
        """析构函数"""
        self.cleanup()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False


# ============================================================================
# TensorRT 转换器
# ============================================================================

class TensorRTConverter(BaseConverter):
    """
    TensorRT 量化转换器
    
    支持 FP32/FP16/INT8/MIXED 精度模式。
    使用 TensorRT Python API 构建引擎。
    """
    
    # 最低版本要求
    MIN_VERSION = "8.6.0"
    
    def __init__(self, config: ConversionConfig):
        """
        初始化 TensorRT 转换器
        
        Args:
            config: 转换配置
        """
        super().__init__(config)
        
        self.trt_logger = None
        if TRT_AVAILABLE:
            self.trt_logger = TRTLogger(verbose=False)
    
    def check_dependencies(self) -> bool:
        """检查依赖"""
        # 使用公共方法检查 ONNX
        if not self._check_onnx_available():
            return False
        
        if not TRT_AVAILABLE:
            self.logger.error(
                "TensorRT is not installed. "
                "Please install TensorRT >= 8.6 from NVIDIA."
            )
            return False
        
        # 检查版本
        if TRT_VERSION:
            version_parts = TRT_VERSION.split(".")[:3]
            min_parts = self.MIN_VERSION.split(".")
            
            for i in range(min(len(version_parts), len(min_parts))):
                if int(version_parts[i]) < int(min_parts[i]):
                    self.logger.error(
                        f"TensorRT version {TRT_VERSION} is too old. "
                        f"Minimum required: {self.MIN_VERSION}"
                    )
                    return False
                elif int(version_parts[i]) > int(min_parts[i]):
                    break
        
        # INT8 需要 PyCUDA（延迟初始化检查）
        if self.config.precision_mode in (PrecisionMode.INT8, PrecisionMode.MIXED):
            if not _init_pycuda():
                self.logger.error(
                    "PyCUDA is not installed or failed to initialize. Required for INT8 calibration. "
                    "Run: pip install pycuda"
                )
                return False
        
        self.logger.info(f"TensorRT version: {TRT_VERSION}")
        return True
    
    def convert(
        self,
        onnx_path: str,
        output_path: str,
        input_shape: Tuple[int, ...],
        dynamic_axes_spec: Optional[Dict] = None,  # 已弃用，保留仅为向后兼容
    ) -> ConversionResult:
        """
        执行 TensorRT 转换
        
        Args:
            onnx_path: 输入 ONNX 模型路径
            output_path: 输出路径
            input_shape: 输入形状 (B, C, H, W)
            dynamic_axes_spec: [已弃用] 不再使用，动态形状配置直接从 YAML 读取：
                               tensorrt.dynamic_batch 和 tensorrt.dynamic_shapes
            
        Returns:
            转换结果
        """
        precision = self.config.precision_mode
        
        self.logger.info(f"TensorRT Conversion: {precision.value}")
        self.logger.info(f"TensorRT version: {TRT_VERSION}")
        
        try:
            # 加载并预处理 ONNX 模型
            model = onnx.load(onnx_path)
            input_name = model.graph.input[0].name
            
            # 预处理：只做必要的兼容性修复（不修改形状）
            model = self._preprocess_onnx_for_tensorrt(model, input_shape, onnx_path)
            
            # 创建 builder 和 network
            builder = trt.Builder(self.trt_logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, self.trt_logger)
            
            # 解析 ONNX 模型（使用预处理后的模型）
            self.logger.info("Parsing ONNX model...")
            onnx_data = model.SerializeToString()
            if not parser.parse(onnx_data):
                errors = []
                for i in range(parser.num_errors):
                    errors.append(str(parser.get_error(i)))
                
                error_str = "; ".join(errors)
                
                # 检测常见错误并给出建议
                suggestions = []
                
                if "Kernel weight dimension" in error_str or "broadcast" in error_str:
                    suggestions.append("此错误通常由 Depthwise Convolution + 动态形状导致")
                    suggestions.append("建议: 在 config.yaml 中设置 stage4_export.dynamic_batch: false")
                    suggestions.append("或者: 使用 onnx-simplifier 简化模型后重试")
                    
                    if "INT64" in error_str:
                        suggestions.append("模型包含 INT64 权重，TensorRT 会自动转换为 INT32")
                    
                    if "Unsupported" in error_str:
                        suggestions.append("模型包含 TensorRT 不支持的算子")
                        suggestions.append("建议: 尝试使用较低的 opset 版本 (如 opset=13)")
                    
                    if suggestions:
                        self.logger.error("=" * 60)
                        self.logger.error("ONNX 解析失败 - 诊断建议:")
                        for s in suggestions:
                            self.logger.error(f"  • {s}")
                        self.logger.error("=" * 60)
                    
                    raise RuntimeError(f"ONNX parse error: {errors}")
            
            self.logger.info(f"Network inputs: {network.num_inputs}")
            self.logger.info(f"Network outputs: {network.num_outputs}")
            
            # 创建 builder config
            config = builder.create_builder_config()
            
            # 设置 workspace
            workspace_bytes = self.config.trt_workspace_gb * (1 << 30)
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
            
            # 设置 timing cache
            if self.config.trt_timing_cache_path:
                self._load_timing_cache(config)
            
            # 设置优化 profile（从 YAML 配置读取动态形状设置）
            self._set_optimization_profile(
                builder, network, config, input_name, input_shape
            )
            
            # 根据精度模式配置
            if precision == PrecisionMode.FP32:
                pass  # 默认 FP32
                
            elif precision == PrecisionMode.FP16:
                config.set_flag(trt.BuilderFlag.FP16)
                self.logger.info("FP16 mode enabled")
                
            elif precision == PrecisionMode.INT8:
                config.set_flag(trt.BuilderFlag.INT8)
                
                # 创建校准器
                calib_manager = CalibrationDataManager(
                    config=self.config,
                    input_shape=input_shape,
                )
                
                calib_cache_path = self._get_calib_cache_path(output_path)
                
                if self.config.calib_method.value == "minmax":
                    calibrator = TensorRTMinMaxCalibrator(
                        calib_data_manager=calib_manager,
                        input_name=input_name,
                        cache_path=calib_cache_path,
                    )
                else:
                    calibrator = TensorRTCalibrator(
                        calib_data_manager=calib_manager,
                        input_name=input_name,
                        cache_path=calib_cache_path,
                        algorithm=self.config.calib_method.value,
                    )
                
                config.int8_calibrator = calibrator
                self.logger.info(f"INT8 mode enabled with {self.config.calib_method.value} calibrator")
                
            elif precision == PrecisionMode.MIXED:
                # 混合精度：INT8 + 敏感层 FP16
                config.set_flag(trt.BuilderFlag.INT8)
                config.set_flag(trt.BuilderFlag.FP16)
                
                # ============================================================
                # 关键修复：强制 TensorRT 遵守层精度设置
                # 没有这个标志，layer.precision 设置会被忽略！
                # ============================================================
                if hasattr(trt.BuilderFlag, 'OBEY_PRECISION_CONSTRAINTS'):
                    # TensorRT 8.6+
                    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
                    self.logger.info("OBEY_PRECISION_CONSTRAINTS enabled")
                elif hasattr(trt.BuilderFlag, 'STRICT_TYPES'):
                    # TensorRT 8.0-8.5
                    config.set_flag(trt.BuilderFlag.STRICT_TYPES)
                    self.logger.info("STRICT_TYPES enabled")
                
                # 创建校准器
                calib_manager = CalibrationDataManager(
                    config=self.config,
                    input_shape=input_shape,
                )
                
                calib_cache_path = self._get_calib_cache_path(output_path)
                
                calibrator = TensorRTCalibrator(
                    calib_data_manager=calib_manager,
                    input_name=input_name,
                    cache_path=calib_cache_path,
                )
                config.int8_calibrator = calibrator
                
                # 设置敏感层精度
                self._set_layer_precision(network, config)
                self.logger.info("MIXED precision mode enabled")
            
            # 构建引擎
            self.logger.info("Building TensorRT engine...")
            start_time = time.time()
            
            serialized_engine = builder.build_serialized_network(network, config)
            
            if serialized_engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            build_time = time.time() - start_time
            self.logger.info(f"Engine built in {build_time:.2f} seconds")
            
            # 保存引擎
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(serialized_engine)
            
            # 保存 timing cache
            if self.config.trt_timing_cache_path:
                self._save_timing_cache(config)
            
            self.logger.info(f"Engine saved to: {output_path}")
            
            # 收集输出文件
            output_files = [output_path]
            if self.config.trt_timing_cache_path and os.path.exists(self.config.trt_timing_cache_path):
                output_files.append(self.config.trt_timing_cache_path)
            
            calib_cache_path = self._get_calib_cache_path(output_path)
            if os.path.exists(calib_cache_path):
                output_files.append(calib_cache_path)
            
            return ConversionResult(
                success=True,
                target_backend=self.config.target_backend,
                precision_mode=precision,
                output_files=output_files,
                build_time_seconds=build_time,
                message="TensorRT engine built successfully",
                stats={
                    "original_size_mb": self.get_file_size_mb(onnx_path),
                    "output_size_mb": self.get_file_size_mb(output_path),
                    "tensorrt_version": TRT_VERSION,
                    "workspace_gb": self.config.trt_workspace_gb,
                    "build_time_seconds": build_time,
                },
            )
            
        except Exception as e:
            self.logger.error(f"TensorRT conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return ConversionResult(
                success=False,
                target_backend=self.config.target_backend,
                precision_mode=precision,
                output_files=[],
                message=str(e),
            )
        finally:
            # 确保 CUDA 上下文被正确清理，避免 PyCUDA 退出错误
            try:
                _cleanup_cuda_context()
            except Exception:
                pass  # 忽略清理错误
    
    def _preprocess_onnx_for_tensorrt(
        self,
        model: Any,
        input_shape: Tuple[int, ...],
        onnx_path: str,
    ) -> Any:
        """
        预处理 ONNX 模型以确保 TensorRT 兼容性
        
        注意：此方法只做必要的兼容性修复，不修改输入形状！
        动态轴配置由 Stage 4 (model_exporter.py) 正确设置，
        这里不应该覆盖。
        
        主要处理：
        1. 清理重复的 initializers
        2. 修复 Conv 节点的 group 属性 (Depthwise Conv 兼容性)
        
        Args:
            model: ONNX ModelProto
            input_shape: 目标输入形状 (B, C, H, W) - 仅用于 Conv group 修复
            onnx_path: 原始 ONNX 文件路径 (用于错误恢复)
            
        Returns:
            处理后的 ONNX 模型
        """
        self.logger.info("正在检查 ONNX 模型兼容性...")
        
        try:
            # Step 1: 记录当前输入形状（用于调试）
            if model.graph.input:
                graph_input = model.graph.input[0]
                current_shape = []
                for dim in graph_input.type.tensor_type.shape.dim:
                    if dim.HasField('dim_param') and dim.dim_param:
                        current_shape.append(f"'{dim.dim_param}'")
                    elif dim.HasField('dim_value'):
                        current_shape.append(str(dim.dim_value))
                    else:
                        current_shape.append("?")
                self.logger.info(f"ONNX 输入形状: [{', '.join(current_shape)}]")
            
            # Step 2: 清理重复的 initializers
            seen_names = set()
            to_remove = []
            for i, init in enumerate(model.graph.initializer):
                if init.name in seen_names:
                    to_remove.append(i)
                else:
                    seen_names.add(init.name)
            
            for i in reversed(to_remove):
                del model.graph.initializer[i]
            
            if to_remove:
                self.logger.debug(f"清理了 {len(to_remove)} 个重复的 initializers")
            
            # Step 3: 修复 Conv 节点 - 确保 group 属性正确
            model = self._fix_conv_nodes(model, input_shape)
            
            self.logger.info("ONNX 兼容性检查完成")
            return model
            
        except Exception as e:
            self.logger.warning(f"ONNX 兼容性检查失败 (使用原始模型): {e}")
            return onnx.load(onnx_path)
    
    def _fix_conv_nodes(self, model: Any, input_shape: Tuple[int, ...]) -> Any:
        """
        修复 Conv 节点的 group 属性（保守版本）
        
        仅修复明确的 Depthwise Convolution 问题，不对普通卷积进行任何修改。
        
        Depthwise Conv 的特征:
        - 权重形状为 [out_ch, 1, kH, kW]，即 in_channels_per_group == 1
        - group 应该等于 out_channels
        - 常见于 MobileNet、EfficientNet 等轻量级模型
        
        普通卷积的特征:
        - 权重形状为 [out_ch, in_ch, kH, kW]
        - group 通常为 1
        - 不应被修改！（之前的 bug 就是错误修改了普通卷积）
        
        Args:
            model: ONNX ModelProto
            input_shape: 输入形状 (B, C, H, W)
            
        Returns:
            修复后的模型
        """
        from onnx import numpy_helper
        
        # 构建 initializer 名称到数据的映射
        initializers = {}
        for init in model.graph.initializer:
            try:
                initializers[init.name] = numpy_helper.to_array(init)
            except Exception:
                continue
        
        fixed_count = 0
        
        for node in model.graph.node:
            if node.op_type != "Conv":
                continue
            
            # 获取权重
            if len(node.input) < 2:
                continue
            
            weight_name = node.input[1]
            if weight_name not in initializers:
                continue
            
            weight = initializers[weight_name]
            if len(weight.shape) != 4:
                continue
            
            # 权重形状: [out_channels, in_channels/groups, kH, kW]
            out_channels = weight.shape[0]
            in_channels_per_group = weight.shape[1]
            
            # 获取当前 group 属性
            current_group = 1
            group_attr_idx = None
            for i, attr in enumerate(node.attribute):
                if attr.name == "group":
                    current_group = attr.i
                    group_attr_idx = i
                    break
            
            # ============================================================
            # 仅修复明确的 Depthwise Convolution
            # 
            # Depthwise Conv 特征:
            # 1. in_channels_per_group == 1 (每个 group 只有 1 个输入通道)
            # 2. out_channels > 1 (排除单通道输入的特殊情况)
            # 3. current_group != out_channels (group 设置不正确)
            #
            # 注意：不再修改普通卷积！这是之前导致 wide_resnet50_2 出错的原因
            # ============================================================
            if in_channels_per_group == 1 and out_channels > 1 and current_group != out_channels:
                # 额外验证：确保这确实是 Depthwise Conv
                # 对于真正的 Depthwise Conv，current_group 通常是 1（未正确设置）
                # 而 out_channels 应该等于实际的输入通道数
                if current_group == 1:
                    self.logger.debug(
                        f"修复 Depthwise Conv: {node.name}, "
                        f"weight_shape={list(weight.shape)}, "
                        f"group: {current_group} -> {out_channels}"
                    )
                    
                    # 更新或添加 group 属性
                    if group_attr_idx is not None:
                        node.attribute[group_attr_idx].i = out_channels
                    else:
                        group_attr = onnx.helper.make_attribute("group", out_channels)
                        node.attribute.append(group_attr)
                    
                    fixed_count += 1
            
            # ============================================================
            # 不再修改普通卷积的 group 属性！
            # 
            # 之前错误的逻辑（已删除）：
            # elif in_channels_per_group > 1:
            #     expected_group = out_channels // in_channels_per_group
            #     # 这个计算是错误的！对于普通卷积 64->128，会得到 group=2
            #     # 但实际应该保持 group=1
            # ============================================================
        
        if fixed_count > 0:
            self.logger.info(f"修复了 {fixed_count} 个 Depthwise Conv 节点的 group 属性")
        
        return model
    
    def _set_optimization_profile(
        self,
        builder: Any,
        network: Any,
        config: Any,
        input_name: str,
        input_shape: Tuple[int, ...],
    ):
        """
        设置优化 profile
        
        直接从 YAML 配置文件读取动态形状设置：
        - tensorrt.dynamic_batch.enabled/min/opt/max
        - tensorrt.dynamic_shapes.enabled/min/opt/max
        """
        profile = builder.create_optimization_profile()
        
        # 解析输入形状
        B, C, H, W = input_shape
        
        # 从配置文件读取动态 batch 设置
        if self.config.trt_dynamic_batch_enabled:
            min_batch = self.config.trt_min_batch
            opt_batch = self.config.trt_opt_batch
            max_batch = self.config.trt_max_batch
            self.logger.info(f"动态 batch: min={min_batch}, opt={opt_batch}, max={max_batch}")
        else:
            min_batch = opt_batch = max_batch = B
        
        # 从配置文件读取动态尺寸设置
        if self.config.trt_dynamic_shapes_enabled and self.config.trt_min_shapes:
            min_h, min_w = self.config.trt_min_shapes
            opt_h, opt_w = self.config.trt_opt_shapes or (H, W)
            max_h, max_w = self.config.trt_max_shapes or (H, W)
            self.logger.info(f"动态尺寸: min={min_h}x{min_w}, opt={opt_h}x{opt_w}, max={max_h}x{max_w}")
        else:
            min_h = opt_h = max_h = H
            min_w = opt_w = max_w = W
        
        min_shape = (min_batch, C, min_h, min_w)
        opt_shape = (opt_batch, C, opt_h, opt_w)
        max_shape = (max_batch, C, max_h, max_w)
        
        self.logger.info(f"Optimization profile for '{input_name}':")
        self.logger.info(f"  min: {min_shape}")
        self.logger.info(f"  opt: {opt_shape}")
        self.logger.info(f"  max: {max_shape}")
        
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
    
    def _set_layer_precision(self, network: Any, config: Any):
        """
        设置层级精度（MIXED 模式）
        
        策略：
        - 敏感层（首尾层、SE 模块等）：强制 FP16
        - 其他层：由 TensorRT 根据 INT8 校准自动决定
        
        注意：需要配合 OBEY_PRECISION_CONSTRAINTS 标志使用
        """
        # 初始化敏感层管理器
        sensitive_manager = SensitiveLayerManager(
            use_default=self.config.use_default_sensitive,
            custom_layers=self.config.sensitive_layers,
        )
        
        sensitive_count = 0
        sensitive_layers = []
        
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            layer_name = layer.name
            
            if sensitive_manager.is_sensitive(layer_name):
                # 敏感层保持 FP16
                try:
                    layer.precision = trt.float16
                    # 设置所有输出为 FP16
                    for j in range(layer.num_outputs):
                        layer.set_output_type(j, trt.float16)
                    sensitive_count += 1
                    sensitive_layers.append(layer_name)
                except Exception as e:
                    # 某些层可能不支持设置精度
                    self.logger.debug(f"Cannot set precision for layer {layer_name}: {e}")
        
        self.logger.info(f"Set {sensitive_count} sensitive layers to FP16")
        
        # 显示前几个敏感层（调试用）
        if sensitive_layers:
            for layer_name in sensitive_layers[:5]:
                self.logger.debug(f"  FP16: {layer_name}")
            if len(sensitive_layers) > 5:
                self.logger.debug(f"  ... and {len(sensitive_layers) - 5} more")
    
    def _get_calib_cache_path(self, output_path: str) -> str:
        """获取校准缓存路径"""
        if self.config.calib_cache_path:
            return self.config.calib_cache_path
        
        output_path = Path(output_path)
        return str(output_path.parent / f"{output_path.stem}_calib.cache")
    
    def _load_timing_cache(self, config: Any):
        """加载 timing cache"""
        if os.path.exists(self.config.trt_timing_cache_path):
            self.logger.info(f"Loading timing cache from: {self.config.trt_timing_cache_path}")
            with open(self.config.trt_timing_cache_path, "rb") as f:
                cache = config.create_timing_cache(f.read())
                config.set_timing_cache(cache, ignore_mismatch=False)
        else:
            cache = config.create_timing_cache(b"")
            config.set_timing_cache(cache, ignore_mismatch=False)
    
    def _save_timing_cache(self, config: Any):
        """保存 timing cache"""
        cache = config.get_timing_cache()
        if cache:
            self.logger.info(f"Saving timing cache to: {self.config.trt_timing_cache_path}")
            os.makedirs(os.path.dirname(self.config.trt_timing_cache_path) or ".", exist_ok=True)
            with open(self.config.trt_timing_cache_path, "wb") as f:
                f.write(cache.serialize())