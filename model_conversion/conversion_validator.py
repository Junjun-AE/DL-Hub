#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 6: 模型转换验证模块 (Conversion Validation)

本模块提供转换后模型的精度验证和性能测试功能。
支持验证 ONNX、TensorRT、OpenVINO 三种格式的模型。

验证指标:
- 余弦相似度 (Cosine Similarity)
- 最大绝对误差 (Max Absolute Difference)
- 均方误差 (MSE)
- 模型压缩比
- 推理延迟 (可选)
- 吞吐量 (可选)

修复说明 (v2.1.0):
- 添加真实数据验证支持，不再仅使用随机数据
- 支持传入 CalibrationDataManager 进行真实数据验证
- 当无校准数据时发出警告并降级到随机数据

作者: Industrial ML Team
版本: 2.1.0 (修复随机数据验证问题)
"""

import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Iterator
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from unified_logger import Logger, console, Timer


# 导入常量配置
try:
    from constants import VALIDATION, PERFORMANCE
except ImportError:
    # 如果常量模块不存在，使用内联默认值
    class _ValidationDefaults:
        FP16_COSINE_THRESHOLD = 0.999
        INT8_COSINE_THRESHOLD = 0.99
        NUM_TEST_SAMPLES = 10
        RANDOM_SEED = 42
    
    class _PerformanceDefaults:
        WARMUP_ITERATIONS = 10
        TEST_ITERATIONS = 100
    
    VALIDATION = _ValidationDefaults()
    PERFORMANCE = _PerformanceDefaults()

# 条件导入
try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

# TensorRT
TRT_AVAILABLE = False
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    pass

# OpenVINO
OV_AVAILABLE = False
try:
    import openvino as ov
    from openvino.runtime import Core
    OV_AVAILABLE = True
except ImportError:
    pass

# PyCUDA - 使用统一的 cuda_utils 模块
# 注意：不使用 pycuda.autoinit，避免与 cuda_utils 冲突
import cuda_utils
from cuda_utils import init_pycuda as _init_pycuda

def _get_cuda():
    """获取 PyCUDA cuda 模块（在初始化后）"""
    return cuda_utils.cuda

def _is_pycuda_available():
    """检查 PyCUDA 是否可用"""
    return cuda_utils.PYCUDA_AVAILABLE

# 向后兼容的别名
PYCUDA_AVAILABLE = False  # 初始值，实际检查用 _is_pycuda_available()
cuda = None  # 初始值，实际使用用 _get_cuda()

# 尝试初始化 PyCUDA
try:
    if _init_pycuda():
        PYCUDA_AVAILABLE = True
        cuda = _get_cuda()
except Exception:
    pass


# ============================================================================
# 日志配置
# ============================================================================

def get_logger(name: str = "ConversionValidator") -> Logger:
    """获取 logger"""
    return Logger.get(name)


logger = Logger.get("root")


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class ValidationMetrics:
    """
    验证指标
    
    存储单次验证的详细指标。
    """
    cosine_sim: float = 0.0
    max_diff: float = 0.0
    mean_diff: float = 0.0
    mse: float = 0.0
    output_shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)


@dataclass 
class PerformanceMetrics:
    """
    性能指标
    
    存储性能测试的结果。
    """
    latency_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    throughput_fps: float = 0.0
    warmup_runs: int = 10
    test_runs: int = 100


# ============================================================================
# 验证结果类 (与 model_converter.py 中的 ValidationResult 保持兼容)
# ============================================================================

@dataclass
class ValidationResult:
    """
    验证结果
    
    与 model_converter.py 中的 ValidationResult 保持兼容的接口。
    """
    passed: bool
    cosine_sim: float
    max_diff: float
    mse: float
    original_size_mb: float
    output_size_mb: float
    compression_ratio: float
    latency_ms: Optional[float] = None
    throughput_fps: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "passed": bool(self.passed),
            "cosine_sim": float(self.cosine_sim),
            "max_diff": float(self.max_diff),
            "mse": float(self.mse),
            "original_size_mb": float(self.original_size_mb),
            "output_size_mb": float(self.output_size_mb),
            "compression_ratio": float(self.compression_ratio),
            "latency_ms": float(self.latency_ms) if self.latency_ms is not None else None,
            "throughput_fps": float(self.throughput_fps) if self.throughput_fps is not None else None,
            "warnings": list(self.warnings),
        }


# ============================================================================
# 推理运行器
# ============================================================================

class ONNXRunner:
    """ONNX Runtime 推理运行器"""
    
    def __init__(self, model_path: str):
        """
        初始化 ONNX 运行器
        
        Args:
            model_path: ONNX 模型路径
        """
        if not ORT_AVAILABLE:
            raise ImportError("ONNX Runtime is required for ONNX validation")
        
        # 选择执行提供者
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        
        # 获取输入/输出信息
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.input_shapes = {inp.name: inp.shape for inp in self.session.get_inputs()}
        
        logger.debug(f"ONNX runner initialized: inputs={self.input_names}, outputs={self.output_names}")
    
    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        运行推理
        
        Args:
            inputs: 输入数据字典
            
        Returns:
            输出数据字典
        """
        # 准备输入
        feed_dict = {}
        for name in self.input_names:
            if name in inputs:
                feed_dict[name] = inputs[name]
            else:
                # 尝试使用第一个输入
                feed_dict[name] = list(inputs.values())[0]
        
        # 运行推理
        outputs = self.session.run(self.output_names, feed_dict)
        
        return {name: output for name, output in zip(self.output_names, outputs)}
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'session') and self.session is not None:
            del self.session
            self.session = None
    
    def __del__(self):
        """析构函数 - 确保 session 被释放"""
        self.cleanup()


class TensorRTRunner:
    """TensorRT 推理运行器"""
    
    def __init__(self, engine_path: str):
        """
        初始化 TensorRT 运行器
        
        Args:
            engine_path: TensorRT engine 文件路径
        """
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT is required for TensorRT validation")
        if not _is_pycuda_available():
            raise ImportError("PyCUDA is required for TensorRT validation")
        
        # 确保 CUDA 上下文已激活
        cuda_manager = cuda_utils.get_global_cuda_manager()
        if cuda_manager is not None:
            cuda_manager.ensure_context_active()
        
        # 加载 engine
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.trt_logger)
        
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}")
        
        # 创建执行上下文
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")
        
        # 获取绑定信息
        self.num_bindings = self.engine.num_bindings
        self.input_names = []
        self.output_names = []
        self.binding_info = {}
        
        for i in range(self.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = self.engine.get_binding_shape(i)
            is_input = self.engine.binding_is_input(i)
            
            self.binding_info[name] = {
                'index': i,
                'dtype': dtype,
                'shape': shape,
                'is_input': is_input,
                'device_mem': None,
            }
            
            if is_input:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        
        # 创建 CUDA stream
        self.stream = _get_cuda().Stream()
        
        logger.debug(f"TensorRT runner initialized: inputs={self.input_names}, outputs={self.output_names}")
    
    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        运行推理
        
        Args:
            inputs: 输入数据字典
            
        Returns:
            输出数据字典
        """
        bindings = [None] * self.num_bindings
        
        # 处理输入
        input_data = None
        for name in self.input_names:
            if name in inputs:
                input_data = inputs[name]
            else:
                input_data = list(inputs.values())[0]
            
            info = self.binding_info[name]
            input_data = np.ascontiguousarray(input_data.astype(info['dtype']))
            
            # 设置动态形状
            self.context.set_binding_shape(info['index'], input_data.shape)
            
            # 分配 GPU 内存
            device_mem = _get_cuda().mem_alloc(input_data.nbytes)
            _get_cuda().memcpy_htod(device_mem, input_data)
            bindings[info['index']] = int(device_mem)
            info['device_mem'] = device_mem
        
        # 处理输出
        outputs = {}
        for name in self.output_names:
            info = self.binding_info[name]
            
            # 获取实际输出形状
            output_shape = self.context.get_binding_shape(info['index'])
            output_size = int(np.prod(output_shape))
            dtype_size = np.dtype(info['dtype']).itemsize
            
            # 分配内存
            device_mem = _get_cuda().mem_alloc(output_size * dtype_size)
            bindings[info['index']] = int(device_mem)
            info['device_mem'] = device_mem
            info['output_shape'] = output_shape
        
        # 执行推理
        success = self.context.execute_v2(bindings)
        if not success:
            raise RuntimeError("TensorRT execution failed")
        
        # 拷贝输出
        for name in self.output_names:
            info = self.binding_info[name]
            output_array = np.empty(info['output_shape'], dtype=info['dtype'])
            _get_cuda().memcpy_dtoh(output_array, info['device_mem'])
            outputs[name] = output_array
            
            # 释放 GPU 内存
            info['device_mem'].free()
            info['device_mem'] = None
        
        # 释放输入内存
        for name in self.input_names:
            info = self.binding_info[name]
            if info['device_mem'] is not None:
                info['device_mem'].free()
                info['device_mem'] = None
        
        return outputs
    
    def cleanup(self):
        """清理资源"""
        for info in self.binding_info.values():
            if info['device_mem'] is not None:
                try:
                    info['device_mem'].free()
                except Exception:
                    pass
        
        del self.context
        del self.engine
        del self.runtime


class OpenVINORunner:
    """OpenVINO 推理运行器"""
    
    def __init__(self, model_path: str):
        """
        初始化 OpenVINO 运行器
        
        Args:
            model_path: OpenVINO 模型路径 (.xml 文件)
        """
        if not OV_AVAILABLE:
            raise ImportError("OpenVINO is required for OpenVINO validation")
        
        self.core = Core()
        
        # 加载模型
        model = self.core.read_model(model=str(model_path))
        self.compiled_model = self.core.compile_model(model, "CPU")
        
        # 获取输入/输出信息
        self.input_names = [inp.any_name for inp in model.inputs]
        self.output_names = [out.any_name for out in model.outputs]
        
        logger.debug(f"OpenVINO runner initialized: inputs={self.input_names}, outputs={self.output_names}")
    
    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        运行推理
        
        Args:
            inputs: 输入数据字典
            
        Returns:
            输出数据字典
        """
        # 准备输入
        input_data = {}
        for name in self.input_names:
            if name in inputs:
                input_data[name] = inputs[name]
            else:
                # 使用第一个输入
                input_data[name] = list(inputs.values())[0]
        
        # 运行推理
        results = self.compiled_model(input_data)
        
        # 转换输出
        outputs = {}
        for i, name in enumerate(self.output_names):
            outputs[name] = results[i]
        
        return outputs
    
    def cleanup(self):
        """清理资源"""
        del self.compiled_model
        del self.core


# ============================================================================
# 转换验证器
# ============================================================================

class ConversionValidator:
    """
    转换验证器
    
    验证转换后模型的精度和性能。
    
    使用方法:
    ---------
    >>> validator = ConversionValidator(enable_perf_test=True)
    >>> result = validator.validate(
    ...     original_onnx_path="model.onnx",
    ...     converted_path="model.engine",
    ...     target_backend=TargetBackend.TENSORRT,
    ...     precision_mode=PrecisionMode.FP16,
    ...     input_shape=(1, 3, 224, 224),
    ... )
    >>> print(f"Passed: {result.passed}, Cosine: {result.cosine_sim}")
    """
    
    def __init__(
        self,
        enable_perf_test: bool = False,
        cosine_threshold_fp16: float = 0.999,
        cosine_threshold_int8: float = 0.99,
        warmup_runs: int = 10,
        test_runs: int = 100,
    ):
        """
        初始化验证器
        
        Args:
            enable_perf_test: 是否启用性能测试
            cosine_threshold_fp16: FP16 余弦相似度阈值
            cosine_threshold_int8: INT8 余弦相似度阈值
            warmup_runs: 预热运行次数
            test_runs: 测试运行次数
        """
        self.enable_perf_test = enable_perf_test
        self.cosine_threshold_fp16 = cosine_threshold_fp16
        self.cosine_threshold_int8 = cosine_threshold_int8
        self.warmup_runs = warmup_runs
        self.test_runs = test_runs
        
        self.logger = Logger.get("ConversionValidator")
        
        # 运行器缓存
        self._onnx_runner: Optional[ONNXRunner] = None
        self._converted_runner: Optional[Any] = None
    
    def validate(
        self,
        original_onnx_path: str,
        converted_path: str,
        target_backend: Any,  # TargetBackend enum
        precision_mode: Any,  # PrecisionMode enum
        input_shape: Tuple[int, ...],
        num_samples: int = 10,
        calib_data_manager: Optional[Any] = None,  # CalibrationDataManager
        validation_data_iterator: Optional[Iterator[np.ndarray]] = None,
    ) -> ValidationResult:
        """
        验证转换后的模型
        
        Args:
            original_onnx_path: 原始 ONNX 模型路径
            converted_path: 转换后模型路径
            target_backend: 目标后端枚举
            precision_mode: 精度模式枚举
            input_shape: 输入形状
            num_samples: 验证样本数
            calib_data_manager: 校准数据管理器（推荐，使用真实数据验证）
            validation_data_iterator: 自定义验证数据迭代器（可选）
            
        Returns:
            验证结果
            
        Note:
            强烈建议提供 calib_data_manager 或 validation_data_iterator，
            使用真实数据进行验证。如果未提供，将降级使用随机数据并发出警告。
        """
        validation_warnings = []
        
        try:
            # 检查精度模式
            if hasattr(precision_mode, 'value'):
                prec_value = precision_mode.value
            else:
                prec_value = str(precision_mode).lower()
            
            # INT8 和 MIXED 精度模式强制要求提供校准数据
            if prec_value in ('int8', 'mixed'):
                if calib_data_manager is None and validation_data_iterator is None:
                    raise ValueError(
                        f"INT8/MIXED 精度模式 ({prec_value}) 必须提供校准数据进行验证。"
                        f"请通过 calib_data_manager 或 validation_data_iterator 参数提供真实数据。"
                        f"使用随机数据进行 INT8 量化验证可能导致严重的精度问题无法被检测到。"
                    )
            
            # 获取文件大小
            original_size_mb = self._get_file_size_mb(original_onnx_path)
            output_size_mb = self._get_file_size_mb(converted_path)
            compression_ratio = original_size_mb / output_size_mb if output_size_mb > 0 else 1.0
            
            # 创建运行器
            self.logger.info(f"Creating ONNX runner for: {original_onnx_path}")
            onnx_runner = ONNXRunner(original_onnx_path)
            
            self.logger.info(f"Creating {target_backend.value} runner for: {converted_path}")
            converted_runner = self._create_runner(converted_path, target_backend)
            
            # 运行验证
            metrics = self._run_validation(
                onnx_runner=onnx_runner,
                converted_runner=converted_runner,
                input_shape=input_shape,
                num_samples=num_samples,
                calib_data_manager=calib_data_manager,
                validation_data_iterator=validation_data_iterator,
            )
            
            # 确定阈值（使用已解析的 prec_value）
            if prec_value in ('int8', 'mixed'):
                threshold = self.cosine_threshold_int8
            else:
                threshold = self.cosine_threshold_fp16
            
            # 判断是否通过
            passed = metrics.cosine_sim >= threshold
            
            if not passed:
                validation_warnings.append(f"Cosine similarity {metrics.cosine_sim:.6f} below threshold {threshold}")
            
            # 性能测试
            latency_ms = None
            throughput_fps = None
            
            if self.enable_perf_test:
                perf_metrics = self._run_performance_test(
                    runner=converted_runner,
                    input_shape=input_shape,
                )
                latency_ms = perf_metrics.latency_ms
                throughput_fps = perf_metrics.throughput_fps
            
            # 清理
            onnx_runner.cleanup()
            converted_runner.cleanup()
            
            return ValidationResult(
                passed=passed,
                cosine_sim=metrics.cosine_sim,
                max_diff=metrics.max_diff,
                mse=metrics.mse,
                original_size_mb=original_size_mb,
                output_size_mb=output_size_mb,
                compression_ratio=compression_ratio,
                latency_ms=latency_ms,
                throughput_fps=throughput_fps,
                warnings=validation_warnings,
            )
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            import traceback
            traceback.print_exc()
            
            return ValidationResult(
                passed=False,
                cosine_sim=0.0,
                max_diff=float('inf'),
                mse=float('inf'),
                original_size_mb=self._get_file_size_mb(original_onnx_path),
                output_size_mb=self._get_file_size_mb(converted_path),
                compression_ratio=1.0,
                warnings=[str(e)],
            )
    
    def _create_runner(self, model_path: str, target_backend: Any) -> Any:
        """创建推理运行器"""
        backend_value = target_backend.value if hasattr(target_backend, 'value') else str(target_backend)
        
        if backend_value in ('ort', 'onnxruntime'):
            return ONNXRunner(model_path)
        elif backend_value in ('tensorrt', 'trt'):
            return TensorRTRunner(model_path)
        elif backend_value in ('openvino', 'ov'):
            return OpenVINORunner(model_path)
        else:
            raise ValueError(f"Unknown backend: {target_backend}")
    
    def _run_validation(
        self,
        onnx_runner: ONNXRunner,
        converted_runner: Any,
        input_shape: Tuple[int, ...],
        num_samples: int,
        calib_data_manager: Optional[Any] = None,
        validation_data_iterator: Optional[Iterator[np.ndarray]] = None,
    ) -> ValidationMetrics:
        """
        运行验证测试
        
        Args:
            onnx_runner: ONNX 运行器
            converted_runner: 转换后模型运行器
            input_shape: 输入形状
            num_samples: 验证样本数
            calib_data_manager: 校准数据管理器（推荐）
            validation_data_iterator: 自定义验证数据迭代器
            
        Returns:
            验证指标
            
        Note:
            数据源优先级: validation_data_iterator > calib_data_manager > 随机数据
        """
        cosine_sims = []
        max_diffs = []
        mses = []
        
        # 确定数据源
        data_source = "random"
        data_iterator = None
        
        if validation_data_iterator is not None:
            data_iterator = validation_data_iterator
            data_source = "custom"
            self.logger.info("使用自定义验证数据迭代器进行验证")
        elif calib_data_manager is not None:
            # 使用校准数据管理器的单样本迭代器
            data_iterator = calib_data_manager.get_single_iterator()
            data_source = "calibration"
            self.logger.info("使用校准数据进行验证（推荐）")
        else:
            # 降级到随机数据
            # 对于 FP16/FP32，随机数据验证通常足够可靠，使用 INFO 级别
            # 对于 INT8/MIXED，这个分支不应该被执行（上层已强制要求校准数据）
            self.logger.info(
                "使用随机数据进行验证。"
                "对于 FP16/FP32 转换，随机数据验证通常足够可靠。"
            )
        
        # 设置随机种子以确保可复现
        np.random.seed(VALIDATION.RANDOM_SEED)
        
        samples_processed = 0
        
        for i in range(num_samples):
            # 获取输入数据
            if data_iterator is not None:
                try:
                    input_data = next(data_iterator)
                    # 确保形状正确
                    if input_data.ndim == 4 and input_data.shape[0] == 1:
                        input_data = input_data.astype(np.float32)
                    else:
                        # 添加 batch 维度
                        input_data = input_data[np.newaxis, ...].astype(np.float32)
                except StopIteration:
                    self.logger.warning(f"真实数据不足，已处理 {samples_processed} 个样本")
                    break
            else:
                # 使用随机数据，但使用合理的范围 [0, 1]
                # 而不是标准正态分布，这更接近实际归一化后的图像
                input_data = np.random.rand(*input_shape).astype(np.float32)
            
            inputs = {'input': input_data}
            
            try:
                # ONNX 推理
                onnx_outputs = onnx_runner.run(inputs)
                
                # 转换模型推理
                converted_outputs = converted_runner.run(inputs)
                
                # 比较输出
                for name, onnx_out in onnx_outputs.items():
                    # 找到对应的转换模型输出
                    if name in converted_outputs:
                        converted_out = converted_outputs[name]
                    else:
                        # 使用第一个输出
                        converted_out = list(converted_outputs.values())[0]
                    
                    # 展平数组
                    onnx_flat = onnx_out.flatten().astype(np.float64)
                    converted_flat = converted_out.flatten().astype(np.float64)
                    
                    # 确保长度一致
                    min_len = min(len(onnx_flat), len(converted_flat))
                    onnx_flat = onnx_flat[:min_len]
                    converted_flat = converted_flat[:min_len]
                    
                    # 计算指标
                    cosine_sim = self._cosine_similarity(onnx_flat, converted_flat)
                    max_diff = np.max(np.abs(onnx_flat - converted_flat))
                    mse = np.mean((onnx_flat - converted_flat) ** 2)
                    
                    cosine_sims.append(cosine_sim)
                    max_diffs.append(max_diff)
                    mses.append(mse)
                    
                    break  # 只比较第一个输出
                    
                samples_processed += 1
                
            except Exception as e:
                self.logger.warning(f"验证样本 {i} 失败: {e}")
                continue
        
        if samples_processed == 0:
            self.logger.error("没有成功验证任何样本")
            return ValidationMetrics(
                cosine_sim=0.0,
                max_diff=float('inf'),
                mean_diff=float('inf'),
                mse=float('inf'),
            )
        
        self.logger.info(
            f"验证完成: {samples_processed}/{num_samples} 样本, "
            f"数据源: {data_source}"
        )
        
        return ValidationMetrics(
            cosine_sim=float(np.mean(cosine_sims)),
            max_diff=float(np.max(max_diffs)),
            mean_diff=float(np.mean(max_diffs)),
            mse=float(np.mean(mses)),
        )
    
    def _run_performance_test(
        self,
        runner: Any,
        input_shape: Tuple[int, ...],
        sample_input: Optional[np.ndarray] = None,
    ) -> PerformanceMetrics:
        """
        运行性能测试
        
        Args:
            runner: 模型运行器
            input_shape: 输入形状
            sample_input: 真实样本输入（可选，推荐提供）
            
        Returns:
            性能指标
        """
        # 确定输入数据
        if sample_input is not None:
            input_data = sample_input.astype(np.float32)
            self.logger.debug("使用真实样本进行性能测试")
        else:
            # 使用 [0, 1] 范围的随机数据，更接近归一化后的图像
            input_data = np.random.rand(*input_shape).astype(np.float32)
            self.logger.debug("使用随机数据进行性能测试")
        
        inputs = {'input': input_data}
        
        # 预热
        for _ in range(self.warmup_runs):
            runner.run(inputs)
        
        # 测试
        latencies = []
        for _ in range(self.test_runs):
            start_time = time.perf_counter()
            runner.run(inputs)
            latencies.append((time.perf_counter() - start_time) * 1000)
        
        latencies = np.array(latencies)
        
        return PerformanceMetrics(
            latency_ms=float(np.mean(latencies)),
            latency_p50_ms=float(np.percentile(latencies, 50)),
            latency_p95_ms=float(np.percentile(latencies, 95)),
            latency_p99_ms=float(np.percentile(latencies, 99)),
            throughput_fps=float(1000.0 / np.mean(latencies)),
            warmup_runs=self.warmup_runs,
            test_runs=self.test_runs,
        )
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def _get_file_size_mb(self, path: str) -> float:
        """
        获取文件或目录大小（MB）
        
        对于 OpenVINO 模型，会自动包含 .xml 和 .bin 文件。
        对于 TensorRT，会计算 .engine 文件大小。
        对于目录，会计算目录内所有文件的总大小。
        """
        path = Path(path)
        
        if path.is_file():
            total_size = path.stat().st_size
            
            # OpenVINO 特殊处理: .xml 文件需要加上对应的 .bin 文件
            if path.suffix.lower() == '.xml':
                bin_path = path.with_suffix('.bin')
                if bin_path.exists():
                    total_size += bin_path.stat().st_size
                    self.logger.debug(f"OpenVINO model: {path.name} + {bin_path.name}")
            
            return total_size / (1024 * 1024)
            
        elif path.is_dir():
            total_size = sum(f.stat().st_size for f in path.glob('**/*') if f.is_file())
            return total_size / (1024 * 1024)
        else:
            return 0.0


# ============================================================================
# 便捷函数
# ============================================================================

def validate_conversion(
    original_onnx_path: str,
    converted_path: str,
    target_backend: str,
    precision: str = "fp16",
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    enable_perf_test: bool = False,
) -> ValidationResult:
    """
    便捷的转换验证函数
    
    Args:
        original_onnx_path: 原始 ONNX 模型路径
        converted_path: 转换后模型路径
        target_backend: 目标后端 (ort/tensorrt/openvino)
        precision: 精度模式 (fp32/fp16/int8)
        input_shape: 输入形状
        enable_perf_test: 是否启用性能测试
        
    Returns:
        验证结果
    """
    # 创建简单的枚举类模拟
    class SimpleBackend:
        def __init__(self, value):
            self.value = value
    
    class SimplePrecision:
        def __init__(self, value):
            self.value = value
    
    validator = ConversionValidator(enable_perf_test=enable_perf_test)
    
    return validator.validate(
        original_onnx_path=original_onnx_path,
        converted_path=converted_path,
        target_backend=SimpleBackend(target_backend.lower()),
        precision_mode=SimplePrecision(precision.lower()),
        input_shape=input_shape,
    )


# ============================================================================
# 测试
# ============================================================================

def _test_validator():
    """测试验证器模块"""
    print("=" * 60)
    print("Testing ConversionValidator module")
    print("=" * 60)
    
    # 测试初始化
    print("\n1. Testing initialization...")
    validator = ConversionValidator(enable_perf_test=False)
    print("   ✅ ConversionValidator initialized successfully")
    
    # 测试 ValidationResult
    print("\n2. Testing ValidationResult...")
    result = ValidationResult(
        passed=True,
        cosine_sim=0.9999,
        max_diff=0.001,
        mse=0.0001,
        original_size_mb=100.0,
        output_size_mb=50.0,
        compression_ratio=2.0,
    )
    print(f"   Result: {result.to_dict()}")
    print("   ✅ ValidationResult works correctly")
    
    print("\n" + "=" * 60)
    print("ConversionValidator tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    _test_validator()