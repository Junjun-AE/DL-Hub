#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX Runtime 量化转换器

支持的精度模式:
- FP32: 直接复制
- FP16: float16 转换
- INT8: 静态量化 (QDQ 格式)
- MIXED: 指定层排除量化

输出格式: .onnx

作者: Industrial ML Team
版本: 1.0.0
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Tuple, Optional, Dict, List

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
)

# 条件导入
try:
    import onnx
    from onnx import numpy_helper
    ONNX_AVAILABLE = True
    
    # ========== ONNX 版本兼容性补丁 ==========
    # ONNX >= 1.16 移除了 onnx.mapping，但旧版 onnxruntime 仍需要它
    # 这里手动恢复 mapping 模块以保持兼容性
    if not hasattr(onnx, 'mapping'):
        # 创建兼容性映射
        class ONNXMapping:
            """ONNX mapping 兼容性模块"""
            # TensorProto 数据类型到 numpy 类型的映射
            TENSOR_TYPE_TO_NP_TYPE = {
                int(onnx.TensorProto.FLOAT): np.float32,
                int(onnx.TensorProto.UINT8): np.uint8,
                int(onnx.TensorProto.INT8): np.int8,
                int(onnx.TensorProto.UINT16): np.uint16,
                int(onnx.TensorProto.INT16): np.int16,
                int(onnx.TensorProto.INT32): np.int32,
                int(onnx.TensorProto.INT64): np.int64,
                int(onnx.TensorProto.BOOL): np.bool_,
                int(onnx.TensorProto.FLOAT16): np.float16,
                int(onnx.TensorProto.DOUBLE): np.float64,
                int(onnx.TensorProto.COMPLEX64): np.complex64,
                int(onnx.TensorProto.COMPLEX128): np.complex128,
                int(onnx.TensorProto.UINT32): np.uint32,
                int(onnx.TensorProto.UINT64): np.uint64,
                int(onnx.TensorProto.STRING): np.object_,
            }
            
            # 尝试添加 BFLOAT16 和 FLOAT8 类型（如果存在）
            if hasattr(onnx.TensorProto, 'BFLOAT16'):
                # bfloat16 在 numpy 中没有原生支持，使用 uint16 存储
                TENSOR_TYPE_TO_NP_TYPE[int(onnx.TensorProto.BFLOAT16)] = np.uint16
            
            # numpy 类型到 TensorProto 数据类型的映射
            NP_TYPE_TO_TENSOR_TYPE = {v: k for k, v in TENSOR_TYPE_TO_NP_TYPE.items()}
        
        # 将兼容模块注入到 onnx
        onnx.mapping = ONNXMapping()
        
except ImportError:
    ONNX_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

# ONNX Runtime 量化工具
try:
    from onnxruntime.quantization import (
        quantize_static,
        quantize_dynamic,
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        CalibrationMethod,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process
    ORT_QUANT_AVAILABLE = True
except ImportError:
    ORT_QUANT_AVAILABLE = False


logger = Logger.get("ORTConverter")


# ============================================================================
# ORT 校准数据读取器
# ============================================================================

class ORTCalibrationDataReader(CalibrationDataReader):
    """
    ONNX Runtime 校准数据读取器
    
    实现 CalibrationDataReader 接口，用于 quantize_static。
    """
    
    def __init__(
        self,
        calib_data_manager: CalibrationDataManager,
        input_name: str,
    ):
        """
        初始化校准数据读取器
        
        Args:
            calib_data_manager: 校准数据管理器
            input_name: 模型输入名称
        """
        self.calib_data_manager = calib_data_manager
        self.input_name = input_name
        self.data_iterator = None
        self.current_index = 0
        
        # 预加载所有数据
        self.data_list = []
        for batch in calib_data_manager.get_single_iterator():
            self.data_list.append(batch)
    
    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        """获取下一个校准样本"""
        if self.current_index >= len(self.data_list):
            return None
        
        data = self.data_list[self.current_index]
        self.current_index += 1
        
        return {self.input_name: data}
    
    def rewind(self):
        """重置迭代器"""
        self.current_index = 0


# ============================================================================
# ONNX Runtime 转换器
# ============================================================================

class ORTConverter(BaseConverter):
    """
    ONNX Runtime 量化转换器
    
    支持 FP32/FP16/INT8/MIXED 精度模式。
    使用 QDQ (Quantize-Dequantize) 格式进行 INT8 量化。
    """
    
    def __init__(self, config: ConversionConfig):
        """
        初始化 ORT 转换器
        
        Args:
            config: 转换配置
        """
        super().__init__(config)
        
        # 校准方法映射
        self.calib_method_map = {
            "entropy": CalibrationMethod.Entropy if ORT_QUANT_AVAILABLE else None,
            "minmax": CalibrationMethod.MinMax if ORT_QUANT_AVAILABLE else None,
        }
    
    def check_dependencies(self) -> bool:
        """检查依赖"""
        # 使用公共方法检查 ONNX
        if not self._check_onnx_available():
            return False
        
        if not ORT_AVAILABLE:
            self.logger.error("ONNX Runtime is not installed. Run: pip install onnxruntime")
            return False
        
        if self.config.precision_mode in (PrecisionMode.INT8, PrecisionMode.MIXED):
            if not ORT_QUANT_AVAILABLE:
                self.logger.error(
                    "ONNX Runtime quantization is not available. "
                    "Run: pip install onnxruntime>=1.14.0"
                )
                return False
        
        return True
    
    def convert(
        self,
        onnx_path: str,
        output_path: str,
        input_shape: Tuple[int, ...],
        dynamic_axes_spec: Optional[Dict] = None,
    ) -> ConversionResult:
        """
        执行 ORT 量化转换
        
        Args:
            onnx_path: 输入 ONNX 模型路径
            output_path: 输出路径
            input_shape: 输入形状 (B, C, H, W)
            dynamic_axes_spec: 动态轴规格
            
        Returns:
            转换结果
        """
        precision = self.config.precision_mode
        
        self.logger.info(f"ORT Conversion: {precision.value}")
        
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
            self.logger.error(f"ORT conversion failed: {e}")
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
        """FP32 转换（直接复制）"""
        self.logger.info("FP32 mode: copying original ONNX model")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        # 复制文件
        shutil.copy2(onnx_path, output_path)
        
        return ConversionResult(
            success=True,
            target_backend=self.config.target_backend,
            precision_mode=PrecisionMode.FP32,
            output_files=[output_path],
            message="FP32 model copied successfully",
            stats={
                "original_size_mb": self.get_file_size_mb(onnx_path),
                "output_size_mb": self.get_file_size_mb(output_path),
            },
        )
    
    def _convert_fp16(self, onnx_path: str, output_path: str) -> ConversionResult:
        """FP16 转换 - 修复版本
        
        修复内容：
        1. 默认优先使用手动 FP16 转换方法（更稳定，避免 onnxconverter_common 崩溃）
        2. 添加完整的异常处理
        3. 添加调试输出以便排查问题
        """
        import sys
        self.logger.info("Converting to FP16...")
        print("[DEBUG] _convert_fp16 开始...", flush=True)
        
        # ============ 关键修复：优先使用手动方法（更稳定） ============
        # onnxconverter_common 的 float16.convert_float_to_float16() 可能导致进程崩溃
        # 手动方法虽然功能简单，但非常稳定
        use_manual_method = True  # 默认使用手动方法
        
        if use_manual_method:
            print("[DEBUG] 使用手动 FP16 转换方法（更稳定）...", flush=True)
            try:
                result = self._convert_fp16_manual(onnx_path, output_path)
                print(f"[DEBUG] 手动 FP16 转换完成: success={result.success}", flush=True)
                return result
            except Exception as e:
                import traceback
                self.logger.error(f"手动 FP16 转换失败: {e}")
                print(f"[DEBUG] 手动 FP16 转换失败: {e}", flush=True)
                print(traceback.format_exc(), flush=True)
                return ConversionResult(
                    success=False,
                    target_backend=self.config.target_backend,
                    precision_mode=PrecisionMode.FP16,
                    output_files=[],
                    message=f"FP16 转换失败: {e}",
                )
        
        # ============ 以下是使用 onnxconverter_common 的备选方法 ============
        try:
            print("[DEBUG] 尝试导入 onnxconverter_common.float16...", flush=True)
            from onnxconverter_common import float16
            print("[DEBUG] 导入成功", flush=True)
        except ImportError as e:
            print(f"[DEBUG] 导入失败: {e}，使用手动方法", flush=True)
            return self._convert_fp16_manual(onnx_path, output_path)
        
        try:
            # 加载模型
            print("[DEBUG] 加载 ONNX 模型...", flush=True)
            model = onnx.load(onnx_path)
            print(f"[DEBUG] 模型加载完成，节点数: {len(model.graph.node)}", flush=True)
            
            # 检查并记录原始模型的动态轴信息
            dynamic_info = self._get_dynamic_axes_info(model)
            if dynamic_info['has_dynamic']:
                self.logger.info(f"检测到动态轴: {dynamic_info['details']}")
            
            # 需要保持 FP32 的算子
            keep_fp32_ops = [
                "Softmax",
                "LayerNormalization", 
                "InstanceNormalization",
                "Exp",
                "Log",
                "Pow",
                "ReduceMean",
                "ReduceSum",
            ]
            
            # 转换为 FP16
            print("[DEBUG] 调用 float16.convert_float_to_float16()...", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            
            model_fp16 = float16.convert_float_to_float16(
                model,
                keep_io_types=True,
                op_block_list=keep_fp32_ops,
            )
            
            print("[DEBUG] FP16 转换完成", flush=True)
            
            # 设置动态batch（如果配置启用）
            if getattr(self.config, 'ort_dynamic_batch_enabled', False) or \
               getattr(self.config, 'trt_dynamic_batch_enabled', False):
                model_fp16 = self._set_dynamic_batch(model_fp16)
                self.logger.info("已为 ONNX Runtime 设置动态 batch")
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            
            # 保存模型
            print(f"[DEBUG] 保存 FP16 模型到: {output_path}", flush=True)
            onnx.save(model_fp16, output_path)
            
            # 验证保存后的模型动态轴
            saved_dynamic_info = self._get_dynamic_axes_info(onnx.load(output_path))
            
            self.logger.info(f"FP16 model saved to: {output_path}")
            if saved_dynamic_info['has_dynamic']:
                self.logger.info(f"保留的动态轴: {saved_dynamic_info['details']}")
            
            return ConversionResult(
                success=True,
                target_backend=self.config.target_backend,
                precision_mode=PrecisionMode.FP16,
                output_files=[output_path],
                message="FP16 conversion successful",
                stats={
                    "original_size_mb": self.get_file_size_mb(onnx_path),
                    "output_size_mb": self.get_file_size_mb(output_path),
                    "keep_fp32_ops": keep_fp32_ops,
                    "dynamic_batch_enabled": saved_dynamic_info['has_dynamic'],
                },
            )
            
        except Exception as e:
            import traceback
            self.logger.error(f"onnxconverter_common FP16 转换失败: {e}")
            print(f"[DEBUG] onnxconverter_common 失败: {e}", flush=True)
            print(traceback.format_exc(), flush=True)
            
            # 回退到手动方法
            print("[DEBUG] 回退到手动 FP16 转换...", flush=True)
            try:
                return self._convert_fp16_manual(onnx_path, output_path)
            except Exception as e2:
                self.logger.error(f"手动 FP16 转换也失败: {e2}")
                return ConversionResult(
                    success=False,
                    target_backend=self.config.target_backend,
                    precision_mode=PrecisionMode.FP16,
                    output_files=[],
                    message=f"FP16 转换失败: {e2}",
                )
    
    def _get_dynamic_axes_info(self, model) -> dict:
        """获取模型的动态轴信息"""
        info = {'has_dynamic': False, 'details': []}
        
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
        
        return info
    
    def _set_dynamic_batch(self, model):
        """为ONNX模型设置动态batch维度"""
        # 设置输入的batch维度为动态
        for inp in model.graph.input:
            if inp.type.tensor_type.shape.dim:
                dim = inp.type.tensor_type.shape.dim[0]
                dim.ClearField('dim_value')
                dim.dim_param = 'batch'
        
        # 设置输出的batch维度为动态
        for out in model.graph.output:
            if out.type.tensor_type.shape.dim:
                dim = out.type.tensor_type.shape.dim[0]
                dim.ClearField('dim_value')
                dim.dim_param = 'batch'
        
        return model
    
    def _convert_fp16_manual(self, onnx_path: str, output_path: str) -> ConversionResult:
        """手动 FP16 转换（不依赖 onnxconverter-common）"""
        self.logger.info("Using manual FP16 conversion...")
        
        model = onnx.load(onnx_path)
        
        # 收集需要转换的 initializer（避免迭代中修改列表）
        new_initializers = []
        indices_to_remove = []
        
        for idx, initializer in enumerate(model.graph.initializer):
            if initializer.data_type == onnx.TensorProto.FLOAT:
                # 转换为 numpy
                np_data = numpy_helper.to_array(initializer)
                
                # 转换为 float16
                np_data_fp16 = np_data.astype(np.float16)
                
                # 创建新 initializer
                new_initializer = numpy_helper.from_array(
                    np_data_fp16,
                    name=initializer.name,
                )
                
                new_initializers.append((idx, new_initializer))
                indices_to_remove.append(idx)
        
        # 批量替换：先删除旧的（倒序删除避免索引偏移），再添加新的
        for idx in reversed(indices_to_remove):
            del model.graph.initializer[idx]
        
        for _, new_init in new_initializers:
            model.graph.initializer.append(new_init)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        # 保存
        onnx.save(model, output_path)
        
        return ConversionResult(
            success=True,
            target_backend=self.config.target_backend,
            precision_mode=PrecisionMode.FP16,
            output_files=[output_path],
            message="FP16 conversion successful (manual)",
            stats={
                "original_size_mb": self.get_file_size_mb(onnx_path),
                "output_size_mb": self.get_file_size_mb(output_path),
            },
        )
    
    def _convert_int8(
        self,
        onnx_path: str,
        output_path: str,
        input_shape: Tuple[int, ...],
    ) -> ConversionResult:
        """
        INT8 静态量化
        
        修复说明:
        1. 添加预处理超时机制（防止卡死）
        2. 提供跳过预处理选项
        3. 使用 onnx.shape_inference 作为轻量替代
        """
        self.logger.info("Converting to INT8 (PTQ)...")
        
        # 获取输入名称
        model = onnx.load(onnx_path)
        input_name = model.graph.input[0].name
        
        # 检查是否跳过预处理（可通过配置控制）
        skip_preprocess = getattr(self.config, 'skip_ort_preprocess', False)
        preprocess_timeout = getattr(self.config, 'preprocess_timeout', 120)  # 默认120秒
        
        # 创建临时目录用于预处理
        with tempfile.TemporaryDirectory() as tmp_dir:
            preprocessed_path = onnx_path  # 默认使用原始模型
            
            if not skip_preprocess:
                preprocessed_path = self._try_preprocess(
                    onnx_path, 
                    tmp_dir, 
                    timeout=preprocess_timeout
                )
            else:
                self.logger.info("Skipping ORT preprocessing (skip_ort_preprocess=True)")
            
            # 创建校准数据管理器
            calib_manager = CalibrationDataManager(
                config=self.config,
                input_shape=input_shape,
            )
            
            # 创建校准数据读取器
            calib_reader = ORTCalibrationDataReader(
                calib_data_manager=calib_manager,
                input_name=input_name,
            )
            
            # 获取校准方法
            calib_method = self.calib_method_map.get(
                self.config.calib_method.value,
                CalibrationMethod.Entropy,
            )
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            
            # 执行量化
            self.logger.info(f"Calibration method: {self.config.calib_method.value}")
            self.logger.info(f"Calibration samples: {len(calib_manager)}")
            
            quantize_static(
                model_input=preprocessed_path,
                model_output=output_path,
                calibration_data_reader=calib_reader,
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                per_channel=True,
                reduce_range=False,
                calibrate_method=calib_method,
            )
        
        self.logger.info(f"INT8 model saved to: {output_path}")
        
        return ConversionResult(
            success=True,
            target_backend=self.config.target_backend,
            precision_mode=PrecisionMode.INT8,
            output_files=[output_path],
            message="INT8 quantization successful",
            stats={
                "original_size_mb": self.get_file_size_mb(onnx_path),
                "output_size_mb": self.get_file_size_mb(output_path),
                "calibration_method": self.config.calib_method.value,
                "calibration_samples": len(calib_manager),
                "quant_format": "QDQ",
                "per_channel": True,
            },
        )
    
    def _try_preprocess(
        self, 
        onnx_path: str, 
        tmp_dir: str, 
        timeout: int = 120
    ) -> str:
        """
        尝试预处理模型（带超时和多种回退策略）
        
        Args:
            onnx_path: 输入ONNX模型路径
            tmp_dir: 临时目录
            timeout: 超时时间（秒）
            
        Returns:
            预处理后的模型路径，失败时返回原始路径
        """
        import signal
        import threading
        
        preprocessed_path = os.path.join(tmp_dir, "preprocessed.onnx")
        
        # 策略1: 尝试使用 onnx.shape_inference（轻量级，通常很快）
        try:
            self.logger.info("Attempting lightweight shape inference...")
            model = onnx.load(onnx_path)
            model_with_shapes = onnx.shape_inference.infer_shapes(model)
            onnx.save(model_with_shapes, preprocessed_path)
            self.logger.info("Lightweight shape inference successful")
            return preprocessed_path
        except Exception as e:
            self.logger.debug(f"Lightweight shape inference failed: {e}")
        
        # 策略2: 尝试 ORT 的 quant_pre_process（带超时）
        self.logger.info(f"Attempting ORT preprocessing (timeout={timeout}s)...")
        
        result = {"success": False, "error": None}
        
        def run_preprocess():
            try:
                quant_pre_process(
                    input_model_path=onnx_path,
                    output_model_path=preprocessed_path,
                    auto_merge=True,
                )
                result["success"] = True
            except Exception as e:
                result["error"] = e
        
        # 在线程中运行预处理
        thread = threading.Thread(target=run_preprocess)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout)
        
        if thread.is_alive():
            # 超时了
            self.logger.warning(
                f"ORT preprocessing timed out after {timeout}s. "
                f"Using original model. "
                f"Tip: Add 'skip_ort_preprocess: true' to config to skip this step."
            )
            return onnx_path
        
        if result["success"] and os.path.exists(preprocessed_path):
            self.logger.info("ORT preprocessing successful")
            return preprocessed_path
        else:
            self.logger.warning(
                f"ORT preprocessing failed: {result['error']}. Using original model."
            )
            return onnx_path
    
    def _convert_mixed(
        self,
        onnx_path: str,
        output_path: str,
        input_shape: Tuple[int, ...],
    ) -> ConversionResult:
        """混合精度量化"""
        self.logger.info("Converting to MIXED precision...")
        
        # 获取输入名称
        model = onnx.load(onnx_path)
        input_name = model.graph.input[0].name
        
        # 初始化敏感层管理器
        sensitive_manager = SensitiveLayerManager(
            use_default=self.config.use_default_sensitive,
            custom_layers=self.config.sensitive_layers,
        )
        
        # 获取需要排除量化的层（敏感层保持 FP16）
        nodes_to_exclude = sensitive_manager.get_sensitive_layers(model)
        
        self.logger.info(f"Sensitive layers (excluded from INT8): {len(nodes_to_exclude)}")
        for layer in nodes_to_exclude[:10]:
            self.logger.info(f"  - {layer}")
        if len(nodes_to_exclude) > 10:
            self.logger.info(f"  ... and {len(nodes_to_exclude) - 10} more")
        
        # 检查是否跳过预处理
        skip_preprocess = getattr(self.config, 'skip_ort_preprocess', False)
        preprocess_timeout = getattr(self.config, 'preprocess_timeout', 120)
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 预处理模型（使用带超时的方法）
            if not skip_preprocess:
                preprocessed_path = self._try_preprocess(
                    onnx_path, 
                    tmp_dir, 
                    timeout=preprocess_timeout
                )
            else:
                self.logger.info("Skipping ORT preprocessing (skip_ort_preprocess=True)")
                preprocessed_path = onnx_path
            
            # 创建校准数据管理器
            calib_manager = CalibrationDataManager(
                config=self.config,
                input_shape=input_shape,
            )
            
            # 创建校准数据读取器
            calib_reader = ORTCalibrationDataReader(
                calib_data_manager=calib_manager,
                input_name=input_name,
            )
            
            # 获取校准方法
            calib_method = self.calib_method_map.get(
                self.config.calib_method.value,
                CalibrationMethod.Entropy,
            )
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            
            # 执行混合精度量化
            quantize_static(
                model_input=preprocessed_path,
                model_output=output_path,
                calibration_data_reader=calib_reader,
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                per_channel=True,
                reduce_range=False,
                calibrate_method=calib_method,
                nodes_to_exclude=nodes_to_exclude,
            )
        
        self.logger.info(f"MIXED precision model saved to: {output_path}")
        
        return ConversionResult(
            success=True,
            target_backend=self.config.target_backend,
            precision_mode=PrecisionMode.MIXED,
            output_files=[output_path],
            message="MIXED precision quantization successful",
            stats={
                "original_size_mb": self.get_file_size_mb(onnx_path),
                "output_size_mb": self.get_file_size_mb(output_path),
                "calibration_method": self.config.calib_method.value,
                "calibration_samples": len(calib_manager),
                "sensitive_layers_count": len(nodes_to_exclude),
                "quant_format": "QDQ",
                "per_channel": True,
            },
        )