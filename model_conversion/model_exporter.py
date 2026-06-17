"""
🚀 Stage 4: ONNX 导出器 (Model Exporter)
========================================

将 PyTorch 模型导出为 ONNX 格式，支持动态轴配置、模型简化和验证。

功能:
1. ONNX 导出 - 支持动态 batch/height/width
2. 自定义算子处理 - 集成 symbolic.py
3. 模型简化 - onnxsim + 可选 onnxoptimizer
4. 模型验证 - 结构检查 + ORT 推理对比

与其他 Stage 的集成:
- Stage 1: 使用 input_spec/output_spec
- Stage 2: 使用推荐的 opset/precision
- Stage 3: 优先使用 optimized_model

使用方法:
---------
>>> from model_exporter import export_model, ModelExporter, ExportConfig
>>> 
>>> # 方式1: 便捷函数
>>> result = export_model(
...     model=model,
...     input_shape=(1, 3, 640, 640),
...     output_path="model.onnx",
...     task_type="det",
... )
>>> print(result.onnx_path)
>>>
>>> # 方式2: 完整控制
>>> config = ExportConfig(
...     opset_version=17,
...     enable_dynamic_batch=True,
...     enable_dynamic_hw=True,
... )
>>> exporter = ModelExporter(config)
>>> result = exporter.export(model, input_shape=(1, 3, 640, 640))

模块结构:
---------
1. 数据结构与异常
2. 动态轴配置器 (DynamicAxesConfigurator)
3. ONNX 导出引擎 (ONNXExportEngine)
4. ONNX 简化器 (ONNXSimplifier)
5. ONNX 验证器 (ONNXValidator)
6. 主导出器 (ModelExporter)
7. 便捷函数与 CLI
8. 测试接口

作者: Model Converter Team
版本: 1.0.0
"""

import io
import os
import tempfile
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

import numpy as np
import torch
import torch.nn as nn
from unified_logger import Logger, console, Timer


# ==================== 日志配置 ====================

logger = Logger.get("model_exporter")


# ==============================================================================
# 第1部分: 数据结构与异常
# ==============================================================================

class TaskType(Enum):
    """任务类型"""
    CLASSIFICATION = "cls"
    DETECTION = "det"
    SEGMENTATION = "seg"


@dataclass
class DynamicAxisConfig:
    """
    单个动态轴配置
    
    包含轴的名称、索引和范围信息 (用于后续 TensorRT Profile)
    """
    name: str              # 轴名称: "batch", "height", "width"
    axis: int              # 轴索引: 0, 2, 3
    min_value: int         # 最小值
    opt_value: int         # 最优值 (TensorRT 优化目标)
    max_value: int         # 最大值


@dataclass
class DynamicAxesSpec:
    """
    完整动态轴规格
    
    包含所有输入输出的动态轴配置
    """
    input_axes: Dict[str, List[DynamicAxisConfig]] = field(default_factory=dict)
    output_axes: Dict[str, List[DynamicAxisConfig]] = field(default_factory=dict)
    
    def to_onnx_format(self) -> Dict[str, Dict[int, str]]:
        """
        转换为 ONNX export 需要的 dynamic_axes 格式
        
        Returns:
            {"input_name": {axis_idx: "axis_name", ...}, ...}
            如果没有任何动态轴，返回 None（让 torch.onnx.export 使用完全静态形状）
        """
        result = {}
        
        for name, axes in self.input_axes.items():
            if axes:  # 只添加非空的轴配置
                result[name] = {ax.axis: ax.name for ax in axes}
        
        for name, axes in self.output_axes.items():
            if axes:  # 只添加非空的轴配置
                result[name] = {ax.axis: ax.name for ax in axes}
        
        # 如果没有任何动态轴，返回 None
        return result if result else None
    
    def to_tensorrt_profile(self) -> Dict[str, Dict[str, Tuple[int, int, int]]]:
        """
        转换为 TensorRT Profile 格式
        
        Returns:
            {"input_name": {"batch": (min, opt, max), ...}, ...}
        """
        result = {}
        
        for name, axes in self.input_axes.items():
            result[name] = {
                ax.name: (ax.min_value, ax.opt_value, ax.max_value)
                for ax in axes
            }
        
        return result
    
    def summary(self) -> str:
        """生成摘要"""
        lines = ["动态轴配置:"]
        
        for name, axes in self.input_axes.items():
            axes_str = ", ".join([
                f"{ax.name}[{ax.axis}]: {ax.min_value}/{ax.opt_value}/{ax.max_value}"
                for ax in axes
            ])
            lines.append(f"  输入 '{name}': {axes_str}")
        
        for name, axes in self.output_axes.items():
            axes_str = ", ".join([f"{ax.name}[{ax.axis}]" for ax in axes])
            lines.append(f"  输出 '{name}': {axes_str}")
        
        return "\n".join(lines)


@dataclass
class SimplifyConfig:
    """ONNX 简化配置"""
    enable_simplify: bool = False          # 修复：默认禁用简化，避免某些模型卡死
    skip_fuse_bn: bool = True              # 跳过 BN 融合 (Stage 3 已做)
    skip_shape_inference: bool = False     # 跳过形状推断
    check_n: int = 3                       # 简化后验证次数
    enable_onnxoptimizer: bool = False     # 是否启用 onnxoptimizer
    optimizer_passes: List[str] = None     # onnxoptimizer passes


@dataclass
class ValidationConfig:
    """验证配置"""
    enable_validation: bool = True         # 是否启用验证
    max_diff_threshold: float = 1e-3       # FP32 最大差异阈值 (复杂模型误差可达 1e-3)
    cosine_threshold: float = 0.999        # 余弦相似度阈值 (> 0.999 即可)
    num_test_samples: int = 3              # 测试样本数
    test_with_random_seed: int = 42        # 随机种子 (确保可复现)
    strict_mode: bool = False              # 严格模式 (使用更严格的阈值)
    input_spec: Any = None                 # InputSpec 对象，用于生成合理的测试数据


@dataclass
class ExportConfig:
    """
    ONNX 导出配置
    
    控制导出过程的各项参数
    """
    # 基本配置
    opset_version: int = 17                # ONNX opset 版本
    input_names: List[str] = None          # 输入名称
    output_names: List[str] = None         # 输出名称
    
    # 动态轴配置
    enable_dynamic_batch: bool = True      # 启用动态 batch
    enable_dynamic_hw: bool = False        # 启用动态 height/width (检测/分割默认开)
    
    # 动态轴范围 (用于 TensorRT)
    batch_range: Tuple[int, int, int] = (1, 4, 16)      # min, opt, max
    height_range: Tuple[int, int, int] = (320, 640, 1280)
    width_range: Tuple[int, int, int] = (320, 640, 1280)
    
    # 导出选项
    do_constant_folding: bool = True       # 常量折叠
    keep_initializers_as_inputs: bool = False
    verbose: bool = False                  # 详细输出
    
    # 简化配置
    simplify: SimplifyConfig = field(default_factory=SimplifyConfig)
    
    # 验证配置
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    
    def __post_init__(self):
        """初始化后处理"""
        if self.input_names is None:
            self.input_names = ["input"]
        if self.output_names is None:
            self.output_names = ["output"]
        if self.simplify is None:
            self.simplify = SimplifyConfig()
        if self.validation is None:
            self.validation = ValidationConfig()


@dataclass
class ValidationResult:
    """验证结果"""
    structure_valid: bool = False          # 结构验证通过
    numerical_valid: bool = False          # 数值验证通过
    max_diff: float = 0.0                  # 最大绝对差异
    mean_diff: float = 0.0                 # 平均绝对差异
    cosine_sim: float = 0.0                # 余弦相似度
    messages: List[str] = field(default_factory=list)
    
    @property
    def passed(self) -> bool:
        """是否通过所有验证"""
        return self.structure_valid and self.numerical_valid


@dataclass
class ExportStats:
    """导出统计"""
    original_node_count: int = 0           # 原始节点数
    simplified_node_count: int = 0         # 简化后节点数
    export_time_ms: float = 0.0            # 导出时间
    simplify_time_ms: float = 0.0          # 简化时间
    validation_time_ms: float = 0.0        # 验证时间
    file_size_mb: float = 0.0              # 文件大小
    
    def summary(self) -> str:
        """生成摘要"""
        reduction = self.original_node_count - self.simplified_node_count
        reduction_pct = (reduction / self.original_node_count * 100) if self.original_node_count > 0 else 0
        
        return f"""
导出统计:
  节点数: {self.original_node_count} → {self.simplified_node_count} (减少 {reduction_pct:.1f}%)
  导出时间: {self.export_time_ms:.1f} ms
  简化时间: {self.simplify_time_ms:.1f} ms
  验证时间: {self.validation_time_ms:.1f} ms
  文件大小: {self.file_size_mb:.2f} MB
"""


@dataclass
class ExportResult:
    """
    导出结果
    
    包含导出的 ONNX 模型及相关信息
    """
    success: bool                          # 是否成功
    onnx_path: str = ""                    # ONNX 文件路径
    onnx_model: Any = None                 # ONNX ModelProto (可选保留)
    dynamic_axes_spec: DynamicAxesSpec = None
    validation_result: ValidationResult = None
    stats: ExportStats = None
    message: str = ""                      # 状态消息
    
    def summary(self) -> str:
        """生成摘要"""
        status = "✅ 成功" if self.success else "❌ 失败"
        lines = [
            "=" * 60,
            f"📦 ONNX 导出结果: {status}",
            "=" * 60,
        ]
        
        if self.success:
            lines.extend([
                f"  文件路径: {self.onnx_path}",
                "",
            ])
            
            if self.dynamic_axes_spec:
                lines.append(self.dynamic_axes_spec.summary())
                lines.append("")
            
            if self.validation_result:
                val = self.validation_result
                lines.extend([
                    "验证结果:",
                    f"  结构验证: {'✅' if val.structure_valid else '❌'}",
                    f"  数值验证: {'✅' if val.numerical_valid else '❌'}",
                    f"  最大差异: {val.max_diff:.2e}",
                    f"  余弦相似度: {val.cosine_sim:.6f}",
                ])
            
            if self.stats:
                lines.append(self.stats.summary())
        else:
            lines.append(f"  错误: {self.message}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# ==================== 异常定义 ====================

class ONNXExportError(Exception):
    """ONNX 导出错误基类"""
    pass


class ExportConfigError(ONNXExportError):
    """配置错误"""
    def __init__(self, param: str, reason: str):
        super().__init__(f"❌ 配置错误 [{param}]: {reason}")
        self.param = param
        self.reason = reason


class DynamicAxesError(ONNXExportError):
    """动态轴配置错误"""
    def __init__(self, message: str):
        super().__init__(f"❌ 动态轴配置错误: {message}")


class OutputInferenceError(ONNXExportError):
    """输出推断错误"""
    def __init__(self, task_type: str, output_shapes: Dict[str, Tuple], message: str = ""):
        msg = f"❌ 无法推断输出动态轴\n  任务类型: {task_type}\n  输出形状: {output_shapes}"
        if message:
            msg += f"\n  详情: {message}"
        msg += "\n💡 请手动指定 output_dynamic_axes 参数"
        super().__init__(msg)
        self.task_type = task_type
        self.output_shapes = output_shapes


class SimplifyError(ONNXExportError):
    """简化错误"""
    def __init__(self, reason: str):
        super().__init__(f"❌ ONNX 简化失败: {reason}")


class ValidationError(ONNXExportError):
    """验证错误"""
    def __init__(self, validation_type: str, reason: str, details: Dict = None):
        msg = f"❌ ONNX 验证失败 [{validation_type}]: {reason}"
        if details:
            for k, v in details.items():
                msg += f"\n  {k}: {v}"
        super().__init__(msg)
        self.validation_type = validation_type
        self.details = details


class DependencyError(ONNXExportError):
    """依赖缺失错误"""
    def __init__(self, package: str, install_cmd: str):
        super().__init__(
            f"❌ 缺少必需依赖: {package}\n"
            f"💡 安装命令: {install_cmd}"
        )


# ==============================================================================
# 第2部分: 依赖检查
# ==============================================================================

def check_dependencies() -> Dict[str, bool]:
    """
    检查必需和可选依赖
    
    Returns:
        {"package_name": is_available}
    """
    deps = {}
    
    # 必需依赖: ONNX
    try:
        import onnx
        deps["onnx"] = True
    except ImportError:
        deps["onnx"] = False
    
    # 必需依赖: ONNX Runtime
    try:
        import onnxruntime
        deps["onnxruntime"] = True
    except ImportError:
        deps["onnxruntime"] = False
    
    # 可选依赖: onnxsim
    try:
        import onnxsim
        deps["onnxsim"] = True
    except ImportError:
        deps["onnxsim"] = False
    
    # 可选依赖: onnxoptimizer
    try:
        import onnxoptimizer
        deps["onnxoptimizer"] = True
    except ImportError:
        deps["onnxoptimizer"] = False
    
    return deps


def ensure_dependencies():
    """确保必需依赖已安装"""
    deps = check_dependencies()
    
    if not deps.get("onnx"):
        raise DependencyError("onnx", "pip install onnx>=1.14")
    
    if not deps.get("onnxruntime"):
        raise DependencyError(
            "onnxruntime",
            "pip install onnxruntime-gpu  # 或 onnxruntime (CPU)"
        )


# ==============================================================================
# 第3部分: 动态轴配置器 (DynamicAxesConfigurator)
# ==============================================================================

class DynamicAxesConfigurator:
    """
    动态轴配置器
    ============
    
    根据任务类型和输入输出规格配置动态轴。
    
    支持的任务:
    - cls (分类): batch 动态，H/W 通常固定
    - det (检测): batch + H/W 动态 (YOLO 风格输出)
    - seg (分割): batch + H/W 动态
    """
    
    # 各任务的默认动态轴范围
    DEFAULT_RANGES = {
        "cls": {
            "batch": (1, 8, 32),
            "height": None,  # 通常固定
            "width": None,
        },
        "det": {
            "batch": (1, 4, 16),
            "height": (320, 640, 1280),
            "width": (320, 640, 1280),
        },
        "seg": {
            "batch": (1, 2, 8),
            "height": (256, 512, 1024),
            "width": (256, 512, 1024),
        },
    }
    
    def __init__(self, config: ExportConfig):
        """
        初始化配置器
        
        Args:
            config: 导出配置
        """
        self.config = config
    
    def configure(
        self,
        task_type: str,
        input_shape: Tuple[int, ...],
        input_names: List[str],
        output_names: List[str],
        output_shapes: Dict[str, Tuple[int, ...]] = None,
    ) -> DynamicAxesSpec:
        """
        配置动态轴
        
        Args:
            task_type: 任务类型 (cls/det/seg)
            input_shape: 输入形状 (B, C, H, W)
            input_names: 输入名称列表
            output_names: 输出名称列表
            output_shapes: 输出形状字典 (可选，用于推断输出动态轴)
            
        Returns:
            DynamicAxesSpec
        """
        if len(input_shape) != 4:
            raise DynamicAxesError(
                f"输入形状必须是 4D (B, C, H, W)，但得到 {len(input_shape)}D"
            )
        
        spec = DynamicAxesSpec()
        
        # 配置输入动态轴
        for name in input_names:
            axes = self._configure_input_axes(task_type, input_shape)
            spec.input_axes[name] = axes
        
        # 配置输出动态轴
        for name in output_names:
            output_shape = output_shapes.get(name) if output_shapes else None
            axes = self._configure_output_axes(task_type, output_shape, name)
            spec.output_axes[name] = axes
        
        return spec
    
    def _configure_input_axes(
        self,
        task_type: str,
        input_shape: Tuple[int, ...],
    ) -> List[DynamicAxisConfig]:
        """配置输入的动态轴"""
        axes = []
        
        # Batch 维度
        if self.config.enable_dynamic_batch:
            batch_range = self.config.batch_range or self.DEFAULT_RANGES[task_type]["batch"]
            axes.append(DynamicAxisConfig(
                name="batch",
                axis=0,
                min_value=batch_range[0],
                opt_value=batch_range[1],
                max_value=batch_range[2],
            ))
        
        # Height/Width 维度 - 完全尊重用户配置
        if self.config.enable_dynamic_hw:
            height_range = self.config.height_range or self.DEFAULT_RANGES[task_type]["height"]
            if height_range:
                axes.append(DynamicAxisConfig(
                    name="height",
                    axis=2,
                    min_value=height_range[0],
                    opt_value=height_range[1],
                    max_value=height_range[2],
                ))
            
            width_range = self.config.width_range or self.DEFAULT_RANGES[task_type]["width"]
            if width_range:
                axes.append(DynamicAxisConfig(
                    name="width",
                    axis=3,
                    min_value=width_range[0],
                    opt_value=width_range[1],
                    max_value=width_range[2],
                ))
        
        return axes
    
    def _configure_output_axes(
        self,
        task_type: str,
        output_shape: Tuple[int, ...] = None,
        output_name: str = "output",
    ) -> List[DynamicAxisConfig]:
        """
        配置输出的动态轴
        
        YOLO 风格输出: [B, num_anchors, 5+num_classes] 或 [B, num_boxes, 6]
        """
        axes = []
        
        # Batch 维度始终与输入保持一致
        if self.config.enable_dynamic_batch:
            batch_range = self.config.batch_range or (1, 4, 16)
            axes.append(DynamicAxisConfig(
                name="batch",
                axis=0,
                min_value=batch_range[0],
                opt_value=batch_range[1],
                max_value=batch_range[2],
            ))
        
        # 根据任务类型配置其他输出轴
        if task_type == "cls":
            # 分类输出: [B, num_classes] - 只有 batch 动态
            pass
        
        elif task_type == "det":
            # YOLO 风格: [B, num_anchors, 5+num_classes]
            # num_anchors 取决于输入尺寸，但通常固定
            # 不额外添加动态轴
            pass
        
        elif task_type == "seg":
            # 分割输出: [B, C, H, W] - H/W 配置与输入保持一致
            if self.config.enable_dynamic_hw:
                height_range = self.config.height_range or (256, 512, 1024)
                width_range = self.config.width_range or (256, 512, 1024)
                
                axes.append(DynamicAxisConfig(
                    name="height",
                    axis=2,
                    min_value=height_range[0],
                    opt_value=height_range[1],
                    max_value=height_range[2],
                ))
                axes.append(DynamicAxisConfig(
                    name="width",
                    axis=3,
                    min_value=width_range[0],
                    opt_value=width_range[1],
                    max_value=width_range[2],
                ))
        
        return axes


# ==============================================================================
# 第4部分: ONNX 导出引擎 (ONNXExportEngine)
# ==============================================================================

class ONNXExportEngine:
    """
    ONNX 导出引擎
    =============
    
    封装 torch.onnx.export 调用，处理自定义算子注册。
    """
    
    def __init__(self, config: ExportConfig):
        """
        初始化导出引擎
        
        Args:
            config: 导出配置
        """
        self.config = config
    
    def export(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        output_path: str,
        dynamic_axes: Dict[str, Dict[int, str]] = None,
        sample_input: Optional[torch.Tensor] = None,
    ) -> Tuple[Any, int]:
        """
        导出 ONNX 模型
        
        Args:
            model: PyTorch 模型
            input_shape: 输入形状
            output_path: 输出路径
            dynamic_axes: 动态轴配置
            sample_input: 真实样本输入（可选，推荐提供以确保导出正确）
            
        Returns:
            (onnx_model, node_count)
        """
        import onnx
        import sys
        
        # ========== 添加详细调试日志 ==========
        logger.debug(f"[EXPORT] 开始导出到: {output_path}")
        sys.stdout.flush()
        
        # 注册自定义 symbolic
        logger.debug("[EXPORT] 注册symbolic...")
        sys.stdout.flush()
        self._register_symbolics()
        
        # 准备输入
        logger.debug("[EXPORT] 获取设备...")
        sys.stdout.flush()
        device = self._get_device(model)
        logger.debug(f"[EXPORT] 设备: {device}")
        
        if sample_input is not None:
            # 使用提供的真实样本
            dummy_input = sample_input.to(device)
            logger.info("使用真实样本数据进行 ONNX 导出")
        else:
            # 使用 [0, 1] 范围的随机数据，更接近归一化后的图像
            # 而不是标准正态分布
            logger.debug(f"[EXPORT] 创建随机输入 {input_shape}...")
            sys.stdout.flush()
            dummy_input = torch.rand(input_shape, device=device)
            logger.debug("使用随机数据进行 ONNX 导出（范围 [0, 1]）")
        
        # 确保模型在 eval 模式
        logger.debug("[EXPORT] 设置eval模式...")
        sys.stdout.flush()
        model.eval()
        
        # 导出参数
        export_kwargs = {
            "opset_version": self.config.opset_version,
            "input_names": self.config.input_names,
            "output_names": self.config.output_names,
            "do_constant_folding": self.config.do_constant_folding,
            "dynamic_axes": dynamic_axes,
            "verbose": self.config.verbose,
        }
        
        # PyTorch 2.0+ 的新参数
        try:
            export_kwargs["keep_initializers_as_inputs"] = self.config.keep_initializers_as_inputs
        except Exception:
            pass
        
        # 导出
        logger.info(f"正在导出 ONNX (opset {self.config.opset_version})...")
        print(f"  [DEBUG] 准备导出，设备={device}", flush=True)
        
        original_device = device
        
        # 步骤1：同步CUDA
        print("  [DEBUG] 同步CUDA...", flush=True)
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [DEBUG] CUDA同步警告: {e}", flush=True)
        
        # 步骤2：将模型和输入移到CPU
        print("  [DEBUG] 将模型移到CPU...", flush=True)
        model_cpu = model.cpu()
        model_cpu.eval()
        dummy_input_cpu = dummy_input.cpu()
        
        # 清理GPU内存
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 步骤3：使用torch.jit.trace预编译模型
        # 这可以解决某些ONNX导出卡住的问题
        print("  [DEBUG] 使用torch.jit.trace预编译模型...", flush=True)
        sys.stdout.flush()
        
        try:
            with torch.no_grad():
                traced_model = torch.jit.trace(model_cpu, dummy_input_cpu)
                traced_model.eval()
            print("  [DEBUG] JIT trace成功", flush=True)
            use_traced = True
        except Exception as e:
            print(f"  [DEBUG] JIT trace失败: {e}，使用原模型", flush=True)
            traced_model = model_cpu
            use_traced = False
        
        # 步骤4：执行导出
        print("  [DEBUG] 执行torch.onnx.export...", flush=True)
        sys.stdout.flush()
        
        try:
            with torch.no_grad():
                torch.onnx.export(
                    traced_model if use_traced else model_cpu,
                    dummy_input_cpu,
                    output_path,
                    **export_kwargs
                )
            print("  [DEBUG] torch.onnx.export完成", flush=True)
        except Exception as e:
            print(f"  [DEBUG] 导出异常: {e}", flush=True)
            raise
        
        # 步骤5：将原始模型移回原设备
        print("  [DEBUG] 将原始模型移回原设备...", flush=True)
        try:
            if original_device.type == 'cuda':
                model.to(original_device)
                print(f"  [DEBUG] 原始模型已移回 {original_device}", flush=True)
        except Exception as e:
            print(f"  [DEBUG] 无法将模型移回原设备: {e}", flush=True)
        
        # 清理
        try:
            if use_traced:
                del traced_model
            del dummy_input_cpu
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        
        logger.debug("[EXPORT] torch.onnx.export 完成")
        sys.stdout.flush()
        
        # 加载并返回
        logger.debug("[EXPORT] 加载导出的模型...")
        sys.stdout.flush()
        onnx_model = onnx.load(output_path)
        
        # 修复 Resize 节点 (PyTorch 可能同时生成 scales 和 sizes)
        logger.debug("[EXPORT] 修复 Resize 节点...")
        sys.stdout.flush()
        onnx_model = self._fix_resize_nodes(onnx_model)
        onnx.save(onnx_model, output_path)
        
        node_count = len(onnx_model.graph.node)
        
        logger.info(f"导出完成，共 {node_count} 个节点")
        sys.stdout.flush()
        
        return onnx_model, node_count
    
    def _fix_resize_nodes(self, onnx_model):
        """
        修复 Resize 节点
        
        PyTorch 导出可能同时生成 scales 和 sizes，但 ONNX Runtime 要求只能有一个。
        此方法检测并修复这种情况。
        """
        import onnx
        from onnx import numpy_helper, TensorProto
        
        graph = onnx_model.graph
        
        # 收集所有初始化器 (常量)
        initializers = {init.name: init for init in graph.initializer}
        
        # 收集所有节点输出
        node_outputs = set()
        for node in graph.node:
            for output in node.output:
                node_outputs.add(output)
        
        # 检查每个 Resize 节点
        nodes_to_check = [node for node in graph.node if node.op_type == 'Resize']
        
        if not nodes_to_check:
            return onnx_model
        
        fixed_count = 0
        
        for node in nodes_to_check:
            # Resize 节点输入: X, roi, scales, sizes (后两个可选)
            if len(node.input) < 4:
                continue
            
            scales_name = node.input[2] if len(node.input) > 2 else ""
            sizes_name = node.input[3] if len(node.input) > 3 else ""
            
            def is_empty_or_none(name):
                """检查输入是否为空或 None"""
                if not name or name == "":
                    return True
                if name in initializers:
                    tensor = numpy_helper.to_array(initializers[name])
                    return tensor.size == 0
                return False
            
            def is_valid_input(name):
                """检查输入是否有效（非空）"""
                if not name or name == "":
                    return False
                if name in initializers:
                    tensor = numpy_helper.to_array(initializers[name])
                    return tensor.size > 0
                # 动态输入 (来自其他节点)
                return name in node_outputs
            
            scales_valid = is_valid_input(scales_name)
            sizes_valid = is_valid_input(sizes_name)
            
            # 如果两个都有效，需要修复 - 优先使用 sizes，清空 scales
            if scales_valid and sizes_valid:
                # 创建空的 scales tensor 名称
                empty_scales_name = f"{node.name}_empty_scales"
                
                # 检查是否已存在
                if empty_scales_name not in initializers:
                    empty_scales = onnx.helper.make_tensor(
                        empty_scales_name,
                        TensorProto.FLOAT,
                        [0],
                        []
                    )
                    graph.initializer.append(empty_scales)
                
                # 更新节点输入
                inputs = list(node.input)
                inputs[2] = empty_scales_name
                del node.input[:]
                node.input.extend(inputs)
                
                fixed_count += 1
                logger.debug(f"修复 Resize 节点: {node.name} (清空 scales, 保留 sizes)")
            
            # 如果只有 scales 有效但 sizes 也存在，清空 sizes
            elif scales_valid and sizes_name and not sizes_valid:
                # 创建空的 sizes tensor
                empty_sizes_name = f"{node.name}_empty_sizes"
                
                if empty_sizes_name not in initializers:
                    empty_sizes = onnx.helper.make_tensor(
                        empty_sizes_name,
                        TensorProto.INT64,
                        [0],
                        []
                    )
                    graph.initializer.append(empty_sizes)
                
                inputs = list(node.input)
                inputs[3] = empty_sizes_name
                del node.input[:]
                node.input.extend(inputs)
                
                fixed_count += 1
                logger.debug(f"修复 Resize 节点: {node.name} (保留 scales, 清空 sizes)")
        
        if fixed_count > 0:
            logger.info(f"已修复 {fixed_count} 个 Resize 节点")
        
        return onnx_model
    
    def _register_symbolics(self):
        """注册自定义 symbolic - 已禁用"""
        logger.debug("跳过 symbolic 注册 (YOLO 已内置支持)")
        return 
        try:
            from symbolic import register_all_symbolics
            register_all_symbolics(self.config.opset_version)
            logger.debug("已注册自定义 symbolic")
        except ImportError:
            logger.warning("symbolic.py 未找到，跳过自定义算子注册")
        except Exception as e:
            logger.warning(f"注册 symbolic 失败: {e}")
    
    def _get_device(self, model: nn.Module) -> torch.device:
        """获取模型设备"""
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device('cpu')


# ==============================================================================
# 第5部分: ONNX 简化器 (ONNXSimplifier)
# ==============================================================================

class ONNXSimplifier:
    """
    ONNX 简化器
    ===========
    
    使用 onnxsim 和可选的 onnxoptimizer 简化 ONNX 模型。
    """
    
    def __init__(self, config: SimplifyConfig):
        """
        初始化简化器
        
        Args:
            config: 简化配置
        """
        self.config = config
    
    def simplify(
        self,
        onnx_model: Any,
        input_shape: Tuple[int, ...],
        dynamic_axes: Dict[str, Dict[int, str]] = None,
    ) -> Tuple[Any, int]:
        """
        简化 ONNX 模型
        
        Args:
            onnx_model: ONNX ModelProto
            input_shape: 输入形状
            dynamic_axes: 动态轴配置
            
        Returns:
            (simplified_model, node_count)
        """
        import onnx
        
        if not self.config.enable_simplify:
            return onnx_model, len(onnx_model.graph.node)
        
        # 检查 onnxsim 是否可用
        try:
            import onnxsim
        except ImportError:
            logger.warning("onnxsim 未安装，跳过简化")
            return onnx_model, len(onnx_model.graph.node)
        
        logger.info("正在简化 ONNX 模型...")
        
        original_node_count = len(onnx_model.graph.node)
        
        # Step 0: 预处理 - 强制设置静态输入形状（解决 EfficientNet 等模型的 TensorRT 兼容问题）
        if dynamic_axes:
            logger.debug("检测到动态轴配置，保留动态维度")
            onnx_model = self._fix_input_shapes_preserve_dynamic(onnx_model, input_shape, dynamic_axes)
        else:
            onnx_model = self._fix_input_shapes(onnx_model, input_shape)
        
        # Step 1: 形状推断 (可选)
        if not self.config.skip_shape_inference:
            try:
                onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
            except Exception as e:
                logger.warning(f"形状推断失败 (继续): {e}")
        
        # Step 2: onnxsim 简化
        # 使用固定形状进行简化
        input_name = onnx_model.graph.input[0].name if onnx_model.graph.input else "input"
        input_shapes = {input_name: list(input_shape)}
        
        # 尝试简化，如果失败则尝试不同策略
        simplified = None
        strategies = [
            # 策略1: 使用固定形状 + 形状推断
            {"skip_shape_inference": False, "input_shapes": input_shapes},
            # 策略2: 使用固定形状，跳过形状推断
            {"skip_shape_inference": True, "input_shapes": input_shapes},
            # 策略3: 不使用输入形状
            {"skip_shape_inference": True, "input_shapes": None},
        ]
        
        last_error = None
        for i, strategy in enumerate(strategies):
            try:
                logger.debug(f"尝试简化策略 {i+1}/{len(strategies)}: {strategy}")
                
                # 根据 onnxsim 版本选择参数
                try:
                    # onnxsim >= 0.4.33 使用新参数名
                    simplified, check = onnxsim.simplify(
                        onnx_model,
                        skip_fuse_bn=self.config.skip_fuse_bn,
                        check_n=self.config.check_n,
                        overwrite_input_shapes=strategy["input_shapes"],
                        skip_shape_inference=strategy["skip_shape_inference"],
                    )
                except TypeError:
                    # 旧版本 onnxsim 使用 input_shapes
                    simplified, check = onnxsim.simplify(
                        onnx_model,
                        skip_fuse_bn=self.config.skip_fuse_bn,
                        check_n=self.config.check_n,
                        input_shapes=strategy["input_shapes"],
                    )
                
                if check:
                    onnx_model = simplified
                    logger.debug(f"简化策略 {i+1} 成功")
                    break
                else:
                    last_error = "简化验证失败"
                    
            except Exception as e:
                last_error = str(e)
                logger.debug(f"简化策略 {i+1} 失败: {e}")
                continue
        else:
            # 所有策略都失败
            logger.warning(f"⚠️ ONNX 简化失败，使用原始模型: {last_error}")
            logger.warning("  提示: 动态形状模型可能无法简化，这不影响模型功能")
            return onnx_model, original_node_count
        
        # Step 3: 后处理 - 再次确保形状正确（保留动态轴）
        if dynamic_axes:
            onnx_model = self._fix_input_shapes_preserve_dynamic(onnx_model, input_shape, dynamic_axes)
            logger.debug("已保留动态轴配置")
        else:
            onnx_model = self._fix_input_shapes(onnx_model, input_shape)
        
        # Step 4: 再次形状推断（仅对静态形状模型）
        # 对于动态形状模型，形状推断可能会导致问题
        if not dynamic_axes:
            try:
                onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
            except Exception as e:
                logger.debug(f"最终形状推断失败: {e}")
        
        # Step 5: 可选 onnxoptimizer
        if self.config.enable_onnxoptimizer:
            onnx_model = self._run_onnxoptimizer(onnx_model)
        
        node_count = len(onnx_model.graph.node)
        logger.info(f"简化完成，节点数: {original_node_count} → {node_count}")
        
        return onnx_model, node_count
    
    def _fix_input_shapes(self, onnx_model: Any, input_shape: Tuple[int, ...]) -> Any:
        """
        修复 ONNX 模型的输入形状
        
        强制将所有输入设置为静态形状，解决 EfficientNet、MobileNet 等模型
        在 TensorRT 中的兼容性问题。
        
        Args:
            onnx_model: ONNX ModelProto
            input_shape: 目标输入形状 (B, C, H, W)
            
        Returns:
            修复后的 ONNX 模型
        """
        import onnx
        from onnx import TensorProto, helper
        
        try:
            # 获取第一个输入
            if not onnx_model.graph.input:
                return onnx_model
            
            graph_input = onnx_model.graph.input[0]
            
            # 创建新的输入，使用静态形状
            new_dim = []
            for i, dim_value in enumerate(input_shape):
                new_dim.append(onnx.helper.make_tensor_value_info(
                    graph_input.name,
                    graph_input.type.tensor_type.elem_type,
                    list(input_shape)
                ).type.tensor_type.shape.dim[i])
            
            # 清除原有的维度
            while len(graph_input.type.tensor_type.shape.dim) > 0:
                graph_input.type.tensor_type.shape.dim.pop()
            
            # 添加新的静态维度
            for dim in new_dim:
                new_dim_proto = graph_input.type.tensor_type.shape.dim.add()
                new_dim_proto.dim_value = dim.dim_value
            
            logger.debug(f"已设置输入 '{graph_input.name}' 的静态形状: {input_shape}")
            
        except Exception as e:
            logger.debug(f"修复输入形状失败 (继续): {e}")
        
        return onnx_model
    def _fix_input_shapes_preserve_dynamic(
        self, 
        onnx_model: Any, 
        input_shape: Tuple[int, ...],
        dynamic_axes: Dict[str, Dict[int, str]]) -> Any:
        """
        修复 ONNX 模型的输入形状，同时保留动态轴
        
        与 _fix_input_shapes 不同，此方法会保留 dynamic_axes 中指定的动态维度。
        
        Args:
            onnx_model: ONNX ModelProto
            input_shape: 目标输入形状 (B, C, H, W)
            dynamic_axes: 动态轴配置，格式 {"input": {0: "batch", 2: "height", ...}}
            
        Returns:
            修复后的 ONNX 模型
        """
        import onnx
        
        try:
            # 获取第一个输入
            if not onnx_model.graph.input:
                return onnx_model
            
            graph_input = onnx_model.graph.input[0]
            input_name = graph_input.name
            
            # 获取该输入的动态轴索引
            dynamic_axis_indices = set()
            if input_name in dynamic_axes:
                dynamic_axis_indices = set(dynamic_axes[input_name].keys())
            
            # 如果没有动态轴，使用原方法
            if not dynamic_axis_indices:
                return self._fix_input_shapes(onnx_model, input_shape)
            
            # 清除原有的维度
            while len(graph_input.type.tensor_type.shape.dim) > 0:
                graph_input.type.tensor_type.shape.dim.pop()
            
            # 添加新的维度（保留动态轴）
            for i, dim_value in enumerate(input_shape):
                new_dim = graph_input.type.tensor_type.shape.dim.add()
                if i in dynamic_axis_indices:
                    # 动态维度：使用 dim_param (字符串名称)
                    axis_name = dynamic_axes[input_name][i]
                    new_dim.dim_param = axis_name
                    logger.debug(f"保留动态轴: dim[{i}] = '{axis_name}'")
                else:
                    # 静态维度：使用 dim_value (整数值)
                    new_dim.dim_value = dim_value
            
            # 同样处理输出
            for graph_output in onnx_model.graph.output:
                output_name = graph_output.name
                if output_name in dynamic_axes:
                    output_dynamic_indices = set(dynamic_axes[output_name].keys())
                    if output_dynamic_indices:
                        output_shape = graph_output.type.tensor_type.shape
                        for i, dim in enumerate(output_shape.dim):
                            if i in output_dynamic_indices:
                                axis_name = dynamic_axes[output_name][i]
                                dim.ClearField('dim_value')
                                dim.dim_param = axis_name
            
            logger.info(f"已设置输入 '{input_name}' 的形状，保留动态轴: {dynamic_axis_indices}")
            
        except Exception as e:
            logger.warning(f"修复输入形状（保留动态轴）失败: {e}")
            return self._fix_input_shapes(onnx_model, input_shape)
        
        return onnx_model

    def _run_onnxoptimizer(self, onnx_model: Any) -> Any:
        """运行 onnxoptimizer"""
        try:
            import onnxoptimizer
        except ImportError:
            logger.warning("onnxoptimizer 未安装，跳过额外优化")
            return onnx_model
        
        passes = self.config.optimizer_passes or [
            'eliminate_identity',
            'eliminate_deadend',
            'eliminate_nop_transpose',
            'fuse_consecutive_transposes',
            'fuse_transpose_into_gemm',
        ]
        
        try:
            onnx_model = onnxoptimizer.optimize(onnx_model, passes)
            logger.debug(f"onnxoptimizer 完成，使用 passes: {passes}")
        except Exception as e:
            logger.warning(f"onnxoptimizer 失败: {e}")
        
        return onnx_model


# ==============================================================================
# 第6部分: ONNX 验证器 (ONNXValidator)
# ==============================================================================

class ONNXValidator:
    """
    ONNX 验证器
    ===========
    
    验证 ONNX 模型的结构完整性和数值正确性。
    """
    
    def __init__(self, config: ValidationConfig):
        """
        初始化验证器
        
        Args:
            config: 验证配置
        """
        self.config = config
    
    def validate(
        self,
        onnx_model: Any,
        onnx_path: str,
        pytorch_model: nn.Module,
        input_shape: Tuple[int, ...],
        sample_inputs: Optional[List[torch.Tensor]] = None,
    ) -> ValidationResult:
        """
        完整验证 ONNX 模型
        
        Args:
            onnx_model: ONNX ModelProto
            onnx_path: ONNX 文件路径
            pytorch_model: 原始 PyTorch 模型
            input_shape: 输入形状
            sample_inputs: 真实样本输入列表（可选，推荐提供）
            
        Returns:
            ValidationResult
        """
        result = ValidationResult()
        
        if not self.config.enable_validation:
            result.structure_valid = True
            result.numerical_valid = True
            result.messages.append("验证已跳过")
            return result
        
        # Step 1: 结构验证
        result = self._validate_structure(onnx_model, result)
        
        if not result.structure_valid:
            return result
        
        # Step 2: 数值验证
        result = self._validate_numerical(
            onnx_path,
            pytorch_model,
            input_shape,
            result,
            sample_inputs=sample_inputs,
        )
        
        return result
    
    def _validate_structure(
        self,
        onnx_model: Any,
        result: ValidationResult,
    ) -> ValidationResult:
        """结构验证"""
        import onnx
        
        logger.info("正在进行结构验证...")
        
        try:
            # 基本检查
            onnx.checker.check_model(onnx_model)
            result.messages.append("✓ 模型结构检查通过")
            
            # 检查权重是否有 NaN/Inf
            for tensor in onnx_model.graph.initializer:
                arr = onnx.numpy_helper.to_array(tensor)
                if np.isnan(arr).any():
                    raise ValidationError(
                        "结构验证",
                        f"权重 '{tensor.name}' 包含 NaN"
                    )
                if np.isinf(arr).any():
                    raise ValidationError(
                        "结构验证",
                        f"权重 '{tensor.name}' 包含 Inf"
                    )
            
            result.messages.append("✓ 权重数值范围检查通过")
            result.structure_valid = True
            
        except onnx.checker.ValidationError as e:
            result.structure_valid = False
            result.messages.append(f"✗ 结构验证失败: {e}")
            raise ValidationError("结构验证", str(e))
        
        return result
    
    def _validate_numerical(
        self,
        onnx_path: str,
        pytorch_model: nn.Module,
        input_shape: Tuple[int, ...],
        result: ValidationResult,
        sample_inputs: Optional[List[torch.Tensor]] = None,
    ) -> ValidationResult:
        """
        数值验证 - 对比 PyTorch 和 ORT 输出
        
        Args:
            onnx_path: ONNX 文件路径
            pytorch_model: PyTorch 模型
            input_shape: 输入形状
            result: 验证结果对象
            sample_inputs: 真实样本输入列表（可选）
            
        Returns:
            更新后的验证结果
        """
        import onnxruntime as ort
        
        logger.info("正在进行数值验证...")
        
        # 获取设备
        device = self._get_device(pytorch_model)
        
        # 创建 ORT Session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if device.type == 'cpu':
            providers = ['CPUExecutionProvider']
        
        session = None  # 初始化为 None，确保 finally 中可以安全检查
        try:
            session = ort.InferenceSession(onnx_path, providers=providers)
        except Exception as e:
            raise ValidationError("数值验证", f"无法创建 ORT Session: {e}")
        
        try:
            # 获取输入输出名称
            input_name = session.get_inputs()[0].name
            output_names = [o.name for o in session.get_outputs()]
        
            # 运行多次测试
            max_diffs = []
            mean_diffs = []
            cosine_sims = []
            
            torch.manual_seed(self.config.test_with_random_seed)
            np.random.seed(self.config.test_with_random_seed)
            
            pytorch_model.eval()
            
            # 确定数据源
            using_real_data = sample_inputs is not None and len(sample_inputs) > 0
            if using_real_data:
                logger.info(f"使用 {len(sample_inputs)} 个真实样本进行数值验证")
            else:
                # 根据 InputSpec 生成合理分布的测试数据
                input_spec = self.config.input_spec
                if input_spec is not None:
                    logger.info("根据 InputSpec 生成模拟标准化后的测试数据")
                else:
                    logger.debug("使用随机数据进行数值验证（范围 [0, 1]）")
            
            def flatten_outputs(output):
                """递归展平嵌套的输出结构"""
                results = []
                if isinstance(output, torch.Tensor):
                    results.append(output.cpu().numpy())
                elif isinstance(output, np.ndarray):
                    results.append(output)
                elif isinstance(output, (list, tuple)):
                    for item in output:
                        results.extend(flatten_outputs(item))
                elif hasattr(output, 'cpu'):
                    # 其他有 cpu 方法的对象
                    results.append(output.cpu().numpy())
                elif hasattr(output, 'numpy'):
                    results.append(output.numpy())
                # 忽略其他类型 (如 None, dict 等)
                return results
            
            for i in range(self.config.num_test_samples):
                # 生成测试输入
                if using_real_data and i < len(sample_inputs):
                    test_input = sample_inputs[i].to(device)
                else:
                    # 根据 InputSpec 生成合理分布的数据
                    test_input = self._generate_test_input(input_shape, device)
                
                # PyTorch 推理
                with torch.no_grad():
                    pt_output = pytorch_model(test_input)
                
                # 处理多输出 (包括嵌套结构)
                pt_outputs = flatten_outputs(pt_output)
                
                if not pt_outputs:
                    logger.warning("PyTorch 模型没有产生有效输出")
                    continue
                
                # ORT 推理
                ort_input = {input_name: test_input.cpu().numpy()}
                ort_outputs = session.run(output_names, ort_input)
                
                # 对比每个输出
                for pt_out, ort_out in zip(pt_outputs, ort_outputs):
                    # 展平以便计算
                    pt_flat = pt_out.flatten()
                    ort_flat = ort_out.flatten()
                    
                    # 最大差异
                    diff = np.abs(pt_flat - ort_flat)
                    max_diffs.append(np.max(diff))
                    mean_diffs.append(np.mean(diff))
                    
                    # 余弦相似度
                    pt_norm = np.linalg.norm(pt_flat)
                    ort_norm = np.linalg.norm(ort_flat)
                    if pt_norm > 0 and ort_norm > 0:
                        cosine = np.dot(pt_flat, ort_flat) / (pt_norm * ort_norm)
                        cosine_sims.append(cosine)
                    else:
                        cosine_sims.append(1.0)  # 全零情况
            
            # 汇总结果
            result.max_diff = float(np.max(max_diffs))
            result.mean_diff = float(np.mean(mean_diffs))
            result.cosine_sim = float(np.min(cosine_sims))
            
            # 智能判断是否通过
            # 策略：cosine_sim 是更可靠的指标，max_diff 可能受数值范围影响
            diff_ok = result.max_diff <= self.config.max_diff_threshold
            cosine_ok = result.cosine_sim >= self.config.cosine_threshold
            
            # 如果 cosine_sim 非常高 (> 0.9999)，即使 max_diff 稍大也认为通过
            cosine_excellent = result.cosine_sim >= 0.9999
            
            if diff_ok and cosine_ok:
                # 完全通过
                result.numerical_valid = True
                result.messages.append("✓ 数值验证通过")
            elif cosine_excellent:
                # cosine 非常好，max_diff 可能是数值范围问题
                result.numerical_valid = True
                result.messages.append(
                    f"✓ 数值验证通过 (cosine={result.cosine_sim:.6f}, "
                    f"max_diff={result.max_diff:.2e} 在可接受范围内)"
                )
                logger.info(f"  数值验证: cosine_sim={result.cosine_sim:.6f} (优秀), max_diff={result.max_diff:.2e}")
            elif cosine_ok and result.max_diff <= 0.01:
                # cosine 通过，max_diff 在 1% 以内，发出警告但通过
                result.numerical_valid = True
                result.messages.append(
                    f"⚠️ 数值验证通过 (有轻微差异): max_diff={result.max_diff:.2e}, "
                    f"cosine_sim={result.cosine_sim:.6f}"
                )
                logger.warning(f"  数值验证有轻微差异: max_diff={result.max_diff:.2e}, cosine={result.cosine_sim:.6f}")
            else:
                result.numerical_valid = False
                result.messages.append(
                    f"✗ 数值验证失败: max_diff={result.max_diff:.2e}, "
                    f"cosine_sim={result.cosine_sim:.6f}"
                )
                raise ValidationError(
                    "数值验证",
                    "PyTorch 与 ONNX Runtime 输出差异过大",
                    {
                        "max_diff": result.max_diff,
                        "threshold": self.config.max_diff_threshold,
                        "cosine_sim": result.cosine_sim,
                        "cosine_threshold": self.config.cosine_threshold,
                    }
                )
            
            return result
        
        finally:
            # ========== 修复：确保 ORT Session 被释放，避免文件锁定 ==========
            if session is not None:
                del session
                import gc
                gc.collect()
    
    def _get_device(self, model: nn.Module) -> torch.device:
        """获取模型设备"""
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device('cpu')
    
    def _generate_test_input(
        self, 
        input_shape: Tuple[int, ...], 
        device: torch.device
    ) -> torch.Tensor:
        """
        根据 InputSpec 生成合理分布的测试数据
        
        这个方法根据模型的预处理配置生成符合真实数据分布的测试输入，
        而不是简单地使用 [0, 1] 范围的随机数据。
        
        对于不同任务类型：
        - CLS (分类): 生成 ImageNet 标准化后的数据分布 (mean≈0, std≈1-2)
        - DET (YOLO): 生成 [0, 1] 范围的数据 (仅 /255 归一化)
        
        Args:
            input_shape: 输入形状 (B, C, H, W)
            device: 目标设备
            
        Returns:
            模拟标准化后的测试张量
        """
        input_spec = self.config.input_spec
        
        if input_spec is None:
            # 没有 InputSpec，使用默认的 [0, 1] 随机数据
            return torch.rand(input_shape, device=device)
        
        # 获取预处理参数
        normalize_mean = getattr(input_spec, 'normalize_mean', (0.0, 0.0, 0.0))
        normalize_std = getattr(input_spec, 'normalize_std', (1.0, 1.0, 1.0))
        
        # 检查是否需要标准化 (mean 不全为 0 或 std 不全为 1)
        needs_normalization = (
            any(m != 0.0 for m in normalize_mean) or 
            any(s != 1.0 for s in normalize_std)
        )
        
        if needs_normalization:
            # 生成模拟标准化后的数据
            # 假设原始图像像素值在 [0, 1] 范围内（/255 后）
            # 标准化公式: (pixel - mean) / std
            # 因此，标准化后的数据范围约为:
            #   最小值: (0 - max(mean)) / min(std)
            #   最大值: (1 - min(mean)) / max(std)
            
            # 对于 ImageNet: mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            # 标准化后范围约为 [-2.1, 2.6]
            
            mean = torch.tensor(normalize_mean, device=device).view(1, 3, 1, 1)
            std = torch.tensor(normalize_std, device=device).view(1, 3, 1, 1)
            
            # 生成 [0, 1] 范围的随机数据，模拟原始归一化后的图像
            raw_data = torch.rand(input_shape, device=device)
            
            # 应用标准化
            normalized_data = (raw_data - mean) / std
            
            logger.debug(
                f"生成标准化测试数据: mean={normalize_mean}, std={normalize_std}, "
                f"数据范围=[{normalized_data.min().item():.2f}, {normalized_data.max().item():.2f}]"
            )
            
            return normalized_data
        else:
            # 不需要标准化 (如 YOLO)，直接生成 [0, 1] 范围数据
            return torch.rand(input_shape, device=device)


# ==============================================================================
# 第7部分: 主导出器 (ModelExporter)
# ==============================================================================

class ModelExporter:
    """
    ONNX 模型导出器
    ===============
    
    整合动态轴配置、导出、简化和验证的完整流程。
    
    使用方法:
    ---------
    >>> config = ExportConfig(opset_version=17)
    >>> exporter = ModelExporter(config)
    >>> result = exporter.export(model, (1, 3, 640, 640), "model.onnx", task_type="det")
    >>> print(result.summary())
    """
    
    def __init__(self, config: ExportConfig = None):
        """
        初始化导出器
        
        Args:
            config: 导出配置，None 则使用默认配置
        """
        # 确保依赖
        ensure_dependencies()
        
        self.config = config or ExportConfig()
        
        # 初始化各组件
        self.axes_configurator = DynamicAxesConfigurator(self.config)
        self.export_engine = ONNXExportEngine(self.config)
        self.simplifier = ONNXSimplifier(self.config.simplify)
        self.validator = ONNXValidator(self.config.validation)
    
    def export(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        output_path: str = None,
        task_type: str = "cls",
        output_shapes: Dict[str, Tuple[int, ...]] = None,
    ) -> ExportResult:
        """
        导出 ONNX 模型
        
        Args:
            model: PyTorch 模型 (优先使用 Stage 3 优化后的模型)
            input_shape: 输入形状 (B, C, H, W)
            output_path: 输出路径，None 则自动生成
            task_type: 任务类型 (cls/det/seg)
            output_shapes: 输出形状字典 (用于推断输出动态轴)
            
        Returns:
            ExportResult
        """
        import onnx
        import time
        
        stats = ExportStats()
        
        # 验证输入
        self._validate_inputs(model, input_shape, task_type)
        
        # 生成输出路径
        if output_path is None:
            output_path = self._generate_output_path()
        
        # 确保输出目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: 配置动态轴
            logger.info("配置动态轴...")
            dynamic_axes_spec = self.axes_configurator.configure(
                task_type=task_type,
                input_shape=input_shape,
                input_names=self.config.input_names,
                output_names=self.config.output_names,
                output_shapes=output_shapes,
            )
            dynamic_axes = dynamic_axes_spec.to_onnx_format()
            
            # 详细记录动态轴配置
            if dynamic_axes:
                logger.info(f"动态轴配置: {dynamic_axes}")
                for input_name, axes in dynamic_axes.items():
                    axis_info = ", ".join([f"dim[{k}]='{v}'" for k, v in axes.items()])
                    logger.info(f"  {input_name}: {axis_info}")
            else:
                logger.info("使用静态形状（无动态轴）")
            
            # Step 2: 导出 ONNX
            print("  [DEBUG] Step 2: 开始ONNX导出...", flush=True)
            print(f"  [DEBUG] 模型类型: {type(model).__name__}", flush=True)
            print(f"  [DEBUG] 模型设备: {next(model.parameters()).device}", flush=True)
            print(f"  [DEBUG] 输入形状: {input_shape}", flush=True)
            print(f"  [DEBUG] 输出路径: {output_path}", flush=True)
            
            start_time = time.time()
            onnx_model, original_node_count = self.export_engine.export(
                model=model,
                input_shape=input_shape,
                output_path=output_path,
                dynamic_axes=dynamic_axes,
            )
            print("  [DEBUG] Step 2: ONNX导出完成", flush=True)
            stats.export_time_ms = (time.time() - start_time) * 1000
            stats.original_node_count = original_node_count
            
            # Step 3: 简化（保留动态轴）
            start_time = time.time()
            onnx_model, simplified_node_count = self.simplifier.simplify(
                onnx_model=onnx_model,
                input_shape=input_shape,
                dynamic_axes=dynamic_axes,
            )
            stats.simplify_time_ms = (time.time() - start_time) * 1000
            stats.simplified_node_count = simplified_node_count
            
            # 保存简化后的模型
            onnx.save(onnx_model, output_path)
            
            # Step 4: 验证
            start_time = time.time()
            validation_result = self.validator.validate(
                onnx_model=onnx_model,
                onnx_path=output_path,
                pytorch_model=model,
                input_shape=input_shape,
            )
            stats.validation_time_ms = (time.time() - start_time) * 1000
            
            # 统计文件大小
            stats.file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            
            # 最终验证：确认动态轴是否正确保留
            self._verify_dynamic_axes(onnx_model, dynamic_axes)
            
            return ExportResult(
                success=True,
                onnx_path=output_path,
                onnx_model=onnx_model,
                dynamic_axes_spec=dynamic_axes_spec,
                validation_result=validation_result,
                stats=stats,
                message="导出成功",
            )
            
        except ONNXExportError as e:
            # 清理可能的临时文件
            if output_path and os.path.exists(output_path):
                os.remove(output_path)
            
            return ExportResult(
                success=False,
                stats=stats,
                message=str(e),
            )
        
        except Exception as e:
            # 清理可能的临时文件
            if output_path and os.path.exists(output_path):
                os.remove(output_path)
            
            raise ONNXExportError(f"未预期的错误: {e}") from e
    
    def _validate_inputs(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        task_type: str,
    ):
        """验证输入参数"""
        # 检查模型
        if model is None:
            raise ExportConfigError("model", "模型不能为 None")
        
        if not isinstance(model, nn.Module):
            raise ExportConfigError("model", f"期望 nn.Module，得到 {type(model)}")
        
        # 检查输入形状
        if input_shape is None:
            raise ExportConfigError("input_shape", "输入形状不能为 None，必须显式指定")
        
        if len(input_shape) != 4:
            raise ExportConfigError(
                "input_shape",
                f"必须是 4D (B, C, H, W)，得到 {len(input_shape)}D: {input_shape}"
            )
        
        # 检查任务类型
        valid_tasks = ("cls", "det", "seg")
        if task_type not in valid_tasks:
            raise ExportConfigError(
                "task_type",
                f"必须是 {valid_tasks} 之一，得到 '{task_type}'"
            )
        
        # 检查 opset
        if self.config.opset_version < 9:
            raise ExportConfigError(
                "opset_version",
                f"opset 必须 >= 9，得到 {self.config.opset_version}"
            )
        
        # 检测可能导致 TensorRT 问题的模型类型
        self._check_model_compatibility(model)
    
    def _check_model_compatibility(self, model: nn.Module):
        """
        检测模型是否可能导致 TensorRT 转换问题
        
        某些模型（如 EfficientNet、MobileNet）使用深度可分离卷积，
        在使用动态 batch 导出 ONNX 时可能导致 TensorRT 解析失败。
        """
        model_name = model.__class__.__name__.lower()
        
        # 可能导致问题的模型类型
        problematic_models = [
            'efficientnet', 'mobilenet', 'mobilenetv2', 'mobilenetv3',
            'mnasnet', 'shufflenet', 'ghostnet', 'regnet',
        ]
        
        # 检查模型名称
        is_problematic = any(pm in model_name for pm in problematic_models)
        
        # 也检查模型是否包含大量深度卷积
        if not is_problematic:
            depthwise_count = 0
            total_conv_count = 0
            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    total_conv_count += 1
                    if module.groups == module.in_channels and module.groups > 1:
                        depthwise_count += 1
            
            # 如果深度卷积占比超过 20%，发出警告
            if total_conv_count > 0 and depthwise_count / total_conv_count > 0.2:
                is_problematic = True
        
        if is_problematic and self.config.enable_dynamic_batch:
            logger.warning("=" * 60)
            logger.warning("⚠️  检测到可能导致 TensorRT 转换问题的模型结构")
            logger.warning(f"   模型类型: {model.__class__.__name__}")
            logger.warning("   问题: 深度可分离卷积 + 动态 batch 可能导致 TensorRT 解析失败")
            logger.warning("   建议: ")
            logger.warning("     1. 在 config.yaml 中设置 stage4_export.dynamic_batch: false")
            logger.warning("     2. 或者在 Stage 5 使用 TensorRT 的动态形状配置")
            logger.warning("=" * 60)
    
    def _verify_dynamic_axes(
        self,
        onnx_model: Any,
        expected_dynamic_axes: Dict[str, Dict[int, str]] = None,
    ):
        """
        验证导出后的 ONNX 模型是否正确保留了动态轴
        
        Args:
            onnx_model: ONNX ModelProto
            expected_dynamic_axes: 预期的动态轴配置
        """
        if not onnx_model.graph.input:
            return
        
        # 检查每个输入的形状
        for graph_input in onnx_model.graph.input:
            input_name = graph_input.name
            shape = graph_input.type.tensor_type.shape
            
            dynamic_dims = []
            static_dims = []
            
            for i, dim in enumerate(shape.dim):
                if dim.HasField('dim_param') and dim.dim_param:
                    # 动态维度
                    dynamic_dims.append((i, dim.dim_param))
                elif dim.HasField('dim_value'):
                    # 静态维度
                    static_dims.append((i, dim.dim_value))
            
            # 记录实际的形状信息
            if dynamic_dims:
                dim_strs = []
                for i, dim in enumerate(shape.dim):
                    if dim.HasField('dim_param') and dim.dim_param:
                        dim_strs.append(f"'{dim.dim_param}'")
                    else:
                        dim_strs.append(str(dim.dim_value))
                shape_str = "[" + ", ".join(dim_strs) + "]"
                logger.info(f"✅ ONNX 输入 '{input_name}' 形状: {shape_str}")
                logger.info(f"   动态轴: {[f'dim[{i}]={name}' for i, name in dynamic_dims]}")
            else:
                dim_strs = [str(dim.dim_value) for dim in shape.dim]
                shape_str = "[" + ", ".join(dim_strs) + "]"
                logger.info(f"📌 ONNX 输入 '{input_name}' 形状: {shape_str} (静态)")
        
        # 检查输出的形状
        for graph_output in onnx_model.graph.output:
            output_name = graph_output.name
            shape = graph_output.type.tensor_type.shape
            
            dynamic_dims = []
            for i, dim in enumerate(shape.dim):
                if dim.HasField('dim_param') and dim.dim_param:
                    dynamic_dims.append((i, dim.dim_param))
            
            if dynamic_dims:
                logger.debug(f"ONNX 输出 '{output_name}' 动态轴: {dynamic_dims}")
    
    def _generate_output_path(self) -> str:
        """生成默认输出路径"""
        timestamp = torch.tensor(0).new_empty(1).uniform_().item()
        return f"model_{int(timestamp * 1e6)}.onnx"


# ==============================================================================
# 第8部分: 便捷函数与 CLI
# ==============================================================================

def export_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    output_path: str = None,
    task_type: str = "cls",
    opset_version: int = 17,
    enable_dynamic_batch: bool = True,
    enable_dynamic_hw: bool = None,
    enable_simplify: bool = True,
    enable_validation: bool = True,
    **kwargs,
) -> ExportResult:
    """
    导出 ONNX 模型 (便捷函数)
    
    Args:
        model: PyTorch 模型
        input_shape: 输入形状 (B, C, H, W)
        output_path: 输出路径
        task_type: 任务类型 (cls/det/seg)
        opset_version: ONNX opset 版本
        enable_dynamic_batch: 启用动态 batch
        enable_dynamic_hw: 启用动态 H/W (det/seg 默认开启)
        enable_simplify: 启用模型简化
        enable_validation: 启用验证
        **kwargs: 其他 ExportConfig 参数
        
    Returns:
        ExportResult
    """
    # 处理动态 H/W 默认值
    if enable_dynamic_hw is None:
        enable_dynamic_hw = task_type in ("det", "seg")
    
    config = ExportConfig(
        opset_version=opset_version,
        enable_dynamic_batch=enable_dynamic_batch,
        enable_dynamic_hw=enable_dynamic_hw,
        simplify=SimplifyConfig(enable_simplify=enable_simplify),
        validation=ValidationConfig(enable_validation=enable_validation),
        **kwargs,
    )
    
    exporter = ModelExporter(config)
    return exporter.export(
        model=model,
        input_shape=input_shape,
        output_path=output_path,
        task_type=task_type,
    )


def run_stage4_export(
    ctx,
    output_path: str = None,
) -> bool:
    """
    Stage 4: ONNX 导出 (Pipeline 集成)
    
    从 PipelineContext 获取模型和配置进行导出。
    
    Args:
        ctx: PipelineContext
        output_path: 输出路径 (可选)
        
    Returns:
        是否成功
    """
    # 获取模型 (优先使用优化后的模型)
    model = getattr(ctx, 'optimized_model', None) or ctx.model
    if model is None:
        logger.error("  ❌ 未找到模型，请先执行 Stage 1 和 Stage 3")
        return False
    
    # 获取输入形状
    input_shape = ctx.input_shape
    if not input_shape:
        logger.error("  ❌ 未找到输入形状")
        return False
    
    # 获取推荐的 opset (从 Stage 2)
    opset = getattr(ctx, 'opset', 17)
    if hasattr(ctx, 'analysis_report') and ctx.analysis_report:
        opset = ctx.analysis_report.advice.recommended_opset
    
    # 生成输出路径
    if output_path is None:
        model_name = getattr(ctx, 'model_name', 'model') or 'model'
        output_path = f"{model_name}.onnx"
    
    try:
        config = ExportConfig(
            opset_version=opset,
            enable_dynamic_batch=True,
            enable_dynamic_hw=ctx.task_type in ('det', 'seg'),
        )
        
        exporter = ModelExporter(config)
        result = exporter.export(
            model=model,
            input_shape=input_shape,
            output_path=output_path,
            task_type=ctx.task_type,
        )
        
        if result.success:
            ctx.onnx_path = result.onnx_path
            logger.info(result.summary())
            logger.info(f"  ✅ ONNX 导出成功: {result.onnx_path}")
            return True
        else:
            logger.error(f"  ❌ ONNX 导出失败: {result.message}")
            return False
            
    except Exception as e:
        logger.error(f"  ❌ ONNX 导出异常: {e}")
        import traceback
        traceback.print_exc()
        return False



# ==============================================================================
# CLI 入口
# ==============================================================================

def main():
    """命令行入口"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="🚀 Stage 4: ONNX 导出器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检查依赖
  python model_exporter.py --check-deps
  
  # 导出模型 (通常配合 main.py 使用)
  # python main.py pipeline -m model.pth -t cls -o model.onnx
        """
    )
    
    parser.add_argument("--check-deps", action="store_true", help="检查依赖")
    
    args = parser.parse_args()
    
    if args.check_deps:
        deps = check_dependencies()
        print("\n📦 依赖检查:")
        print("=" * 40)
        for pkg, available in deps.items():
            status = "✅ 已安装" if available else "❌ 未安装"
            print(f"  {pkg}: {status}")
        print("=" * 40)
        return 0
    
    parser.print_help()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())