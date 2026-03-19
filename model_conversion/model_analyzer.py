"""
🔍 Stage 2: 模型分析器 (Model Analyzer)
========================================

分析 PyTorch 模型的兼容性、性能和转换建议。

功能:
1. 算子兼容性检测 (ONNX/TensorRT/OpenVINO)
2. 性能分析 (FLOPs/参数量/内存估算)
3. 转换建议 (opset版本/精度/潜在问题)
4. 报告生成 (console/json/html)

使用方法:
---------
>>> from model_analyzer import ModelAnalyzer, analyze_model
>>> 
>>> # 方式1: 使用便捷函数
>>> report = analyze_model(model, input_shape=(1, 3, 224, 224))
>>> print(report)
>>>
>>> # 方式2: 使用分析器类
>>> analyzer = ModelAnalyzer()
>>> report = analyzer.analyze(model, input_shape=(1, 3, 224, 224))
>>> analyzer.print_report(report)
>>> analyzer.save_report(report, 'report.html')

模块结构:
---------
1. 数据结构 (DataStructures)
2. 算子兼容性检测器 (OpCompatibilityChecker)
3. 性能分析器 (ModelProfiler)
4. 转换建议器 (ConversionAdvisor)
5. 主分析器与报告生成 (ModelAnalyzer)
6. 便捷函数
7. 简化测试接口
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Set

import torch
import torch.nn as nn
from unified_logger import Logger, console, Timer


# ==================== 日志配置 ====================

logger = Logger.get("model_analyzer")


# ==============================================================================
# 第1部分: 数据结构 (DataStructures)
# ==============================================================================

class Severity(Enum):
    """问题严重程度"""
    ERROR = "error"      # 必须解决，否则无法转换
    WARNING = "warning"  # 建议解决，可能影响精度/性能
    INFO = "info"        # 提示信息


class RecommendedPrecision(Enum):
    """推荐精度模式"""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    
    @classmethod
    def from_string(cls, s: str) -> "RecommendedPrecision":
        """从字符串创建枚举"""
        mapping = {
            "fp32": cls.FP32,
            "fp16": cls.FP16,
            "int8": cls.INT8,
        }
        s_lower = s.lower().strip()
        if s_lower not in mapping:
            raise ValueError(f"Unknown precision: {s}")
        return mapping[s_lower]


class Backend(Enum):
    """目标后端"""
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"


@dataclass
class OpInfo:
    """算子信息"""
    name: str                    # 层名称 (如 layer1.conv1)
    op_type: str                 # 算子类型 (如 Conv2d)
    params: Dict[str, Any] = field(default_factory=dict)
    input_shapes: List[Tuple] = field(default_factory=list)
    output_shapes: List[Tuple] = field(default_factory=list)


@dataclass
class CompatibilityResult:
    """兼容性检查结果"""
    backend: str
    supported: bool
    unsupported_ops: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ProfilingResult:
    """性能分析结果"""
    total_flops: int             # 总 FLOPs
    total_params: int            # 总参数量
    total_memory_mb: float       # 估计内存 (MB)
    layer_details: List[Dict] = field(default_factory=list)
    flops_calculation_failed: bool = False  # FLOPs 计算是否失败
    flops_error_message: str = ""           # 失败原因


@dataclass
class Issue:
    """问题/建议"""
    severity: Severity
    message: str
    solution: str = ""


@dataclass
class ConversionAdvice:
    """转换建议"""
    recommended_opset: int
    recommended_precision: RecommendedPrecision  # 使用枚举类型
    issues: List[Issue] = field(default_factory=list)
    
    @property
    def precision_str(self) -> str:
        """获取精度字符串（向后兼容）"""
        return self.recommended_precision.value


@dataclass
class AnalysisReport:
    """完整分析报告"""
    model_name: str
    task_type: str
    input_shape: Tuple
    op_count: Dict[str, int]
    compatibility: Dict[str, CompatibilityResult]
    profiling: ProfilingResult
    advice: ConversionAdvice
    
    def __str__(self) -> str:
        """字符串表示"""
        return self._to_console()
    
    def _to_console(self) -> str:
        """控制台格式"""
        p = self.profiling
        a = self.advice
        
        lines = [
            "=" * 60,
            "📊 模型分析报告",
            "=" * 60,
            "",
            "📋 基本信息",
            f"  模型名称:  {self.model_name}",
            f"  任务类型:  {self.task_type}",
            f"  输入形状:  {self.input_shape}",
            "",
            "📦 算子统计 (Top 10)",
        ]
        
        for op, count in sorted(self.op_count.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"  {op}: {count}")
        
        # 复杂度评估
        complexity = self._assess_complexity()
        
        # FLOPs 显示
        if p.flops_calculation_failed:
            flops_str = "N/A (计算失败)"
            # 添加警告
            lines.extend([
                "",
                "⚠️ 警告: FLOPs 计算失败",
                f"  原因: {p.flops_error_message or '未知错误'}",
                "  建议: 检查模型是否包含不支持的动态操作",
            ])
        else:
            flops_str = f"{p.total_flops / 1e9:.2f} G"
        
        lines.extend([
            "",
            "⚡ 性能分析",
            f"  FLOPs:     {flops_str}",
            f"  参数量:    {p.total_params / 1e6:.2f} M",
            f"  内存估算:  {p.total_memory_mb:.1f} MB",
            f"  复杂度:    {complexity['level']}",
            "",
            "🔍 后端兼容性",
        ])
        
        for backend, result in self.compatibility.items():
            status = "✅" if result.supported else "❌"
            lines.append(f"  {backend}: {status}")
            if result.unsupported_ops:
                lines.append(f"    不支持: {', '.join(result.unsupported_ops)}")
            for warn in result.warnings[:2]:
                lines.append(f"    ⚠️ {warn}")
        
        lines.extend([
            "",
            "💡 转换建议",
            f"  推荐 opset:  {a.recommended_opset}",
            f"  推荐精度:    {a.recommended_precision.value}",
        ])
        
        if a.issues:
            lines.append("")
            lines.append("⚠️ 注意事项")
            for issue in a.issues:
                icon = {'error': '❌', 'warning': '⚠️', 'info': 'ℹ️'}.get(
                    issue.severity.value, 'ℹ️')
                lines.append(f"  {icon} {issue.message}")
                if issue.solution:
                    lines.append(f"     → {issue.solution}")
        
        lines.append("")
        lines.append("=" * 60)
        return '\n'.join(lines)
    
    def _assess_complexity(self) -> dict:
        """评估模型复杂度"""
        flops_g = self.profiling.total_flops / 1e9
        params_m = self.profiling.total_params / 1e6
        
        if flops_g < 1 and params_m < 5:
            level = "轻量级 (适合边缘部署)"
        elif flops_g < 10 and params_m < 50:
            level = "中等 (适合 GPU 推理)"
        elif flops_g < 100 and params_m < 200:
            level = "较重 (建议 FP16/INT8)"
        else:
            level = "重量级 (需要高端 GPU)"
        
        return {"level": level, "flops_g": flops_g, "params_m": params_m}


# ==============================================================================
# 第2部分: 算子兼容性检测器 (OpCompatibilityChecker)
# ==============================================================================

class OpCompatibilityChecker:
    """
    算子兼容性检查器
    ================
    检查模型算子在各目标后端的支持情况
    """
    
    # ONNX 算子支持 (op_type: min_opset)
    # 包含 PyTorch nn.Module 类名到 ONNX 算子的映射
    ONNX_SUPPORT = {
        # ===== 卷积相关 =====
        'Conv2d': 1, 'Conv1d': 1, 'Conv3d': 1,
        'ConvTranspose2d': 1, 'ConvTranspose1d': 1, 'ConvTranspose3d': 1,
        
        # ===== 归一化 =====
        'BatchNorm2d': 9, 'BatchNorm1d': 9, 'BatchNorm3d': 9,
        'LayerNorm': 17, 'GroupNorm': 18, 'InstanceNorm2d': 9,
        'InstanceNorm1d': 9, 'InstanceNorm3d': 9,
        
        # ===== 激活函数 =====
        'ReLU': 1, 'ReLU6': 1, 'LeakyReLU': 1, 'PReLU': 9,
        'GELU': 20, 'Sigmoid': 1, 'Softmax': 1, 'LogSoftmax': 1,
        'Tanh': 1, 'SiLU': 14, 'Hardswish': 14, 'Hardsigmoid': 14,
        'Mish': 9, 'ELU': 6, 'SELU': 6, 'Softplus': 1,
        'Hardtanh': 1, 'CELU': 12,
        
        # ===== 池化 =====
        'MaxPool2d': 1, 'MaxPool1d': 1, 'MaxPool3d': 1,
        'AvgPool2d': 1, 'AvgPool1d': 1, 'AvgPool3d': 1,
        'AdaptiveAvgPool2d': 1, 'AdaptiveMaxPool2d': 1,
        'AdaptiveAvgPool1d': 1, 'AdaptiveMaxPool1d': 1,
        'AdaptiveAvgPool3d': 1, 'AdaptiveMaxPool3d': 1,
        'LPPool2d': 2, 'LPPool1d': 2,
        
        # ===== 线性/全连接 =====
        'Linear': 1, 'Bilinear': 9,
        'Embedding': 1, 'EmbeddingBag': 1,
        
        # ===== Dropout =====
        'Dropout': 1, 'Dropout2d': 1, 'Dropout3d': 1,
        'AlphaDropout': 1, 'FeatureAlphaDropout': 1,
        
        # ===== 形状操作 =====
        'Flatten': 1, 'Unflatten': 1,
        'Identity': 1, 'Sequential': 1,
        'Concat': 1,      # torch.cat -> ONNX Concat
        'Split': 2,       # torch.split / torch.chunk
        'Squeeze': 1, 'Unsqueeze': 1,
        'Reshape': 5, 'View': 5,
        'Transpose': 1, 'Permute': 1,
        'Slice': 1, 'Gather': 1, 'Scatter': 9,
        'Expand': 8, 'Tile': 6,
        
        # ===== 上采样/下采样 =====
        'Upsample': 9, 
        'UpsamplingNearest2d': 9, 'UpsamplingBilinear2d': 11,
        'PixelShuffle': 11, 'PixelUnshuffle': 11,
        
        # ===== 填充 =====
        'ZeroPad2d': 1, 'ConstantPad2d': 1, 'ConstantPad1d': 1,
        'ReflectionPad2d': 1, 'ReflectionPad1d': 1,
        'ReplicationPad2d': 1, 'ReplicationPad1d': 1, 'ReplicationPad3d': 1,
        
        # ===== 注意力 =====
        'MultiheadAttention': 14,
        
        # ===== RNN =====
        'LSTM': 1, 'GRU': 1, 'RNN': 1,
        'LSTMCell': 1, 'GRUCell': 1, 'RNNCell': 1,
        
        # ===== 容器 (不产生算子，仅结构) =====
        'ModuleList': 1, 'ModuleDict': 1,
        'ParameterList': 1, 'ParameterDict': 1,
        
        # ===== 其他常用 =====
        'Softmax2d': 1,
        'CrossMapLRN2d': 1, 'LocalResponseNorm': 1,
    }
    
    # TensorRT 不支持的算子
    TRT_UNSUPPORTED = {
        'NonZero',           # 动态输出形状
        'Where',             # 条件动态 (TRT<9)
        'TopK',              # k 必须为常量
        'RoIAlign',          # 需要 Plugin
        'DeformConv2d',      # 需要 Plugin
        'ModulatedDeformConv',  # 需要 Plugin
        'NMSRotated',        # 需要 Plugin
    }
    
    # TensorRT 受限的算子
    TRT_LIMITED = {
        'GridSample': '仅支持 bilinear/nearest, align_corners=True',
        'LayerNorm': 'TRT<8.6 需要 Plugin，TRT>=8.6 原生支持',
        'GroupNorm': '建议分解为 InstanceNorm + 重塑，或使用 Plugin',
        'ScatterND': '动态索引受限，建议使用固定索引',
        'InstanceNorm': '动态 batch size 受限',
        'Resize': 'mode=cubic 需要 opset>=11',
        'MultiheadAttention': '建议拆分为基础算子以获得更好优化',
    }
    
    # OpenVINO 不支持的算子
    OV_UNSUPPORTED = {
        'DeformConv2d',
        'RoIAlign',
        'ModulatedDeformConv',
    }
    
    def check_all(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        backends: List[str] = None,
    ) -> Dict[str, CompatibilityResult]:
        """
        检查所有后端兼容性
        
        Args:
            model: PyTorch 模型
            input_shape: 输入形状 (B, C, H, W)
            backends: 要检查的后端列表
            
        Returns:
            各后端的兼容性结果
        """
        backends = backends or ['onnx', 'tensorrt', 'openvino']
        
        # 提取模型算子
        ops = self._extract_ops(model, input_shape)
        op_types = set(op.op_type for op in ops)
        
        results = {}
        for backend in backends:
            results[backend] = self._check_backend(op_types, backend)
        
        return results
    
    def _extract_ops(self, model: nn.Module,
                     input_shape: Tuple[int, ...]) -> List[OpInfo]:
        """提取模型中的算子"""
        ops = []
        
        for name, module in model.named_modules():
            if len(list(module.children())) > 0:
                continue  # 跳过容器模块
            
            op_type = module.__class__.__name__
            params = {}
            
            # 提取关键参数
            if hasattr(module, 'kernel_size'):
                params['kernel_size'] = module.kernel_size
            if hasattr(module, 'stride'):
                params['stride'] = module.stride
            if hasattr(module, 'in_channels'):
                params['in_channels'] = module.in_channels
            if hasattr(module, 'out_channels'):
                params['out_channels'] = module.out_channels
            if hasattr(module, 'in_features'):
                params['in_features'] = module.in_features
            if hasattr(module, 'out_features'):
                params['out_features'] = module.out_features
            
            ops.append(OpInfo(
                name=name,
                op_type=op_type,
                params=params,
            ))
        
        return ops
    
    def _check_backend(self, op_types: Set[str],
                       backend: str) -> CompatibilityResult:
        """检查单个后端"""
        unsupported = []
        warnings = []
        
        if backend == 'onnx':
            for op in op_types:
                if op not in self.ONNX_SUPPORT:
                    # 跳过基础/容器类型
                    if not self._is_trivial_op(op):
                        # 检查是否是已知的第三方安全模块
                        if not self._is_known_safe_module(op):
                            warnings.append(f"{op}: 未在标准支持列表中，可能需要验证导出")
        
        elif backend == 'tensorrt':
            for op in op_types:
                if op in self.TRT_UNSUPPORTED:
                    unsupported.append(op)
                elif op in self.TRT_LIMITED:
                    warnings.append(f"{op}: {self.TRT_LIMITED[op]}")
        
        elif backend == 'openvino':
            for op in op_types:
                if op in self.OV_UNSUPPORTED:
                    unsupported.append(op)
        
        return CompatibilityResult(
            backend=backend,
            supported=len(unsupported) == 0,
            unsupported_ops=unsupported,
            warnings=warnings,
        )
    
    def _is_trivial_op(self, op_type: str) -> bool:
        """是否是基础算子 (肯定支持)"""
        trivial = {
            'Identity', 'Sequential', 'ModuleList', 'ModuleDict', 
            'ParameterList', 'ParameterDict',
        }
        return op_type in trivial
    
    def _is_known_safe_module(self, op_type: str) -> bool:
        """是否是已知安全的第三方模块"""
        # timm 库的常用模块
        TIMM_SAFE = {
            'BatchNormAct2d', 'SelectAdaptivePool2d', 'ClassifierHead',
            'SqueezeExcite', 'EffectiveSEModule', 'SEModule',
            'InvertedResidual', 'DepthwiseSeparableConv',
            'EfficientNet', 'MobileNetV3', 'ResNet', 'VisionTransformer',
            'ConvBnAct', 'SeparableConv2d', 'MixedConv2d',
            'CondConv2d', 'StdConv2d', 'ScaledStdConv2d',
            'GatherExcite', 'GlobalContext', 'CBAM', 'EcaModule',
            'Attention', 'WindowAttention', 'MultiScaleAttention',
        }
        
        # ultralytics/YOLO 的模块
        YOLO_SAFE = {
            'Conv', 'DWConv', 'DWConvTranspose2d', 'Focus', 
            'GhostConv', 'GhostBottleneck',
            'SPP', 'SPPF', 'SPPFast',
            'Bottleneck', 'BottleneckCSP', 'C3', 'C2f', 'C3k2', 'C2PSA',
            'Concat', 'Detect', 'Segment', 'Pose', 'Classify',
            'DFL', 'Proto', 'Ensemble',
            'TransformerLayer', 'TransformerBlock',
            'RepConv', 'DownC', 'SPPCSPC',
        }
        
        # detectron2 的模块
        DETECTRON_SAFE = {
            'CNNBlockBase', 'Conv2d', 'FrozenBatchNorm2d',
            'ShapeSpec', 'Backbone', 'ResNetBlockBase',
        }
        
        return op_type in TIMM_SAFE or op_type in YOLO_SAFE or op_type in DETECTRON_SAFE
    
    def get_min_opset(self, model: nn.Module,
                      input_shape: Tuple[int, ...]) -> int:
        """获取模型需要的最低 ONNX opset"""
        ops = self._extract_ops(model, input_shape)
        
        min_opset = 11  # 默认最低
        for op in ops:
            required = self.ONNX_SUPPORT.get(op.op_type, 11)
            min_opset = max(min_opset, required)
        
        return min_opset


# ==============================================================================
# 第3部分: 性能分析器 (ModelProfiler)
# ==============================================================================

class ModelProfiler:
    """
    模型性能分析器
    ==============
    计算 FLOPs、参数量、内存估算
    """
    
    def profile(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        precision: str = 'fp32',
    ) -> ProfilingResult:
        """
        分析模型性能
        
        Args:
            model: PyTorch 模型
            input_shape: 输入形状 (B, C, H, W)
            precision: 精度 (fp32/fp16/int8)
            
        Returns:
            ProfilingResult
        """
        model.eval()
        
        # 计算参数量
        total_params, layer_params = self._count_params(model)
        
        # 计算 FLOPs
        total_flops, layer_flops, flops_failed, flops_error = self._count_flops(model, input_shape)
        
        # 估算内存
        total_memory = self._estimate_memory(model, input_shape, precision)
        
        # 逐层详情
        layer_details = self._build_layer_details(
            model, layer_params, layer_flops, precision
        )
        
        return ProfilingResult(
            total_flops=total_flops,
            total_params=total_params,
            total_memory_mb=total_memory,
            layer_details=layer_details,
            flops_calculation_failed=flops_failed,
            flops_error_message=flops_error,
        )
    
    def _count_params(self, model: nn.Module) -> Tuple[int, Dict[str, int]]:
        """统计参数量"""
        total = 0
        layer_params = {}
        
        for name, param in model.named_parameters():
            count = param.numel()
            total += count
            
            # 归属到层
            layer_name = name.rsplit('.', 1)[0] if '.' in name else name
            layer_params[layer_name] = layer_params.get(layer_name, 0) + count
        
        return total, layer_params
    
    def _count_flops(self, model: nn.Module,
                     input_shape: Tuple[int, ...]) -> Tuple[int, Dict[str, int]]:
        """计算 FLOPs"""
        total_flops = 0
        layer_flops = {}
        hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                nonlocal total_flops
                flops = self._calc_module_flops(module, input, output)
                layer_flops[name] = flops
                total_flops += flops
            return hook
        
        # 注册 hooks
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 叶子模块
                hooks.append(module.register_forward_hook(make_hook(name)))
        
        # 前向传播
        flops_failed = False
        flops_error = ""
        try:
            device = next(model.parameters()).device
            dummy_input = torch.randn(input_shape, device=device)
            with torch.no_grad():
                model(dummy_input)
        except Exception as e:
            flops_failed = True
            flops_error = str(e)
            logger.warning(f"FLOPs 计算时前向传播失败: {e}")
        finally:
            for hook in hooks:
                hook.remove()
        
        return total_flops, layer_flops, flops_failed, flops_error
    
    def _calc_module_flops(self, module: nn.Module,
                           input: Tuple, output: Any) -> int:
        """
        计算单个模块的 FLOPs
        
        注意: FLOPs = Floating Point Operations (浮点运算次数)
        对于乘加操作，通常计为 2 FLOPs (1 乘法 + 1 加法)
        """
        try:
            # ===== 卷积 =====
            if isinstance(module, nn.Conv2d):
                out_shape = output.shape  # (N, C_out, H_out, W_out)
                return (2 * module.in_channels * module.out_channels *
                        module.kernel_size[0] * module.kernel_size[1] *
                        out_shape[2] * out_shape[3] // module.groups)
            
            elif isinstance(module, nn.Conv1d):
                out_shape = output.shape  # (N, C_out, L_out)
                return (2 * module.in_channels * module.out_channels *
                        module.kernel_size[0] * out_shape[2] // module.groups)
            
            elif isinstance(module, nn.Conv3d):
                out_shape = output.shape
                return (2 * module.in_channels * module.out_channels *
                        module.kernel_size[0] * module.kernel_size[1] * module.kernel_size[2] *
                        out_shape[2] * out_shape[3] * out_shape[4] // module.groups)
            
            elif isinstance(module, nn.ConvTranspose2d):
                in_shape = input[0].shape
                return (2 * module.in_channels * module.out_channels *
                        module.kernel_size[0] * module.kernel_size[1] *
                        in_shape[2] * in_shape[3] // module.groups)
            
            # ===== 全连接 =====
            elif isinstance(module, nn.Linear):
                batch_size = input[0].shape[0] if len(input[0].shape) >= 2 else 1
                return 2 * module.in_features * module.out_features * batch_size
            
            # ===== 归一化 =====
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d)):
                # 均值 + 方差 + 归一化 + 缩放偏移 ≈ 4 ops per element
                return 4 * input[0].numel()
            
            elif isinstance(module, nn.LayerNorm):
                return 5 * input[0].numel()
            
            elif isinstance(module, (nn.GroupNorm, nn.InstanceNorm2d)):
                return 5 * input[0].numel()
            
            # ===== 激活函数 =====
            elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.PReLU)):
                return input[0].numel()
            
            elif isinstance(module, nn.GELU):
                # GELU 近似需要更多计算
                return 10 * input[0].numel()
            
            elif isinstance(module, (nn.Sigmoid, nn.Tanh)):
                return 4 * input[0].numel()
            
            elif isinstance(module, nn.SiLU):
                # SiLU = x * sigmoid(x)
                return 5 * input[0].numel()
            
            elif isinstance(module, (nn.Softmax, nn.LogSoftmax)):
                return 3 * input[0].numel()
            
            elif isinstance(module, (nn.Hardswish, nn.Hardsigmoid)):
                return 3 * input[0].numel()
            
            # ===== 池化 =====
            elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
                if isinstance(output, torch.Tensor):
                    kernel = module.kernel_size
                    if isinstance(kernel, int):
                        kernel = (kernel, kernel)
                    return output.numel() * kernel[0] * kernel[1]
                return 0
            
            elif isinstance(module, (nn.MaxPool1d, nn.AvgPool1d)):
                if isinstance(output, torch.Tensor):
                    kernel = module.kernel_size
                    if isinstance(kernel, int):
                        kernel = kernel
                    else:
                        kernel = kernel[0]
                    return output.numel() * kernel
                return 0
            
            elif isinstance(module, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
                if isinstance(output, torch.Tensor) and len(input) > 0:
                    in_h, in_w = input[0].shape[2], input[0].shape[3]
                    out_h, out_w = output.shape[2], output.shape[3]
                    avg_kernel = max((in_h // max(out_h, 1)) * (in_w // max(out_w, 1)), 1)
                    return output.numel() * avg_kernel
                return 0
            
            elif isinstance(module, (nn.AdaptiveAvgPool1d, nn.AdaptiveMaxPool1d)):
                if isinstance(output, torch.Tensor) and len(input) > 0:
                    in_len = input[0].shape[2]
                    out_len = output.shape[2]
                    avg_kernel = max(in_len // max(out_len, 1), 1)
                    return output.numel() * avg_kernel
                return 0
            
            # ===== 上采样 =====
            elif isinstance(module, nn.Upsample) or module.__class__.__name__ == 'Upsample':
                if isinstance(output, torch.Tensor):
                    mode = getattr(module, 'mode', 'nearest')
                    if mode == 'nearest':
                        return output.numel()
                    else:
                        # 双线性/三线性插值
                        return output.numel() * 8
                return 0
            
            # ===== 注意力 =====
            elif isinstance(module, nn.MultiheadAttention):
                if len(input) > 0 and isinstance(input[0], torch.Tensor):
                    seq_len = input[0].shape[0]
                    batch = input[0].shape[1]
                    embed_dim = module.embed_dim
                    num_heads = module.num_heads
                    
                    # Q, K, V 投影 + 输出投影
                    proj_flops = 4 * 2 * seq_len * batch * embed_dim * embed_dim
                    # QK^T + AV
                    attn_flops = 2 * 2 * batch * num_heads * seq_len * seq_len * (embed_dim // num_heads)
                    # Softmax
                    softmax_flops = 3 * batch * num_heads * seq_len * seq_len
                    
                    return proj_flops + attn_flops + softmax_flops
                return 0
            
            # ===== RNN =====
            elif isinstance(module, nn.LSTM):
                if len(input) > 0 and isinstance(input[0], torch.Tensor):
                    seq_len, batch, input_size = input[0].shape
                    hidden_size = module.hidden_size
                    num_layers = module.num_layers
                    bidirectional = 2 if module.bidirectional else 1
                    # 4 个门，每个门有 input 和 hidden 的矩阵乘法
                    flops_per_step = 4 * 2 * (input_size + hidden_size) * hidden_size
                    return seq_len * batch * num_layers * bidirectional * flops_per_step
                return 0
            
            elif isinstance(module, nn.GRU):
                if len(input) > 0 and isinstance(input[0], torch.Tensor):
                    seq_len, batch, input_size = input[0].shape
                    hidden_size = module.hidden_size
                    num_layers = module.num_layers
                    bidirectional = 2 if module.bidirectional else 1
                    # 3 个门
                    flops_per_step = 3 * 2 * (input_size + hidden_size) * hidden_size
                    return seq_len * batch * num_layers * bidirectional * flops_per_step
                return 0
            
            # ===== Embedding =====
            elif isinstance(module, nn.Embedding):
                # Embedding 本质是查表，不涉及计算
                return 0
            
            # ===== Dropout / Identity =====
            elif isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.Identity)):
                return 0
            
            # ===== Flatten =====
            elif isinstance(module, nn.Flatten):
                return 0
            
        except Exception as e:
            logger.debug(f"FLOPs 计算异常 ({module.__class__.__name__}): {e}")
        
        return 0
    
    def _estimate_memory(self, model: nn.Module,
                         input_shape: Tuple[int, ...],
                         precision: str) -> float:
        """
        估算推理内存 (MB)
        
        包括:
        1. 模型权重
        2. 输入张量
        3. 中间激活值（考虑峰值内存）
        """
        bytes_per_elem = {'fp32': 4, 'fp16': 2, 'int8': 1, 'bf16': 2}.get(precision, 4)
        
        # 1. 模型权重内存
        param_memory = sum(p.numel() for p in model.parameters()) * bytes_per_elem
        
        # 2. 输入张量内存
        input_elements = 1
        for dim in input_shape:
            input_elements *= dim
        input_memory = input_elements * bytes_per_elem
        
        # 3. 中间激活值内存
        activation_sizes = []
        hooks = []
        
        def capture_activation(module, input, output):
            """捕获输出张量大小"""
            if isinstance(output, torch.Tensor):
                activation_sizes.append(output.numel() * bytes_per_elem)
            elif isinstance(output, (tuple, list)):
                for o in output:
                    if isinstance(o, torch.Tensor):
                        activation_sizes.append(o.numel() * bytes_per_elem)
        
        # 只在叶子模块上注册 hook
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                hooks.append(module.register_forward_hook(capture_activation))
        
        activation_memory = 0
        try:
            device = next(model.parameters()).device
            dummy = torch.randn(input_shape, device=device)
            with torch.no_grad():
                model(dummy)
            
            if activation_sizes:
                # 取最大的几个激活值之和（模拟峰值内存）
                sorted_sizes = sorted(activation_sizes, reverse=True)
                top_k = min(3, len(sorted_sizes))
                activation_memory = sum(sorted_sizes[:top_k])
                
        except Exception as e:
            logger.debug(f"内存估算前向传播失败: {e}")
            activation_memory = param_memory * 0.5
        finally:
            for h in hooks:
                h.remove()
        
        # 4. 总内存 (MB)
        total_bytes = param_memory + input_memory + activation_memory
        return total_bytes / (1024 * 1024)
    
    def _build_layer_details(
        self,
        model: nn.Module,
        layer_params: Dict[str, int],
        layer_flops: Dict[str, int],
        precision: str,
    ) -> List[Dict]:
        """构建逐层详情"""
        details = []
        bytes_per_elem = {'fp32': 4, 'fp16': 2, 'int8': 1}.get(precision, 4)
        
        for name, module in model.named_modules():
            if len(list(module.children())) > 0:
                continue
            
            params = layer_params.get(name, 0)
            flops = layer_flops.get(name, 0)
            memory = params * bytes_per_elem / (1024 * 1024)
            
            details.append({
                'name': name,
                'op_type': module.__class__.__name__,
                'params': params,
                'flops': flops,
                'memory_mb': round(memory, 4),
            })
        
        return details


# ==============================================================================
# 第4部分: 转换建议器 (ConversionAdvisor)
# ==============================================================================

class ConversionAdvisor:
    """
    转换建议器
    ==========
    基于分析结果给出 opset、精度、优化建议
    """
    
    # opset 与后端的最佳匹配
    OPSET_RECOMMENDATIONS = {
        'tensorrt': 17,
        'openvino': 17,
        'onnxruntime': 18,
        'default': 17,
    }
    
    # 精度敏感的算子 (不建议 INT8)
    PRECISION_SENSITIVE_OPS = {
        'LayerNorm', 'Softmax', 'GELU', 'Sigmoid',
        'Add',  # 残差连接
        'MultiheadAttention',
    }
    
    def advise(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        compatibility: Dict[str, CompatibilityResult],
        profiling: ProfilingResult,
        target_backend: str = 'tensorrt',
    ) -> ConversionAdvice:
        """
        生成转换建议
        
        Args:
            model: PyTorch 模型
            input_shape: 输入形状
            compatibility: 兼容性检查结果
            profiling: 性能分析结果
            target_backend: 目标后端
            
        Returns:
            ConversionAdvice
        """
        issues = []
        
        # 1. 推荐 opset
        recommended_opset = self._recommend_opset(
            model, input_shape, compatibility, target_backend
        )
        
        # 2. 推荐精度
        recommended_precision = self._recommend_precision(
            model, profiling, target_backend
        )
        
        # 3. 收集问题和建议
        issues.extend(self._check_compatibility_issues(compatibility, target_backend))
        issues.extend(self._check_performance_issues(profiling))
        issues.extend(self._check_model_issues(model))
        
        return ConversionAdvice(
            recommended_opset=recommended_opset,
            recommended_precision=recommended_precision,
            issues=issues,
        )
    
    def _recommend_opset(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        compatibility: Dict[str, CompatibilityResult],
        target_backend: str,
    ) -> int:
        """推荐 opset 版本"""
        checker = OpCompatibilityChecker()
        min_required = checker.get_min_opset(model, input_shape)
        
        backend_recommended = self.OPSET_RECOMMENDATIONS.get(
            target_backend,
            self.OPSET_RECOMMENDATIONS['default']
        )
        
        recommended = max(min_required, backend_recommended)
        return min(recommended, 18)
    
    def _recommend_precision(
        self,
        model: nn.Module,
        profiling: ProfilingResult,
        target_backend: str,
    ) -> RecommendedPrecision:
        """推荐量化精度"""
        # 检查敏感算子
        sensitive_count = 0
        total_count = 0
        
        for layer in profiling.layer_details:
            total_count += 1
            if layer['op_type'] in self.PRECISION_SENSITIVE_OPS:
                sensitive_count += 1
        
        sensitive_ratio = sensitive_count / max(total_count, 1)
        
        # 决策逻辑
        if sensitive_ratio > 0.3:
            return RecommendedPrecision.FP16
        elif profiling.total_params > 100_000_000:  # >100M 参数
            return RecommendedPrecision.INT8  # 大模型推荐 INT8
        elif profiling.total_flops > 10_000_000_000:  # >10G FLOPs
            return RecommendedPrecision.FP16
        else:
            return RecommendedPrecision.FP16
    
    def _check_compatibility_issues(
        self,
        compatibility: Dict[str, CompatibilityResult],
        target_backend: str,
    ) -> List[Issue]:
        """检查兼容性问题"""
        issues = []
        
        if target_backend in compatibility:
            result = compatibility[target_backend]
            
            if result.unsupported_ops:
                issues.append(Issue(
                    severity=Severity.ERROR,
                    message=f"目标后端 {target_backend} 不支持以下算子: {result.unsupported_ops}",
                    solution="需要替换为等效算子或使用自定义 Plugin",
                ))
            
            for warning in result.warnings:
                issues.append(Issue(
                    severity=Severity.WARNING,
                    message=warning,
                    solution="检查算子参数是否在支持范围内",
                ))
        
        return issues
    
    def _check_performance_issues(
        self,
        profiling: ProfilingResult,
    ) -> List[Issue]:
        """检查性能问题"""
        issues = []
        
        if profiling.total_memory_mb > 4096:  # >4GB
            issues.append(Issue(
                severity=Severity.WARNING,
                message=f"模型预估内存占用 {profiling.total_memory_mb:.0f}MB，可能超出显存限制",
                solution="考虑使用 FP16 或 INT8 量化减少内存占用",
            ))
        
        if profiling.total_flops > 100_000_000_000:  # >100G FLOPs
            issues.append(Issue(
                severity=Severity.INFO,
                message=f"模型计算量较大 ({profiling.total_flops/1e9:.1f}G FLOPs)",
                solution="考虑使用 TensorRT FP16 或 INT8 加速",
            ))
        
        return issues
    
    def _check_model_issues(self, model: nn.Module) -> List[Issue]:
        """检查模型结构问题"""
        issues = []
        custom_modules = set()
        
        # 安全的容器/包装类
        SAFE_WRAPPERS = {
            'Sequential', 'ModuleList', 'ModuleDict', 
            'ParameterList', 'ParameterDict',
            'Identity', 'Dropout', 'Dropout2d', 'Dropout3d',
        }
        
        # 已知可安全导出的第三方模块 (timm / ultralytics / detectron2)
        KNOWN_SAFE_MODULES = {
            # ===== timm 库模块 =====
            'BatchNormAct2d', 'SelectAdaptivePool2d', 'ClassifierHead',
            'SqueezeExcite', 'EffectiveSEModule', 'SEModule', 'EcaModule',
            'InvertedResidual', 'DepthwiseSeparableConv', 'ConvBnAct',
            'EfficientNet', 'MobileNetV3', 'ResNet', 'VisionTransformer',
            'SeparableConv2d', 'MixedConv2d', 'CondConv2d',
            'StdConv2d', 'ScaledStdConv2d', 'GatherExcite', 
            'GlobalContext', 'CBAM', 'Attention', 'WindowAttention',
            'MultiScaleAttention', 'PatchEmbed', 'Mlp', 'GatedMlp',
            'LayerScale', 'DropPath', 'SplitBatchNorm2d', 'GroupNormAct',
            
            # ===== ultralytics/YOLO 模块 =====
            'Conv', 'DWConv', 'DWConvTranspose2d', 'Focus', 
            'GhostConv', 'GhostBottleneck', 'RepConv',
            'SPP', 'SPPF', 'SPPFast', 'SPPCSPC',
            'Bottleneck', 'BottleneckCSP', 'C3', 'C2f', 'C3k2', 'C2PSA',
            'Concat', 'Detect', 'Segment', 'Pose', 'Classify', 'OBB',
            'DFL', 'Proto', 'Ensemble', 'DownC',
            'TransformerLayer', 'TransformerBlock', 'C3TR',
            'CrossConv', 'Sum', 'Contract', 'Expand',
            
            # ===== detectron2 模块 =====
            'CNNBlockBase', 'FrozenBatchNorm2d', 'ShapeSpec',
            'Backbone', 'ResNetBlockBase', 'BasicStem', 'BottleneckBlock',
        }
        
        for name, module in model.named_modules():
            # 只检测叶子模块
            if len(list(module.children())) > 0:
                continue
                
            class_name = module.__class__.__name__
            module_path = module.__class__.__module__
            
            # 跳过标准 PyTorch 模块
            if module_path.startswith(('torch.nn', 'torchvision')):
                continue
            
            # 跳过安全的包装类
            if class_name in SAFE_WRAPPERS:
                continue
            
            # 跳过已知可安全导出的模块
            if class_name in KNOWN_SAFE_MODULES:
                continue
            
            # 记录真正未知的自定义模块
            custom_modules.add(class_name)
        
        if custom_modules:
            unique_custom = list(custom_modules)[:5]
            issues.append(Issue(
                severity=Severity.INFO,
                message=f"检测到自定义模块: {unique_custom}",
                solution="确保这些模块的 forward 方法可以被 ONNX 追踪",
            ))
        
        return issues


# ==============================================================================
# 第5部分: 主分析器与报告生成 (ModelAnalyzer)
# ==============================================================================

class ModelAnalyzer:
    """
    模型分析器
    ==========
    
    功能:
    1. 算子兼容性检测 (ONNX/TensorRT/OpenVINO)
    2. 性能分析 (FLOPs/参数量/内存)
    3. 转换建议 (opset/精度/问题预警)
    4. 报告生成 (console/json/html)
    
    使用:
    -----
    >>> analyzer = ModelAnalyzer()
    >>> report = analyzer.analyze(model, input_shape=(1,3,224,224))
    >>> analyzer.print_report(report)
    >>> analyzer.save_report(report, 'report.html')
    """
    
    def __init__(self):
        self.compat_checker = OpCompatibilityChecker()
        self.profiler = ModelProfiler()
        self.advisor = ConversionAdvisor()
    
    # ==================== 分析 ====================
    
    def analyze(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        task_type: str = 'cls',
        model_name: str = None,
        target_backends: List[str] = None,
    ) -> AnalysisReport:
        """
        分析模型
        
        Args:
            model: PyTorch 模型
            input_shape: 输入形状 (B, C, H, W)
            task_type: 任务类型 (cls/det/seg)
            model_name: 模型名称
            target_backends: 目标后端列表
            
        Returns:
            AnalysisReport
        """
        target_backends = target_backends or ['onnx', 'tensorrt', 'openvino']
        model_name = model_name or model.__class__.__name__
        
        logger.debug(f"  分析模型: {model_name}, 输入: {input_shape}")
        
        # 1. 算子兼容性
        compatibility = self.compat_checker.check_all(
            model, input_shape, target_backends
        )
        
        # 2. 性能分析
        profiling = self.profiler.profile(model, input_shape)
        
        # 3. 转换建议
        advice = self.advisor.advise(
            model, input_shape, compatibility, profiling,
            target_backend=target_backends[0]
        )
        
        # 4. 统计算子
        op_count = self._count_ops(model)
        
        report = AnalysisReport(
            model_name=model_name,
            task_type=task_type,
            input_shape=input_shape,
            op_count=op_count,
            compatibility=compatibility,
            profiling=profiling,
            advice=advice,
        )
        
        return report
    
    def _count_ops(self, model: nn.Module) -> Dict[str, int]:
        """统计算子数量"""
        op_count = {}
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                op_type = module.__class__.__name__
                op_count[op_type] = op_count.get(op_type, 0) + 1
        return op_count
    
    # ==================== 报告生成 ====================
    
    def print_report(self, report: AnalysisReport, format: str = 'console'):
        """打印报告"""
        print(self.generate_report(report, format))
    
    def save_report(self, report: AnalysisReport,
                    path: str, format: str = None):
        """保存报告到文件"""
        if format is None:
            if path.endswith('.json'):
                format = 'json'
            elif path.endswith('.html'):
                format = 'html'
            else:
                format = 'console'
        
        content = self.generate_report(report, format)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"📄 报告已保存: {path}")
    
    def generate_report(self, report: AnalysisReport,
                        format: str = 'console') -> str:
        """生成指定格式的报告"""
        if format == 'json':
            return self._report_to_json(report)
        elif format == 'html':
            return self._report_to_html(report)
        else:
            return str(report)
    
    def _report_to_json(self, report: AnalysisReport) -> str:
        """JSON 格式报告"""
        from dataclasses import asdict
        
        def convert(obj):
            if hasattr(obj, 'value'):
                return obj.value
            elif hasattr(obj, '__dataclass_fields__'):
                return {k: convert(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(i) for i in obj]
            return obj
        
        data = {
            'model_name': report.model_name,
            'task_type': report.task_type,
            'input_shape': report.input_shape,
            'op_count': report.op_count,
            'compatibility': convert(report.compatibility),
            'profiling': convert(report.profiling),
            'advice': convert(report.advice),
            'generated_at': datetime.now().isoformat(),
        }
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    def _report_to_html(self, report: AnalysisReport) -> str:
        """HTML 格式报告"""
        p = report.profiling
        a = report.advice
        
        # 算子统计表格
        op_rows = ""
        for op, count in sorted(report.op_count.items(), key=lambda x: -x[1])[:15]:
            op_rows += f"<tr><td>{op}</td><td>{count}</td></tr>"
        
        # 兼容性表格
        compat_rows = ""
        for backend, result in report.compatibility.items():
            status = "✅" if result.supported else "❌"
            issues = ", ".join(result.unsupported_ops) or "-"
            warnings = "<br>".join(result.warnings[:2]) if result.warnings else "-"
            compat_rows += f"<tr><td>{backend}</td><td>{status}</td><td>{issues}</td><td style='font-size:12px'>{warnings}</td></tr>"
        
        # 问题列表
        issues_html = ""
        for issue in a.issues:
            color = {'error': '#e74c3c', 'warning': '#f39c12', 'info': '#3498db'}.get(
                issue.severity.value, '#95a5a6')
            issues_html += f"""
            <div style="border-left: 3px solid {color}; padding-left: 10px; margin: 10px 0;">
                <strong>{issue.message}</strong><br>
                <small style="color: #666;">→ {issue.solution}</small>
            </div>"""
        
        # 复杂度评估
        complexity = report._assess_complexity()
        
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>模型分析报告 - {report.model_name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 900px; margin: 0 auto; background: white; 
                     padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; margin-bottom: 30px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 8px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #3498db; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .metric {{ font-size: 28px; font-weight: bold; color: #3498db; }}
        .metric-box {{ display: inline-block; text-align: center; padding: 20px; 
                      margin: 10px; background: #ecf0f1; border-radius: 8px; min-width: 150px; }}
        .metric-label {{ font-size: 14px; color: #7f8c8d; }}
        .complexity {{ background: #e8f4fd; padding: 10px 20px; border-radius: 5px; 
                      display: inline-block; margin-top: 10px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>📊 模型分析报告</h1>
    
    <h2>📋 基本信息</h2>
    <table>
        <tr><th width="150">模型名称</th><td>{report.model_name}</td></tr>
        <tr><th>任务类型</th><td>{report.task_type}</td></tr>
        <tr><th>输入形状</th><td>{report.input_shape}</td></tr>
    </table>
    
    <h2>⚡ 性能指标</h2>
    <div style="text-align: center;">
        <div class="metric-box">
            <div class="metric">{p.total_flops/1e9:.2f}G</div>
            <div class="metric-label">FLOPs</div>
        </div>
        <div class="metric-box">
            <div class="metric">{p.total_params/1e6:.2f}M</div>
            <div class="metric-label">参数量</div>
        </div>
        <div class="metric-box">
            <div class="metric">{p.total_memory_mb:.1f}MB</div>
            <div class="metric-label">内存估算</div>
        </div>
    </div>
    <div style="text-align: center;">
        <div class="complexity">📊 复杂度等级: <strong>{complexity['level']}</strong></div>
    </div>
    
    <h2>📦 算子统计 (Top 15)</h2>
    <table>
        <tr><th>算子类型</th><th>数量</th></tr>
        {op_rows}
    </table>
    
    <h2>🔍 后端兼容性</h2>
    <table>
        <tr><th>后端</th><th>状态</th><th>不支持的算子</th><th>警告</th></tr>
        {compat_rows}
    </table>
    
    <h2>💡 转换建议</h2>
    <table>
        <tr><th width="150">推荐 opset</th><td>{a.recommended_opset}</td></tr>
        <tr><th>推荐精度</th><td>{a.recommended_precision.value}</td></tr>
    </table>
    
    {f'<h2>⚠️ 注意事项</h2>{issues_html}' if issues_html else ''}
    
    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; 
                   color: #95a5a6; font-size: 12px;">
        生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </footer>
</div>
</body>
</html>"""


# ==============================================================================
# 第6部分: 便捷函数
# ==============================================================================

def analyze_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    task_type: str = 'cls',
    model_name: str = None,
    target_backends: List[str] = None,
) -> AnalysisReport:
    """
    分析模型 (便捷函数)
    
    Args:
        model: PyTorch 模型
        input_shape: 输入形状 (B, C, H, W)
        task_type: 任务类型 (cls/det/seg)
        model_name: 模型名称
        target_backends: 目标后端列表
        
    Returns:
        AnalysisReport
    """
    return ModelAnalyzer().analyze(
        model=model,
        input_shape=input_shape,
        task_type=task_type,
        model_name=model_name,
        target_backends=target_backends,
    )


# ==============================================================================
# CLI 入口
# ==============================================================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="🔍 Stage 2: 模型分析器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析模型 (需要配合 model_importer 使用)
  # 请使用 main.py 进行完整分析
        """
    )
    
    args = parser.parse_args()
    parser.print_help()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())