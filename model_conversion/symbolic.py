"""
🔧 Stage 4: 自定义算子 Symbolic 注册 (symbolic.py)
==================================================

为 ONNX 导出提供自定义算子的 symbolic 函数实现。

功能:
1. Symbolic 注册基础设施 - 统一管理自定义算子注册
2. 激活函数 Symbolic - GELU, SiLU, Mish, Hardswish 等
3. 注意力机制 Symbolic - Scaled Dot-Product Attention
4. CV 专用算子 Symbolic - Deformable Conv, ROI Align, NMS 等
5. 用户扩展接口 - 支持用户自定义算子注册

使用方法:
---------
>>> from symbolic import SymbolicRegistry, register_all_symbolics
>>> 
>>> # 方式1: 一键注册所有内置 symbolic
>>> register_all_symbolics()
>>>
>>> # 方式2: 手动控制
>>> registry = SymbolicRegistry()
>>> registry.register_all()
>>> print(registry.list_registered())
>>>
>>> # 方式3: 注册用户自定义 symbolic
>>> from symbolic import register_user_symbolic
>>> register_user_symbolic("my_namespace::my_op", my_symbolic_fn)

模块结构:
---------
1. 注册基础设施 (SymbolicRegistry)
2. 激活函数 Symbolic
3. 注意力机制 Symbolic
4. CV 专用算子 Symbolic
5. 用户扩展接口
6. 管理与诊断工具

作者: Model Converter Team
版本: 1.0.0
"""

import logging
import threading
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Callable, Optional, Any, Tuple, Set

import torch
import torch.nn as nn

# ==================== 日志配置 ====================

logger = logging.getLogger(__name__)


# ==============================================================================
# 第1部分: 数据结构与异常
# ==============================================================================

class SymbolicCategory(Enum):
    """Symbolic 类别"""
    ACTIVATION = "activation"
    ATTENTION = "attention"
    CV_OPS = "cv_ops"
    NORMALIZATION = "normalization"
    USER_DEFINED = "user_defined"


@dataclass
class SymbolicInfo:
    """Symbolic 函数信息"""
    op_name: str                          # 算子名称 (如 "aten::gelu")
    symbolic_fn: Callable                 # symbolic 函数
    category: SymbolicCategory            # 类别
    min_opset: int = 9                    # 最低支持 opset
    description: str = ""                 # 描述
    

class SymbolicRegistrationError(Exception):
    """Symbolic 注册错误"""
    def __init__(self, op_name: str, reason: str):
        super().__init__(f"❌ Symbolic 注册失败 [{op_name}]: {reason}")
        self.op_name = op_name
        self.reason = reason


class SymbolicNotFoundError(Exception):
    """Symbolic 未找到"""
    def __init__(self, op_name: str, available: List[str] = None):
        msg = f"❌ 未找到 symbolic: {op_name}"
        if available:
            msg += f"\n可用的 symbolic: {', '.join(available[:10])}"
        super().__init__(msg)


# ==============================================================================
# 第2部分: 注册基础设施 (SymbolicRegistry)
# ==============================================================================

class SymbolicRegistry:
    """
    Symbolic 注册表
    ===============
    
    统一管理所有自定义算子的 symbolic 函数注册。
    
    使用方法:
    ---------
    >>> registry = SymbolicRegistry()
    >>> registry.register_builtin_symbolics()
    >>> registry.register_all()  # 注册到 torch.onnx
    """
    
    _instance = None
    _lock = threading.Lock()  # 线程安全锁
    
    def __new__(cls):
        """线程安全的单例模式"""
        if cls._instance is None:
            with cls._lock:
                # 双重检查锁定
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            self._registry: Dict[str, SymbolicInfo] = {}
            self._registered_to_torch: Set[str] = set()
            self._initialized = True
    
    def register(
        self,
        op_name: str,
        symbolic_fn: Callable,
        category: SymbolicCategory = SymbolicCategory.USER_DEFINED,
        min_opset: int = 9,
        description: str = "",
        overwrite: bool = False,
    ) -> bool:
        """
        注册 symbolic 函数
        
        Args:
            op_name: 算子名称 (如 "aten::gelu", "torchvision::nms")
            symbolic_fn: symbolic 函数
            category: 类别
            min_opset: 最低支持 opset
            description: 描述
            overwrite: 是否覆盖已有注册
            
        Returns:
            是否成功注册（已存在且不覆盖时返回False）
        """
        if op_name in self._registry and not overwrite:
            # 不抛出异常，只记录日志并跳过
            logger.debug(f"symbolic 已存在，跳过注册: {op_name}")
            return False
        
        self._registry[op_name] = SymbolicInfo(
            op_name=op_name,
            symbolic_fn=symbolic_fn,
            category=category,
            min_opset=min_opset,
            description=description,
        )
        logger.debug(f"注册 symbolic: {op_name}")
        return True
    
    def get(self, op_name: str) -> SymbolicInfo:
        """获取 symbolic 信息"""
        if op_name not in self._registry:
            raise SymbolicNotFoundError(op_name, list(self._registry.keys()))
        return self._registry[op_name]
    
    def has(self, op_name: str) -> bool:
        """检查是否已注册"""
        return op_name in self._registry
    
    def list_registered(self, category: SymbolicCategory = None) -> List[str]:
        """列出已注册的 symbolic"""
        if category is None:
            return list(self._registry.keys())
        return [
            name for name, info in self._registry.items()
            if info.category == category
        ]
    
    def register_to_torch(self, op_name: str, opset_version: int = 17, skip_if_registered: bool = True) -> bool:
        """
        将指定 symbolic 注册到 torch.onnx
        
        Args:
            op_name: 算子名称
            opset_version: opset 版本
            skip_if_registered: 如果已注册则跳过（避免重复注册警告）
            
        Returns:
            是否成功注册
        """
        if op_name not in self._registry:
            raise SymbolicNotFoundError(op_name)
        
        # 检查是否已注册，避免重复注册警告
        if skip_if_registered and op_name in self._registered_to_torch:
            logger.debug(f"已注册，跳过: {op_name}")
            return True
        
        info = self._registry[op_name]
        
        if opset_version < info.min_opset:
            logger.warning(
                f"opset {opset_version} < 最低要求 {info.min_opset}，"
                f"跳过注册 {op_name}"
            )
            return False
        
        try:
            # 解析命名空间和算子名
            if "::" in op_name:
                namespace, op = op_name.split("::", 1)
            else:
                namespace, op = "aten", op_name
            
            # 注册到 torch.onnx
            torch.onnx.register_custom_op_symbolic(
                f"{namespace}::{op}",
                info.symbolic_fn,
                opset_version
            )
            
            self._registered_to_torch.add(op_name)
            logger.debug(f"已注册到 torch.onnx: {op_name} (opset {opset_version})")
            return True
            
        except Exception as e:
            # 检查是否为"已存在"错误，如果是则视为成功
            error_msg = str(e).lower()
            if "already" in error_msg or "exist" in error_msg or "registered" in error_msg:
                self._registered_to_torch.add(op_name)
                logger.debug(f"算子已存在，标记为已注册: {op_name}")
                return True
            logger.warning(f"注册 {op_name} 到 torch.onnx 失败: {e}")
            return False
    
    def register_all(self, opset_version: int = 17) -> Dict[str, bool]:
        """
        将所有已注册的 symbolic 注册到 torch.onnx
        
        Args:
            opset_version: opset 版本
            
        Returns:
            {op_name: success} 的字典
        """
        results = {}
        for op_name in self._registry:
            results[op_name] = self.register_to_torch(op_name, opset_version)
        return results
    
    def unregister_all(self) -> None:
        """取消所有 torch.onnx 注册 (重置状态)"""
        self._registered_to_torch.clear()
        logger.debug("已清除所有 torch.onnx 注册状态")
    
    def get_registration_status(self) -> Dict[str, Dict[str, Any]]:
        """获取注册状态"""
        status = {}
        for op_name, info in self._registry.items():
            status[op_name] = {
                "category": info.category.value,
                "min_opset": info.min_opset,
                "registered_to_torch": op_name in self._registered_to_torch,
                "description": info.description,
            }
        return status
    
    def clear(self) -> None:
        """清空注册表"""
        self._registry.clear()
        self._registered_to_torch.clear()


# ==============================================================================
# 第3部分: 激活函数 Symbolic
# ==============================================================================

class ActivationSymbolics:
    """
    激活函数 Symbolic 实现
    ======================
    
    包含常见激活函数的 ONNX symbolic 实现。
    """
    
    @staticmethod
    def gelu_symbolic(g, input, approximate: str = 'none'):
        """
        GELU 激活函数的 symbolic 实现
        
        GELU(x) = x * Φ(x)
        其中 Φ(x) 是标准正态分布的累积分布函数
        
        Args:
            g: ONNX graph
            input: 输入张量
            approximate: 近似方法 ('none' 或 'tanh')
        """
        # 检查 approximate 参数类型
        if hasattr(approximate, 'node'):
            # 如果是 ONNX 节点，尝试获取其值
            approx_str = 'none'
        else:
            approx_str = str(approximate) if approximate else 'none'
        
        if approx_str == 'tanh':
            # 近似公式: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            sqrt_2_over_pi = g.op(
                "Constant", 
                value_t=torch.tensor(0.7978845608028654, dtype=torch.float32)
            )
            coeff = g.op(
                "Constant",
                value_t=torch.tensor(0.044715, dtype=torch.float32)
            )
            one = g.op("Constant", value_t=torch.tensor(1.0, dtype=torch.float32))
            half = g.op("Constant", value_t=torch.tensor(0.5, dtype=torch.float32))
            
            # x^3
            x_cubed = g.op("Pow", input, g.op("Constant", value_t=torch.tensor(3.0)))
            
            # x + 0.044715 * x^3
            inner = g.op("Add", input, g.op("Mul", coeff, x_cubed))
            
            # sqrt(2/pi) * (x + 0.044715 * x^3)
            tanh_arg = g.op("Mul", sqrt_2_over_pi, inner)
            
            # tanh(...)
            tanh_val = g.op("Tanh", tanh_arg)
            
            # 1 + tanh(...)
            one_plus_tanh = g.op("Add", one, tanh_val)
            
            # 0.5 * x * (1 + tanh(...))
            return g.op("Mul", g.op("Mul", half, input), one_plus_tanh)
        
        else:
            # 标准 GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
            sqrt_2 = g.op(
                "Constant", 
                value_t=torch.tensor(1.4142135623730951, dtype=torch.float32)
            )
            one = g.op("Constant", value_t=torch.tensor(1.0, dtype=torch.float32))
            half = g.op("Constant", value_t=torch.tensor(0.5, dtype=torch.float32))
            
            # x / sqrt(2)
            erf_arg = g.op("Div", input, sqrt_2)
            
            # erf(x / sqrt(2))
            erf_val = g.op("Erf", erf_arg)
            
            # 1 + erf(...)
            one_plus_erf = g.op("Add", one, erf_val)
            
            # 0.5 * (1 + erf(...))
            cdf = g.op("Mul", half, one_plus_erf)
            
            # x * cdf
            return g.op("Mul", input, cdf)
    
    @staticmethod
    def silu_symbolic(g, input):
        """
        SiLU (Swish) 激活函数的 symbolic 实现
        
        SiLU(x) = x * sigmoid(x)
        """
        sigmoid = g.op("Sigmoid", input)
        return g.op("Mul", input, sigmoid)
    
    @staticmethod
    def mish_symbolic(g, input):
        """
        Mish 激活函数的 symbolic 实现
        
        Mish(x) = x * tanh(softplus(x))
                = x * tanh(ln(1 + exp(x)))
        """
        # softplus(x) = ln(1 + exp(x))
        one = g.op("Constant", value_t=torch.tensor(1.0, dtype=torch.float32))
        exp_x = g.op("Exp", input)
        one_plus_exp = g.op("Add", one, exp_x)
        softplus = g.op("Log", one_plus_exp)
        
        # tanh(softplus(x))
        tanh_sp = g.op("Tanh", softplus)
        
        # x * tanh(softplus(x))
        return g.op("Mul", input, tanh_sp)
    
    @staticmethod
    def hardswish_symbolic(g, input):
        """
        HardSwish 激活函数的 symbolic 实现
        
        HardSwish(x) = x * HardSigmoid(x)
                     = x * clip((x + 3) / 6, 0, 1)
        """
        # 使用 ONNX HardSigmoid: alpha=1/6, beta=0.5
        # HardSigmoid(x) = max(0, min(1, alpha*x + beta))
        hardsigmoid = g.op("HardSigmoid", input, alpha_f=1.0/6.0, beta_f=0.5)
        return g.op("Mul", input, hardsigmoid)
    
    @staticmethod
    def hardsigmoid_symbolic(g, input):
        """
        HardSigmoid 激活函数的 symbolic 实现
        
        HardSigmoid(x) = clip((x + 3) / 6, 0, 1)
        """
        return g.op("HardSigmoid", input, alpha_f=1.0/6.0, beta_f=0.5)
    
    @staticmethod
    def leaky_relu_symbolic(g, input, negative_slope=0.01):
        """
        LeakyReLU 激活函数的 symbolic 实现
        """
        # 获取 negative_slope 的实际值
        if hasattr(negative_slope, 'node'):
            slope = 0.01  # 默认值
        else:
            slope = float(negative_slope)
        
        return g.op("LeakyRelu", input, alpha_f=slope)
    
    @classmethod
    def register_all(cls, registry: SymbolicRegistry) -> None:
        """注册所有激活函数 symbolic"""
        
        registry.register(
            "aten::gelu",
            cls.gelu_symbolic,
            category=SymbolicCategory.ACTIVATION,
            min_opset=9,  # 需要 Erf
            description="GELU 激活函数"
        )
        
        registry.register(
            "aten::silu",
            cls.silu_symbolic,
            category=SymbolicCategory.ACTIVATION,
            min_opset=9,
            description="SiLU (Swish) 激活函数"
        )
        
        registry.register(
            "aten::mish",
            cls.mish_symbolic,
            category=SymbolicCategory.ACTIVATION,
            min_opset=9,
            description="Mish 激活函数"
        )
        
        registry.register(
            "aten::hardswish",
            cls.hardswish_symbolic,
            category=SymbolicCategory.ACTIVATION,
            min_opset=14,  # HardSigmoid opset 14 更稳定
            description="HardSwish 激活函数"
        )
        
        registry.register(
            "aten::hardsigmoid",
            cls.hardsigmoid_symbolic,
            category=SymbolicCategory.ACTIVATION,
            min_opset=14,
            description="HardSigmoid 激活函数"
        )


# ==============================================================================
# 第4部分: 注意力机制 Symbolic
# ==============================================================================

class AttentionSymbolics:
    """
    注意力机制 Symbolic 实现
    ========================
    
    包含注意力机制相关算子的 ONNX symbolic 实现。
    """
    
    @staticmethod
    def scaled_dot_product_attention_symbolic(
        g, 
        query, 
        key, 
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,    # PyTorch 2.2+ 新增
        *args,               # 兼容未来可能的新参数
        **kwargs,
    ):
        """
        Scaled Dot-Product Attention 的 symbolic 实现
        
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
        
        支持 PyTorch 2.0/2.1/2.2+ 的所有参数签名:
        - PyTorch 2.0: (q, k, v, attn_mask, dropout_p, is_causal)
        - PyTorch 2.1: (q, k, v, attn_mask, dropout_p, is_causal, scale)
        - PyTorch 2.2+: (q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa)
        
        Args:
            g: ONNX graph
            query: [B, num_heads, seq_len, head_dim]
            key: [B, num_heads, seq_len, head_dim]
            value: [B, num_heads, seq_len, head_dim]
            attn_mask: 注意力掩码
            dropout_p: dropout 概率 (导出时忽略)
            is_causal: 是否使用因果掩码
            scale: 缩放因子
            enable_gqa: 是否启用 Grouped Query Attention (PyTorch 2.2+)
        """
        # 获取 head_dim 用于计算 scale
        # 注意: 在 symbolic 中无法直接获取形状，使用默认 scale
        scale_is_none = (
            scale is None or 
            (hasattr(scale, 'node') and scale.node().kind() == 'prim::Constant' and
             scale.node().output().type().kind() == 'NoneType')
        )
        
        if scale_is_none:
            # 使用 sqrt(64) 作为默认值，实际会在运行时正确计算
            default_scale = g.op(
                "Constant", 
                value_t=torch.tensor(0.125, dtype=torch.float32)  # 1/sqrt(64)
            )
            scale_factor = default_scale
        else:
            scale_factor = scale
        
        # K^T: 转置最后两个维度
        key_transposed = g.op("Transpose", key, perm_i=[0, 1, 3, 2])
        
        # Q @ K^T
        attn_weights = g.op("MatMul", query, key_transposed)
        
        # 缩放
        attn_weights = g.op("Mul", attn_weights, scale_factor)
        
        # 应用注意力掩码
        mask_is_none = (
            attn_mask is None or
            (hasattr(attn_mask, 'node') and 
             attn_mask.node().kind() == 'prim::Constant' and
             attn_mask.node().output().type().kind() == 'NoneType')
        )
        
        if not mask_is_none:
            attn_weights = g.op("Add", attn_weights, attn_mask)
        
        # Softmax
        attn_weights = g.op("Softmax", attn_weights, axis_i=-1)
        
        # 注意: dropout 在导出时不应用 (eval mode)
        # 注意: enable_gqa 在导出时不特殊处理，假设已经处理好了 Q/K/V 的形状
        
        # 输出: attn_weights @ V
        output = g.op("MatMul", attn_weights, value)
        
        return output
    
    @staticmethod
    def multi_head_attention_forward_symbolic(
        g,
        query,
        key, 
        value,
        embed_dim_to_check,
        num_heads,
        in_proj_weight,
        in_proj_bias,
        bias_k,
        bias_v,
        add_zero_attn,
        dropout_p,
        out_proj_weight,
        out_proj_bias,
        training=False,
        key_padding_mask=None,
        need_weights=False,
        attn_mask=None,
        use_separate_proj_weight=False,
        q_proj_weight=None,
        k_proj_weight=None,
        v_proj_weight=None,
        static_k=None,
        static_v=None,
        average_attn_weights=True,
        is_causal=False,
    ):
        """
        MultiHeadAttention forward 的 symbolic 实现
        
        这是一个复杂的算子，完整实现需要处理很多情况。
        这里提供基础实现，适用于大多数情况。
        """
        # 注意: 这个 symbolic 较复杂，PyTorch 通常会自动分解
        # 如果需要完整支持，建议升级 opset 或使用 torch.onnx.export 的内置支持
        
        # 基础实现: 使用线性变换 + scaled_dot_product_attention
        # 实际中 PyTorch >= 2.0 已经有较好的 ONNX 支持
        
        raise NotImplementedError(
            "MultiHeadAttention 的完整 symbolic 实现较复杂，"
            "建议使用 opset >= 14 或分解为基础算子"
        )
    
    @classmethod
    def register_all(cls, registry: SymbolicRegistry) -> None:
        """注册所有注意力机制 symbolic"""
        
        registry.register(
            "aten::scaled_dot_product_attention",
            cls.scaled_dot_product_attention_symbolic,
            category=SymbolicCategory.ATTENTION,
            min_opset=13,  # 需要较高 opset 支持
            description="Scaled Dot-Product Attention"
        )


# ==============================================================================
# 第5部分: CV 专用算子 Symbolic
# ==============================================================================

class CVOpsSymbolics:
    """
    CV 专用算子 Symbolic 实现
    =========================
    
    包含计算机视觉任务常用算子的 ONNX symbolic 实现。
    """
    
    @staticmethod
    def grid_sample_symbolic(
        g, 
        input, 
        grid, 
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    ):
        """
        grid_sample 的 symbolic 实现
        
        Args:
            input: [N, C, H_in, W_in]
            grid: [N, H_out, W_out, 2]
            mode: 'bilinear' | 'nearest' | 'bicubic'
            padding_mode: 'zeros' | 'border' | 'reflection'
            align_corners: bool
        """
        # 处理参数
        mode_str = str(mode) if not hasattr(mode, 'node') else 'bilinear'
        padding_str = str(padding_mode) if not hasattr(padding_mode, 'node') else 'zeros'
        align = bool(align_corners) if not hasattr(align_corners, 'node') else False
        
        # ONNX GridSample 参数映射
        mode_map = {'bilinear': 'linear', 'nearest': 'nearest', 'bicubic': 'cubic'}
        onnx_mode = mode_map.get(mode_str, 'linear')
        
        padding_map = {'zeros': 'zeros', 'border': 'border', 'reflection': 'reflection'}
        onnx_padding = padding_map.get(padding_str, 'zeros')
        
        return g.op(
            "GridSample",
            input,
            grid,
            align_corners_i=1 if align else 0,
            mode_s=onnx_mode,
            padding_mode_s=onnx_padding,
        )
    
    @staticmethod
    def interpolate_symbolic(
        g,
        input,
        size=None,
        scale_factor=None,
        mode='nearest',
        align_corners=None,
        recompute_scale_factor=None,
        antialias=False,
    ):
        """
        F.interpolate / nn.Upsample 的 symbolic 实现
        
        支持 nearest, linear, bilinear, bicubic, trilinear 模式
        正确处理动态形状的 Resize 算子
        """
        # 处理 mode 参数
        mode_str = str(mode) if not hasattr(mode, 'node') else 'nearest'
        
        # ONNX Resize 模式映射
        mode_map = {
            'nearest': 'nearest',
            'linear': 'linear', 
            'bilinear': 'linear',
            'bicubic': 'cubic',
            'trilinear': 'linear',
        }
        onnx_mode = mode_map.get(mode_str, 'nearest')
        
        # 坐标变换模式
        if align_corners and not hasattr(align_corners, 'node'):
            coord_mode = 'align_corners'
        else:
            coord_mode = 'half_pixel' if onnx_mode != 'nearest' else 'asymmetric'
        
        # 构建空的 roi
        empty_roi = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))
        
        # 检查是否有 output_size
        has_size = size is not None and not (
            hasattr(size, 'node') and 
            size.node().kind() == 'prim::Constant' and
            size.node().output().type().kind() == 'NoneType'
        )
        
        if has_size:
            # 使用 output size
            # 需要构建完整的 4D sizes tensor: [N, C, H_out, W_out]
            # 先获取输入的 N, C 维度
            input_shape = g.op("Shape", input)
            
            # 取前两个维度 [N, C]
            zero = g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64))
            two = g.op("Constant", value_t=torch.tensor([2], dtype=torch.int64))
            nc_dims = g.op("Slice", input_shape, zero, two, zero)
            
            # 确保 size 是正确的格式
            if hasattr(size, 'node') and size.node().kind() == 'prim::ListConstruct':
                # size 是列表，需要 Cast 为 int64
                size = g.op("Cast", size, to_i=7)  # 7 = int64
            
            # 拼接 [N, C] + [H_out, W_out]
            output_size = g.op("Concat", nc_dims, size, axis_i=0)
            
            empty_scales = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))
            
            return g.op(
                "Resize",
                input,
                empty_roi,
                empty_scales,
                output_size,
                coordinate_transformation_mode_s=coord_mode,
                mode_s=onnx_mode,
                nearest_mode_s='floor' if onnx_mode == 'nearest' else '',
            )
        else:
            # 使用 scale_factor
            # 需要构建完整的 4D scales tensor: [1, 1, scale_h, scale_w]
            
            # 检查 scale_factor 的类型
            if scale_factor is None or (
                hasattr(scale_factor, 'node') and 
                scale_factor.node().kind() == 'prim::Constant' and
                scale_factor.node().output().type().kind() == 'NoneType'
            ):
                # 默认 scale = 2
                scales = g.op("Constant", value_t=torch.tensor([1.0, 1.0, 2.0, 2.0], dtype=torch.float32))
            elif hasattr(scale_factor, 'node'):
                # scale_factor 是 graph 节点
                # 尝试提取常量值
                try:
                    sf_node = scale_factor.node()
                    if sf_node.kind() == 'prim::Constant':
                        # 获取常量值
                        sf_val = sf_node.output().toIValue()
                        if isinstance(sf_val, (int, float)):
                            scales = g.op("Constant", value_t=torch.tensor([1.0, 1.0, float(sf_val), float(sf_val)], dtype=torch.float32))
                        elif isinstance(sf_val, (list, tuple)) and len(sf_val) == 2:
                            scales = g.op("Constant", value_t=torch.tensor([1.0, 1.0, float(sf_val[0]), float(sf_val[1])], dtype=torch.float32))
                        else:
                            # 默认
                            scales = g.op("Constant", value_t=torch.tensor([1.0, 1.0, 2.0, 2.0], dtype=torch.float32))
                    elif sf_node.kind() == 'prim::ListConstruct':
                        # 是列表，需要在前面添加 [1, 1]
                        prefix = g.op("Constant", value_t=torch.tensor([1.0, 1.0], dtype=torch.float32))
                        sf_float = g.op("Cast", scale_factor, to_i=1)  # 1 = float32
                        scales = g.op("Concat", prefix, sf_float, axis_i=0)
                    else:
                        # 其他情况，假设是 2D tensor [scale_h, scale_w]
                        prefix = g.op("Constant", value_t=torch.tensor([1.0, 1.0], dtype=torch.float32))
                        sf_float = g.op("Cast", scale_factor, to_i=1)
                        scales = g.op("Concat", prefix, sf_float, axis_i=0)
                except Exception:
                    # 回退：使用默认 scale
                    scales = g.op("Constant", value_t=torch.tensor([1.0, 1.0, 2.0, 2.0], dtype=torch.float32))
            else:
                # Python 数值
                if isinstance(scale_factor, (int, float)):
                    scales = g.op("Constant", value_t=torch.tensor([1.0, 1.0, float(scale_factor), float(scale_factor)], dtype=torch.float32))
                elif isinstance(scale_factor, (list, tuple)) and len(scale_factor) == 2:
                    scales = g.op("Constant", value_t=torch.tensor([1.0, 1.0, float(scale_factor[0]), float(scale_factor[1])], dtype=torch.float32))
                else:
                    scales = g.op("Constant", value_t=torch.tensor([1.0, 1.0, 2.0, 2.0], dtype=torch.float32))
            
            # 空的 sizes
            empty_sizes = g.op("Constant", value_t=torch.tensor([], dtype=torch.int64))
            
            return g.op(
                "Resize",
                input,
                empty_roi,
                scales,
                empty_sizes,
                coordinate_transformation_mode_s=coord_mode,
                mode_s=onnx_mode,
                nearest_mode_s='floor' if onnx_mode == 'nearest' else '',
            )
    
    @staticmethod
    def roi_align_symbolic(
        g,
        input,
        rois,
        spatial_scale,
        pooled_height,
        pooled_width,
        sampling_ratio,
        aligned=False,
    ):
        """
        ROI Align 的 symbolic 实现
        
        Args:
            input: [N, C, H, W] 特征图
            rois: [num_rois, 5] (batch_idx, x1, y1, x2, y2)
            spatial_scale: 空间缩放因子
            pooled_height: 输出高度
            pooled_width: 输出宽度
            sampling_ratio: 采样比例
            aligned: 是否使用对齐模式
        """
        # 提取参数值
        scale = float(spatial_scale) if not hasattr(spatial_scale, 'node') else 1.0
        ph = int(pooled_height) if not hasattr(pooled_height, 'node') else 7
        pw = int(pooled_width) if not hasattr(pooled_width, 'node') else 7
        sr = int(sampling_ratio) if not hasattr(sampling_ratio, 'node') else 0
        align = bool(aligned) if not hasattr(aligned, 'node') else False
        
        # 分离 batch_indices 和 rois
        # ONNX RoiAlign 期望 rois 格式: [num_rois, 4] 和 batch_indices: [num_rois]
        batch_indices = g.op(
            "Cast",
            g.op("Slice", rois, 
                 g.op("Constant", value_t=torch.tensor([0])),
                 g.op("Constant", value_t=torch.tensor([1])),
                 g.op("Constant", value_t=torch.tensor([1]))),
            to_i=7  # int64
        )
        batch_indices = g.op("Squeeze", batch_indices, axes_i=[1])
        
        rois_coords = g.op("Slice", rois,
                          g.op("Constant", value_t=torch.tensor([1])),
                          g.op("Constant", value_t=torch.tensor([5])),
                          g.op("Constant", value_t=torch.tensor([1])))
        
        return g.op(
            "RoiAlign",
            input,
            rois_coords,
            batch_indices,
            output_height_i=ph,
            output_width_i=pw,
            sampling_ratio_i=sr,
            spatial_scale_f=scale,
            coordinate_transformation_mode_s='half_pixel' if align else 'output_half_pixel',
        )
    
    @staticmethod
    def nms_symbolic(
        g,
        boxes,
        scores,
        iou_threshold,
    ):
        """
        非极大值抑制 (NMS) 的 symbolic 实现
        
        Args:
            boxes: [num_boxes, 4]
            scores: [num_boxes]
            iou_threshold: IoU 阈值
        """
        # 获取阈值
        iou_thresh = float(iou_threshold) if not hasattr(iou_threshold, 'node') else 0.5
        
        # ONNX NMS 期望的输入格式:
        # boxes: [num_batches, num_classes, num_boxes, 4]
        # scores: [num_batches, num_classes, num_boxes]
        
        # 添加 batch 和 class 维度
        boxes_4d = g.op("Unsqueeze", boxes, axes_i=[0, 1])
        scores_3d = g.op("Unsqueeze", scores, axes_i=[0, 1])
        
        # 参数
        max_output = g.op("Constant", value_t=torch.tensor([10000], dtype=torch.int64))
        iou_th = g.op("Constant", value_t=torch.tensor([iou_thresh], dtype=torch.float32))
        score_th = g.op("Constant", value_t=torch.tensor([0.0], dtype=torch.float32))
        
        # 调用 ONNX NMS
        selected_indices = g.op(
            "NonMaxSuppression",
            boxes_4d,
            scores_3d,
            max_output,
            iou_th,
            score_th,
        )
        
        # 提取 box indices (第3列)
        return g.op("Slice", selected_indices,
                   g.op("Constant", value_t=torch.tensor([2])),
                   g.op("Constant", value_t=torch.tensor([3])),
                   g.op("Constant", value_t=torch.tensor([1])))
    
    @staticmethod
    def deformable_conv2d_symbolic(
        g,
        input,
        weight,
        offset,
        mask,
        bias,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        n_weight_grps,
        n_offset_grps,
        use_mask,
    ):
        """
        可变形卷积 (Deformable Conv2d) 的 symbolic 实现
        
        注意: ONNX 标准不直接支持 Deformable Conv，
        需要使用 custom op 或 TensorRT plugin
        """
        # 这个算子需要 TensorRT plugin 或自定义实现
        # 这里抛出明确错误
        raise NotImplementedError(
            "Deformable Conv2d 需要后端特定支持:\n"
            "  - TensorRT: 使用 DCNv2 plugin\n"
            "  - ONNX Runtime: 使用自定义算子\n"
            "建议在模型中使用标准卷积替代，或在后续阶段处理"
        )
    
    @staticmethod
    def upsample_nearest2d_symbolic(g, input, output_size, scales_h=None, scales_w=None):
        """
        aten::upsample_nearest2d 的 symbolic 实现
        
        专门处理 nn.Upsample(mode='nearest') 的 ONNX 导出
        
        Args:
            g: ONNX graph
            input: 输入 tensor [N, C, H, W]
            output_size: 输出尺寸 [H_out, W_out] 或 None
            scales_h: 高度缩放因子 (标量)
            scales_w: 宽度缩放因子 (标量)
        """
        empty_roi = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))
        
        # 检查是否有 output_size
        has_output_size = output_size is not None and not (
            hasattr(output_size, 'node') and 
            output_size.node().kind() == 'prim::Constant' and
            output_size.node().output().type().kind() == 'NoneType'
        )
        
        if has_output_size:
            # 使用 output_size
            # 需要构建完整的 4D sizes: [N, C, H_out, W_out]
            input_shape = g.op("Shape", input)
            
            # 取前两个维度 [N, C]
            zero = g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64))
            two = g.op("Constant", value_t=torch.tensor([2], dtype=torch.int64))
            nc_dims = g.op("Slice", input_shape, zero, two, zero)
            
            # output_size 转为 int64
            if hasattr(output_size, 'node') and output_size.node().kind() == 'prim::ListConstruct':
                output_size_i64 = g.op("Cast", output_size, to_i=7)  # 7 = int64
            else:
                output_size_i64 = output_size
            
            # 拼接 [N, C, H_out, W_out]
            full_size = g.op("Concat", nc_dims, output_size_i64, axis_i=0)
            
            empty_scales = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))
            
            return g.op(
                "Resize",
                input,
                empty_roi,
                empty_scales,
                full_size,
                coordinate_transformation_mode_s='asymmetric',
                mode_s='nearest',
                nearest_mode_s='floor',
            )
        else:
            # 使用 scales
            # 从 scales_h, scales_w 构建 4D scales: [1, 1, scales_h, scales_w]
            
            # 尝试提取标量值
            def get_scale_value(scale, default=2.0):
                if scale is None:
                    return default
                if hasattr(scale, 'node'):
                    try:
                        node = scale.node()
                        if node.kind() == 'prim::Constant':
                            val = node.output().toIValue()
                            if val is not None:
                                return float(val)
                    except (RuntimeError, AttributeError, TypeError):
                        pass
                    return default
                return float(scale) if isinstance(scale, (int, float)) else default
            
            sh = get_scale_value(scales_h, 2.0)
            sw = get_scale_value(scales_w, 2.0)
            
            scales = g.op("Constant", value_t=torch.tensor([1.0, 1.0, sh, sw], dtype=torch.float32))
            empty_sizes = g.op("Constant", value_t=torch.tensor([], dtype=torch.int64))
            
            return g.op(
                "Resize",
                input,
                empty_roi,
                scales,
                empty_sizes,
                coordinate_transformation_mode_s='asymmetric',
                mode_s='nearest',
                nearest_mode_s='floor',
            )
    
    @staticmethod
    def upsample_bilinear2d_symbolic(g, input, output_size, align_corners, scales_h=None, scales_w=None):
        """
        aten::upsample_bilinear2d 的 symbolic 实现
        
        专门处理 nn.Upsample(mode='bilinear') 的 ONNX 导出
        """
        empty_roi = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))
        
        # 坐标变换模式
        if align_corners and not hasattr(align_corners, 'node'):
            coord_mode = 'align_corners'
        elif hasattr(align_corners, 'node'):
            try:
                val = align_corners.node().output().toIValue()
                coord_mode = 'align_corners' if val else 'half_pixel'
            except (RuntimeError, AttributeError):
                coord_mode = 'half_pixel'
        else:
            coord_mode = 'half_pixel'
        
        # 检查是否有 output_size
        has_output_size = output_size is not None and not (
            hasattr(output_size, 'node') and 
            output_size.node().kind() == 'prim::Constant' and
            output_size.node().output().type().kind() == 'NoneType'
        )
        
        if has_output_size:
            # 使用 output_size
            input_shape = g.op("Shape", input)
            zero = g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64))
            two = g.op("Constant", value_t=torch.tensor([2], dtype=torch.int64))
            nc_dims = g.op("Slice", input_shape, zero, two, zero)
            
            if hasattr(output_size, 'node') and output_size.node().kind() == 'prim::ListConstruct':
                output_size_i64 = g.op("Cast", output_size, to_i=7)
            else:
                output_size_i64 = output_size
            
            full_size = g.op("Concat", nc_dims, output_size_i64, axis_i=0)
            empty_scales = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))
            
            return g.op(
                "Resize",
                input,
                empty_roi,
                empty_scales,
                full_size,
                coordinate_transformation_mode_s=coord_mode,
                mode_s='linear',
            )
        else:
            # 使用 scales
            def get_scale_value(scale, default=2.0):
                if scale is None:
                    return default
                if hasattr(scale, 'node'):
                    try:
                        node = scale.node()
                        if node.kind() == 'prim::Constant':
                            val = node.output().toIValue()
                            if val is not None:
                                return float(val)
                    except (RuntimeError, AttributeError, TypeError):
                        pass
                    return default
                return float(scale) if isinstance(scale, (int, float)) else default
            
            sh = get_scale_value(scales_h, 2.0)
            sw = get_scale_value(scales_w, 2.0)
            
            scales = g.op("Constant", value_t=torch.tensor([1.0, 1.0, sh, sw], dtype=torch.float32))
            empty_sizes = g.op("Constant", value_t=torch.tensor([], dtype=torch.int64))
            
            return g.op(
                "Resize",
                input,
                empty_roi,
                scales,
                empty_sizes,
                coordinate_transformation_mode_s=coord_mode,
                mode_s='linear',
            )
    
    @classmethod
    def register_all(cls, registry: SymbolicRegistry) -> None:
        """注册所有 CV 算子 symbolic"""
        
        registry.register(
            "aten::grid_sampler",
            cls.grid_sample_symbolic,
            category=SymbolicCategory.CV_OPS,
            min_opset=16,  # GridSample 在 opset 16 更完整
            description="Grid Sample (空间变换)"
        )
        
        # 注意: upsample_nearest2d 和 upsample_bilinear2d 使用 PyTorch 内置的 symbolic
        # 因为 PyTorch 内置的对动态形状支持更好
        # 如果遇到问题，可以取消下面的注释启用自定义实现
        #
        # registry.register(
        #     "aten::upsample_nearest2d",
        #     cls.upsample_nearest2d_symbolic,
        #     category=SymbolicCategory.CV_OPS,
        #     min_opset=11,
        #     description="Nearest 上采样"
        # )
        # 
        # registry.register(
        #     "aten::upsample_bilinear2d", 
        #     cls.upsample_bilinear2d_symbolic,
        #     category=SymbolicCategory.CV_OPS,
        #     min_opset=11,
        #     description="Bilinear 上采样"
        # )
        
        registry.register(
            "torchvision::roi_align",
            cls.roi_align_symbolic,
            category=SymbolicCategory.CV_OPS,
            min_opset=16,
            description="ROI Align"
        )
        
        registry.register(
            "torchvision::nms",
            cls.nms_symbolic,
            category=SymbolicCategory.CV_OPS,
            min_opset=11,
            description="Non-Maximum Suppression"
        )


# ==============================================================================
# 第6部分: 用户扩展接口
# ==============================================================================

class CustomSymbolicHandler(ABC):
    """
    自定义 Symbolic 处理器基类
    ==========================
    
    用户可以继承此类添加自定义算子的 symbolic 实现。
    
    使用方法:
    ---------
    >>> class MySymbolicHandler(CustomSymbolicHandler):
    ...     def register_custom_ops(self, registry):
    ...         registry.register("my_ns::my_op", self.my_op_symbolic, ...)
    ...     
    ...     def my_op_symbolic(self, g, input):
    ...         return g.op("MyOp", input)
    ...
    >>> handler = MySymbolicHandler()
    >>> handler.register_to(SymbolicRegistry())
    """
    
    @abstractmethod
    def register_custom_ops(self, registry: SymbolicRegistry) -> None:
        """
        注册自定义算子
        
        子类必须实现此方法
        
        Args:
            registry: Symbolic 注册表
        """
        pass
    
    def register_to(self, registry: SymbolicRegistry) -> None:
        """注册到指定注册表"""
        self.register_custom_ops(registry)


def register_user_symbolic(
    op_name: str,
    symbolic_fn: Callable,
    min_opset: int = 9,
    description: str = "",
) -> None:
    """
    注册用户自定义 symbolic (便捷函数)
    
    Args:
        op_name: 算子名称 (如 "my_namespace::my_op")
        symbolic_fn: symbolic 函数
        min_opset: 最低支持 opset
        description: 描述
        
    使用方法:
    ---------
    >>> def my_op_symbolic(g, input, param):
    ...     return g.op("MyCustomOp", input, param_i=param)
    ...
    >>> register_user_symbolic("my_ns::my_op", my_op_symbolic)
    """
    registry = SymbolicRegistry()
    registry.register(
        op_name=op_name,
        symbolic_fn=symbolic_fn,
        category=SymbolicCategory.USER_DEFINED,
        min_opset=min_opset,
        description=description,
    )


# ==============================================================================
# 第7部分: 管理与诊断工具
# ==============================================================================

# 全局单例注册表
_global_registry: Optional[SymbolicRegistry] = None


def register_builtin_symbolics(registry: SymbolicRegistry = None) -> SymbolicRegistry:
    """
    注册所有内置 symbolic
    
    使用全局单例模式，避免重复注册
    
    Args:
        registry: 注册表，None 则使用默认单例
        
    Returns:
        SymbolicRegistry
    """
    global _global_registry
    
    # 使用全局单例
    if registry is None:
        if _global_registry is not None:
            # 已经注册过，直接返回
            return _global_registry
        registry = SymbolicRegistry()
        _global_registry = registry
    
    # 注册各类 symbolic（重复注册会自动跳过）
    ActivationSymbolics.register_all(registry)
    AttentionSymbolics.register_all(registry)
    CVOpsSymbolics.register_all(registry)
    
    logger.info(f"已注册 {len(registry.list_registered())} 个内置 symbolic")
    
    return registry


def register_all_symbolics(opset_version: int = 17) -> Dict[str, bool]:
    """
    一键注册所有 symbolic 到 torch.onnx
    
    Args:
        opset_version: opset 版本
        
    Returns:
        {op_name: success} 的字典
    """
    registry = register_builtin_symbolics()
    return registry.register_all(opset_version)


def diagnose_unsupported_ops(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    opset_version: int = 17,
) -> Dict[str, List[str]]:
    """
    诊断模型中不支持的算子
    
    Args:
        model: PyTorch 模型
        input_shape: 输入形状
        opset_version: opset 版本
        
    Returns:
        {"unsupported": [...], "warnings": [...]}
    """
    import io
    from contextlib import redirect_stderr
    
    unsupported = []
    warnings_list = []
    
    # 创建示例输入
    device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
    dummy_input = torch.randn(input_shape, device=device)
    
    # 尝试导出并捕获错误
    model.eval()
    buffer = io.BytesIO()
    stderr_capture = io.StringIO()
    
    try:
        with redirect_stderr(stderr_capture):
            torch.onnx.export(
                model,
                dummy_input,
                buffer,
                opset_version=opset_version,
                do_constant_folding=False,
                verbose=False,
            )
    except Exception as e:
        error_msg = str(e)
        # 解析错误信息，提取不支持的算子
        if "Unsupported" in error_msg or "not supported" in error_msg.lower():
            unsupported.append(error_msg)
    
    # 解析警告
    stderr_content = stderr_capture.getvalue()
    if stderr_content:
        for line in stderr_content.split('\n'):
            if 'Warning' in line or 'warning' in line:
                warnings_list.append(line.strip())
    
    return {
        "unsupported": unsupported,
        "warnings": warnings_list,
    }


def check_op_support(op_name: str) -> Dict[str, Any]:
    """
    检查算子的支持情况
    
    Args:
        op_name: 算子名称
        
    Returns:
        支持信息字典
    """
    registry = SymbolicRegistry()
    
    result = {
        "op_name": op_name,
        "has_custom_symbolic": registry.has(op_name),
        "custom_info": None,
    }
    
    if registry.has(op_name):
        info = registry.get(op_name)
        result["custom_info"] = {
            "category": info.category.value,
            "min_opset": info.min_opset,
            "description": info.description,
        }
    
    return result


def get_supported_ops_summary() -> Dict[str, List[str]]:
    """
    获取所有支持的算子摘要
    
    Returns:
        按类别分组的算子列表
    """
    registry = register_builtin_symbolics()
    
    summary = {}
    for category in SymbolicCategory:
        ops = registry.list_registered(category)
        if ops:
            summary[category.value] = ops
    
    return summary



# ==============================================================================
# CLI 入口
# ==============================================================================

def main():
    """命令行入口"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="🔧 Stage 4: 自定义算子 Symbolic 注册",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 列出所有 symbolic
  python symbolic.py --list
  
  # 检查特定算子
  python symbolic.py --check aten::gelu
        """
    )
    
    parser.add_argument("--list", action="store_true", help="列出所有 symbolic")
    parser.add_argument("--check", type=str, help="检查特定算子支持")
    
    args = parser.parse_args()
    
    if args.list:
        summary = get_supported_ops_summary()
        print("\n📋 支持的算子列表")
        print("=" * 40)
        for category, ops in summary.items():
            print(f"\n🏷️ {category}:")
            for op in ops:
                print(f"   - {op}")
        return 0
    
    if args.check:
        info = check_op_support(args.check)
        print(f"\n🔍 算子支持检查: {args.check}")
        print("=" * 40)
        if info["has_custom_symbolic"]:
            print("  ✅ 有自定义 symbolic")
            ci = info["custom_info"]
            print(f"     类别: {ci['category']}")
            print(f"     最低 opset: {ci['min_opset']}")
            print(f"     描述: {ci['description']}")
        else:
            print("  ⚠️ 无自定义 symbolic (使用 PyTorch 默认)")
        return 0
    
    parser.print_help()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
