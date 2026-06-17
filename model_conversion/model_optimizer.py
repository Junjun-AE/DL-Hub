"""
🔧 Stage 3: 图优化器 (Graph Optimizer)
======================================

对 PyTorch 模型进行导出前优化，提升推理性能。

功能:
1. Conv+BN 融合 - 将 Conv + BatchNorm 合并为单个 Conv
2. Conv+BN+Act 融合 - 将 Conv + BatchNorm + 激活函数 合并
3. Linear+BN 融合 - 将 Linear + BatchNorm1d 合并
4. 冗余层移除 - 移除 Identity、Dropout 等

支持的激活函数:
- ReLU, ReLU6, LeakyReLU, PReLU
- SiLU (Swish), GELU, Mish
- Hardswish, Hardsigmoid

使用方法:
---------
>>> from model_optimizer import optimize_model, GraphOptimizer, OptimConfig
>>> 
>>> # 方式1: 便捷函数
>>> result = optimize_model(model, input_shape=(1, 3, 224, 224))
>>> optimized_model = result.model
>>> print(result.stats.summary())
>>>
>>> # 方式2: 完整控制
>>> config = OptimConfig(enable_conv_bn_act_fusion=True)
>>> optimizer = GraphOptimizer(config)
>>> result = optimizer.optimize(model, input_shape=(1, 3, 224, 224))

作者: Model Converter Team
版本: 1.1.0 (修复 YOLO 模型融合问题)
"""

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Type, Any, Union
from collections import OrderedDict

import torch
import torch.nn as nn
from unified_logger import Logger, console, Timer


# ==================== 日志配置 ====================

logger = Logger.get("model_optimizer")


# ==============================================================================
# 第1部分: 数据结构
# ==============================================================================

@dataclass
class OptimConfig:
    """
    优化配置
    
    控制启用哪些优化 Pass 以及验证参数
    """
    # 融合优化
    enable_conv_bn_fusion: bool = True        # Conv + BN 融合
    enable_conv_bn_act_fusion: bool = True    # Conv + BN + Act 融合
    enable_linear_bn_fusion: bool = True      # Linear + BN 融合
    
    # 冗余消除
    enable_identity_elimination: bool = True  # 移除 Identity
    enable_dropout_elimination: bool = True   # 移除 Dropout
    
    # 验证配置
    verify_output: bool = True                # 是否验证输出一致性
    verify_tolerance: float = 1e-3            # 验证容差 (Conv+BN 融合通常有 1e-4 ~ 1e-3 的误差)
    
    # 日志配置
    verbose: bool = False                     # 详细日志


@dataclass
class OptimizationStats:
    """
    优化统计
    
    记录优化前后的算子数量变化和各项优化的效果
    """
    # 算子统计
    original_op_count: Dict[str, int] = field(default_factory=dict)
    optimized_op_count: Dict[str, int] = field(default_factory=dict)
    
    # 融合统计
    conv_bn_fused: int = 0
    conv_bn_act_fused: int = 0
    linear_bn_fused: int = 0
    
    # 跳过统计（新增）
    conv_bn_skipped: int = 0  # 跳过的融合数（如 YOLO 自定义模块）
    
    # 消除统计
    identity_removed: int = 0
    dropout_removed: int = 0
    
    # 验证结果
    verification_passed: bool = True
    output_diff: float = 0.0
    
    # 失败原因追踪（新增）
    failed_operations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_failure(self, operation: str, reason: str) -> None:
        """
        记录失败的优化操作
        
        Args:
            operation: 操作名称
            reason: 失败原因
        """
        self.failed_operations.append(f"{operation}: {reason}")
    
    def add_warning(self, message: str) -> None:
        """
        记录警告信息
        
        Args:
            message: 警告消息
        """
        self.warnings.append(message)
    
    @property
    def has_failures(self) -> bool:
        """是否有失败的操作"""
        return len(self.failed_operations) > 0
    
    @property
    def total_fusions(self) -> int:
        """总融合数"""
        return self.conv_bn_fused + self.conv_bn_act_fused + self.linear_bn_fused
    
    @property
    def total_eliminations(self) -> int:
        """总消除数"""
        return self.identity_removed + self.dropout_removed
    
    def summary(self) -> str:
        """生成优化报告摘要（简洁版，单次输出）"""
        original_total = sum(self.original_op_count.values())
        optimized_total = sum(self.optimized_op_count.values())
        reduction = original_total - optimized_total
        reduction_pct = (reduction / original_total * 100) if original_total > 0 else 0
        
        lines = [
            "",
            "  📊 算子变化: {} → {} (减少 {} 个, {:.1f}%)".format(
                original_total, optimized_total, reduction, reduction_pct
            ),
            "  🔗 融合: Conv+BN {} 对, Conv+BN+Act {} 对, Linear+BN {} 对".format(
                self.conv_bn_fused, self.conv_bn_act_fused, self.linear_bn_fused
            ),
            "  🗑️ 移除: Identity {} 个, Dropout {} 个".format(
                self.identity_removed, self.dropout_removed
            ),
            "  ✅ 验证: {} (最大差异: {:.2e})".format(
                '通过' if self.verification_passed else '❌ 未通过',
                self.output_diff
            ),
        ]
        
        # 添加警告信息
        if self.warnings:
            lines.append("  ⚠️ 警告: {}".format(len(self.warnings)))
            for w in self.warnings[:3]:  # 只显示前3个
                lines.append(f"     - {w}")
            if len(self.warnings) > 3:
                lines.append(f"     ... 还有 {len(self.warnings) - 3} 个警告")
        
        # 添加失败信息
        if self.failed_operations:
            lines.append("  ❌ 失败操作: {}".format(len(self.failed_operations)))
            for f in self.failed_operations[:3]:  # 只显示前3个
                lines.append(f"     - {f}")
            if len(self.failed_operations) > 3:
                lines.append(f"     ... 还有 {len(self.failed_operations) - 3} 个失败")
        
        return '\n'.join(lines)


@dataclass
class OptimizeResult:
    """
    优化结果
    
    包含优化后的模型、统计信息和状态
    """
    model: nn.Module                          # 优化后的模型
    stats: OptimizationStats                  # 优化统计
    success: bool = True                      # 是否成功
    message: str = ""                         # 状态消息


# ==============================================================================
# 第2部分: 工具类 - ModuleUtils
# ==============================================================================

class ModuleUtils:
    """
    模块操作工具
    
    提供模型模块的增删改查功能
    """
    
    @staticmethod
    def get_module(model: nn.Module, name: str) -> Optional[nn.Module]:
        """
        根据名称获取模块
        
        Args:
            model: 模型
            name: 模块名称，如 'layer1.0.conv1'
            
        Returns:
            找到的模块，或 None
        """
        if not name:
            return model
        
        parts = name.split('.')
        module = model
        
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            elif hasattr(module, '_modules') and part in module._modules:
                module = module._modules[part]
            else:
                return None
        
        return module
    
    @staticmethod
    def set_module(model: nn.Module, name: str, new_module: nn.Module):
        """
        设置/替换模块
        
        Args:
            model: 模型
            name: 模块名称
            new_module: 新模块
        """
        parts = name.split('.')
        parent = model
        
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        setattr(parent, parts[-1], new_module)
    
    @staticmethod
    def del_module(model: nn.Module, name: str):
        """
        删除模块（替换为 Identity）
        
        Args:
            model: 模型
            name: 模块名称
        """
        ModuleUtils.set_module(model, name, nn.Identity())
    
    @staticmethod
    def get_parent_name(name: str) -> Tuple[str, str]:
        """
        获取父模块名称和子模块名称
        
        Args:
            name: 完整模块名称
            
        Returns:
            (父模块名称, 子模块名称)
        """
        parts = name.rsplit('.', 1)
        if len(parts) == 1:
            return '', parts[0]
        return parts[0], parts[1]
    
    @staticmethod
    def deep_copy(model: nn.Module) -> nn.Module:
        """
        深拷贝模型，带有容错机制
        
        Args:
            model: 原模型
            
        Returns:
            拷贝的模型
            
        Note:
            如果 deepcopy 失败（如 CUDA 张量锁等对象），
            将 fallback 到保存/加载 state_dict 的方式
        """
        try:
            return copy.deepcopy(model)
        except Exception as e:
            # Fallback: 通过 torch.save/load 复制
            import io
            import warnings
            warnings.warn(
                f"deepcopy failed ({e}), falling back to torch.save/load. "
                "Some non-persistent buffers may not be copied."
            )
            
            # 获取设备
            device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
            
            try:
                # 使用 torch.save/load 到内存缓冲区
                buffer = io.BytesIO()
                torch.save(model, buffer)
                buffer.seek(0)
                model_copy = torch.load(buffer, map_location=device, weights_only=False)
                return model_copy
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to copy model. deepcopy error: {e}, "
                    f"torch.save/load error: {e2}. "
                    "Consider manually creating a new model instance and copying state_dict."
                )
    
    @staticmethod
    def count_ops(model: nn.Module) -> Dict[str, int]:
        """
        统计模型中各类算子数量
        
        Args:
            model: 模型
            
        Returns:
            算子类型到数量的映射
        """
        op_count = {}
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 叶子模块
                op_type = module.__class__.__name__
                op_count[op_type] = op_count.get(op_type, 0) + 1
        return op_count
    
    @staticmethod
    def get_device(model: nn.Module) -> torch.device:
        """
        获取模型所在设备
        
        Args:
            model: 模型
            
        Returns:
            设备
        """
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device('cpu')


# ==============================================================================
# 第3部分: 融合优化器 - FusionOptimizer
# ==============================================================================

class FusionOptimizer:
    """
    融合优化器
    
    实现 Conv+BN、Conv+BN+Act、Linear+BN 的融合
    """
    
    # 支持融合的激活函数类型
    FUSABLE_ACTIVATIONS: Set[Type] = {
        nn.ReLU,
        nn.ReLU6,
        nn.LeakyReLU,
        nn.PReLU,
        nn.SiLU,        # Swish
        nn.GELU,
        nn.Mish,
        nn.Hardswish,
        nn.Hardsigmoid,
    }
    
    # 支持融合的卷积类型
    CONV_TYPES: Set[Type] = {nn.Conv1d, nn.Conv2d, nn.Conv3d}
    
    # 支持融合的 BatchNorm 类型
    BN_TYPES: Set[Type] = {nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d}
    
    # ========== 新增：需要跳过的自定义 Conv 模块类名 ==========
    # 这些模块内部有自己的 forward 逻辑，不能简单地替换子模块
    SKIP_CUSTOM_CONV_MODULES: Set[str] = {
        'Conv',          # ultralytics YOLO
        'DWConv',        # ultralytics YOLO  
        'ConvBN',        # 通用
        'ConvBnAct',     # 通用
        'ConvBNReLU',    # torchvision
        'ConvBNActivation',  # torchvision
        'ConvNormActivation',  # torchvision
        'Conv2dNormActivation',  # torchvision
        'DepthwiseSeparableConv',  # 通用
        'GhostConv',     # GhostNet
        'RepConv',       # YOLOv7
        'C2f',           # YOLOv8
        'C3',            # YOLOv5
        'Bottleneck',    # 通用（但需要小心）
    }
    
    def __init__(self, verbose: bool = False):
        """
        初始化融合优化器
        
        Args:
            verbose: 是否输出详细日志
        """
        self.verbose = verbose
        self.skipped_count = 0  # 记录跳过的数量
    
    # ---------------------- Conv + BN 融合 ----------------------
    
    def fuse_conv_bn(self, model: nn.Module) -> Tuple[nn.Module, int]:
        """
        融合 Conv + BatchNorm
        
        将连续的 Conv 和 BatchNorm 合并为单个 Conv，减少内存访问。
        
        数学原理:
        - BN: y = γ * (x - μ) / √(σ² + ε) + β
        - 融合: W_new = γ * W / √(σ² + ε), b_new = γ * (b - μ) / √(σ² + ε) + β
        
        Args:
            model: 输入模型
            
        Returns:
            (融合后的模型, 融合的对数)
        """
        fused_count = 0
        self.skipped_count = 0
        
        # 查找可融合的 Conv-BN 对
        fusable_pairs = self._find_conv_bn_pairs(model)
        
        for conv_name, bn_name in fusable_pairs:
            conv = ModuleUtils.get_module(model, conv_name)
            bn = ModuleUtils.get_module(model, bn_name)
            
            if conv is None or bn is None:
                continue
            
            # ========== 新增：检查是否在自定义模块内 ==========
            parent_name, _ = ModuleUtils.get_parent_name(conv_name)
            if parent_name:
                parent = ModuleUtils.get_module(model, parent_name)
                if parent is not None:
                    parent_class = parent.__class__.__name__
                    if parent_class in self.SKIP_CUSTOM_CONV_MODULES:
                        if self.verbose:
                            logger.debug(f"  跳过融合: {conv_name} 在自定义模块 {parent_class} 内")
                        self.skipped_count += 1
                        continue
            # ================================================
            
            # 执行融合
            fused_conv = self._fuse_conv_bn_weights(conv, bn)
            
            # 如果融合失败，跳过
            if fused_conv is None:
                continue
            
            # 替换模块
            ModuleUtils.set_module(model, conv_name, fused_conv)
            ModuleUtils.set_module(model, bn_name, nn.Identity())
            
            fused_count += 1
            
            if self.verbose:
                logger.debug(f"  融合: {conv_name} + {bn_name}")
        
        return model, fused_count
    
    def get_skipped_count(self) -> int:
        """获取跳过的融合数量"""
        return self.skipped_count
    
    def _find_conv_bn_pairs(self, model: nn.Module) -> List[Tuple[str, str]]:
        """
        查找可融合的 Conv-BN 对
        
        支持两种模式:
        1. Sequential 中连续的 Conv-BN
        2. 自定义模块中的 self.conv + self.bn
        """
        pairs = []
        
        for name, module in model.named_modules():
            # 模式1: Sequential
            if isinstance(module, nn.Sequential):
                pairs.extend(self._find_pairs_in_sequential(module, name))
            
            # 模式2: 检查模块的直接子模块
            # ========== 修改：跳过自定义 Conv 模块 ==========
            module_class = module.__class__.__name__
            if module_class not in self.SKIP_CUSTOM_CONV_MODULES:
                pairs.extend(self._find_pairs_in_module(module, name))
        
        return pairs
    
    def _find_pairs_in_sequential(self, seq: nn.Sequential, 
                                   prefix: str) -> List[Tuple[str, str]]:
        """在 Sequential 中查找连续的 Conv-BN 对"""
        pairs = []
        children = list(seq.named_children())
        
        for i in range(len(children) - 1):
            name1, mod1 = children[i]
            name2, mod2 = children[i + 1]
            
            if self._is_conv(mod1) and self._is_bn(mod2):
                # 检查通道数匹配
                if self._channels_match(mod1, mod2):
                    conv_name = f"{prefix}.{name1}" if prefix else name1
                    bn_name = f"{prefix}.{name2}" if prefix else name2
                    pairs.append((conv_name, bn_name))
        
        return pairs
    
    def _find_pairs_in_module(self, module: nn.Module, 
                               prefix: str) -> List[Tuple[str, str]]:
        """在模块中查找 conv/bn 属性对"""
        pairs = []
        
        # ========== 新增：跳过自定义 Conv 模块 ==========
        module_class = module.__class__.__name__
        if module_class in self.SKIP_CUSTOM_CONV_MODULES:
            return pairs
        # ================================================
        
        # 常见的命名模式
        conv_bn_patterns = [
            ('conv', 'bn'),
            ('conv', 'norm'),
            ('conv1', 'bn1'), ('conv2', 'bn2'), ('conv3', 'bn3'),
            ('dwconv', 'bn'), ('pwconv', 'bn'),
            ('conv_dw', 'bn1'), ('conv_pw', 'bn2'),
        ]
        
        for conv_attr, bn_attr in conv_bn_patterns:
            if hasattr(module, conv_attr) and hasattr(module, bn_attr):
                conv = getattr(module, conv_attr)
                bn = getattr(module, bn_attr)
                
                if self._is_conv(conv) and self._is_bn(bn):
                    if self._channels_match(conv, bn):
                        conv_name = f"{prefix}.{conv_attr}" if prefix else conv_attr
                        bn_name = f"{prefix}.{bn_attr}" if prefix else bn_attr
                        pairs.append((conv_name, bn_name))
        
        return pairs
    
    def _fuse_conv_bn_weights(self, conv: nn.Module, bn: nn.Module) -> nn.Module:
        """
        融合 Conv 和 BN 的权重
        
        Args:
            conv: 卷积层
            bn: BatchNorm 层
            
        Returns:
            融合后的卷积层，如果无法融合则返回 None
        """
        # 安全检查：确保 BN 有有效的 running stats
        if bn.running_mean is None or bn.running_var is None:
            logger.debug(f"  跳过融合: BN 没有有效的 running stats")
            return None
        
        # 安全检查：确保是标准的 BatchNorm（不是自定义的组合模块）
        if not isinstance(bn, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            logger.debug(f"  跳过融合: BN 类型不支持 {type(bn)}")
            return None
        
        # 获取 BN 参数
        gamma = bn.weight.data if bn.weight is not None else torch.ones_like(bn.running_mean)
        beta = bn.bias.data if bn.bias is not None else torch.zeros_like(bn.running_mean)
        mean = bn.running_mean.data
        var = bn.running_var.data
        eps = bn.eps
        
        # ========== 检查 BN 是否已经被融合（变成恒等变换） ==========
        identity_threshold = 1e-4
        
        mean_is_zero = mean.abs().max().item() < identity_threshold
        var_is_one = (var - 1.0).abs().max().item() < identity_threshold
        gamma_is_one = (gamma - 1.0).abs().max().item() < identity_threshold
        beta_is_zero = beta.abs().max().item() < identity_threshold
        
        if mean_is_zero and var_is_one and gamma_is_one and beta_is_zero:
            logger.debug(f"  跳过融合: BN 已是恒等变换（可能已被预融合）")
            return None
        
        if mean_is_zero and var_is_one:
            logger.debug(f"  跳过融合: BN mean=0/var=1，可能已被部分融合")
            return None
        # ===================================================================
        
        # 计算缩放因子
        std = torch.sqrt(var + eps)
        scale = gamma / std                       # γ / √(σ² + ε)
        
        # 确定卷积类型并创建融合后的卷积
        if isinstance(conv, nn.Conv1d):
            fused_conv = nn.Conv1d(
                conv.in_channels, conv.out_channels, conv.kernel_size,
                conv.stride, conv.padding, conv.dilation,
                conv.groups, True, conv.padding_mode
            )
            # Conv1d weight shape: [out_channels, in_channels/groups, kernel_size]
            scale_view = scale.view(-1, 1, 1)
        elif isinstance(conv, nn.Conv2d):
            fused_conv = nn.Conv2d(
                conv.in_channels, conv.out_channels, conv.kernel_size,
                conv.stride, conv.padding, conv.dilation,
                conv.groups, True, conv.padding_mode
            )
            # Conv2d weight shape: [out_channels, in_channels/groups, kH, kW]
            scale_view = scale.view(-1, 1, 1, 1)
        elif isinstance(conv, nn.Conv3d):
            fused_conv = nn.Conv3d(
                conv.in_channels, conv.out_channels, conv.kernel_size,
                conv.stride, conv.padding, conv.dilation,
                conv.groups, True, conv.padding_mode
            )
            # Conv3d weight shape: [out_channels, in_channels/groups, kD, kH, kW]
            scale_view = scale.view(-1, 1, 1, 1, 1)
        else:
            logger.debug(f"  跳过融合: 不支持的卷积类型 {type(conv)}")
            return None
        
        # 复制到相同设备
        fused_conv = fused_conv.to(conv.weight.device)
        
        # 融合权重: W_new = W * γ / √(σ² + ε)
        fused_conv.weight.data = conv.weight.data * scale_view
        
        # 融合偏置: b_new = γ * (b - μ) / √(σ² + ε) + β
        if conv.bias is not None:
            fused_conv.bias.data = (conv.bias.data - mean) * scale + beta
        else:
            fused_conv.bias.data = -mean * scale + beta
        
        return fused_conv
    
    # ---------------------- Conv + BN + Act 融合 ----------------------
    
    def fuse_conv_bn_act(self, model: nn.Module) -> Tuple[nn.Module, int]:
        """
        融合 Conv + BatchNorm + Activation
        
        注意: 激活函数本身不能被数学融合进卷积，
        但我们可以将它们放入一个高效的自定义模块中，
        或者在 Conv+BN 融合后简化结构。
        
        这里的实现是: 找到 Conv-BN-Act 三元组，融合 Conv-BN，
        然后用一个 Sequential(fused_conv, act) 替换。
        
        Args:
            model: 输入模型
            
        Returns:
            (融合后的模型, 融合的三元组数)
        """
        fused_count = 0
        
        # 查找 Conv-BN-Act 三元组
        triplets = self._find_conv_bn_act_triplets(model)
        
        for conv_name, bn_name, act_name in triplets:
            conv = ModuleUtils.get_module(model, conv_name)
            bn = ModuleUtils.get_module(model, bn_name)
            act = ModuleUtils.get_module(model, act_name)
            
            if conv is None or bn is None or act is None:
                continue
            
            # 先融合 Conv + BN
            fused_conv = self._fuse_conv_bn_weights(conv, bn)
            
            # 如果融合失败，跳过
            if fused_conv is None:
                continue
            
            # 创建融合模块 (Conv + Act)
            fused_module = ConvActFused(fused_conv, copy.deepcopy(act))
            
            # 替换模块
            ModuleUtils.set_module(model, conv_name, fused_module)
            ModuleUtils.set_module(model, bn_name, nn.Identity())
            ModuleUtils.set_module(model, act_name, nn.Identity())
            
            fused_count += 1
            
            if self.verbose:
                logger.debug(f"  融合: {conv_name} + {bn_name} + {act_name}")
        
        return model, fused_count
    
    def _find_conv_bn_act_triplets(self, model: nn.Module) -> List[Tuple[str, str, str]]:
        """查找 Conv-BN-Act 三元组"""
        triplets = []
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential):
                triplets.extend(self._find_triplets_in_sequential(module, name))
        
        return triplets
    
    def _find_triplets_in_sequential(self, seq: nn.Sequential, 
                                      prefix: str) -> List[Tuple[str, str, str]]:
        """在 Sequential 中查找 Conv-BN-Act 三元组"""
        triplets = []
        children = list(seq.named_children())
        
        for i in range(len(children) - 2):
            name1, mod1 = children[i]
            name2, mod2 = children[i + 1]
            name3, mod3 = children[i + 2]
            
            if (self._is_conv(mod1) and self._is_bn(mod2) and 
                self._is_fusable_activation(mod3)):
                if self._channels_match(mod1, mod2):
                    conv_name = f"{prefix}.{name1}" if prefix else name1
                    bn_name = f"{prefix}.{name2}" if prefix else name2
                    act_name = f"{prefix}.{name3}" if prefix else name3
                    triplets.append((conv_name, bn_name, act_name))
        
        return triplets
    
    # ---------------------- Linear + BN 融合 ----------------------
    
    def fuse_linear_bn(self, model: nn.Module) -> Tuple[nn.Module, int]:
        """
        融合 Linear + BatchNorm1d
        
        Args:
            model: 输入模型
            
        Returns:
            (融合后的模型, 融合的对数)
        """
        fused_count = 0
        
        # 查找 Linear-BN 对
        pairs = self._find_linear_bn_pairs(model)
        
        for linear_name, bn_name in pairs:
            linear = ModuleUtils.get_module(model, linear_name)
            bn = ModuleUtils.get_module(model, bn_name)
            
            if linear is None or bn is None:
                continue
            
            # 执行融合
            fused_linear = self._fuse_linear_bn_weights(linear, bn)
            
            # 如果融合失败，跳过
            if fused_linear is None:
                continue
            
            # 替换模块
            ModuleUtils.set_module(model, linear_name, fused_linear)
            ModuleUtils.set_module(model, bn_name, nn.Identity())
            
            fused_count += 1
            
            if self.verbose:
                logger.debug(f"  融合: {linear_name} + {bn_name}")
        
        return model, fused_count
    
    def _find_linear_bn_pairs(self, model: nn.Module) -> List[Tuple[str, str]]:
        """查找 Linear-BN 对"""
        pairs = []
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential):
                children = list(module.named_children())
                
                for i in range(len(children) - 1):
                    name1, mod1 = children[i]
                    name2, mod2 = children[i + 1]
                    
                    if isinstance(mod1, nn.Linear) and isinstance(mod2, nn.BatchNorm1d):
                        if mod1.out_features == mod2.num_features:
                            linear_name = f"{name}.{name1}" if name else name1
                            bn_name = f"{name}.{name2}" if name else name2
                            pairs.append((linear_name, bn_name))
        
        return pairs
    
    def _fuse_linear_bn_weights(self, linear: nn.Linear, 
                                 bn: nn.BatchNorm1d) -> Optional[nn.Linear]:
        """融合 Linear 和 BN 的权重"""
        # 安全检查
        if bn.running_mean is None or bn.running_var is None:
            logger.debug(f"  跳过 Linear+BN 融合: BN 没有有效的 running stats")
            return None
        
        # 获取 BN 参数
        gamma = bn.weight.data if bn.weight is not None else torch.ones_like(bn.running_mean)
        beta = bn.bias.data if bn.bias is not None else torch.zeros_like(bn.running_mean)
        mean = bn.running_mean.data
        var = bn.running_var.data
        eps = bn.eps
        
        # 计算缩放因子
        std = torch.sqrt(var + eps)
        scale = gamma / std
        
        # 创建融合后的 Linear
        fused_linear = nn.Linear(
            linear.in_features, linear.out_features, True
        )
        
        # 复制到相同设备
        fused_linear = fused_linear.to(linear.weight.device)
        
        # 融合权重: W_new = W * γ / √(σ² + ε)
        fused_linear.weight.data = linear.weight.data * scale.unsqueeze(1)
        
        # 融合偏置
        if linear.bias is not None:
            fused_linear.bias.data = (linear.bias.data - mean) * scale + beta
        else:
            fused_linear.bias.data = -mean * scale + beta
        
        return fused_linear
    
    # ---------------------- 辅助方法 ----------------------
    
    def _is_conv(self, module: nn.Module) -> bool:
        """检查是否是卷积层"""
        return isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d))
    
    def _is_bn(self, module: nn.Module) -> bool:
        """
        检查是否是纯粹的 BatchNorm 层
        
        排除 timm 的 BatchNormAct2d 等组合模块
        """
        # 首先检查是否是 BatchNorm 类型
        if not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            return False
        
        # 排除组合模块（如 timm 的 BatchNormAct2d）
        class_name = module.__class__.__name__
        excluded_names = {
            'BatchNormAct2d', 'BatchNormAct1d', 'BatchNormAct3d',
            'FrozenBatchNorm2d', 'SyncBatchNorm',
            'BatchNormAct',  # 通用排除
        }
        if class_name in excluded_names:
            return False
        
        # 检查模块是否有额外的子模块（纯 BN 不应该有子模块）
        if len(list(module.children())) > 0:
            return False
        
        # 检查是否有 act 或 activation 属性（组合模块的特征）
        if hasattr(module, 'act') or hasattr(module, 'activation'):
            return False
        
        return True
    
    def _is_fusable_activation(self, module: nn.Module) -> bool:
        """检查是否是可融合的激活函数"""
        return type(module) in self.FUSABLE_ACTIVATIONS
    
    def _channels_match(self, conv: nn.Module, bn: nn.Module) -> bool:
        """检查 Conv 输出通道数是否与 BN 特征数匹配"""
        return conv.out_channels == bn.num_features


class ConvActFused(nn.Module):
    """
    融合的 Conv + Activation 模块
    
    用于替换 Conv + BN + Act 三元组（BN 已融入 Conv）
    """
    
    def __init__(self, conv: nn.Module, act: nn.Module):
        super().__init__()
        self.conv = conv
        self.act = act
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


# ==============================================================================
# 第4部分: 冗余消除器 - EliminationOptimizer
# ==============================================================================

class EliminationOptimizer:
    """
    冗余消除优化器
    
    移除模型中的冗余层，如 Identity、Dropout 等
    """
    
    # 可移除的模块类型
    IDENTITY_TYPES: Set[Type] = {nn.Identity}
    
    DROPOUT_TYPES: Set[Type] = {
        nn.Dropout,
        nn.Dropout1d,
        nn.Dropout2d,
        nn.Dropout3d,
        nn.AlphaDropout,
        nn.FeatureAlphaDropout,
    }
    
    def __init__(self, verbose: bool = False):
        """
        初始化冗余消除优化器
        
        Args:
            verbose: 是否输出详细日志
        """
        self.verbose = verbose
    
    def eliminate_identity(self, model: nn.Module) -> Tuple[nn.Module, int]:
        """
        移除 Identity 层
        
        Identity 层在推理时没有任何作用，可以安全移除。
        
        Args:
            model: 输入模型
            
        Returns:
            (处理后的模型, 移除的数量)
        """
        removed_count = 0
        
        # 收集要移除的 Identity 层
        identity_names = []
        for name, module in model.named_modules():
            if type(module) in self.IDENTITY_TYPES:
                identity_names.append(name)
        
        # 移除 (实际上在 Sequential 中可以重建，其他情况保留 Identity 避免破坏结构)
        for name in identity_names:
            parent_name, child_name = ModuleUtils.get_parent_name(name)
            parent = ModuleUtils.get_module(model, parent_name)
            
            if isinstance(parent, nn.Sequential):
                # Sequential 中可以安全移除
                if self._remove_from_sequential(parent, child_name):
                    removed_count += 1
                    if self.verbose:
                        logger.debug(f"  移除 Identity: {name}")
        
        return model, removed_count
    
    def eliminate_dropout(self, model: nn.Module) -> Tuple[nn.Module, int]:
        """
        移除 Dropout 层
        
        在 eval 模式下，Dropout 不起作用，可以安全移除。
        
        Args:
            model: 输入模型（应该在 eval 模式）
            
        Returns:
            (处理后的模型, 移除的数量)
        """
        removed_count = 0
        
        # 收集 Dropout 层
        dropout_layers = []
        for name, module in model.named_modules():
            if type(module) in self.DROPOUT_TYPES:
                dropout_layers.append(name)
        
        # 替换为 Identity
        for name in dropout_layers:
            ModuleUtils.set_module(model, name, nn.Identity())
            removed_count += 1
            if self.verbose:
                logger.debug(f"  移除 Dropout: {name}")
        
        return model, removed_count
    
    def _remove_from_sequential(self, seq: nn.Sequential, child_name: str) -> bool:
        """从 Sequential 中移除指定子模块"""
        try:
            # 重建 Sequential，跳过指定的子模块
            new_modules = OrderedDict()
            for name, module in seq.named_children():
                if name != child_name:
                    new_modules[name] = module
            
            # 更新 Sequential
            seq._modules = new_modules
            return True
        except Exception as e:
            logger.debug(f"  无法从 Sequential 移除 {child_name}: {e}")
            return False


# ==============================================================================
# 第5部分: 主优化器 - GraphOptimizer
# ==============================================================================

class GraphOptimizer:
    """
    图优化器主类
    
    协调各个优化 Pass 的执行，验证优化结果
    
    使用示例:
    ---------
    >>> config = OptimConfig(enable_conv_bn_fusion=True)
    >>> optimizer = GraphOptimizer(config)
    >>> result = optimizer.optimize(model, input_shape=(1, 3, 224, 224))
    >>> if result.success:
    ...     optimized_model = result.model
    ...     print(result.stats.summary())
    """
    
    def __init__(self, config: Optional[OptimConfig] = None):
        """
        初始化图优化器
        
        Args:
            config: 优化配置，默认使用 OptimConfig()
        """
        self.config = config or OptimConfig()
        self.fusion = FusionOptimizer(verbose=self.config.verbose)
        self.elimination = EliminationOptimizer(verbose=self.config.verbose)
    
    def optimize(
        self, 
        model: nn.Module, 
        input_shape: Tuple[int, ...]
    ) -> OptimizeResult:
        """
        优化模型
        
        执行流程:
        1. 深拷贝模型
        2. 记录原始输出（用于验证）
        3. 执行融合优化
        4. 执行冗余消除
        5. 验证输出一致性
        6. 生成统计报告
        
        Args:
            model: 输入模型
            input_shape: 输入形状，如 (1, 3, 224, 224)
            
        Returns:
            OptimizeResult
        """
        # 初始化统计
        stats = OptimizationStats()
        
        # 记录原始算子数量
        stats.original_op_count = ModuleUtils.count_ops(model)
        
        # 深拷贝模型
        optimized_model = ModuleUtils.deep_copy(model)
        optimized_model.eval()
        
        # 获取设备
        device = ModuleUtils.get_device(model)
        
        # 创建固定的验证输入（确保前后使用相同输入）
        verify_input = None
        original_output = None
        if self.config.verify_output:
            torch.manual_seed(42)  # 固定随机种子
            verify_input = torch.randn(input_shape, device=device)
            model.eval()
            with torch.no_grad():
                output = model(verify_input)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                original_output = output.clone()
        
        try:
            # ========== Pass 1: 融合优化 ==========
            
            # 1.1 Conv + BN + Act 融合 (优先于 Conv + BN，因为它更完整)
            if self.config.enable_conv_bn_act_fusion:
                optimized_model, count = self.fusion.fuse_conv_bn_act(optimized_model)
                stats.conv_bn_act_fused = count
                if count > 0:
                    logger.info(f"    Conv+BN+Act 融合: {count} 对")
            
            # 1.2 Conv + BN 融合
            if self.config.enable_conv_bn_fusion:
                optimized_model, count = self.fusion.fuse_conv_bn(optimized_model)
                stats.conv_bn_fused = count
                stats.conv_bn_skipped = self.fusion.get_skipped_count()
                if count > 0:
                    logger.info(f"    Conv+BN 融合: {count} 对")
                if stats.conv_bn_skipped > 0:
                    logger.info(f"    Conv+BN 跳过(自定义模块): {stats.conv_bn_skipped} 对")
            
            # 1.3 Linear + BN 融合
            if self.config.enable_linear_bn_fusion:
                optimized_model, count = self.fusion.fuse_linear_bn(optimized_model)
                stats.linear_bn_fused = count
                if count > 0:
                    logger.info(f"    Linear+BN 融合: {count} 对")
            
            # ========== Pass 2: 冗余消除 ==========
            
            # 2.1 Dropout 移除
            if self.config.enable_dropout_elimination:
                optimized_model, count = self.elimination.eliminate_dropout(optimized_model)
                stats.dropout_removed = count
                if count > 0:
                    logger.info(f"    Dropout 移除: {count} 个")
            
            # 2.2 Identity 移除
            if self.config.enable_identity_elimination:
                optimized_model, count = self.elimination.eliminate_identity(optimized_model)
                stats.identity_removed = count
                if count > 0:
                    logger.info(f"    Identity 移除: {count} 个")
            
            # ========== 验证阶段 ==========
            
            if self.config.verify_output and original_output is not None and verify_input is not None:
                optimized_model.eval()
                with torch.no_grad():
                    output = optimized_model(verify_input)
                    if isinstance(output, (tuple, list)):
                        output = output[0]
                    optimized_output = output
                
                passed, diff = self._verify_output(original_output, optimized_output)
                stats.verification_passed = passed
                stats.output_diff = diff
                
                if not passed:
                    logger.warning(f"    ⚠️ 输出验证未通过 (diff={diff:.2e})")
            
            # 记录优化后算子数量
            stats.optimized_op_count = ModuleUtils.count_ops(optimized_model)
            
            return OptimizeResult(
                model=optimized_model,
                stats=stats,
                success=True,
                message="优化成功"
            )
            
        except Exception as e:
            logger.error(f"    ❌ 优化过程出错: {e}")
            import traceback
            traceback.print_exc()
            return OptimizeResult(
                model=model,  # 返回原始模型
                stats=stats,
                success=False,
                message=f"优化失败: {e}"
            )
    
    def _get_model_output(
        self, 
        model: nn.Module, 
        input_shape: Tuple[int, ...], 
        device: torch.device
    ) -> torch.Tensor:
        """获取模型输出"""
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(input_shape, device=device)
            output = model(dummy_input)
            # 处理多输出情况
            if isinstance(output, (tuple, list)):
                output = output[0]
            return output.clone()
    
    def _verify_output(
        self, 
        original: torch.Tensor, 
        optimized: torch.Tensor
    ) -> Tuple[bool, float]:
        """
        验证输出一致性
        
        使用多种指标判断：
        1. 最大绝对差异 < tolerance
        2. 或者余弦相似度 > 0.9999 (即使绝对差异稍大也通过)
        
        Args:
            original: 原始模型输出
            optimized: 优化后模型输出
            
        Returns:
            (是否通过, 最大差异)
        """
        try:
            diff = (original - optimized).abs().max().item()
            
            # 方法1: 绝对差异
            if diff < self.config.verify_tolerance:
                return True, diff
            
            # 方法2: 余弦相似度 (对于大值输出更合理)
            orig_flat = original.flatten().float()
            opt_flat = optimized.flatten().float()
            
            # 计算余弦相似度
            cos_sim = torch.nn.functional.cosine_similarity(
                orig_flat.unsqueeze(0), 
                opt_flat.unsqueeze(0)
            ).item()
            
            # 余弦相似度 > 0.9999 认为通过
            if cos_sim > 0.9999:
                return True, diff
            
            return False, diff
        except Exception as e:
            logger.warning(f"输出验证出错: {e}")
            return False, float('inf')


# ==============================================================================
# 第6部分: 便捷函数
# ==============================================================================

def optimize_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    config: Optional[OptimConfig] = None,
) -> OptimizeResult:
    """
    便捷函数：优化模型
    
    Args:
        model: PyTorch 模型
        input_shape: 输入形状，如 (1, 3, 224, 224)
        config: 优化配置，默认启用所有优化
        
    Returns:
        OptimizeResult
        
    使用示例:
    ---------
    >>> from model_optimizer import optimize_model
    >>> result = optimize_model(model, (1, 3, 224, 224))
    >>> if result.success:
    ...     model = result.model
    ...     print(result.stats.summary())
    """
    optimizer = GraphOptimizer(config)
    return optimizer.optimize(model, input_shape)



# ==============================================================================
# 主程序入口
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=20,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    print(__doc__)
    print("\n请使用 main.py 进行完整的模型优化流程")
