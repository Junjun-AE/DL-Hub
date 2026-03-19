#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一异常层次结构模块

提供清晰的异常继承层次，便于错误处理和调试。

Author: Model Converter Team
Version: 1.0.0
"""

from typing import Optional, List, Any


# ============================================================================
# 基础异常类
# ============================================================================

class ConverterError(Exception):
    """
    模型转换器基础异常
    
    所有自定义异常的基类，提供统一的错误信息格式。
    """
    
    def __init__(self, message: str, details: Optional[str] = None):
        """
        初始化异常
        
        Args:
            message: 错误消息
            details: 详细信息（可选）
        """
        self.message = message
        self.details = details
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """格式化错误消息"""
        if self.details:
            return f"{self.message}\n详细信息: {self.details}"
        return self.message


# ============================================================================
# 依赖相关异常
# ============================================================================

class DependencyError(ConverterError):
    """
    依赖缺失或版本不兼容异常
    
    当必需的库未安装或版本不满足要求时抛出。
    """
    
    def __init__(
        self,
        package: str,
        required_version: Optional[str] = None,
        install_cmd: Optional[str] = None,
    ):
        """
        初始化依赖异常
        
        Args:
            package: 缺失的包名
            required_version: 要求的版本（可选）
            install_cmd: 安装命令（可选）
        """
        self.package = package
        self.required_version = required_version
        self.install_cmd = install_cmd
        
        message = f"依赖缺失: {package}"
        if required_version:
            message += f" >= {required_version}"
        
        details = None
        if install_cmd:
            details = f"请运行: {install_cmd}"
        
        super().__init__(message, details)


# ============================================================================
# 模型导入相关异常
# ============================================================================

class ModelImportError(ConverterError):
    """
    模型导入异常
    
    当模型文件无法加载或格式不支持时抛出。
    """
    
    def __init__(
        self,
        model_path: str,
        reason: str,
        supported_formats: Optional[List[str]] = None,
    ):
        """
        初始化模型导入异常
        
        Args:
            model_path: 模型文件路径
            reason: 失败原因
            supported_formats: 支持的格式列表（可选）
        """
        self.model_path = model_path
        self.reason = reason
        self.supported_formats = supported_formats
        
        message = f"模型导入失败: {model_path}\n原因: {reason}"
        
        details = None
        if supported_formats:
            details = f"支持的格式: {', '.join(supported_formats)}"
        
        super().__init__(message, details)


class SecurityError(ConverterError):
    """
    安全相关异常
    
    当检测到不安全的操作（如加载不受信任的模型文件）时抛出。
    """
    
    def __init__(self, message: str, mitigation: Optional[str] = None):
        """
        初始化安全异常
        
        Args:
            message: 安全问题描述
            mitigation: 缓解措施（可选）
        """
        self.mitigation = mitigation
        details = mitigation if mitigation else "请确认操作来源可信"
        super().__init__(f"安全警告: {message}", details)


# ============================================================================
# 校准相关异常
# ============================================================================

class CalibrationError(ConverterError):
    """
    校准数据异常
    
    当校准数据无效、不足或处理失败时抛出。
    """
    
    def __init__(
        self,
        message: str,
        data_path: Optional[str] = None,
        num_samples: Optional[int] = None,
    ):
        """
        初始化校准异常
        
        Args:
            message: 错误消息
            data_path: 校准数据路径（可选）
            num_samples: 样本数量（可选）
        """
        self.data_path = data_path
        self.num_samples = num_samples
        
        details_parts = []
        if data_path:
            details_parts.append(f"数据路径: {data_path}")
        if num_samples is not None:
            details_parts.append(f"样本数量: {num_samples}")
        
        details = "\n".join(details_parts) if details_parts else None
        super().__init__(message, details)


# ============================================================================
# 导出相关异常
# ============================================================================

class ExportError(ConverterError):
    """
    ONNX 导出异常
    
    当模型无法导出为 ONNX 格式时抛出。
    """
    
    def __init__(
        self,
        message: str,
        unsupported_ops: Optional[List[str]] = None,
        opset_version: Optional[int] = None,
    ):
        """
        初始化导出异常
        
        Args:
            message: 错误消息
            unsupported_ops: 不支持的算子列表（可选）
            opset_version: opset 版本（可选）
        """
        self.unsupported_ops = unsupported_ops
        self.opset_version = opset_version
        
        details_parts = []
        if opset_version:
            details_parts.append(f"opset 版本: {opset_version}")
        if unsupported_ops:
            details_parts.append(f"不支持的算子: {', '.join(unsupported_ops)}")
        
        details = "\n".join(details_parts) if details_parts else None
        super().__init__(message, details)


class ExportConfigError(ExportError):
    """
    导出配置异常
    
    当导出配置无效时抛出。
    """
    
    def __init__(self, param: str, reason: str):
        """
        初始化配置异常
        
        Args:
            param: 无效的参数名
            reason: 原因
        """
        self.param = param
        super().__init__(f"配置错误 [{param}]: {reason}")


# ============================================================================
# 转换相关异常
# ============================================================================

class ConversionError(ConverterError):
    """
    量化转换异常
    
    当模型无法转换到目标格式时抛出。
    """
    
    def __init__(
        self,
        message: str,
        backend: Optional[str] = None,
        precision: Optional[str] = None,
    ):
        """
        初始化转换异常
        
        Args:
            message: 错误消息
            backend: 目标后端（可选）
            precision: 精度模式（可选）
        """
        self.backend = backend
        self.precision = precision
        
        details_parts = []
        if backend:
            details_parts.append(f"目标后端: {backend}")
        if precision:
            details_parts.append(f"精度模式: {precision}")
        
        details = "\n".join(details_parts) if details_parts else None
        super().__init__(message, details)


# ============================================================================
# 验证相关异常
# ============================================================================

class ValidationError(ConverterError):
    """
    验证失败异常
    
    当模型验证（精度对比、性能测试）失败时抛出。
    """
    
    def __init__(
        self,
        message: str,
        metric: Optional[str] = None,
        expected: Optional[float] = None,
        actual: Optional[float] = None,
    ):
        """
        初始化验证异常
        
        Args:
            message: 错误消息
            metric: 验证指标名称（可选）
            expected: 期望值（可选）
            actual: 实际值（可选）
        """
        self.metric = metric
        self.expected = expected
        self.actual = actual
        
        details_parts = []
        if metric:
            details_parts.append(f"验证指标: {metric}")
        if expected is not None:
            details_parts.append(f"期望值: {expected}")
        if actual is not None:
            details_parts.append(f"实际值: {actual}")
        
        details = "\n".join(details_parts) if details_parts else None
        super().__init__(message, details)


# ============================================================================
# 优化相关异常
# ============================================================================

class OptimizationError(ConverterError):
    """
    优化失败异常
    
    当图优化（融合、消除等）失败时抛出。
    """
    
    def __init__(
        self,
        message: str,
        optimization_type: Optional[str] = None,
        layer_name: Optional[str] = None,
    ):
        """
        初始化优化异常
        
        Args:
            message: 错误消息
            optimization_type: 优化类型（可选）
            layer_name: 相关层名称（可选）
        """
        self.optimization_type = optimization_type
        self.layer_name = layer_name
        
        details_parts = []
        if optimization_type:
            details_parts.append(f"优化类型: {optimization_type}")
        if layer_name:
            details_parts.append(f"相关层: {layer_name}")
        
        details = "\n".join(details_parts) if details_parts else None
        super().__init__(message, details)


# ============================================================================
# 符号注册相关异常
# ============================================================================

class SymbolicError(ConverterError):
    """
    Symbolic 注册异常基类
    """
    pass


class SymbolicRegistrationError(SymbolicError):
    """
    Symbolic 注册失败异常
    """
    
    def __init__(self, op_name: str, reason: str):
        """
        初始化注册异常
        
        Args:
            op_name: 算子名称
            reason: 失败原因
        """
        self.op_name = op_name
        super().__init__(f"Symbolic 注册失败 [{op_name}]: {reason}")


class SymbolicNotFoundError(SymbolicError):
    """
    Symbolic 未找到异常
    """
    
    def __init__(self, op_name: str, available: Optional[List[str]] = None):
        """
        初始化未找到异常
        
        Args:
            op_name: 算子名称
            available: 可用的 symbolic 列表（可选）
        """
        self.op_name = op_name
        self.available = available
        
        details = None
        if available:
            details = f"可用的 symbolic: {', '.join(available[:10])}"
            if len(available) > 10:
                details += f" ... (共 {len(available)} 个)"
        
        super().__init__(f"未找到 symbolic: {op_name}", details)


# ============================================================================
# 配置相关异常
# ============================================================================

class ConfigError(ConverterError):
    """
    配置错误异常
    
    当配置文件无效或缺少必需参数时抛出。
    """
    
    def __init__(
        self,
        message: str,
        config_path: Optional[str] = None,
        invalid_keys: Optional[List[str]] = None,
    ):
        """
        初始化配置异常
        
        Args:
            message: 错误消息
            config_path: 配置文件路径（可选）
            invalid_keys: 无效的配置键列表（可选）
        """
        self.config_path = config_path
        self.invalid_keys = invalid_keys
        
        details_parts = []
        if config_path:
            details_parts.append(f"配置文件: {config_path}")
        if invalid_keys:
            details_parts.append(f"无效配置: {', '.join(invalid_keys)}")
        
        details = "\n".join(details_parts) if details_parts else None
        super().__init__(message, details)


# ============================================================================
# 便捷函数
# ============================================================================

def raise_dependency_error(
    package: str,
    required_version: Optional[str] = None,
) -> None:
    """
    抛出依赖错误的便捷函数
    
    Args:
        package: 包名
        required_version: 要求的版本
    """
    install_cmd = f"pip install {package}"
    if required_version:
        install_cmd += f">={required_version}"
    
    raise DependencyError(
        package=package,
        required_version=required_version,
        install_cmd=install_cmd,
    )
