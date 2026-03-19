"""
配置模块
"""

from .model_registry import MODEL_REGISTRY, get_model_config, get_all_families, get_all_scales

__all__ = ['MODEL_REGISTRY', 'get_model_config', 'get_all_families', 'get_all_scales']
