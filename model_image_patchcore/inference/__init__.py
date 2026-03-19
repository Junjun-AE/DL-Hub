# -*- coding: utf-8 -*-
"""推理模块"""

try:
    from .predictor import (
        PatchCorePredictor,
        PredictionResult,
        create_visualization,
    )
    __all__ = [
        'PatchCorePredictor',
        'PredictionResult',
        'create_visualization',
    ]
except ImportError:
    # torch不可用时，仅导出PredictionResult
    from .predictor import PredictionResult
    __all__ = ['PredictionResult']
