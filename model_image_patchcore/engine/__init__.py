# -*- coding: utf-8 -*-
"""训练引擎模块"""

try:
    from .trainer import (
        PatchCoreTrainer,
        TrainingCallback,
        TrainingResult,
    )
    __all__ = [
        'PatchCoreTrainer',
        'TrainingCallback',
        'TrainingResult',
    ]
except ImportError:
    # torch不可用时，仅导出数据类
    from .trainer import TrainingCallback, TrainingResult
    __all__ = ['TrainingCallback', 'TrainingResult']
