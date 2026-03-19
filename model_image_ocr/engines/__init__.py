# -*- coding: utf-8 -*-
"""OCR引擎模块"""

from .ocr_engine import (
    PaddleOCREngine,
    ONNXOCREngine,
    TensorRTOCREngine,
    OCRResult,
    create_engine,
)

__all__ = [
    'PaddleOCREngine',
    'ONNXOCREngine', 
    'TensorRTOCREngine',
    'OCRResult',
    'create_engine',
]
