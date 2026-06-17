# -*- coding: utf-8 -*-
"""
OCR Toolkit - 文本检测与识别工具
================================

功能:
1. 文本检测 - 定位图像中的文字区域
2. 文本识别 - 识别文字内容
3. 端到端OCR - 检测+识别一体化
4. 模型导出 - 导出为ONNX/TensorRT格式

支持的模型:
- PaddleOCR (推荐，无需训练)
- PPOCR v4 (最新版本)
- 自定义模型

部署支持:
- ONNX Runtime
- TensorRT (FP16/FP32)
"""

__version__ = "1.0.0"
__author__ = "OCR Toolkit"

# 模型配置
MODEL_CONFIGS = {
    # 检测模型
    'det': {
        'ch_PP-OCRv4_det': {
            'name': 'PP-OCRv4 检测',
            'lang': 'ch',
            'description': '中英文检测，最新版本',
            'input_size': (640, 640),
            'url': 'https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar'
        },
        'ch_PP-OCRv3_det': {
            'name': 'PP-OCRv3 检测',
            'lang': 'ch',
            'description': '中英文检测，轻量版',
            'input_size': (640, 640),
            'url': 'https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar'
        },
        'en_PP-OCRv3_det': {
            'name': 'PP-OCRv3 英文检测',
            'lang': 'en',
            'description': '英文检测',
            'input_size': (640, 640),
            'url': 'https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar'
        },
    },
    # 识别模型
    'rec': {
        'ch_PP-OCRv4_rec': {
            'name': 'PP-OCRv4 识别',
            'lang': 'ch',
            'description': '中英文识别，最新版本',
            'input_height': 48,
            'url': 'https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar'
        },
        'ch_PP-OCRv3_rec': {
            'name': 'PP-OCRv3 识别',
            'lang': 'ch',
            'description': '中英文识别，轻量版',
            'input_height': 48,
            'url': 'https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar'
        },
        'en_PP-OCRv4_rec': {
            'name': 'PP-OCRv4 英文识别',
            'lang': 'en',
            'description': '英文识别',
            'input_height': 48,
            'url': 'https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar'
        },
        'japan_PP-OCRv3_rec': {
            'name': 'PP-OCRv3 日文识别',
            'lang': 'japan',
            'description': '日文识别',
            'input_height': 48,
            'url': 'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/japan_PP-OCRv3_rec_infer.tar'
        },
    },
    # 方向分类模型
    'cls': {
        'ch_ppocr_mobile_v2.0_cls': {
            'name': '方向分类器',
            'description': '0°/180°方向分类',
            'url': 'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar'
        },
    }
}

# 支持的语言
SUPPORTED_LANGUAGES = {
    'ch': '中文',
    'en': '英文',
    'japan': '日文',
    'korean': '韩文',
    'german': '德文',
    'french': '法文',
}

# TensorRT 配置
TENSORRT_CONFIG = {
    'workspace_size': 4,  # GB
    'fp16': True,
    'int8': False,
    'max_batch_size': 8,
    'dynamic_shapes': True,
}
