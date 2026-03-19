"""
SevSeg-YOLO: 工业缺陷检测 + 严重程度评分 + 零标注近似分割
==========================================================

一行代码完成完整推理:

    from sevseg_yolo import SevSegYOLO

    model = SevSegYOLO("best.pt")
    result = model.predict("image.jpg")

    for det in result.detections:
        print(det.severity, det.class_name, det.mask.shape)

    result.visualize().save("output.jpg")
"""

__version__ = "2.0.0"

# 主入口 (推荐)
from sevseg_yolo.model import SevSegYOLO, SevSegResult, Detection

# 底层组件 (高级用户)
from sevseg_yolo.mask_generator_v3 import MaskGeneratorV3
from sevseg_yolo.mask_generator_v2 import MaskGeneratorV2
from sevseg_yolo.evaluation import full_score_evaluation
from sevseg_yolo.utils import (
    predict_with_masks,
    register_feature_hooks,
    get_model_input_hw,
    bbox_to_model_space,
)

__all__ = [
    # 主入口
    "SevSegYOLO",
    "SevSegResult",
    "Detection",
    # 底层
    "MaskGeneratorV3",
    "MaskGeneratorV2",
    "full_score_evaluation",
    "predict_with_masks",
    "register_feature_hooks",
    "get_model_input_hw",
    "bbox_to_model_space",
]
