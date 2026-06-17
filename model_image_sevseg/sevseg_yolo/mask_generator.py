"""MaskGenerator — default module.

Re-exports MaskGeneratorV3 (bimodal channel selection + Canny edge-guided
upsampling) as the default. V2 is available via mask_generator_v2.

Usage:
    from sevseg_yolo.mask_generator import MaskGeneratorV3
    mg = MaskGeneratorV3()
    mask = mg.generate(bbox=..., feat_p3_fpn=..., feat_p4_fpn=..., guide_crop=...)
"""
from sevseg_yolo.mask_generator_v3 import MaskGeneratorV3
from sevseg_yolo.mask_generator_v2 import MaskGeneratorV2

__all__ = ["MaskGeneratorV3", "MaskGeneratorV2"]
