"""SevSeg-YOLO utility functions.

Provides:
    - register_feature_hooks(): Register hooks to extract P3/P4/P5 features
    - bbox_to_model_space(): Convert bbox from original to model coordinates
    - predict_with_masks(): High-level API for detection + scoring + mask generation
"""
from __future__ import annotations

import numpy as np
import cv2


def register_feature_hooks(model):
    """Register forward hooks to extract multi-scale features from a SevSeg-YOLO model.

    Args:
        model: YOLO model instance (from ultralytics.YOLO)

    Returns:
        tuple: (hooked_dict, hooks_list)
            - hooked_dict: dict that will be populated with "p3", "p4", "p5", "layer2"
            - hooks_list: list of hook handles (call h.remove() to clean up)

    Example:
        >>> from ultralytics import YOLO
        >>> model = YOLO("best.pt")
        >>> hooked, hooks = register_feature_hooks(model)
        >>> model("image.jpg")  # triggers hooks
        >>> print(hooked["p3"].shape)  # (C, H, W)
        >>> for h in hooks: h.remove()  # cleanup
    """
    from ultralytics.nn.modules.head import ScoreDetect

    raw = model.model
    hooked = {}
    hooks = []

    # Hook backbone layer 2 (for n/s scales with layer2 features)
    if hasattr(raw, 'model') and len(raw.model) > 2:
        hooks.append(raw.model[2].register_forward_hook(
            lambda m, i, o: hooked.update({"layer2": o.detach().cpu().numpy()[0]})))

    # Hook ScoreDetect head input to get P3/P4/P5
    for md in raw.modules():
        if isinstance(md, ScoreDetect):
            def _make_hook():
                def _h(mod, inp, out):
                    if isinstance(inp, tuple) and len(inp) > 0:
                        x = inp[0]
                        if isinstance(x, list) and len(x) >= 2:
                            hooked["p3"] = x[0].detach().cpu().numpy()[0]
                            hooked["p4"] = x[1].detach().cpu().numpy()[0]
                            if len(x) > 2:
                                hooked["p5"] = x[2].detach().cpu().numpy()[0]
                return _h
            hooks.append(md.register_forward_hook(_make_hook()))
            break

    return hooked, hooks


def get_model_input_hw(p3_feat):
    """Infer model input (H, W) from P3 feature map shape.

    Args:
        p3_feat: P3 feature map, shape (C, H, W)

    Returns:
        tuple: (input_H, input_W) — actual model input size (may differ from 640x640 due to auto letterbox)
    """
    return (p3_feat.shape[1] * 8, p3_feat.shape[2] * 8)


def bbox_to_model_space(bbox, orig_hw, model_hw):
    """Convert bbox from original image coordinates to model input space.

    YOLO uses auto-letterbox which may produce non-square inputs (e.g., 320x640 for 1000x2048 images).
    This function handles the correct gain + padding calculation.

    Args:
        bbox: [x1, y1, x2, y2] in original image coordinates
        orig_hw: (orig_H, orig_W)
        model_hw: (model_H, model_W) from get_model_input_hw()

    Returns:
        list: [mx1, my1, mx2, my2] in model input space
    """
    oh, ow = orig_hw
    mh, mw = model_hw
    gain = min(mh / oh, mw / ow)
    pad_x = round((mw - ow * gain) / 2 - 0.1)
    pad_y = round((mh - oh * gain) / 2 - 0.1)
    x1, y1, x2, y2 = bbox
    return [max(0, min(x1 * gain + pad_x, mw)),
            max(0, min(y1 * gain + pad_y, mh)),
            max(0, min(x2 * gain + pad_x, mw)),
            max(0, min(y2 * gain + pad_y, mh))]


def predict_with_masks(model, source, conf=0.15, imgsz=640, device="0",
                       mask_config=None):
    """High-level API: detection + severity scoring + mask generation in one call.

    Args:
        model: YOLO model instance or path to weights
        source: Image path, directory, or numpy array (BGR)
        conf: Confidence threshold
        imgsz: Input image size
        device: Device string ("0", "cpu", etc.)
        mask_config: MaskGenerator config dict, default S6-tuned params

    Returns:
        list of dicts, each containing:
            - "bbox": [x1, y1, x2, y2] (original image coordinates)
            - "class_id": int
            - "confidence": float
            - "severity": float [0, 10]
            - "mask": np.ndarray {0,1} at bbox resolution (rh, rw)
            - "fill_ratio": float (mask fill ratio)

    Example:
        >>> from ultralytics import YOLO
        >>> from sevseg_yolo.utils import predict_with_masks
        >>> model = YOLO("best.pt")
        >>> results = predict_with_masks(model, "image.jpg")
        >>> for det in results:
        ...     print(f"Severity: {det['severity']:.1f}, Mask fill: {det['fill_ratio']:.3f}")
    """
    from ultralytics import YOLO
    from sevseg_yolo.mask_generator_v2 import MaskGeneratorV2

    # Load model if path
    if isinstance(model, (str,)):
        model = YOLO(model)

    # Default S6-tuned config
    if mask_config is None:
        mask_config = dict(
            topk_channels=48, guided_radius=6, guided_eps=0.005,
            adaptive_block=15, adaptive_C=-5, morph_close_k=7, morph_open_k=3,
        )

    mg = MaskGeneratorV2(**mask_config)
    hooked, hooks = register_feature_hooks(model)

    # Read image if path
    if isinstance(source, str):
        orig = cv2.imread(source)
    else:
        orig = source
    if orig is None:
        return []

    oh, ow = orig.shape[:2]

    # Run detection
    hooked.clear()
    preds = model(source, conf=conf, imgsz=imgsz, device=device, verbose=False)

    if not preds or preds[0].boxes is None or len(preds[0].boxes) == 0 or "p3" not in hooked:
        for h in hooks: h.remove()
        return []

    model_hw = get_model_input_hw(hooked["p3"])
    detections = []

    for i in range(len(preds[0].boxes)):
        bb = preds[0].boxes.xyxy[i].cpu().numpy()
        cls_id = int(preds[0].boxes.cls[i].item())
        confidence = preds[0].boxes.conf[i].item()
        severity = 0.0
        if hasattr(preds[0], '_raw_pred') and preds[0]._raw_pred is not None:
            raw = preds[0]._raw_pred
            if raw.shape[-1] >= 7:
                severity = raw[i, 6].item() * 10
        elif preds[0].boxes.data.shape[-1] >= 7:
            severity = preds[0].boxes.data[i, 6].item() * 10

        bx1, by1 = max(0, int(bb[0])), max(0, int(bb[1]))
        bx2, by2 = min(ow, int(bb[2])), min(oh, int(bb[3]))
        rh, rw = by2 - by1, bx2 - bx1
        if rh < 3 or rw < 3:
            continue

        # Model space bbox
        bbox_model = bbox_to_model_space(bb[:4].tolist(), (oh, ow), model_hw)

        # Guide crop (original image coordinates — correct!)
        guide_crop = orig[by1:by2, bx1:bx2]

        # Generate mask
        mask_raw = mg.generate(
            bbox=bbox_model,
            feat_layer2=hooked.get("layer2"),
            feat_p3_fpn=hooked["p3"],
            feat_p4_fpn=hooked["p4"],
            feat_p5_fpn=hooked.get("p5"),
            guide_crop=guide_crop,
            input_hw=model_hw,
        )

        # Resize to original bbox size
        mask = cv2.resize(mask_raw, (rw, rh), interpolation=cv2.INTER_NEAREST)

        detections.append({
            "bbox": [bx1, by1, bx2, by2],
            "class_id": cls_id,
            "confidence": round(confidence, 4),
            "severity": round(severity, 2),
            "mask": mask,
            "fill_ratio": round(float(mask.mean()), 4),
        })

    for h in hooks:
        h.remove()

    return detections
