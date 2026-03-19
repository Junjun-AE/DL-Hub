"""SevSeg-YOLO prediction/inference module (§21.5 P6).

Extends the standard DetectionPredictor to handle the extra severity score
column in ScoreDetect postprocess output (B, K, 7).

Usage:
    from sevseg_yolo.predict import ScoreDetectionPredictor

    predictor = ScoreDetectionPredictor(overrides={"model": "best.pt"})
    results = predictor("image.jpg")
    # results[0].severity → severity scores on [0, 10] scale
"""

from __future__ import annotations

import torch
import numpy as np


class ScoreDetectionPredictor:
    """SevSeg-YOLO predictor with severity score support.

    Wraps the Ultralytics DetectionPredictor to:
    1. Extract severity scores from the 7th column of model output.
    2. Denormalize scores from [0, 1] to [0, 10] scale.
    3. Attach scores to prediction results.

    Args:
        overrides: Dict of argument overrides (model, source, conf, etc.).
    """

    def __init__(self, overrides: dict | None = None):
        from ultralytics.models.yolo.detect.predict import DetectionPredictor
        self._base_predictor = DetectionPredictor(overrides=overrides or {})

    def __call__(self, source=None, **kwargs):
        """Run prediction on source image(s).

        Args:
            source: Image path, directory, URL, or numpy array.
            **kwargs: Additional prediction arguments.

        Returns:
            List of ScoreDetectionResult objects.
        """
        results = self._base_predictor(source=source, **kwargs)
        return [self._enhance_result(r) for r in results]

    @staticmethod
    def _enhance_result(result):
        """Attach denormalized severity scores to a detection result.

        The ScoreDetect.postprocess output format is (B, K, 7):
            [x1, y1, x2, y2, max_cls_prob, cls_index, severity_01]

        This method extracts the severity column and denormalizes it.
        """
        # Check if result has the extra column
        if hasattr(result, "boxes") and result.boxes is not None:
            data = result.boxes.data  # (K, 6+) tensor

            if data.shape[-1] >= 7:
                # Extract severity from column 6 (0-indexed)
                severity_01 = data[:, 6]  # [0, 1]
                severity_10 = severity_01 * 10.0  # [0, 10]

                # Attach to result
                result.severity = severity_10.cpu().numpy()
            else:
                # No severity column (standard detection model)
                result.severity = np.zeros(data.shape[0])

        return result


def run_inference(
    model_path: str,
    source,
    conf: float = 0.25,
    imgsz: int = 640,
    device: str = "",
) -> list[dict]:
    """Convenience function for SevSeg-YOLO inference.

    Args:
        model_path: Path to trained SevSeg-YOLO weights.
        source: Image path, directory, or numpy array.
        conf: Confidence threshold.
        imgsz: Input image size.
        device: Device string ('', '0', 'cpu', etc.).

    Returns:
        List of dicts, each containing:
            'boxes': (K, 4) array of [x1, y1, x2, y2]
            'scores': (K,) confidence scores
            'classes': (K,) class indices
            'severity': (K,) severity scores on [0, 10] scale
            'class_names': list of class name strings
    """
    predictor = ScoreDetectionPredictor(overrides={
        "model": model_path,
        "conf": conf,
        "imgsz": imgsz,
        "device": device,
    })

    results = predictor(source=source)
    outputs = []

    for r in results:
        boxes = r.boxes
        output = {
            "boxes": boxes.xyxy.cpu().numpy() if boxes is not None else np.zeros((0, 4)),
            "scores": boxes.conf.cpu().numpy() if boxes is not None else np.zeros(0),
            "classes": boxes.cls.cpu().numpy().astype(int) if boxes is not None else np.zeros(0, dtype=int),
            "severity": getattr(r, "severity", np.zeros(0)),
            "class_names": [r.names[int(c)] for c in (boxes.cls.cpu().numpy() if boxes is not None else [])],
        }
        outputs.append(output)

    return outputs
