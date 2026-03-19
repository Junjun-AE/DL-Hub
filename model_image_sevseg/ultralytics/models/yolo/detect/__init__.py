# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .predict import DetectionPredictor
from .train import DetectionTrainer, ScoreDetectionTrainer
from .val import DetectionValidator, ScoreDetectionValidator

__all__ = (
    "DetectionPredictor",
    "DetectionTrainer",
    "DetectionValidator",
    "ScoreDetectionTrainer",
    "ScoreDetectionValidator",
)
