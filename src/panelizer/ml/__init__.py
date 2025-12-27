"""ML-based panel detection (optional dependency).

This module requires extra dependencies:
    uv sync --extra ml

Example usage:
    >>> from panelizer.ml import YOLODetector, is_ml_available
    >>>
    >>> if is_ml_available():
    ...     detector = YOLODetector()
    ...     result = detector.detect(image)
"""

from .base import Detector, MLDetector
from .exceptions import MLDependencyError, MLError, ModelNotFoundError


def is_ml_available() -> bool:
    """Check if ML dependencies (ultralytics, torch) are installed.

    Returns:
        True if ML detection is available
    """
    try:
        import ultralytics  # noqa: F401

        return True
    except ImportError:
        return False


__all__ = [
    "Detector",
    "MLDetector",
    "MLError",
    "MLDependencyError",
    "ModelNotFoundError",
    "is_ml_available",
]

# Conditionally export YOLODetector if dependencies are available
# This allows type checking to work even without deps installed
try:
    from .yolo_detector import YOLODetector  # noqa: F401

    __all__.append("YOLODetector")
except ImportError:
    # ultralytics not installed - YOLODetector won't be available
    pass
