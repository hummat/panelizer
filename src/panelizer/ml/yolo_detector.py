"""YOLO-based panel detector using best-comic-panel-detection model."""

import sys
import time
from typing import TYPE_CHECKING, List, Optional

from PIL import Image

from ..cv.detector import DetectionResult
from ..schema import Panel
from .base import MLDetector
from .exceptions import MLDependencyError
from .model_manager import get_yolo_model_path

if TYPE_CHECKING:
    from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]

# Track if we've warned about CPU usage
_cpu_warning_shown = False


def _check_ultralytics_available() -> bool:
    """Check if ultralytics is installed."""
    try:
        import ultralytics  # noqa: F401

        return True
    except ImportError:
        return False


def _get_best_device() -> str:
    """Auto-detect the best available device.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class YOLODetector(MLDetector):
    """YOLO-based panel detector using best-comic-panel-detection model.

    This detector uses the YOLOv12x model fine-tuned for comic panel detection,
    available from HuggingFace (mosesb/best-comic-panel-detection).

    Example:
        >>> detector = YOLODetector()
        >>> result = detector.detect(image)
        >>> for panel in result.panels:
        ...     print(f"Panel {panel.id}: {panel.bbox} (conf: {panel.confidence:.2f})")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.25,
        device: Optional[str] = None,
    ):
        """Initialize YOLO detector.

        Args:
            model_path: Path to YOLO weights file. If None, downloads from HuggingFace.
            confidence_threshold: Minimum confidence for panel detection (0-1).
            device: Device for inference ('cuda', 'mps', 'cpu', or None for auto).
        """
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._requested_device = device
        self._model: Optional["YOLO"] = None
        self._available: Optional[bool] = None
        self._actual_device: Optional[str] = None

    def is_available(self) -> bool:
        """Check if YOLO backend is available."""
        if self._available is None:
            self._available = _check_ultralytics_available()
        return self._available

    @property
    def model_name(self) -> str:
        return "YOLOv12x (best-comic-panel-detection)"

    @property
    def device(self) -> str:
        """Get the device being used for inference."""
        if self._actual_device is None:
            if self._requested_device and self._requested_device != "auto":
                self._actual_device = self._requested_device
            else:
                self._actual_device = _get_best_device()
        return self._actual_device

    def _ensure_model_loaded(self) -> "YOLO":
        """Lazy-load the YOLO model."""
        if self._model is not None:
            return self._model

        if not self.is_available():
            raise MLDependencyError("ultralytics is not installed. Install with: uv sync --extra ml")

        from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]

        # Get model path (downloads if necessary)
        model_path = self._model_path or get_yolo_model_path()

        self._model = YOLO(model_path)

        # Warn about CPU usage
        global _cpu_warning_shown
        if self.device == "cpu" and not _cpu_warning_shown:
            print(
                "Warning: Running YOLO on CPU. Inference will be slow (~2-5s/page). "
                "Use --device cuda or --device mps for faster inference.",
                file=sys.stderr,
            )
            _cpu_warning_shown = True

        return self._model

    def detect(self, image: Image.Image) -> DetectionResult:
        """Detect panels using YOLO model.

        Args:
            image: PIL Image to process

        Returns:
            DetectionResult with detected panels
        """
        start_time = time.perf_counter()

        model = self._ensure_model_loaded()

        # Run inference
        results = model.predict(
            source=image,
            conf=self._confidence_threshold,
            device=self.device,
            verbose=False,
        )

        # Parse results
        panels: List[Panel] = []
        confidences: List[float] = []

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for i, box in enumerate(boxes):  # pyright: ignore[reportArgumentType]
                    # YOLO returns xyxy format, convert to xywh
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = xyxy
                    x, y = int(x1), int(y1)
                    w, h = int(x2 - x1), int(y2 - y1)

                    conf = float(box.conf[0])
                    confidences.append(conf)

                    panels.append(
                        Panel(
                            id=f"p-{i}",
                            bbox=(x, y, w, h),
                            confidence=conf,
                        )
                    )

        # Compute page-level confidence from panel confidences
        page_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        processing_time = time.perf_counter() - start_time

        return DetectionResult(
            panels=panels,
            confidence=page_confidence,
            gutters=None,  # YOLO doesn't detect gutters
            processing_time=round(processing_time, 3),
        )
