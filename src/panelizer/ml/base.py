"""Abstract base classes for ML detectors."""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from PIL import Image

from ..cv.detector import DetectionResult


@runtime_checkable
class Detector(Protocol):
    """Protocol for all panel detectors (CV, YOLO, SAM).

    This protocol enables duck typing - any class with a matching
    detect() method can be used as a detector.
    """

    def detect(self, image: Image.Image) -> DetectionResult:
        """Detect panels in an image.

        Args:
            image: PIL Image to process

        Returns:
            DetectionResult with panels, confidence, and metadata
        """
        ...


class MLDetector(ABC):
    """Abstract base class for ML-based panel detectors.

    Subclasses must implement:
    - detect(): Run inference and return DetectionResult
    - is_available(): Check if backend dependencies are installed
    - model_name: Human-readable model identifier
    """

    @abstractmethod
    def detect(self, image: Image.Image) -> DetectionResult:
        """Detect panels using ML model.

        Args:
            image: PIL Image to process

        Returns:
            DetectionResult with detected panels
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the ML backend is available.

        Returns:
            True if required dependencies are installed
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...
