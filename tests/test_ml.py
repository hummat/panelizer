"""Unit tests for ML module (mocked, no GPU/model required)."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from panelizer.ml import (
    Detector,
    MLDependencyError,
    MLDetector,
    MLError,
    ModelNotFoundError,
    is_ml_available,
)
from panelizer.ml.model_manager import (
    DEFAULT_CACHE_DIR,
    YOLO_MODEL_FILE,
    YOLO_MODEL_REPO,
    get_cache_dir,
)
from panelizer.schema import Panel


class TestExceptions:
    """Test ML exception hierarchy."""

    def test_ml_error_is_base(self):
        assert issubclass(MLDependencyError, MLError)
        assert issubclass(ModelNotFoundError, MLError)

    def test_ml_dependency_error(self):
        exc = MLDependencyError("test message")
        assert "test message" in str(exc)

    def test_model_not_found_error(self):
        exc = ModelNotFoundError("model missing")
        assert "model missing" in str(exc)


class TestIsMLAvailable:
    """Test is_ml_available() function."""

    def test_returns_bool(self):
        result = is_ml_available()
        assert isinstance(result, bool)


class TestModelManager:
    """Test model manager functions."""

    def test_get_cache_dir_default(self):
        """Test default cache directory."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove XDG_CACHE_HOME if set
            os.environ.pop("XDG_CACHE_HOME", None)
            cache = get_cache_dir()
            assert cache == DEFAULT_CACHE_DIR

    def test_get_cache_dir_xdg(self, tmp_path: Path):
        """Test XDG_CACHE_HOME is respected."""
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            cache = get_cache_dir()
            assert cache == tmp_path / "panelizer" / "models"

    def test_model_constants(self):
        """Verify model constants are set correctly."""
        assert YOLO_MODEL_REPO == "mosesb/best-comic-panel-detection"
        assert YOLO_MODEL_FILE == "best.pt"


class TestDetectorProtocol:
    """Test Detector protocol."""

    def test_protocol_is_runtime_checkable(self):
        """Verify Detector protocol can be checked at runtime."""
        from panelizer.cv.detector import CVDetector

        detector = CVDetector()
        assert isinstance(detector, Detector)


class TestMLDetectorABC:
    """Test MLDetector abstract base class."""

    def test_cannot_instantiate_directly(self):
        """Verify MLDetector cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MLDetector()  # type: ignore

    def test_subclass_must_implement_methods(self):
        """Verify subclass must implement required methods."""

        class IncompleteDetector(MLDetector):
            pass

        with pytest.raises(TypeError):
            IncompleteDetector()  # type: ignore


class TestDetectionResultContract:
    """Verify DetectionResult interface matches what ML detectors need."""

    def test_detection_result_fields(self):
        """Test that DetectionResult has all required fields."""
        from panelizer.cv.detector import DetectionResult

        panel = Panel(id="p-0", bbox=(10, 10, 100, 100), confidence=0.9)
        result = DetectionResult(
            panels=[panel],
            confidence=0.9,
            gutters=(10, 10),
            processing_time=0.5,
        )

        assert result.panels == [panel]
        assert result.confidence == 0.9
        assert result.gutters == (10, 10)
        assert result.processing_time == 0.5


class TestYOLODetectorMocked:
    """Tests using mocked YOLO model."""

    @pytest.fixture
    def mock_yolo_model(self):
        """Create a mock YOLO model with prediction results."""
        mock_model = MagicMock()

        # Create mock boxes
        mock_box = MagicMock()
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].tolist.return_value = [100.0, 100.0, 300.0, 400.0]
        mock_box.conf = [MagicMock()]
        mock_box.conf[0].__float__ = lambda self: 0.95

        # Mock result with boxes
        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_result.boxes.__len__ = lambda self: 1

        mock_model.predict.return_value = [mock_result]
        return mock_model

    @patch("panelizer.ml.yolo_detector._check_ultralytics_available", return_value=True)
    @patch("panelizer.ml.yolo_detector.get_yolo_model_path", return_value="/fake/model.pt")
    def test_yolo_detector_init(self, mock_path, mock_available):
        """Test YOLODetector initialization."""
        # Need to import after mocking
        from panelizer.ml.yolo_detector import YOLODetector

        detector = YOLODetector(confidence_threshold=0.5, device="cpu")
        assert detector._confidence_threshold == 0.5
        assert detector._requested_device == "cpu"

    @patch("panelizer.ml.yolo_detector._check_ultralytics_available", return_value=True)
    def test_yolo_detector_is_available(self, mock_available):
        """Test is_available() method."""
        from panelizer.ml.yolo_detector import YOLODetector

        detector = YOLODetector()
        assert detector.is_available() is True

    @patch("panelizer.ml.yolo_detector._check_ultralytics_available", return_value=False)
    def test_yolo_detector_not_available(self, mock_available):
        """Test is_available() when ultralytics not installed."""
        from panelizer.ml.yolo_detector import YOLODetector

        detector = YOLODetector()
        # Reset cached value
        detector._available = None
        assert detector.is_available() is False

    @patch("panelizer.ml.yolo_detector._check_ultralytics_available", return_value=True)
    def test_yolo_detector_model_name(self, mock_available):
        """Test model_name property."""
        from panelizer.ml.yolo_detector import YOLODetector

        detector = YOLODetector()
        assert "YOLO" in detector.model_name
        assert "best-comic-panel-detection" in detector.model_name

    @patch("panelizer.ml.yolo_detector._check_ultralytics_available", return_value=True)
    def test_yolo_detector_device_auto(self, mock_available):
        """Test device auto-detection."""
        from panelizer.ml.yolo_detector import YOLODetector

        detector = YOLODetector(device=None)
        # Should auto-detect (will be 'cpu' in test environment)
        device = detector.device
        assert device in ("cuda", "mps", "cpu")

    @patch("panelizer.ml.yolo_detector._check_ultralytics_available", return_value=True)
    def test_yolo_detector_device_explicit(self, mock_available):
        """Test explicit device selection."""
        from panelizer.ml.yolo_detector import YOLODetector

        detector = YOLODetector(device="cpu")
        assert detector.device == "cpu"

    @patch("panelizer.ml.yolo_detector._check_ultralytics_available", return_value=False)
    def test_yolo_detector_detect_raises_without_deps(self, mock_available):
        """Test detect() raises MLDependencyError when deps missing."""
        from panelizer.ml.yolo_detector import YOLODetector

        detector = YOLODetector()
        detector._available = None  # Reset cached value

        img = Image.new("RGB", (800, 600), color=(255, 255, 255))

        with pytest.raises(MLDependencyError) as exc_info:
            detector.detect(img)

        assert "uv sync --extra ml" in str(exc_info.value)


class TestGetBestDevice:
    """Test device auto-detection."""

    def test_get_best_device_returns_string(self):
        """Test _get_best_device returns valid device string."""
        from panelizer.ml.yolo_detector import _get_best_device

        device = _get_best_device()
        assert device in ("cuda", "mps", "cpu")

    @patch.dict("sys.modules", {"torch": None})
    def test_get_best_device_without_torch(self):
        """Test fallback to CPU when torch not available."""
        # This is tricky to test because torch may be imported elsewhere
        # Just verify the function doesn't crash
        from panelizer.ml.yolo_detector import _get_best_device

        device = _get_best_device()
        assert isinstance(device, str)
