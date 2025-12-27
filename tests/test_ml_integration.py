"""Integration tests requiring ML dependencies.

Run with: uv run --extra ml pytest tests/test_ml_integration.py

These tests are skipped if ML dependencies are not installed.
"""

import pytest
from PIL import Image, ImageDraw

from panelizer.ml import is_ml_available

# Skip all tests if ML not available
pytestmark = pytest.mark.skipif(
    not is_ml_available(),
    reason="ML dependencies not installed (uv sync --extra ml)",
)


class TestYOLODetectorReal:
    """Real YOLO detector tests (downloads model, uses actual inference)."""

    @pytest.fixture(scope="class")
    def detector(self):
        """Shared detector instance to avoid repeated model loading."""
        from panelizer.ml import YOLODetector

        return YOLODetector(device="cpu")  # Force CPU for CI/testing

    def test_detect_synthetic_panels(self, detector):
        """Test detection on synthetic image with clear panel borders."""
        # Create image with clear black borders on white background
        img = Image.new("RGB", (800, 1200), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Draw thick black panel borders (typical comic style)
        border_width = 5

        # Top-left panel
        draw.rectangle([50, 50, 380, 550], outline=(0, 0, 0), width=border_width)
        # Top-right panel
        draw.rectangle([420, 50, 750, 550], outline=(0, 0, 0), width=border_width)
        # Bottom panel (full width)
        draw.rectangle([50, 600, 750, 1150], outline=(0, 0, 0), width=border_width)

        result = detector.detect(img)

        # Should detect something (may not be exactly 3 on synthetic)
        assert isinstance(result.panels, list)
        assert result.source == "yolo" if hasattr(result, "source") else True
        assert result.processing_time is not None
        assert result.processing_time > 0

    def test_detect_blank_page(self, detector):
        """Test on blank white page."""
        img = Image.new("RGB", (800, 600), color=(255, 255, 255))
        result = detector.detect(img)

        # May or may not find panels on blank page
        assert isinstance(result.panels, list)
        assert 0.0 <= result.confidence <= 1.0

    def test_detect_solid_black_page(self, detector):
        """Test on solid black page."""
        img = Image.new("RGB", (800, 600), color=(0, 0, 0))
        result = detector.detect(img)

        assert isinstance(result.panels, list)
        assert isinstance(result.confidence, float)

    def test_detect_returns_valid_bboxes(self, detector):
        """Verify detected bboxes are valid (x, y, w, h format)."""
        img = Image.new("RGB", (800, 600), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Draw a clear panel
        draw.rectangle([100, 100, 700, 500], outline=(0, 0, 0), width=8)

        result = detector.detect(img)

        for panel in result.panels:
            x, y, w, h = panel.bbox
            assert x >= 0
            assert y >= 0
            assert w > 0
            assert h > 0
            assert x + w <= 800
            assert y + h <= 600

    def test_detect_returns_valid_confidence(self, detector):
        """Verify confidence scores are in valid range."""
        img = Image.new("RGB", (800, 600), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([100, 100, 700, 500], outline=(0, 0, 0), width=5)

        result = detector.detect(img)

        for panel in result.panels:
            assert 0.0 <= panel.confidence <= 1.0

    def test_detector_properties(self, detector):
        """Test detector property methods."""
        assert detector.is_available() is True
        assert "YOLO" in detector.model_name
        assert detector.device in ("cuda", "mps", "cpu")


class TestModelDownload:
    """Test model download functionality."""

    def test_model_downloads_on_first_use(self, detector):
        """Verify model is downloaded/cached on first detect() call."""
        # The detector fixture already ran detect(), so model should be loaded
        assert detector._model is not None

    @pytest.fixture(scope="class")
    def detector(self):
        """Create detector for download tests."""
        from panelizer.ml import YOLODetector

        detector = YOLODetector(device="cpu")
        # Trigger model load
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        detector.detect(img)
        return detector
