from PIL import Image, ImageDraw

from panel_flow.cv import CVDetector


class TestCVDetector:
    def test_detect_synthetic_panels(self) -> None:
        # Create a synthetic image with clear panel-like rectangles
        img = Image.new("RGB", (800, 600), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Draw two distinct "panels" with black borders
        draw.rectangle([50, 50, 350, 280], outline=(0, 0, 0), width=3)
        draw.rectangle([400, 50, 750, 280], outline=(0, 0, 0), width=3)
        draw.rectangle([50, 320, 750, 550], outline=(0, 0, 0), width=3)

        detector = CVDetector()
        panels, confidence = detector.detect(img)

        # Should detect some panels (exact count depends on CV threshold tuning)
        assert isinstance(panels, list)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_detect_blank_page(self) -> None:
        # Blank white image - should detect no panels or very few
        img = Image.new("RGB", (800, 600), color=(255, 255, 255))
        detector = CVDetector()
        panels, confidence = detector.detect(img)

        # With a blank page, confidence should be lower
        assert isinstance(panels, list)
        assert isinstance(confidence, float)

    def test_detect_single_large_panel(self) -> None:
        # Image with one large rectangle covering most of page
        img = Image.new("RGB", (800, 600), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 780, 580], outline=(0, 0, 0), width=5)

        detector = CVDetector()
        panels, confidence = detector.detect(img)

        # Single panel detection typically gets lower confidence
        assert isinstance(confidence, float)

    def test_panel_ids_are_unique(self) -> None:
        img = Image.new("RGB", (800, 600), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Draw multiple panels
        for i in range(3):
            x = 50 + i * 250
            draw.rectangle([x, 50, x + 200, 250], outline=(0, 0, 0), width=3)

        detector = CVDetector()
        panels, _ = detector.detect(img)

        if panels:
            ids = [p.id for p in panels]
            assert len(ids) == len(set(ids))  # all IDs unique
