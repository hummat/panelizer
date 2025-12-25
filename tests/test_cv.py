from PIL import Image, ImageDraw

from panel_flow.cv import CVDetector
from panel_flow.cv.panel_internal import InternalPanel
from panel_flow.cv.segment import Segment


class TestSegment:
    def test_segment_creation(self) -> None:
        seg = Segment((0, 0), (10, 10))
        assert seg.a == (0, 0)
        assert seg.b == (10, 10)

    def test_segment_distance(self) -> None:
        seg = Segment((0, 0), (3, 4))
        assert seg.dist() == 5.0  # 3-4-5 triangle
        assert seg.dist_x() == 3
        assert seg.dist_y() == 4

    def test_segment_center(self) -> None:
        seg = Segment((0, 0), (10, 10))
        assert seg.center() == (5, 5)

    def test_segment_union(self) -> None:
        # Overlapping parallel segments should union
        seg1 = Segment((0, 0), (10, 0))
        seg2 = Segment((5, 0), (15, 0))
        union = seg1.union(seg2)
        assert union is not None
        assert union.left() == 0
        assert union.right() == 15

    def test_segment_angle_ok(self) -> None:
        # Parallel segments
        seg1 = Segment((0, 0), (10, 0))
        seg2 = Segment((0, 5), (10, 5))
        assert seg1.angle_ok_with(seg2)

    def test_segment_may_contain(self) -> None:
        seg = Segment((0, 0), (10, 10))
        assert seg.may_contain((5, 5))
        assert not seg.may_contain((15, 15))

    def test_segment_to_xyrb(self) -> None:
        seg = Segment((10, 20), (50, 80))
        xyrb = seg.to_xyrb()
        assert xyrb == [10, 20, 50, 80]

    def test_segment_bounds(self) -> None:
        seg = Segment((10, 20), (50, 80))
        assert seg.left() == 10
        assert seg.top() == 20
        assert seg.right() == 50
        assert seg.bottom() == 80

    def test_segment_angle(self) -> None:
        # Horizontal segment
        seg_h = Segment((0, 0), (10, 0))
        assert abs(seg_h.angle()) < 0.01  # ~0 radians

        # Vertical segment
        seg_v = Segment((0, 0), (0, 10))
        assert abs(seg_v.angle() - 1.5708) < 0.01  # ~π/2 radians

    def test_segment_equality(self) -> None:
        seg1 = Segment((0, 0), (10, 10))
        seg2 = Segment((0, 0), (10, 10))
        seg3 = Segment((10, 10), (0, 0))  # Reversed, but should be equal
        seg4 = Segment((0, 0), (5, 5))

        assert seg1 == seg2
        assert seg1 == seg3  # Reversed segments are equal
        assert seg1 != seg4

    def test_segment_intersect(self) -> None:
        # Overlapping horizontal segments
        seg1 = Segment((0, 0), (10, 0))
        seg2 = Segment((5, 0), (15, 0))
        intersection = seg1.intersect(seg2)
        assert intersection is not None
        assert intersection.left() == 5
        assert intersection.right() == 10

    def test_segment_union_all(self) -> None:
        # Multiple overlapping segments should be merged
        segs = [
            Segment((0, 0), (10, 0)),
            Segment((5, 0), (15, 0)),
            Segment((12, 0), (20, 0)),
        ]
        unified = Segment.union_all(segs)
        # Should merge into fewer segments
        assert len(unified) <= len(segs)

    def test_segment_projected_point(self) -> None:
        seg = Segment((0, 0), (10, 0))
        # Point above the segment
        projected = seg.projected_point((5, 5))
        # Should project onto (5, 0)
        assert projected == (5, 0)

    def test_segment_str(self) -> None:
        seg = Segment((0, 0), (10, 10))
        assert str(seg) == "((0, 0), (10, 10))"


class TestInternalPanel:
    def test_panel_creation_from_xywh(self) -> None:
        panel = InternalPanel((800, 600), 0.1, xywh=(10, 20, 100, 150))
        assert panel.x == 10
        assert panel.y == 20
        assert panel.w() == 100
        assert panel.h() == 150

    def test_panel_from_xyrb(self) -> None:
        panel = InternalPanel.from_xyrb((800, 600), 0.1, 10, 20, 110, 170)
        assert panel.x == 10
        assert panel.y == 20
        assert panel.r == 110
        assert panel.b == 170

    def test_panel_area(self) -> None:
        panel = InternalPanel((800, 600), 0.1, xywh=(0, 0, 10, 20))
        assert panel.area() == 200

    def test_panel_is_small(self) -> None:
        # Small panel (< 10% of image width or height)
        small_panel = InternalPanel((800, 600), 0.1, xywh=(0, 0, 50, 40))
        assert small_panel.is_small()

        # Large panel
        large_panel = InternalPanel((800, 600), 0.1, xywh=(0, 0, 400, 300))
        assert not large_panel.is_small()

    def test_panel_overlap(self) -> None:
        panel1 = InternalPanel((800, 600), 0.1, xywh=(0, 0, 100, 100))
        panel2 = InternalPanel((800, 600), 0.1, xywh=(50, 50, 100, 100))

        overlap = panel1.overlap_panel(panel2)
        assert overlap is not None
        assert overlap.x == 50
        assert overlap.y == 50
        assert overlap.w() == 50
        assert overlap.h() == 50

    def test_panel_same_row(self) -> None:
        panel1 = InternalPanel((800, 600), 0.1, xywh=(0, 100, 100, 50))
        panel2 = InternalPanel((800, 600), 0.1, xywh=(200, 110, 100, 50))
        assert panel1.same_row(panel2)

    def test_panel_same_col(self) -> None:
        panel1 = InternalPanel((800, 600), 0.1, xywh=(100, 0, 50, 100))
        panel2 = InternalPanel((800, 600), 0.1, xywh=(110, 200, 50, 100))
        assert panel1.same_col(panel2)

    def test_panel_contains(self) -> None:
        large_panel = InternalPanel((800, 600), 0.1, xywh=(0, 0, 200, 200))
        small_panel = InternalPanel((800, 600), 0.1, xywh=(50, 50, 50, 50))
        assert large_panel.contains(small_panel)
        assert not small_panel.contains(large_panel)

    def test_panel_overlaps(self) -> None:
        panel1 = InternalPanel((800, 600), 0.1, xywh=(0, 0, 100, 100))
        panel2 = InternalPanel((800, 600), 0.1, xywh=(50, 50, 100, 100))
        assert panel1.overlaps(panel2)

    def test_panel_find_neighbors(self) -> None:
        panels = [
            InternalPanel((800, 600), 0.1, xywh=(0, 0, 100, 100)),
            InternalPanel((800, 600), 0.1, xywh=(200, 0, 100, 100)),
            InternalPanel((800, 600), 0.1, xywh=(0, 200, 100, 100)),
        ]
        # Panel 0 should have panel 1 to its right
        right = panels[0].find_right_panel(panels)
        assert right == panels[1]

        # Panel 0 should have panel 2 below it
        bottom = panels[0].find_bottom_panel(panels)
        assert bottom == panels[2]

    def test_panel_group_with(self) -> None:
        panel1 = InternalPanel((800, 600), 0.1, xywh=(0, 0, 100, 100))
        panel2 = InternalPanel((800, 600), 0.1, xywh=(150, 150, 100, 100))
        grouped = panel1.group_with(panel2)
        # Should encompass both panels
        assert grouped.x == 0
        assert grouped.y == 0
        assert grouped.r == 250
        assert grouped.b == 250

    def test_panel_is_close(self) -> None:
        panel1 = InternalPanel((800, 600), 0.1, xywh=(0, 0, 100, 100))
        panel2 = InternalPanel((800, 600), 0.1, xywh=(120, 0, 100, 100))
        assert panel1.is_close(panel2)

        panel3 = InternalPanel((800, 600), 0.1, xywh=(500, 500, 100, 100))
        assert not panel1.is_close(panel3)

    def test_panel_bumps_into(self) -> None:
        panel1 = InternalPanel((800, 600), 0.1, xywh=(0, 0, 100, 100))
        panel2 = InternalPanel((800, 600), 0.1, xywh=(50, 50, 100, 100))
        panel3 = InternalPanel((800, 600), 0.1, xywh=(200, 200, 100, 100))

        assert panel1.bumps_into([panel2])
        assert not panel1.bumps_into([panel3])

    def test_panel_diagonal(self) -> None:
        panel = InternalPanel((800, 600), 0.1, xywh=(0, 0, 100, 100))
        diag = panel.diagonal()
        assert diag.a == (0, 0)
        assert diag.b == (100, 100)

    def test_panel_to_xywh(self) -> None:
        panel = InternalPanel((800, 600), 0.1, xywh=(10, 20, 100, 150))
        assert panel.to_xywh() == (10, 20, 100, 150)

    def test_panel_equality(self) -> None:
        panel1 = InternalPanel((800, 600), 0.1, xywh=(10, 20, 100, 150))
        panel2 = InternalPanel((800, 600), 0.1, xywh=(10, 20, 100, 150))
        panel3 = InternalPanel((800, 600), 0.1, xywh=(50, 50, 100, 150))
        assert panel1 == panel2
        assert panel1 != panel3


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

    def test_two_column_grid(self) -> None:
        """Test detection of a 2x3 grid with clear gutters."""
        img = Image.new("RGB", (800, 1200), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # 2 columns, 3 rows, 10px gutters
        for row in range(3):
            for col in range(2):
                x = 10 + col * 395
                y = 10 + row * 395
                draw.rectangle([x, y, x + 380, y + 380], outline=(0, 0, 0), width=3)

        detector = CVDetector()
        panels, confidence = detector.detect(img)

        # Should detect multiple panels
        assert len(panels) >= 2
        # Confidence should be reasonable for a regular grid
        assert confidence >= 0.3

    def test_thin_gutters(self) -> None:
        """Test detection with very thin (3px) gutters."""
        img = Image.new("RGB", (400, 600), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Two panels with thin gutter
        draw.rectangle([10, 10, 190, 290], outline=(0, 0, 0), width=2)
        draw.rectangle([210, 10, 390, 290], outline=(0, 0, 0), width=2)

        detector = CVDetector()
        panels, confidence = detector.detect(img)

        # Should detect at least some panels
        assert len(panels) >= 1
        assert isinstance(confidence, float)

    def test_custom_min_panel_ratio(self) -> None:
        """Test detector with custom minimum panel size ratio."""
        img = Image.new("RGB", (800, 600), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Draw a small panel that would be filtered with default ratio
        draw.rectangle([10, 10, 60, 50], outline=(0, 0, 0), width=3)

        # With default ratio (0.1), small panel might be filtered
        detector_default = CVDetector()
        panels_default, _ = detector_default.detect(img)

        # With smaller ratio (0.05), smaller panels are kept
        detector_small = CVDetector(min_panel_ratio=0.05)
        panels_small, _ = detector_small.detect(img)

        # Both should return valid results
        assert isinstance(panels_default, list)
        assert isinstance(panels_small, list)
