import numpy as np

from panel_flow.cv.panel_internal import InternalPanel
from panel_flow.cv.pipeline import expand_panels, group_big_panels, group_small_panels
from panel_flow.cv.segment import Segment


def rect_polygon(x: int, y: int, w: int, h: int) -> np.ndarray:
    return np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]], dtype=np.int32)


class TestGroupSmallPanels:
    def test_groups_close_small_panels_into_one(self) -> None:
        img_size = (1000, 1000)
        small_ratio = 0.2

        p1 = InternalPanel(img_size, small_ratio, polygon=rect_polygon(10, 10, 50, 50))
        p2 = InternalPanel(img_size, small_ratio, polygon=rect_polygon(80, 10, 50, 50))

        grouped = group_small_panels([p1, p2])
        assert len(grouped) == 1
        assert grouped[0].splittable is False
        assert grouped[0].x <= 10
        assert grouped[0].r >= 130


class TestExpandPanels:
    def test_expands_to_neighbour_edges_using_min_gutter(self) -> None:
        img_size = (600, 300)
        ratio = 0.05

        # Horizontal row: p2 right edge is under-detected (should expand towards p3).
        p1 = InternalPanel.from_xyrb(img_size, ratio, 200, 0, 290, 100)  # r=290
        p2 = InternalPanel.from_xyrb(img_size, ratio, 300, 0, 380, 100)  # r=380 (should expand to 390)
        p3 = InternalPanel.from_xyrb(img_size, ratio, 400, 0, 490, 100)  # x=400

        # Vertical column: p4 bottom edge is under-detected (should expand towards p5).
        p4 = InternalPanel.from_xyrb(img_size, ratio, 0, 0, 90, 80)  # b=80 (should expand to 90)
        p5 = InternalPanel.from_xyrb(img_size, ratio, 0, 100, 90, 180)  # y=100
        p6 = InternalPanel.from_xyrb(img_size, ratio, 0, 190, 90, 270)  # provides min gutter=10

        expanded = expand_panels([p1, p2, p3, p4, p5, p6])

        # min horizontal gutter from p1->p2 is 10 => p2.r should become p3.x - 10 = 390
        assert p2.r == 390

        # min vertical gutter from p5->p6 is 10 => p4.b should become p5.y - 10 = 90
        assert p4.b == 90
        assert expanded


class TestGroupBigPanels:
    def test_groups_when_no_strong_segments_inside_union(self) -> None:
        img_size = (400, 200)
        ratio = 0.05
        p1 = InternalPanel.from_xyrb(img_size, ratio, 0, 0, 100, 100)
        p2 = InternalPanel.from_xyrb(img_size, ratio, 110, 0, 210, 100)

        grouped = group_big_panels([p1, p2], segments=[])
        assert len(grouped) == 1
        assert grouped[0].x == 0
        assert grouped[0].r == 210

    def test_does_not_group_when_strong_segment_present(self) -> None:
        img_size = (400, 200)
        ratio = 0.05
        p1 = InternalPanel.from_xyrb(img_size, ratio, 0, 0, 100, 100)
        p2 = InternalPanel.from_xyrb(img_size, ratio, 110, 0, 210, 100)

        # Long segment inside the union should prevent grouping.
        barrier = Segment((105, 0), (105, 100))

        grouped = group_big_panels([p1, p2], segments=[barrier])
        assert len(grouped) == 2
