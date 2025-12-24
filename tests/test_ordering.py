from panel_flow.ordering import order_panels
from panel_flow.schema import ReadingDirection


class TestOrderPanels:
    def test_empty(self) -> None:
        assert order_panels([]) == []

    def test_single_panel(self) -> None:
        bboxes = [(100, 100, 200, 200)]
        assert order_panels(bboxes) == [0]

    def test_ltr_row(self) -> None:
        # Two panels side by side, should be ordered left to right
        bboxes = [
            (300, 100, 200, 200),  # right panel
            (50, 100, 200, 200),  # left panel
        ]
        result = order_panels(bboxes, ReadingDirection.LTR)
        assert result == [1, 0]  # left first, then right

    def test_rtl_row(self) -> None:
        # Two panels side by side, should be ordered right to left
        bboxes = [
            (300, 100, 200, 200),  # right panel
            (50, 100, 200, 200),  # left panel
        ]
        result = order_panels(bboxes, ReadingDirection.RTL)
        assert result == [0, 1]  # right first, then left

    def test_two_rows_ltr(self) -> None:
        # Top row: two panels, bottom row: one panel
        bboxes = [
            (50, 10, 100, 100),  # top-left
            (200, 10, 100, 100),  # top-right
            (100, 200, 100, 100),  # bottom-center
        ]
        result = order_panels(bboxes, ReadingDirection.LTR)
        # Top row first (L->R), then bottom row
        assert result == [0, 1, 2]

    def test_default_direction_is_ltr(self) -> None:
        bboxes = [
            (300, 100, 200, 200),
            (50, 100, 200, 200),
        ]
        result = order_panels(bboxes)  # default direction
        assert result == [1, 0]  # LTR order
