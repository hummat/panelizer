from panelizer.ordering import order_panels, same_col, same_row
from panelizer.schema import ReadingDirection


class TestSameCol:
    def test_same_col_no_overlap(self) -> None:
        # Two panels completely separate horizontally
        bbox1 = (0, 0, 100, 100)  # x=0 to x=100
        bbox2 = (200, 0, 100, 100)  # x=200 to x=300 (no overlap)
        assert not same_col(bbox1, bbox2)

    def test_same_col_one_contains_other(self) -> None:
        # bbox1 contains bbox2 horizontally
        bbox1 = (0, 0, 300, 100)  # x=0 to x=300
        bbox2 = (50, 200, 100, 100)  # x=50 to x=150 (contained)
        assert same_col(bbox1, bbox2)


class TestSameRow:
    def test_same_row_when_first_below_second(self) -> None:
        # bbox1 is below bbox2 vertically but at same y level (triggers swap)
        bbox1 = (0, 100, 100, 50)  # y=100
        bbox2 = (200, 50, 100, 100)  # y=50, extends to y=150
        assert same_row(bbox1, bbox2)

    def test_same_row_when_one_contains_other(self) -> None:
        # bbox2 is completely inside bbox1 vertically (triggers containment check)
        bbox1 = (0, 0, 100, 200)  # y=0 to y=200
        bbox2 = (200, 50, 100, 50)  # y=50 to y=100 (contained)
        assert same_row(bbox1, bbox2)


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
