from typing import List, Tuple

from .schema import ReadingDirection


def order_panels(
    bboxes: List[Tuple[int, int, int, int]], direction: ReadingDirection = ReadingDirection.LTR
) -> List[int]:
    """
    Orders panel indices based on a row-major heuristic.
    bboxes are (x, y, w, h).
    """
    if not bboxes:
        return []

    # Sort primarily by top (y) and secondarily by left (x)
    # To handle rows properly, we can group panels that are vertically overlapping significantly
    indices = list(range(len(bboxes)))

    # Simple approach: sort by Y first. If Y difference is small, sort by X.
    # A more robust approach would be recursive XY-cut, but let's start simple.
    avg_h = sum(b[3] for b in bboxes) / len(bboxes)
    row_threshold = avg_h * 0.5

    def sort_key(idx: int):
        x, y, w, h = bboxes[idx]
        # Quantize Y to group into rows
        row = y // row_threshold
        if direction == ReadingDirection.RTL:
            return (row, -x)
        return (row, x)

    return sorted(indices, key=sort_key)
