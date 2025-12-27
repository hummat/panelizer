"""Panel reading order using topological sort with spatial relationships.

Based on Kumiko's fix_panels_numbering algorithm which handles complex layouts
better than simple row-quantization.
"""

from typing import List, Optional, Tuple

from .schema import ReadingDirection

# Type alias for bounding boxes: (x, y, width, height)
BBox = Tuple[int, int, int, int]


def _to_xyrb(bbox: BBox) -> Tuple[int, int, int, int]:
    """Convert (x, y, w, h) to (x, y, right, bottom)."""
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def same_row(bbox1: BBox, bbox2: BBox) -> bool:
    """Check if two panels are in the same row (33% vertical overlap threshold).

    Two panels are considered in the same row if:
    - One is contained within the other vertically, OR
    - Their vertical intersection is at least 1/3 of the smaller panel's height
    """
    x1, y1, r1, b1 = _to_xyrb(bbox1)
    x2, y2, r2, b2 = _to_xyrb(bbox2)

    # Sort by y coordinate
    if y1 > y2:
        y1, b1, y2, b2 = y2, b2, y1, b1

    # Strictly above (no overlap)
    if y2 > b1:
        return False

    # One contained in the other
    if b2 < b1:
        return True

    # Check intersection ratio
    intersection_y = min(b1, b2) - y2
    min_h = min(b1 - y1, b2 - y2)
    return min_h == 0 or intersection_y / min_h >= 1 / 3


def same_col(bbox1: BBox, bbox2: BBox) -> bool:
    """Check if two panels are in the same column (33% horizontal overlap threshold).

    Two panels are considered in the same column if:
    - One is contained within the other horizontally, OR
    - Their horizontal intersection is at least 1/3 of the smaller panel's width
    """
    x1, y1, r1, b1 = _to_xyrb(bbox1)
    x2, y2, r2, b2 = _to_xyrb(bbox2)

    # Sort by x coordinate
    if x1 > x2:
        x1, r1, x2, r2 = x2, r2, x1, r1

    # Strictly left (no overlap)
    if x2 > r1:
        return False

    # One contained in the other
    if r2 < r1:
        return True

    # Check intersection ratio
    intersection_x = min(r1, r2) - x2
    min_w = min(r1 - x1, r2 - x2)
    return min_w == 0 or intersection_x / min_w >= 1 / 3


def find_top_panel(idx: int, bboxes: List[BBox]) -> Optional[int]:
    """Find the panel directly above the given panel (same column, closest bottom edge)."""
    bbox = bboxes[idx]
    _, y, _, _ = _to_xyrb(bbox)

    candidates = []
    for i, other in enumerate(bboxes):
        if i == idx:
            continue
        _, _, _, b = _to_xyrb(other)
        # Panel is above (its bottom <= our top) and in same column
        if b <= y and same_col(bbox, other):
            candidates.append((i, b))

    if not candidates:
        return None
    # Return the one with the largest bottom (closest to our top)
    return max(candidates, key=lambda x: x[1])[0]


def find_all_left_panels(idx: int, bboxes: List[BBox]) -> List[int]:
    """Find all panels to the left of the given panel in the same row."""
    bbox = bboxes[idx]
    x, _, _, _ = _to_xyrb(bbox)

    result = []
    for i, other in enumerate(bboxes):
        if i == idx:
            continue
        _, _, r, _ = _to_xyrb(other)
        # Panel is to the left (its right <= our left) and in same row
        if r <= x and same_row(bbox, other):
            result.append(i)
    return result


def find_all_right_panels(idx: int, bboxes: List[BBox]) -> List[int]:
    """Find all panels to the right of the given panel in the same row."""
    bbox = bboxes[idx]
    _, _, r, _ = _to_xyrb(bbox)

    result = []
    for i, other in enumerate(bboxes):
        if i == idx:
            continue
        x, _, _, _ = _to_xyrb(other)
        # Panel is to the right (its left >= our right) and in same row
        if x >= r and same_row(bbox, other):
            result.append(i)
    return result


def order_panels(bboxes: List[BBox], direction: ReadingDirection = ReadingDirection.LTR) -> List[int]:
    """Order panel indices using topological sort with spatial relationships.

    Uses Kumiko's fix_panels_numbering algorithm:
    - For each panel, find neighbors that must come before it
    - If any "before" neighbor comes after in current order, swap and restart
    - Repeat until no swaps needed

    For LTR: before neighbors = top panel + all left panels
    For RTL: before neighbors = top panel + all right panels

    Args:
        bboxes: List of (x, y, width, height) tuples
        direction: Reading direction (LTR or RTL)

    Returns:
        List of indices in reading order
    """
    if not bboxes:
        return []

    if len(bboxes) == 1:
        return [0]

    # Start with a simple sort as initial ordering
    # Sort by y first, then by x (or -x for RTL)
    indices = list(range(len(bboxes)))
    indices.sort(key=lambda i: (bboxes[i][1], bboxes[i][0] if direction == ReadingDirection.LTR else -bboxes[i][0]))

    # Iteratively fix ordering violations
    max_iterations = len(bboxes) * len(bboxes)  # Prevent infinite loops
    iteration = 0

    while iteration < max_iterations:
        changed = False
        iteration += 1

        for pos, idx in enumerate(indices):
            # Find all neighbors that should come before this panel
            neighbors_before = []

            # Top panel should come before
            top = find_top_panel(idx, bboxes)
            if top is not None:
                neighbors_before.append(top)

            # Left/right panels (depending on direction) should come before
            if direction == ReadingDirection.RTL:
                neighbors_before.extend(find_all_right_panels(idx, bboxes))
            else:
                neighbors_before.extend(find_all_left_panels(idx, bboxes))

            # Check if any neighbor comes after current position
            for neighbor in neighbors_before:
                neighbor_pos = indices.index(neighbor)
                if pos < neighbor_pos:
                    # Violation: neighbor should come before but comes after
                    # Move current panel after the neighbor
                    indices.insert(neighbor_pos, indices.pop(pos))
                    changed = True
                    break

            if changed:
                break  # Restart the whole loop with reordered indices

        if not changed:
            break

    return indices
