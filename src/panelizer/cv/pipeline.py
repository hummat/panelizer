import math
from typing import Callable, Dict, List, Tuple

import cv2 as cv
import numpy as np

from .panel_internal import InternalPanel
from .segment import Segment


def sobel_edges(gray: np.ndarray) -> np.ndarray:
    """Apply Sobel edge detection to grayscale image."""
    ddepth = cv.CV_16S
    grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    sobel = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return sobel


def get_contours(sobel: np.ndarray) -> List[np.ndarray]:
    """Threshold and find contours from Sobel edge image."""
    _, thresh = cv.threshold(sobel, 100, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2:]
    return list(contours)


def detect_segments(gray: np.ndarray, img_size: Tuple[int, int], min_panel_ratio: float) -> List[Segment]:
    """
    Detect line segments using LSD (Line Segment Detector).
    Returns segments longer than minimum panel size threshold.
    """
    lsd = cv.createLineSegmentDetector(0)
    dlines = lsd.detect(gray)

    min_dist = min(img_size) * min_panel_ratio
    segments = []

    # Cap segments at 500 to avoid performance issues
    while len(segments) == 0 or len(segments) > 500:
        segments = []

        if dlines is None or dlines[0] is None:
            break

        for dline in dlines[0]:
            x0 = int(round(dline[0][0]))
            y0 = int(round(dline[0][1]))
            x1 = int(round(dline[0][2]))
            y1 = int(round(dline[0][3]))

            a = x0 - x1
            b = y0 - y1
            dist = math.sqrt(a**2 + b**2)
            if dist >= min_dist:
                segments.append(Segment((x0, y0), (x1, y1)))

        # If we got too many segments, raise threshold
        if len(segments) > 500:
            min_dist *= 1.1
        else:
            break

    return segments


def initial_panels(
    contours: List[np.ndarray], img_size: Tuple[int, int], min_panel_ratio: float
) -> List[InternalPanel]:
    """
    Convert contours to initial panel polygons.
    Filters out very small contours.
    """
    panels = []
    for contour in contours:
        arclength = cv.arcLength(contour, True)
        epsilon = 0.001 * arclength
        approx = cv.approxPolyDP(contour, epsilon, True)

        panel = InternalPanel(img_size, min_panel_ratio, polygon=approx)
        if panel.is_very_small():
            continue

        panels.append(panel)

    return panels


def split_panels(panels: List[InternalPanel], segments: List[Segment]) -> Tuple[List[InternalPanel], List[float]]:
    """
    Iteratively split panels using detected segments.
    Continues until no more panels can be split.
    """
    split_coverages: List[float] = []
    did_split = True
    while did_split:
        did_split = False
        # Sort by area descending to split largest panels first
        for p in sorted(panels, key=lambda p: p.area(), reverse=True):
            split = p.split(segments)
            if split is not None:
                did_split = True
                panels.remove(p)
                panels += split.subpanels
                split_coverages.append(split.segments_coverage())
                break

    return panels, split_coverages


def exclude_small(panels: List[InternalPanel], min_panel_ratio: float) -> List[InternalPanel]:
    """Filter out panels smaller than minimum size threshold."""
    return [p for p in panels if not p.is_small()]


def merge_panels(panels: List[InternalPanel]) -> List[InternalPanel]:
    """
    Merge panels where one contains more than 50% of another.
    Handles speech bubbles or text that split panels incorrectly.
    """
    panels_to_remove = []
    for i, p1 in enumerate(panels):
        for p2 in panels[i + 1 :]:
            if p1.contains(p2):
                panels_to_remove.append(p2)
                p1 = p1.merge(p2, panels)
            elif p2.contains(p1):
                panels_to_remove.append(p1)
                p2 = p2.merge(p1, panels)

    for p in set(panels_to_remove):
        if p in panels:
            panels.remove(p)

    return panels


def deoverlap_panels(panels: List[InternalPanel]) -> List[InternalPanel]:
    """
    Fix slight overlaps from panel splitting.
    Adjusts boundaries to eliminate overlaps.
    """
    for p1 in panels:
        for p2 in panels:
            if p1 == p2:
                continue

            opanel = p1.overlap_panel(p2)
            if not opanel:
                continue

            # Adjust vertical overlap
            if opanel.w() < opanel.h() and p1.r == opanel.r:
                p1.r = opanel.x
                p2.x = opanel.r
                continue

            # Adjust horizontal overlap
            if opanel.w() > opanel.h() and p1.b == opanel.b:
                p1.b = opanel.y
                p2.y = opanel.b
                continue

    return panels


def group_small_panels(panels: List[InternalPanel]) -> List[InternalPanel]:
    """
    Group small panels that are close together into bigger ones (Kumiko).

    Uses convex hull of grouped polygons, and marks the grouped panel as non-splittable.
    """
    small_panels = [p for p in panels if p.is_small()]
    if len(small_panels) < 2:
        return panels

    parent = list(range(len(small_panels)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[rj] = ri

    for i, p1 in enumerate(small_panels):
        for j in range(i + 1, len(small_panels)):
            p2 = small_panels[j]
            if p1.is_close(p2):
                union(i, j)

    groups: Dict[int, List[InternalPanel]] = {}
    for i, p in enumerate(small_panels):
        root = find(i)
        groups.setdefault(root, []).append(p)

    grouped_panels: List[InternalPanel] = []
    to_remove: List[InternalPanel] = []

    for group in groups.values():
        if len(group) < 2:
            continue

        polygons = []
        for p in group:
            if p.polygon is None:
                x, y, w, h = p.to_xywh()
                polygons.append(np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]], dtype=np.int32))
            else:
                polygons.append(p.polygon)

        big_hull = cv.convexHull(np.concatenate(polygons))
        big_panel = InternalPanel(group[0].img_size, group[0].small_panel_ratio, polygon=big_hull, splittable=False)
        grouped_panels.append(big_panel)
        to_remove.extend(group)

    if not grouped_panels:
        return panels

    remaining = [p for p in panels if p not in to_remove]
    return remaining + grouped_panels


def actual_gutters(panels: List[InternalPanel], func: Callable[[List[int]], int] = min) -> Dict[str, int]:
    """
    Estimate gutters between panels (Kumiko).

    Returns {"x","y","r","b"} where "r"/"b" are the negative gutter values used for expansion.
    """
    gutters_x: List[int] = []
    gutters_y: List[int] = []

    for p in panels:
        left_panel = p.find_left_panel(panels)
        if left_panel:
            gutters_x.append(p.x - left_panel.r)

        top_panel = p.find_top_panel(panels)
        if top_panel:
            gutters_y.append(p.y - top_panel.b)

    if not gutters_x:
        gutters_x = [1]
    if not gutters_y:
        gutters_y = [1]

    gx = func(gutters_x)
    gy = func(gutters_y)
    return {"x": gx, "y": gy, "r": -gx, "b": -gy}


def expand_panels(panels: List[InternalPanel]) -> List[InternalPanel]:
    """Expand panels to their neighbour's edge or the detected frame around all panels (Kumiko)."""
    if not panels:
        return panels

    gutters = actual_gutters(panels)
    opposite = {"x": "r", "r": "x", "y": "b", "b": "y"}

    for p in panels:
        for d in ("x", "y", "r", "b"):
            neighbour = p.find_neighbour_panel(d, panels)
            if neighbour is not None:
                newcoord = getattr(neighbour, opposite[d]) + gutters[d]
            else:
                if d in ("x", "y"):
                    min_panel = min(panels, key=lambda q: getattr(q, d))
                else:
                    min_panel = max(panels, key=lambda q: getattr(q, d))
                newcoord = getattr(min_panel, d)

            if d in ("r", "b"):
                if newcoord > getattr(p, d):
                    setattr(p, d, newcoord)
            else:
                if newcoord < getattr(p, d):
                    setattr(p, d, newcoord)

    return panels


def group_big_panels(panels: List[InternalPanel], segments: List[Segment]) -> List[InternalPanel]:
    """Group big panels together when the union doesn't bump and has no strong gutter segments (Kumiko)."""
    grouped = True
    while grouped:
        grouped = False
        for i, p1 in enumerate(panels):
            for p2 in panels[i + 1 :]:
                p3 = p1.group_with(p2)

                other_panels = [p for p in panels if p not in [p1, p2]]
                if p3.bumps_into(other_panels):
                    continue

                big_segments: List[Segment] = []
                for s in segments:
                    if p3.contains_segment(s) and s.dist() > p3.diagonal().dist() / 5:
                        if s not in big_segments:
                            big_segments.append(s)

                if big_segments:
                    continue

                panels.append(p3)
                panels.remove(p1)
                panels.remove(p2)
                grouped = True
                break

            if grouped:
                break

    return panels


def detect_panels(
    img: np.ndarray,
    min_panel_ratio: float = 0.1,
    *,
    panel_expansion: bool = True,
    small_panel_grouping: bool = True,
    big_panel_grouping: bool = True,
) -> Tuple[List[InternalPanel], List[float]]:
    """
    Main detection pipeline.
    Returns panels and list of split coverage scores for confidence calculation.
    """
    img_size = (img.shape[1], img.shape[0])  # (width, height)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 1. Sobel edge detection
    sobel = sobel_edges(gray)

    # 2. Threshold + contours
    contours = get_contours(sobel)

    # 3. LSD gutter detection
    segments = detect_segments(gray, img_size, min_panel_ratio)
    segments = Segment.union_all(segments)

    # 4. Initial panels from contours
    panels = initial_panels(contours, img_size, min_panel_ratio)

    if small_panel_grouping:
        panels = group_small_panels(panels)

    # 5. Refinement passes
    # Split panels using segments
    panels, split_coverages = split_panels(panels, segments)

    panels = exclude_small(panels, min_panel_ratio)
    panels = merge_panels(panels)
    panels = deoverlap_panels(panels)
    panels = exclude_small(panels, min_panel_ratio)

    if panel_expansion:
        panels = expand_panels(panels)

    # Fallback: if no panels detected, return full page as single panel
    if not panels:
        panels.append(InternalPanel(img_size, min_panel_ratio, xywh=(0, 0, img_size[0], img_size[1])))

    if big_panel_grouping:
        panels = group_big_panels(panels, segments)

    return panels, split_coverages
