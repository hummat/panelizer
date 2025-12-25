import math
from typing import List, Tuple

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


def detect_segments(
    gray: np.ndarray, img_size: Tuple[int, int], min_panel_ratio: float
) -> List[Segment]:
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


def split_panels(panels: List[InternalPanel], segments: List[Segment]) -> List[InternalPanel]:
    """
    Iteratively split panels using detected segments.
    Continues until no more panels can be split.
    """
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
                break

    return panels


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


def detect_panels(
    img: np.ndarray, min_panel_ratio: float = 0.1
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

    # Track split coverage for confidence
    split_coverages = []

    # 5. Refinement passes
    # Split panels using segments
    original_panel_count = len(panels)
    panels = split_panels(panels, segments)
    # If we split any panels, record their coverage (simplified for now)
    if len(panels) > original_panel_count:
        split_coverages.append(0.7)  # Placeholder - actual coverage tracked in Split class

    panels = exclude_small(panels, min_panel_ratio)
    panels = merge_panels(panels)
    panels = deoverlap_panels(panels)

    # Fallback: if no panels detected, return full page as single panel
    if len(panels) == 0:
        panels.append(
            InternalPanel(img_size, min_panel_ratio, xywh=(0, 0, img_size[0], img_size[1]))
        )

    return panels, split_coverages
