import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np

from .confidence import compute_line_variance
from .panel_internal import InternalPanel
from .segment import Segment

if TYPE_CHECKING:
    from .debug import DebugContext


@dataclass
class PipelineResult:
    """Result from detect_panels containing panels and cached intermediate data."""

    panels: List[InternalPanel]
    split_coverages: List[float]
    gray: np.ndarray  # Grayscale image (for confidence scoring)


def denoise(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Apply Gaussian blur to reduce halftone textures and scanner noise."""
    return cv.GaussianBlur(gray, (ksize, ksize), 0)


def sobel_edges(gray: np.ndarray) -> np.ndarray:
    """Apply Sobel edge detection to grayscale image."""
    ddepth = cv.CV_16S
    grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    sobel = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return sobel


def canny_edges(gray: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    """Apply Canny edge detection with hysteresis thresholding."""
    return cv.Canny(gray, low_threshold, high_threshold)


def morphological_close(edges: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Apply morphological closing to bridge small gaps in edges."""
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (ksize, ksize))
    return cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)


def get_contours(edges: np.ndarray, use_otsu: bool = True) -> List[np.ndarray]:
    """Threshold and find contours from edge image.

    Args:
        edges: Edge image (Sobel or Canny output)
        use_otsu: If True, use Otsu's adaptive thresholding; otherwise fixed threshold
    """
    if use_otsu:
        _, thresh = cv.threshold(edges, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    else:
        _, thresh = cv.threshold(edges, 100, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2:]
    return list(contours)


def _compute_axis_alignment(dx: float, dy: float) -> float:
    """
    Compute how axis-aligned a segment is (0-1 score).
    1.0 = perfectly horizontal or vertical
    0.0 = 45 degrees (worst case for gutter detection)
    """
    if dx == 0 and dy == 0:
        return 0.0
    # Angle from horizontal (0 to 90 degrees)
    angle_rad = math.atan2(abs(dy), abs(dx))
    angle_deg = math.degrees(angle_rad)
    # Score: 0 deg or 90 deg = 1.0, 45 deg = 0.0
    # Map: 0->1, 45->0, 90->1
    if angle_deg <= 45:
        return 1.0 - (angle_deg / 45.0)
    else:
        return (angle_deg - 45) / 45.0


def detect_segments(
    gray: np.ndarray,
    img_size: Tuple[int, int],
    min_segment_ratio: float,
    *,
    max_segments: int = 500,
    prefer_axis_aligned: bool = True,
    use_lsd_nfa: bool = False,
) -> List[Segment]:
    """
    Detect line segments using LSD (Line Segment Detector).
    Returns segments longer than minimum threshold, capped at max_segments.

    Args:
        gray: Grayscale image
        img_size: (width, height) tuple
        min_segment_ratio: Minimum segment length as fraction of smaller image dimension
        max_segments: Maximum segments to return
        prefer_axis_aligned: Prefer horizontal/vertical segments (typical comic gutters)
        use_lsd_nfa: Use LSD_REFINE_ADV mode for NFA quality scores (slower)
    """
    # Choose LSD mode: ADV gives NFA quality scores but is slower
    lsd_mode = cv.LSD_REFINE_ADV if use_lsd_nfa else cv.LSD_REFINE_NONE
    lsd = cv.createLineSegmentDetector(lsd_mode)
    result = lsd.detect(gray)

    if result is None or result[0] is None:
        return []

    lines = result[0]
    # NFA scores available only in ADV mode: result = (lines, widths, precs, nfa)
    nfa_scores = result[3] if use_lsd_nfa and len(result) > 3 and result[3] is not None else None

    min_dist = min(img_size) * min_segment_ratio

    # Collect segments with scoring
    scored_segments: List[Tuple[float, Segment]] = []
    for i, dline in enumerate(lines):
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))

        dx = x1 - x0
        dy = y1 - y0
        dist = math.sqrt(dx**2 + dy**2)

        if dist < min_dist:
            continue

        # Base score is length (normalized by image size for consistency)
        score = dist / min(img_size)

        # Axis alignment bonus: multiply by 1.0-2.0 based on alignment
        if prefer_axis_aligned:
            alignment = _compute_axis_alignment(dx, dy)
            score *= 1.0 + alignment  # Range: 1.0 (diagonal) to 2.0 (axis-aligned)

        # NFA quality bonus: higher NFA = better detection confidence
        # NFA is log-scale, typically -1 (weak) to 10+ (strong)
        if nfa_scores is not None:
            nfa = float(nfa_scores[i][0])
            # Map NFA to multiplier: -1->0.5, 0->1.0, 5->1.5, 10->2.0
            nfa_mult = max(0.5, min(2.0, 1.0 + nfa / 10.0))
            score *= nfa_mult

        scored_segments.append((score, Segment((x0, y0), (x1, y1))))

    # Keep highest-scoring segments
    if len(scored_segments) > max_segments:
        scored_segments.sort(key=lambda x: x[0], reverse=True)
        scored_segments = scored_segments[:max_segments]

    return [seg for _, seg in scored_segments]


def initial_panels(
    contours: List[np.ndarray],
    img_size: Tuple[int, int],
    min_panel_ratio: float,
    *,
    use_polygon: bool = False,
) -> List[InternalPanel]:
    """
    Convert contours to initial panel bounding boxes.
    Filters out very small contours.

    Args:
        contours: List of contours from findContours
        img_size: Image (width, height)
        min_panel_ratio: Minimum panel size ratio
        use_polygon: If True, use approxPolyDP for polygon data (needed for splitting).
                     If False, use boundingRect only (faster, simpler).
    """
    panels = []
    for contour in contours:
        if use_polygon:
            arclength = cv.arcLength(contour, True)
            epsilon = 0.001 * arclength
            approx = cv.approxPolyDP(contour, epsilon, True)
            panel = InternalPanel(img_size, min_panel_ratio, polygon=approx)
        else:
            x, y, w, h = cv.boundingRect(contour)
            panel = InternalPanel(img_size, min_panel_ratio, xywh=(x, y, w, h))

        if panel.is_very_small():
            continue

        panels.append(panel)

    return panels


def _is_gutter_line(gray: np.ndarray, segment: Segment, max_variance: float = 400.0) -> bool:
    """
    Check if a segment line looks like a real gutter (low pixel variance)
    vs artwork being cut (high pixel variance).
    """
    x0, y0 = segment.a
    x1, y1 = segment.b
    variance = compute_line_variance(gray, x0, y0, x1, y1)
    return variance <= max_variance


def split_panels(
    panels: List[InternalPanel], segments: List[Segment], gray: Optional[np.ndarray] = None
) -> Tuple[List[InternalPanel], List[float]]:
    """
    Iteratively split panels using detected segments.
    Continues until no more panels can be split.

    Args:
        panels: List of panels to potentially split
        segments: Detected line segments
        gray: Grayscale image for gutter color consistency check (optional)

    Sets split_coverage on each subpanel created via splitting.
    """
    split_coverages: List[float] = []
    did_split = True
    while did_split:
        did_split = False
        # Sort by area descending to split largest panels first
        for p in sorted(panels, key=lambda p: p.area(), reverse=True):
            split = p.split(segments)
            if split is not None:
                # Validate split using gutter color consistency
                if gray is not None and not _is_gutter_line(gray, split.segment):
                    # High variance along split = cutting through artwork, reject
                    continue

                did_split = True
                panels.remove(p)
                coverage = split.segments_coverage()
                # Set split_coverage on each subpanel
                for subpanel in split.subpanels:
                    subpanel.split_coverage = coverage
                panels += split.subpanels
                split_coverages.append(coverage)
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


def collect_all_gutters(panels: List[InternalPanel]) -> Tuple[List[int], List[int]]:
    """
    Collect all individual gutter widths between panels.

    Returns (gutters_x, gutters_y) lists for variance analysis.
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

    return gutters_x, gutters_y


def actual_gutters(panels: List[InternalPanel], func: Callable[[List[int]], int] = min) -> Dict[str, int]:
    """
    Estimate gutters between panels (Kumiko).

    Returns {"x","y","r","b"} where "r"/"b" are the negative gutter values used for expansion.
    """
    gutters_x, gutters_y = collect_all_gutters(panels)

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


def remove_contained_panels(
    panels: List[InternalPanel], threshold: float = 0.9, *, prefer_smaller: bool = True
) -> List[InternalPanel]:
    """
    Remove panels that are almost entirely contained within another panel.

    Args:
        panels: List of panels to filter
        threshold: Remove panel if this fraction is inside another (default 0.9 = 90%)
        prefer_smaller: If True, keep smaller panels (better segmentation).
                       If False, keep larger panels (original behavior for speech bubbles).
    """
    to_remove: List[InternalPanel] = []

    for p1 in panels:
        for p2 in panels:
            if p1 is p2:
                continue

            overlap = p1.overlap_panel(p2)
            if not overlap:
                continue

            # Check if p1 is mostly inside p2
            p1_area = p1.area()
            if p1_area > 0 and overlap.area() / p1_area >= threshold:
                # p1 is contained in p2
                if prefer_smaller:
                    # Keep smaller panels (better segmentation) - remove the container
                    to_remove.append(p2)
                else:
                    # Keep larger panel (original behavior) - remove the contained
                    to_remove.append(p1)
                break

    return [p for p in panels if p not in to_remove]


def _is_axis_aligned(segment: Segment, tolerance_deg: float = 15.0) -> bool:
    """Check if a segment is approximately horizontal or vertical."""
    angle_rad = segment.angle()
    angle_deg = abs(math.degrees(angle_rad))
    # Normalize to 0-90 range
    if angle_deg > 90:
        angle_deg = 180 - angle_deg
    # Check if near horizontal (0°) or vertical (90°)
    return angle_deg <= tolerance_deg or angle_deg >= (90 - tolerance_deg)


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

                # Only axis-aligned segments can block grouping (not diagonal motion lines)
                big_segments: List[Segment] = []
                for s in segments:
                    if not _is_axis_aligned(s):
                        continue
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
    min_segment_ratio: Optional[float] = None,
    panel_expansion: bool = False,
    small_panel_grouping: bool = False,
    big_panel_grouping: bool = False,
    panel_splitting: bool = False,
    use_denoising: bool = True,
    use_canny: bool = False,
    use_morphological_close: bool = False,
    max_segments: int = 500,
    prefer_axis_aligned: bool = True,
    use_lsd_nfa: bool = False,
    debug: Optional["DebugContext"] = None,
) -> PipelineResult:
    """
    Main detection pipeline.
    Returns PipelineResult with panels, split coverages, and cached intermediate data.

    Args:
        img: BGR image (OpenCV format)
        min_panel_ratio: Minimum panel size as fraction of image dimensions
        min_segment_ratio: Minimum segment length as fraction of image (default: min_panel_ratio / 2)
        panel_expansion: Whether to expand panels to fill gutters
        small_panel_grouping: Whether to group small nearby panels
        big_panel_grouping: Whether to group large adjacent panels
        panel_splitting: Whether to split panels using detected segments (expensive)
        use_denoising: Whether to apply Gaussian blur before edge detection
        use_canny: Whether to use Canny edges (True) or Sobel edges (False)
        use_morphological_close: Whether to apply morphological closing to bridge edge gaps
        max_segments: Maximum segments to keep from LSD
        prefer_axis_aligned: Prefer horizontal/vertical segments for gutters
        use_lsd_nfa: Use LSD NFA quality scores for filtering
        debug: Optional debug context for step-by-step visualization
    """
    # Default segment ratio is half of panel ratio (keep shorter lines)
    if min_segment_ratio is None:
        min_segment_ratio = min_panel_ratio / 2
    img_size = (img.shape[1], img.shape[0])  # (width, height)

    # Debug: set base image
    if debug and debug.enabled:
        debug.set_base_image(img)
        debug.add_step("Input image", [])
        debug.add_image("Input image")

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if debug and debug.enabled:
        debug.add_image("Grayscale", gray)

    # 1. Preprocessing: optional denoising
    if use_denoising:
        gray_processed = denoise(gray)
        if debug and debug.enabled:
            debug.add_image("Denoised", gray_processed)
    else:
        gray_processed = gray

    # 2. Edge detection: Canny or Sobel
    if use_canny:
        edges = canny_edges(gray_processed)
        if debug and debug.enabled:
            debug.add_image("Canny edges", edges)
    else:
        edges = sobel_edges(gray_processed)
        if debug and debug.enabled:
            debug.add_image("Sobel edges", edges)

    # 3. Optional morphological closing to bridge gaps
    if use_morphological_close:
        edges = morphological_close(edges)
        if debug and debug.enabled:
            debug.add_image("Morphological close", edges)

    # 4. Threshold (Otsu) + contours
    contours = get_contours(edges)
    if debug and debug.enabled:
        debug.draw_contours(contours)
        debug.add_step("Contours detected", [])
        debug.add_image("Contours")

    # 5. LSD gutter detection (only needed for splitting or big panel grouping)
    segments: List[Segment] = []
    if panel_splitting or big_panel_grouping:
        segments = detect_segments(
            gray_processed,
            img_size,
            min_segment_ratio,
            max_segments=max_segments,
            prefer_axis_aligned=prefer_axis_aligned,
            use_lsd_nfa=use_lsd_nfa,
        )
        if debug and debug.enabled:
            debug.draw_segments(segments)
            debug.add_step("Segments detected", [])
            debug.add_image("Segments")

    # 6. Initial panels from contours
    # Use polygon approximation when splitting or grouping needs polygon data
    use_polygon = panel_splitting or big_panel_grouping
    panels = initial_panels(contours, img_size, min_panel_ratio, use_polygon=use_polygon)
    if debug and debug.enabled:
        debug.draw_panels(panels)
        debug.add_step("Initial panels", panels)
        debug.add_image("Initial panels")

    if small_panel_grouping:
        panels = group_small_panels(panels)
        if debug and debug.enabled:
            debug.draw_panels(panels)
            debug.add_step("Small panels grouped", panels)
            debug.add_image("Small panels grouped")

    # 7. Panel splitting (expensive O(n²) polygon ops, often no-op)
    split_coverages: List[float] = []
    if panel_splitting:
        panels, split_coverages = split_panels(panels, segments, gray=gray)
        if debug and debug.enabled:
            debug.draw_panels(panels)
            debug.add_step("Panels split", panels)
            debug.add_image("Panels split")

    # 8. Refinement: filter small, merge overlaps, deoverlap
    panels = exclude_small(panels, min_panel_ratio)
    # merge_panels and deoverlap_panels only needed when splitting creates overlaps
    if panel_splitting:
        panels = merge_panels(panels)
        panels = deoverlap_panels(panels)
        panels = exclude_small(panels, min_panel_ratio)
    if debug and debug.enabled:
        debug.draw_panels(panels)
        debug.add_step("Panels refined", panels)
        debug.add_image("Panels refined")

    if panel_expansion:
        panels = expand_panels(panels)
        if debug and debug.enabled:
            debug.draw_panels(panels)
            debug.add_step("Panels expanded", panels)
            debug.add_image("Panels expanded")

    # Fallback: if no panels detected, return full page as single panel
    if not panels:
        panels.append(InternalPanel(img_size, min_panel_ratio, xywh=(0, 0, img_size[0], img_size[1])))

    if big_panel_grouping:
        panels = group_big_panels(panels, segments)
        if debug and debug.enabled:
            debug.draw_panels(panels)
            debug.add_step("Big panels grouped", panels)
            debug.add_image("Big panels grouped")

    # Final cleanup: remove panels contained within others (false positives)
    panels = remove_contained_panels(panels)
    if debug and debug.enabled:
        debug.draw_panels(panels)
        debug.add_step("Contained panels removed", panels)
        debug.add_image("Contained panels removed")

    # Final result
    if debug and debug.enabled:
        debug.draw_panels(panels, color="green")
        debug.add_step("Final result", panels)
        debug.add_image("Final result")

    return PipelineResult(
        panels=panels,
        split_coverages=split_coverages,
        gray=gray,
    )
