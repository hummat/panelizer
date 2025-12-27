"""Per-panel and page-level confidence scoring heuristics."""

from typing import List, Optional, Tuple

from .panel_internal import InternalPanel


def compute_panel_confidence(
    panel: InternalPanel,
    all_panels: List[InternalPanel],
    page_area: int,
    *,
    split_coverage: Optional[float] = None,
) -> float:
    """
    Compute confidence score for a single panel based on multiple heuristics.

    Heuristics used:
    1. Aspect ratio - panels with extreme ratios (< 0.2 or > 5) are penalized
    2. Size factor - very small (< 3%) or very large (> 70%) panels are penalized
    3. Rectangularity - how well the polygon fits its bounding box
    4. Gutter quality - clear, consistent gaps to neighbors
    5. Split quality - if panel was created via split, uses segment coverage

    Returns a score between 0.0 and 1.0.
    """
    scores: List[Tuple[float, float]] = []  # (score, weight)

    # 1. Aspect ratio factor (weight: 1.0)
    # Most comic panels have aspect ratios between 0.3 and 3.0
    aspect = _aspect_ratio_score(panel)
    scores.append((aspect, 1.0))

    # 2. Size factor (weight: 1.0)
    # Very small or very large panels are less confident
    size = _size_score(panel, page_area)
    scores.append((size, 1.0))

    # 3. Rectangularity factor (weight: 0.8)
    # Rectangular panels are more common and more confident
    rect = _rectangularity_score(panel)
    scores.append((rect, 0.8))

    # 4. Gutter quality factor (weight: 1.2)
    # Panels with clear, consistent gutters to neighbors are more confident
    gutter = _gutter_quality_score(panel, all_panels)
    scores.append((gutter, 1.2))

    # 5. Split quality factor (weight: 0.5)
    # If panel was created via split, use the segment coverage
    if split_coverage is not None:
        scores.append((split_coverage, 0.5))

    # Weighted average
    total_weight = sum(w for _, w in scores)
    weighted_sum = sum(s * w for s, w in scores)
    confidence = weighted_sum / total_weight if total_weight > 0 else 0.5

    return max(0.0, min(1.0, confidence))


def _aspect_ratio_score(panel: InternalPanel) -> float:
    """
    Score based on aspect ratio.

    Ideal: 0.4 to 2.5 (common comic panel ratios)
    Acceptable: 0.2 to 5.0
    Poor: outside this range (very thin strips or extreme squares)
    """
    w, h = panel.w(), panel.h()
    if h == 0 or w == 0:
        return 0.2

    aspect = w / h

    # Ideal range: full score
    if 0.4 <= aspect <= 2.5:
        return 1.0

    # Acceptable range: linear falloff
    if 0.2 <= aspect < 0.4:
        return 0.6 + 0.4 * (aspect - 0.2) / 0.2
    if 2.5 < aspect <= 5.0:
        return 0.6 + 0.4 * (5.0 - aspect) / 2.5

    # Outside acceptable range: low score
    if aspect < 0.2:
        return max(0.2, 0.6 * aspect / 0.2)
    # aspect > 5.0
    return max(0.2, 0.6 * 5.0 / aspect)


def _size_score(panel: InternalPanel, page_area: int) -> float:
    """
    Score based on panel size relative to page.

    Ideal: 5% to 50% of page area
    Acceptable: 3% to 70%
    Poor: outside this range
    """
    if page_area == 0:
        return 0.5

    area_ratio = panel.area() / page_area

    # Ideal range: full score
    if 0.05 <= area_ratio <= 0.50:
        return 1.0

    # Small panels
    if area_ratio < 0.05:
        if area_ratio >= 0.03:
            return 0.7 + 0.3 * (area_ratio - 0.03) / 0.02
        if area_ratio >= 0.01:
            return 0.4 + 0.3 * (area_ratio - 0.01) / 0.02
        return max(0.2, 0.4 * area_ratio / 0.01)

    # Large panels
    if area_ratio <= 0.70:
        return 0.7 + 0.3 * (0.70 - area_ratio) / 0.20
    if area_ratio <= 0.90:
        return 0.4 + 0.3 * (0.90 - area_ratio) / 0.20
    # Very large (splash page)
    return 0.4


def _rectangularity_score(panel: InternalPanel) -> float:
    """
    Score based on how rectangular the panel is.

    Uses polygon area vs bounding box area ratio.
    Pure rectangles score 1.0, irregular shapes score lower.
    """
    if panel.polygon is None:
        # No polygon data - assume rectangular (from xywh)
        return 0.9

    import cv2 as cv

    # Polygon area
    poly_area = cv.contourArea(panel.polygon)
    if poly_area <= 0:
        return 0.5

    # Bounding box area
    bbox_area = panel.area()
    if bbox_area <= 0:
        return 0.5

    # Rectangularity = polygon_area / bbox_area
    # Perfect rectangle = 1.0
    rect_ratio = poly_area / bbox_area

    # Most panels are nearly rectangular (0.85+)
    if rect_ratio >= 0.90:
        return 1.0
    if rect_ratio >= 0.75:
        return 0.7 + 0.3 * (rect_ratio - 0.75) / 0.15
    if rect_ratio >= 0.50:
        return 0.4 + 0.3 * (rect_ratio - 0.50) / 0.25
    return max(0.2, 0.4 * rect_ratio / 0.50)


def _gutter_quality_score(panel: InternalPanel, all_panels: List[InternalPanel]) -> float:
    """
    Score based on gutter consistency with neighbors.

    Clear, regular gutters to adjacent panels indicate confident detection.
    """
    if len(all_panels) <= 1:
        # Single panel - can't assess gutter quality
        return 0.7

    neighbor_scores: List[float] = []

    # Check each direction
    left = panel.find_left_panel(all_panels)
    if left:
        gap = panel.x - left.r
        neighbor_scores.append(_gap_score(gap, panel.img_size[0]))

    right = panel.find_right_panel(all_panels)
    if right:
        gap = right.x - panel.r
        neighbor_scores.append(_gap_score(gap, panel.img_size[0]))

    top = panel.find_top_panel(all_panels)
    if top:
        gap = panel.y - top.b
        neighbor_scores.append(_gap_score(gap, panel.img_size[1]))

    bottom = panel.find_bottom_panel(all_panels)
    if bottom:
        gap = bottom.y - panel.b
        neighbor_scores.append(_gap_score(gap, panel.img_size[1]))

    if not neighbor_scores:
        # No neighbors found - edge panel, slightly lower confidence
        return 0.75

    return sum(neighbor_scores) / len(neighbor_scores)


def _gap_score(gap: int, dimension: int) -> float:
    """
    Score a gap (gutter) between panels.

    Ideal gutter: 0.5% to 5% of page dimension
    Acceptable: 0% to 10%
    """
    if dimension == 0:
        return 0.5

    gap_ratio = gap / dimension

    # Negative gap = overlap (bad)
    if gap_ratio < 0:
        return max(0.1, 0.5 + gap_ratio * 5)  # Penalize overlaps heavily

    # No gap (touching) - okay but not ideal
    if gap_ratio < 0.005:
        return 0.7 + 0.3 * gap_ratio / 0.005

    # Ideal gutter range
    if 0.005 <= gap_ratio <= 0.05:
        return 1.0

    # Slightly large gutter
    if gap_ratio <= 0.10:
        return 0.7 + 0.3 * (0.10 - gap_ratio) / 0.05

    # Very large gap - probably missing a panel
    return max(0.3, 0.7 * 0.10 / gap_ratio)


def compute_page_confidence(
    panel_confidences: List[float],
    panel_areas: List[int],
    page_area: int,
    *,
    panel_count_factor: Optional[float] = None,
    coverage_factor: Optional[float] = None,
) -> float:
    """
    Compute overall page detection confidence.

    Combines:
    1. Area-weighted mean of panel confidences
    2. Panel count reasonableness (2-12 is healthy)
    3. Page coverage (70-95% is healthy)
    """
    if not panel_confidences:
        return 0.1

    # 1. Area-weighted mean of panel confidences
    total_area = sum(panel_areas)
    if total_area > 0:
        weighted_conf = sum(c * a for c, a in zip(panel_confidences, panel_areas, strict=True)) / total_area
    else:
        weighted_conf = sum(panel_confidences) / len(panel_confidences)

    # 2. Panel count factor
    if panel_count_factor is None:
        n = len(panel_confidences)
        if 2 <= n <= 12:
            panel_count_factor = 1.0
        elif n == 1:
            panel_count_factor = 0.7  # Could be splash page
        elif n == 0:
            panel_count_factor = 0.1
        else:
            panel_count_factor = 0.5  # Too many panels

    # 3. Coverage factor
    if coverage_factor is None:
        if page_area > 0:
            coverage = total_area / page_area
            if 0.70 <= coverage <= 0.95:
                coverage_factor = 1.0
            elif 0.50 <= coverage < 0.70:
                coverage_factor = 0.7 + 0.3 * (coverage - 0.50) / 0.20
            elif coverage > 0.95:
                coverage_factor = 0.8  # Slight overlap or bleed
            else:
                coverage_factor = max(0.4, coverage / 0.50 * 0.7)
        else:
            coverage_factor = 0.8

    # Combine using geometric mean (penalizes weak signals)
    combined = (weighted_conf * panel_count_factor * coverage_factor) ** (1 / 3)
    return max(0.0, min(1.0, combined))
