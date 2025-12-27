"""Per-panel and page-level confidence scoring heuristics."""

import statistics
from typing import List, Optional, Tuple

import numpy as np

from .panel_internal import InternalPanel


def sample_line_pixels(gray: np.ndarray, x0: int, y0: int, x1: int, y1: int, num_samples: int = 20) -> np.ndarray:
    """Sample pixel values along a line segment."""
    h, w = gray.shape[:2]

    # Generate sample points along the segment
    t_values = np.linspace(0, 1, num_samples)
    xs = (x0 + t_values * (x1 - x0)).astype(int)
    ys = (y0 + t_values * (y1 - y0)).astype(int)

    # Clamp to image bounds
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)

    return gray[ys, xs]


def compute_line_variance(gray: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    """Compute variance of pixel values along a line segment."""
    pixels = sample_line_pixels(gray, x0, y0, x1, y1)
    if len(pixels) == 0:
        return 0.0
    return float(np.var(pixels))


def compute_edge_strength(
    panel: InternalPanel,
    magnitude: np.ndarray,
    border_width: int = 3,
) -> float:
    """
    Compute average gradient magnitude along a panel's border.

    Bleed Aware: If a panel edge coincides with the image boundary, it is
    ignored for scoring purposes (as it likely lacks a drawn border).

    Args:
        panel: The panel to analyze
        magnitude: Precomputed gradient magnitude image (same size as original)
        border_width: Width of border region to sample (pixels)

    Returns:
        Normalized edge strength score between 0.0 and 1.0
    """
    img_h, img_w = magnitude.shape[:2]
    x, y, w, h = panel.to_xywh()

    # Sample border regions, but only if they aren't on the image boundary
    border_samples: List[np.ndarray] = []

    # Top border (skip if at top of image)
    if y > 2:
        top_y1 = max(0, y - border_width)
        top_y2 = min(img_h, y + border_width)
        if top_y2 > top_y1:
            border_samples.append(magnitude[top_y1:top_y2, x : x + w].flatten())

    # Bottom border (skip if at bottom of image)
    if y + h < img_h - 2:
        bot_y1 = max(0, y + h - border_width)
        bot_y2 = min(img_h, y + h + border_width)
        if bot_y2 > bot_y1:
            border_samples.append(magnitude[bot_y1:bot_y2, x : x + w].flatten())

    # Left border (skip if at left of image)
    if x > 2:
        left_x1 = max(0, x - border_width)
        left_x2 = min(img_w, x + border_width)
        if left_x2 > left_x1:
            border_samples.append(magnitude[y : y + h, left_x1:left_x2].flatten())

    # Right border (skip if at right of image)
    if x + w < img_w - 2:
        right_x1 = max(0, x + w - border_width)
        right_x2 = min(img_w, x + w + border_width)
        if right_x2 > right_x1:
            border_samples.append(magnitude[y : y + h, right_x1:right_x2].flatten())

    if not border_samples:
        # Full page panel or all edges bleed - assume reasonable confidence
        # if the rest of the metrics pass.
        return 0.8

    # Combine all border samples
    all_samples = np.concatenate(border_samples)
    if len(all_samples) == 0:
        return 0.8

    # Average gradient magnitude along borders
    avg_magnitude = float(np.mean(all_samples))

    # Normalize to 0-1 score
    # Typical strong edges have magnitudes around 50-200
    # Weak edges are below 30
    if avg_magnitude >= 100:
        return 1.0
    elif avg_magnitude >= 50:
        return 0.7 + 0.3 * (avg_magnitude - 50) / 50
    elif avg_magnitude >= 20:
        return 0.4 + 0.3 * (avg_magnitude - 20) / 30
    else:
        return max(0.2, 0.4 * avg_magnitude / 20)


def compute_panel_confidence(
    panel: InternalPanel,
    all_panels: List[InternalPanel],
    page_area: int,
    *,
    gray: Optional[np.ndarray] = None,
    split_coverage: Optional[float] = None,
    edge_strength: Optional[float] = None,
) -> float:
    """
    Compute confidence score for a single panel based on multiple heuristics.

    Heuristics used:
    1. Aspect ratio - panels with extreme ratios (< 0.2 or > 5) are penalized
    2. Size factor - very small (< 3%) or very large (> 70%) panels are penalized
    3. Rectangularity - how well the polygon fits its bounding box
    4. Gutter quality - clear, consistent gaps to neighbors
    5. Gutter color - low variance along edges (if gray image provided)
    6. Split quality - if panel was created via split, uses segment coverage
    7. Edge strength - average gradient magnitude along panel border (if provided)

    Returns a score between 0.0 and 1.0.
    """
    scores: List[Tuple[float, float]] = []  # (score, weight)

    # 1. Aspect ratio factor (weight: 1.0)
    aspect = _aspect_ratio_score(panel)
    scores.append((aspect, 1.0))

    # 2. Size factor (weight: 1.0)
    size = _size_score(panel, page_area)
    scores.append((size, 1.0))

    # 3. Rectangularity factor (weight: 0.8)
    rect = _rectangularity_score(panel)
    scores.append((rect, 0.8))

    # 4. Gutter quality factor (weight: 1.2)
    gutter = _gutter_quality_score(panel, all_panels)
    scores.append((gutter, 1.2))

    # 5. Gutter color consistency (weight: 1.5)
    if gray is not None:
        color_conf = _gutter_color_score(panel, gray)
        scores.append((color_conf, 1.5))

    # 6. Split quality factor (weight: 0.5)
    if split_coverage is not None:
        scores.append((split_coverage, 0.5))

    # 7. Edge strength factor (weight: 1.0)
    if edge_strength is not None:
        scores.append((edge_strength, 1.0))

    # Weighted average
    total_weight = sum(w for _, w in scores)
    weighted_sum = sum(s * w for s, w in scores)
    confidence = weighted_sum / total_weight if total_weight > 0 else 0.5

    return max(0.0, min(1.0, confidence))


def _gutter_color_score(panel: InternalPanel, gray: np.ndarray) -> float:
    """
    Score based on pixel variance along the panel edges.
    Low variance (solid color) = good gutter.
    High variance (artwork) = bad split/detection.
    """
    x, y, w, h = panel.to_xywh()
    img_h, img_w = gray.shape[:2]

    # Sample all 4 edges, but slightly inset to avoid the border line itself
    # We want to sample the "gutter" side of the border.
    variances = []

    # Helper to score variance: 0-100 is excellent, 400+ is poor
    def variance_to_score(v: float) -> float:
        if v <= 100:
            return 1.0
        if v >= 600:
            return 0.2
        return 1.0 - 0.8 * (v - 100) / 500

    # Top (check if not bleed)
    if y > 2:
        variances.append(compute_line_variance(gray, x, y, x + w, y))
    # Bottom (check if not bleed)
    if y + h < img_h - 2:
        variances.append(compute_line_variance(gray, x, y + h, x + w, y + h))
    # Left (check if not bleed)
    if x > 2:
        variances.append(compute_line_variance(gray, x, y, x, y + h))
    # Right (check if not bleed)
    if x + w < img_w - 2:
        variances.append(compute_line_variance(gray, x + w, y, x + w, y + h))

    if not variances:
        return 0.8  # All edges bleed, can't judge

    edge_scores = [variance_to_score(v) for v in variances]
    return sum(edge_scores) / len(edge_scores)


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
    gutter_variance_factor: Optional[float] = None,
    gutters_x: Optional[List[int]] = None,
    gutters_y: Optional[List[int]] = None,
) -> float:
    """
    Compute overall page detection confidence.

    Combines:
    1. Area-weighted mean of panel confidences
    2. Panel count reasonableness (2-12 is healthy)
    3. Page coverage (70-95% is healthy)
    4. Gutter consistency (low variance in gutter widths is healthy)
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

    # 4. Gutter variance factor
    # Comic grids usually have consistent spacing; high variance signals detection failure
    if gutter_variance_factor is None:
        gutter_variance_factor = _compute_gutter_variance_score(gutters_x, gutters_y)

    # Combine using geometric mean (penalizes weak signals)
    combined = (weighted_conf * panel_count_factor * coverage_factor * gutter_variance_factor) ** (1 / 4)
    return max(0.0, min(1.0, combined))


def _compute_gutter_variance_score(
    gutters_x: Optional[List[int]] = None,
    gutters_y: Optional[List[int]] = None,
) -> float:
    """
    Score based on consistency of gutter widths.

    Low variance = consistent grid = high confidence
    High variance = irregular spacing = potential detection failure
    """
    all_gutters: List[int] = []
    if gutters_x:
        all_gutters.extend(gutters_x)
    if gutters_y:
        all_gutters.extend(gutters_y)

    if len(all_gutters) < 2:
        # Not enough data to compute variance
        return 0.85

    # Filter out negative gutters (overlaps) for variance calculation
    positive_gutters = [g for g in all_gutters if g > 0]
    if len(positive_gutters) < 2:
        return 0.7  # Mostly overlapping panels

    mean_gutter = statistics.mean(positive_gutters)
    if mean_gutter == 0:
        return 0.8

    # Coefficient of variation (CV) = stdev / mean
    # Low CV = consistent gutters
    stdev = statistics.stdev(positive_gutters)
    cv = stdev / mean_gutter

    # Ideal: CV < 0.3 (gutters within 30% of mean)
    # Acceptable: CV < 0.6
    # Poor: CV >= 0.6
    if cv < 0.3:
        return 1.0
    elif cv < 0.6:
        return 0.7 + 0.3 * (0.6 - cv) / 0.3
    else:
        return max(0.4, 0.7 * 0.6 / cv)
