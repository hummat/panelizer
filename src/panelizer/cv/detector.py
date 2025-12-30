import time
from typing import TYPE_CHECKING, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from ..schema import Panel
from .confidence import compute_edge_strength, compute_page_confidence, compute_panel_confidence
from .pipeline import actual_gutters, collect_all_gutters, detect_panels

if TYPE_CHECKING:
    from .debug import DebugContext


class DetectionResult:
    """Result of panel detection including metadata."""

    def __init__(
        self,
        panels: List[Panel],
        confidence: float,
        gutters: Optional[Tuple[int, int]] = None,
        processing_time: Optional[float] = None,
    ):
        self.panels = panels
        self.confidence = confidence
        self.gutters = gutters
        self.processing_time = processing_time


def _clamp_bbox_xywh(bbox: Tuple[int, int, int, int], *, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox

    if img_w <= 0 or img_h <= 0:
        return (0, 0, 0, 0)

    x = max(0, min(int(x), img_w - 1))
    y = max(0, min(int(y), img_h - 1))

    w = int(w)
    h = int(h)
    if w <= 0:
        w = 1
    if h <= 0:
        h = 1

    w = min(w, img_w - x)
    h = min(h, img_h - y)
    return (x, y, w, h)


class CVDetector:
    def __init__(
        self,
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
        skip_scoring: bool = False,
        max_dimension: int = 2000,
    ) -> None:
        """
        Initialize CV detector.

        Args:
            min_panel_ratio: Minimum panel size as fraction of page dimensions (default 0.1 = 10%)
            min_segment_ratio: Minimum segment length as fraction (default: min_panel_ratio / 2)
            panel_expansion: Whether to expand panels to fill gutters (off = raw detection)
            small_panel_grouping: Whether to group small nearby panels (off = raw detection)
            big_panel_grouping: Whether to group large adjacent panels (off = raw detection)
            panel_splitting: Whether to split panels using detected segments (expensive, often no-op)
            use_denoising: Whether to apply Gaussian blur before edge detection
            use_canny: Whether to use Canny edges (True) or Sobel edges (False)
            use_morphological_close: Whether to apply morphological closing (off = raw edges)
            max_segments: Maximum segments to keep from LSD (longest first)
            prefer_axis_aligned: Prefer horizontal/vertical segments (typical comic gutters)
            use_lsd_nfa: Use LSD NFA quality scores (slower but more accurate filtering)
            skip_scoring: Skip confidence scoring (faster, use when ML fallback not needed)
            max_dimension: Downscale images larger than this (0 = no limit, default 2000)
        """
        self.min_panel_ratio = min_panel_ratio
        self.min_segment_ratio = min_segment_ratio
        self.panel_expansion = panel_expansion
        self.small_panel_grouping = small_panel_grouping
        self.big_panel_grouping = big_panel_grouping
        self.panel_splitting = panel_splitting
        self.use_denoising = use_denoising
        self.use_canny = use_canny
        self.use_morphological_close = use_morphological_close
        self.max_segments = max_segments
        self.prefer_axis_aligned = prefer_axis_aligned
        self.use_lsd_nfa = use_lsd_nfa
        self.skip_scoring = skip_scoring
        self.max_dimension = max_dimension

    def detect(self, image: Image.Image, *, debug: Optional["DebugContext"] = None) -> DetectionResult:
        """
        Detects panels in a PIL image using Kumiko-based CV pipeline.
        Returns a DetectionResult with panels, confidence, gutters, and timing.

        Args:
            image: PIL Image to process
            debug: Optional debug context for step-by-step visualization
        """
        start_time = time.perf_counter()

        # Downscale if image exceeds max_dimension (reduces noise and speeds up processing)
        orig_w, orig_h = image.size
        scale = 1.0
        if self.max_dimension > 0:
            max_dim = max(orig_w, orig_h)
            if max_dim > self.max_dimension:
                scale = self.max_dimension / max_dim
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Convert PIL to OpenCV (BGR)
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img_h, img_w = img.shape[:2]
        page_area = img_w * img_h

        # Run detection pipeline (returns cached gray and magnitude)
        result = detect_panels(
            img,
            self.min_panel_ratio,
            min_segment_ratio=self.min_segment_ratio,
            panel_expansion=self.panel_expansion,
            small_panel_grouping=self.small_panel_grouping,
            big_panel_grouping=self.big_panel_grouping,
            panel_splitting=self.panel_splitting,
            use_denoising=self.use_denoising,
            use_canny=self.use_canny,
            use_morphological_close=self.use_morphological_close,
            max_segments=self.max_segments,
            prefer_axis_aligned=self.prefer_axis_aligned,
            use_lsd_nfa=self.use_lsd_nfa,
            debug=debug,
        )
        internal_panels = result.panels

        # Helper to scale bbox back to original image size
        def scale_bbox(x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
            if scale == 1.0:
                return (x, y, w, h)
            return (
                int(x / scale),
                int(y / scale),
                int(w / scale),
                int(h / scale),
            )

        # Skip scoring if requested (faster when ML fallback not needed)
        if self.skip_scoring:
            panels = []
            for i, p in enumerate(internal_panels):
                x, y, w, h = _clamp_bbox_xywh(p.to_xywh(), img_w=img_w, img_h=img_h)
                x, y, w, h = scale_bbox(x, y, w, h)
                panels.append(Panel(id=f"p-{i}", bbox=(x, y, w, h), confidence=1.0))

            gutters_dict = actual_gutters(internal_panels)
            gx, gy = gutters_dict["x"], gutters_dict["y"]
            if scale != 1.0:
                gx, gy = int(gx / scale), int(gy / scale)
            processing_time = time.perf_counter() - start_time

            return DetectionResult(
                panels=panels,
                confidence=1.0,
                gutters=(gx, gy),
                processing_time=round(processing_time, 3),
            )

        # Use cached gray from pipeline, compute magnitude for scoring
        gray = result.gray
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Compute per-panel confidence scores (including edge strength and gutter color)
        panel_confidences = []
        for p in internal_panels:
            edge_strength = compute_edge_strength(p, magnitude)
            conf = compute_panel_confidence(
                p,
                internal_panels,
                page_area,
                gray=gray,
                split_coverage=p.split_coverage,
                edge_strength=edge_strength,
            )
            panel_confidences.append(conf)

        # Convert to schema Panel objects (scale back to original size)
        panels = []
        for i, (p, conf) in enumerate(zip(internal_panels, panel_confidences, strict=True)):
            x, y, w, h = _clamp_bbox_xywh(p.to_xywh(), img_w=img_w, img_h=img_h)
            x, y, w, h = scale_bbox(x, y, w, h)
            panels.append(
                Panel(
                    id=f"p-{i}",
                    bbox=(x, y, w, h),
                    confidence=conf,
                )
            )

        # Collect all gutters for variance analysis
        gutters_x, gutters_y = collect_all_gutters(internal_panels)

        # Calculate overall page confidence (with gutter variance)
        panel_areas = [p.bbox[2] * p.bbox[3] for p in panels]
        confidence = compute_page_confidence(
            panel_confidences,
            panel_areas,
            page_area,
            gutters_x=gutters_x,
            gutters_y=gutters_y,
        )

        # Compute gutters from internal panels (scale back)
        gutters_dict = actual_gutters(internal_panels)
        gx, gy = gutters_dict["x"], gutters_dict["y"]
        if scale != 1.0:
            gx, gy = int(gx / scale), int(gy / scale)

        processing_time = time.perf_counter() - start_time

        return DetectionResult(
            panels=panels,
            confidence=float(confidence),
            gutters=(gx, gy),
            processing_time=round(processing_time, 3),
        )
