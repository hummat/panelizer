from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from ..schema import Panel
from .pipeline import detect_panels


class CVDetector:
    def __init__(
        self,
        min_panel_ratio: float = 0.1,
        *,
        panel_expansion: bool = True,
        small_panel_grouping: bool = True,
        big_panel_grouping: bool = True,
    ) -> None:
        """
        Initialize CV detector.
        min_panel_ratio: minimum panel size as fraction of page dimensions (default 0.1 = 10%)
        """
        self.min_panel_ratio = min_panel_ratio
        self.panel_expansion = panel_expansion
        self.small_panel_grouping = small_panel_grouping
        self.big_panel_grouping = big_panel_grouping

    def detect(self, image: Image.Image) -> Tuple[List[Panel], float]:
        """
        Detects panels in a PIL image using Kumiko-based CV pipeline.
        Returns a list of Panels and a confidence score.
        """
        # Convert PIL to OpenCV (BGR)
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Run detection pipeline
        internal_panels, split_coverages = detect_panels(
            img,
            self.min_panel_ratio,
            panel_expansion=self.panel_expansion,
            small_panel_grouping=self.small_panel_grouping,
            big_panel_grouping=self.big_panel_grouping,
        )

        # Convert to schema Panel objects
        panels = []
        for i, p in enumerate(internal_panels):
            x, y, w, h = p.to_xywh()
            panels.append(
                Panel(
                    id=f"p-{i}",
                    bbox=(x, y, w, h),
                    confidence=0.9,  # Individual panel confidence
                )
            )

        # Calculate overall confidence
        confidence = self._compute_confidence(panels, split_coverages, img.shape[1] * img.shape[0])

        return panels, float(confidence)

    def _compute_confidence(self, panels: List[Panel], split_coverages: List[float], page_area: int) -> float:
        """
        Compute overall detection confidence based on multiple signals.
        """
        if not panels:
            return 0.1

        # 1. Panel count factor (2-12 is healthy)
        n = len(panels)
        if 2 <= n <= 12:
            count_factor = 1.0
        elif n == 1:
            count_factor = 0.7  # Could be splash page
        else:
            count_factor = 0.5  # Too many panels, likely over-split

        # 2. Coverage factor (70-95% is healthy)
        panel_area = sum(p.bbox[2] * p.bbox[3] for p in panels)
        coverage = panel_area / page_area
        if 0.7 <= coverage <= 0.95:
            coverage_factor = 1.0
        else:
            coverage_factor = 0.8

        # 3. Split quality factor
        if split_coverages:
            split_factor = sum(split_coverages) / len(split_coverages)
        else:
            split_factor = 0.8  # No splits needed, assume okay

        # Combine factors (geometric mean to penalize weak signals)
        confidence = (count_factor * coverage_factor * split_factor) ** (1 / 3)
        return min(1.0, max(0.0, confidence))
