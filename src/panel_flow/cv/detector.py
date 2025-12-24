from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from ..schema import Panel


class CVDetector:
    def detect(self, image: Image.Image) -> Tuple[List[Panel], float]:
        """
        Detects panels in a PIL image using classic CV.
        Returns a list of Panels and a confidence score.
        """
        # Convert PIL to OpenCV (BGR)
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Gaussian Blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. Adaptive Threshold
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # 3. Contour Detection
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        panels = []
        page_area = img.shape[0] * img.shape[1]

        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h

            # Filter out noise (tiny contours or extreme aspect ratios)
            if area < page_area * 0.01:  # Less than 1% of page
                continue
            if (
                w > img.shape[1] * 0.95 and h > img.shape[0] * 0.95
            ):  # Probably the whole page border
                continue

            panels.append(
                Panel(
                    id=f"p-{i}",
                    bbox=(x, y, w, h),
                    confidence=0.9,  # Base confidence for CV detection
                )
            )

        # 4. Confidence Estimation
        # Simple heuristic: if we found 2-10 panels and they cover a good chunk of the page
        detected_area = sum(p.bbox[2] * p.bbox[3] for p in panels)
        coverage = detected_area / page_area

        # Penalty if coverage is too low or too high (single panel usually means failure)
        confidence = 0.5
        if 2 <= len(panels) <= 12:
            confidence = min(1.0, coverage * 1.2)
        elif len(panels) == 1:
            confidence = 0.3  # Single panel detected is suspicious in comics unless splash

        return panels, float(confidence)
