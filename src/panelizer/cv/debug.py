"""Debug visualization for the CV detection pipeline.

Provides step-by-step visualization of the panel detection process,
inspired by Kumiko's debug output but with a simpler implementation.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2 as cv
import numpy as np

from .panel_internal import InternalPanel
from .segment import Segment


@dataclass
class DebugStep:
    """A single step in the detection pipeline."""

    name: str
    panel_count: int
    elapsed_ms: float
    image_filename: Optional[str] = None


@dataclass
class DebugContext:
    """Context for tracking debug information during detection.

    Usage:
        ctx = DebugContext(enabled=True, output_dir=Path("debug_output"))
        ctx.set_base_image(img)
        ctx.add_step("Initial", panels)
        ctx.add_image("Sobel edges", sobel_img)
        ...
        ctx.save_html()
    """

    enabled: bool = False
    output_dir: Optional[Path] = None
    steps: List[DebugStep] = field(default_factory=list)
    _base_img: Optional[np.ndarray] = None
    _current_img: Optional[np.ndarray] = None
    _start_time: float = field(default_factory=time.perf_counter)
    _last_step_time: float = field(default_factory=time.perf_counter)
    _img_counter: int = 0

    # Color palette for drawing
    COLORS = {
        "white": (255, 255, 255),
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "yellow": (0, 255, 255),
        "purple": (255, 0, 255),
        "gray": (150, 150, 150),
    }

    def set_base_image(self, img: np.ndarray) -> None:
        """Set the base image for drawing overlays."""
        if not self.enabled:
            return
        self._base_img = img.copy()
        self._current_img = img.copy()

    def reset_overlay(self) -> None:
        """Reset the current image to the base image."""
        if not self.enabled or self._base_img is None:
            return
        self._current_img = self._base_img.copy()

    def add_step(self, name: str, panels: List[InternalPanel]) -> None:
        """Record a pipeline step with panel count and timing."""
        if not self.enabled:
            return

        now = time.perf_counter()
        elapsed_ms = (now - self._last_step_time) * 1000
        self._last_step_time = now

        step = DebugStep(
            name=name,
            panel_count=len(panels),
            elapsed_ms=elapsed_ms,
        )
        self.steps.append(step)

    def add_image(self, label: str, img: Optional[np.ndarray] = None) -> None:
        """Save an intermediate image to the output directory."""
        if not self.enabled or self.output_dir is None:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Use provided image or current overlay
        save_img = img if img is not None else self._current_img
        if save_img is None:
            return

        # Generate filename
        clean_label = label.lower().replace(" ", "-").replace("/", "-")
        filename = f"{self._img_counter:02d}-{clean_label}.jpg"
        self._img_counter += 1

        filepath = self.output_dir / filename
        cv.imwrite(str(filepath), save_img)

        # Associate with current step if any
        if self.steps:
            self.steps[-1].image_filename = filename

        # Reset overlay for next step
        self.reset_overlay()

    def draw_contours(self, contours: List[np.ndarray], color: str = "green", thickness: int = 2) -> None:
        """Draw contours on the current overlay image."""
        if not self.enabled or self._current_img is None:
            return

        cv_color = self.COLORS.get(color, self.COLORS["green"])
        for contour in contours:
            cv.drawContours(self._current_img, [contour], 0, cv_color, thickness)

    def draw_segments(self, segments: List[Segment], color: str = "green", thickness: int = 2) -> None:
        """Draw line segments on the current overlay image."""
        if not self.enabled or self._current_img is None:
            return

        cv_color = self.COLORS.get(color, self.COLORS["green"])
        for seg in segments:
            pt1 = (int(seg.a[0]), int(seg.a[1]))
            pt2 = (int(seg.b[0]), int(seg.b[1]))
            cv.line(self._current_img, pt1, pt2, cv_color, thickness)

    def draw_panels(self, panels: List[InternalPanel], color: str = "red", thickness: int = 3) -> None:
        """Draw panel bounding boxes on the current overlay image."""
        if not self.enabled or self._current_img is None:
            return

        cv_color = self.COLORS.get(color, self.COLORS["red"])
        for p in panels:
            cv.rectangle(self._current_img, (p.x, p.y), (p.r, p.b), cv_color, thickness)

    def total_time_ms(self) -> float:
        """Get total processing time in milliseconds."""
        return (time.perf_counter() - self._start_time) * 1000

    def save_html(self) -> Optional[Path]:
        """Generate an HTML report showing all debug steps."""
        if not self.enabled or self.output_dir is None:
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)

        html_parts = [
            "<!DOCTYPE html>",
            "<html><head>",
            "<meta charset='utf-8'>",
            "<title>Panelizer Debug Output</title>",
            "<style>",
            "body { font-family: system-ui, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }",
            "h1 { color: #333; }",
            ".step { border: 1px solid #ddd; margin: 20px 0; padding: 15px; border-radius: 8px; }",
            ".step-header { display: flex; justify-content: space-between; align-items: center; }",
            ".step-name { font-weight: bold; font-size: 1.2em; }",
            ".step-stats { color: #666; font-size: 0.9em; }",
            ".step-image { max-width: 100%; margin-top: 15px; border: 1px solid #eee; }",
            ".summary { background: #f5f5f5; padding: 15px; border-radius: 8px; margin-bottom: 20px; }",
            "</style>",
            "</head><body>",
            "<h1>Panelizer Detection Pipeline</h1>",
            f"<div class='summary'>Total processing time: {self.total_time_ms():.1f}ms</div>",
        ]

        for i, step in enumerate(self.steps):
            html_parts.append("<div class='step'>")
            html_parts.append("<div class='step-header'>")
            html_parts.append(f"<span class='step-name'>{i + 1}. {step.name}</span>")
            html_parts.append(f"<span class='step-stats'>{step.panel_count} panels | {step.elapsed_ms:.1f}ms</span>")
            html_parts.append("</div>")

            if step.image_filename:
                html_parts.append(f"<img class='step-image' src='{step.image_filename}' alt='{step.name}'>")

            html_parts.append("</div>")

        html_parts.extend(["</body></html>"])

        html_path = self.output_dir / "index.html"
        html_path.write_text("\n".join(html_parts))
        return html_path
