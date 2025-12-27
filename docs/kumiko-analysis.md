# Kumiko Analysis

> **Note:** This document analyzes [Kumiko](https://github.com/njean42/kumiko), an active Python-based comic panel detector. It serves as the primary reference for Panelizer's Stage 1 CV pipeline.

## Summary
Kumiko is a sophisticated tool for slicing comics into panels using classic Computer Vision. While originally a C++ project, the current version is implemented in Python and provides a robust, non-ML alternative for panel extraction.

## Technical Deep Dive

The core logic resides in `lib/page.py` and `lib/panel.py`. The pipeline is a deterministic sequence of image transformations:

1.  **Gradient Analysis (Sobel)**:
    - Uses `cv.Sobel` for both X and Y directions.
    - Combines them using `cv.addWeighted` to create a gradient map.
    - This is more robust than simple thresholding as it identifies transitions (borders) regardless of absolute brightness.

2.  **Adaptive Filtering**:
    - Implements **Panel Expansion** (`expand_panels`): A recursive algorithm that snaps panel edges to detected gutters or neighboring panels. It uses a "gutter estimation" phase to determine the whitespace between panels.
    - **Small Panel Grouping**: Uses a clustering algorithm (in `lib/panel.py`) to merge adjacent tiny fragments, which often occur in action scenes with shattered borders.

3.  **Polygon-based Splitting**:
    - Unlike simpler tools that only use bounding boxes, Kumiko maintains **polygons**.
    - It uses **recursive XY-cuts** assisted by the Line Segment Detector (LSD) to split merged panels. If a line segment significantly bisects a detected contour, it triggers a split.

4.  **License Handling**:
    - A unique feature is built-in support for `.license` sidecar files (JSON), allowing metadata and licensing info to travel with the image.

## Key Features
- **Deterministic & Fast**: Runs entirely on CPU with predictable performance.
- **Advanced Post-processing**: The expansion and grouping logic handles "messy" traditional comic layouts better than standard contour detection.
- **Manga Support**: Built-in `ltr`/`rtl` toggle for reading direction.

## Limitations
- **Parameter Sensitivity**: While robust, the `small_panel_ratio` and Sobel parameters are fixed, which can lead to failures on very high-resolution or very low-contrast scans.
- **No Probabilistic Fallback**: If CV fails, it doesn't provide a confidence score, it just returns its best (potentially wrong) guess.

## Use in Panelizer
Panelizer's **Stage 1** is an evolution of this pipeline. We have:
- Ported the core Sobel/LSD logic.
- Added **Confidence Scoring** (based on panel count, coverage, and split strength).
- Integrated the `.license` sidecar concept into our schema.

## License
**GNU AGPLv3**.

## References
- [Kumiko GitHub](https://github.com/njean42/kumiko)