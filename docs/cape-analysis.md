# C.A.P.E. Analysis

> **Note:** This document analyzes [C.A.P.E. (Comic Analysis and Panel Extraction)](https://github.com/CodeMinion/C.A.P.E), a pioneer in human-in-the-loop comic editing.

## Summary
C.A.P.E. is a legacy desktop application (last updated 2018) that introduced **adaptive parameter tuning** and a professional-grade panel editor. It is the architectural inspiration for Panelizer's "Proposal vs. Truth" philosophy.

## Technical Deep Dive

The most innovative part of C.A.P.E. is the `findBestPanels` algorithm in `panelextractor.py`:

1.  **Adaptive Binary Search**:
    - Instead of one-shot detection, it runs a loop to find the best `erosion` and `dilation` iterations.
    - It uses a **Binary Search** to find a parameter set that results in a panel count closest to a "Goal" (usually 5-8 panels per page).
    - It validates layouts using `isGoodLayout`, which checks for overlapping panels or panels that cover unrealistic portions of the page.

2.  **Border Logic**:
    - `hasDarkBorders`: Specifically checks corners and edges to decide if the comic uses black or white gutters, adjusting the inversion logic accordingly.

3.  **Delaunay Triangulation**:
    - Uses `cv2.Subdiv2D` (Delaunay) for certain internal geometry checks, a technique rarely seen in other comic tools.

## Key Features
- **Human-in-the-loop**: The first tool to treat CV as a "proposal" that the user corrects in a UI.
- **Layout Validation**: Proactively rejects detections that don't "look like" a comic page (e.g., too much overlap).

## Limitations
- **Legacy Stack**: Built for Python 2.7 and older Electron versions.
- **Speed**: The iterative binary search is slow, as it re-runs extraction multiple times per page.
- **Heuristic-Heavy**: Relies heavily on the assumption that pages have a "target" number of panels.

## Use in Panelizer
We adopt C.A.P.E.'s **User-is-Sacred** approach. Overrides are stored separately and automation never clobbers human work. We improve on their "Adaptive Search" by using a much faster one-shot **Stage 2 (ML)** fallback instead of iterative CV loops.

## License
**ISC License**.

## References
- [C.A.P.E. GitHub](https://github.com/CodeMinion/C.A.P.E)