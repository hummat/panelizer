# Comic Trim Analysis

> **Note:** This document analyzes **Comic Trim**, an Android comic reader known for its heuristic "Vignette" mode.

## Summary
Comic Trim is a popular Android reader that attempts panel-by-panel reading using lightweight computer vision. It represents the "middle ground" between basic readers and ML-powered tools.

## Technical Deep Dive

1.  **Uniform Edge Requirement**:
    - The app explicitly states it works best on comics with "uniform edges" and "isolated vignettes".
    - This indicates a reliance on **Connected Components** or simple **Contour Detection** algorithms looking for whitespace gutters.
    
2.  **Processing**:
    - Runs a fast, on-device pass using OpenCV (or similar) to find white/black separators.
    - Does not appear to use deep learning, as it fails predictably on overlapping or borderless panels.

## Key Features
- **Mobile-First**: Designed specifically for reading CBZ on phone screens.
- **Fast**: Heuristics are instant, requiring no heavy model inference.

## Limitations
- **Fragile**: Fails on "modern" layouts (overlapping panels, splash pages, bleeding art).
- **Discontinued/Dormant**: Updates have slowed or stopped.
- **Closed Source**: Implementation details are hidden.

## Use in Panelizer
Comic Trim serves as a **warning** against relying solely on Stage 1 (CV) heuristics.
- Its failure modes on complex pages validate our need for **Stage 2 (ML Fallback)**.
- However, its speed validates our decision to *start* with CV before resorting to heavy models.

## License
**Proprietary / Freemium**.

## References
- [Google Play Store](https://play.google.com/store/apps/details?id=com.comic.trim)