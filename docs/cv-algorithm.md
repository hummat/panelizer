# Stage 1: CV Algorithm & Confidence Scoring

This document details the "Classic CV" pipeline implementation (`src/panelizer/cv/`) used for on-device panel detection. It combines techniques from [Kumiko](https://github.com/njean42/kumiko) with a novel confidence scoring system to determine when to fallback to ML.

## Pipeline Steps

The detection pipeline (`pipeline.py`) processes each page image through the following stages:

1. **Preprocessing**:
   - Convert to Grayscale.
   - **Sobel Edge Detection**: Computes gradients (x/y) to highlight boundaries.
   - **Line Segment Detection (LSD)**: Finds strong linear features using OpenCV's LSD. These are crucial for handling panel borders that aren't perfect rectangles.

2. **Initial Proposals**:
   - **Contour Detection**: derived from the Sobel edges.
   - **Polygon Approximation**: `cv.approxPolyDP` simplifies contours into polygons.
   - **Size Filtering**: Polygons smaller than `min_panel_ratio` (default 10% of page) are discarded.

3. **Refinement Loop**:
   - **Segment Splitting**: Iteratively splits panels if internal line segments (from LSD) suggest a cut. This handles "grid" layouts where contours might merge adjacent panels.
   - **Small Panel Grouping**: Clusters tiny adjacent panels (e.g., shattered glass effect) into a single logical panel.
   - **Merge & De-overlap**: Merges panels where one contains >50% of another. Adjusts boundaries to remove slight overlaps.
   - **Expansion**: Expands panels to "snap" to gutters or page edges (based on `actual_gutters` analysis).

4. **Grouping**:
   - **Big Panel Grouping**: Merges adjacent panels if no strong *axis-aligned* gutter segment exists between them. Diagonal segments (e.g., motion lines, speed lines) are ignored since real gutters are nearly always horizontal or vertical. This handles panels drawn without clear borders while avoiding false splits from artistic elements.

## Confidence Scoring

A key differentiator of Panelizer is its ability to self-assess quality. The confidence score (0.0 - 1.0) determines if the page should be flagged for Stage 2 (ML/YOLO).

The score is a geometric mean of three factors, implemented in `detector.py`:

$$ Confidence = \\sqrt[3]{F_{count} \\times F_{coverage} \\times F_{split}} $$

### 1. Panel Count Factor ($F_{count}$)

Checks if the number of detected panels is "reasonable" for a comic page.

- **1 panel**: $0.7$ (Ambiguous: could be a splash page, or a failure to detect splits).
- **2-12 panels**: $1.0$ (Healthy range).
- **>12 panels**: $0.5$ (Likely over-fragmentation/noise).

### 2. Coverage Factor ($F_{coverage}$)

Checks what percentage of the page area is covered by panels.

- **70% - 95%**: $1.0$ (Healthy: typical comic page with margins).
- **Otherwise**: $0.8$ (Too empty or too full, suggesting missed panels or false positives).

### 3. Split Quality Factor ($F_{split}$)

If panels were split using line segments (LSD), how "strong" were those segments?

- Average of `split_coverages` (fraction of the split line covered by actual detected image segments).
- If no splits occurred: $0.8$ (Conservative baseline).

### Thresholds

- **> 0.8**: High confidence. Use result.
- **< 0.8**: Low confidence. Send to Stage 2 (YOLO) or flag for human review.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_panel_ratio` | 0.10 | Minimum panel dimension as fraction of page size. |
| `panel_expansion` | `True` | Snap panels to gutters/edges. |
| `small_panel_grouping` | `True` | Merge tiny clusters. |
| `big_panel_grouping` | `True` | Merge borderless adjacent panels. |

## Known Limitations

CV-based detection works well for pages with clear panel borders but struggles with:

- **Motion lines / speed lines**: Artistic lines inside panels can create false contours that fragment a single panel into multiple pieces. The axis-aligned segment filter helps with grouping, but contour detection itself may still produce fragments.
- **Borderless panels**: Panels that blend into each other without clear edges.
- **Complex artistic layouts**: Overlapping panels, irregular shapes, panels-within-panels.
- **Low contrast borders**: Faint or stylized panel borders that don't produce strong edges.

When CV confidence is low (< 0.8), the page should be processed by Stage 2 (ML/YOLO) or flagged for manual review.

