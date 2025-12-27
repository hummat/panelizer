# YOLOv12 Analysis (best-comic-panel-detection)

> **Note:** This document analyzes the `best-comic-panel-detection` model, a fine-tuned YOLOv12x model hosted on HuggingFace. It is the default choice for Panelizer's Stage 2 ML fallback.

## Summary
This is a state-of-the-art object detection model fine-tuned specifically for comic panels. It achieves near-perfect performance on standard validation benchmarks.

## Architecture

- **Base Model**: YOLOv12x (You Only Look Once, v12, Extra Large variant).
- **Framework**: PyTorch / Ultralytics.
- **Task**: Object Detection (Bounding Box Regression).
- **Class**: Single class: `Comic Panel`.

## Performance

| Metric | Value |
|--------|-------|
| mAP50 | **0.991** |
| mAP50-95 | **0.985** |

These metrics indicate it is incredibly reliable at finding panel bounding boxes, even with complex overlap.

## Key Features
- **Speed**: YOLO architectures are designed for real-time inference.
- **Simplicity**: Returns clean bounding boxes `[x, y, w, h]` and confidence scores.
- **Integration**: The `ultralytics` Python package provides a one-line API (`model.predict()`).

## Limitations
- **Bounding Boxes Only**: Cannot perfectly capture non-rectangular (e.g., circular, slanted) panels. It will return the bounding rectangle.
- **Model Size**: The `x` (Extra Large) variant is heavy (~hundreds of MBs), requiring a GPU for fast batch processing.
- **No Order**: Like all object detectors, it returns unordered boxes. Ordering must be done in post-processing.

## Use in Panelizer
This model is the engine for **Stage 2 (ML Fallback)**.
- **Trigger**: When Stage 1 (CV) confidence is low (< 0.8).
- **Role**: Provides robust detection for "hard" pages where heuristics fail.
- **Post-processing**: We apply the same heuristic ordering (row-major) to the YOLO outputs as we do for CV outputs.

## License
**Apache 2.0**.

## References
- [HuggingFace Model Page](https://huggingface.co/mosesb/best-comic-panel-detection)
- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
