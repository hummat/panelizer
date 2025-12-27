# SAM-Comic Analysis

> **Note:** This document analyzes the evolution of Meta's Segment Anything Model (SAM) and its application to comic panel extraction, specifically focusing on `segment-anything-comic` and newer foundation models.

## Summary
The Segment Anything Model (SAM) family provides pixel-perfect polygon masks, which are superior to YOLO's bounding boxes for irregular panel shapes (circles, jagged explosions, slanted layouts). While the original SAM was slow, subsequent versions have significantly improved speed and prompting capabilities.

## Model Evolution

### SAM 1 (April 2023)
The original foundation model that proved zero-shot segmentation was possible at scale.
- **Architecture**: ViT-H/L/B backbone with a lightweight mask decoder.
- **Comic Application**: Fine-tuned by projects like `segment-anything-comic` to handle gutters and "negative frames".
- **Limitation**: Very slow; required point or box prompts; struggled with high-resolution global context.

### SAM 2 (July 2024)
A unified model for image and video segmentation.
- **Architecture**: Replaces ViT with **Hiera** (a hierarchical Vision Transformer) for richer embeddings.
- **Improvements**: **6x faster** than SAM 1 for image tasks. Introduced a memory mechanism and an occlusion head.
- **Relevance**: The speed increase makes it viable for local batch processing. Its ability to track objects in video could assist in "Motion Comic" conversion or consistent panel tracking across similar layouts.

### SAM 3 (November 2025)
The latest generation of promptable segmentation.
- **Architecture**: Fully integrated multi-modal transformer.
- **Improvements**: Native **Text-to-Segmentation**. You can prompt for "all circular panels" or "the explosion panel" directly without providing coordinates.
- **Relevance**: Drastically simplifies the "Stage 3" ordering and semantic understanding by allowing natural language queries for specific panel types or content.

## Comic-Specific Fine-tuning

Researchers have found that for comics, the best results come from:
1.  **Fine-tuning the Mask Decoder**: Keeping the heavy image encoder frozen while teaching the decoder the specific "language" of comic gutters and panel boundaries.
2.  **Synthetic Data**: Training on procedurally generated comic layouts to handle the "infinite" variety of panel shapes.
3.  **Corner Detection**: Using SAM backbones to detect panel corners as anchors for arbitrary polygons.

## Key Features for Panelizer
- **True Shapes**: Handles non-rectangular panels perfectly (essential for 5-10% of modern comics).
- **Negative Space**: Can identify panels defined only by the absence of art (e.g., white space surrounding a figure).
- **Zero-Shot Potential**: Requires very little data to adapt to new artistic styles.

## Limitations
- **Compute Cost**: Even SAM 2/3 require significant VRAM (8GB+) compared to classic CV or YOLO.
- **Overkill**: For standard 9-panel grids, YOLOv12 remains the efficiency king.
- **Post-processing**: Masks must be converted to simplified polygons for the PWA viewer to render efficiently via CSS/SVG.

## Use in Panelizer
SAM is the engine for **"High Precision" mode**.

- **Current Status**: Not integrated in v0.1.
- **Future Role**: 
    - **Trigger**: If YOLO detection suggests high overlap or low confidence in box boundaries.
    - **Semantic Query (SAM 3)**: Allowing users to type "fix the slanted panel" and having SAM 3 re-segment that specific area.

## License
**Apache 2.0**.

## References
- [Segment Anything 1 Paper (2023)](https://arxiv.org/abs/2304.02643)
- [Segment Anything 2 (Meta AI)](https://ai.meta.com/blog/segment-anything-2/)
- [SAM 3 Release Notes (November 2025)](https://roboflow.com/model/sam-3)
- [segment-anything-comic GitHub](https://github.com/Vrroom/segment-anything-comic)