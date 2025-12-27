# SAM-Comic Analysis

> **Note:** This document analyzes `segment-anything-comic`, a fine-tuned version of Meta's Segment Anything Model (SAM) for comics.

## Summary
This project fine-tunes the mask decoder of SAM to specifically segment comic panels. Unlike YOLO (which gives boxes), SAM generates pixel-perfect polygon masks.

## Architecture

- **Base Model**: Meta SAM (ViT-H or similar backbone).
- **Modification**: Fine-tuned mask decoder on comic datasets.
- **Output**: Binary masks / Polygons.

## Key Features
- **True Shapes**: Handles circles, jagged explosions, and slanted panels perfectly.
- **Zero-Shot Potential**: Inherits SAM's ability to "understand" objects generally.

## Limitations
- **Speed**: SAM is significantly slower than YOLO or classic CV. It is a heavy Vision Transformer.
- **Compute Cost**: Requires substantial GPU VRAM.
- **Overkill**: Most comic panels (>95%) are rectangular. Using SAM for everything is inefficient.

## Use in Panelizer
SAM-Comic is considered for a future **"High Precision" mode** or specific fallback for irregular layouts.

- **Current Status**: Not integrated in v0.1.
- **Future Role**: If YOLO detection suggests high overlap (indicating non-rectangular shapes) or if a user specifically requests "High Precision" processing, SAM could be invoked to generate polygons instead of boxes.

## License
**Apache 2.0** (Inherited from SAM).

## References
- [GitHub Repository](https://github.com/Vrroom/segment-anything-comic)
- [Segment Anything Paper](https://arxiv.org/abs/2304.02643)
