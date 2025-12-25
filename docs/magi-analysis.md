# Magi Model Analysis

## Summary

**Magi does not have a specialized panel ordering model.** Panel ordering is heuristic-based, not learned.

## Architecture

| Component | Model |
|-----------|-------|
| Detection | Conditional DETR (Detection Transformer) |
| OCR | VisionEncoderDecoderModel |
| Character crop embedding | ViTMAE (Vision Transformer Masked Autoencoder) |
| Panel ordering | **Heuristics (row-major LTR)** |

## Panel Ordering Implementation

From `modelling_magi.py`:

```python
def sort_panels_and_text_bboxes_in_reading_order(
    self,
    batch_panel_bboxes,
    batch_text_bboxes,
):
    for batch_index in range(len(batch_text_bboxes)):
        panel_bboxes = batch_panel_bboxes[batch_index]
        text_bboxes = batch_text_bboxes[batch_index]

        sorted_panel_indices = sort_panels(panel_bboxes)
        sorted_panels = [panel_bboxes[i] for i in sorted_panel_indices]
        sorted_text_indices = sort_text_boxes_in_reading_order(text_bboxes, sorted_panels)
```

The ordering uses two utility functions:
- `sort_panels()`: Arranges panel bboxes in reading order
- `sort_text_boxes_in_reading_order()`: Arranges text boxes based on sorted panels

## What Makes Magi Special

The "specialized" aspect is **multi-task detection using Conditional DETR**, which detects:
- Characters (class label 0)
- Text (class label 1)
- Panels (class label 2)

Plus learned associations:
- Text-to-character matching
- Character-to-character matching
- Dialogue detection (`is_this_text_a_dialogue`)

## Versions

| Version | Base Model | Notes |
|---------|-----------|-------|
| v1 | Conditional DETR + ViTMAE | Multi-task detection, heuristic ordering |
| v2 | Same as v1 | Chapter-wide character tracking |
| v3 | Florence-2 (DaViT + LM) | New vision tower, ordering not documented |

## Implications for Panel Flow

1. **Magi's ordering is not learned** — it's the same row-major heuristics we already plan to use
2. **The real differentiator** is multi-task detection with learned associations
3. **VLM for ordering may be more sophisticated** than Magi's heuristics
4. **Question:** Does Conditional DETR multi-task detection add enough value over separate YOLO + heuristic ordering?

## References

- [Magi model architecture code](https://huggingface.co/ragavsachdeva/magi/blob/main/modelling_magi.py)
- [Magi GitHub](https://github.com/ragavsachdeva/magi)
- [Magi v1 paper](https://arxiv.org/abs/2401.10224)
