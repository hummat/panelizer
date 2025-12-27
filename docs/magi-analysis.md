# Magi Model Analysis

> **Note:** This document analyzes the [Magi](https://github.com/ragavsachdeva/magi) project, specifically versions v1 (2024) and v3 (2025).

## Summary
Magi is the state-of-the-art research project for **end-to-end comic understanding**. It doesn't just find panels; it transcribes the entire reading experience into prose or structured data.

## Technical Deep Dive

1.  **Evolution (v1 to v3)**:
    - **v1 (CVPR'24)**: Focused on "The Manga Whisperer". Uses a multi-task Conditional DETR to detect Panels, Text, and Characters simultaneously.
    - **v3 (2025)**: "From Panels to Prose". Upgraded to a Large Vision-Language Model (LVLM) backbone (Florence-2/DaViT). This allows for semantic understanding (e.g., "who is speaking even if the bubble is between characters").

2.  **Reading Order Heuristics**:
    - Despite its complexity, Magi's **Panel Ordering** is still largely geometric. It uses the detected panels to create a "reading flow" but relies on text box sequence to verify the flow.
    - For Manga (RTL), it uses a specialized sorting weight that prioritizes the top-right over the top-left.

3.  **Multi-Task Associations**:
    - Creates "hyper-edges" between Text -> Character -> Panel. This allows it to know which character a dialogue belongs to, even in "borderless" or "floating" text layouts.

## Key Features
- **Semantic Understanding**: The only tool that "reads" the content to help determine order.
- **Chapter-wide Tracking**: Magi v2/v3 can track a character's identity across the whole book.

## Limitations
- **Academic License**: Strictly **Non-Commercial / Research Only**.
- **Resource Intensive**: Requires significant VRAM (8GB+) for inference.
- **Manga-Centric**: While it works for western comics, most of its training data and bias are towards Manga (RTL).

## Use in Panelizer
Magi serves as our **Stage 3 (Ordering) benchmark**. While we don't use their code due to licensing, we study their "Dialogue Association" patterns to improve our own VLM-based ordering prompts.

## License
**Academic Research Only**.

## References
- [Magi GitHub](https://github.com/ragavsachdeva/magi)
- [Manga Whisperer Paper (2024)](https://arxiv.org/abs/2401.10224)
- [From Panels to Prose (2025)](https://arxiv.org/abs/2501.00000) (approximate cite)