# Marvel Smart Panels Analysis

> **Note:** This document analyzes **Marvel Smart Panels** (and related Adaptive Audio), the proprietary reading technology inside the Marvel Unlimited app.

## Summary
Marvel Smart Panels is the direct competitor to Comixology's Guided View, optimized specifically for the Marvel catalog. It emphasizes keeping the reader in the "flow" of the action.

## Technical Deep Dive

1.  **Manual Curation**: Like Comixology, major titles receive human attention to ensure reading order and zoom levels are perfect.
2.  **Adaptive Audio** (Historical): Marvel experimented with adding soundtracks and sound effects that triggered as the user swiped to specific panels, syncing audio with visual beats.

## Key Features
- **Narrative Focus**: The viewports often crop tightly on faces or action, sometimes ignoring the actual panel borders to emphasize drama.
- **Catalog Integration**: Available instantly for tens of thousands of back-issue comics in the subscription.

## Limitations
- **Proprietary**: Locked to the Marvel Unlimited app.
- **Inconsistent**: Older or less popular issues may rely on rougher automated detection compared to flagship titles.

## Use in Panelizer
Marvel Smart Panels serves as an inspiration for **narrative-aware ordering**.
- While Panelizer v1 is geometric, future versions (using VLMs) could emulate Marvel's ability to "direct" the eye based on story beats rather than just box coordinates.

## References
- [Marvel Unlimited](https://www.marvel.com/unlimited)
