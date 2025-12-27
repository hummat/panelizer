# Panels App Analysis

> **Note:** This document analyzes the [Panels](https://panels.app/) iOS/iPadOS app, widely considered the commercial gold standard for modern comic reading.

## Summary
Panels is a premium comic reader that integrates proprietary machine learning ("Panels View") to offer a Guided View experience on local files. It represents the "bar to beat" for UX quality.

## Technical Deep Dive (Inferred)

Being closed-source, its architecture is inferred from behavior:

1.  **CoreML Integration**: Likely uses Apple's Vision framework or custom CoreML models for on-device detection.
2.  **Saliency Analysis**: "Smart Crop" features suggest it identifies key subjects (faces, text bubbles) to center the view, not just panel boundaries.
3.  **Experimental Labs**: Keeps ML features in a "Labs" section, acknowledging that automation is imperfect—validating Panelizer's "Proposal" philosophy.

## Key Features
- **UX Polish**: Best-in-class gestures, animations, and library management.
- **iCloud Sync**: Seamlessly syncs reading progress (and files) across devices.
- **Smart Crop**: Intelligently removes white borders from pages.

## Limitations
- **Proprietary**: Closed ecosystem.
- **Apple Only**: No Android or Web version.
- **No Overrides**: If the "Panels View" detection is wrong, the user cannot correct it; they must disable the feature for that page.

## Use in Panelizer
Panels is our **UX North Star**.
- We aim to match its "smoothness" and "smart zoom" behaviors in our PWA viewer.
- We differentiate by offering **Platform Independence** (Web/Android/Windows) and **Editability** (fixing the detection mistakes that Panels can't).

## License
**Proprietary / Commercial**.

## References
- [Official Website](https://panels.app/)