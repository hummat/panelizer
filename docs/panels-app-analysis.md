# Panels App Analysis

> **Note:** This document analyzes the [Panels](https://panels.app/) iOS/iPadOS app.
>
> **Update (Dec 2025):** The Panels app has announced it will **shut down on December 31, 2025**, and its source code will be released under **Apache 2.0** around **January 4, 2026**.

## Summary
Panels was a premium comic reader for iOS that integrated proprietary machine learning ("Panels View") to offer a Guided View experience on local files. For years, it represented the "bar to beat" for UX quality.

With the announcement of its open-sourcing, it transitions from a commercial competitor to a potential **foundational reference** for open-source readers like Panelizer.

## Technical Deep Dive (Known & Inferred)

1.  **CoreML Integration**:
    - Uses on-device CoreML models for detection.
    - "Panels View" (experimental) allowed panel-by-panel navigation.
    
2.  **Smart Crop**:
    - A feature to intelligently remove white borders, likely using saliency or edge detection.

3.  **Experimental Labs**:
    - Kept ML features in a "Labs" section, acknowledging the difficulty of perfect automation.

## Key Features
- **UX Polish**: Best-in-class gestures, animations, and library management.
- **iCloud Sync**: Seamlessly syncs reading progress across Apple devices.
- **Smart Crop**: Intelligent border removal.
- **Manga Support**: Native RTL support and metadata tagging.

## Limitations
- **Apple Only**: Strictly iOS/iPadOS/macOS.
- **No Overrides**: Users could not correct failed detections (prior to shutdown).

## Use in Panelizer
Panels was our "UX North Star". With its open-sourcing in 2026, it becomes a **primary technical resource**.

**Strategic Shifts:**
1.  **Code Analysis**: Once open-sourced, we should evaluate their ML models (CoreML) and see if they can be converted to ONNX/YOLO formats for cross-platform use.
2.  **UX Patterns**: We can study their implementation of smooth zoom transitions and gesture handling.
3.  **Native iOS**: If Panelizer ever needs a native iOS client, the Panels codebase could serve as the base.

## License
**Proprietary** (Until Dec 31, 2025) -> **Apache 2.0** (From Jan 4, 2026).

## References
- [Official Website](https://panels.app/)
- [Shutdown Announcement](https://panels.art)
