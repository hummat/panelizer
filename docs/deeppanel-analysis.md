# DeepPanel Analysis

> **Note:** This document analyzes [DeepPanel](https://github.com/pedrovgs/DeepPanel), a machine learning approach optimized for mobile devices.

## Summary
DeepPanel is a research project focused on running comic panel segmentation directly on Android and iOS devices. It uses a custom U-Net CNN architecture trained on a private dataset.

## Architecture

- **Model**: U-Net (Encoder-Decoder CNN).
- **Task**: Semantic Segmentation (Pixel-wise classification).
- **Classes**: 3 classes per pixel:
    1.  Background
    2.  Panel Border
    3.  Panel Content
- **Framework**: TensorFlow / TFLite.

## Key Features
- **Mobile Optimized**: Designed to run < 1 second on mobile CPUs/NPUs.
- **Semantic Segmentation**: Can theoretically handle irregular shapes better than bounding boxes.
- **Reading Order**: Attempts to simulate human reading behavior for grouping, though implementation details are sparse.

## Limitations
- **Private Dataset**: Trained on 550 pages that are not public. Retraining requires creating a new dataset from scratch.
- **No Confidence Score**: Semantic segmentation masks don't inherently provide a "panel confidence" score like object detectors (YOLO) do.
- **Maintenance**: Last updated ~2021.
- **Complexity**: Integrating a U-Net and post-processing masks is more complex than bounding box regression.

## Use in Panelizer
DeepPanel is **not** currently used in Panelizer because:
1.  **YOLOv12 (Stage 2)** offers state-of-the-art accuracy and speed with a simpler bounding-box output, which is sufficient for 95% of comics.
2.  **Data Availability**: We cannot improve DeepPanel without its dataset.
3.  **Maintenance**: The project is effectively dormant.

However, its mobile libraries (Android/iOS) are excellent references if we ever build native apps.

## License
**Apache 2.0**.

## References
- [DeepPanel GitHub](https://github.com/pedrovgs/DeepPanel)
- [DeepPanel Android](https://github.com/pedrovgs/DeepPanelAndroid)
