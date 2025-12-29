# Smart Comic Reader Analysis

> **Note:** This document analyzes [Smart Comic Reader](https://smartcomicreader.com/), an iOS comic reader app known for its automated panel detection capabilities.

## Summary
Smart Comic Reader (developed by Onne van Dijk) focuses heavily on the "guided view" experience. Unlike general-purpose readers that treat panel-by-panel reading as a secondary feature, this app centers its value proposition on automatically processing pages to show one panel at a time.

## Key Features

- **Aggressive Automation**: Automatically zooms and pans between panels.
- **Loose Edge Tolerance**: Marketing claims it works on comics "where panels don't need well-defined edges," suggesting a segmentation approach that might be more robust than simple contour finding (or uses heuristics for borderless panels).
- **On-Device Processing**: Appears to process files locally on the device.

## Relevance to Panelizer

Smart Comic Reader validates the demand for:
1.  **Automated Processing**: Users want to just open a file and read, without manual cropping.
2.  **Resilience**: The ability to handle "imperfect" layouts (borderless, overlapping) is a key differentiator.

## Differentiators
- **Panelizer's Edge**: Smart Comic Reader is an iOS app. Panelizer targets cross-platform usage (PWA) and, crucially, allows **saving/exporting** the detection data and **manual overrides** (it is unclear if Smart Comic Reader allows editing the detected panels).

## References
- [Official Website](https://smartcomicreader.com/)
