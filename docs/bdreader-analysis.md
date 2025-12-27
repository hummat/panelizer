# BDReader Analysis

> **Note:** This document analyzes [BDReader](https://sourceforge.net/projects/bdreader/), an early open-source pioneer in panel-by-panel comic reading.

## Summary
BDReader (Band Dessinée Reader) was one of the first open-source projects to attempt automated panel detection for comfortable reading on small screens. Developed in C++ with Qt, it targeted early PDAs and netbooks.

## Technical Deep Dive

The core approach relied on classical Computer Vision heuristics:
1.  **Page Segmentation**: Detected panel frames using edge detection and contour analysis.
2.  **Comfort Mode**: The "Auto-detection" feature broke the page into individual views.
3.  **Navigation**: Users click/tap to advance to the next detected panel.

## Key Features
- **Auto-detection**: One of the earliest implementations of "Guided View" outside of commercial walled gardens.
- **Privacy**: Fully offline, local file reading.
- **Cross-Platform**: Built on Qt, supporting Windows and Linux.

## Limitations
- **Abandoned**: Last updated around 2015.
- **Heuristic Only**: Lacks machine learning capabilities, meaning it fails on complex/irregular layouts.
- **UI/UX**: Dated interface compared to modern mobile standards.
- **No Editing**: If detection failed, the user was stuck.

## Use in Panelizer
BDReader serves as a **historical reference**. It demonstrated that heuristic detection is feasible for ~70% of comics but also highlighted the "uncanny valley" of frustration when automation fails without an escape hatch (overrides).

## License
**GPL v2**.

## References
- [SourceForge Project](https://sourceforge.net/projects/bdreader/)
