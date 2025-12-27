# Comic Smart Panels Analysis

> **Note:** This document analyzes [Comic Smart Panels](https://github.com/zoran123456/Comic-Smart-Panels), a project focused on the manual creation and viewing of guided comic experiences.

## Summary
Comic Smart Panels is a suite of tools (Creator and Viewer) for defining "smart panels" on digital comics. Unlike Panelizer, which aims for automation first, this project focuses on a robust manual authoring workflow.

## Technical Deep Dive

The project was built on the Microsoft .NET stack (WPF):

1.  **Data Format (CPD)**:
    - Used a custom JSON-based format (`.cpd` or `.cpr`) to store panel definitions.
    - Stored panel coordinates and reading order.
    
2.  **Creator Tool**:
    - A dedicated GUI for drawing vector-like boxes over pages.
    - Featured "Grid Snapping" to speed up manual annotation.
    - Allowed relative positioning (percentages) to handle different screen sizes.

3.  **Viewer**:
    - A standard comic reader that respected the `.cpd` sidecar files to trigger zoom animations.

## Key Features
- **Human-First**: Prioritizes perfect human curation over imperfect automation.
- **Vector Definitions**: Panels are defined geometrically, not just as raster crops.
- **Sidecar Metadata**: Keeps the original comic file pristine (non-destructive).

## Limitations
- **Manual Labor**: Requires users to draw *every* panel by hand. No automation assistance.
- **Windows Only**: Built on WPF, limiting its reach to Windows desktops.
- **Abandoned**: Last significant activity was around 2015.

## Use in Panelizer
This project is a strong reference for **Stage 4 (Human-in-the-loop)**.
- We adopt the "sidecar metadata" philosophy (our `panels.json` vs their `.cpd`).
- We admire the "Creator" UI patterns (snap-to-grid, drag-to-draw) for our own Override Editor.

## License
**MIT** (Implied by GitHub presence, though explicit license file may be missing).

## References
- [GitHub Repository](https://github.com/zoran123456/Comic-Smart-Panels)
