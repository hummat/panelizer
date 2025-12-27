# Comixology Guided View Analysis

> **Note:** This document analyzes **Comixology Guided View**, the commercial industry standard for digital comic reading.

## Summary
Introduced by Comixology (now Amazon), Guided View defined the modern expectation for reading comics on small screens. It transforms a static page into a cinematic, panel-by-panel slideshow.

## Technical Deep Dive

Unlike the automated tools in Panelizer, Guided View relies on **publisher-supplied metadata**:

1.  **Authoring**: Publishers or dedicated teams manually define the viewports and transitions for each page.
2.  **Format**: A proprietary XML or JSON overlay that travels with the DRM-protected file.
3.  **Hybrid Automation**: Tools like "Kindle Create" now offer "auto-detection" to assist authors, but the final output is human-verified.

## Key Features
- **Cinematic Flow**: Transitions (pans, zooms) are timed to match the narrative rhythm.
- **Immersion**: Hides the "page" context to focus solely on the current story beat.
- **Cross-Device**: A single file works on phone, tablet, and desktop web.

## Limitations
- **Walled Garden**: Only available for comics purchased/rented within the Amazon/Comixology ecosystem.
- **Labor Intensive**: Requires manual effort for every single book.
- **DRM**: Users do not own the files or the metadata.

## Use in Panelizer
Guided View is the **functional benchmark**.
- Panelizer's goal is to bring this "premium" reading experience to **local, DRM-free files**.
- We attempt to replicate the *result* (panel-by-panel reading) using *automation* (CV/ML) instead of manual labor.

## References
- [Comixology/Amazon Kindle](https://www.amazon.com/kindle-dbs/comics-store/home)
