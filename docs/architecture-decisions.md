# Architecture Decisions

This document synthesizes the key architectural decisions for Panelizer, derived from our comprehensive analysis of existing tools and commercial products.

## 1. The "Proposal vs. Truth" Philosophy
**Decision:** All automated outputs (CV or ML) are treated as **proposals**. User overrides are treated as **truth** and are stored separately.
**Why:**
- Heuristic tools (BDReader, Comic Trim) fail on ~30% of pages, causing user frustration.
- Pure ML tools (YOLO, DeepPanel) lack 100% accuracy on reading order.
- **Result:** We must implement a "Stage 4" Override Editor that saves to a `panels.json` sidecar. Automation never overwrites a user's manual correction.

## 2. The Hybrid "CV-First" Pipeline
**Decision:** Use classic Computer Vision (Kumiko-style) for Stage 1, and fall back to YOLOv12 (Stage 2) only when confidence is low.
**Why:**
- **Speed:** CV is instantaneous on-device. YOLO requires loading heavy models (or cloud API calls).
- **Cost:** Running a VLM or heavy GPU model on every page is wasteful when 70% of pages are simple grids.
- **Result:** We implement a `Confidence Score` in Stage 1 to act as the gatekeeper.

## 3. PWA for "Guided View" UX
**Decision:** Build the viewer as a Progressive Web App (PWA) with CSS-based smooth zooming.
**Why:**
- **UX Standard:** Commercial apps (Panels, Comixology) use smooth "cinematic" transitions. Static crops are not enough.
- **Platform Agnostic:** A PWA works on iOS (via "Add to Home Screen"), Android, and Desktop without separate codebases.
- **Privacy:** Files are processed locally via the File API; no comic book content is ever uploaded to a server.

## 4. Reading Order via VLM (Future)
**Decision:** Use Heuristics (row-major) as the default, but reserve Vision-Language Models (VLMs) for "ambiguous" layouts.
**Why:**
- Tools like **Magi** show that understanding *content* (who is speaking?) is required for perfect ordering.
- Geometry alone fails on complex Manga/Western layouts.
- **Result:** We will add a "Cloud Assist" button for pages where heuristics produce low confidence, sending the layout to a VLM to solve the sorting problem.

## 5. Data Format Compatibility
**Decision:** Use a JSON schema that loosely aligns with C.A.P.E.'s `.cpanel` format but extends it for modern needs.
**Why:**
- It allows potential import/export with legacy tools.
- It validates the "Sidecar Metadata" approach which keeps the original CBZ file pristine.

## References

Our decisions are backed by deep dives into the following tools and technologies:

### Open Source & Research
- **[Kumiko](kumiko-analysis.md)** (Stage 1 CV Reference)
*   **[C.A.P.E.](cape-analysis.md)** (Override & Editor Philosophy)
*   **[YOLOv12](yolov12-analysis.md)** (Stage 2 ML Engine)
*   **[Magi](magi-analysis.md)** (Reading Order Benchmark)
*   **[SAM-Comic](sam-comic-analysis.md)** (Irregular Polygon Fallback)
*   **[DeepPanel](deeppanel-analysis.md)** (Mobile Segmentation)

### Legacy & Heuristic Readers
*   **[BDReader](bdreader-analysis.md)** (Early CV experiments)
*   **[Comic Trim](comic-trim-analysis.md)** (Heuristic limits on mobile)
*   **[Comic Smart Panels](comic-smart-panels-analysis.md)** (Manual authoring patterns)

### Commercial Gold Standards
*   **[Panels App](panels-app-analysis.md)** (UX & smooth transitions)
*   **[Comixology Guided View](comixology-analysis.md)** (Cinematic narrative flow)
*   **[Marvel Smart Panels](marvel-smart-panels-analysis.md)** (Adaptive storytelling)
