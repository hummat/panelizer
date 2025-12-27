# Panelizer

A pragmatic, human-in-the-loop system for **panel-by-panel comic reading** from **local, DRM-free files** (CBZ/PDF), combining classic computer vision, modern foundation models, and a lightweight cross-platform viewer.

> Motivation: there is currently **no good open solution** for Guided-View-style reading of local comics. Existing readers rely on heuristics; research code stops at detection; commercial solutions are locked ecosystems.

This project aims to close that gap.

---

## Prior art & Analysis

We have conducted deep technical analyses of existing tools to inform Panelizer's architecture.

### Detection tools (no integrated reader)

| Tool | Method | Analysis | Notes |
|------|--------|----------|-------|
| **[Kumiko](https://github.com/njean42/kumiko)** | OpenCV contours | [Analysis](docs/kumiko-analysis.md) | Active. Primary reference for our Stage 1 CV. |
| **[C.A.P.E.](https://github.com/CodeMinion/C.A.P.E)** | OpenCV + Editor | [Analysis](docs/cape-analysis.md) | **Closest vision.** pioneered the "User is Sacred" override philosophy. |
| **[DeepPanel](https://github.com/pedrovgs/DeepPanel)** | CNN (TFLite) | [Analysis](docs/deeppanel-analysis.md) | Mobile-optimized. Good reference for native apps. |
| **[best-comic-panel-detection](https://huggingface.co/mosesb/best-comic-panel-detection)** | YOLOv12 | [Analysis](docs/yolov12-analysis.md) | mAP ~99%. Our choice for Stage 2 ML fallback. |
| **[segment-anything-comic](https://github.com/Vrroom/segment-anything-comic)** | SAM fine-tuned | [Analysis](docs/sam-comic-analysis.md) | Handles irregular polygons. Research-grade. |
| **[Magi](https://github.com/ragavsachdeva/magi)** | Deep learning | [Analysis](docs/magi-analysis.md) | **Only tool that does reading order.** Manga-focused. |

### Readers with panel detection

| Tool | Detection | Analysis | Status |
|------|-----------|----------|--------|
| **[BDReader](https://sourceforge.net/projects/bdreader/)** | Heuristics | [Analysis](docs/bdreader-analysis.md) | Abandoned. Shows limits of pure heuristics. |
| **[Comic Smart Panels](https://github.com/zoran123456/Comic-Smart-Panels)** | Manual | [Analysis](docs/comic-smart-panels-analysis.md) | Good "Creator" UI patterns. |
| **[Panels app](https://panels.app/)** | ML (Proprietary) | [Analysis](docs/panels-app-analysis.md) | **Sunsetting / Open Source** (Jan 2026). The UX bar to beat. |
| **[Comic Trim](https://play.google.com/store/apps/details?id=com.comic.trim)** | Heuristics | [Analysis](docs/comic-trim-analysis.md) | Android-only. Fails on complex layouts. |

### Commercial Ecosystem

- **[Comixology Guided View](docs/comixology-analysis.md)** — The industry standard for "cinematic" reading.
- **[Marvel Smart Panels](docs/marvel-smart-panels-analysis.md)** — Narrative-focused adaptive zooming.

### Gap analysis

| Stage | Kumiko | C.A.P.E. | Magi | Panels app | **Panelizer** |
|-------|--------|----------|------|------------|---------------|
| CV detection | ✓ | ✓ | — | — | ✓ |
| ML fallback | — | — | ✓ | ✓ | ✓ |
| Reading order | Heuristic | Heuristic | ✓ | ? | ✓ (VLM) |
| Human editing | Planned | ✓ | — | — | ✓ |
| Viewer | Basic | — | — | ✓ | ✓ (PWA) |
| Cross-platform | Desktop | Desktop | Desktop | Mobile | Web/Mobile |

See **[Architecture Decisions](docs/architecture-decisions.md)** for how these analyses shaped our technical strategy.

---

## High-level idea

Split the problem into what each technique is actually good at:

1. **Classic CV (on device / offline)** — Fast, cheap, interpretable. Handles most "easy" pages.
2. **Deep learning (offline or cloud)** — Robust panel segmentation for irregular layouts.
3. **Vision-capable LLMs (cloud, optional)** — Infer **reading order** using semantic + visual context.
4. **Human-in-the-loop overrides** — Required for correctness and UX.
5. **Cross-platform PWA viewer** — Reads cached metadata; no native code required.

Detection is treated as a **proposal**, not ground truth.

---

## Architecture overview

```
CBZ / PDF
   │
   ▼
[ Page extraction ]
   │
   ├─▶ Classic CV (reference: Kumiko)
   │      ├─ confidence ≥ threshold → order heuristically → panels.json
   │      └─ confidence < threshold ↓
   │
   └─▶ ML detection (YOLO or SAM-comic)
          └─ panel bboxes/polygons
                 │
                 ├─▶ Heuristic ordering (row-major LTR/RTL)
                 │      └─ if unambiguous → panels.json
                 │
                 └─▶ VLM ordering (optional, for complex layouts)
                        └─ ordered panels → panels.json
                               │
                               ▼
                      PWA Viewer (panel-by-panel)
                               │
                               ▼
                      Human overrides → panels.json (updated)
```

All results are cached. Models are run **once per book**, not per read.

---

## Stage 1 — Classic CV panel detection (on device)

**Goal:** cover the majority of pages without ML.

Pipeline (per page image):

1. Grayscale → Gaussian blur
2. Adaptive threshold (handles varying page backgrounds)
3. Contour detection → filter by area, aspect ratio
4. Gutter analysis (whitespace between panels)
5. Recursive XY-cut or connected components
6. Bounding box extraction
7. Heuristic ordering (row-major LTR, or RTL for manga)
8. **Confidence estimation** (contour clarity, gutter strength, layout regularity)

Reference implementation: Kumiko's approach, extended with confidence scoring.

**Pros:** Fast, offline, explainable, no API costs.

**Cons:** Fails on borderless panels, irregular layouts, art bleeding into gutters.

**Output:** panels with confidence scores; low-confidence pages flagged for Stage 2.

---

## Stage 2 — ML panel detection (offline / cloud)

**Goal:** robust geometry for pages where CV fails.

Candidate models (ready to use):

| Model | Output | Speed | Notes |
|-------|--------|-------|-------|
| **[best-comic-panel-detection](https://huggingface.co/mosesb/best-comic-panel-detection)** | Bboxes | Fast | YOLOv12, mAP 99%, easiest integration |
| **[segment-anything-comic](https://github.com/Vrroom/segment-anything-comic)** | Polygons | Slow | Handles irregular panels |
| **[DeepPanel](https://github.com/pedrovgs/DeepPanel)** | Bboxes | ~400ms | Best for mobile/native if needed |

**Recommendation:** Start with YOLO model. Add SAM-comic for irregular layouts if needed.

Trigger: CV confidence below threshold, or user-flagged pages.

Output: panel bboxes (or polygons), **unordered**.

---

## Stage 3 — Reading order

**Goal:** determine panel sequence.

### 3a. Heuristic ordering (default)

- Row-major: top-to-bottom, then LTR (western) or RTL (manga)
- Works for 80%+ of pages
- Free, instant, deterministic

### 3b. Learned ordering (for complex layouts)

Options:

1. **Adapt Magi's ordering logic** — extract and reuse their ordering component
2. **VLM inference** — send annotated image to GPT-4V/Claude, ask for order
3. **Train lightweight classifier** — predict layout type, apply corresponding rule

VLM approach:
- Input: page image with labeled panel bboxes
- Prompt: "Given panels A-F, provide reading order for [LTR/RTL] comic. Note any ambiguity."
- Output: ordered list + confidence

**Recommendation:** Start with heuristics. Add VLM only for flagged pages (cost control).

---

## Stage 4 — Human-in-the-loop overrides (essential)

No automation is perfect. This is what turns "90% correct" into "100% usable".

Reference: C.A.P.E.'s Electron editor (study the UI patterns).

The viewer must support:
- Drag-to-reorder panels
- Adjust panel boundaries (resize/move)
- Merge panels (e.g., splash pages detected as multiple)
- Split panels (single detection covering two)
- Mark as "full page" (skip panel-by-panel)
- Flag page for re-processing

Override behavior:
- Stored separately from auto-detected data
- **Never overwritten** by re-running detection
- Exportable (for sharing corrections without sharing images)

---

## Stage 5 — Cross-platform viewer (PWA)

**Why PWA:**
- No native Android/iOS development
- Installable ("Add to Home Screen")
- Works on desktop, Android, iOS
- File API sufficient for "open file" workflow

Core features:
- Open local CBZ/PDF (via `<input type="file">`)
- Load/display `panels.json`
- Panel-by-panel navigation (tap/swipe/keys)
- Smooth zoom transitions
- Reading progress persistence (IndexedDB)
- Override editing UI

Backend (optional):
- Static hosting or FastHTML for app shell
- **Comics never leave device**

iOS notes:
- No File System Access API (can't "open folder")
- PWA storage can be evicted under pressure
- Workaround: select files individually, cache in IndexedDB

---

## Data format (draft)

```json
{
  "version": 1,
  "book_hash": "sha256:...",
  "pages": [
    {
      "index": 0,
      "size": [1800, 2700],
      "panels": [
        {"id": "p0-A", "bbox": [100, 50, 800, 600], "confidence": 0.95},
        {"id": "p0-B", "bbox": [900, 50, 800, 600], "confidence": 0.91}
      ],
      "order": ["p0-A", "p0-B"],
      "order_confidence": 0.88,
      "source": "cv",
      "user_override": false
    }
  ],
  "overrides": {
    "p0-A": {"bbox": [110, 55, 790, 590]},
    "page_3": {"order": ["p3-B", "p3-A", "p3-C"]}
  },
  "metadata": {
    "reading_direction": "ltr",
    "created": "2025-01-15T10:30:00+00:00",
    "tool_version": "0.1.0"
  }
}
```

Design notes:
- `overrides` separate from auto-detected data (never clobbered)
- `book_hash` for cache invalidation if source changes
- `source` tracks provenance (cv / yolo / sam / vlm / manual)
- Compatible with C.A.P.E.'s `.cpanel` format where possible

---

## Non-goals

- Replacing Comixology / Kindle / commercial readers
- Perfect automation without human input
- Real-time on-device deep learning (batch processing is fine)
- DRM circumvention
- Cloud-hosted comic storage
- Native mobile apps (PWA is sufficient)

---

## Open questions

1. **Confidence calibration** — how to reliably estimate when CV is "good enough"?
2. **YOLO vs SAM** — is bbox sufficient, or do we need polygons for irregular panels?
3. **Magi integration** — can we extract just the ordering component?
4. **Override sharing** — format for sharing corrections without copyrighted images?
5. **C.A.P.E. compatibility** — worth maintaining `.cpanel` format compatibility?

---

## Roadmap

### Phase 1: Core detection (2 weekends)
- [ ] CV detector with confidence scoring (reference: Kumiko)
- [ ] YOLO model integration (HuggingFace)
- [ ] JSON schema + validation
- [ ] CLI for batch processing CBZ

### Phase 2: Viewer MVP (2-3 weekends)
- [ ] PWA shell
- [ ] Panel-by-panel navigation
- [ ] Basic touch/swipe/keyboard controls
- [ ] Reading progress persistence

### Phase 3: Human editing (2 weekends)
- [ ] Override UI (reference: C.A.P.E.)
- [ ] Drag-to-reorder, resize, merge/split
- [ ] Override persistence

### Phase 4: Polish + ordering (ongoing)
- [ ] Mobile gesture refinement
- [ ] VLM ordering for flagged pages
- [ ] SAM-comic for irregular panels
- [ ] Override export/import

---

## Status

🚧 **Early design / prototyping**

Current focus:
- Finalizing architecture
- CV detector prototype
- Evaluating YOLO model

---

## Development

```bash
uv sync                  # install dependencies
uv run ruff format .     # format
uv run ruff check .      # lint
uv run pyright           # type check
uv run pytest            # test (80% coverage required)
```

See `AGENTS.md` for contribution guidelines.

---

## References

### Detection
- [Kumiko](https://github.com/njean42/kumiko) — CV panel extraction
- [C.A.P.E.](https://github.com/CodeMinion/C.A.P.E) — CV + editor (study the UI)
- [best-comic-panel-detection](https://huggingface.co/mosesb/best-comic-panel-detection) — YOLOv12
- [segment-anything-comic](https://github.com/Vrroom/segment-anything-comic) — SAM fine-tuned
- [DeepPanel](https://github.com/pedrovgs/DeepPanel) — Mobile CNN

### Ordering
- [Magi](https://github.com/ragavsachdeva/magi) — Panel ordering + OCR

### Research
- [Max Halford's tutorial](https://maxhalford.github.io/blog/comic-book-panel-segmentation/) — scikit-image approach
- [Manga109 dataset](http://www.manga109.org/)
- [CoMix benchmark](https://arxiv.org/abs/2407.03550) — multi-task comic understanding
- [Comics Understanding survey](https://arxiv.org/abs/2409.09502) — comprehensive 2024 overview

### Commercial (study the UX)
- [Panels app](https://panels.app/) — ML-based guided view
- Comixology Guided View — manual curation benchmark
