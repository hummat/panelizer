# AGENTS.md

This file provides guidance to CLI coding agents when working with code in this repository.

## Project Overview

Panel Flow is a pragmatic, human-in-the-loop system for panel-by-panel comic reading from local, DRM-free files (CBZ/PDF). It combines classic computer vision, modern foundation models, and a lightweight cross-platform viewer.

**Current Status:** Early-stage prototyping. Core detection modules under development.

## Architecture Philosophy

The system uses a staged detection pipeline where each technique handles what it does best:

1. **Stage 1 - Classic CV (on-device):** Fast, offline contour detection for "easy" pages
2. **Stage 2 - ML detection (offline/cloud):** YOLO or SAM-based segmentation for complex layouts
3. **Stage 3 - Reading order:** Heuristics (row-major LTR/RTL) with optional VLM inference for ambiguous cases
4. **Stage 4 - Human-in-the-loop:** Override UI for corrections (stored separately, never overwritten)
5. **Stage 5 - PWA Viewer:** Cross-platform panel navigation

Detection results are treated as **proposals**, not ground truth. User overrides are sacred and must never be clobbered by re-running automation.

## Project Structure & Module Organization

Project layout:
- `src/`: core library modules (extraction, detection, ordering, metadata schema)
- `tests/`: automated tests with synthetic/public-domain inputs (never copyrighted comics)
- `docs/`: extended design notes that would bloat the README
- `examples/`: small, reproducible sample configs/outputs (no copyrighted images)

## Key Technical Constraints

### Data Format
All panel metadata uses a JSON schema with:
- Separation of auto-detected data from user `overrides` (overrides NEVER overwritten)
- `book_hash` for cache invalidation
- `source` provenance tracking (cv/yolo/sam/vlm/manual)
- `confidence` scores for automated detections

See README.md lines 218-257 for full schema draft.

### Detection Confidence
- Stage 1 CV must estimate confidence (contour clarity, gutter strength, layout regularity)
- Low-confidence pages trigger Stage 2 ML detection
- No single threshold exists yet - calibration is an open question

### Reading Direction
- Western comics: LTR (left-to-right)
- Manga: RTL (right-to-left)
- Stored in metadata, affects heuristic ordering

### PWA Constraints
- iOS has no File System Access API - must use file input individually
- PWA storage can be evicted under pressure on iOS - cache in IndexedDB
- Comics **never leave device** (no cloud storage)

## Reference Implementations

When implementing stages, refer to these projects:
- **Stage 1 CV:** Kumiko's contour detection approach (https://github.com/njean42/kumiko)
- **Stage 2 ML:** best-comic-panel-detection YOLOv12 model (https://huggingface.co/mosesb/best-comic-panel-detection)
- **Stage 3 Ordering:** Magi's ordering logic (https://github.com/ragavsachdeva/magi)
- **Stage 4 Override UI:** C.A.P.E.'s Electron editor patterns (https://github.com/CodeMinion/C.A.P.E)

## Coding Style & Naming Conventions

- Keep Markdown concise and skimmable; prefer fenced code blocks for formats/commands
- For new code, default to:
  - **Python:** 4-space indentation, type hints where practical, modules in `snake_case.py`
  - **CLI/tools:** commands and files in `kebab-case` where applicable (e.g., `panel-flow`)
- Name data/schema files explicitly (e.g., `schema/panel-metadata.schema.json`)

## Testing Guidelines

- Add tests alongside new behavior; keep them deterministic and fast
- If using Python, prefer `pytest` with names like `tests/test_*.py`
- **Never commit copyrighted comic pages** - use generated images or tiny mocked inputs
- Test fixtures should be synthetic or public-domain only

## Build, Test, and Development Commands

Uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Setup
uv sync

# Format (ruff)
uv run ruff format .

# Lint (ruff + pyright)
uv run ruff check .
uv run pyright

# Test (pytest with coverage, 80% minimum)
uv run pytest
```

Run in order: format → lint → test. All must pass before commit.

## Commit & Pull Request Guidelines

- Commit messages are short and imperative (e.g., "Add initial README with project overview")
- Follow existing patterns visible in git history
- PRs should include: clear description, motivation, and any spec changes to `README.md`
- For UI/viewer work, include screenshots/GIFs and note input device coverage (mouse/touch)
- **Keep docs in sync:** Update `AGENTS.md` and `README.md` when adding/changing tooling, architecture, or project structure

## Security & Configuration Tips

- Never commit secrets (API keys, tokens). Use `.env` locally and keep it untracked
- Avoid logging file paths or content that could leak personal comic libraries; sanitize examples
- No secrets (API keys, tokens) in code - use `.env` locally

## Key Design Decisions

### Why PWA instead of native apps?
- No iOS/Android native development required
- Installable via "Add to Home Screen"
- File API sufficient for "open file" workflow
- Works across desktop, Android, iOS

### Why separate overrides?
- Detection automation is never perfect
- User corrections must persist across re-processing
- Overrides can be exported/shared without sharing copyrighted images

### Why confidence scoring?
- Determines when to fall back from CV to ML
- Helps users identify pages needing review
- Enables cost control for VLM/API-based ordering

## Open Questions (as of current design)

1. How to reliably calibrate CV confidence thresholds?
2. Are bboxes sufficient, or do irregular panels need polygons (SAM)?
3. Can Magi's ordering component be extracted standalone?
4. What format for sharing override corrections without copyrighted images?
5. Is C.A.P.E. `.cpanel` format compatibility worthwhile?

## Non-Goals

Do NOT implement or suggest:
- Cloud-hosted comic storage
- DRM circumvention
- Real-time on-device deep learning (batch processing is expected)
- Native mobile apps (PWA is intentional choice)
- Perfect automation without human input
- Replacing commercial readers like Comixology

## Documentation Standards

- Keep Markdown concise and skimmable
- Use fenced code blocks for formats/commands
- README.md is the architectural source of truth
- AGENTS.md (this file) contains contribution/structure guidelines and Claude Code guidance
