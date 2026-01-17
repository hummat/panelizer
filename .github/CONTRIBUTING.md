# Contributing to panelizer

Thanks for your interest in contributing! This document covers development setup and guidelines.

## Development Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Quick Start

```bash
# Clone the repository
git clone https://github.com/hummat/panelizer.git
cd panelizer

# Install development dependencies
uv sync --group dev

# For ML features (optional)
uv sync --group ml      # GPU support
uv sync --group ml-cpu  # CPU only

# Run all checks
uv run ruff format .
uv run ruff check . --fix
uv run pyright
uv run pytest
```

## Code Style

- Python 3.11+
- 120-character line limit (see `[tool.ruff]` in pyproject.toml)
- Type hints where practical
- Run checks in order: `ruff format` → `ruff check --fix` → `pyright` → `pytest`

## Architecture

Before making changes, read:
- `AGENTS.md` — project overview and architecture philosophy
- `docs/architecture-decisions.md` — detailed design decisions

### Key Principles

- Detection results are **proposals**, not ground truth
- User overrides are sacred and must never be clobbered
- Never commit copyrighted comic pages — use synthetic/public-domain test inputs

## Pull Request Process

1. **Create an issue first** for non-trivial changes
2. **Fork and branch** from `main`
3. **Make your changes** following the style guide
4. **Run all checks** — all must pass before commit
5. **Update documentation** if needed (README.md, AGENTS.md)
6. **Submit PR** using the template

### Commit Messages

- Use present tense: "Add feature" not "Added feature"
- Keep the first line under 72 characters
- Reference issues: "Fix panel detection (#42)"

## Testing

- Framework: pytest with 79% coverage minimum
- Never commit copyrighted images — use generated/mocked inputs
- Test fixtures should be synthetic or public-domain only

## Questions?

- Open a [Discussion](https://github.com/hummat/panelizer/discussions) for questions
- Check existing [Issues](https://github.com/hummat/panelizer/issues) for known problems
