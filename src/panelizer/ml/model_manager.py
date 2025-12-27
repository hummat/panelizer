"""Model download and cache management."""

import os
from pathlib import Path
from typing import Optional

from .exceptions import MLDependencyError, ModelNotFoundError

# HuggingFace model info
YOLO_MODEL_REPO = "mosesb/best-comic-panel-detection"
YOLO_MODEL_FILE = "best.pt"

# Default cache location
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "panelizer" / "models"


def get_cache_dir() -> Path:
    """Get model cache directory, respecting XDG_CACHE_HOME.

    Returns:
        Path to the cache directory for storing model weights
    """
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache) / "panelizer" / "models"
    return DEFAULT_CACHE_DIR


def get_yolo_model_path(cache_dir: Optional[Path] = None) -> str:
    """Get path to YOLO model, downloading from HuggingFace if necessary.

    Args:
        cache_dir: Override default cache directory

    Returns:
        Path to the model weights file

    Raises:
        MLDependencyError: If huggingface_hub is not installed
        ModelNotFoundError: If model download fails
    """
    cache = cache_dir or get_cache_dir()
    model_path = cache / YOLO_MODEL_FILE

    if model_path.exists():
        return str(model_path)

    # Download from HuggingFace
    return _download_from_huggingface(cache)


def _download_from_huggingface(cache_dir: Path) -> str:
    """Download model from HuggingFace Hub.

    Args:
        cache_dir: Directory to store the downloaded model

    Returns:
        Path to the downloaded model file

    Raises:
        MLDependencyError: If huggingface_hub is not installed
        ModelNotFoundError: If download fails
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise MLDependencyError("huggingface_hub is not installed. Install with: uv sync --extra ml") from e

    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        model_path = hf_hub_download(
            repo_id=YOLO_MODEL_REPO,
            filename=YOLO_MODEL_FILE,
            local_dir=cache_dir,
        )
        return model_path
    except Exception as e:
        raise ModelNotFoundError(f"Failed to download model from {YOLO_MODEL_REPO}: {e}") from e


def clear_model_cache(cache_dir: Optional[Path] = None) -> None:
    """Remove cached models.

    Args:
        cache_dir: Override default cache directory
    """
    import shutil

    cache = cache_dir or get_cache_dir()
    if cache.exists():
        shutil.rmtree(cache)
