"""ML module exceptions."""


class MLError(Exception):
    """Base exception for ML module errors."""

    pass


class MLDependencyError(MLError):
    """Raised when ML dependencies are not installed."""

    pass


class ModelNotFoundError(MLError):
    """Raised when model weights cannot be found or downloaded."""

    pass
