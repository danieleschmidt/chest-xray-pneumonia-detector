"""Core package for training a pneumonia detection model."""

from importlib.metadata import PackageNotFoundError, version

from .pipeline import TrainingConfig, run_training

try:  # pragma: no cover - metadata absent during local development
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover - fallback if not installed
    __version__ = "0.0.0"

__all__ = ["TrainingConfig", "run_training", "__version__"]
