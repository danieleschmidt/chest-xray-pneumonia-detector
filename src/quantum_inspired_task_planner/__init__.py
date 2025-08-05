"""Quantum-Inspired Task Planner.

A sophisticated task planning system using quantum computing principles
for optimal scheduling and resource allocation.
"""

from importlib.metadata import PackageNotFoundError, version
from .quantum_scheduler import QuantumScheduler, QuantumTask, TaskPriority, TaskStatus

try:  # pragma: no cover - metadata absent during local development
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover - fallback if not installed
    __version__ = "0.1.0"

__all__ = ["QuantumScheduler", "QuantumTask", "TaskPriority", "TaskStatus", "__version__"]
