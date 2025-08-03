"""
Database layer for the pneumonia detection system.
Provides data persistence, caching, and audit logging capabilities.
"""

from .connection import DatabaseManager, get_database
from .models import (
    Base,
    PredictionRecord,
    AuditLog,
    ModelVersion,
    UserSession,
    ImageMetadata
)
from .repositories import (
    PredictionRepository,
    AuditRepository,
    ModelVersionRepository,
    UserRepository
)
from .cache import CacheManager
from .migrations import MigrationManager

__all__ = [
    "DatabaseManager",
    "get_database",
    "Base",
    "PredictionRecord", 
    "AuditLog",
    "ModelVersion",
    "UserSession",
    "ImageMetadata",
    "PredictionRepository",
    "AuditRepository", 
    "ModelVersionRepository",
    "UserRepository",
    "CacheManager",
    "MigrationManager"
]