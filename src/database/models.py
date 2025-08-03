"""
SQLAlchemy ORM models for the pneumonia detection system.
Implements HIPAA-compliant data models with audit logging and encryption.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Boolean, Text, 
    ForeignKey, Index, JSON, LargeBinary, Enum as SQLAEnum
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

from .connection import Base


class PredictionStatus(enum.Enum):
    """Prediction processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AuditEventType(enum.Enum):
    """Types of audit events for HIPAA compliance."""
    PHI_ACCESS = "phi_access"
    PREDICTION_REQUEST = "prediction_request"
    MODEL_PREDICTION = "model_prediction"
    DATA_EXPORT = "data_export"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    ADMIN_ACTION = "admin_action"
    SECURITY_EVENT = "security_event"
    DATA_MODIFICATION = "data_modification"
    SYSTEM_ERROR = "system_error"


class ModelStatus(enum.Enum):
    """Model deployment status."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class PredictionRecord(Base):
    """
    Records of all predictions made by the system.
    Maintains audit trail for medical compliance.
    """
    __tablename__ = "prediction_records"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Request Information
    request_id = Column(String, unique=True, nullable=False, index=True)
    session_id = Column(String, ForeignKey("user_sessions.id"), nullable=True)
    
    # Image Information
    image_hash = Column(String, nullable=False, index=True)  # SHA256 of image
    image_metadata_id = Column(String, ForeignKey("image_metadata.id"), nullable=True)
    
    # Prediction Results
    prediction = Column(String, nullable=False)  # NORMAL/PNEUMONIA
    confidence = Column(Float, nullable=False)
    class_probabilities = Column(JSON, nullable=False)
    
    # Model Information
    model_version = Column(String, nullable=False)
    model_id = Column(String, ForeignKey("model_versions.id"), nullable=True)
    
    # Performance Metrics
    inference_time_ms = Column(Float, nullable=False)
    processing_time_ms = Column(Float, nullable=False)
    
    # Quality Assessment
    image_quality_score = Column(Float, nullable=True)
    diagnostic_quality = Column(Boolean, default=True)
    
    # Clinical Context
    clinical_recommendation = Column(Text, nullable=True)
    risk_level = Column(String, nullable=True)
    urgency_level = Column(String, nullable=True)
    
    # Status and Timestamps
    status = Column(SQLAEnum(PredictionStatus), default=PredictionStatus.PENDING)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    session = relationship("UserSession", back_populates="predictions")
    image_metadata = relationship("ImageMetadata", back_populates="predictions")
    model = relationship("ModelVersion", back_populates="predictions")
    
    # Indexes for performance
    __table_args__ = (
        Index("idx_prediction_created_at", "created_at"),
        Index("idx_prediction_model_version", "model_version"),
        Index("idx_prediction_status", "status"),
        Index("idx_prediction_confidence", "confidence"),
    )
    
    def __repr__(self):
        return f"<PredictionRecord(id={self.id}, prediction={self.prediction}, confidence={self.confidence:.3f})>"
    
    @hybrid_property
    def is_high_confidence(self):
        """Check if prediction has high confidence."""
        return self.confidence >= 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "request_id": self.request_id,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "class_probabilities": self.class_probabilities,
            "model_version": self.model_version,
            "inference_time_ms": self.inference_time_ms,
            "image_quality_score": self.image_quality_score,
            "clinical_recommendation": self.clinical_recommendation,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
        }


class AuditLog(Base):
    """
    HIPAA-compliant audit logging for all system activities.
    Maintains immutable record of all access and modifications.
    """
    __tablename__ = "audit_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Event Information
    event_type = Column(SQLAEnum(AuditEventType), nullable=False)
    event_description = Column(Text, nullable=False)
    
    # User Information (anonymized for privacy)
    user_id = Column(String, nullable=True)  # Hashed user identifier
    session_id = Column(String, nullable=True)
    client_ip = Column(String, nullable=True)  # Anonymized IP
    user_agent = Column(Text, nullable=True)
    
    # Request Information
    request_id = Column(String, nullable=True)
    endpoint = Column(String, nullable=True)
    method = Column(String, nullable=True)
    
    # Medical Data Access (encrypted)
    phi_accessed = Column(Boolean, default=False)
    data_classification = Column(String, nullable=True)  # PHI, PII, PUBLIC
    
    # System Information
    system_component = Column(String, nullable=False)
    process_id = Column(String, nullable=True)
    
    # Event Details (JSON)
    event_data = Column(JSON, nullable=True)
    
    # Outcome
    success = Column(Boolean, nullable=False)
    error_message = Column(Text, nullable=True)
    
    # Timestamps (immutable)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    
    # Integrity and Security
    checksum = Column(String, nullable=True)  # For tamper detection
    
    # Indexes for audit queries
    __table_args__ = (
        Index("idx_audit_timestamp", "timestamp"),
        Index("idx_audit_event_type", "event_type"),
        Index("idx_audit_user_id", "user_id"),
        Index("idx_audit_phi_accessed", "phi_accessed"),
        Index("idx_audit_success", "success"),
    )
    
    def __repr__(self):
        return f"<AuditLog(id={self.id}, event_type={self.event_type.value}, timestamp={self.timestamp})>"


class ModelVersion(Base):
    """
    Version tracking for ML models with deployment status.
    Maintains model lineage and performance metrics.
    """
    __tablename__ = "model_versions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Model Identification
    model_name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    model_type = Column(String, nullable=False)  # CNN, Transfer Learning, etc.
    
    # File Information
    file_path = Column(String, nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    file_checksum = Column(String, nullable=False)  # SHA256
    
    # Training Information
    training_dataset = Column(String, nullable=True)
    training_date = Column(DateTime, nullable=True)
    training_duration_hours = Column(Float, nullable=True)
    
    # Performance Metrics
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    auc_roc = Column(Float, nullable=True)
    
    # Deployment Information
    status = Column(SQLAEnum(ModelStatus), default=ModelStatus.DEVELOPMENT)
    deployment_date = Column(DateTime, nullable=True)
    
    # Model Configuration
    architecture_config = Column(JSON, nullable=True)
    hyperparameters = Column(JSON, nullable=True)
    
    # Metadata
    description = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    predictions = relationship("PredictionRecord", back_populates="model")
    
    # Indexes
    __table_args__ = (
        Index("idx_model_name_version", "model_name", "version", unique=True),
        Index("idx_model_status", "status"),
        Index("idx_model_created_at", "created_at"),
    )
    
    def __repr__(self):
        return f"<ModelVersion(name={self.model_name}, version={self.version}, status={self.status.value})>"


class UserSession(Base):
    """
    User session tracking for audit compliance.
    Anonymous session management without storing PII.
    """
    __tablename__ = "user_sessions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Session Information
    session_token = Column(String, unique=True, nullable=False, index=True)
    anonymous_user_id = Column(String, nullable=False, index=True)  # Hashed identifier
    
    # Session Metadata
    client_ip_hash = Column(String, nullable=True)  # Hashed IP for privacy
    user_agent_hash = Column(String, nullable=True)  # Hashed for privacy
    
    # Session Lifecycle
    started_at = Column(DateTime, default=func.now(), nullable=False)
    last_activity = Column(DateTime, default=func.now(), onupdate=func.now())
    ended_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Usage Statistics
    prediction_count = Column(Integer, default=0)
    
    # Relationships
    predictions = relationship("PredictionRecord", back_populates="session")
    
    # Indexes
    __table_args__ = (
        Index("idx_session_token", "session_token"),
        Index("idx_session_active", "is_active"),
        Index("idx_session_started_at", "started_at"),
    )
    
    def __repr__(self):
        return f"<UserSession(id={self.id}, started_at={self.started_at})>"


class ImageMetadata(Base):
    """
    Metadata for processed medical images.
    Stores technical information without image content.
    """
    __tablename__ = "image_metadata"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Image Identification
    image_hash = Column(String, unique=True, nullable=False, index=True)
    original_filename = Column(String, nullable=True)  # Sanitized filename
    
    # Image Properties
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    channels = Column(Integer, nullable=False)
    format = Column(String, nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    
    # Quality Metrics
    sharpness_score = Column(Float, nullable=True)
    contrast_score = Column(Float, nullable=True)
    brightness_score = Column(Float, nullable=True)
    quality_score = Column(Float, nullable=True)
    
    # Processing Information
    preprocessing_applied = Column(JSON, nullable=True)
    enhancement_methods = Column(JSON, nullable=True)
    
    # Medical Metadata (optional)
    imaging_modality = Column(String, default="X-RAY")
    body_part = Column(String, default="CHEST")
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    predictions = relationship("PredictionRecord", back_populates="image_metadata")
    
    # Indexes
    __table_args__ = (
        Index("idx_image_hash", "image_hash"),
        Index("idx_image_quality", "quality_score"),
        Index("idx_image_created_at", "created_at"),
    )
    
    def __repr__(self):
        return f"<ImageMetadata(id={self.id}, dimensions={self.width}x{self.height})>"


class SystemConfiguration(Base):
    """
    System configuration and feature flags.
    Allows runtime configuration changes.
    """
    __tablename__ = "system_configurations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Configuration
    key = Column(String, unique=True, nullable=False, index=True)
    value = Column(Text, nullable=False)
    value_type = Column(String, nullable=False)  # string, int, float, bool, json
    
    # Metadata
    description = Column(Text, nullable=True)
    category = Column(String, nullable=True)
    
    # Change Tracking
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    updated_by = Column(String, nullable=True)
    
    def __repr__(self):
        return f"<SystemConfiguration(key={self.key}, value={self.value})>"
    
    @property
    def parsed_value(self):
        """Parse value based on type."""
        if self.value_type == "int":
            return int(self.value)
        elif self.value_type == "float":
            return float(self.value)
        elif self.value_type == "bool":
            return self.value.lower() in ("true", "1", "yes")
        elif self.value_type == "json":
            return json.loads(self.value)
        else:
            return self.value