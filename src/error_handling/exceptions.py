"""Custom exceptions for the pneumonia detection system."""

from typing import Optional, Dict, Any


class PneumoniaDetectorError(Exception):
    """Base exception for pneumonia detector errors."""
    
    def __init__(self, message: str, error_code: str = "UNKNOWN_ERROR", details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ModelError(PneumoniaDetectorError):
    """Errors related to model loading or inference."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "MODEL_ERROR", details)


class ValidationError(PneumoniaDetectorError):
    """Errors related to input validation."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "VALIDATION_ERROR", details)


class SecurityError(PneumoniaDetectorError):
    """Errors related to security violations."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "SECURITY_ERROR", details)


class ResourceError(PneumoniaDetectorError):
    """Errors related to resource constraints."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "RESOURCE_ERROR", details)


class ConfigurationError(PneumoniaDetectorError):
    """Errors related to system configuration."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "CONFIGURATION_ERROR", details)


class TrainingError(PneumoniaDetectorError):
    """Errors related to model training."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "TRAINING_ERROR", details)


class QuantumSchedulerError(PneumoniaDetectorError):
    """Errors related to quantum task scheduling."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "QUANTUM_SCHEDULER_ERROR", details)