"""
Service layer for the pneumonia detection system.
Provides business logic abstraction for the API layer.
"""

from .prediction_service import PredictionService, PredictionRequest, PredictionResult, DiagnosticReportGenerator

__all__ = [
    "PredictionService",
    "PredictionRequest", 
    "PredictionResult",
    "DiagnosticReportGenerator"
]