"""
Pydantic models for API request/response schemas.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union

from pydantic import BaseModel, Field, validator


class PredictionResponse(BaseModel):
    """Response model for single image prediction."""
    
    prediction: int = Field(..., description="Predicted class (0=Normal, 1=Pneumonia)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence score")
    class_name: str = Field(..., description="Human-readable class name")
    model_version: str = Field(..., description="Version of the model used")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 1,
                "confidence": 0.87,
                "class_name": "Pneumonia",
                "model_version": "v1.0.0",
                "processing_time_ms": 245,
                "timestamp": "2025-07-27T10:30:00Z"
            }
        }


class BatchPredictionItem(BaseModel):
    """Single item in batch prediction response."""
    
    filename: str = Field(..., description="Original filename")
    prediction: int = Field(..., description="Predicted class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    class_name: str = Field(..., description="Human-readable class name")


class BatchPredictionResponse(BaseModel):
    """Response model for batch image prediction."""
    
    predictions: List[BatchPredictionItem] = Field(..., description="List of predictions")
    batch_size: int = Field(..., description="Number of images in batch")
    total_processing_time_ms: int = Field(..., description="Total processing time in milliseconds")
    timestamp: Optional[datetime] = Field(None, description="Batch processing timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "filename": "chest_xray_1.jpg",
                        "prediction": 0,
                        "confidence": 0.92,
                        "class_name": "Normal"
                    },
                    {
                        "filename": "chest_xray_2.jpg", 
                        "prediction": 1,
                        "confidence": 0.85,
                        "class_name": "Pneumonia"
                    }
                ],
                "batch_size": 2,
                "total_processing_time_ms": 450,
                "timestamp": "2025-07-27T10:30:00Z"
            }
        }


class HealthCheck(BaseModel):
    """Individual health check result."""
    
    name: str = Field(..., description="Name of the health check")
    status: str = Field(..., description="Status of the check (healthy/unhealthy)")
    message: Optional[str] = Field(None, description="Additional information")
    duration_ms: Optional[int] = Field(None, description="Check duration in milliseconds")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    checks: List[HealthCheck] = Field(..., description="Individual health checks")
    uptime_seconds: int = Field(..., description="Application uptime in seconds")
    version: str = Field(..., description="Application version")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-07-27T10:30:00Z",
                "checks": [
                    {
                        "name": "model_availability",
                        "status": "healthy",
                        "message": "Model loaded successfully",
                        "duration_ms": 5
                    },
                    {
                        "name": "disk_space",
                        "status": "healthy", 
                        "message": "Sufficient disk space available",
                        "duration_ms": 2
                    }
                ],
                "uptime_seconds": 3600,
                "version": "0.2.0"
            }
        }


class ModelInfoResponse(BaseModel):
    """Response model for model information endpoint."""
    
    model_name: str = Field(..., description="Name of the model")
    model_version: str = Field(..., description="Version of the model")
    model_type: str = Field(..., description="Type of the model (e.g., CNN)")
    input_shape: List[int] = Field(..., description="Expected input shape")
    output_shape: List[int] = Field(..., description="Output shape")
    parameters: int = Field(..., description="Number of model parameters")
    size_mb: float = Field(..., description="Model size in megabytes")
    classes: List[str] = Field(..., description="List of class names")
    loaded_at: datetime = Field(..., description="When the model was loaded")
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "pneumonia_detector_cnn",
                "model_version": "v1.0.0",
                "model_type": "CNN",
                "input_shape": [224, 224, 3],
                "output_shape": [1],
                "parameters": 1235467,
                "size_mb": 4.7,
                "classes": ["Normal", "Pneumonia"],
                "loaded_at": "2025-07-27T08:00:00Z"
            }
        }


class ErrorResponse(BaseModel):
    """Response model for error responses."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: datetime = Field(..., description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "validation_error",
                "message": "Invalid image format",
                "timestamp": "2025-07-27T10:30:00Z",
                "details": {
                    "field": "file",
                    "expected": "image/*",
                    "received": "text/plain"
                }
            }
        }


class MetricsResponse(BaseModel):
    """Response model for application metrics."""
    
    predictions_total: int = Field(..., description="Total number of predictions made")
    predictions_per_minute: float = Field(..., description="Recent predictions per minute")
    average_processing_time_ms: float = Field(..., description="Average processing time")
    error_rate: float = Field(..., description="Error rate percentage")
    uptime_seconds: int = Field(..., description="Application uptime")
    memory_usage_mb: float = Field(..., description="Current memory usage")
    cpu_usage_percent: float = Field(..., description="Current CPU usage")


class GradCAMRequest(BaseModel):
    """Request model for Grad-CAM visualization."""
    
    layer_name: Optional[str] = Field(None, description="Target layer for Grad-CAM")
    class_index: Optional[int] = Field(None, description="Target class index (0 or 1)")
    
    @validator('class_index')
    def validate_class_index(cls, v):
        if v is not None and v not in [0, 1]:
            raise ValueError('class_index must be 0 (Normal) or 1 (Pneumonia)')
        return v


class GradCAMResponse(BaseModel):
    """Response model for Grad-CAM visualization."""
    
    image_data: str = Field(..., description="Base64-encoded Grad-CAM overlay image")
    layer_name: str = Field(..., description="Layer used for visualization")
    class_index: int = Field(..., description="Target class index")
    class_name: str = Field(..., description="Target class name")
    heatmap_intensity: float = Field(..., description="Average heatmap intensity")


class ModelUploadRequest(BaseModel):
    """Request model for model upload/update."""
    
    model_name: str = Field(..., description="Name of the model")
    model_version: str = Field(..., description="Version of the model")
    description: Optional[str] = Field(None, description="Model description")
    tags: Optional[List[str]] = Field(None, description="Model tags")


class PredictionRequest(BaseModel):
    """Request model for prediction when using JSON payload."""
    
    image_data: str = Field(..., description="Base64-encoded image data")
    filename: Optional[str] = Field(None, description="Original filename")
    
    @validator('image_data')
    def validate_image_data(cls, v):
        import base64
        try:
            base64.b64decode(v)
        except Exception:
            raise ValueError('image_data must be valid base64-encoded data')
        return v