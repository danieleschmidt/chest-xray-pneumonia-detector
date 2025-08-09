"""
Pydantic schemas for quantum-inspired task planner API request/response models.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, validator


class TaskCreateRequest(BaseModel):
    """Request schema for creating a new task."""
    name: str = Field(..., description="Task name", min_length=1, max_length=200)
    description: Optional[str] = Field(None, description="Detailed task description")
    priority: str = Field("medium", description="Task priority level")
    dependencies: Optional[List[str]] = Field(None, description="List of task dependency IDs")
    estimated_duration_minutes: Optional[int] = Field(60, description="Estimated task duration in minutes", ge=1)
    resource_requirements: Optional[Dict[str, float]] = Field(None, description="Resource requirements by type")


class TaskResponse(BaseModel):
    """Response schema for task information."""
    id: str = Field(..., description="Unique task identifier")
    name: str = Field(..., description="Task name")
    description: str = Field(..., description="Task description")
    priority: str = Field(..., description="Task priority level")
    status: str = Field(..., description="Current task status")
    dependencies: List[str] = Field(..., description="Task dependency IDs")
    estimated_duration_minutes: int = Field(..., description="Estimated duration in minutes")
    created_at: datetime = Field(..., description="Task creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Task start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Task completion timestamp")


class ScheduleOptimizeRequest(BaseModel):
    """Request schema for schedule optimization."""
    algorithm: str = Field("annealing", description="Optimization algorithm to use")
    max_iterations: Optional[int] = Field(1000, description="Maximum optimization iterations", ge=10, le=10000)
    temperature: Optional[float] = Field(100.0, description="Initial annealing temperature", gt=0.0)
    cooling_rate: Optional[float] = Field(0.95, description="Annealing cooling rate", gt=0.0, lt=1.0)


class ScheduleResponse(BaseModel):
    """Response schema for schedule optimization results."""
    optimal_schedule: List[str] = Field(..., description="Optimal task execution order")
    energy: float = Field(..., description="Final optimization energy/cost")
    iterations: int = Field(..., description="Number of optimization iterations performed")
    convergence_achieved: bool = Field(..., description="Whether optimization converged")
    execution_time: float = Field(..., description="Optimization execution time in seconds")


class ResourceAddRequest(BaseModel):
    """Request schema for adding a new resource."""
    resource_id: str = Field(..., description="Unique resource identifier", min_length=1)
    type: str = Field(..., description="Resource type")
    capacity: float = Field(..., description="Total resource capacity", gt=0.0)


class ResourceUtilizationResponse(BaseModel):
    """Response schema for resource utilization information."""
    utilization: Dict[str, Dict[str, Any]] = Field(..., description="Resource utilization data by resource ID")


class PredictionRequest(BaseModel):
    """Request schema for pneumonia detection predictions."""
    image_data: str = Field(..., description="Base64 encoded image data")
    include_grad_cam: bool = Field(False, description="Include Grad-CAM visualization")


class PredictionResponse(BaseModel):
    """Response schema for pneumonia detection predictions."""
    prediction: str = Field(..., description="Predicted class (NORMAL or PNEUMONIA)")
    confidence: float = Field(..., description="Prediction confidence score", ge=0.0, le=1.0)
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_version: str = Field(..., description="Model version used for prediction")


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: List[PredictionResponse] = Field(..., description="List of individual predictions")
    batch_size: int = Field(..., description="Number of images processed")
    total_processing_time_ms: float = Field(..., description="Total batch processing time")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    failed_predictions: int = Field(..., description="Number of failed predictions")


class HealthResponse(BaseModel):
    """Response schema for health check."""
    healthy: bool = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    components: Dict[str, Any] = Field(..., description="Component health status")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ModelInfoResponse(BaseModel):
    """Response schema for model information."""
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    model_type: str = Field(..., description="Model architecture type")
    training_date: Optional[datetime] = Field(None, description="Model training date")
    model_size_mb: float = Field(..., description="Model size in MB")
    input_shape: List[int] = Field(..., description="Expected input shape")
    output_classes: List[str] = Field(..., description="Output class names")
    performance_metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    deployment_info: Dict[str, Any] = Field(..., description="Deployment information")


class ErrorResponse(BaseModel):
    """Response schema for API errors."""
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: datetime = Field(..., description="Error timestamp")
    path: str = Field(..., description="Request path")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class MetricsResponse(BaseModel):
    """Response schema for metrics endpoint."""
    total_predictions: int = Field(..., description="Total number of predictions made")
    predictions_per_minute: float = Field(..., description="Recent predictions per minute")
    average_processing_time_ms: float = Field(..., description="Average processing time")
    error_rate: float = Field(..., description="Error rate percentage")
    model_versions: Dict[str, int] = Field(..., description="Predictions by model version")
    uptime_seconds: float = Field(..., description="Service uptime")


class ValidationError(BaseModel):
    """Schema for validation errors."""
    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Validation error message")
    value: Any = Field(..., description="Invalid value provided")


class ImageValidationResponse(BaseModel):
    """Response schema for image validation."""
    valid: bool = Field(..., description="Whether the image is valid")
    format: str = Field(..., description="Image format")
    dimensions: tuple = Field(..., description="Image dimensions (width, height)")
    size_bytes: int = Field(..., description="Image size in bytes")
    issues: List[str] = Field(default_factory=list, description="Validation issues found")


class PredictionHistory(BaseModel):
    """Schema for prediction history tracking."""
    prediction_id: str = Field(..., description="Unique prediction identifier")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    prediction: str = Field(..., description="Prediction result")
    confidence: float = Field(..., description="Prediction confidence")
    model_version: str = Field(..., description="Model version used")
    processing_time_ms: float = Field(..., description="Processing time")
    image_hash: str = Field(..., description="SHA256 hash of input image")
    user_agent: Optional[str] = Field(None, description="Client user agent")
    ip_address: Optional[str] = Field(None, description="Client IP address")


# Validators
@validator('confidence', 'probabilities')
def validate_confidence_range(cls, v):
    """Validate confidence scores are within valid range."""
    if isinstance(v, dict):
        for key, value in v.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f'Probability for {key} must be between 0.0 and 1.0')
    elif isinstance(v, float):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
    return v