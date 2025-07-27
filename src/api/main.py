"""
FastAPI application for Chest X-Ray Pneumonia Detector API.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .models import (
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse
)
from .dependencies import get_model_service, get_logger
from .middleware import MetricsMiddleware, SecurityMiddleware, LoggingMiddleware
from ..monitoring.metrics import metrics_registry
from ..monitoring.health import HealthChecker


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger = logging.getLogger(__name__)
    logger.info("Starting Chest X-Ray Pneumonia Detector API")
    
    # Initialize health checker
    health_checker = HealthChecker()
    app.state.health_checker = health_checker
    
    # Initialize metrics
    metrics_registry.init_metrics()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Chest X-Ray Pneumonia Detector API")


# Create FastAPI application
app = FastAPI(
    title="Chest X-Ray Pneumonia Detector API",
    description="AI-powered pneumonia detection from chest X-ray images",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

app.add_middleware(LoggingMiddleware)
app.add_middleware(MetricsMiddleware)
app.add_middleware(SecurityMiddleware)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get(
    "/",
    response_model=Dict[str, str],
    summary="API Information",
    description="Get basic information about the API"
)
async def root():
    """Root endpoint providing API information."""
    return {
        "name": "Chest X-Ray Pneumonia Detector API",
        "version": "0.2.0",
        "description": "AI-powered pneumonia detection from chest X-ray images",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the health status of the API and its dependencies"
)
async def health_check(request: Request):
    """Health check endpoint."""
    health_checker = request.app.state.health_checker
    health_status = await health_checker.check_health()
    
    status_code = status.HTTP_200_OK if health_status.status == "healthy" else status.HTTP_503_SERVICE_UNAVAILABLE
    
    return JSONResponse(
        status_code=status_code,
        content=health_status.dict()
    )


@app.get(
    "/health/ready",
    response_model=Dict[str, str],
    summary="Readiness Check",
    description="Check if the API is ready to serve requests"
)
async def readiness_check(request: Request):
    """Readiness check endpoint for Kubernetes."""
    health_checker = request.app.state.health_checker
    is_ready = await health_checker.check_readiness()
    
    if is_ready:
        return {"status": "ready"}
    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )


@app.get(
    "/health/live",
    response_model=Dict[str, str],
    summary="Liveness Check",
    description="Check if the API is alive"
)
async def liveness_check():
    """Liveness check endpoint for Kubernetes."""
    return {"status": "alive"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Single Image Prediction",
    description="Predict pneumonia from a single chest X-ray image"
)
async def predict_single(
    file: UploadFile = File(..., description="Chest X-ray image file"),
    model_service=Depends(get_model_service),
    logger=Depends(get_logger)
):
    """Predict pneumonia from a single chest X-ray image."""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="File must be an image"
            )
        
        # Read image data
        image_data = await file.read()
        
        # Make prediction
        start_time = time.time()
        prediction = await model_service.predict_single(image_data)
        processing_time = int((time.time() - start_time) * 1000)  # ms
        
        # Update metrics
        metrics_registry.prediction_counter.inc()
        metrics_registry.prediction_duration.observe(processing_time / 1000)
        
        logger.info(
            "Single prediction completed",
            extra={
                "filename": file.filename,
                "prediction": prediction.prediction,
                "confidence": prediction.confidence,
                "processing_time_ms": processing_time
            }
        )
        
        return PredictionResponse(
            prediction=prediction.prediction,
            confidence=prediction.confidence,
            class_name=prediction.class_name,
            model_version=prediction.model_version,
            processing_time_ms=processing_time,
            timestamp=prediction.timestamp
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        metrics_registry.error_counter.labels(error_type="prediction_error").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed"
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch Image Prediction",
    description="Predict pneumonia from multiple chest X-ray images"
)
async def predict_batch(
    files: List[UploadFile] = File(..., description="Multiple chest X-ray image files"),
    model_service=Depends(get_model_service),
    logger=Depends(get_logger)
):
    """Predict pneumonia from multiple chest X-ray images."""
    try:
        if len(files) > 10:  # Limit batch size
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Maximum 10 files allowed per batch"
            )
        
        # Validate all files
        for file in files:
            if not file.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"File {file.filename} must be an image"
                )
        
        # Read all image data
        image_data_list = []
        filenames = []
        for file in files:
            image_data = await file.read()
            image_data_list.append(image_data)
            filenames.append(file.filename)
        
        # Make batch prediction
        start_time = time.time()
        predictions = await model_service.predict_batch(image_data_list)
        processing_time = int((time.time() - start_time) * 1000)  # ms
        
        # Combine predictions with filenames
        prediction_results = []
        for filename, prediction in zip(filenames, predictions):
            prediction_results.append({
                "filename": filename,
                "prediction": prediction.prediction,
                "confidence": prediction.confidence,
                "class_name": prediction.class_name
            })
        
        # Update metrics
        metrics_registry.batch_prediction_counter.inc()
        metrics_registry.batch_size_histogram.observe(len(files))
        metrics_registry.prediction_duration.observe(processing_time / 1000)
        
        logger.info(
            "Batch prediction completed",
            extra={
                "batch_size": len(files),
                "processing_time_ms": processing_time
            }
        )
        
        return BatchPredictionResponse(
            predictions=prediction_results,
            batch_size=len(files),
            total_processing_time_ms=processing_time,
            timestamp=predictions[0].timestamp if predictions else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}", exc_info=True)
        metrics_registry.error_counter.labels(error_type="batch_prediction_error").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch prediction failed"
        )


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Model Information",
    description="Get information about the current model"
)
async def get_model_info(
    model_service=Depends(get_model_service),
    logger=Depends(get_logger)
):
    """Get information about the current model."""
    try:
        model_info = await model_service.get_model_info()
        return model_info
        
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get model information"
        )


@app.post(
    "/model/reload",
    response_model=Dict[str, str],
    summary="Reload Model",
    description="Reload the model from disk"
)
async def reload_model(
    model_service=Depends(get_model_service),
    logger=Depends(get_logger)
):
    """Reload the model from disk."""
    try:
        await model_service.reload_model()
        logger.info("Model reloaded successfully")
        
        return {
            "message": "Model reloaded successfully",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        
    except Exception as e:
        logger.error(f"Failed to reload model: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reload model"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger = logging.getLogger(__name__)
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    metrics_registry.error_counter.labels(error_type="unhandled_exception").inc()
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            message="An unexpected error occurred",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ).dict()
    )


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )