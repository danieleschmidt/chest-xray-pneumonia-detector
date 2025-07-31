"""
Production-ready FastAPI application for chest X-ray pneumonia detection.
Implements enterprise-grade patterns for healthcare ML inference.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest

from ..monitoring.metrics import ModelMetrics
from ..monitoring.health_checks import HealthChecker
from .models import PneumoniaDetectionModel
from .schemas import PredictionRequest, PredictionResponse, HealthResponse
from .middleware import SecurityMiddleware, RateLimitMiddleware
from .utils import validate_image, preprocess_image

# Initialize metrics
model_metrics = ModelMetrics()
health_checker = HealthChecker()

# Prometheus metrics
PREDICTION_COUNTER = Counter('predictions_total', 'Total predictions made', ['model_version', 'status'])
PREDICTION_DURATION = Histogram('prediction_duration_seconds', 'Time spent on predictions')
REQUEST_COUNTER = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])

logger = logging.getLogger(__name__)

# Global model instance
model_instance: Optional[PneumoniaDetectionModel] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown."""
    # Startup
    logger.info("Starting up Pneumonia Detection API")
    global model_instance
    model_instance = PneumoniaDetectionModel()
    await model_instance.load_model()
    logger.info("Model loaded successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Pneumonia Detection API")
    if model_instance:
        await model_instance.cleanup()

app = FastAPI(
    title="Chest X-Ray Pneumonia Detection API",
    description="Enterprise-grade API for pneumonia detection from chest X-ray images",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(SecurityMiddleware)
app.add_middleware(RateLimitMiddleware, calls=100, period=60)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Configure for production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time and request tracking."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Record metrics
    REQUEST_COUNTER.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    return response

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Comprehensive health check endpoint."""
    health_status = await health_checker.check_all()
    status_code = status.HTTP_200_OK if health_status["healthy"] else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(content=health_status, status_code=status_code)

@app.get("/health/liveness", tags=["Health"])
async def liveness_probe():
    """Kubernetes liveness probe endpoint."""
    return {"status": "alive", "timestamp": time.time()}

@app.get("/health/readiness", tags=["Health"])
async def readiness_probe():
    """Kubernetes readiness probe endpoint."""
    if model_instance and model_instance.is_loaded():
        return {"status": "ready", "timestamp": time.time()}
    raise HTTPException(status_code=503, detail="Service not ready")

@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_pneumonia(
    file: UploadFile = File(..., description="Chest X-ray image file"),
    model_version: Optional[str] = Field(None, description="Specific model version to use")
):
    """
    Predict pneumonia from chest X-ray image.
    
    Returns confidence scores and interpretability information.
    """
    if not model_instance:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        # Validate file
        await validate_image(file)
        
        # Process image
        image_data = await file.read()
        processed_image = await preprocess_image(image_data)
        
        # Make prediction with timing
        with PREDICTION_DURATION.time():
            prediction_result = await model_instance.predict(
                processed_image, 
                model_version=model_version
            )
        
        # Record metrics
        PREDICTION_COUNTER.labels(
            model_version=prediction_result.model_version,
            status="success"
        ).inc()
        
        # Log prediction for monitoring
        model_metrics.record_prediction(
            confidence=prediction_result.confidence,
            prediction=prediction_result.prediction,
            model_version=prediction_result.model_version
        )
        
        return prediction_result
        
    except Exception as e:
        PREDICTION_COUNTER.labels(
            model_version=model_version or "default",
            status="error"
        ).inc()
        
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=List[PredictionResponse], tags=["Prediction"])
async def predict_batch(
    files: List[UploadFile] = File(..., description="Multiple chest X-ray images"),
    model_version: Optional[str] = Field(None, description="Specific model version to use")
):
    """Batch prediction endpoint for multiple images."""
    if not model_instance:
        raise HTTPException(status_code=503, detail="Model not available")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size limited to 10 images")
    
    results = []
    tasks = []
    
    for file in files:
        task = predict_single_image(file, model_version)
        tasks.append(task)
    
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in batch
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "filename": files[i].filename,
                    "error": str(result),
                    "prediction": None,
                    "confidence": None
                })
            else:
                processed_results.append(result)
        
        return processed_results
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

async def predict_single_image(file: UploadFile, model_version: Optional[str]):
    """Helper function for single image prediction in batch."""
    await validate_image(file)
    image_data = await file.read()
    processed_image = await preprocess_image(image_data)
    return await model_instance.predict(processed_image, model_version=model_version)

@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get information about the loaded model."""
    if not model_instance:
        raise HTTPException(status_code=503, detail="Model not available")
    
    return await model_instance.get_model_info()

@app.post("/model/reload", tags=["Model"])
async def reload_model(model_version: Optional[str] = None):
    """Reload the model (admin endpoint)."""
    if not model_instance:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        await model_instance.reload_model(model_version)
        return {"message": "Model reloaded successfully", "version": model_version}
    except Exception as e:
        logger.error(f"Model reload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time(),
            "path": request.url.path
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler for unhandled exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": time.time(),
            "path": request.url.path
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        access_log=True,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    )