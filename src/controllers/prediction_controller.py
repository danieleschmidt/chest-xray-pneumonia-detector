"""
REST API controllers for pneumonia detection endpoints.
Implements clean separation between API layer and business logic.
"""

import asyncio
import logging
import time
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from ..services.prediction_service import PredictionService, PredictionRequest, DiagnosticReportGenerator
from ..api.schemas import PredictionResponse, BatchPredictionResponse, ModelInfoResponse
from ..api.utils import validate_image, calculate_image_hash, sanitize_filename
from ..monitoring.metrics import ModelMetrics


logger = logging.getLogger(__name__)


class PredictionController:
    """Controller for prediction-related endpoints."""
    
    def __init__(self):
        self.prediction_service = PredictionService()
        self.metrics = ModelMetrics()
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize the controller and underlying services."""
        if not self._initialized:
            await self.prediction_service.initialize()
            self._initialized = True
            logger.info("PredictionController initialized")
    
    async def predict_single(
        self,
        file: UploadFile,
        model_version: Optional[str] = None,
        include_gradcam: bool = False,
        confidence_threshold: float = 0.5
    ) -> PredictionResponse:
        """
        Handle single image prediction request.
        
        Args:
            file: Uploaded image file
            model_version: Specific model version to use
            include_gradcam: Whether to include Grad-CAM visualization
            confidence_threshold: Confidence threshold for classification
            
        Returns:
            Prediction response with results
        """
        if not self._initialized:
            await self.initialize()
        
        request_start = time.time()
        
        try:
            # Validate uploaded file
            validation_result = await validate_image(file)
            logger.info(f"Image validation passed: {validation_result}")
            
            # Read file content
            file_content = await file.read()
            
            # Create prediction request
            prediction_request = PredictionRequest(
                image_data=file_content,
                model_version=model_version,
                include_interpretability=include_gradcam,
                confidence_threshold=confidence_threshold,
                quality_check=True
            )
            
            # Make prediction
            result = await self.prediction_service.predict(prediction_request)
            
            # Calculate image hash for tracking
            image_hash = calculate_image_hash(file_content)
            
            # Create response
            response = PredictionResponse(
                prediction=result.prediction,
                confidence=result.confidence,
                class_probabilities=result.probabilities,
                model_version=result.model_version,
                inference_time=result.processing_time_ms / 1000,
                prediction_id=f"pred_{int(time.time() * 1000)}",
                timestamp=time.time(),
                interpretability=result.interpretability_data,
                metadata={
                    "image_hash": image_hash,
                    "filename": await sanitize_filename(file.filename or "unknown"),
                    "file_size": len(file_content),
                    "confidence_category": result.confidence_category.value,
                    "quality_score": result.image_quality_score,
                    "clinical_recommendation": result.clinical_recommendation,
                    "processing_time_total": (time.time() - request_start) * 1000
                }
            )
            
            # Log successful prediction
            logger.info(
                f"Prediction completed: {result.prediction} "
                f"(confidence: {result.confidence:.3f}, "
                f"time: {result.processing_time_ms:.1f}ms)"
            )
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}"
            )
    
    async def predict_batch(
        self,
        files: List[UploadFile],
        model_version: Optional[str] = None,
        confidence_threshold: float = 0.5
    ) -> BatchPredictionResponse:
        """
        Handle batch prediction request.
        
        Args:
            files: List of uploaded image files
            model_version: Specific model version to use
            confidence_threshold: Confidence threshold for classification
            
        Returns:
            Batch prediction response
        """
        if not self._initialized:
            await self.initialize()
        
        batch_start = time.time()
        max_batch_size = 10
        
        if len(files) > max_batch_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Batch size limited to {max_batch_size} images"
            )
        
        # Validate all files first
        try:
            validation_tasks = [validate_image(file) for file in files]
            await asyncio.gather(*validation_tasks)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File validation failed: {str(e)}"
            )
        
        # Create prediction requests
        prediction_requests = []
        for file in files:
            file_content = await file.read()
            request = PredictionRequest(
                image_data=file_content,
                model_version=model_version,
                confidence_threshold=confidence_threshold,
                quality_check=True
            )
            prediction_requests.append(request)
        
        # Process batch
        try:
            results = await self.prediction_service.predict_batch(prediction_requests)
            
            # Convert to response format
            predictions = []
            successful_count = 0
            failed_count = 0
            
            for i, result in enumerate(results):
                if result.prediction != "ERROR":
                    successful_count += 1
                    
                    response = PredictionResponse(
                        prediction=result.prediction,
                        confidence=result.confidence,
                        class_probabilities=result.probabilities,
                        model_version=result.model_version,
                        inference_time=result.processing_time_ms / 1000,
                        prediction_id=f"batch_pred_{int(time.time() * 1000)}_{i}",
                        timestamp=time.time(),
                        metadata={
                            "filename": await sanitize_filename(files[i].filename or f"batch_image_{i}"),
                            "batch_index": i,
                            "confidence_category": result.confidence_category.value,
                            "quality_score": result.image_quality_score
                        }
                    )
                else:
                    failed_count += 1
                    # Create error response
                    response = PredictionResponse(
                        prediction="ERROR",
                        confidence=0.0,
                        class_probabilities={"ERROR": 1.0},
                        model_version="unknown",
                        inference_time=0.0,
                        prediction_id=f"batch_error_{int(time.time() * 1000)}_{i}",
                        timestamp=time.time(),
                        metadata={
                            "filename": await sanitize_filename(files[i].filename or f"batch_image_{i}"),
                            "batch_index": i,
                            "error": result.metadata.get("error", "Unknown error")
                        }
                    )
                
                predictions.append(response)
            
            batch_time = (time.time() - batch_start) * 1000
            
            return BatchPredictionResponse(
                predictions=predictions,
                batch_size=len(files),
                total_processing_time_ms=batch_time,
                successful_predictions=successful_count,
                failed_predictions=failed_count
            )
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Batch prediction failed: {str(e)}"
            )
    
    async def get_model_info(self) -> ModelInfoResponse:
        """Get comprehensive model information."""
        if not self._initialized:
            await self.initialize()
        
        try:
            model_info = await self.prediction_service.model.get_model_info()
            return model_info
        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve model information"
            )
    
    async def reload_model(self, model_version: Optional[str] = None) -> Dict[str, Any]:
        """Reload model with optional version specification."""
        if not self._initialized:
            await self.initialize()
        
        try:
            await self.prediction_service.reload_models(model_version)
            return {
                "message": "Models reloaded successfully",
                "model_version": model_version,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to reload model: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model reload failed: {str(e)}"
            )
    
    async def generate_diagnostic_report(
        self,
        prediction_response: PredictionResponse,
        patient_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a comprehensive diagnostic report."""
        try:
            # Convert response to result format for report generation
            from ..services.prediction_service import PredictionResult, PredictionConfidence
            
            confidence_category = PredictionConfidence.HIGH if prediction_response.confidence > 0.8 else PredictionConfidence.MEDIUM
            
            result = PredictionResult(
                prediction=prediction_response.prediction,
                confidence=prediction_response.confidence,
                confidence_category=confidence_category,
                probabilities=prediction_response.class_probabilities,
                model_version=prediction_response.model_version,
                processing_time_ms=prediction_response.inference_time * 1000,
                image_quality_score=prediction_response.metadata.get("quality_score", 1.0),
                clinical_recommendation=prediction_response.metadata.get("clinical_recommendation", "No specific recommendation"),
                interpretability_data=prediction_response.interpretability
            )
            
            report = DiagnosticReportGenerator.generate_report(result, patient_id)
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate diagnostic report: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate diagnostic report"
            )
    
    async def get_prediction_metrics(self) -> Dict[str, Any]:
        """Get prediction performance metrics."""
        try:
            metrics = await self.prediction_service.get_model_performance_metrics()
            return metrics
        except Exception as e:
            logger.error(f"Failed to get metrics: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve metrics"
            )
    
    async def cleanup(self) -> None:
        """Clean up controller resources."""
        if self.prediction_service:
            await self.prediction_service.cleanup()
        self._initialized = False
        logger.info("PredictionController cleaned up")