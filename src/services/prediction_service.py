"""
Core prediction service implementing business logic for pneumonia detection.
Provides high-level interface for model inference with comprehensive error handling.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np
from PIL import Image

from ..api.models import PneumoniaDetectionModel, ModelEnsemble
from ..api.schemas import PredictionResponse
from ..monitoring.metrics import ModelMetrics
from ..model_registry import ModelRegistry


logger = logging.getLogger(__name__)


class PredictionConfidence(Enum):
    """Confidence level categories for predictions."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class PredictionRequest:
    """Internal prediction request structure."""
    image_data: bytes
    model_version: Optional[str] = None
    include_interpretability: bool = False
    confidence_threshold: float = 0.5
    quality_check: bool = True
    preprocessing_options: Optional[Dict[str, Any]] = None


@dataclass
class PredictionResult:
    """Enhanced prediction result with clinical context."""
    prediction: str
    confidence: float
    confidence_category: PredictionConfidence
    probabilities: Dict[str, float]
    model_version: str
    processing_time_ms: float
    image_quality_score: float
    clinical_recommendation: str
    interpretability_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class PredictionService:
    """Main service class for pneumonia detection predictions."""
    
    def __init__(self):
        self.model = PneumoniaDetectionModel()
        self.ensemble_model: Optional[ModelEnsemble] = None
        self.model_registry = ModelRegistry()
        self.metrics = ModelMetrics()
        self._is_initialized = False
        
    async def initialize(self, use_ensemble: bool = False) -> None:
        """Initialize the prediction service."""
        try:
            if use_ensemble:
                # Load ensemble of models for better accuracy
                ensemble_config = [
                    {"path": "saved_models/pneumonia_cnn_v1.keras", "version": "v1.0"},
                    {"path": "saved_models/pneumonia_mobilenet_v1.keras", "version": "mobilenet_v1"}
                ]
                self.ensemble_model = ModelEnsemble(ensemble_config)
                await self.ensemble_model.load_models()
                logger.info("Ensemble model initialized")
            else:
                # Load single model
                await self.model.load_model()
                logger.info("Single model initialized")
            
            self._is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize prediction service: {str(e)}")
            raise RuntimeError(f"Service initialization failed: {str(e)}")
    
    async def predict(self, request: PredictionRequest) -> PredictionResult:
        """
        Make a prediction with comprehensive analysis.
        
        Args:
            request: Prediction request containing image and options
            
        Returns:
            Enhanced prediction result with clinical context
        """
        if not self._is_initialized:
            raise RuntimeError("Prediction service not initialized")
        
        start_time = time.time()
        
        try:
            # Preprocess image
            from ..api.utils import preprocess_image, validate_medical_image_quality
            processed_image = await preprocess_image(request.image_data)
            
            # Quality assessment
            quality_metrics = {}
            if request.quality_check:
                quality_metrics = validate_medical_image_quality(processed_image[0])
                
                # Warn if quality is poor
                if not quality_metrics.get('diagnostic_quality', True):
                    logger.warning("Low quality image detected for prediction")
            
            # Make prediction
            if self.ensemble_model and self.ensemble_model.is_loaded():
                prediction_response = await self.ensemble_model.predict(processed_image)
            else:
                prediction_response = await self.model.predict(
                    processed_image, 
                    model_version=request.model_version
                )
            
            # Add interpretability if requested
            interpretability_data = None
            if request.include_interpretability:
                try:
                    gradcam_result = await self.model.predict_with_gradcam(processed_image)
                    interpretability_data = gradcam_result.get("interpretation")
                except Exception as e:
                    logger.warning(f"Failed to generate interpretability data: {str(e)}")
            
            # Calculate confidence category
            confidence_category = self._categorize_confidence(prediction_response.confidence)
            
            # Generate clinical recommendation
            clinical_recommendation = self._generate_clinical_recommendation(
                prediction_response.prediction,
                prediction_response.confidence,
                quality_metrics
            )
            
            # Calculate total processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Record metrics
            self.metrics.record_prediction(
                confidence=prediction_response.confidence,
                prediction=prediction_response.prediction,
                model_version=prediction_response.model_version,
                inference_time=processing_time / 1000
            )
            
            return PredictionResult(
                prediction=prediction_response.prediction,
                confidence=prediction_response.confidence,
                confidence_category=confidence_category,
                probabilities=prediction_response.class_probabilities,
                model_version=prediction_response.model_version,
                processing_time_ms=processing_time,
                image_quality_score=quality_metrics.get('quality_score', 1.0),
                clinical_recommendation=clinical_recommendation,
                interpretability_data=interpretability_data,
                metadata={
                    "quality_metrics": quality_metrics,
                    "preprocessing_applied": True,
                    "ensemble_used": self.ensemble_model is not None
                }
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    async def predict_batch(self, requests: List[PredictionRequest]) -> List[PredictionResult]:
        """Process multiple prediction requests efficiently."""
        if not self._is_initialized:
            raise RuntimeError("Prediction service not initialized")
        
        results = []
        
        # Process in batches to manage memory
        batch_size = 10
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [self.predict(request) for request in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions in batch
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Batch prediction {i+j} failed: {str(result)}")
                    # Create error result
                    error_result = PredictionResult(
                        prediction="ERROR",
                        confidence=0.0,
                        confidence_category=PredictionConfidence.LOW,
                        probabilities={"ERROR": 1.0},
                        model_version="unknown",
                        processing_time_ms=0.0,
                        image_quality_score=0.0,
                        clinical_recommendation="Processing failed - please retry",
                        metadata={"error": str(result)}
                    )
                    results.append(error_result)
                else:
                    results.append(result)
        
        return results
    
    def _categorize_confidence(self, confidence: float) -> PredictionConfidence:
        """Categorize prediction confidence for clinical use."""
        if confidence >= 0.95:
            return PredictionConfidence.VERY_HIGH
        elif confidence >= 0.80:
            return PredictionConfidence.HIGH
        elif confidence >= 0.60:
            return PredictionConfidence.MEDIUM
        else:
            return PredictionConfidence.LOW
    
    def _generate_clinical_recommendation(
        self, 
        prediction: str, 
        confidence: float, 
        quality_metrics: Dict[str, Any]
    ) -> str:
        """Generate clinical recommendation based on prediction and image quality."""
        
        # Base recommendations
        if prediction == "PNEUMONIA":
            if confidence >= 0.90:
                base_rec = "Strong indication of pneumonia. Consider immediate clinical evaluation."
            elif confidence >= 0.70:
                base_rec = "Possible pneumonia detected. Clinical correlation recommended."
            else:
                base_rec = "Weak pneumonia signal. Additional imaging or clinical assessment advised."
        else:  # NORMAL
            if confidence >= 0.90:
                base_rec = "No clear signs of pneumonia detected."
            elif confidence >= 0.70:
                base_rec = "Likely normal chest X-ray. Continue routine monitoring if symptomatic."
            else:
                base_rec = "Inconclusive results. Consider repeat imaging or clinical evaluation."
        
        # Add quality-based modifiers
        quality_modifier = ""
        if not quality_metrics.get('diagnostic_quality', True):
            quality_modifier = " Note: Image quality may limit diagnostic accuracy."
        elif not quality_metrics.get('is_sharp', True):
            quality_modifier = " Note: Image sharpness could be improved for better assessment."
        
        # Add disclaimer
        disclaimer = " This AI assessment is for screening purposes only and should not replace professional medical diagnosis."
        
        return base_rec + quality_modifier + disclaimer
    
    async def get_model_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive model performance metrics."""
        model_info = await self.model.get_model_info()
        
        return {
            "model_info": model_info,
            "prediction_metrics": self.metrics.get_summary(),
            "service_status": {
                "initialized": self._is_initialized,
                "model_loaded": self.model.is_loaded(),
                "ensemble_available": self.ensemble_model is not None
            }
        }
    
    async def reload_models(self, model_version: Optional[str] = None) -> None:
        """Reload models with optional version specification."""
        logger.info(f"Reloading models (version: {model_version})")
        
        # Reload main model
        await self.model.reload_model(model_version)
        
        # Reload ensemble if available
        if self.ensemble_model:
            await self.ensemble_model.load_models()
        
        logger.info("Models reloaded successfully")
    
    async def cleanup(self) -> None:
        """Clean up service resources."""
        if self.model:
            await self.model.cleanup()
        
        if self.ensemble_model:
            for model in self.ensemble_model.models:
                await model.cleanup()
        
        self._is_initialized = False
        logger.info("Prediction service cleaned up")


class DiagnosticReportGenerator:
    """Generate comprehensive diagnostic reports from predictions."""
    
    @staticmethod
    def generate_report(prediction_result: PredictionResult, patient_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate a structured diagnostic report."""
        
        report = {
            "report_id": f"CXR_{int(time.time() * 1000)}",
            "timestamp": time.time(),
            "patient_id": patient_id,
            "study_type": "Chest X-Ray Pneumonia Screening",
            "ai_analysis": {
                "prediction": prediction_result.prediction,
                "confidence": prediction_result.confidence,
                "confidence_level": prediction_result.confidence_category.value,
                "model_version": prediction_result.model_version
            },
            "image_quality": {
                "quality_score": prediction_result.image_quality_score,
                "diagnostic_quality": prediction_result.image_quality_score > 0.7
            },
            "clinical_assessment": {
                "recommendation": prediction_result.clinical_recommendation,
                "urgency": "HIGH" if prediction_result.prediction == "PNEUMONIA" and prediction_result.confidence > 0.8 else "ROUTINE"
            },
            "technical_details": {
                "processing_time_ms": prediction_result.processing_time_ms,
                "interpretability_available": prediction_result.interpretability_data is not None
            },
            "disclaimer": "This AI-generated report is for screening purposes only. Clinical correlation and professional medical interpretation are required for diagnosis and treatment decisions."
        }
        
        return report