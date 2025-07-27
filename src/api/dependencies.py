"""
Dependency injection for FastAPI endpoints.
"""

import logging
from typing import Any

from fastapi import Depends, HTTPException, status
from ..monitoring.logging import get_logger


class ModelService:
    """Mock model service for API endpoints."""
    
    def __init__(self):
        self.model = None
        self.model_version = "v1.0.0"
        self.model_loaded = False
        
    async def predict_single(self, image_data: bytes) -> Any:
        """Mock single prediction."""
        # This would be replaced with actual model inference
        import time
        from datetime import datetime
        
        # Simulate processing time
        import asyncio
        await asyncio.sleep(0.1)
        
        # Mock prediction result
        class MockPrediction:
            def __init__(self):
                self.prediction = 1  # Mock: pneumonia detected
                self.confidence = 0.87
                self.class_name = "Pneumonia"
                self.model_version = "v1.0.0"
                self.timestamp = datetime.utcnow()
        
        return MockPrediction()
    
    async def predict_batch(self, image_data_list: list) -> list:
        """Mock batch prediction."""
        predictions = []
        for _ in image_data_list:
            prediction = await self.predict_single(b"mock_data")
            predictions.append(prediction)
        return predictions
    
    async def get_model_info(self) -> Any:
        """Get model information."""
        from datetime import datetime
        from ..api.models import ModelInfoResponse
        
        return ModelInfoResponse(
            model_name="pneumonia_detector_cnn",
            model_version=self.model_version,
            model_type="CNN",
            input_shape=[224, 224, 3],
            output_shape=[1],
            parameters=1235467,
            size_mb=4.7,
            classes=["Normal", "Pneumonia"],
            loaded_at=datetime.utcnow()
        )
    
    async def reload_model(self):
        """Reload the model."""
        # Mock model reload
        import asyncio
        await asyncio.sleep(0.5)
        self.model_loaded = True


# Global model service instance
_model_service = ModelService()


def get_model_service() -> ModelService:
    """Dependency to get model service."""
    if not _model_service.model_loaded:
        # Initialize model service if not already done
        _model_service.model_loaded = True
    
    return _model_service


def get_logger(name: str = __name__) -> logging.Logger:
    """Dependency to get logger instance."""
    return get_logger(name)


def get_current_user():
    """Dependency to get current authenticated user."""
    # This would be replaced with actual authentication
    # For now, return a mock user
    return {
        "user_id": "mock_user",
        "username": "test_user",
        "role": "user"
    }