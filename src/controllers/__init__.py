"""
Controller layer for API request handling.
Implements clean separation between API endpoints and business logic.
"""

from .prediction_controller import PredictionController

__all__ = ["PredictionController"]