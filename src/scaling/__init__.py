"""Auto-scaling system for dynamic resource management."""

from .auto_scaler import (
    AutoScaler,
    ScalingStrategy,
    LinearScalingStrategy,
    PredictiveScalingStrategy,
    MetricThreshold,
    ScalingEvent,
    auto_scaler
)

__all__ = [
    'AutoScaler',
    'ScalingStrategy',
    'LinearScalingStrategy',
    'PredictiveScalingStrategy',
    'MetricThreshold',
    'ScalingEvent',
    'auto_scaler'
]