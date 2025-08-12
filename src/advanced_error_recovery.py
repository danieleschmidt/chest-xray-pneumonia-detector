"""
Advanced Error Recovery System for Medical AI Training
Implements sophisticated error handling and recovery mechanisms.
"""

import logging
import traceback
import time
import json
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import threading
from queue import Queue
import pickle

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorReport:
    """Detailed error report for analysis and recovery."""
    timestamp: float
    error_type: str
    severity: ErrorSeverity
    message: str
    stack_trace: str
    context: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_actions: List[str] = None
    
    def __post_init__(self):
        if self.recovery_actions is None:
            self.recovery_actions = []


class RecoveryStrategy:
    """Base class for error recovery strategies."""
    
    def __init__(self, name: str, max_attempts: int = 3):
        self.name = name
        self.max_attempts = max_attempts
        self.attempt_count = 0
        
    def can_recover(self, error: ErrorReport) -> bool:
        """Check if this strategy can handle the error."""
        return self.attempt_count < self.max_attempts
        
    def recover(self, error: ErrorReport, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Attempt error recovery. Returns (success, updated_context)."""
        self.attempt_count += 1
        return False, context
        
    def reset(self):
        """Reset attempt counter."""
        self.attempt_count = 0


class ModelCheckpointRecovery(RecoveryStrategy):
    """Recovery strategy using model checkpoints."""
    
    def __init__(self, checkpoint_dir: Path, max_attempts: int = 3):
        super().__init__("ModelCheckpointRecovery", max_attempts)
        self.checkpoint_dir = Path(checkpoint_dir)
        
    def can_recover(self, error: ErrorReport) -> bool:
        """Can recover from training failures."""
        training_errors = ["NaN loss", "exploding gradients", "model divergence", "OOM"]
        return (super().can_recover(error) and 
                any(err in error.message for err in training_errors) and
                self._has_valid_checkpoint())
                
    def _has_valid_checkpoint(self) -> bool:
        """Check if valid checkpoint exists."""
        if not self.checkpoint_dir.exists():
            return False
            
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.keras"))
        return len(checkpoint_files) > 0
        
    def recover(self, error: ErrorReport, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Recover by loading last valid checkpoint."""
        super().recover(error, context)
        
        try:
            # Find most recent checkpoint
            checkpoint_files = sorted(
                self.checkpoint_dir.glob("checkpoint_*.keras"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            if not checkpoint_files:
                logger.error("No checkpoint files found for recovery")
                return False, context
                
            latest_checkpoint = checkpoint_files[0]
            logger.info(f"Attempting recovery from checkpoint: {latest_checkpoint}")
            
            # Load model from checkpoint
            if "model" in context:
                try:
                    context["model"] = tf.keras.models.load_model(latest_checkpoint)
                    logger.info("Model successfully loaded from checkpoint")
                    
                    # Reset optimizer state
                    if "optimizer" in context:
                        context["optimizer"] = tf.keras.optimizers.Adam(
                            learning_rate=context.get("learning_rate", 0.001)
                        )
                        context["model"].compile(
                            optimizer=context["optimizer"],
                            loss=context.get("loss", "binary_crossentropy"),
                            metrics=context.get("metrics", ["accuracy"])
                        )
                        
                    return True, context
                    
                except Exception as e:
                    logger.error(f"Failed to load checkpoint: {e}")
                    return False, context
                    
        except Exception as e:
            logger.error(f"Checkpoint recovery failed: {e}")
            return False, context
            
        return False, context


class DataCorruptionRecovery(RecoveryStrategy):
    """Recovery strategy for data corruption issues."""
    
    def __init__(self, backup_data_dir: Optional[Path] = None):
        super().__init__("DataCorruptionRecovery", max_attempts=2)
        self.backup_data_dir = backup_data_dir
        
    def can_recover(self, error: ErrorReport) -> bool:
        """Can recover from data loading errors."""
        data_errors = ["corrupted", "invalid format", "permission denied", "not found"]
        return (super().can_recover(error) and
                any(err in error.message.lower() for err in data_errors))
                
    def recover(self, error: ErrorReport, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Recover by switching to backup data or regenerating."""
        super().recover(error, context)
        
        try:
            # First attempt: use backup data
            if self.attempt_count == 1 and self.backup_data_dir and self.backup_data_dir.exists():
                logger.info(f"Switching to backup data: {self.backup_data_dir}")
                context["data_dir"] = self.backup_data_dir
                return True, context
                
            # Second attempt: regenerate dummy data
            elif self.attempt_count == 2:
                logger.info("Generating dummy data for recovery")
                context["use_dummy_data"] = True
                context["dummy_samples"] = context.get("dummy_samples", 100)
                return True, context
                
        except Exception as e:
            logger.error(f"Data recovery failed: {e}")
            return False, context
            
        return False, context


class MemoryRecovery(RecoveryStrategy):
    """Recovery strategy for memory issues."""
    
    def __init__(self):
        super().__init__("MemoryRecovery", max_attempts=3)
        
    def can_recover(self, error: ErrorReport) -> bool:
        """Can recover from memory errors."""
        memory_errors = ["OOM", "out of memory", "memory limit", "allocation failed"]
        return (super().can_recover(error) and
                any(err in error.message for err in memory_errors))
                
    def recover(self, error: ErrorReport, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Recover by reducing memory usage."""
        super().recover(error, context)
        
        try:
            if self.attempt_count == 1:
                # Reduce batch size
                old_batch_size = context.get("batch_size", 32)
                new_batch_size = max(1, old_batch_size // 2)
                context["batch_size"] = new_batch_size
                logger.info(f"Reduced batch size from {old_batch_size} to {new_batch_size}")
                
            elif self.attempt_count == 2:
                # Clear TensorFlow memory
                tf.keras.backend.clear_session()
                if tf.config.list_physical_devices('GPU'):
                    for gpu in tf.config.experimental.list_physical_devices('GPU'):
                        tf.config.experimental.set_memory_growth(gpu, True)
                logger.info("Cleared TensorFlow session and enabled memory growth")
                
            elif self.attempt_count == 3:
                # Reduce model complexity
                if "model_params" in context:
                    params = context["model_params"]
                    params["hidden_units"] = max(16, params.get("hidden_units", 64) // 2)
                    params["num_layers"] = max(1, params.get("num_layers", 3) - 1)
                    logger.info("Reduced model complexity for memory conservation")
                    
            return True, context
            
        except Exception as e:
            logger.error(f"Memory recovery failed: {e}")
            return False, context


class GradientRecovery(RecoveryStrategy):
    """Recovery strategy for gradient-related issues."""
    
    def __init__(self):
        super().__init__("GradientRecovery", max_attempts=3)
        
    def can_recover(self, error: ErrorReport) -> bool:
        """Can recover from gradient issues."""
        gradient_errors = ["exploding", "vanishing", "NaN", "gradient"]
        return (super().can_recover(error) and
                any(err in error.message for err in gradient_errors))
                
    def recover(self, error: ErrorReport, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Recover by adjusting training parameters."""
        super().recover(error, context)
        
        try:
            if self.attempt_count == 1:
                # Reduce learning rate
                old_lr = context.get("learning_rate", 0.001)
                new_lr = old_lr * 0.1
                context["learning_rate"] = new_lr
                logger.info(f"Reduced learning rate from {old_lr} to {new_lr}")
                
            elif self.attempt_count == 2:
                # Add gradient clipping
                context["gradient_clip_norm"] = 1.0
                context["gradient_clip_value"] = 0.5
                logger.info("Added gradient clipping")
                
            elif self.attempt_count == 3:
                # Change optimizer
                context["optimizer_type"] = "RMSprop"
                context["learning_rate"] = 0.0001
                logger.info("Switched to RMSprop optimizer with very low learning rate")
                
            return True, context
            
        except Exception as e:
            logger.error(f"Gradient recovery failed: {e}")
            return False, context


class AdvancedErrorRecoverySystem:
    """
    Comprehensive error recovery system for medical AI training.
    
    Features:
    - Multi-strategy error recovery
    - Error pattern analysis
    - Automatic fallback mechanisms
    - Recovery history tracking
    - Predictive error prevention
    """
    
    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        backup_data_dir: Optional[Path] = None,
        max_recovery_attempts: int = 5,
        error_log_file: Optional[Path] = None
    ):
        """
        Initialize advanced error recovery system.
        
        Args:
            checkpoint_dir: Directory for model checkpoints
            backup_data_dir: Directory for backup data
            max_recovery_attempts: Maximum total recovery attempts
            error_log_file: File to log error reports
        """
        self.max_recovery_attempts = max_recovery_attempts
        self.error_log_file = error_log_file
        self.error_history = []
        self.recovery_success_rate = {}
        
        # Initialize recovery strategies
        self.recovery_strategies = [
            ModelCheckpointRecovery(checkpoint_dir or Path("./checkpoints")),
            DataCorruptionRecovery(backup_data_dir),
            MemoryRecovery(),
            GradientRecovery()
        ]
        
        logger.info("Initialized AdvancedErrorRecoverySystem")
        
    def capture_error(
        self,
        exception: Exception,
        context: Dict[str, Any],
        severity: ErrorSeverity = ErrorSeverity.MEDIUM
    ) -> ErrorReport:
        """Capture detailed error information."""
        
        error_report = ErrorReport(
            timestamp=time.time(),
            error_type=type(exception).__name__,
            severity=severity,
            message=str(exception),
            stack_trace=traceback.format_exc(),
            context=self._sanitize_context(context)
        )
        
        self.error_history.append(error_report)
        self._log_error(error_report)
        
        return error_report
        
    def _sanitize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize context for logging (remove large objects)."""
        sanitized = {}
        
        for key, value in context.items():
            if isinstance(value, (str, int, float, bool, list, dict)):
                if isinstance(value, (list, dict)) and len(str(value)) > 1000:
                    sanitized[key] = f"<large_{type(value).__name__}>"
                else:
                    sanitized[key] = value
            elif hasattr(value, '__name__'):
                sanitized[key] = value.__name__
            else:
                sanitized[key] = str(type(value))
                
        return sanitized
        
    def _log_error(self, error_report: ErrorReport):
        """Log error report to file."""
        if self.error_log_file:
            try:
                with open(self.error_log_file, 'a') as f:
                    f.write(json.dumps(asdict(error_report), default=str) + '\n')
            except Exception as e:
                logger.error(f"Failed to log error: {e}")
                
    def attempt_recovery(
        self,
        error_report: ErrorReport,
        context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Attempt error recovery using available strategies."""
        
        logger.info(f"Attempting recovery for {error_report.error_type}: {error_report.message}")
        
        # Find applicable recovery strategies
        applicable_strategies = [
            strategy for strategy in self.recovery_strategies
            if strategy.can_recover(error_report)
        ]
        
        if not applicable_strategies:
            logger.warning("No applicable recovery strategies found")
            return False, context
            
        # Sort strategies by success rate (if available)
        applicable_strategies.sort(
            key=lambda s: self.recovery_success_rate.get(s.name, 0.5),
            reverse=True
        )
        
        # Attempt recovery with each strategy
        for strategy in applicable_strategies:
            try:
                logger.info(f"Trying recovery strategy: {strategy.name}")
                
                success, updated_context = strategy.recover(error_report, context.copy())
                
                if success:
                    logger.info(f"Recovery successful using {strategy.name}")
                    error_report.recovery_attempted = True
                    error_report.recovery_successful = True
                    error_report.recovery_actions.append(strategy.name)
                    
                    # Update success rate
                    self._update_success_rate(strategy.name, True)
                    
                    return True, updated_context
                else:
                    logger.warning(f"Recovery failed with {strategy.name}")
                    self._update_success_rate(strategy.name, False)
                    
            except Exception as e:
                logger.error(f"Recovery strategy {strategy.name} raised exception: {e}")
                self._update_success_rate(strategy.name, False)
                
        # No strategy succeeded
        error_report.recovery_attempted = True
        error_report.recovery_successful = False
        logger.error("All recovery strategies failed")
        
        return False, context
        
    def _update_success_rate(self, strategy_name: str, success: bool):
        """Update success rate for recovery strategy."""
        if strategy_name not in self.recovery_success_rate:
            self.recovery_success_rate[strategy_name] = 0.5  # Initial neutral rate
            
        # Exponential moving average
        alpha = 0.1
        current_rate = self.recovery_success_rate[strategy_name]
        new_rate = (1 - alpha) * current_rate + alpha * (1.0 if success else 0.0)
        self.recovery_success_rate[strategy_name] = new_rate
        
    def analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze error patterns for predictive prevention."""
        if len(self.error_history) < 5:
            return {"message": "Insufficient error history for analysis"}
            
        # Error frequency analysis
        error_types = {}
        error_severities = {}
        recovery_rates = {}
        
        for error in self.error_history:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
            error_severities[error.severity.value] = error_severities.get(error.severity.value, 0) + 1
            
            if error.recovery_attempted:
                key = error.error_type
                if key not in recovery_rates:
                    recovery_rates[key] = {"attempts": 0, "successes": 0}
                recovery_rates[key]["attempts"] += 1
                if error.recovery_successful:
                    recovery_rates[key]["successes"] += 1
                    
        # Calculate success rates
        for error_type in recovery_rates:
            attempts = recovery_rates[error_type]["attempts"]
            successes = recovery_rates[error_type]["successes"]
            recovery_rates[error_type]["success_rate"] = successes / attempts if attempts > 0 else 0
            
        return {
            "total_errors": len(self.error_history),
            "error_types": error_types,
            "error_severities": error_severities,
            "recovery_rates": recovery_rates,
            "strategy_success_rates": self.recovery_success_rate
        }
        
    def predict_failure_risk(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Predict failure risk based on context and history."""
        risks = {}
        
        # Memory risk
        batch_size = context.get("batch_size", 32)
        model_complexity = context.get("model_complexity", "medium")
        
        if batch_size > 64:
            risks["memory_risk"] = min(1.0, batch_size / 128)
        else:
            risks["memory_risk"] = 0.1
            
        # Gradient risk
        learning_rate = context.get("learning_rate", 0.001)
        if learning_rate > 0.01:
            risks["gradient_risk"] = min(1.0, learning_rate / 0.1)
        else:
            risks["gradient_risk"] = 0.1
            
        # Data risk
        if context.get("use_dummy_data", False):
            risks["data_risk"] = 0.1
        else:
            risks["data_risk"] = 0.3
            
        return risks
        
    def get_preventive_recommendations(self, context: Dict[str, Any]) -> List[str]:
        """Get recommendations to prevent common errors."""
        recommendations = []
        risks = self.predict_failure_risk(context)
        
        if risks.get("memory_risk", 0) > 0.7:
            recommendations.append("Consider reducing batch size to prevent OOM errors")
            
        if risks.get("gradient_risk", 0) > 0.7:
            recommendations.append("Consider reducing learning rate to prevent gradient issues")
            
        if risks.get("data_risk", 0) > 0.5:
            recommendations.append("Verify data integrity before training")
            
        # Add recommendations based on error history
        error_types = [e.error_type for e in self.error_history[-10:]]  # Last 10 errors
        
        if error_types.count("ResourceExhaustedError") > 2:
            recommendations.append("Frequent memory errors detected - consider model optimization")
            
        if error_types.count("InvalidArgumentError") > 2:
            recommendations.append("Frequent argument errors - validate input shapes and types")
            
        return recommendations
        
    def save_recovery_state(self, filepath: Path):
        """Save recovery system state."""
        state = {
            "error_history": [asdict(error) for error in self.error_history],
            "recovery_success_rate": self.recovery_success_rate,
            "timestamp": time.time()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
            
        logger.info(f"Saved recovery state to {filepath}")
        
    def load_recovery_state(self, filepath: Path):
        """Load recovery system state."""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
                
            # Reconstruct error history
            self.error_history = []
            for error_data in state["error_history"]:
                error_data["severity"] = ErrorSeverity(error_data["severity"])
                error_report = ErrorReport(**error_data)
                self.error_history.append(error_report)
                
            self.recovery_success_rate = state["recovery_success_rate"]
            
            logger.info(f"Loaded recovery state from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load recovery state: {e}")


# Context manager for automatic error handling
class ErrorRecoveryContext:
    """Context manager for automatic error handling and recovery."""
    
    def __init__(self, recovery_system: AdvancedErrorRecoverySystem, context: Dict[str, Any]):
        self.recovery_system = recovery_system
        self.context = context
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_value, traceback_obj):
        if exc_value is not None:
            error_report = self.recovery_system.capture_error(
                exc_value, 
                self.context,
                ErrorSeverity.HIGH if exc_type.__name__ in ["SystemError", "MemoryError"] else ErrorSeverity.MEDIUM
            )
            
            success, updated_context = self.recovery_system.attempt_recovery(error_report, self.context)
            
            if success:
                logger.info("Error recovered successfully, continuing execution")
                self.context.update(updated_context)
                return True  # Suppress the exception
            else:
                logger.error("Error recovery failed, re-raising exception")
                return False  # Re-raise the exception
                
        return False


# Example usage
if __name__ == "__main__":
    # Initialize recovery system
    recovery_system = AdvancedErrorRecoverySystem(
        checkpoint_dir=Path("./checkpoints"),
        error_log_file=Path("./error_log.jsonl")
    )
    
    # Example training context
    training_context = {
        "model_type": "CNN",
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 10
    }
    
    # Get preventive recommendations
    recommendations = recovery_system.get_preventive_recommendations(training_context)
    print(f"Preventive recommendations: {recommendations}")
    
    print("Advanced Error Recovery System initialized successfully!")