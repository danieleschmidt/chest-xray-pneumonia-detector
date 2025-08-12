"""
Comprehensive Validation Framework for Medical AI
Implements multi-layer validation for data integrity, model safety, and clinical compliance.
"""

import logging
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
import json
import hashlib
from PIL import Image
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import warnings

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    validator_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    warnings: List[str]
    errors: List[str]
    timestamp: float


class BaseValidator:
    """Base class for all validators."""
    
    def __init__(self, name: str, critical: bool = False):
        self.name = name
        self.critical = critical
        
    def validate(self, data: Any, context: Dict[str, Any]) -> ValidationResult:
        """Perform validation and return result."""
        raise NotImplementedError
        
    def _create_result(
        self, 
        passed: bool, 
        score: float, 
        details: Dict[str, Any] = None,
        warnings: List[str] = None,
        errors: List[str] = None
    ) -> ValidationResult:
        """Create validation result."""
        import time
        return ValidationResult(
            validator_name=self.name,
            passed=passed,
            score=score,
            details=details or {},
            warnings=warnings or [],
            errors=errors or [],
            timestamp=time.time()
        )


class DataIntegrityValidator(BaseValidator):
    """Validates data integrity and quality."""
    
    def __init__(self):
        super().__init__("DataIntegrityValidator", critical=True)
        
    def validate(self, data: Any, context: Dict[str, Any]) -> ValidationResult:
        """Validate data integrity."""
        errors = []
        warnings = []
        details = {}
        score = 1.0
        
        try:
            if isinstance(data, np.ndarray):
                # Check for NaN/Inf values
                nan_count = np.sum(np.isnan(data))
                inf_count = np.sum(np.isinf(data))
                
                if nan_count > 0:
                    errors.append(f"Found {nan_count} NaN values")
                    score *= 0.5
                    
                if inf_count > 0:
                    errors.append(f"Found {inf_count} infinite values")
                    score *= 0.5
                    
                # Check data range
                data_min, data_max = np.min(data), np.max(data)
                if data_max - data_min == 0:
                    warnings.append("Data has zero variance")
                    score *= 0.8
                    
                # Check for suspicious patterns
                if len(data.shape) > 1 and np.std(data) < 1e-6:
                    warnings.append("Very low data variance detected")
                    score *= 0.9
                    
                details.update({
                    "shape": data.shape,
                    "dtype": str(data.dtype),
                    "min_value": float(data_min),
                    "max_value": float(data_max),
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data)),
                    "nan_count": int(nan_count),
                    "inf_count": int(inf_count)
                })
                
            elif isinstance(data, (list, tuple)):
                # Validate list/tuple data
                if len(data) == 0:
                    errors.append("Empty data provided")
                    score = 0.0
                else:
                    # Check consistency
                    if hasattr(data[0], 'shape'):
                        shapes = [item.shape for item in data if hasattr(item, 'shape')]
                        if len(set(shapes)) > 1:
                            warnings.append("Inconsistent data shapes detected")
                            score *= 0.8
                            
                details["length"] = len(data)
                details["type"] = type(data).__name__
                
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            score = 0.0
            
        passed = len(errors) == 0
        return self._create_result(passed, score, details, warnings, errors)


class ImageQualityValidator(BaseValidator):
    """Validates medical image quality and properties."""
    
    def __init__(self, expected_channels: int = 3, min_resolution: Tuple[int, int] = (128, 128)):
        super().__init__("ImageQualityValidator", critical=False)
        self.expected_channels = expected_channels
        self.min_resolution = min_resolution
        
    def validate(self, data: Any, context: Dict[str, Any]) -> ValidationResult:
        """Validate image quality."""
        errors = []
        warnings = []
        details = {}
        score = 1.0
        
        try:
            if isinstance(data, np.ndarray) and len(data.shape) >= 3:
                # Check image dimensions
                height, width = data.shape[:2]
                channels = data.shape[2] if len(data.shape) > 2 else 1
                
                if height < self.min_resolution[0] or width < self.min_resolution[1]:
                    warnings.append(f"Low resolution: {width}x{height} < {self.min_resolution}")
                    score *= 0.7
                    
                if channels != self.expected_channels:
                    warnings.append(f"Unexpected channels: {channels} != {self.expected_channels}")
                    score *= 0.8
                    
                # Check pixel value range
                pixel_min, pixel_max = np.min(data), np.max(data)
                if pixel_min < 0:
                    warnings.append("Negative pixel values detected")
                    score *= 0.9
                    
                if pixel_max > 255 and data.dtype != np.float32:
                    warnings.append("Pixel values exceed typical range")
                    score *= 0.9
                    
                # Check for completely black/white images
                if np.std(data) < 1e-6:
                    errors.append("Image has no variation (completely uniform)")
                    score *= 0.3
                    
                # Check contrast
                contrast_ratio = (pixel_max - pixel_min) / (pixel_max + 1e-8)
                if contrast_ratio < 0.1:
                    warnings.append("Very low contrast detected")
                    score *= 0.8
                    
                # Check for potential artifacts
                if len(data.shape) == 3 and channels == 3:
                    # Check for pure green/blue channels (common in medical imaging errors)
                    green_dominance = np.mean(data[:, :, 1]) / (np.mean(data[:, :, 0]) + 1e-8)
                    blue_dominance = np.mean(data[:, :, 2]) / (np.mean(data[:, :, 0]) + 1e-8)
                    
                    if green_dominance > 2.0 or blue_dominance > 2.0:
                        warnings.append("Suspicious color channel distribution")
                        score *= 0.9
                        
                details.update({
                    "width": int(width),
                    "height": int(height),
                    "channels": int(channels),
                    "pixel_min": float(pixel_min),
                    "pixel_max": float(pixel_max),
                    "contrast_ratio": float(contrast_ratio),
                    "dtype": str(data.dtype)
                })
                
        except Exception as e:
            errors.append(f"Image validation error: {str(e)}")
            score = 0.0
            
        passed = len(errors) == 0
        return self._create_result(passed, score, details, warnings, errors)


class ModelArchitectureValidator(BaseValidator):
    """Validates model architecture for medical AI compliance."""
    
    def __init__(self):
        super().__init__("ModelArchitectureValidator", critical=True)
        
    def validate(self, data: Any, context: Dict[str, Any]) -> ValidationResult:
        """Validate model architecture."""
        errors = []
        warnings = []
        details = {}
        score = 1.0
        
        try:
            if isinstance(data, tf.keras.Model):
                model = data
                
                # Check input shape
                if model.input_shape is None:
                    errors.append("Model input shape not defined")
                    score *= 0.5
                else:
                    input_shape = model.input_shape
                    if len(input_shape) < 2:
                        errors.append("Invalid input shape")
                        score *= 0.7
                        
                    details["input_shape"] = input_shape
                    
                # Check output shape for medical classification
                if hasattr(model, 'output_shape'):
                    output_shape = model.output_shape
                    if isinstance(output_shape, tuple) and len(output_shape) > 1:
                        num_classes = output_shape[-1]
                        if num_classes == 1:
                            # Binary classification
                            last_layer = model.layers[-1]
                            if hasattr(last_layer, 'activation'):
                                if last_layer.activation.__name__ != 'sigmoid':
                                    warnings.append("Binary classification should use sigmoid activation")
                                    score *= 0.9
                        else:
                            # Multi-class classification
                            last_layer = model.layers[-1]
                            if hasattr(last_layer, 'activation'):
                                if last_layer.activation.__name__ != 'softmax':
                                    warnings.append("Multi-class classification should use softmax activation")
                                    score *= 0.9
                                    
                    details["output_shape"] = output_shape
                    
                # Count parameters
                total_params = model.count_params()
                trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
                
                # Check for reasonable parameter count
                if total_params > 50_000_000:  # 50M parameters
                    warnings.append("Very large model - consider efficiency for medical deployment")
                    score *= 0.8
                elif total_params < 1000:
                    warnings.append("Very small model - may lack capacity")
                    score *= 0.9
                    
                # Check for dropout layers (important for medical AI)
                has_dropout = any('dropout' in layer.name.lower() for layer in model.layers)
                if not has_dropout and total_params > 10000:
                    warnings.append("No dropout layers found - consider adding for regularization")
                    score *= 0.9
                    
                # Check for batch normalization
                has_batch_norm = any('batch' in layer.name.lower() for layer in model.layers)
                
                # Validate layer types for medical imaging
                conv_layers = [layer for layer in model.layers if 'conv' in layer.__class__.__name__.lower()]
                pool_layers = [layer for layer in model.layers if 'pool' in layer.__class__.__name__.lower()]
                
                if len(conv_layers) == 0 and 'image' in context.get('data_type', '').lower():
                    warnings.append("No convolutional layers for image data")
                    score *= 0.8
                    
                details.update({
                    "total_params": int(total_params),
                    "trainable_params": int(trainable_params),
                    "num_layers": len(model.layers),
                    "has_dropout": has_dropout,
                    "has_batch_norm": has_batch_norm,
                    "conv_layers": len(conv_layers),
                    "pool_layers": len(pool_layers)
                })
                
        except Exception as e:
            errors.append(f"Model validation error: {str(e)}")
            score = 0.0
            
        passed = len(errors) == 0
        return self._create_result(passed, score, details, warnings, errors)


class TrainingConfigValidator(BaseValidator):
    """Validates training configuration for medical AI."""
    
    def __init__(self):
        super().__init__("TrainingConfigValidator", critical=False)
        
    def validate(self, data: Any, context: Dict[str, Any]) -> ValidationResult:
        """Validate training configuration."""
        errors = []
        warnings = []
        details = {}
        score = 1.0
        
        try:
            config = data if isinstance(data, dict) else context
            
            # Check learning rate
            lr = config.get('learning_rate', 0.001)
            if lr > 0.1:
                warnings.append(f"Very high learning rate: {lr}")
                score *= 0.8
            elif lr < 1e-6:
                warnings.append(f"Very low learning rate: {lr}")
                score *= 0.9
                
            # Check batch size
            batch_size = config.get('batch_size', 32)
            if batch_size < 8:
                warnings.append("Small batch size may cause training instability")
                score *= 0.9
            elif batch_size > 128:
                warnings.append("Large batch size may require learning rate adjustment")
                score *= 0.9
                
            # Check epochs
            epochs = config.get('epochs', 10)
            if epochs < 5:
                warnings.append("Very few epochs - model may underfit")
                score *= 0.8
            elif epochs > 1000:
                warnings.append("Very many epochs - risk of overfitting")
                score *= 0.9
                
            # Check validation split
            val_split = config.get('validation_split', 0.2)
            if val_split < 0.1:
                warnings.append("Small validation split may not be representative")
                score *= 0.8
            elif val_split > 0.5:
                warnings.append("Large validation split reduces training data")
                score *= 0.9
                
            # Check for early stopping
            callbacks = config.get('callbacks', [])
            has_early_stopping = any('early' in str(callback).lower() for callback in callbacks)
            if not has_early_stopping:
                warnings.append("No early stopping - consider adding to prevent overfitting")
                score *= 0.9
                
            # Check optimizer
            optimizer = config.get('optimizer', 'adam')
            if isinstance(optimizer, str):
                if optimizer.lower() not in ['adam', 'rmsprop', 'sgd', 'adamw']:
                    warnings.append(f"Uncommon optimizer: {optimizer}")
                    score *= 0.9
                    
            # Medical AI specific checks
            loss_function = config.get('loss', 'binary_crossentropy')
            if 'medical' in context.get('domain', '').lower():
                if 'class_weight' not in config and 'sample_weight' not in config:
                    warnings.append("Consider class weighting for medical data imbalance")
                    score *= 0.9
                    
            details.update({
                "learning_rate": float(lr),
                "batch_size": int(batch_size),
                "epochs": int(epochs),
                "validation_split": float(val_split),
                "optimizer": str(optimizer),
                "loss_function": str(loss_function),
                "has_early_stopping": has_early_stopping
            })
            
        except Exception as e:
            errors.append(f"Config validation error: {str(e)}")
            score = 0.0
            
        passed = len(errors) == 0
        return self._create_result(passed, score, details, warnings, errors)


class ModelPerformanceValidator(BaseValidator):
    """Validates model performance metrics for medical applications."""
    
    def __init__(self, min_accuracy: float = 0.7, min_recall: float = 0.8):
        super().__init__("ModelPerformanceValidator", critical=True)
        self.min_accuracy = min_accuracy
        self.min_recall = min_recall
        
    def validate(self, data: Any, context: Dict[str, Any]) -> ValidationResult:
        """Validate model performance."""
        errors = []
        warnings = []
        details = {}
        score = 1.0
        
        try:
            if isinstance(data, dict) and 'metrics' in data:
                metrics = data['metrics']
                
                # Check accuracy
                accuracy = metrics.get('accuracy', metrics.get('val_accuracy', 0))
                if accuracy < self.min_accuracy:
                    errors.append(f"Accuracy {accuracy:.3f} below minimum {self.min_accuracy}")
                    score *= 0.5
                elif accuracy < self.min_accuracy * 1.1:
                    warnings.append(f"Accuracy {accuracy:.3f} close to minimum threshold")
                    score *= 0.8
                    
                # Check recall (critical for medical diagnosis)
                recall = metrics.get('recall', 0)
                if recall > 0 and recall < self.min_recall:
                    errors.append(f"Recall {recall:.3f} below minimum {self.min_recall} (critical for medical AI)")
                    score *= 0.3
                    
                # Check precision
                precision = metrics.get('precision', 0)
                if precision > 0 and precision < 0.6:
                    warnings.append(f"Low precision {precision:.3f} may indicate false positives")
                    score *= 0.8
                    
                # Check F1 score
                f1 = metrics.get('f1_score', 0)
                if f1 > 0 and f1 < 0.7:
                    warnings.append(f"Low F1 score {f1:.3f} indicates poor overall performance")
                    score *= 0.8
                    
                # Check for overfitting
                train_acc = metrics.get('accuracy', 0)
                val_acc = metrics.get('val_accuracy', 0)
                if train_acc > 0 and val_acc > 0:
                    acc_gap = train_acc - val_acc
                    if acc_gap > 0.15:  # 15% gap
                        warnings.append(f"Large accuracy gap {acc_gap:.3f} indicates overfitting")
                        score *= 0.7
                        
                # Check loss values
                loss = metrics.get('loss', float('inf'))
                val_loss = metrics.get('val_loss', float('inf'))
                
                if loss == float('inf') or val_loss == float('inf'):
                    errors.append("Invalid loss values detected")
                    score = 0.0
                elif val_loss > loss * 2:
                    warnings.append("Validation loss much higher than training loss")
                    score *= 0.8
                    
                details.update({
                    "accuracy": float(accuracy),
                    "val_accuracy": float(val_acc),
                    "recall": float(recall),
                    "precision": float(precision),
                    "f1_score": float(f1),
                    "loss": float(loss),
                    "val_loss": float(val_loss),
                    "accuracy_gap": float(acc_gap) if train_acc > 0 and val_acc > 0 else 0.0
                })
                
        except Exception as e:
            errors.append(f"Performance validation error: {str(e)}")
            score = 0.0
            
        passed = len(errors) == 0 and score >= 0.7  # Require high performance for medical AI
        return self._create_result(passed, score, details, warnings, errors)


class SecurityValidator(BaseValidator):
    """Validates security aspects of medical AI systems."""
    
    def __init__(self):
        super().__init__("SecurityValidator", critical=True)
        
    def validate(self, data: Any, context: Dict[str, Any]) -> ValidationResult:
        """Validate security aspects."""
        errors = []
        warnings = []
        details = {}
        score = 1.0
        
        try:
            # Check for data encryption in context
            if 'encryption_enabled' in context:
                if not context['encryption_enabled']:
                    errors.append("Data encryption not enabled for medical data")
                    score *= 0.5
                    
            # Check for secure data paths
            data_paths = context.get('data_paths', [])
            for path in data_paths:
                path_str = str(path)
                if '/tmp' in path_str or '\\temp' in path_str:
                    warnings.append(f"Data stored in temporary directory: {path_str}")
                    score *= 0.8
                    
            # Check for logging of sensitive information
            log_config = context.get('logging_config', {})
            if log_config.get('log_predictions', False):
                warnings.append("Prediction logging enabled - ensure no PHI is logged")
                score *= 0.9
                
            # Check model serialization security
            if 'model_save_path' in context:
                save_path = context['model_save_path']
                if not str(save_path).endswith(('.keras', '.h5', '.pb')):
                    warnings.append("Using pickle for model serialization - security risk")
                    score *= 0.7
                    
            # Check for differential privacy
            if 'differential_privacy' in context:
                if not context['differential_privacy']:
                    warnings.append("Differential privacy not enabled for sensitive medical data")
                    score *= 0.9
                    
            details.update({
                "encryption_enabled": context.get('encryption_enabled', False),
                "secure_paths": len([p for p in data_paths if '/tmp' not in str(p)]),
                "total_paths": len(data_paths),
                "differential_privacy": context.get('differential_privacy', False)
            })
            
        except Exception as e:
            errors.append(f"Security validation error: {str(e)}")
            score = 0.0
            
        passed = len(errors) == 0
        return self._create_result(passed, score, details, warnings, errors)


class ComprehensiveValidationFramework:
    """
    Comprehensive validation framework for medical AI systems.
    
    Provides multi-layer validation including:
    - Data integrity and quality
    - Model architecture compliance
    - Training configuration optimization
    - Performance validation
    - Security and privacy compliance
    """
    
    def __init__(
        self,
        validators: Optional[List[BaseValidator]] = None,
        strict_mode: bool = False,
        save_reports: bool = True,
        report_dir: Optional[Path] = None
    ):
        """
        Initialize comprehensive validation framework.
        
        Args:
            validators: Custom list of validators
            strict_mode: Fail on any validator failure
            save_reports: Save validation reports
            report_dir: Directory to save reports
        """
        self.strict_mode = strict_mode
        self.save_reports = save_reports
        self.report_dir = Path(report_dir) if report_dir else Path("./validation_reports")
        
        # Initialize default validators if none provided
        if validators is None:
            self.validators = [
                DataIntegrityValidator(),
                ImageQualityValidator(),
                ModelArchitectureValidator(),
                TrainingConfigValidator(),
                ModelPerformanceValidator(),
                SecurityValidator()
            ]
        else:
            self.validators = validators
            
        if self.save_reports:
            self.report_dir.mkdir(exist_ok=True)
            
        logger.info(f"Initialized validation framework with {len(self.validators)} validators")
        
    def validate_all(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Tuple[bool, List[ValidationResult]]:
        """
        Run all validators and return overall result.
        
        Args:
            data: Data to validate (can contain multiple items)
            context: Additional context information
            
        Returns:
            Tuple of (overall_passed, validation_results)
        """
        context = context or {}
        results = []
        overall_passed = True
        
        for validator in self.validators:
            try:
                # Determine which data to pass to validator
                validator_data = data
                if isinstance(data, dict):
                    # Try to find relevant data for validator
                    if validator.name == "ModelArchitectureValidator" and 'model' in data:
                        validator_data = data['model']
                    elif validator.name == "TrainingConfigValidator" and 'config' in data:
                        validator_data = data['config']
                    elif validator.name == "ModelPerformanceValidator" and 'metrics' in data:
                        validator_data = {'metrics': data['metrics']}
                        
                result = validator.validate(validator_data, context)
                results.append(result)
                
                # Check if validation passed
                if not result.passed:
                    if validator.critical or self.strict_mode:
                        overall_passed = False
                    logger.warning(f"Validator {validator.name} failed: {result.errors}")
                else:
                    logger.info(f"Validator {validator.name} passed (score: {result.score:.3f})")
                    
            except Exception as e:
                error_result = ValidationResult(
                    validator_name=validator.name,
                    passed=False,
                    score=0.0,
                    details={},
                    warnings=[],
                    errors=[f"Validator exception: {str(e)}"],
                    timestamp=time.time()
                )
                results.append(error_result)
                overall_passed = False
                logger.error(f"Validator {validator.name} raised exception: {e}")
                
        # Save report if enabled
        if self.save_reports:
            self._save_validation_report(results, overall_passed)
            
        return overall_passed, results
        
    def _save_validation_report(self, results: List[ValidationResult], overall_passed: bool):
        """Save validation report to file."""
        import time
        
        report = {
            "timestamp": time.time(),
            "overall_passed": overall_passed,
            "strict_mode": self.strict_mode,
            "total_validators": len(results),
            "passed_validators": sum(1 for r in results if r.passed),
            "failed_validators": sum(1 for r in results if not r.passed),
            "average_score": np.mean([r.score for r in results]),
            "results": [
                {
                    "validator": r.validator_name,
                    "passed": r.passed,
                    "score": r.score,
                    "details": r.details,
                    "warnings": r.warnings,
                    "errors": r.errors
                }
                for r in results
            ]
        }
        
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        report_file = self.report_dir / f"validation_report_{timestamp_str}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"Saved validation report: {report_file}")
        
    def get_validation_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate summary of validation results."""
        critical_failures = [r for r in results if not r.passed and any(v.critical for v in self.validators if v.name == r.validator_name)]
        warnings_count = sum(len(r.warnings) for r in results)
        errors_count = sum(len(r.errors) for r in results)
        
        return {
            "total_validators": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "critical_failures": len(critical_failures),
            "total_warnings": warnings_count,
            "total_errors": errors_count,
            "average_score": np.mean([r.score for r in results]),
            "min_score": min(r.score for r in results),
            "max_score": max(r.score for r in results)
        }


# Example usage
if __name__ == "__main__":
    # Initialize validation framework
    framework = ComprehensiveValidationFramework(
        strict_mode=False,
        save_reports=True
    )
    
    # Example data and context
    dummy_data = {
        "training_data": np.random.random((100, 150, 150, 3)),
        "labels": np.random.randint(0, 2, (100,)),
        "config": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 50,
            "validation_split": 0.2
        },
        "metrics": {
            "accuracy": 0.85,
            "val_accuracy": 0.82,
            "recall": 0.88,
            "precision": 0.83,
            "f1_score": 0.85,
            "loss": 0.3,
            "val_loss": 0.4
        }
    }
    
    context = {
        "domain": "medical",
        "data_type": "image",
        "encryption_enabled": True,
        "differential_privacy": False
    }
    
    # Run validation
    passed, results = framework.validate_all(dummy_data, context)
    summary = framework.get_validation_summary(results)
    
    print(f"Validation {'PASSED' if passed else 'FAILED'}")
    print(f"Summary: {summary}")
    print("Comprehensive Validation Framework test completed!")