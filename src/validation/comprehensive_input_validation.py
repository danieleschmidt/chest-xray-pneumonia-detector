"""
Comprehensive input validation system for medical AI pipeline.
Provides multi-layered validation with medical data compliance.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import mimetypes
import hashlib
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime, date
import json
import cv2
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    MEDICAL_GRADE = "medical_grade"


class DataType(Enum):
    """Supported data types for validation."""
    IMAGE = "image"
    MEDICAL_IMAGE = "medical_image"
    PATIENT_DATA = "patient_data"
    MODEL_PARAMETERS = "model_parameters"
    TRAINING_DATA = "training_data"
    PREDICTION_INPUT = "prediction_input"


@dataclass
class ValidationResult:
    """Result of validation operation."""
    is_valid: bool
    data_type: DataType
    validation_level: ValidationLevel
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    sanitized_data: Optional[Any] = None
    confidence_score: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MedicalImageValidator:
    """Specialized validator for medical images."""
    
    # Medical image standards
    DICOM_EXTENSIONS = {'.dcm', '.dicom'}
    MEDICAL_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.dcm', '.dicom'}
    
    # Medical image size constraints
    MIN_RESOLUTION = (64, 64)
    MAX_RESOLUTION = (8192, 8192)
    MIN_FILE_SIZE = 1024  # 1KB
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    
    # Medical imaging modalities
    VALID_MODALITIES = {
        'X-RAY', 'XRAY', 'CR', 'DX',  # Chest X-rays
        'CT', 'COMPUTED_TOMOGRAPHY',
        'MRI', 'MAGNETIC_RESONANCE',
        'US', 'ULTRASOUND',
        'MG', 'MAMMOGRAPHY'
    }
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
    
    def validate_medical_image(self, image_path: str, 
                             metadata: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Comprehensive medical image validation."""
        result = ValidationResult(
            is_valid=True,
            data_type=DataType.MEDICAL_IMAGE,
            validation_level=self.validation_level
        )
        
        try:
            path = Path(image_path)
            
            # Basic file existence check
            if not path.exists():
                result.is_valid = False
                result.errors.append(f"File does not exist: {image_path}")
                return result
            
            # File extension validation
            if path.suffix.lower() not in self.MEDICAL_IMAGE_EXTENSIONS:
                result.warnings.append(f"Uncommon medical image extension: {path.suffix}")
            
            # File size validation
            file_size = path.stat().st_size
            if file_size < self.MIN_FILE_SIZE:
                result.is_valid = False
                result.errors.append(f"File too small: {file_size} bytes")
            elif file_size > self.MAX_FILE_SIZE:
                result.is_valid = False
                result.errors.append(f"File too large: {file_size} bytes")
            
            # MIME type validation
            mime_type, _ = mimetypes.guess_type(str(path))
            if mime_type and not mime_type.startswith('image/'):
                result.is_valid = False
                result.errors.append(f"Invalid MIME type: {mime_type}")
            
            # Image content validation
            try:
                if path.suffix.lower() in self.DICOM_EXTENSIONS:
                    # DICOM validation (would require pydicom)
                    result.warnings.append("DICOM validation requires pydicom library")
                else:
                    # Standard image validation
                    self._validate_standard_image(path, result)
                
            except Exception as img_error:
                result.is_valid = False
                result.errors.append(f"Image validation failed: {img_error}")
            
            # Medical metadata validation
            if metadata:
                self._validate_medical_metadata(metadata, result)
            
            # Advanced validations for medical grade
            if self.validation_level == ValidationLevel.MEDICAL_GRADE:
                self._perform_medical_grade_validation(path, result)
            
            # Calculate confidence score
            result.confidence_score = self._calculate_confidence_score(result)
            
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Validation error: {e}")
            logger.error(f"Medical image validation failed: {e}")
        
        return result
    
    def _validate_standard_image(self, path: Path, result: ValidationResult):
        """Validate standard image formats."""
        with Image.open(path) as img:
            # Verify image integrity
            img.verify()
            
        # Re-open for analysis (verify() closes the image)
        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode
            format_name = img.format
            
            result.metadata.update({
                'width': width,
                'height': height,
                'mode': mode,
                'format': format_name,
                'file_size': path.stat().st_size
            })
            
            # Resolution validation
            if width < self.MIN_RESOLUTION[0] or height < self.MIN_RESOLUTION[1]:
                result.errors.append(f"Resolution too low: {width}x{height}")
            
            if width > self.MAX_RESOLUTION[0] or height > self.MAX_RESOLUTION[1]:
                result.errors.append(f"Resolution too high: {width}x{height}")
            
            # Color mode validation for medical images
            if mode not in ['L', 'RGB', 'RGBA']:
                result.warnings.append(f"Unusual color mode for medical image: {mode}")
            
            # Aspect ratio check
            aspect_ratio = width / height
            if aspect_ratio < 0.1 or aspect_ratio > 10:
                result.warnings.append(f"Unusual aspect ratio: {aspect_ratio:.2f}")
    
    def _validate_medical_metadata(self, metadata: Dict[str, Any], result: ValidationResult):
        """Validate medical-specific metadata."""
        # Modality validation
        if 'modality' in metadata:
            modality = str(metadata['modality']).upper()
            if modality not in self.VALID_MODALITIES:
                result.warnings.append(f"Unknown modality: {modality}")
        
        # Patient age validation
        if 'patient_age' in metadata:
            age = metadata['patient_age']
            try:
                age_int = int(age)
                if not (0 <= age_int <= 150):
                    result.errors.append(f"Invalid patient age: {age}")
            except (ValueError, TypeError):
                result.errors.append(f"Invalid age format: {age}")
        
        # Study date validation
        if 'study_date' in metadata:
            try:
                study_date = pd.to_datetime(metadata['study_date'])
                if study_date > pd.Timestamp.now():
                    result.errors.append("Study date cannot be in the future")
                if study_date < pd.Timestamp('1900-01-01'):
                    result.errors.append("Study date too old")
            except Exception:
                result.errors.append(f"Invalid study date: {metadata['study_date']}")
    
    def _perform_medical_grade_validation(self, path: Path, result: ValidationResult):
        """Perform medical-grade validation checks."""
        try:
            # Pixel data analysis
            img_array = np.array(Image.open(path))
            
            # Check for corrupted pixels
            if np.any(np.isnan(img_array)) or np.any(np.isinf(img_array)):
                result.errors.append("Image contains invalid pixel values")
            
            # Dynamic range analysis
            if img_array.dtype == np.uint8:
                pixel_range = (img_array.min(), img_array.max())
                if pixel_range[1] - pixel_range[0] < 50:  # Low dynamic range
                    result.warnings.append(f"Low dynamic range: {pixel_range}")
            
            # Noise analysis (simplified)
            if len(img_array.shape) == 2:  # Grayscale
                # Calculate local variance as noise estimate
                kernel = np.ones((3, 3)) / 9
                local_mean = cv2.filter2D(img_array.astype(np.float32), -1, kernel)
                local_var = cv2.filter2D((img_array.astype(np.float32) - local_mean)**2, -1, kernel)
                avg_noise = np.mean(local_var)
                
                result.metadata['estimated_noise_level'] = float(avg_noise)
                
                if avg_noise > 1000:  # High noise threshold
                    result.warnings.append(f"High noise level detected: {avg_noise:.2f}")
            
            # Check for potential artifacts
            # (This is a simplified check - real medical validation would be more complex)
            height, width = img_array.shape[:2]
            
            # Check edges for potential cropping artifacts
            edge_pixels = np.concatenate([
                img_array[0, :].flatten(),   # Top edge
                img_array[-1, :].flatten(),  # Bottom edge
                img_array[:, 0].flatten(),   # Left edge
                img_array[:, -1].flatten()   # Right edge
            ])
            
            if len(np.unique(edge_pixels)) < 5:  # Very uniform edges
                result.warnings.append("Potential cropping artifacts detected")
            
        except Exception as e:
            result.warnings.append(f"Advanced validation failed: {e}")

    def _calculate_confidence_score(self, result: ValidationResult) -> float:
        """Calculate confidence score based on validation results."""
        base_score = 1.0
        
        # Deduct for errors and warnings
        base_score -= len(result.errors) * 0.3
        base_score -= len(result.warnings) * 0.1
        
        # Bonus for complete metadata
        if len(result.metadata) > 5:
            base_score += 0.1
        
        return max(0.0, min(1.0, base_score))


class ParameterValidator:
    """Validator for model parameters and training configurations."""
    
    # Parameter ranges for common ML parameters
    PARAMETER_RANGES = {
        'learning_rate': (1e-6, 1.0),
        'batch_size': (1, 1024),
        'epochs': (1, 1000),
        'dropout_rate': (0.0, 0.95),
        'regularization': (0.0, 1.0),
        'momentum': (0.0, 1.0),
        'beta_1': (0.0, 1.0),
        'beta_2': (0.0, 1.0),
        'epsilon': (1e-10, 1e-3)
    }
    
    # Valid choices for categorical parameters
    VALID_CHOICES = {
        'optimizer': ['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta'],
        'activation': ['relu', 'sigmoid', 'tanh', 'softmax', 'linear', 'elu', 'selu'],
        'loss': ['binary_crossentropy', 'categorical_crossentropy', 'mse', 'mae'],
        'metrics': ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    }
    
    def validate_parameters(self, parameters: Dict[str, Any], 
                          validation_level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
        """Validate model and training parameters."""
        result = ValidationResult(
            is_valid=True,
            data_type=DataType.MODEL_PARAMETERS,
            validation_level=validation_level
        )
        
        sanitized_params = {}
        
        for param_name, param_value in parameters.items():
            try:
                # Range validation
                if param_name in self.PARAMETER_RANGES:
                    min_val, max_val = self.PARAMETER_RANGES[param_name]
                    
                    try:
                        numeric_value = float(param_value)
                        if not (min_val <= numeric_value <= max_val):
                            result.errors.append(
                                f"{param_name} ({numeric_value}) outside valid range [{min_val}, {max_val}]"
                            )
                        else:
                            sanitized_params[param_name] = numeric_value
                    except (ValueError, TypeError):
                        result.errors.append(f"Invalid numeric value for {param_name}: {param_value}")
                
                # Choice validation
                elif param_name in self.VALID_CHOICES:
                    valid_options = self.VALID_CHOICES[param_name]
                    param_str = str(param_value).lower()
                    
                    if param_str not in valid_options:
                        result.errors.append(
                            f"{param_name} ({param_value}) not in valid options: {valid_options}"
                        )
                    else:
                        sanitized_params[param_name] = param_str
                
                # Special validations
                elif param_name == 'img_size':
                    self._validate_image_size(param_value, result, sanitized_params)
                
                elif param_name == 'class_weights':
                    self._validate_class_weights(param_value, result, sanitized_params)
                
                else:
                    # Unknown parameter - add to sanitized but warn
                    sanitized_params[param_name] = param_value
                    if validation_level in [ValidationLevel.STRICT, ValidationLevel.MEDICAL_GRADE]:
                        result.warnings.append(f"Unknown parameter: {param_name}")
            
            except Exception as e:
                result.errors.append(f"Validation error for {param_name}: {e}")
        
        # Parameter compatibility checks
        self._check_parameter_compatibility(sanitized_params, result)
        
        result.sanitized_data = sanitized_params
        result.is_valid = len(result.errors) == 0
        
        return result
    
    def _validate_image_size(self, img_size, result: ValidationResult, sanitized_params: Dict[str, Any]):
        """Validate image size parameter."""
        try:
            if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
                width, height = map(int, img_size)
                if width < 32 or height < 32:
                    result.errors.append(f"Image size too small: {width}x{height}")
                elif width > 2048 or height > 2048:
                    result.warnings.append(f"Very large image size: {width}x{height}")
                else:
                    sanitized_params['img_size'] = (width, height)
            else:
                result.errors.append(f"Invalid image size format: {img_size}")
        except (ValueError, TypeError):
            result.errors.append(f"Invalid image size values: {img_size}")
    
    def _validate_class_weights(self, class_weights, result: ValidationResult, sanitized_params: Dict[str, Any]):
        """Validate class weights parameter."""
        try:
            if isinstance(class_weights, (list, tuple)):
                weights = [float(w) for w in class_weights]
                if any(w < 0 for w in weights):
                    result.errors.append("Class weights cannot be negative")
                elif all(w == 0 for w in weights):
                    result.errors.append("All class weights cannot be zero")
                else:
                    sanitized_params['class_weights'] = weights
            elif isinstance(class_weights, dict):
                weights = {k: float(v) for k, v in class_weights.items()}
                if any(w < 0 for w in weights.values()):
                    result.errors.append("Class weights cannot be negative")
                else:
                    sanitized_params['class_weights'] = weights
            else:
                result.errors.append(f"Invalid class weights format: {type(class_weights)}")
        except (ValueError, TypeError) as e:
            result.errors.append(f"Invalid class weights values: {e}")
    
    def _check_parameter_compatibility(self, params: Dict[str, Any], result: ValidationResult):
        """Check for parameter compatibility issues."""
        # Check optimizer-specific parameters
        if 'optimizer' in params:
            optimizer = params['optimizer']
            
            if optimizer == 'sgd' and 'momentum' not in params:
                result.warnings.append("SGD optimizer typically benefits from momentum parameter")
            
            if optimizer == 'adam':
                if 'beta_1' in params and params['beta_1'] >= 1.0:
                    result.errors.append("Adam beta_1 should be < 1.0")
                if 'beta_2' in params and params['beta_2'] >= 1.0:
                    result.errors.append("Adam beta_2 should be < 1.0")
        
        # Check batch size vs learning rate
        if 'batch_size' in params and 'learning_rate' in params:
            batch_size = params['batch_size']
            learning_rate = params['learning_rate']
            
            # Rule of thumb: larger batch sizes often need higher learning rates
            if batch_size > 32 and learning_rate < 1e-4:
                result.warnings.append(
                    f"Large batch size ({batch_size}) with small learning rate ({learning_rate}) "
                    "may slow convergence"
                )


class ComprehensiveInputValidator:
    """Main validator orchestrating all validation types."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.image_validator = MedicalImageValidator(validation_level)
        self.parameter_validator = ParameterValidator()
        
        # Validation statistics
        self.validation_count = 0
        self.validation_history: List[ValidationResult] = []
    
    def validate(self, data: Any, data_type: DataType, 
                metadata: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Main validation entry point."""
        self.validation_count += 1
        
        try:
            if data_type == DataType.MEDICAL_IMAGE:
                result = self.image_validator.validate_medical_image(data, metadata)
            
            elif data_type == DataType.MODEL_PARAMETERS:
                result = self.parameter_validator.validate_parameters(data, self.validation_level)
            
            elif data_type == DataType.PATIENT_DATA:
                result = self._validate_patient_data(data)
            
            elif data_type == DataType.TRAINING_DATA:
                result = self._validate_training_data(data)
            
            elif data_type == DataType.PREDICTION_INPUT:
                result = self._validate_prediction_input(data)
            
            else:
                result = ValidationResult(
                    is_valid=False,
                    data_type=data_type,
                    validation_level=self.validation_level,
                    errors=[f"Unsupported data type: {data_type}"]
                )
            
            # Store validation history
            self.validation_history.append(result)
            if len(self.validation_history) > 1000:  # Keep last 1000 validations
                self.validation_history = self.validation_history[-1000:]
            
            return result
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                data_type=data_type,
                validation_level=self.validation_level,
                errors=[f"Validation error: {e}"]
            )
    
    def _validate_patient_data(self, patient_data: Dict[str, Any]) -> ValidationResult:
        """Validate patient data for HIPAA compliance."""
        result = ValidationResult(
            is_valid=True,
            data_type=DataType.PATIENT_DATA,
            validation_level=self.validation_level
        )
        
        sanitized_data = {}
        
        # Required fields for medical AI
        required_fields = ['patient_id'] if self.validation_level != ValidationLevel.MEDICAL_GRADE else [
            'patient_id', 'study_date', 'modality'
        ]
        
        for field in required_fields:
            if field not in patient_data:
                result.errors.append(f"Missing required field: {field}")
        
        # Validate individual fields
        for field, value in patient_data.items():
            if field == 'patient_id':
                # Hash patient ID for privacy
                if not re.match(r'^[A-Za-z0-9_-]+$', str(value)):
                    result.errors.append(f"Invalid patient ID format: {value}")
                else:
                    sanitized_data['patient_id_hash'] = hashlib.sha256(
                        str(value).encode('utf-8')
                    ).hexdigest()[:16]
            
            elif field == 'age':
                try:
                    age = int(value)
                    if not (0 <= age <= 150):
                        result.errors.append(f"Invalid age: {age}")
                    else:
                        sanitized_data['age'] = age
                except (ValueError, TypeError):
                    result.errors.append(f"Invalid age format: {value}")
            
            elif field == 'sex':
                valid_values = {'M', 'F', 'O', 'U'}  # Male, Female, Other, Unknown
                if str(value).upper() not in valid_values:
                    result.errors.append(f"Invalid sex value: {value}")
                else:
                    sanitized_data['sex'] = str(value).upper()
            
            else:
                sanitized_data[field] = value
        
        result.sanitized_data = sanitized_data
        result.is_valid = len(result.errors) == 0
        
        return result
    
    def _validate_training_data(self, training_data: Dict[str, Any]) -> ValidationResult:
        """Validate training data configuration."""
        result = ValidationResult(
            is_valid=True,
            data_type=DataType.TRAINING_DATA,
            validation_level=self.validation_level
        )
        
        # Check required paths
        required_paths = ['train_dir', 'val_dir']
        for path_key in required_paths:
            if path_key not in training_data:
                result.errors.append(f"Missing required path: {path_key}")
            else:
                path = Path(training_data[path_key])
                if not path.exists():
                    result.errors.append(f"Path does not exist: {path}")
                elif not path.is_dir():
                    result.errors.append(f"Path is not a directory: {path}")
        
        # Validate data balance
        if 'train_dir' in training_data and 'val_dir' in training_data:
            try:
                train_path = Path(training_data['train_dir'])
                val_path = Path(training_data['val_dir'])
                
                # Count images in each class
                train_counts = self._count_images_by_class(train_path)
                val_counts = self._count_images_by_class(val_path)
                
                result.metadata.update({
                    'train_counts': train_counts,
                    'val_counts': val_counts
                })
                
                # Check for class imbalance
                if train_counts:
                    max_count = max(train_counts.values())
                    min_count = min(train_counts.values())
                    imbalance_ratio = max_count / max(min_count, 1)
                    
                    if imbalance_ratio > 10:
                        result.warnings.append(
                            f"High class imbalance detected: {imbalance_ratio:.2f}:1"
                        )
                
            except Exception as e:
                result.warnings.append(f"Could not analyze data balance: {e}")
        
        result.is_valid = len(result.errors) == 0
        return result
    
    def _validate_prediction_input(self, prediction_input: Any) -> ValidationResult:
        """Validate input for prediction."""
        result = ValidationResult(
            is_valid=True,
            data_type=DataType.PREDICTION_INPUT,
            validation_level=self.validation_level
        )
        
        # Handle different input types
        if isinstance(prediction_input, str):  # File path
            path = Path(prediction_input)
            if not path.exists():
                result.errors.append(f"Input file does not exist: {prediction_input}")
            else:
                # Validate as medical image
                img_result = self.image_validator.validate_medical_image(prediction_input)
                result.errors.extend(img_result.errors)
                result.warnings.extend(img_result.warnings)
                result.metadata.update(img_result.metadata)
        
        elif isinstance(prediction_input, np.ndarray):  # Numpy array
            if prediction_input.ndim not in [2, 3, 4]:
                result.errors.append(f"Invalid array dimensions: {prediction_input.ndim}")
            
            if prediction_input.size == 0:
                result.errors.append("Empty array provided")
            
            # Check for invalid values
            if np.any(np.isnan(prediction_input)) or np.any(np.isinf(prediction_input)):
                result.errors.append("Array contains invalid values (NaN or Inf)")
        
        else:
            result.errors.append(f"Unsupported input type: {type(prediction_input)}")
        
        result.is_valid = len(result.errors) == 0
        return result
    
    def _count_images_by_class(self, data_dir: Path) -> Dict[str, int]:
        """Count images in each class subdirectory."""
        counts = {}
        
        if not data_dir.exists():
            return counts
        
        for class_dir in data_dir.iterdir():
            if class_dir.is_dir():
                image_count = len(list(class_dir.glob('*.jpg'))) + \
                             len(list(class_dir.glob('*.jpeg'))) + \
                             len(list(class_dir.glob('*.png')))
                counts[class_dir.name] = image_count
        
        return counts
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics."""
        if not self.validation_history:
            return {'message': 'No validations performed'}
        
        recent_validations = self.validation_history[-100:]  # Last 100
        
        success_rate = sum(1 for v in recent_validations if v.is_valid) / len(recent_validations)
        
        error_counts = defaultdict(int)
        data_type_counts = defaultdict(int)
        
        for validation in recent_validations:
            data_type_counts[validation.data_type.value] += 1
            for error in validation.errors:
                error_counts[error] += 1
        
        return {
            'total_validations': self.validation_count,
            'recent_success_rate': success_rate * 100,
            'data_type_distribution': dict(data_type_counts),
            'common_errors': sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            'validation_level': self.validation_level.value
        }
    
    def export_validation_report(self, file_path: str):
        """Export comprehensive validation report."""
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'validation_level': self.validation_level.value,
            'statistics': self.get_validation_statistics(),
            'recent_validations': [
                {
                    'timestamp': v.timestamp.isoformat(),
                    'data_type': v.data_type.value,
                    'is_valid': v.is_valid,
                    'error_count': len(v.errors),
                    'warning_count': len(v.warnings),
                    'confidence_score': v.confidence_score
                }
                for v in self.validation_history[-50:]  # Last 50
            ]
        }
        
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Validation report exported to {file_path}")


def create_validator(validation_level: ValidationLevel = ValidationLevel.STANDARD) -> ComprehensiveInputValidator:
    """Factory function to create a configured validator."""
    return ComprehensiveInputValidator(validation_level)


if __name__ == '__main__':
    # Example usage
    validator = create_validator(ValidationLevel.MEDICAL_GRADE)
    
    # Test parameter validation
    test_params = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 10,
        'optimizer': 'adam',
        'img_size': (224, 224)
    }
    
    result = validator.validate(test_params, DataType.MODEL_PARAMETERS)
    print(f"Parameter validation: {result.is_valid}")
    if result.errors:
        print(f"Errors: {result.errors}")
    
    # Get statistics
    stats = validator.get_validation_statistics()
    print(f"Validation statistics: {stats}")
    
    # Export report
    validator.export_validation_report('/tmp/validation_report.json')
