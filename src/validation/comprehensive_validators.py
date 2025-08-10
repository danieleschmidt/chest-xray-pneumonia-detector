# Comprehensive Data and Model Validation Framework
# Implements robust validation for medical AI systems

import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import logging
from PIL import Image, ImageStat
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import hashlib
from datetime import datetime
import warnings


@dataclass
class ValidationResult:
    """Container for validation results."""
    is_valid: bool
    score: float
    issues: List[str]
    metadata: Dict[str, Any]
    timestamp: datetime


class ImageQualityValidator:
    """Validates image quality for medical AI applications."""
    
    def __init__(self, min_resolution: Tuple[int, int] = (128, 128),
                 max_resolution: Tuple[int, int] = (4096, 4096),
                 required_channels: int = 3):
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.required_channels = required_channels
        
    def validate_image(self, image_path: str) -> ValidationResult:
        """Comprehensive image quality validation."""
        issues = []
        metadata = {}
        
        try:
            # Load and analyze image
            with Image.open(image_path) as img:
                width, height = img.size
                mode = img.mode
                
                metadata.update({
                    'width': width,
                    'height': height,
                    'mode': mode,
                    'format': img.format,
                    'file_size': Path(image_path).stat().st_size
                })
                
                # Check resolution
                if width < self.min_resolution[0] or height < self.min_resolution[1]:
                    issues.append(f"Resolution too low: {width}x{height}")
                    
                if width > self.max_resolution[0] or height > self.max_resolution[1]:
                    issues.append(f"Resolution too high: {width}x{height}")
                
                # Check channels
                if mode == 'RGB' and self.required_channels == 3:
                    pass
                elif mode == 'L' and self.required_channels == 1:
                    pass
                else:
                    issues.append(f"Invalid channel configuration: {mode}")
                
                # Check for corruption
                img_array = np.array(img)
                if np.any(np.isnan(img_array)) or np.any(np.isinf(img_array)):
                    issues.append("Image contains NaN or infinite values")
                
                # Analyze image statistics
                stats = ImageStat.Stat(img)
                mean_brightness = np.mean(stats.mean)
                std_brightness = np.mean(stats.stddev)
                
                metadata.update({
                    'mean_brightness': mean_brightness,
                    'std_brightness': std_brightness
                })
                
                # Check for blank images
                if std_brightness < 1.0:
                    issues.append("Image appears to be blank or uniform")
                
                # Check for over/under exposure
                if mean_brightness < 10:
                    issues.append("Image appears underexposed")
                elif mean_brightness > 245:
                    issues.append("Image appears overexposed")
        
        except Exception as e:
            issues.append(f"Failed to validate image: {str(e)}")
            
        # Calculate quality score
        quality_score = max(0.0, 1.0 - len(issues) * 0.2)
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            score=quality_score,
            issues=issues,
            metadata=metadata,
            timestamp=datetime.now()
        )
    
    def validate_batch(self, image_paths: List[str]) -> Dict[str, ValidationResult]:
        """Validate multiple images in batch."""
        results = {}
        for path in image_paths:
            results[path] = self.validate_image(path)
        return results


class ModelArchitectureValidator:
    """Validates CNN model architecture for medical applications."""
    
    def __init__(self):
        self.recommended_layers = ['conv2d', 'maxpooling2d', 'dropout', 'dense']
        self.min_layers = 5
        self.max_layers = 100
        
    def validate_model(self, model: tf.keras.Model) -> ValidationResult:
        """Comprehensive model architecture validation."""
        issues = []
        metadata = {}
        
        try:
            # Basic model info
            total_layers = len(model.layers)
            trainable_params = model.count_params()
            
            metadata.update({
                'total_layers': total_layers,
                'trainable_params': trainable_params,
                'model_name': model.name,
                'input_shape': str(model.input_shape) if hasattr(model, 'input_shape') else 'Unknown'
            })
            
            # Check layer count
            if total_layers < self.min_layers:
                issues.append(f"Too few layers: {total_layers}")
            elif total_layers > self.max_layers:
                issues.append(f"Too many layers: {total_layers}")
            
            # Analyze layer types
            layer_types = {}
            for layer in model.layers:
                layer_type = layer.__class__.__name__.lower()
                layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
            
            metadata['layer_types'] = layer_types
            
            # Check for recommended layer types
            has_conv = any('conv' in layer_type for layer_type in layer_types.keys())
            has_pooling = any('pool' in layer_type for layer_type in layer_types.keys())
            has_dropout = 'dropout' in layer_types
            has_dense = 'dense' in layer_types
            
            if not has_conv:
                issues.append("Model lacks convolutional layers")
            if not has_pooling:
                issues.append("Model lacks pooling layers")
            if not has_dropout:
                issues.append("Model lacks dropout regularization")
            if not has_dense:
                issues.append("Model lacks dense classification layers")
            
            # Check parameter count
            if trainable_params < 1000:
                issues.append("Model has very few parameters - may underfit")
            elif trainable_params > 50_000_000:
                issues.append("Model has excessive parameters - may overfit")
            
            # Check for gradient flow issues
            try:
                # Test forward pass with dummy data
                if hasattr(model, 'input_shape') and model.input_shape:
                    dummy_input = tf.random.normal((1,) + model.input_shape[1:])
                    _ = model(dummy_input, training=False)
                else:
                    issues.append("Cannot determine model input shape")
            except Exception as e:
                issues.append(f"Forward pass failed: {str(e)}")
            
        except Exception as e:
            issues.append(f"Model validation failed: {str(e)}")
        
        # Calculate architecture score
        architecture_score = max(0.0, 1.0 - len(issues) * 0.15)
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            score=architecture_score,
            issues=issues,
            metadata=metadata,
            timestamp=datetime.now()
        )


class DatasetValidator:
    """Validates datasets for medical AI training."""
    
    def __init__(self):
        self.min_samples_per_class = 100
        self.max_imbalance_ratio = 10.0
        
    def validate_dataset(self, X: np.ndarray, y: np.ndarray,
                        dataset_name: str = "Unknown") -> ValidationResult:
        """Comprehensive dataset validation."""
        issues = []
        metadata = {}
        
        try:
            # Basic dataset info
            num_samples, *feature_shape = X.shape
            num_classes = len(np.unique(y))
            
            metadata.update({
                'dataset_name': dataset_name,
                'num_samples': num_samples,
                'feature_shape': feature_shape,
                'num_classes': num_classes,
                'data_type': str(X.dtype)
            })
            
            # Check sample count
            if num_samples < 100:
                issues.append(f"Too few samples: {num_samples}")
            
            # Analyze class distribution
            unique_classes, class_counts = np.unique(y, return_counts=True)
            class_distribution = dict(zip(unique_classes, class_counts))
            metadata['class_distribution'] = {str(k): int(v) for k, v in class_distribution.items()}
            
            # Check for minimum samples per class
            min_count = np.min(class_counts)
            max_count = np.max(class_counts)
            
            if min_count < self.min_samples_per_class:
                issues.append(f"Some classes have too few samples: {min_count}")
            
            # Check class imbalance
            imbalance_ratio = max_count / min_count
            metadata['imbalance_ratio'] = float(imbalance_ratio)
            
            if imbalance_ratio > self.max_imbalance_ratio:
                issues.append(f"Severe class imbalance: {imbalance_ratio:.2f}")
            
            # Check for data quality issues
            if np.any(np.isnan(X)):
                issues.append("Dataset contains NaN values")
            if np.any(np.isinf(X)):
                issues.append("Dataset contains infinite values")
            
            # Check data range
            x_min, x_max = X.min(), X.max()
            metadata.update({'data_min': float(x_min), 'data_max': float(x_max)})
            
            if x_min == x_max:
                issues.append("All feature values are identical")
            
            # Check for potential data leakage (duplicate samples)
            if len(X.shape) > 2:
                # For images, check for exact duplicates
                X_flat = X.reshape(num_samples, -1)
                unique_samples = np.unique(X_flat, axis=0)
                if len(unique_samples) < num_samples:
                    duplicate_count = num_samples - len(unique_samples)
                    issues.append(f"Found {duplicate_count} duplicate samples")
                    metadata['duplicate_samples'] = duplicate_count
            
        except Exception as e:
            issues.append(f"Dataset validation failed: {str(e)}")
        
        # Calculate dataset quality score
        quality_score = max(0.0, 1.0 - len(issues) * 0.1)
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            score=quality_score,
            issues=issues,
            metadata=metadata,
            timestamp=datetime.now()
        )


class ModelPerformanceValidator:
    """Validates model performance and detects potential issues."""
    
    def __init__(self):
        self.min_accuracy = 0.6
        self.min_precision = 0.7
        self.min_recall = 0.7
        self.min_f1 = 0.65
        
    def validate_performance(self, y_true: np.ndarray, y_pred: np.ndarray,
                           y_prob: Optional[np.ndarray] = None) -> ValidationResult:
        """Comprehensive performance validation."""
        issues = []
        metadata = {}
        
        try:
            # Calculate basic metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            metadata.update({
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            })
            
            # Check against thresholds
            if accuracy < self.min_accuracy:
                issues.append(f"Accuracy too low: {accuracy:.3f}")
            if precision < self.min_precision:
                issues.append(f"Precision too low: {precision:.3f}")
            if recall < self.min_recall:
                issues.append(f"Recall too low: {recall:.3f}")
            if f1 < self.min_f1:
                issues.append(f"F1-score too low: {f1:.3f}")
            
            # Check for potential overfitting indicators
            if accuracy > 0.99 and len(np.unique(y_true)) > 1:
                issues.append("Suspiciously high accuracy - potential overfitting")
            
            # Analyze prediction distribution
            pred_distribution = np.bincount(y_pred.astype(int))
            if len(pred_distribution) > 1:
                pred_entropy = -np.sum(pred_distribution / np.sum(pred_distribution) * 
                                     np.log2(pred_distribution / np.sum(pred_distribution) + 1e-8))
                metadata['prediction_entropy'] = float(pred_entropy)
                
                if pred_entropy < 0.1:
                    issues.append("Model predictions lack diversity")
            
            # Analyze probability calibration if available
            if y_prob is not None:
                # Check for overconfident predictions
                max_probs = np.max(y_prob, axis=1) if y_prob.ndim > 1 else y_prob
                avg_confidence = np.mean(max_probs)
                metadata['average_confidence'] = float(avg_confidence)
                
                if avg_confidence > 0.95:
                    issues.append("Model appears overconfident")
                elif avg_confidence < 0.6:
                    issues.append("Model appears underconfident")
            
        except Exception as e:
            issues.append(f"Performance validation failed: {str(e)}")
        
        # Calculate performance score
        if 'accuracy' in metadata and 'f1_score' in metadata:
            performance_score = (metadata['accuracy'] + metadata['f1_score']) / 2
        else:
            performance_score = 0.0
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            score=performance_score,
            issues=issues,
            metadata=metadata,
            timestamp=datetime.now()
        )


class ComprehensiveValidationSuite:
    """Complete validation suite for medical AI systems."""
    
    def __init__(self):
        self.image_validator = ImageQualityValidator()
        self.model_validator = ModelArchitectureValidator()
        self.dataset_validator = DatasetValidator()
        self.performance_validator = ModelPerformanceValidator()
        self.validation_history = []
        
    def run_full_validation(self,
                          model: tf.keras.Model,
                          train_data: Tuple[np.ndarray, np.ndarray],
                          test_data: Tuple[np.ndarray, np.ndarray],
                          predictions: Optional[np.ndarray] = None,
                          image_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run comprehensive validation suite."""
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'model_validation': None,
            'dataset_validation': None,
            'performance_validation': None,
            'image_validation': None,
            'overall_score': 0.0,
            'pass_criteria': False
        }
        
        scores = []
        all_issues = []
        
        # Validate model architecture
        try:
            model_result = self.model_validator.validate_model(model)
            validation_results['model_validation'] = {
                'is_valid': model_result.is_valid,
                'score': model_result.score,
                'issues': model_result.issues,
                'metadata': model_result.metadata
            }
            scores.append(model_result.score)
            all_issues.extend([f"Model: {issue}" for issue in model_result.issues])
        except Exception as e:
            validation_results['model_validation'] = {'error': str(e)}
        
        # Validate training dataset
        try:
            X_train, y_train = train_data
            dataset_result = self.dataset_validator.validate_dataset(X_train, y_train, "training")
            validation_results['dataset_validation'] = {
                'is_valid': dataset_result.is_valid,
                'score': dataset_result.score,
                'issues': dataset_result.issues,
                'metadata': dataset_result.metadata
            }
            scores.append(dataset_result.score)
            all_issues.extend([f"Dataset: {issue}" for issue in dataset_result.issues])
        except Exception as e:
            validation_results['dataset_validation'] = {'error': str(e)}
        
        # Validate model performance
        if predictions is not None:
            try:
                X_test, y_test = test_data
                # Convert predictions to class labels if needed
                if predictions.ndim > 1 and predictions.shape[1] > 1:
                    y_pred = np.argmax(predictions, axis=1)
                    y_prob = predictions
                else:
                    y_pred = (predictions.flatten() > 0.5).astype(int)
                    y_prob = predictions
                
                perf_result = self.performance_validator.validate_performance(
                    y_test, y_pred, y_prob
                )
                validation_results['performance_validation'] = {
                    'is_valid': perf_result.is_valid,
                    'score': perf_result.score,
                    'issues': perf_result.issues,
                    'metadata': perf_result.metadata
                }
                scores.append(perf_result.score)
                all_issues.extend([f"Performance: {issue}" for issue in perf_result.issues])
            except Exception as e:
                validation_results['performance_validation'] = {'error': str(e)}
        
        # Validate images if provided
        if image_paths:
            try:
                image_results = self.image_validator.validate_batch(image_paths[:10])  # Sample validation
                valid_images = sum(1 for result in image_results.values() if result.is_valid)
                avg_score = np.mean([result.score for result in image_results.values()])
                
                validation_results['image_validation'] = {
                    'total_images': len(image_paths),
                    'sampled_images': len(image_results),
                    'valid_images': valid_images,
                    'average_score': float(avg_score),
                    'sample_results': {path: {
                        'is_valid': result.is_valid,
                        'score': result.score,
                        'issues': result.issues
                    } for path, result in list(image_results.items())[:5]}
                }
                scores.append(avg_score)
            except Exception as e:
                validation_results['image_validation'] = {'error': str(e)}
        
        # Calculate overall score
        if scores:
            validation_results['overall_score'] = float(np.mean(scores))
            validation_results['pass_criteria'] = (
                validation_results['overall_score'] > 0.7 and 
                len(all_issues) == 0
            )
        
        validation_results['all_issues'] = all_issues
        validation_results['total_issues'] = len(all_issues)
        
        # Store in history
        self.validation_history.append(validation_results)
        
        return validation_results
    
    def generate_validation_report(self, results: Dict[str, Any], 
                                 output_path: str = "validation_report.json"):
        """Generate comprehensive validation report."""
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate summary
        summary = {
            'validation_passed': results['pass_criteria'],
            'overall_score': results['overall_score'],
            'total_issues': results['total_issues'],
            'critical_issues': [issue for issue in results.get('all_issues', []) 
                              if any(word in issue.lower() for word in ['failed', 'error', 'critical'])],
            'recommendations': self._generate_recommendations(results)
        }
        
        summary_path = output_path.replace('.json', '_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        # Model recommendations
        if results.get('model_validation', {}).get('score', 1.0) < 0.8:
            recommendations.append("Consider improving model architecture")
        
        # Dataset recommendations  
        dataset_meta = results.get('dataset_validation', {}).get('metadata', {})
        if dataset_meta.get('imbalance_ratio', 1.0) > 5.0:
            recommendations.append("Address class imbalance with resampling or weighted loss")
        
        # Performance recommendations
        perf_meta = results.get('performance_validation', {}).get('metadata', {})
        if perf_meta.get('accuracy', 1.0) < 0.8:
            recommendations.append("Improve model performance through hyperparameter tuning")
        
        # General recommendations
        if results['total_issues'] > 5:
            recommendations.append("Address critical validation issues before deployment")
        
        return recommendations


if __name__ == "__main__":
    # Demonstration of comprehensive validation
    
    print("Initializing comprehensive validation suite...")
    validator = ComprehensiveValidationSuite()
    
    # Create dummy data for demonstration
    X_train = np.random.randn(1000, 224, 224, 3)
    y_train = np.random.randint(0, 2, 1000)
    X_test = np.random.randn(200, 224, 224, 3)
    y_test = np.random.randint(0, 2, 200)
    
    # Create dummy model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Generate dummy predictions
    predictions = np.random.rand(200, 1)
    
    # Run validation
    results = validator.run_full_validation(
        model=model,
        train_data=(X_train, y_train),
        test_data=(X_test, y_test),
        predictions=predictions
    )
    
    # Generate report
    summary = validator.generate_validation_report(results, "demo_validation_report.json")
    
    print(f"Validation completed!")
    print(f"Overall score: {results['overall_score']:.3f}")
    print(f"Pass criteria: {results['pass_criteria']}")
    print(f"Total issues: {results['total_issues']}")