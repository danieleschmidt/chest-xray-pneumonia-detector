"""
Mock models and components for testing.

This module provides mock implementations of ML models and components
to enable fast, reliable testing without actual model training.
"""

import numpy as np
from unittest.mock import MagicMock, Mock
import tensorflow as tf
from typing import Tuple, Optional, Dict, Any


class MockModel:
    """Mock TensorFlow/Keras model for testing."""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=1):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weights = None
        self.compiled = False
        self.trained = False
        
    def compile(self, optimizer=None, loss=None, metrics=None, **kwargs):
        """Mock model compilation."""
        self.compiled = True
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics or []
        
    def fit(self, x=None, y=None, batch_size=32, epochs=1, 
            validation_data=None, callbacks=None, **kwargs):
        """Mock model training."""
        self.trained = True
        
        # Create mock history
        history = Mock()
        history.history = {
            'loss': np.random.random(epochs).tolist(),
            'accuracy': np.random.uniform(0.8, 0.95, epochs).tolist(),
            'val_loss': np.random.random(epochs).tolist(),
            'val_accuracy': np.random.uniform(0.75, 0.9, epochs).tolist()
        }
        return history
        
    def predict(self, x, batch_size=32, **kwargs):
        """Mock model prediction."""
        if isinstance(x, np.ndarray):
            batch_size = len(x)
        elif hasattr(x, '__len__'):
            batch_size = len(x)
            
        if self.num_classes == 1:
            # Binary classification
            return np.random.uniform(0, 1, (batch_size, 1))
        else:
            # Multi-class classification
            return np.random.uniform(0, 1, (batch_size, self.num_classes))
            
    def evaluate(self, x=None, y=None, batch_size=32, **kwargs):
        """Mock model evaluation."""
        return [0.1, 0.85]  # [loss, accuracy]
        
    def save(self, filepath, **kwargs):
        """Mock model saving."""
        self.saved_path = filepath
        
    def load_weights(self, filepath, **kwargs):
        """Mock weight loading."""
        self.weights_path = filepath
        
    def summary(self):
        """Mock model summary."""
        return f"Mock Model Summary\nInput shape: {self.input_shape}\nOutput classes: {self.num_classes}"


class MockDataGenerator:
    """Mock data generator for testing."""
    
    def __init__(self, data_dir, batch_size=32, target_size=(224, 224), 
                 class_mode='binary', **kwargs):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.class_mode = class_mode
        self.samples = 100  # Mock sample count
        self.num_classes = 2 if class_mode == 'binary' else 10
        
    def __iter__(self):
        """Make generator iterable."""
        return self
        
    def __next__(self):
        """Generate mock batch."""
        batch_images = np.random.random((self.batch_size, *self.target_size, 3))
        
        if self.class_mode == 'binary':
            batch_labels = np.random.randint(0, 2, (self.batch_size, 1))
        else:
            batch_labels = np.random.randint(0, self.num_classes, (self.batch_size,))
            
        return batch_images, batch_labels
        
    def __len__(self):
        """Return number of batches."""
        return self.samples // self.batch_size


class MockMLflowLogger:
    """Mock MLflow logger for testing."""
    
    def __init__(self):
        self.logged_params = {}
        self.logged_metrics = {}
        self.logged_artifacts = []
        self.active_run = True
        
    def log_param(self, key, value):
        """Mock parameter logging."""
        self.logged_params[key] = value
        
    def log_metric(self, key, value, step=None):
        """Mock metric logging."""
        if key not in self.logged_metrics:
            self.logged_metrics[key] = []
        self.logged_metrics[key].append((value, step))
        
    def log_artifact(self, local_path, artifact_path=None):
        """Mock artifact logging."""
        self.logged_artifacts.append((local_path, artifact_path))
        
    def start_run(self, run_name=None, experiment_id=None):
        """Mock run start."""
        self.active_run = True
        self.run_name = run_name
        return Mock(info=Mock(run_id="mock_run_id"))
        
    def end_run(self):
        """Mock run end."""
        self.active_run = False


class MockImageProcessor:
    """Mock image processor for testing."""
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        
    def preprocess_image(self, image_path):
        """Mock image preprocessing."""
        return np.random.random((*self.target_size, 3))
        
    def augment_image(self, image):
        """Mock image augmentation."""
        return image + np.random.normal(0, 0.1, image.shape)
        
    def normalize_image(self, image):
        """Mock image normalization."""
        return (image - 0.5) / 0.5


def create_mock_training_environment():
    """Create complete mock training environment."""
    return {
        'model': MockModel(),
        'train_generator': MockDataGenerator('/mock/train'),
        'val_generator': MockDataGenerator('/mock/val'),
        'logger': MockMLflowLogger(),
        'processor': MockImageProcessor()
    }


def create_mock_inference_environment():
    """Create mock inference environment."""
    model = MockModel()
    model.trained = True
    
    return {
        'model': model,
        'processor': MockImageProcessor()
    }


# Pytest fixtures for easy use in tests
import pytest

@pytest.fixture
def mock_model():
    """Fixture providing mock model."""
    return MockModel()

@pytest.fixture
def mock_data_generator():
    """Fixture providing mock data generator."""
    return MockDataGenerator('/mock/data')

@pytest.fixture
def mock_mlflow_logger():
    """Fixture providing mock MLflow logger."""
    return MockMLflowLogger()

@pytest.fixture
def mock_training_env():
    """Fixture providing complete mock training environment."""
    return create_mock_training_environment()

@pytest.fixture
def mock_inference_env():
    """Fixture providing mock inference environment."""
    return create_mock_inference_environment()