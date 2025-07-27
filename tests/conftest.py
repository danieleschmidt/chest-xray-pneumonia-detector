"""
Pytest configuration and shared fixtures for the Chest X-Ray Pneumonia Detector tests.
"""

import os
import tempfile
from pathlib import Path
from typing import Generator, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a sample medical image for testing."""
    # Create a 224x224 grayscale image with some realistic patterns
    image = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
    
    # Add some patterns that might resemble lung structures
    center_x, center_y = 112, 112
    radius = 80
    y, x = np.ogrid[:224, :224]
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    image[mask] = np.random.randint(100, 200, np.sum(mask))
    
    return image


@pytest.fixture
def sample_rgb_image() -> np.ndarray:
    """Create a sample RGB medical image for testing."""
    # Create a 224x224x3 RGB image
    return np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_path(tmp_path: Path, sample_image: np.ndarray) -> Path:
    """Save sample image to a temporary file and return path."""
    image_path = tmp_path / "test_image.jpg"
    pil_image = Image.fromarray(sample_image, mode='L')
    pil_image.save(image_path)
    return image_path


@pytest.fixture
def sample_dataset_structure(tmp_path: Path) -> Tuple[Path, Path, Path]:
    """Create a sample dataset directory structure for testing."""
    # Create directory structure
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    test_dir = tmp_path / "test"
    
    for split_dir in [train_dir, val_dir, test_dir]:
        normal_dir = split_dir / "NORMAL"
        pneumonia_dir = split_dir / "PNEUMONIA"
        normal_dir.mkdir(parents=True)
        pneumonia_dir.mkdir(parents=True)
        
        # Create sample images
        for i in range(5):
            # Normal images
            normal_image = np.random.randint(50, 150, (224, 224), dtype=np.uint8)
            normal_path = normal_dir / f"normal_{i}.jpg"
            Image.fromarray(normal_image, mode='L').save(normal_path)
            
            # Pneumonia images
            pneumonia_image = np.random.randint(30, 120, (224, 224), dtype=np.uint8)
            pneumonia_path = pneumonia_dir / f"pneumonia_{i}.jpg"
            Image.fromarray(pneumonia_image, mode='L').save(pneumonia_path)
    
    return train_dir, val_dir, test_dir


@pytest.fixture
def mock_tensorflow():
    """Mock TensorFlow for tests that don't need actual ML operations."""
    import sys
    from unittest.mock import MagicMock
    
    # Create mock modules
    tf_mock = MagicMock()
    keras_mock = MagicMock()
    
    # Configure common TensorFlow functions
    tf_mock.keras = keras_mock
    tf_mock.constant.return_value = MagicMock()
    tf_mock.reduce_mean.return_value = MagicMock()
    
    # Mock model creation
    model_mock = MagicMock()
    model_mock.fit.return_value = MagicMock()
    model_mock.predict.return_value = np.array([[0.8], [0.2], [0.9]])
    model_mock.evaluate.return_value = [0.1, 0.95]  # loss, accuracy
    
    keras_mock.Sequential.return_value = model_mock
    keras_mock.models.load_model.return_value = model_mock
    
    # Add to sys.modules
    sys.modules['tensorflow'] = tf_mock
    sys.modules['tensorflow.keras'] = keras_mock
    
    yield tf_mock
    
    # Cleanup
    if 'tensorflow' in sys.modules:
        del sys.modules['tensorflow']
    if 'tensorflow.keras' in sys.modules:
        del sys.modules['tensorflow.keras']


@pytest.fixture
def mock_mlflow():
    """Mock MLflow for tests that don't need actual experiment tracking."""
    import sys
    from unittest.mock import MagicMock
    
    mlflow_mock = MagicMock()
    mlflow_mock.start_run.return_value.__enter__ = MagicMock()
    mlflow_mock.start_run.return_value.__exit__ = MagicMock()
    mlflow_mock.log_param = MagicMock()
    mlflow_mock.log_metric = MagicMock()
    mlflow_mock.log_artifact = MagicMock()
    
    sys.modules['mlflow'] = mlflow_mock
    
    yield mlflow_mock
    
    if 'mlflow' in sys.modules:
        del sys.modules['mlflow']


@pytest.fixture
def sample_training_config() -> dict:
    """Provide sample training configuration for tests."""
    return {
        "epochs": 2,
        "batch_size": 4,
        "learning_rate": 0.001,
        "dropout_rate": 0.5,
        "num_classes": 1,
        "img_size": (150, 150),
        "use_transfer_learning": True,
        "base_model_name": "MobileNetV2",
        "seed": 42,
    }


@pytest.fixture
def sample_predictions() -> np.ndarray:
    """Provide sample model predictions for testing evaluation functions."""
    return np.array([
        [0.9],  # High confidence positive
        [0.1],  # High confidence negative
        [0.7],  # Medium confidence positive
        [0.3],  # Medium confidence negative
        [0.8],  # High confidence positive
    ])


@pytest.fixture
def sample_labels() -> np.ndarray:
    """Provide sample ground truth labels for testing evaluation functions."""
    return np.array([1, 0, 1, 0, 1])  # Binary labels


@pytest.fixture
def mock_model_file(tmp_path: Path) -> Path:
    """Create a mock model file for testing."""
    model_path = tmp_path / "test_model.keras"
    # Create a dummy file to simulate a model
    model_path.write_text("mock model content")
    return model_path


@pytest.fixture(scope="session")
def gpu_available() -> bool:
    """Check if GPU is available for testing."""
    try:
        import tensorflow as tf
        return len(tf.config.list_physical_devices('GPU')) > 0
    except ImportError:
        return False


@pytest.fixture
def environment_variables():
    """Set up test environment variables."""
    original_env = os.environ.copy()
    
    # Set test environment variables
    test_env = {
        "PYTHONPATH": str(Path(__file__).parent.parent / "src"),
        "ENVIRONMENT": "test",
        "LOG_LEVEL": "DEBUG",
        "MLFLOW_TRACKING_URI": "file:///tmp/test_mlruns",
        "MODEL_CACHE_SIZE": "1",
    }
    
    os.environ.update(test_env)
    
    yield test_env
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def disable_warnings():
    """Disable specific warnings during testing."""
    import warnings
    
    # Disable common ML library warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    
    yield
    
    # Re-enable warnings
    warnings.resetwarnings()


# Pytest markers for organizing tests
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::FutureWarning"),
]


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "security: mark test as security test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "medical: mark test as medical data related"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add unit marker to tests in unit directory
        if "unit" in item.fspath.dirname:
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to tests in integration directory
        if "integration" in item.fspath.dirname:
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker to integration tests
        if "integration" in item.fspath.dirname:
            item.add_marker(pytest.mark.slow)
        
        # Add performance marker to performance tests
        if "performance" in item.fspath.dirname:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)


@pytest.fixture(autouse=True)
def cleanup_test_artifacts():
    """Automatically cleanup test artifacts after each test."""
    yield
    
    # Cleanup common test artifacts
    cleanup_paths = [
        Path("test_output.png"),
        Path("test_model.keras"),
        Path("test_predictions.csv"),
        Path("test_metrics.json"),
    ]
    
    for path in cleanup_paths:
        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass  # Ignore cleanup errors