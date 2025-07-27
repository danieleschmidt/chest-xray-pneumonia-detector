import pytest
import tempfile
import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    pil_image = Image.fromarray(image)
    return pil_image


@pytest.fixture
def sample_dataset_structure(temp_dir, sample_image):
    """Create a sample dataset structure for testing."""
    dataset_path = Path(temp_dir) / "dataset"
    
    # Create directory structure
    for split in ["train", "val", "test"]:
        for class_name in ["NORMAL", "PNEUMONIA"]:
            class_dir = dataset_path / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Create sample images
            for i in range(5):
                image_path = class_dir / f"image_{i}.jpg"
                sample_image.save(image_path)
    
    return dataset_path


@pytest.fixture
def mock_model_config():
    """Mock model configuration for testing."""
    return {
        "input_shape": (224, 224, 3),
        "num_classes": 1,
        "architecture": "MobileNetV2",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 5
    }


@pytest.fixture
def sample_trained_model_path(temp_dir):
    """Path for a sample trained model."""
    return os.path.join(temp_dir, "test_model.keras")


@pytest.fixture(scope="session")
def test_data_dir():
    """Session-scoped test data directory."""
    test_dir = Path(__file__).parent / "fixtures" / "test_data"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


@pytest.fixture
def mock_environment_variables():
    """Mock environment variables for testing."""
    env_vars = {
        "MLFLOW_TRACKING_URI": "sqlite:///test_mlflow.db",
        "MODEL_REGISTRY_PATH": "./test_models",
        "LOG_LEVEL": "DEBUG"
    }
    
    # Store original values
    original_values = {}
    for key, value in env_vars.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield env_vars
    
    # Restore original values
    for key, original_value in original_values.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture
def skip_slow_tests():
    """Skip slow tests unless explicitly requested."""
    if not pytest.config.getoption("--run-slow"):
        pytest.skip("slow test not requested")


def pytest_addoption(parser):
    """Add command line options for pytest."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow tests"
    )
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests"
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "integration: mark test as integration")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "e2e: mark test as end-to-end")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options."""
    if config.getoption("--run-slow"):
        return
    
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)