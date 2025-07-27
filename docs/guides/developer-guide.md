# Developer Guide

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Project Structure](#project-structure)
3. [Development Workflow](#development-workflow)
4. [Code Standards](#code-standards)
5. [Testing Guidelines](#testing-guidelines)
6. [API Development](#api-development)
7. [Model Development](#model-development)
8. [Contributing Guidelines](#contributing-guidelines)
9. [Debugging and Profiling](#debugging-and-profiling)
10. [Deployment](#deployment)

## Development Environment Setup

### Prerequisites

- Python 3.8+ (3.11 recommended)
- Git
- Docker and Docker Compose
- VS Code (recommended) with Python extension

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/your-org/chest-xray-pneumonia-detector.git
cd chest-xray-pneumonia-detector

# Use development container (recommended)
code .
# VS Code will prompt to reopen in container

# OR setup locally
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
pre-commit install
```

### Development Container

The project includes a complete development container configuration:

```bash
# Open in VS Code
code .
# Click "Reopen in Container" when prompted

# Or use Docker directly
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
docker-compose exec app bash
```

### Environment Variables

Copy and customize the development environment:

```bash
cp .env.example .env
```

Key development settings:

```env
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
RELOAD=true
```

## Project Structure

```
chest-xray-pneumonia-detector/
├── src/                          # Source code
│   ├── api/                      # FastAPI application
│   │   ├── main.py              # API entry point
│   │   ├── models.py            # Pydantic models
│   │   ├── middleware.py        # Custom middleware
│   │   └── dependencies.py     # Dependency injection
│   ├── monitoring/              # Monitoring and observability
│   │   ├── metrics.py           # Prometheus metrics
│   │   ├── health.py            # Health checks
│   │   └── logging.py           # Structured logging
│   ├── security/                # Security components
│   │   ├── scanner.py           # Security scanning
│   │   ├── secrets_manager.py   # Secrets management
│   │   └── compliance.py        # HIPAA compliance
│   ├── chest_xray_pneumonia_detector/  # Core ML package
│   │   ├── __init__.py
│   │   └── pipeline.py
│   ├── config.py                # Configuration management
│   ├── data_loader.py           # Data loading utilities
│   ├── model_builder.py         # Model architectures
│   ├── train_engine.py          # Training pipeline
│   ├── inference.py             # Inference pipeline
│   └── ...                      # Other modules
├── tests/                       # Test suite
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   ├── performance/             # Performance tests
│   ├── conftest.py             # Test configuration
│   └── fixtures/               # Test data
├── docs/                        # Documentation
│   ├── guides/                  # User guides
│   ├── api/                     # API documentation
│   ├── security/                # Security documentation
│   └── deployment/              # Deployment guides
├── docker/                      # Docker configurations
├── .github/                     # GitHub workflows
├── .devcontainer/              # Development container
├── .vscode/                    # VS Code configuration
├── pyproject.toml              # Python project configuration
├── Dockerfile                  # Container definition
├── docker-compose.yml         # Service orchestration
├── Makefile                    # Development commands
└── README.md                   # Project overview
```

### Module Organization

#### Core Modules

- **`data_loader.py`**: Data loading, preprocessing, and augmentation
- **`model_builder.py`**: Neural network architectures and model creation
- **`train_engine.py`**: Training pipeline with MLflow integration
- **`inference.py`**: Batch inference and prediction pipeline
- **`predict_utils.py`**: Single prediction and Grad-CAM visualization

#### API Modules

- **`api/main.py`**: FastAPI application with endpoints
- **`api/models.py`**: Request/response schemas
- **`api/middleware.py`**: Custom middleware for logging, metrics, security
- **`api/dependencies.py`**: Dependency injection for services

#### Monitoring Modules

- **`monitoring/metrics.py`**: Prometheus metrics collection
- **`monitoring/health.py`**: Health checks and readiness probes
- **`monitoring/logging.py`**: Structured logging and audit trails

#### Security Modules

- **`security/scanner.py`**: Security scanning and SBOM generation
- **`security/secrets_manager.py`**: Secure secrets management
- **`security/compliance.py`**: HIPAA compliance utilities

## Development Workflow

### Git Workflow

We use Git Flow with the following branches:

- **`main`**: Production-ready code
- **`develop`**: Integration branch for features
- **`feature/*`**: Feature development branches
- **`hotfix/*`**: Critical bug fixes
- **`release/*`**: Release preparation

#### Creating a Feature

```bash
# Start from develop branch
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/add-new-model-architecture

# Make changes and commit
git add .
git commit -m "feat: add ResNet-50 architecture support"

# Push and create pull request
git push origin feature/add-new-model-architecture
```

### Code Review Process

1. **Create Pull Request**: Include detailed description and testing instructions
2. **Automated Checks**: CI pipeline runs tests, linting, and security scans
3. **Peer Review**: At least one team member reviews the code
4. **Address Feedback**: Make requested changes
5. **Merge**: Squash and merge after approval

### Pre-commit Hooks

Pre-commit hooks automatically run before each commit:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files

# Skip hooks (emergency only)
git commit --no-verify -m "emergency fix"
```

## Code Standards

### Python Style Guide

We follow PEP 8 with these modifications:

- **Line length**: 88 characters (Black default)
- **Imports**: Organized with isort
- **Docstrings**: Google-style docstrings
- **Type hints**: Required for all public functions

#### Example Function

```python
from typing import List, Optional, Tuple
import numpy as np


def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
    augment: Optional[dict] = None
) -> np.ndarray:
    """Preprocess medical image for model input.
    
    Args:
        image: Input image array in HWC format
        target_size: Target dimensions (height, width)
        normalize: Whether to normalize pixel values to [0, 1]
        augment: Augmentation parameters (rotation, brightness, etc.)
        
    Returns:
        Preprocessed image array
        
    Raises:
        ValueError: If image dimensions are invalid
        
    Example:
        >>> image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        >>> processed = preprocess_image(image, target_size=(224, 224))
        >>> processed.shape
        (224, 224, 3)
    """
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image, got {image.ndim}D")
    
    # Implementation here
    return processed_image
```

### Code Quality Tools

#### Linting with Ruff

```bash
# Check code quality
ruff check src/ tests/

# Fix auto-fixable issues
ruff check --fix src/ tests/

# Configuration in pyproject.toml
[tool.ruff]
line-length = 88
select = ["E", "W", "F", "I", "B", "C4", "UP"]
```

#### Formatting with Black

```bash
# Format code
black src/ tests/

# Check formatting
black --check src/ tests/

# Configuration in pyproject.toml
[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310', 'py311']
```

#### Type Checking with MyPy

```bash
# Check types
mypy src/

# Configuration in pyproject.toml
[tool.mypy]
python_version = "3.8"
disallow_untyped_defs = true
```

### Documentation Standards

#### Docstring Format

```python
def train_model(
    data_dir: str,
    model_config: dict,
    training_config: dict
) -> Tuple[Model, dict]:
    """Train a pneumonia detection model.
    
    This function implements the complete training pipeline including
    data loading, model creation, training loop, and validation.
    
    Args:
        data_dir: Path to training data directory containing
            train/, val/, and test/ subdirectories
        model_config: Model architecture configuration including:
            - model_type: Architecture type ('cnn', 'resnet', 'mobilenet')
            - input_shape: Input dimensions (height, width, channels)
            - num_classes: Number of output classes
        training_config: Training parameters including:
            - epochs: Number of training epochs
            - batch_size: Training batch size
            - learning_rate: Initial learning rate
            
    Returns:
        Tuple containing:
            - trained_model: Compiled Keras model
            - training_history: Dictionary with training metrics
            
    Raises:
        FileNotFoundError: If data_dir doesn't exist
        ValueError: If model_config or training_config are invalid
        
    Example:
        >>> model_config = {
        ...     'model_type': 'cnn',
        ...     'input_shape': (224, 224, 3),
        ...     'num_classes': 2
        ... }
        >>> training_config = {
        ...     'epochs': 10,
        ...     'batch_size': 32,
        ...     'learning_rate': 0.001
        ... }
        >>> model, history = train_model('data/', model_config, training_config)
    """
```

#### API Documentation

Use OpenAPI/Swagger documentation for all endpoints:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Chest X-Ray Pneumonia Detector API",
    description="AI-powered pneumonia detection from chest X-ray images",
    version="0.2.0"
)

class PredictionRequest(BaseModel):
    """Request model for image prediction."""
    
    image_data: str = Field(..., description="Base64-encoded image data")
    model_version: Optional[str] = Field(None, description="Specific model version")

@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict pneumonia from chest X-ray",
    description="Analyze a chest X-ray image and return pneumonia prediction",
    responses={
        200: {"description": "Prediction successful"},
        400: {"description": "Invalid image data"},
        500: {"description": "Prediction failed"}
    }
)
async def predict(request: PredictionRequest):
    """Predict pneumonia from chest X-ray image."""
```

## Testing Guidelines

### Test Structure

```python
import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.data_loader import preprocess_image
from src.model_builder import create_cnn_model


class TestImagePreprocessing:
    """Test suite for image preprocessing functions."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    def test_preprocess_image_basic(self, sample_image):
        """Test basic image preprocessing."""
        processed = preprocess_image(sample_image, target_size=(224, 224))
        
        assert processed.shape == (224, 224, 3)
        assert processed.dtype == np.float32
        assert 0.0 <= processed.min() <= processed.max() <= 1.0
    
    def test_preprocess_image_invalid_input(self):
        """Test preprocessing with invalid input."""
        invalid_image = np.random.random((100,))  # 1D array
        
        with pytest.raises(ValueError, match="Expected 3D image"):
            preprocess_image(invalid_image)
    
    @patch('src.data_loader.apply_augmentation')
    def test_preprocess_with_augmentation(self, mock_augment, sample_image):
        """Test preprocessing with data augmentation."""
        mock_augment.return_value = sample_image
        augment_config = {'rotation': 20, 'brightness': 0.2}
        
        processed = preprocess_image(
            sample_image, 
            augment=augment_config
        )
        
        mock_augment.assert_called_once_with(sample_image, augment_config)
```

### Test Categories

#### Unit Tests (`tests/unit/`)

Test individual functions and classes in isolation:

```bash
# Run unit tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/unit/ --cov=src --cov-report=html
```

#### Integration Tests (`tests/integration/`)

Test component interactions and end-to-end workflows:

```bash
# Run integration tests
pytest tests/integration/ -v --timeout=300

# Run specific integration test
pytest tests/integration/test_training_pipeline.py -v
```

#### Performance Tests (`tests/performance/`)

Test performance characteristics and benchmarks:

```bash
# Run performance tests
pytest tests/performance/ -v --timeout=600

# Run with performance profiling
pytest tests/performance/ --benchmark-json=benchmark.json
```

### Test Data Management

#### Fixtures

Use pytest fixtures for reusable test data:

```python
@pytest.fixture(scope="session")
def test_dataset():
    """Create a test dataset for integration tests."""
    # Create temporary directory structure
    # Generate sample images
    # Return dataset paths

@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.predict.return_value = np.array([[0.8], [0.2]])
    return model
```

#### Test Database

Use separate database for testing:

```bash
# Set test database URL
export DATABASE_URL="postgresql://test:test@localhost:5432/test_db"

# Run tests with test database
pytest tests/integration/test_database.py
```

### Continuous Integration

Tests run automatically on:

- **Pull requests**: Full test suite
- **Push to develop**: Unit and integration tests
- **Daily schedule**: Performance and security tests

```yaml
# .github/workflows/ci.yml
- name: Run tests
  run: |
    pytest tests/ -v \
      --cov=src \
      --cov-report=xml \
      --junitxml=test-results.xml
```

## API Development

### FastAPI Best Practices

#### Request/Response Models

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime

class PredictionRequest(BaseModel):
    """Request model with validation."""
    
    image_data: str = Field(..., description="Base64-encoded image")
    model_version: Optional[str] = Field(None, description="Model version")
    
    @validator('image_data')
    def validate_image_data(cls, v):
        """Validate base64 image data."""
        import base64
        try:
            base64.b64decode(v)
        except Exception:
            raise ValueError('Invalid base64 image data')
        return v

class PredictionResponse(BaseModel):
    """Response model with examples."""
    
    prediction: int = Field(..., description="Predicted class")
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 1,
                "confidence": 0.87,
                "timestamp": "2025-07-27T10:30:00Z"
            }
        }
```

#### Error Handling

```python
from fastapi import HTTPException, status

class APIError(Exception):
    """Custom API exception."""
    
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    """Handle custom API errors."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Predict with proper error handling."""
    try:
        # Validate input
        if not request.image_data:
            raise APIError("Image data is required", status.HTTP_400_BAD_REQUEST)
        
        # Make prediction
        result = await model_service.predict(request.image_data)
        return PredictionResponse(**result)
        
    except ValueError as e:
        raise APIError(f"Invalid input: {e}", status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise APIError("Prediction failed", status.HTTP_500_INTERNAL_SERVER_ERROR)
```

#### Dependency Injection

```python
from fastapi import Depends
from typing import Annotated

class ModelService:
    """Model service for predictions."""
    
    def __init__(self):
        self.model = None
        self.load_model()
    
    async def predict(self, image_data: bytes) -> dict:
        """Make prediction."""
        # Implementation
        pass

# Global service instance
model_service = ModelService()

def get_model_service() -> ModelService:
    """Dependency to get model service."""
    return model_service

@app.post("/predict")
async def predict(
    request: PredictionRequest,
    model_service: Annotated[ModelService, Depends(get_model_service)]
):
    """Endpoint with dependency injection."""
    result = await model_service.predict(request.image_data)
    return result
```

### API Testing

```python
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_predict_endpoint():
    """Test prediction endpoint."""
    # Prepare test data
    with open("test_image.jpg", "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", f, "image/jpeg")}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0

def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    
    health_data = response.json()
    assert health_data["status"] == "healthy"
```

## Model Development

### Model Architecture

#### Creating New Architectures

```python
import tensorflow as tf
from typing import Tuple, Optional

def create_custom_cnn(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    dropout_rate: float = 0.5
) -> tf.keras.Model:
    """Create custom CNN architecture.
    
    Args:
        input_shape: Input dimensions (height, width, channels)
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # Convolutional blocks
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Classification head
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    if num_classes == 1:
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
    else:
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        loss = 'categorical_crossentropy'
    
    model = tf.keras.Model(inputs, outputs, name='custom_cnn')
    
    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=['accuracy']
    )
    
    return model
```

#### Transfer Learning

```python
def create_transfer_model(
    base_model_name: str = 'MobileNetV2',
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 1,
    trainable_layers: int = 0
) -> tf.keras.Model:
    """Create transfer learning model.
    
    Args:
        base_model_name: Pre-trained model name
        input_shape: Input dimensions
        num_classes: Number of output classes
        trainable_layers: Number of top layers to make trainable
        
    Returns:
        Transfer learning model
    """
    # Get base model
    base_models = {
        'MobileNetV2': tf.keras.applications.MobileNetV2,
        'ResNet50': tf.keras.applications.ResNet50,
        'VGG16': tf.keras.applications.VGG16,
    }
    
    base_model_class = base_models[base_model_name]
    base_model = base_model_class(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Make top layers trainable if specified
    if trainable_layers > 0:
        for layer in base_model.layers[-trainable_layers:]:
            layer.trainable = True
    
    # Add custom head
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    if num_classes == 1:
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model
```

### Training Pipeline

#### Custom Training Loop

```python
import mlflow
from typing import Dict, Any

class TrainingPipeline:
    """Custom training pipeline with MLflow integration."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.history = None
    
    def setup_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Setup training and validation datasets."""
        # Load and preprocess data
        train_ds = self.create_dataset(self.config['train_dir'], training=True)
        val_ds = self.create_dataset(self.config['val_dir'], training=False)
        return train_ds, val_ds
    
    def create_model(self) -> tf.keras.Model:
        """Create and compile model."""
        if self.config['use_transfer_learning']:
            model = create_transfer_model(
                base_model_name=self.config['base_model'],
                num_classes=self.config['num_classes']
            )
        else:
            model = create_custom_cnn(
                input_shape=self.config['input_shape'],
                num_classes=self.config['num_classes']
            )
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.config['learning_rate']),
            loss=self.config['loss'],
            metrics=['accuracy']
        )
        
        return model
    
    def setup_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """Setup training callbacks."""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.2,
                patience=5,
                min_lr=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint(
                self.config['checkpoint_path'],
                save_best_only=True,
                monitor='val_loss'
            )
        ]
        
        return callbacks
    
    def train(self) -> Dict[str, Any]:
        """Execute training pipeline."""
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(self.config)
            
            # Setup data and model
            train_ds, val_ds = self.setup_data()
            self.model = self.create_model()
            callbacks = self.setup_callbacks()
            
            # Train model
            self.history = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self.config['epochs'],
                callbacks=callbacks,
                verbose=1
            )
            
            # Log metrics
            for metric, values in self.history.history.items():
                for epoch, value in enumerate(values):
                    mlflow.log_metric(metric, value, step=epoch)
            
            # Save model
            self.model.save(self.config['model_path'])
            mlflow.log_artifact(self.config['model_path'])
            
            return {
                'model': self.model,
                'history': self.history.history,
                'final_loss': self.history.history['val_loss'][-1],
                'final_accuracy': self.history.history['val_accuracy'][-1]
            }
```

### Model Evaluation

```python
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(
    model: tf.keras.Model,
    test_data: tf.data.Dataset,
    class_names: List[str]
) -> Dict[str, Any]:
    """Comprehensive model evaluation."""
    
    # Get predictions
    y_true = []
    y_pred = []
    y_prob = []
    
    for batch_x, batch_y in test_data:
        predictions = model.predict(batch_x)
        y_true.extend(batch_y.numpy())
        y_prob.extend(predictions)
        y_pred.extend((predictions > 0.5).astype(int))
    
    # Calculate metrics
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'accuracy': report['accuracy'],
        'precision': report['macro avg']['precision'],
        'recall': report['macro avg']['recall'],
        'f1_score': report['macro avg']['f1-score']
    }
```

## Contributing Guidelines

### Contribution Process

1. **Fork Repository**: Create your own fork
2. **Create Feature Branch**: Branch from `develop`
3. **Implement Changes**: Follow coding standards
4. **Add Tests**: Maintain test coverage
5. **Update Documentation**: Keep docs current
6. **Submit Pull Request**: Include detailed description

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings introduced
```

### Code Review Checklist

#### Functionality
- [ ] Code works as intended
- [ ] Edge cases handled
- [ ] Error handling implemented
- [ ] Performance considerations addressed

#### Code Quality
- [ ] Follows coding standards
- [ ] Well-documented
- [ ] No code duplication
- [ ] Appropriate abstractions

#### Testing
- [ ] Adequate test coverage
- [ ] Tests are meaningful
- [ ] Edge cases tested
- [ ] Integration tests included

#### Security
- [ ] No hardcoded secrets
- [ ] Input validation implemented
- [ ] HIPAA compliance maintained
- [ ] Security best practices followed

## Debugging and Profiling

### Debugging Techniques

#### Logging

```python
import logging

logger = logging.getLogger(__name__)

def debug_prediction(image_path: str, model: tf.keras.Model):
    """Debug prediction with detailed logging."""
    logger.debug(f"Processing image: {image_path}")
    
    # Load and preprocess image
    image = load_image(image_path)
    logger.debug(f"Image shape: {image.shape}, dtype: {image.dtype}")
    
    processed = preprocess_image(image)
    logger.debug(f"Processed shape: {processed.shape}, range: [{processed.min():.3f}, {processed.max():.3f}]")
    
    # Make prediction
    prediction = model.predict(processed[np.newaxis, ...])
    logger.debug(f"Raw prediction: {prediction}")
    
    confidence = float(prediction[0][0])
    class_name = "Pneumonia" if confidence > 0.5 else "Normal"
    
    logger.info(f"Prediction: {class_name} (confidence: {confidence:.3f})")
    
    return {
        'prediction': int(confidence > 0.5),
        'confidence': confidence,
        'class_name': class_name
    }
```

#### Profiling

```python
import cProfile
import pstats
from functools import wraps

def profile_function(func):
    """Decorator to profile function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()
            
        # Print stats
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        
        return result
    return wrapper

@profile_function
def train_model_profiled(**kwargs):
    """Train model with profiling."""
    return train_model(**kwargs)
```

#### Memory Profiling

```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    """Function with memory profiling."""
    # Memory-intensive operations
    large_array = np.random.random((10000, 10000))
    processed = process_array(large_array)
    return processed

# Run with: python -m memory_profiler script.py
```

### Performance Optimization

#### GPU Optimization

```python
import tensorflow as tf

# Configure GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Mixed precision training
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# XLA compilation
@tf.function(experimental_relax_shapes=True, jit_compile=True)
def optimized_prediction(model, inputs):
    """Optimized prediction with XLA."""
    return model(inputs)
```

#### Data Pipeline Optimization

```python
def create_optimized_dataset(data_dir: str, batch_size: int) -> tf.data.Dataset:
    """Create optimized data pipeline."""
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(data_dir),
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    
    # Optimization techniques
    dataset = dataset.cache()  # Cache in memory
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch next batch
    dataset = dataset.map(
        preprocess_batch, 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    return dataset
```

## Deployment

### Docker Deployment

```bash
# Build production image
docker build -t chest-xray-detector:latest .

# Run container
docker run -p 8080:8080 \
  -e ENVIRONMENT=production \
  -e GPU_ENABLED=true \
  chest-xray-detector:latest

# Using docker-compose
docker-compose up -d
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chest-xray-detector
spec:
  replicas: 3
  selector:
    matchLabels:
      app: chest-xray-detector
  template:
    metadata:
      labels:
        app: chest-xray-detector
    spec:
      containers:
      - name: api
        image: chest-xray-detector:latest
        ports:
        - containerPort: 8080
        env:
        - name: ENVIRONMENT
          value: "production"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

### Environment-Specific Configuration

#### Development
```env
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
RELOAD=true
```

#### Staging
```env
ENVIRONMENT=staging
DEBUG=false
LOG_LEVEL=INFO
HIPAA_COMPLIANT=true
```

#### Production
```env
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING
HIPAA_COMPLIANT=true
GPU_ENABLED=true
```

This developer guide provides comprehensive information for contributing to and developing the Chest X-Ray Pneumonia Detector project. Follow these guidelines to maintain code quality, ensure security compliance, and contribute effectively to the project.