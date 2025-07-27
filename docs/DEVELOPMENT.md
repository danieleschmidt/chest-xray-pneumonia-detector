# Development Guide

This guide provides comprehensive information for developers contributing to the Chest X-Ray Pneumonia Detector project.

## Table of Contents

- [Quick Start](#quick-start)
- [Development Environment](#development-environment)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Building and Deployment](#building-and-deployment)
- [Debugging](#debugging)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Git
- Docker (optional but recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/chest-xray-pneumonia-detector.git
   cd chest-xray-pneumonia-detector
   ```

2. **Run the setup script**
   ```bash
   ./scripts/setup-dev.sh
   ```

3. **Activate virtual environment**
   ```bash
   source venv/bin/activate
   ```

4. **Verify installation**
   ```bash
   make test
   python -m src.version_cli
   ```

## Development Environment

### Option 1: Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
make test
```

### Option 2: Dev Container

Use the provided `.devcontainer/devcontainer.json` for a consistent development environment:

```bash
# With VS Code
code .
# Then: Ctrl+Shift+P -> "Dev Containers: Reopen in Container"

# With command line
devcontainer up --workspace-folder .
```

### Option 3: Docker Development

```bash
# Build development image
docker-compose build dev

# Start development environment
docker-compose up -d dev

# Enter development container
docker-compose exec dev bash
```

## Project Structure

```
chest-xray-pneumonia-detector/
├── .devcontainer/          # Development container config
├── .github/                # GitHub workflows and templates
│   ├── workflows/          # CI/CD workflows
│   └── ISSUE_TEMPLATE/     # Issue templates
├── .vscode/                # VS Code configuration
├── docs/                   # Documentation
│   ├── adr/               # Architecture Decision Records
│   ├── api/               # API documentation
│   ├── guides/            # User guides
│   └── runbooks/          # Operational runbooks
├── monitoring/             # Monitoring configuration
├── scripts/               # Build and utility scripts
├── src/                   # Source code
│   ├── chest_xray_pneumonia_detector/  # Main package
│   └── monitoring/        # Monitoring utilities
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   ├── e2e/              # End-to-end tests
│   └── performance/      # Performance tests
├── docker-compose.yml     # Multi-service Docker setup
├── Dockerfile            # Multi-stage Docker build
├── Makefile              # Build automation
├── pyproject.toml        # Python project configuration
└── requirements*.txt     # Python dependencies
```

### Source Code Organization

```
src/
├── chest_xray_pneumonia_detector/  # Core package
├── architecture_review.py          # Architecture validation
├── config.py                      # Configuration management
├── data_loader.py                 # Data loading utilities
├── data_split.py                  # Dataset splitting
├── evaluate.py                    # Model evaluation
├── grad_cam.py                    # Gradient-weighted Class Activation Mapping
├── image_utils.py                 # Image processing utilities
├── inference.py                   # Batch inference
├── input_validation.py            # Input validation
├── model_builder.py               # Model architecture definitions
├── model_registry.py              # Model versioning and storage
├── monitoring/                    # Monitoring and observability
│   ├── health_checks.py          # Health check endpoints
│   ├── logging_config.py         # Logging configuration
│   └── metrics.py                # Metrics collection
├── performance_benchmark.py       # Performance testing
├── predict_utils.py               # Single prediction utilities
├── train_engine.py                # Training orchestration
└── version_cli.py                 # Version management
```

## Development Workflow

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes
# ... edit files ...

# Run tests
make test

# Run linting
make lint

# Commit changes
git add .
git commit -m "feat: add your feature description"

# Push branch
git push origin feature/your-feature-name

# Create pull request
gh pr create --title "Add your feature" --body "Description of changes"
```

### 2. Pre-commit Hooks

Pre-commit hooks automatically run on each commit:

- **Code formatting**: Black, Ruff
- **Linting**: Ruff, Bandit
- **Type checking**: MyPy (if configured)
- **Tests**: Pytest (quick tests only)

### 3. Code Review Process

1. **Create Pull Request**: Include clear description and test evidence
2. **Automated Checks**: Ensure CI passes
3. **Code Review**: At least one reviewer approval required
4. **Testing**: Verify changes work as expected
5. **Merge**: Squash and merge to main branch

## Code Standards

### Python Style

- **Formatter**: Black (line length: 88)
- **Linter**: Ruff with strict configuration
- **Import sorting**: isort (integrated with Ruff)
- **Type hints**: Required for public APIs

### Code Quality

```python
# Good example
def process_image(
    image_path: Path, 
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Process medical image for model inference.
    
    Args:
        image_path: Path to the image file
        target_size: Target dimensions for resizing
        
    Returns:
        Preprocessed image array
        
    Raises:
        ValueError: If image cannot be loaded
    """
    try:
        image = Image.open(image_path)
        image = image.resize(target_size)
        return np.array(image)
    except Exception as e:
        raise ValueError(f"Failed to process image {image_path}: {e}") from e
```

### Documentation Standards

- **Docstrings**: Google style for all public functions
- **Type hints**: Required for function signatures
- **Comments**: Explain why, not what
- **README updates**: Update when adding features

### Security Guidelines

- **No secrets**: Never commit API keys, passwords, or tokens
- **Input validation**: Validate all user inputs
- **Error handling**: Don't expose sensitive information in errors
- **Dependencies**: Keep dependencies updated and secure

## Testing

### Test Categories

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Test performance characteristics

### Running Tests

```bash
# All tests
make test

# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Performance tests (slow)
pytest tests/performance/ -v -m slow

# Specific test file
pytest tests/test_data_loader.py -v

# With coverage
pytest --cov=src --cov-report=html

# Parallel execution
pytest -n auto
```

### Writing Tests

```python
# tests/unit/test_image_utils.py
import pytest
from pathlib import Path
from src.image_utils import process_image

class TestImageProcessing:
    def test_process_valid_image(self, sample_image_path):
        """Test processing a valid image."""
        result = process_image(sample_image_path)
        
        assert result is not None
        assert result.shape == (224, 224, 3)
        assert result.dtype == np.uint8
    
    def test_process_invalid_path(self):
        """Test error handling for invalid path."""
        with pytest.raises(ValueError, match="Failed to process image"):
            process_image(Path("nonexistent.jpg"))
    
    @pytest.mark.parametrize("size", [(128, 128), (256, 256), (512, 512)])
    def test_different_sizes(self, sample_image_path, size):
        """Test processing with different target sizes."""
        result = process_image(sample_image_path, target_size=size)
        assert result.shape[:2] == size
```

### Test Configuration

Tests are configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
minversion = "6.0"
addopts = "-ra -q --strict-markers"
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]
```

## Building and Deployment

### Local Build

```bash
# Python package
make build

# Docker images
./scripts/build.sh all

# Specific Docker target
./scripts/build.sh production
```

### Development Deployment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production Deployment

See [deployment documentation](./guides/deployment.md) for production deployment procedures.

## Debugging

### Local Debugging

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with debugging
python -m src.train_engine --epochs 1 --batch_size 2 --use_dummy_data

# Use Python debugger
import pdb; pdb.set_trace()
```

### VS Code Debugging

Use the provided `.vscode/launch.json` configurations:

- **Python: Train Model**: Debug training pipeline
- **Python: Run Inference**: Debug inference pipeline
- **Python: Current File**: Debug current file
- **Python: Pytest Current File**: Debug current test file

### Container Debugging

```bash
# Debug production container
docker run -it --rm pneumonia-detector:latest bash

# Debug with volumes
docker run -it --rm -v $(pwd):/app pneumonia-detector:latest bash

# Check container health
docker-compose exec api python -m src.monitoring.health_checks --check all
```

### Common Issues

1. **Import Errors**: Check PYTHONPATH and virtual environment
2. **Memory Issues**: Monitor memory usage during training
3. **GPU Issues**: Verify CUDA installation and GPU availability
4. **File Permission Issues**: Check file ownership in containers

## Performance Considerations

### Model Performance

```bash
# Benchmark inference speed
python -m src.performance_benchmark

# Profile model training
python -m cProfile -o profile.stats -m src.train_engine --epochs 1

# Analyze profile
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
"
```

### Memory Optimization

- Use data generators for large datasets
- Implement batch processing for inference
- Monitor memory usage with the monitoring tools
- Use appropriate batch sizes for available memory

### Development Performance

```bash
# Fast test subset
pytest tests/unit/ -x --tb=short

# Parallel testing
pytest -n auto

# Skip slow tests
pytest -m "not slow"
```

## Troubleshooting

### Common Development Issues

#### Virtual Environment Issues

```bash
# Recreate virtual environment
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -e .
pip install -r requirements-dev.txt
```

#### Dependency Conflicts

```bash
# Check dependency tree
pip-tree

# Update all dependencies
pip install --upgrade -r requirements.txt

# Check for security issues
safety check
```

#### Test Failures

```bash
# Run failed tests in verbose mode
pytest --lf -v

# Run specific test with debugging
pytest tests/test_failing.py::test_function -v -s

# Clear test cache
pytest --cache-clear
```

#### Docker Issues

```bash
# Rebuild without cache
docker-compose build --no-cache

# Clean Docker system
docker system prune -a

# Check container logs
docker-compose logs service-name
```

### Getting Help

1. **Check existing issues**: Search GitHub issues for similar problems
2. **Read error messages**: Most errors include helpful information
3. **Check logs**: Enable debug logging for detailed information
4. **Ask for help**: Create an issue with detailed information

### Debug Information Collection

When reporting issues, include:

```bash
# System information
python --version
pip --version
docker --version

# Package information
pip list
python -m src.version_cli

# Health check
python -m src.monitoring.health_checks --check all

# Recent logs
tail -n 50 logs/application.log
```

## Next Steps

- Review the [Architecture Documentation](../ARCHITECTURE.md)
- Check the [API Documentation](./api/README.md)
- Read the [User Guides](./guides/README.md)
- Explore the [Runbooks](./runbooks/README.md)

For questions or suggestions about this development guide, please create an issue or contribute improvements via pull request.