#!/bin/bash

# Post-create script for development container setup
set -e

echo "🚀 Setting up Chest X-Ray Pneumonia Detector development environment..."

# Install the package in development mode
echo "📦 Installing package in development mode..."
pip install -e .

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/{train,val,test}/{NORMAL,PNEUMONIA}
mkdir -p saved_models
mkdir -p reports
mkdir -p logs
mkdir -p .artifacts

# Set up MLflow tracking
echo "🔬 Initializing MLflow tracking..."
mkdir -p mlruns

# Generate .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "🔐 Creating .env file from template..."
    cp .env.example .env
fi

# Set up Jupyter Lab configuration
echo "🔬 Setting up Jupyter Lab..."
mkdir -p ~/.jupyter
cat > ~/.jupyter/jupyter_lab_config.py << 'EOF'
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
c.ServerApp.token = ''
c.ServerApp.password = ''
EOF

# Install Jupyter extensions
jupyter labextension install @jupyterlab/git

# Run initial security scan
echo "🔒 Running initial security scan..."
python -m bandit -r src/ -f json -o .artifacts/bandit-report.json || true

# Run initial linting
echo "✨ Running initial code quality checks..."
python -m ruff check src/ tests/ --output-format=json --output-file=.artifacts/ruff-report.json || true
python -m ruff format src/ tests/ || true

# Generate initial test coverage report
echo "🧪 Running initial test suite..."
python -m pytest tests/ --cov=src --cov-report=html:.artifacts/coverage-html --cov-report=json:.artifacts/coverage.json || true

# Create sample data structure documentation
cat > data/README.md << 'EOF'
# Dataset Directory Structure

This directory contains the chest X-ray dataset organized for pneumonia detection.

## Structure
```
data/
├── train/
│   ├── NORMAL/     # Normal chest X-ray images for training
│   └── PNEUMONIA/  # Pneumonia chest X-ray images for training
├── val/
│   ├── NORMAL/     # Normal chest X-ray images for validation
│   └── PNEUMONIA/  # Pneumonia chest X-ray images for validation
└── test/
    ├── NORMAL/     # Normal chest X-ray images for testing
    └── PNEUMONIA/  # Pneumonia chest X-ray images for testing
```

## Usage
To populate this directory with data, use the data splitting utility:

```bash
python -m src.data_split --input_dir /path/to/raw_dataset --output_dir data --val_frac 0.1 --test_frac 0.1
```

## Data Requirements
- Images should be in common formats (JPEG, PNG, DICOM)
- Minimum resolution: 150x150 pixels
- Maximum file size: 50MB per image
- Total dataset size: Recommend 1000+ images per class for production use
EOF

echo "✅ Development environment setup complete!"
echo "🔧 Available commands:"
echo "  - make test: Run test suite"
echo "  - make lint: Run code quality checks"
echo "  - make security: Run security scans"
echo "  - make train: Train model with default parameters"
echo "  - make serve: Start MLflow server"
echo "  - jupyter lab: Start Jupyter Lab server"

# Start MLflow server in background for development
echo "🔬 Starting MLflow server..."
nohup mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root file:///workspace/mlruns > logs/mlflow.log 2>&1 &