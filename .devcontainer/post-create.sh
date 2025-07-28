#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# 🔬 Chest X-Ray Pneumonia Detector - Post-Creation Setup Script
# ═══════════════════════════════════════════════════════════════════════════════

set -e

echo "🚀 Starting post-creation setup for Chest X-Ray Pneumonia Detector..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install additional system dependencies
echo "🔧 Installing system dependencies..."
sudo apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    vim \
    htop \
    tree \
    jq \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Install development dependencies
if [ -f "requirements-dev.txt" ]; then
    echo "📋 Installing development requirements..."
    pip install -r requirements-dev.txt
fi

# Install main project requirements
if [ -f "requirements.txt" ]; then
    echo "📋 Installing main requirements..."
    pip install -r requirements.txt
fi

# Install project in development mode
echo "🔧 Installing project in development mode..."
pip install -e .

# Set up pre-commit hooks
echo "🎣 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p \
    data/{train,val,test} \
    saved_models \
    checkpoints \
    reports \
    logs \
    cache \
    notebooks \
    mlruns

# Set appropriate permissions
echo "🔐 Setting permissions..."
chmod +x scripts/*.sh 2>/dev/null || true
chmod +x .devcontainer/*.sh

# Configure git (if not already configured)
echo "⚙️ Configuring git..."
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global core.autocrlf input

# Set up Jupyter configuration
echo "📊 Configuring Jupyter..."
jupyter --generate-config 2>/dev/null || true
echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py 2>/dev/null || true
echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py 2>/dev/null || true
echo "c.NotebookApp.port = 8888" >> ~/.jupyter/jupyter_notebook_config.py 2>/dev/null || true

# Install additional Python packages for ML development
echo "🧠 Installing additional ML packages..."
pip install --quiet \
    jupyterlab \
    matplotlib \
    seaborn \
    plotly \
    scikit-image \
    opencv-python-headless \
    pillow

# Set up environment file
echo "🌍 Setting up environment configuration..."
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo "✅ Created .env from .env.example"
fi

# Display helpful information
echo ""
echo "✅ Post-creation setup completed successfully!"
echo ""
echo "🔧 Available Commands:"
echo "  • make help          - Show available make targets"
echo "  • pytest            - Run test suite"
echo "  • ruff check .       - Run linting"
echo "  • black .            - Format code"
echo "  • pre-commit run --all-files - Run all pre-commit hooks"
echo ""
echo "📊 Development Tools:"
echo "  • Jupyter Lab: http://localhost:8888"
echo "  • MLflow UI: http://localhost:5000"
echo "  • API Server: http://localhost:8000"
echo ""
echo "🚀 Ready for development!"