#!/bin/bash
set -e

# Development environment setup script
# This script sets up a complete development environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "🚀 Setting up Chest X-Ray Pneumonia Detector development environment"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "📋 Using Python $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [[ ! -d "venv" ]]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
if [[ -f "requirements.txt" ]]; then
    pip install -r requirements.txt
fi

if [[ -f "requirements-dev.txt" ]]; then
    pip install -r requirements-dev.txt
fi

# Install package in development mode
echo "🔧 Installing package in development mode..."
pip install -e .

# Install pre-commit hooks
echo "🎣 Installing pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
    echo "✅ Pre-commit hooks installed"
else
    echo "⚠️  pre-commit not found, skipping hooks installation"
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data/{train,val,test}/{NORMAL,PNEUMONIA}
mkdir -p saved_models
mkdir -p reports
mkdir -p logs
mkdir -p checkpoints

# Copy environment template
if [[ ! -f ".env" ]] && [[ -f ".env.example" ]]; then
    echo "📄 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please review and update .env file with your configuration"
fi

# Run initial tests to verify setup
echo "🧪 Running initial tests..."
if command -v pytest &> /dev/null; then
    pytest --version > /dev/null && echo "✅ pytest is working"
    
    # Run a quick test to verify everything is working
    if [[ -d "tests" ]]; then
        echo "🔍 Running quick verification tests..."
        pytest tests/ -x --tb=short -q --disable-warnings || echo "⚠️  Some tests failed, but environment is set up"
    fi
else
    echo "⚠️  pytest not found, skipping test verification"
fi

# Verify code quality tools
echo "🔍 Verifying code quality tools..."
if command -v ruff &> /dev/null; then
    echo "✅ ruff is available"
else
    echo "⚠️  ruff not found"
fi

if command -v black &> /dev/null; then
    echo "✅ black is available"
else
    echo "⚠️  black not found"
fi

if command -v bandit &> /dev/null; then
    echo "✅ bandit is available"
else
    echo "⚠️  bandit not found"
fi

# Check if Docker is available for containerized development
if command -v docker &> /dev/null; then
    echo "🐳 Docker is available"
    if command -v docker-compose &> /dev/null; then
        echo "🐙 Docker Compose is available"
        echo "💡 You can use 'docker-compose up -d' for containerized development"
    fi
else
    echo "⚠️  Docker not found - containerized development not available"
fi

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "📋 Quick start guide:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Review and update .env file with your configuration"
echo "  3. Run tests: make test"
echo "  4. Start training: make train-dummy"
echo "  5. Check code quality: make lint"
echo ""
echo "🔧 Available make commands:"
make help
echo ""
echo "📚 Documentation:"
echo "  - README.md: Project overview and usage"
echo "  - ARCHITECTURE.md: System architecture and design"
echo "  - CONTRIBUTING.md: Contribution guidelines"
echo "  - docs/ROADMAP.md: Project roadmap and milestones"
echo ""
echo "🐛 Need help? Check the troubleshooting section in README.md"