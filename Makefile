.PHONY: help install install-dev test lint format security clean build dev docs

# Default target
help:
	@echo "Available commands:"
	@echo "  install     - Install production dependencies"
	@echo "  install-dev - Install development dependencies"
	@echo "  test        - Run tests with coverage"
	@echo "  lint        - Run linting checks"
	@echo "  format      - Format code with black and ruff"
	@echo "  security    - Run security scans"
	@echo "  clean       - Clean build artifacts and cache"
	@echo "  build       - Build package"
	@echo "  dev         - Start development environment"
	@echo "  docs        - Generate documentation"

# Installation
install:
	pip install -e .
	pip install -r requirements.txt

install-dev:
	pip install -e .
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

# Testing
test:
	pytest --cov=src --cov-report=html --cov-report=term-missing -v

test-fast:
	pytest -x --tb=short

test-integration:
	pytest -m integration -v

test-unit:
	pytest -m "not integration" -v

# Code Quality
lint:
	ruff check src tests
	black --check src tests
	bandit -r src -ll

format:
	ruff check --fix src tests
	black src tests

# Security
security:
	bandit -r src -ll
	pip-audit
	safety check

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Building
build: clean
	python -m build

# Development
dev:
	@echo "Starting development environment..."
	@echo "Available commands:"
	@echo "  make test     - Run tests"
	@echo "  make lint     - Run linting"
	@echo "  make format   - Format code"

# Documentation
docs:
	@echo "Documentation generation not yet implemented"
	@echo "Future: Sphinx or MkDocs documentation"

# MLflow
mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

# Training shortcuts
train-dummy:
	python -m src.train_engine --epochs 5 --batch_size 16 --use_dummy_data

train-quick:
	python -m src.train_engine --epochs 10 --batch_size 32 --use_dummy_data

# Version management
version:
	python -m src.version_cli

# Dataset utilities
dataset-stats:
	python -m src.dataset_stats --data_dir data

# Performance benchmarking
benchmark:
	python -m src.performance_benchmark

# Model validation
validate-model:
	python -m src.model_architecture_validation

# Security scan
scan-security:
	python -m src.dependency_security_scan