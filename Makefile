# Makefile for Chest X-Ray Pneumonia Detector
# 
# Usage:
#   make help              Show this help message
#   make install           Install the project in development mode
#   make test              Run the test suite
#   make lint              Run code quality checks
#   make format            Format code using black and ruff
#   make security          Run security scans
#   make build             Build Docker image
#   make up                Start services with docker-compose
#   make down              Stop services
#   make clean             Clean up build artifacts and caches

.PHONY: help install install-dev test test-unit test-integration test-performance lint format security clean build push up down logs shell docs serve-docs train inference gradcam benchmark validate audit release

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
DOCKER := docker
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := chest-xray-pneumonia-detector
VERSION := $(shell python -c "import toml; print(toml.load('pyproject.toml')['project']['version'])")
DOCKER_TAG := $(PROJECT_NAME):$(VERSION)
DOCKER_LATEST := $(PROJECT_NAME):latest

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
NC := \033[0m # No Color

define print_info
	@echo "$(CYAN)$(1)$(NC)"
endef

define print_success
	@echo "$(GREEN)$(1)$(NC)"
endef

define print_warning
	@echo "$(YELLOW)$(1)$(NC)"
endef

define print_error
	@echo "$(RED)$(1)$(NC)"
endef

help: ## Show this help message
	@echo "$(BLUE)Chest X-Ray Pneumonia Detector - Development Commands$(NC)"
	@echo ""
	@echo "$(YELLOW)Available commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(CYAN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)Docker commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## .*[Dd]ocker/ {printf "  $(PURPLE)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Development Environment Setup
install: ## Install the project in development mode
	$(call print_info,"Installing project in development mode...")
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e .
	$(call print_success,"Installation completed!")

install-dev: ## Install development dependencies
	$(call print_info,"Installing development dependencies...")
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .
	pre-commit install
	$(call print_success,"Development environment setup completed!")

# Testing
test: ## Run the complete test suite
	$(call print_info,"Running complete test suite...")
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html --cov-report=xml
	$(call print_success,"All tests completed!")

test-unit: ## Run unit tests only
	$(call print_info,"Running unit tests...")
	$(PYTHON) -m pytest tests/unit/ -v --cov=src

test-integration: ## Run integration tests only
	$(call print_info,"Running integration tests...")
	$(PYTHON) -m pytest tests/integration/ -v --timeout=300

test-performance: ## Run performance tests only
	$(call print_info,"Running performance tests...")
	$(PYTHON) -m pytest tests/performance/ -v --timeout=600

test-security: ## Run security-related tests
	$(call print_info,"Running security tests...")
	$(PYTHON) -m pytest tests/ -m security -v

test-watch: ## Run tests in watch mode
	$(call print_info,"Starting test watcher...")
	$(PYTHON) -m pytest-watch tests/

# Code Quality
lint: ## Run code quality checks
	$(call print_info,"Running code quality checks...")
	@echo "$(YELLOW)Running ruff...$(NC)"
	$(PYTHON) -m ruff check src/ tests/
	@echo "$(YELLOW)Running black check...$(NC)"
	$(PYTHON) -m black --check src/ tests/
	@echo "$(YELLOW)Running mypy...$(NC)"
	$(PYTHON) -m mypy src/
	@echo "$(YELLOW)Running bandit...$(NC)"
	$(PYTHON) -m bandit -r src/
	$(call print_success,"All linting checks passed!")

format: ## Format code using black and ruff
	$(call print_info,"Formatting code...")
	$(PYTHON) -m black src/ tests/
	$(PYTHON) -m ruff check --fix src/ tests/
	$(PYTHON) -m isort src/ tests/
	$(call print_success,"Code formatting completed!")

security: ## Run security scans
	$(call print_info,"Running security scans...")
	@echo "$(YELLOW)Running bandit security scan...$(NC)"
	$(PYTHON) -m bandit -r src/ -f json -o .artifacts/bandit-report.json
	@echo "$(YELLOW)Running safety check...$(NC)"
	$(PYTHON) -m safety check --json --output .artifacts/safety-report.json
	@echo "$(YELLOW)Running pip-audit...$(NC)"
	$(PYTHON) -m pip-audit --format=json --output=.artifacts/pip-audit-report.json
	$(call print_success,"Security scans completed!")

# Docker Operations
build: ## Build Docker image
	$(call print_info,"Building Docker image...")
	$(DOCKER) build \
		--build-arg BUILD_DATE=$$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
		--build-arg VCS_REF=$$(git rev-parse --short HEAD) \
		--build-arg VERSION=$(VERSION) \
		-t $(DOCKER_TAG) \
		-t $(DOCKER_LATEST) \
		.
	$(call print_success,"Docker image built: $(DOCKER_TAG)")

build-dev: ## Build development Docker image
	$(call print_info,"Building development Docker image...")
	$(DOCKER) build --target development -t $(PROJECT_NAME):dev .
	$(call print_success,"Development Docker image built!")

build-gpu: ## Build GPU-enabled Docker image
	$(call print_info,"Building GPU-enabled Docker image...")
	$(DOCKER) build --target gpu-runtime -t $(PROJECT_NAME):gpu .
	$(call print_success,"GPU Docker image built!")

push: build ## Build and push Docker image to registry
	$(call print_info,"Pushing Docker image to registry...")
	$(DOCKER) push $(DOCKER_TAG)
	$(DOCKER) push $(DOCKER_LATEST)
	$(call print_success,"Docker image pushed!")

# Docker Compose Operations
up: ## Start services with docker-compose
	$(call print_info,"Starting services...")
	$(DOCKER_COMPOSE) up -d
	$(call print_success,"Services started! App available at http://localhost:8080")

up-dev: ## Start development services
	$(call print_info,"Starting development services...")
	$(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.dev.yml up -d
	$(call print_success,"Development services started!")

down: ## Stop services
	$(call print_info,"Stopping services...")
	$(DOCKER_COMPOSE) down
	$(call print_success,"Services stopped!")

down-volumes: ## Stop services and remove volumes
	$(call print_warning,"Stopping services and removing volumes...")
	$(DOCKER_COMPOSE) down -v
	$(call print_success,"Services stopped and volumes removed!")

logs: ## Show service logs
	$(DOCKER_COMPOSE) logs -f

logs-app: ## Show application logs only
	$(DOCKER_COMPOSE) logs -f app

shell: ## Open shell in running container
	$(DOCKER_COMPOSE) exec app bash

shell-dev: ## Open shell in development container
	$(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.dev.yml exec app bash

# ML Operations
train: ## Train model with dummy data
	$(call print_info,"Training model...")
	$(PYTHON) -m src.train_engine --use_dummy_data --epochs 5 --batch_size 16

train-real: ## Train model with real data (requires data setup)
	$(call print_info,"Training model with real data...")
	$(PYTHON) -m src.train_engine --train_dir data/train --val_dir data/val --epochs 20

inference: ## Run inference on test data
	$(call print_info,"Running inference...")
	$(PYTHON) -m src.inference --model_path saved_models/pneumonia_cnn_v1.keras --data_dir data/test --output_csv predictions.csv

gradcam: ## Generate Grad-CAM visualization
	$(call print_info,"Generating Grad-CAM visualization...")
	$(PYTHON) -m src.predict_utils --model_path saved_models/pneumonia_cnn_v1.keras --img_path sample.jpg --output_path gradcam.png

benchmark: ## Run performance benchmarks
	$(call print_info,"Running performance benchmarks...")
	$(PYTHON) -m src.performance_benchmark --model_path saved_models/pneumonia_cnn_v1.keras

validate: ## Validate model architecture
	$(call print_info,"Validating model architecture...")
	$(PYTHON) -m src.model_architecture_validation --model_path saved_models/pneumonia_cnn_v1.keras

# Data Operations
data-split: ## Split dataset into train/val/test
	$(call print_info,"Splitting dataset...")
	$(PYTHON) -m src.data_split --input_dir raw_data --output_dir data --val_frac 0.1 --test_frac 0.1

data-stats: ## Generate dataset statistics
	$(call print_info,"Generating dataset statistics...")
	$(PYTHON) -m src.dataset_stats data/ --plot_png dataset_stats.png

# MLflow Operations
serve-mlflow: ## Start MLflow tracking server
	$(call print_info,"Starting MLflow server...")
	mlflow server --host 0.0.0.0 --port 5000 &
	$(call print_success,"MLflow server started at http://localhost:5000")

# API Operations
serve-api: ## Start FastAPI development server
	$(call print_info,"Starting API server...")
	$(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload

serve-api-prod: ## Start production API server
	$(call print_info,"Starting production API server...")
	gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8080

# Jupyter Operations
jupyter: ## Start Jupyter Lab server
	$(call print_info,"Starting Jupyter Lab...")
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Documentation
docs: ## Generate documentation
	$(call print_info,"Generating documentation...")
	$(PYTHON) -m sphinx-build -b html docs/ docs/_build/html
	$(call print_success,"Documentation generated in docs/_build/html/")

serve-docs: docs ## Serve documentation locally
	$(call print_info,"Serving documentation...")
	$(PYTHON) -m http.server 8000 --directory docs/_build/html
	$(call print_success,"Documentation available at http://localhost:8000")

# Monitoring
monitor: ## Start monitoring stack
	$(call print_info,"Starting monitoring stack...")
	$(DOCKER_COMPOSE) -f docker-compose.monitoring.yml up -d
	$(call print_success,"Monitoring available at http://localhost:3000 (Grafana)")

# Security and Compliance
audit: ## Run comprehensive security audit
	$(call print_info,"Running comprehensive security audit...")
	$(PYTHON) -m src.dependency_security_scan
	$(MAKE) security
	$(call print_success,"Security audit completed!")

sbom: ## Generate Software Bill of Materials
	$(call print_info,"Generating SBOM...")
	$(PYTHON) -m pip-licenses --format=json --output-file=.artifacts/sbom.json
	$(call print_success,"SBOM generated!")

# Release Management
release: ## Create a new release
	$(call print_info,"Creating new release...")
	@echo "Current version: $(VERSION)"
	@read -p "Enter new version: " NEW_VERSION && \
	sed -i 's/version = "$(VERSION)"/version = "'$$NEW_VERSION'"/' pyproject.toml && \
	git add pyproject.toml && \
	git commit -m "bump: version $(VERSION) → $$NEW_VERSION" && \
	git tag -a "v$$NEW_VERSION" -m "Release version $$NEW_VERSION" && \
	echo "$(GREEN)Release $$NEW_VERSION created! Push with: git push origin main --tags$(NC)"

# Cleanup
clean: ## Clean up build artifacts and caches
	$(call print_info,"Cleaning up...")
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ htmlcov/ .coverage .coverage.*
	rm -f bandit-report.* safety-report.* pip-audit-report.*
	$(call print_success,"Cleanup completed!")

clean-docker: ## Clean up Docker images and containers
	$(call print_warning,"Cleaning up Docker resources...")
	$(DOCKER) system prune -f
	$(DOCKER) image prune -f
	$(call print_success,"Docker cleanup completed!")

# Environment Management
env-create: ## Create .env file from template
	$(call print_info,"Creating .env file...")
	cp .env.example .env
	$(call print_success,".env file created! Please update with your settings.")

env-check: ## Check environment configuration
	$(call print_info,"Checking environment configuration...")
	$(PYTHON) -c "from src.config import load_config; print('Environment configuration OK')"

# CI/CD Helpers
ci-install: ## Install dependencies for CI
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt -r requirements-dev.txt
	$(PIP) install -e .

ci-test: ## Run tests for CI
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=xml --cov-report=term --timeout=300

ci-lint: ## Run linting for CI
	$(PYTHON) -m ruff check src/ tests/
	$(PYTHON) -m black --check src/ tests/
	$(PYTHON) -m mypy src/

ci-security: ## Run security checks for CI
	$(PYTHON) -m bandit -r src/ -f json -o bandit-report.json
	$(PYTHON) -m safety check
	$(PYTHON) -m pip-audit

ci-build: ## Build for CI
	$(PYTHON) -m build
	$(DOCKER) build -t test-image .

# Health checks
health: ## Check system health
	$(call print_info,"Checking system health...")
	@echo "$(YELLOW)Python version:$(NC)"
	$(PYTHON) --version
	@echo "$(YELLOW)Package version:$(NC)"
	$(PYTHON) -c "import src; print(src.__version__)" 2>/dev/null || echo "Package not installed"
	@echo "$(YELLOW)Dependencies:$(NC)"
	$(PIP) check
	$(call print_success,"Health check completed!")

# Version information
version: ## Show version information
	@echo "$(BLUE)Chest X-Ray Pneumonia Detector$(NC)"
	@echo "Version: $(GREEN)$(VERSION)$(NC)"
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Git commit: $$(git rev-parse --short HEAD 2>/dev/null || echo 'not available')"
	@echo "Build date: $$(date -u +'%Y-%m-%dT%H:%M:%SZ')"