# Multi-stage Docker build for Chest X-Ray Pneumonia Detector
FROM python:3.10-slim-bullseye as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Development stage
FROM base as development

# Install development dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Change ownership to appuser
RUN chown -R appuser:appuser /app

USER appuser

# Default command for development
CMD ["python", "-m", "src.train_engine", "--help"]

# Production stage
FROM base as production

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt ./

# Install production dependencies only
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY pyproject.toml ./

# Install package
RUN pip install --no-cache-dir .

# Create necessary directories
RUN mkdir -p saved_models data logs && \
    chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command
CMD ["python", "-m", "src.version_cli"]

# API stage (for future web API)
FROM production as api

# Install additional API dependencies
USER root
RUN pip install --no-cache-dir fastapi uvicorn[standard]
USER appuser

# Expose port
EXPOSE 8000

# API health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# API command (placeholder for future implementation)
CMD ["python", "-c", "print('API server not yet implemented')"]

# Inference stage (optimized for inference workloads)
FROM production as inference

# Copy pre-trained models (if available)
COPY --chown=appuser:appuser saved_models/ ./saved_models/

# Set inference-specific environment variables
ENV MODEL_PATH=/app/saved_models/pneumonia_cnn_v1.keras \
    BATCH_SIZE=32 \
    IMG_SIZE=224,224

# Default inference command
CMD ["python", "-m", "src.inference", "--help"]