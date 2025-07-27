# Multi-stage Docker build for Chest X-Ray Pneumonia Detector
# Stage 1: Build stage with development dependencies
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=0.2.0

# Add metadata labels
LABEL org.opencontainers.image.title="Chest X-Ray Pneumonia Detector"
LABEL org.opencontainers.image.description="AI-powered pneumonia detection from chest X-ray images"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL org.opencontainers.image.vendor="Your Organization"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.source="https://github.com/your-org/chest-xray-pneumonia-detector"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-dev \
    pkg-config \
    python3-dev \
    python3-opencv \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements*.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/
COPY pyproject.toml pytest.ini ./

# Install the package
RUN pip install --no-cache-dir -e .

# Run tests and security checks in build stage
RUN python -m pytest tests/ --tb=short -v
RUN python -m bandit -r src/ -f json -o bandit-report.json || true
RUN python -m ruff check src/ tests/ || true

# Stage 2: Production runtime stage
FROM python:3.11-slim as runtime

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src
ENV APP_ENV=production

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgomp1 \
    python3-opencv \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install only production dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code from builder stage
COPY --from=builder /app/src ./src/
COPY --from=builder /app/pyproject.toml ./

# Install the package in production mode
RUN pip install --no-cache-dir -e . --no-deps

# Create necessary directories
RUN mkdir -p /app/data /app/saved_models /app/logs /app/reports && \
    chown -R appuser:appuser /app

# Copy health check script
COPY docker/healthcheck.py ./
RUN chmod +x healthcheck.py

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python healthcheck.py || exit 1

# Default command (can be overridden)
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]

# Stage 3: Development stage (optional)
FROM builder as development

# Install additional development tools
RUN pip install --no-cache-dir \
    jupyterlab \
    pre-commit \
    mypy \
    types-requests \
    types-Pillow

# Set development environment
ENV APP_ENV=development
ENV DEBUG=true

# Expose additional ports for development
EXPOSE 8080 8888 5000

# Development command
CMD ["bash"]

# Stage 4: GPU-enabled runtime (optional)
FROM nvidia/cuda:11.8-runtime-ubuntu22.04 as gpu-runtime

# Install Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy and install requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application
COPY --from=builder /app/src ./src/
COPY --from=builder /app/pyproject.toml ./

# Install package
RUN pip install --no-cache-dir -e . --no-deps

# Create directories and set permissions
RUN mkdir -p /app/data /app/saved_models /app/logs /app/reports && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]