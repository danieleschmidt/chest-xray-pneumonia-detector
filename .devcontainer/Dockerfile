ARG VARIANT="3.11-bullseye"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

# Install system dependencies for medical imaging and ML
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    build-essential \
    curl \
    git \
    graphviz \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk-3-dev \
    pkg-config \
    python3-dev \
    python3-opencv \
    sqlite3 \
    wget \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements*.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install \
    -r /tmp/pip-tmp/requirements.txt \
    -r /tmp/pip-tmp/requirements-dev.txt \
    && rm -rf /tmp/pip-tmp

# Install additional development tools
RUN pip3 install \
    jupyterlab \
    pre-commit \
    bandit[toml] \
    safety \
    pip-audit \
    mypy \
    types-requests \
    types-Pillow \
    ipywidgets \
    nbconvert

# Install MLflow server dependencies
RUN pip3 install \
    mlflow[extras] \
    boto3 \
    psycopg2-binary \
    pymongo

# Create workspace directory
RUN mkdir -p /workspace
WORKDIR /workspace

# Set environment variables for development
ENV PYTHONPATH=/workspace/src
ENV MLFLOW_TRACKING_URI=http://localhost:5000
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Configure git for development container
RUN git config --global --add safe.directory /workspace