# Quick Start Guide

## Overview

The Chest X-Ray Pneumonia Detector is an AI-powered system for detecting pneumonia from chest X-ray images. This guide will help you get started quickly with the system.

## Prerequisites

- Python 3.8 or higher
- Docker and Docker Compose (for containerized deployment)
- At least 4GB of RAM
- 10GB of available disk space

## Installation Methods

### Method 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/chest-xray-pneumonia-detector.git
cd chest-xray-pneumonia-detector

# Copy environment file
cp .env.example .env

# Start the services
make up
```

The API will be available at http://localhost:8080

### Method 2: Local Development

```bash
# Clone the repository
git clone https://github.com/your-org/chest-xray-pneumonia-detector.git
cd chest-xray-pneumonia-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .

# Run tests to verify installation
make test

# Train a model with dummy data
make train

# Start the API server
make serve-api
```

## Quick Usage Examples

### 1. Health Check

```bash
curl http://localhost:8080/health
```

### 2. Single Image Prediction

```bash
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/chest_xray.jpg"
```

### 3. Batch Prediction

```bash
curl -X POST "http://localhost:8080/predict/batch" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

### 4. Model Information

```bash
curl http://localhost:8080/model/info
```

## Command Line Interface

### Train a Model

```bash
# Train with dummy data (for testing)
python -m src.train_engine --use_dummy_data --epochs 5

# Train with real data
python -m src.train_engine \
  --train_dir data/train \
  --val_dir data/val \
  --epochs 20 \
  --batch_size 32
```

### Run Inference

```bash
# Single file inference
python -m src.inference \
  --model_path saved_models/pneumonia_cnn_v1.keras \
  --data_dir data/test \
  --output_csv predictions.csv
```

### Generate Grad-CAM Visualization

```bash
python -m src.predict_utils \
  --model_path saved_models/pneumonia_cnn_v1.keras \
  --img_path sample_xray.jpg \
  --output_path gradcam_output.png
```

### Dataset Statistics

```bash
python -m src.dataset_stats data/ \
  --plot_png dataset_stats.png \
  --json_output stats.json
```

## Web Interface

Visit http://localhost:8080/docs for the interactive API documentation (Swagger UI) or http://localhost:8080/redoc for ReDoc documentation.

## Monitoring

- **Metrics**: http://localhost:9090 (Prometheus)
- **Dashboards**: http://localhost:3000 (Grafana, admin/admin)
- **MLflow**: http://localhost:5000

## Configuration

Key configuration options in `.env`:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8080

# Model Configuration
DEFAULT_MODEL_PATH=saved_models/pneumonia_cnn_v1.keras
IMAGE_SIZE=224,224

# Security
HIPAA_COMPLIANT=true
AUDIT_LOG_ENABLED=true

# Performance
GPU_ENABLED=false
CPU_WORKERS=4
```

## Data Format

### Input Images
- **Formats**: JPEG, PNG, DICOM
- **Size**: Minimum 150x150 pixels
- **Channels**: Grayscale or RGB
- **File size**: Maximum 50MB

### Expected Directory Structure for Training

```
data/
├── train/
│   ├── NORMAL/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── PNEUMONIA/
│       ├── image3.jpg
│       └── image4.jpg
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

## Common Issues and Solutions

### 1. Model Not Found

**Error**: `Model file not found`

**Solution**: 
```bash
# Train a model first
make train
# Or download a pre-trained model (if available)
```

### 2. Memory Issues

**Error**: `Out of memory during training`

**Solutions**:
- Reduce batch size: `--batch_size 16`
- Use smaller image size: `--img_size 150 150`
- Enable GPU if available: `GPU_ENABLED=true`

### 3. Permission Errors

**Error**: `Permission denied accessing files`

**Solution**:
```bash
# Fix file permissions
chmod -R 755 data/
chmod -R 755 saved_models/
```

### 4. Port Already in Use

**Error**: `Port 8080 is already in use`

**Solutions**:
- Change port in `.env`: `API_PORT=8081`
- Or stop existing service: `make down`

## Security Considerations

This system handles medical data and must comply with HIPAA regulations:

1. **Enable HIPAA compliance**: Set `HIPAA_COMPLIANT=true` in `.env`
2. **Secure secrets**: Never commit real credentials to version control
3. **Enable audit logging**: Set `AUDIT_LOG_ENABLED=true`
4. **Use HTTPS**: Configure SSL certificates for production
5. **Network security**: Use firewalls and VPNs for production deployment

## Next Steps

- Read the [User Guide](user-guide.md) for detailed usage instructions
- Check the [API Documentation](../api/README.md) for complete API reference
- See [Deployment Guide](../deployment/README.md) for production deployment
- Review [Security Guide](../security/README.md) for compliance requirements

## Getting Help

- **Documentation**: Check the `docs/` directory
- **Issues**: Report bugs and feature requests on GitHub
- **Community**: Join our discussion forums
- **Support**: Contact support for enterprise customers

## Performance Expectations

### Inference Performance
- **Single image**: < 2 seconds
- **Batch processing**: > 100 images per minute
- **Memory usage**: < 2GB for inference
- **Accuracy**: > 90% on validation datasets

### System Requirements
- **Development**: 4GB RAM, 2 CPU cores
- **Production**: 8GB RAM, 4 CPU cores, SSD storage
- **GPU**: Optional, improves training speed significantly