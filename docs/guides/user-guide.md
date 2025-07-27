# User Guide

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Using the API](#using-the-api)
4. [Command Line Tools](#command-line-tools)
5. [Model Training](#model-training)
6. [Data Management](#data-management)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Security and Compliance](#security-and-compliance)
9. [Troubleshooting](#troubleshooting)

## Overview

The Chest X-Ray Pneumonia Detector is a comprehensive AI system designed to assist healthcare professionals in detecting pneumonia from chest X-ray images. The system provides:

- **High-accuracy predictions** using state-of-the-art CNN models
- **HIPAA-compliant** data handling and storage
- **Real-time inference** capabilities
- **Batch processing** for large datasets
- **Model interpretability** through Grad-CAM visualizations
- **Comprehensive monitoring** and audit trails

## Getting Started

### System Requirements

**Minimum Requirements:**
- Python 3.8+
- 4GB RAM
- 10GB available storage
- CPU with 2+ cores

**Recommended for Production:**
- Python 3.11+
- 16GB RAM
- 100GB SSD storage
- GPU with 8GB+ VRAM (for training)
- Docker and Kubernetes support

### Installation

See the [Quick Start Guide](quick-start.md) for installation instructions.

## Using the API

### Authentication

The API supports multiple authentication methods:

```python
import requests

# For development (no auth required)
headers = {}

# For production (JWT token)
headers = {"Authorization": "Bearer your-jwt-token"}

# For API key authentication
headers = {"X-API-Key": "your-api-key"}
```

### Making Predictions

#### Single Image Prediction

```python
import requests

url = "http://localhost:8080/predict"
files = {"file": open("chest_xray.jpg", "rb")}

response = requests.post(url, files=files)
result = response.json()

print(f"Prediction: {result['class_name']}")
print(f"Confidence: {result['confidence']:.2f}")
```

**Response Example:**
```json
{
  "prediction": 1,
  "confidence": 0.87,
  "class_name": "Pneumonia",
  "model_version": "v1.0.0",
  "processing_time_ms": 245,
  "timestamp": "2025-07-27T10:30:00Z"
}
```

#### Batch Prediction

```python
import requests

url = "http://localhost:8080/predict/batch"
files = [
    ("files", open("image1.jpg", "rb")),
    ("files", open("image2.jpg", "rb")),
    ("files", open("image3.jpg", "rb"))
]

response = requests.post(url, files=files)
results = response.json()

for pred in results["predictions"]:
    print(f"{pred['filename']}: {pred['class_name']} ({pred['confidence']:.2f})")
```

#### Error Handling

```python
import requests

try:
    response = requests.post(url, files=files, timeout=30)
    response.raise_for_status()
    result = response.json()
except requests.exceptions.RequestException as e:
    print(f"API request failed: {e}")
except ValueError as e:
    print(f"Invalid JSON response: {e}")
```

### Model Management

#### Get Model Information

```python
response = requests.get("http://localhost:8080/model/info")
model_info = response.json()

print(f"Model: {model_info['model_name']} v{model_info['model_version']}")
print(f"Parameters: {model_info['parameters']:,}")
print(f"Size: {model_info['size_mb']:.1f} MB")
```

#### Reload Model

```python
response = requests.post("http://localhost:8080/model/reload")
if response.status_code == 200:
    print("Model reloaded successfully")
```

### Health Monitoring

```python
response = requests.get("http://localhost:8080/health")
health = response.json()

print(f"Status: {health['status']}")
print(f"Uptime: {health['uptime_seconds']} seconds")

for check in health['checks']:
    print(f"- {check['name']}: {check['status']}")
```

## Command Line Tools

### Training Pipeline

#### Basic Training

```bash
# Train with dummy data for testing
python -m src.train_engine \
  --use_dummy_data \
  --epochs 5 \
  --batch_size 16

# Train with real data
python -m src.train_engine \
  --train_dir data/train \
  --val_dir data/val \
  --epochs 20 \
  --batch_size 32 \
  --learning_rate 0.001
```

#### Advanced Training Options

```bash
python -m src.train_engine \
  --train_dir data/train \
  --val_dir data/val \
  --epochs 50 \
  --batch_size 32 \
  --num_classes 1 \
  --use_transfer_learning \
  --base_model_name MobileNetV2 \
  --learning_rate 0.001 \
  --dropout_rate 0.5 \
  --save_model_path saved_models/pneumonia_cnn_v2.keras \
  --checkpoint_path saved_models/best_checkpoint.keras \
  --plot_path training_history.png \
  --early_stopping_patience 10 \
  --seed 42
```

#### Training with MLflow

```bash
python -m src.train_engine \
  --train_dir data/train \
  --val_dir data/val \
  --mlflow_experiment pneumonia-detector \
  --mlflow_run_name "experiment-001" \
  --mlflow_tracking_uri http://localhost:5000
```

### Inference Pipeline

#### Single Directory Inference

```bash
python -m src.inference \
  --model_path saved_models/pneumonia_cnn_v1.keras \
  --data_dir data/test \
  --output_csv predictions.csv \
  --num_classes 1
```

#### Custom Inference

```bash
python -m src.inference \
  --model_path saved_models/pneumonia_cnn_v1.keras \
  --data_dir /path/to/xray/images \
  --output_csv results.csv \
  --batch_size 16 \
  --img_size 224 224
```

### Model Interpretability

#### Grad-CAM Visualization

```bash
python -m src.predict_utils \
  --model_path saved_models/pneumonia_cnn_v1.keras \
  --img_path chest_xray_sample.jpg \
  --last_conv_layer_name conv_pw_13_relu \
  --output_path gradcam_visualization.png \
  --img_size 224 224
```

### Data Management

#### Dataset Splitting

```bash
python -m src.data_split \
  --input_dir raw_dataset/ \
  --output_dir data/ \
  --val_frac 0.15 \
  --test_frac 0.15 \
  --seed 42 \
  --move
```

#### Dataset Statistics

```bash
python -m src.dataset_stats data/ \
  --plot_png dataset_distribution.png \
  --json_output dataset_stats.json \
  --csv_output dataset_summary.csv \
  --sort_by count
```

### Model Validation

#### Architecture Validation

```bash
python -m src.model_architecture_validation \
  --model_path saved_models/pneumonia_cnn_v1.keras \
  --test_input_shape 224 224 3 \
  --output_report validation_report.json
```

#### Performance Benchmarking

```bash
python -m src.performance_benchmark \
  --model_path saved_models/pneumonia_cnn_v1.keras \
  --data_dir data/test \
  --batch_sizes 1 4 8 16 32 \
  --num_iterations 100 \
  --output_report benchmark_results.json
```

## Model Training

### Data Preparation

#### Directory Structure

Organize your data in the following structure:

```
data/
├── train/
│   ├── NORMAL/
│   │   ├── normal_001.jpg
│   │   ├── normal_002.jpg
│   │   └── ...
│   └── PNEUMONIA/
│       ├── pneumonia_001.jpg
│       ├── pneumonia_002.jpg
│       └── ...
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

#### Image Requirements

- **Format**: JPEG, PNG, or DICOM
- **Size**: Minimum 150x150 pixels
- **Quality**: Clear, well-contrasted X-ray images
- **Labeling**: Accurate classification (Normal vs Pneumonia)

### Training Configuration

#### Model Architectures

1. **Custom CNN**: Basic convolutional neural network
2. **Transfer Learning**: Pre-trained models (VGG16, ResNet, MobileNetV2)
3. **Attention Models**: CNNs with Squeeze-and-Excitation blocks

#### Hyperparameter Tuning

```bash
# Experiment with different learning rates
for lr in 0.001 0.0001 0.00001; do
  python -m src.train_engine \
    --learning_rate $lr \
    --mlflow_run_name "lr_${lr}" \
    --epochs 20
done

# Test different batch sizes
for bs in 16 32 64; do
  python -m src.train_engine \
    --batch_size $bs \
    --mlflow_run_name "bs_${bs}" \
    --epochs 20
done
```

### Training Monitoring

#### MLflow Integration

1. Start MLflow server: `mlflow server --host 0.0.0.0 --port 5000`
2. View experiments at http://localhost:5000
3. Compare runs and track metrics
4. Download trained models

#### TensorBoard Integration

```bash
# Generate TensorBoard logs during training
python -m src.train_engine \
  --tensorboard_log_dir logs/tensorboard \
  --epochs 50

# Start TensorBoard
tensorboard --logdir logs/tensorboard --port 6006
```

## Data Management

### Data Validation

#### Automated Quality Checks

```bash
python -c "
from src.image_utils import validate_dataset
report = validate_dataset('data/train')
print(f'Valid images: {report[\"valid_images\"]}')
print(f'Invalid images: {report[\"invalid_images\"]}')
print(f'Class distribution: {report[\"class_distribution\"]}')
"
```

#### Manual Review Process

1. **Visual Inspection**: Review sample images from each class
2. **Quality Assessment**: Check image clarity and contrast
3. **Label Verification**: Ensure correct classification
4. **Bias Detection**: Check for demographic or equipment bias

### Data Augmentation

Training includes automatic data augmentation:

- **Rotation**: ±20 degrees
- **Brightness**: 0.7-1.3x
- **Zoom**: ±20%
- **Horizontal flip**: 50% probability
- **Contrast adjustment**: Automatic

### Data Privacy

#### HIPAA Compliance

```python
from src.security.compliance import hipaa_compliance

# Scan for PHI in text
phi_findings = hipaa_compliance.scan_for_phi(text_data)
print(f"PHI found: {len(phi_findings)} instances")

# Anonymize data
anonymized_text, mapping = hipaa_compliance.anonymize_text(text_data)

# Generate patient ID
patient_id = hipaa_compliance.generate_patient_id()
print(f"Anonymous ID: {patient_id}")
```

## Monitoring and Logging

### Application Monitoring

#### Metrics Collection

Available metrics at http://localhost:9090/metrics:

- **Request metrics**: HTTP requests, response times
- **Prediction metrics**: Prediction count, confidence distribution
- **System metrics**: CPU, memory, disk usage
- **Model metrics**: Model load time, inference duration
- **Error metrics**: Error count by type

#### Dashboard Setup

1. Access Grafana at http://localhost:3000
2. Login with admin/admin
3. Import pre-configured dashboards
4. Set up alerts for critical metrics

### Logging

#### Structured Logging

All logs are in JSON format for easy parsing:

```json
{
  "timestamp": "2025-07-27T10:30:00Z",
  "level": "INFO",
  "logger": "src.api.main",
  "message": "Prediction completed",
  "extra": {
    "request_id": "req-123",
    "user_id": "user-456",
    "prediction": 1,
    "confidence": 0.87,
    "processing_time_ms": 245
  }
}
```

#### Audit Logging

HIPAA-compliant audit logs track:

- Data access events
- Model inference operations
- Security events
- Administrative actions

### Performance Monitoring

#### Key Performance Indicators

- **Throughput**: Predictions per minute
- **Latency**: Average response time
- **Accuracy**: Model performance metrics
- **Availability**: System uptime percentage
- **Resource utilization**: CPU, memory, GPU usage

## Security and Compliance

### HIPAA Compliance

#### Data Protection

1. **Encryption**: AES-256 for data at rest and in transit
2. **Access Control**: Role-based authentication and authorization
3. **Audit Trails**: Comprehensive logging of all data access
4. **Data Minimization**: Only collect necessary information

#### Configuration

```env
# Enable HIPAA compliance mode
HIPAA_COMPLIANT=true
AUDIT_LOG_ENABLED=true
PHI_ENCRYPTION_KEY=your-secure-encryption-key

# Data retention (7 years for medical records)
DATA_RETENTION_DAYS=2555
```

### Security Scanning

#### Automated Security Checks

```bash
# Run comprehensive security scan
python -m src.security.scanner \
  --output-dir .security-reports \
  --verbose

# Check for secrets in code
python -c "
from src.security.scanner import SecurityScanner
scanner = SecurityScanner()
results = scanner.scan_for_secrets()
print(f'Secrets found: {results[\"secrets_found\"]}')
"
```

#### Vulnerability Management

1. **Dependency scanning**: Regular updates and security patches
2. **Code analysis**: Static analysis with Bandit
3. **Container scanning**: Docker image vulnerability assessment
4. **Penetration testing**: Regular security assessments

### Access Control

#### User Management

```python
from src.security.auth import UserManager

user_manager = UserManager()

# Create user
user_manager.create_user(
    username="doctor1",
    email="doctor1@hospital.com",
    role="clinician"
)

# Authenticate user
token = user_manager.authenticate("doctor1", "password")

# Check permissions
can_access = user_manager.check_permission(token, "view_patient_data")
```

## Troubleshooting

### Common Issues

#### Model Loading Errors

**Problem**: Model file not found or corrupted

**Solutions**:
```bash
# Verify model file exists
ls -la saved_models/

# Check model file integrity
python -c "
import tensorflow as tf
model = tf.keras.models.load_model('saved_models/pneumonia_cnn_v1.keras')
print('Model loaded successfully')
"

# Retrain model if corrupted
make train
```

#### Memory Issues

**Problem**: Out of memory during inference or training

**Solutions**:
```bash
# Reduce batch size
python -m src.train_engine --batch_size 8

# Use smaller image size
python -m src.train_engine --img_size 150 150

# Enable mixed precision training
python -m src.train_engine --use_mixed_precision
```

#### Performance Issues

**Problem**: Slow inference times

**Solutions**:
```bash
# Enable GPU acceleration
export GPU_ENABLED=true

# Optimize model for inference
python -c "
import tensorflow as tf
model = tf.keras.models.load_model('saved_models/model.keras')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
"

# Use batch processing
python -m src.inference --batch_size 32
```

#### Data Quality Issues

**Problem**: Poor model performance

**Diagnostics**:
```bash
# Check data distribution
python -m src.dataset_stats data/

# Validate image quality
python -c "
from src.image_utils import validate_dataset
report = validate_dataset('data/train')
print(report)
"

# Review training metrics
# Check MLflow at http://localhost:5000
```

### Getting Help

#### Log Analysis

1. **Application logs**: Check `logs/app.log`
2. **Error logs**: Look for ERROR level messages
3. **Audit logs**: Review `logs/audit.log` for security events
4. **System logs**: Check Docker/Kubernetes logs

#### Support Channels

- **Documentation**: Comprehensive guides in `docs/`
- **GitHub Issues**: Report bugs and feature requests
- **Community Forum**: Get help from other users
- **Enterprise Support**: Priority support for business customers

#### Performance Tuning

1. **Monitor metrics**: Use Grafana dashboards
2. **Profile code**: Use Python profiling tools
3. **Optimize database**: Index frequently queried fields
4. **Scale infrastructure**: Add more compute resources

### Best Practices

#### Development

1. **Version control**: Use Git for all code changes
2. **Testing**: Maintain >90% test coverage
3. **Code review**: Require peer review for all changes
4. **Documentation**: Keep docs up to date

#### Production

1. **Monitoring**: Set up comprehensive monitoring and alerting
2. **Backup**: Regular backups of models and data
3. **Security**: Regular security scans and updates
4. **Compliance**: Maintain HIPAA compliance documentation

#### Model Management

1. **Versioning**: Track all model versions with MLflow
2. **Validation**: Validate models before deployment
3. **A/B testing**: Test new models against current production
4. **Rollback**: Maintain ability to quickly rollback models