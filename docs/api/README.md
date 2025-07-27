# API Documentation

## Overview

The Chest X-Ray Pneumonia Detector API provides a RESTful interface for pneumonia detection from chest X-ray images. The API is built with FastAPI and includes comprehensive documentation, monitoring, and security features.

**Base URL**: `http://localhost:8080` (development)

**API Version**: v0.2.0

## Authentication

### Development Environment
No authentication required for development.

### Production Environment
The API supports multiple authentication methods:

- **JWT Tokens**: Bearer token authentication
- **API Keys**: X-API-Key header authentication
- **OAuth 2.0**: For third-party integrations

```bash
# JWT Authentication
curl -H "Authorization: Bearer your-jwt-token" http://api-url/predict

# API Key Authentication
curl -H "X-API-Key: your-api-key" http://api-url/predict
```

## Core Endpoints

### Health Check

Check the health status of the API and its dependencies.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-07-27T10:30:00Z",
  "checks": [
    {
      "name": "model_availability",
      "status": "healthy",
      "message": "Model loaded successfully",
      "duration_ms": 5
    },
    {
      "name": "disk_space",
      "status": "healthy",
      "message": "Sufficient disk space available",
      "duration_ms": 2
    }
  ],
  "uptime_seconds": 3600,
  "version": "0.2.0"
}
```

**Example**:
```bash
curl http://localhost:8080/health
```

### Readiness Check

Kubernetes readiness probe endpoint.

**Endpoint**: `GET /health/ready`

**Response**:
```json
{
  "status": "ready"
}
```

### Liveness Check

Kubernetes liveness probe endpoint.

**Endpoint**: `GET /health/live`

**Response**:
```json
{
  "status": "alive"
}
```

## Prediction Endpoints

### Single Image Prediction

Predict pneumonia from a single chest X-ray image.

**Endpoint**: `POST /predict`

**Content-Type**: `multipart/form-data`

**Parameters**:
- `file` (required): Image file (JPEG, PNG, DICOM)

**Response**:
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

**Example**:
```bash
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@chest_xray.jpg"
```

**Python Example**:
```python
import requests

url = "http://localhost:8080/predict"
files = {"file": ("chest_xray.jpg", open("chest_xray.jpg", "rb"), "image/jpeg")}

response = requests.post(url, files=files)
result = response.json()

print(f"Prediction: {result['class_name']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### Batch Image Prediction

Predict pneumonia from multiple chest X-ray images in a single request.

**Endpoint**: `POST /predict/batch`

**Content-Type**: `multipart/form-data`

**Parameters**:
- `files` (required): Multiple image files (max 10 files)

**Response**:
```json
{
  "predictions": [
    {
      "filename": "chest_xray_1.jpg",
      "prediction": 0,
      "confidence": 0.92,
      "class_name": "Normal"
    },
    {
      "filename": "chest_xray_2.jpg",
      "prediction": 1,
      "confidence": 0.85,
      "class_name": "Pneumonia"
    }
  ],
  "batch_size": 2,
  "total_processing_time_ms": 450,
  "timestamp": "2025-07-27T10:30:00Z"
}
```

**Example**:
```bash
curl -X POST "http://localhost:8080/predict/batch" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"
```

**Python Example**:
```python
import requests

url = "http://localhost:8080/predict/batch"
files = [
    ("files", ("image1.jpg", open("image1.jpg", "rb"), "image/jpeg")),
    ("files", ("image2.jpg", open("image2.jpg", "rb"), "image/jpeg")),
    ("files", ("image3.jpg", open("image3.jpg", "rb"), "image/jpeg"))
]

response = requests.post(url, files=files)
results = response.json()

for pred in results["predictions"]:
    print(f"{pred['filename']}: {pred['class_name']} ({pred['confidence']:.2f})")
```

## Model Management

### Get Model Information

Retrieve information about the currently loaded model.

**Endpoint**: `GET /model/info`

**Response**:
```json
{
  "model_name": "pneumonia_detector_cnn",
  "model_version": "v1.0.0",
  "model_type": "CNN",
  "input_shape": [224, 224, 3],
  "output_shape": [1],
  "parameters": 1235467,
  "size_mb": 4.7,
  "classes": ["Normal", "Pneumonia"],
  "loaded_at": "2025-07-27T08:00:00Z"
}
```

**Example**:
```bash
curl http://localhost:8080/model/info
```

### Reload Model

Reload the model from disk (useful after model updates).

**Endpoint**: `POST /model/reload`

**Response**:
```json
{
  "message": "Model reloaded successfully",
  "timestamp": "2025-07-27T10:35:00Z"
}
```

**Example**:
```bash
curl -X POST http://localhost:8080/model/reload
```

## Monitoring Endpoints

### Metrics

Prometheus-formatted metrics for monitoring.

**Endpoint**: `GET /metrics`

**Content-Type**: `text/plain`

**Response**: Prometheus metrics format

**Example**:
```bash
curl http://localhost:8080/metrics
```

### API Information

Get basic API information and available endpoints.

**Endpoint**: `GET /`

**Response**:
```json
{
  "name": "Chest X-Ray Pneumonia Detector API",
  "version": "0.2.0",
  "description": "AI-powered pneumonia detection from chest X-ray images",
  "docs": "/docs",
  "health": "/health",
  "metrics": "/metrics"
}
```

## Error Handling

### Error Response Format

All errors follow a consistent format:

```json
{
  "error": "validation_error",
  "message": "Invalid image format",
  "timestamp": "2025-07-27T10:30:00Z",
  "details": {
    "field": "file",
    "expected": "image/*",
    "received": "text/plain"
  }
}
```

### HTTP Status Codes

- **200 OK**: Successful request
- **400 Bad Request**: Invalid input or malformed request
- **401 Unauthorized**: Authentication required
- **403 Forbidden**: Insufficient permissions
- **404 Not Found**: Endpoint not found
- **413 Payload Too Large**: File size exceeds limit
- **422 Unprocessable Entity**: Validation errors
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Server error
- **503 Service Unavailable**: Service temporarily unavailable

### Common Errors

#### Invalid Image Format

```json
{
  "error": "validation_error",
  "message": "File must be an image",
  "timestamp": "2025-07-27T10:30:00Z"
}
```

#### File Too Large

```json
{
  "error": "file_too_large",
  "message": "File size exceeds 50MB limit",
  "timestamp": "2025-07-27T10:30:00Z"
}
```

#### Model Not Available

```json
{
  "error": "model_error",
  "message": "Model is not loaded",
  "timestamp": "2025-07-27T10:30:00Z"
}
```

#### Rate Limit Exceeded

```json
{
  "error": "rate_limit_exceeded",
  "message": "Too many requests. Please try again later.",
  "timestamp": "2025-07-27T10:30:00Z"
}
```

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Default limit**: 100 requests per minute per IP
- **Burst limit**: 10 requests per second
- **Headers included in response**:
  - `X-RateLimit-Limit`: Request limit
  - `X-RateLimit-Remaining`: Remaining requests
  - `X-RateLimit-Reset`: Reset time (Unix timestamp)

## Request/Response Headers

### Request Headers

```http
Content-Type: multipart/form-data
Authorization: Bearer your-token (if authentication enabled)
X-API-Key: your-api-key (alternative authentication)
User-Agent: YourApp/1.0
```

### Response Headers

```http
Content-Type: application/json
X-Request-ID: req-12345678
X-Response-Time: 245ms
X-Model-Version: v1.0.0
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
```

## File Upload Requirements

### Supported Formats

- **JPEG**: .jpg, .jpeg
- **PNG**: .png
- **DICOM**: .dcm, .dicom

### File Size Limits

- **Maximum size**: 50MB per file
- **Batch limit**: 10 files per batch request
- **Total batch size**: 200MB maximum

### Image Requirements

- **Minimum resolution**: 150x150 pixels
- **Maximum resolution**: 4096x4096 pixels
- **Color modes**: Grayscale, RGB
- **Bit depth**: 8-bit or 16-bit

## Interactive Documentation

### Swagger UI

Access interactive API documentation at: `http://localhost:8080/docs`

Features:
- Try out API endpoints directly
- View request/response schemas
- Authentication testing
- Code generation in multiple languages

### ReDoc

Alternative documentation interface at: `http://localhost:8080/redoc`

Features:
- Clean, readable documentation
- Detailed schema descriptions
- Code examples
- Download OpenAPI specification

## SDK and Client Libraries

### Python SDK

```python
from chest_xray_detector import Client

# Initialize client
client = Client(base_url="http://localhost:8080")

# Single prediction
result = client.predict("chest_xray.jpg")
print(f"Prediction: {result.class_name} ({result.confidence:.2f})")

# Batch prediction
results = client.predict_batch(["image1.jpg", "image2.jpg"])
for result in results:
    print(f"{result.filename}: {result.class_name}")

# Model information
model_info = client.get_model_info()
print(f"Model: {model_info.model_name} v{model_info.model_version}")
```

### JavaScript SDK

```javascript
import { ChestXrayDetectorClient } from 'chest-xray-detector-js';

const client = new ChestXrayDetectorClient('http://localhost:8080');

// Single prediction
const result = await client.predict(imageFile);
console.log(`Prediction: ${result.class_name} (${result.confidence.toFixed(2)})`);

// Batch prediction
const results = await client.predictBatch([file1, file2, file3]);
results.predictions.forEach(pred => {
    console.log(`${pred.filename}: ${pred.class_name}`);
});
```

### cURL Examples

#### Health Check
```bash
curl -w "\nStatus: %{http_code}\nTime: %{time_total}s\n" \
  http://localhost:8080/health
```

#### Single Prediction with Verbose Output
```bash
curl -X POST \
  -H "Accept: application/json" \
  -F "file=@chest_xray.jpg" \
  -w "\nStatus: %{http_code}\nTime: %{time_total}s\n" \
  http://localhost:8080/predict | jq .
```

#### Batch Prediction
```bash
curl -X POST \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  http://localhost:8080/predict/batch | jq .
```

## WebSocket Support (Future Feature)

Real-time prediction updates via WebSocket:

```javascript
const ws = new WebSocket('ws://localhost:8080/ws/predict');

ws.onopen = function() {
    // Send image data for real-time prediction
    ws.send(imageData);
};

ws.onmessage = function(event) {
    const result = JSON.parse(event.data);
    console.log('Real-time prediction:', result);
};
```

## Security Considerations

### HIPAA Compliance

When `HIPAA_COMPLIANT=true`:
- All requests are logged for audit purposes
- PHI data is automatically detected and anonymized
- Encryption is enforced for all data transmission
- Access controls are strictly enforced

### Data Privacy

- Images are not stored permanently
- Temporary files are securely deleted after processing
- No patient identifying information is logged
- All processing happens in memory when possible

### Security Headers

The API includes security headers:
```http
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
Content-Security-Policy: default-src 'self'
```

## Performance Optimization

### Best Practices

1. **Use batch prediction** for multiple images
2. **Resize images** to 224x224 before upload
3. **Use appropriate image formats** (JPEG for photos)
4. **Implement client-side caching** for repeated requests
5. **Use compression** for large files

### Performance Metrics

- **Single prediction**: < 2 seconds average
- **Batch prediction**: > 100 images per minute
- **Concurrent requests**: Up to 10 simultaneous
- **Memory usage**: < 2GB for inference

## Support and Resources

### Documentation Links

- [Quick Start Guide](../guides/quick-start.md)
- [User Guide](../guides/user-guide.md)
- [Developer Guide](../guides/developer-guide.md)
- [Security Documentation](../security/README.md)

### API Status

Check API status and uptime at: `http://localhost:8080/health`

### Contact Information

- **Issues**: Report on GitHub Issues
- **Security**: security@your-organization.com
- **Support**: support@your-organization.com