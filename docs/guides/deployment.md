# Deployment Guide

This guide covers deployment strategies and procedures for the Chest X-Ray Pneumonia Detector in various environments.

## Table of Contents

- [Overview](#overview)
- [Local Development](#local-development)
- [Staging Environment](#staging-environment)
- [Production Deployment](#production-deployment)
- [Container Orchestration](#container-orchestration)
- [Configuration Management](#configuration-management)
- [Monitoring and Observability](#monitoring-and-observability)
- [Security Considerations](#security-considerations)
- [Backup and Recovery](#backup-and-recovery)
- [Troubleshooting](#troubleshooting)

## Overview

The Chest X-Ray Pneumonia Detector supports multiple deployment patterns:

- **Local Development**: Docker Compose for local testing
- **Staging**: Containerized deployment with basic monitoring
- **Production**: High-availability deployment with full monitoring
- **Cloud Platforms**: AWS, GCP, Azure deployment options

## Local Development

### Quick Start with Docker Compose

```bash
# Clone repository
git clone https://github.com/your-org/chest-xray-pneumonia-detector.git
cd chest-xray-pneumonia-detector

# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f

# Access services
# - MLflow UI: http://localhost:5000
# - Prometheus: http://localhost:9090 (with monitoring profile)
# - Grafana: http://localhost:3000 (with monitoring profile)
```

### Development Services

```bash
# Start with monitoring
docker-compose --profile monitoring up -d

# Run inference test
docker-compose exec api python -m src.inference --help

# Check health
docker-compose exec api python -m src.monitoring.health_checks
```

## Staging Environment

### Prerequisites

- Docker and Docker Compose
- At least 4GB RAM
- 20GB available disk space
- Network access for container registry

### Staging Deployment

```bash
# Create staging directory
mkdir -p /opt/pneumonia-detector-staging
cd /opt/pneumonia-detector-staging

# Download docker-compose files
curl -O https://raw.githubusercontent.com/your-org/chest-xray-pneumonia-detector/main/docker-compose.yml
curl -O https://raw.githubusercontent.com/your-org/chest-xray-pneumonia-detector/main/docker-compose.override.yml

# Create environment file
cat > .env << EOF
# Staging Environment Configuration
ENVIRONMENT=staging
LOG_LEVEL=INFO
MLFLOW_TRACKING_URI=postgresql://mlflow:mlflow@postgres:5432/mlflow

# Model Configuration
MODEL_PATH=/app/saved_models/pneumonia_cnn_v1.keras

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Security
ENABLE_CONSOLE_LOGGING=false
EOF

# Pull latest images
docker-compose pull

# Start services
docker-compose up -d

# Wait for services to be ready
sleep 30

# Verify deployment
docker-compose exec api python -m src.monitoring.health_checks --check all
```

### Staging Verification

```bash
# Check service health
curl http://localhost:8000/health

# Run test inference (when API is implemented)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image_path": "/path/to/test/image.jpg"}'

# Check metrics
curl http://localhost:8000/metrics
```

## Production Deployment

### Architecture Overview

```
                    [Load Balancer]
                          |
            [API Gateway / Reverse Proxy]
                          |
        +------------------+------------------+
        |                                     |
   [API Service]                     [API Service]
        |                                     |
        +------------[Database]---------------+
                          |
                 [Model Storage]
                          |
                 [Monitoring Stack]
```

### Infrastructure Requirements

**Minimum Production Requirements:**
- 2 CPU cores per API instance
- 4GB RAM per API instance
- 50GB storage for models and data
- Load balancer (nginx, HAProxy, or cloud LB)
- PostgreSQL database
- Redis for caching (optional)

**Recommended Production Setup:**
- 4 CPU cores per API instance
- 8GB RAM per API instance
- 100GB SSD storage
- Multi-region deployment
- Managed database service
- CDN for static assets

### Production Docker Deployment

#### 1. Create Production Environment

```bash
# Create production directory
sudo mkdir -p /opt/pneumonia-detector
cd /opt/pneumonia-detector

# Create production docker-compose file
cat > docker-compose.prod.yml << 'EOF'
version: '3.8'

services:
  api:
    image: ghcr.io/your-org/pneumonia-detector:latest
    restart: unless-stopped
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
      - DATABASE_URL=postgresql://pneumonia:secure_password@db:5432/pneumonia_prod
      - MODEL_PATH=/app/models/pneumonia_cnn_v1.keras
    volumes:
      - model-data:/app/models:ro
      - app-logs:/app/logs
    ports:
      - "8000:8000"
    depends_on:
      - db
      - redis
    healthcheck:
      test: ["CMD", "python", "-m", "src.monitoring.health_checks", "--check", "liveness"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: postgres:13
    restart: unless-stopped
    environment:
      - POSTGRES_DB=pneumonia_prod
      - POSTGRES_USER=pneumonia
      - POSTGRES_PASSWORD=secure_password
    volumes:
      - db-data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:6-alpine
    restart: unless-stopped
    volumes:
      - redis-data:/data
    ports:
      - "6379:6379"

  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - api

  prometheus:
    image: prom/prometheus
    restart: unless-stopped
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    restart: unless-stopped
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=secure_admin_password
    volumes:
      - grafana-data:/var/lib/grafana
    ports:
      - "3000:3000"

volumes:
  model-data:
    driver: local
  app-logs:
    driver: local
  db-data:
    driver: local
  redis-data:
    driver: local
  prometheus-data:
    driver: local
  grafana-data:
    driver: local
EOF
```

#### 2. Configure Nginx

```bash
# Create nginx configuration
cat > nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream api_backend {
        server api:8000;
    }

    server {
        listen 80;
        server_name your-domain.com;
        
        # Redirect to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";

        # API endpoints
        location /api/ {
            proxy_pass http://api_backend/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        # Health check
        location /health {
            proxy_pass http://api_backend/health;
            access_log off;
        }

        # Metrics (restrict access)
        location /metrics {
            proxy_pass http://api_backend/metrics;
            allow 10.0.0.0/8;
            allow 172.16.0.0/12;
            allow 192.168.0.0/16;
            deny all;
        }
    }
}
EOF
```

#### 3. Deploy to Production

```bash
# Set secure permissions
sudo chown -R root:root /opt/pneumonia-detector
sudo chmod 600 /opt/pneumonia-detector/.env

# Pull latest images
docker-compose -f docker-compose.prod.yml pull

# Start production services
docker-compose -f docker-compose.prod.yml up -d

# Verify deployment
docker-compose -f docker-compose.prod.yml ps
docker-compose -f docker-compose.prod.yml logs --tail=50
```

### Production Verification

```bash
# Health check
curl -f https://your-domain.com/health

# API test
curl -X POST https://your-domain.com/api/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{"image_path": "/path/to/test/image.jpg"}'

# Metrics (from internal network)
curl http://localhost:9090/metrics
```

## Container Orchestration

### Kubernetes Deployment

#### 1. Namespace and ConfigMap

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: pneumonia-detector

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: pneumonia-detector-config
  namespace: pneumonia-detector
data:
  LOG_LEVEL: "INFO"
  ENVIRONMENT: "production"
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
```

#### 2. Secrets

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: pneumonia-detector-secrets
  namespace: pneumonia-detector
type: Opaque
data:
  DATABASE_URL: <base64-encoded-database-url>
  API_KEY: <base64-encoded-api-key>
```

#### 3. Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pneumonia-detector
  namespace: pneumonia-detector
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pneumonia-detector
  template:
    metadata:
      labels:
        app: pneumonia-detector
    spec:
      containers:
      - name: api
        image: ghcr.io/your-org/pneumonia-detector:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: pneumonia-detector-config
        - secretRef:
            name: pneumonia-detector-secrets
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
          readOnly: true
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-storage-pvc
```

#### 4. Service and Ingress

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: pneumonia-detector-service
  namespace: pneumonia-detector
spec:
  selector:
    app: pneumonia-detector
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: pneumonia-detector-ingress
  namespace: pneumonia-detector
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.your-domain.com
    secretName: pneumonia-detector-tls
  rules:
  - host: api.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: pneumonia-detector-service
            port:
              number: 80
```

#### 5. Deploy to Kubernetes

```bash
# Apply configurations
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# Verify deployment
kubectl get pods -n pneumonia-detector
kubectl get services -n pneumonia-detector
kubectl get ingress -n pneumonia-detector

# Check logs
kubectl logs -n pneumonia-detector deployment/pneumonia-detector
```

## Configuration Management

### Environment Variables

```bash
# Production environment variables
export ENVIRONMENT=production
export LOG_LEVEL=INFO
export LOG_FORMAT=structured

# Database configuration
export DATABASE_URL=postgresql://user:pass@host:port/db
export DATABASE_POOL_SIZE=20
export DATABASE_MAX_OVERFLOW=30

# Model configuration
export MODEL_PATH=/app/models/pneumonia_cnn_v1.keras
export MODEL_CACHE_SIZE=100
export BATCH_SIZE=32

# API configuration
export API_HOST=0.0.0.0
export API_PORT=8000
export API_WORKERS=4
export API_TIMEOUT=30

# Security configuration
export API_KEY_REQUIRED=true
export RATE_LIMIT_PER_MINUTE=100
export CORS_ORIGINS=https://your-domain.com

# Monitoring configuration
export METRICS_ENABLED=true
export HEALTH_CHECK_ENABLED=true
export PROMETHEUS_PORT=9090
```

### Configuration Files

```yaml
# config/production.yaml
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  timeout: 30

database:
  url: "postgresql://user:pass@host:port/db"
  pool_size: 20
  max_overflow: 30

model:
  path: "/app/models/pneumonia_cnn_v1.keras"
  cache_size: 100
  batch_size: 32

logging:
  level: "INFO"
  format: "structured"
  file: "/app/logs/application.log"

monitoring:
  metrics_enabled: true
  health_checks_enabled: true
  prometheus_port: 9090

security:
  api_key_required: true
  rate_limit_per_minute: 100
  cors_origins:
    - "https://your-domain.com"
```

## Monitoring and Observability

### Health Checks

```bash
# Application health
curl -f http://localhost:8000/health

# Readiness probe
curl -f http://localhost:8000/ready

# Liveness probe
curl -f http://localhost:8000/alive

# Detailed health check
python -m src.monitoring.health_checks --check all --format json
```

### Metrics Collection

```bash
# Prometheus metrics
curl http://localhost:8000/metrics

# Application metrics
curl http://localhost:8000/metrics | grep pneumonia_detector

# Custom metrics via CLI
python -m src.monitoring.metrics --serve --port 9090
```

### Logging

```bash
# Structured logs
tail -f /var/log/pneumonia-detector/application.log | jq .

# Error logs
grep '"level":"ERROR"' /var/log/pneumonia-detector/application.log

# Performance logs
grep '"event_type":"http_request"' /var/log/pneumonia-detector/application.log
```

### Alerting

Configure alerts for:
- High error rates
- High response times
- Service availability
- Resource utilization
- Security events

## Security Considerations

### Network Security

```bash
# Firewall rules (example for UFW)
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 80/tcp      # HTTP
sudo ufw allow 443/tcp     # HTTPS
sudo ufw allow from 10.0.0.0/8 to any port 9090  # Prometheus (internal only)
sudo ufw enable
```

### Container Security

```yaml
# Security context for Kubernetes
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  runAsGroup: 1000
  fsGroup: 1000
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  capabilities:
    drop:
      - ALL
```

### TLS Configuration

```bash
# Generate self-signed certificate (for testing)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout key.pem -out cert.pem \
  -subj "/C=US/ST=State/L=City/O=Org/CN=your-domain.com"

# Use Let's Encrypt for production
certbot certonly --webroot -w /var/www/html -d your-domain.com
```

## Backup and Recovery

### Database Backup

```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups/pneumonia-detector"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup database
pg_dump $DATABASE_URL > $BACKUP_DIR/db_backup_$DATE.sql

# Backup models
tar -czf $BACKUP_DIR/models_backup_$DATE.tar.gz /app/models/

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "*.sql" -mtime +30 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

### Model Backup

```bash
# Backup model registry
rsync -av /app/models/ user@backup-server:/backups/models/

# Backup MLflow artifacts
rsync -av /app/mlruns/ user@backup-server:/backups/mlruns/
```

### Recovery Procedures

```bash
# Database recovery
psql $DATABASE_URL < /backups/db_backup_20231201_120000.sql

# Model recovery
tar -xzf /backups/models_backup_20231201_120000.tar.gz -C /

# Container recovery
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d
```

## Troubleshooting

### Common Issues

#### 1. Service Won't Start

```bash
# Check logs
docker-compose logs service-name

# Check resource usage
docker stats

# Check configuration
docker-compose config
```

#### 2. High Memory Usage

```bash
# Monitor memory
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Adjust container limits
# In docker-compose.yml:
deploy:
  resources:
    limits:
      memory: 2G
    reservations:
      memory: 1G
```

#### 3. Database Connection Issues

```bash
# Test database connectivity
docker-compose exec api python -c "
import psycopg2
conn = psycopg2.connect('$DATABASE_URL')
print('Connection successful')
conn.close()
"

# Check database logs
docker-compose logs db
```

#### 4. Model Loading Errors

```bash
# Check model file
docker-compose exec api ls -la /app/models/

# Test model loading
docker-compose exec api python -c "
import tensorflow as tf
model = tf.keras.models.load_model('/app/models/pneumonia_cnn_v1.keras')
print('Model loaded successfully')
"
```

### Performance Optimization

```bash
# Optimize Docker images
docker build --target production --tag pneumonia-detector:optimized .

# Use multi-stage builds
# Optimize layer caching
# Remove unnecessary dependencies

# Database optimization
# Add indexes
# Optimize queries
# Configure connection pooling

# Application optimization
# Enable caching
# Optimize batch sizes
# Use async processing
```

### Monitoring Deployment Health

```bash
# Comprehensive health check script
#!/bin/bash
echo "=== Deployment Health Check ==="

# Service availability
curl -f http://localhost:8000/health || echo "❌ Health check failed"

# Database connectivity
docker-compose exec api python -c "
try:
    import psycopg2
    conn = psycopg2.connect('$DATABASE_URL')
    conn.close()
    print('✅ Database connection OK')
except Exception as e:
    print(f'❌ Database connection failed: {e}')
"

# Model availability
docker-compose exec api python -c "
try:
    import tensorflow as tf
    model = tf.keras.models.load_model('/app/models/pneumonia_cnn_v1.keras')
    print('✅ Model loading OK')
except Exception as e:
    print(f'❌ Model loading failed: {e}')
"

# Metrics endpoint
curl -f http://localhost:8000/metrics > /dev/null && echo "✅ Metrics OK" || echo "❌ Metrics failed"

echo "=== Health Check Complete ==="
```

For additional support, refer to the [troubleshooting documentation](../runbooks/troubleshooting.md) or create an issue in the repository.