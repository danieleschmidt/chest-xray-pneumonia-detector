# Advanced Deployment Guide

This guide covers the advanced deployment capabilities implemented for the Chest X-Ray Pneumonia Detection system, designed for production healthcare environments.

## Overview

The enhanced SDLC implementation provides enterprise-grade deployment strategies with:

- **Advanced CI/CD Pipeline**: Matrix testing across Python versions and OS platforms
- **Production-Ready API**: FastAPI with comprehensive monitoring and security
- **Kubernetes Deployment**: Full orchestration with auto-scaling and health checks
- **Blue-Green Deployment**: Zero-downtime deployments with automatic rollback
- **Advanced Monitoring**: ML-specific metrics, bias detection, and drift monitoring
- **Security**: Container scanning, SARIF reports, and comprehensive security policies

## Quick Start

### 1. Enhanced CI/CD Pipeline

The new `docs/workflows/advanced-ci.yml` template provides:

> **Note**: Copy this file to `.github/workflows/advanced-ci.yml` to enable advanced CI/CD.
> Manual copy required due to GitHub security restrictions on workflow modifications.

```bash
# Triggers matrix testing across Python 3.8-3.12 and multiple OS
git push origin main

# Manual deployment trigger
gh workflow run "Advanced CI/CD Pipeline" --ref main
```

**Features:**
- Matrix testing (Python 3.8-3.12, Ubuntu/macOS)
- Advanced security scanning (Bandit, Trivy, Semgrep)
- Container vulnerability scanning
- SBOM generation
- Blue-green deployment to staging/production

### 2. Production API Deployment

Deploy the FastAPI application:

```bash
# Local development
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Production deployment
python scripts/automated-model-deployment.py \
    --model-version v1.2.0 \
    --strategy blue_green
```

**API Features:**
- Asynchronous request handling
- Batch prediction endpoints
- Comprehensive health checks
- Prometheus metrics integration
- Rate limiting and security middleware
- Model version management

### 3. Kubernetes Deployment

Deploy to Kubernetes cluster:

```bash
# Apply all manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n pneumonia-detection

# Scale deployment
kubectl scale deployment pneumonia-detection-api --replicas=5 -n pneumonia-detection
```

**Kubernetes Features:**
- HPA with CPU/memory/custom metrics
- Pod disruption budgets
- Network policies for security
- Persistent volume for model storage
- Service mesh integration ready

## Deployment Strategies

### Blue-Green Deployment

Zero-downtime deployment with full rollback capability:

```bash
python scripts/automated-model-deployment.py \
    --model-version v1.3.0 \
    --strategy blue_green \
    --config deployment-config.yaml
```

**Process:**
1. Deploy new version to "green" environment
2. Health checks and smoke tests
3. Gradual traffic switching (10% → 100%)
4. Monitor metrics for 5 minutes
5. Automatic rollback if issues detected

### Canary Deployment

Gradual rollout with risk mitigation:

```bash
python scripts/automated-model-deployment.py \
    --model-version v1.3.0 \
    --strategy canary
```

**Process:**
1. Deploy to 10% of traffic
2. Monitor canary metrics
3. Gradually increase to 25%, 50%, 75%, 100%
4. Automatic rollback on anomalies

### Rolling Deployment

Standard Kubernetes rolling update:

```bash
python scripts/automated-model-deployment.py \
    --model-version v1.3.0 \
    --strategy rolling
```

## Monitoring and Observability

### Advanced Dashboards

Grafana dashboards include:

- **Model Performance**: Prediction rate, latency, error rate
- **ML Metrics**: Confidence distribution, prediction classes
- **Bias Detection**: Demographic fairness metrics
- **Drift Detection**: Model performance degradation
- **Infrastructure**: CPU, memory, network utilization

Access: `http://grafana.yourdomain.com/dashboard/ml-model-performance`

### Alerting

Comprehensive alerting rules:

```yaml
# High-level alerts
- ModelHighErrorRate: > 5% error rate
- ModelLatencyHigh: > 2s 95th percentile
- ModelDriftDetected: Drift score > 0.7
- BiasDetected: Demographic bias > 0.8

# Infrastructure alerts  
- HighCPUUsage: > 80% CPU utilization
- HighMemoryUsage: > 85% memory utilization
- PodRestartingTooOften: Frequent restarts
```

### Metrics Collection

Key metrics automatically collected:

```python
# Prometheus metrics
predictions_total{model_version, status}
prediction_duration_seconds
model_confidence_bucket
bias_detection_score{demographic_group}
data_quality_score
model_drift_score
```

## Security

### Container Security

- Non-root user execution
- Read-only root filesystem  
- Security context constraints
- Vulnerability scanning in CI
- Base image security updates

### Network Security

- Network policies restrict pod communication
- TLS termination at ingress
- Rate limiting (100 req/min per IP)
- CORS configuration for production
- Security headers (HSTS, CSP, etc.)

### Data Security

- PHI encryption with AES-256
- Secure model storage
- Audit logging for all predictions
- HIPAA compliance measures
- Data retention policies

## Performance Optimization

### Auto-scaling

Horizontal Pod Autoscaler configuration:

```yaml
# Scale based on:
- CPU utilization: > 70%
- Memory utilization: > 80%  
- Custom metric: > 100 req/sec per pod

# Scale range: 3-20 replicas
# Scale-up: 50% increase every 60s
# Scale-down: 10% decrease every 300s
```

### Resource Optimization

Optimized resource allocation:

```yaml
requests:
  cpu: 500m      # 0.5 CPU cores
  memory: 1Gi    # 1GB RAM
limits:
  cpu: 2000m     # 2 CPU cores  
  memory: 4Gi    # 4GB RAM
```

### Caching Strategy

Model and prediction caching:

- Model loading cache (in-memory)
- Response caching for identical images
- Redis cluster for distributed caching
- Cache invalidation on model updates

## Disaster Recovery

### Backup Strategy

- Automated model backups to S3
- Database backups every 6 hours
- Configuration backup with GitOps
- Cross-region replication

### Recovery Procedures

```bash
# Rollback deployment
python scripts/automated-model-deployment.py \
    --rollback deploy-v1.2.0-1643723400

# Restore from backup
kubectl apply -f backup/k8s-manifests/
```

### Business Continuity

- Multi-region deployment capability
- Load balancer health checks
- Automated failover procedures
- SLA: 99.9% availability target

## Cost Optimization

### Resource Management

- Spot instances for non-critical workloads
- Vertical pod autoscaling
- Resource quotas and limits
- Node auto-scaling based on demand

### Monitoring Costs

```bash
# Cost monitoring queries
sum(rate(container_cpu_usage_seconds_total[5m])) * 0.05  # CPU cost
sum(container_memory_usage_bytes) / 1024^3 * 0.01       # Memory cost
```

## Compliance

### HIPAA Compliance

- Encrypted data at rest and in transit
- Audit logging for all access
- User authentication and authorization
- Business Associate Agreements (BAA)
- Regular compliance audits

### SOC 2 Compliance

- Access controls implementation
- Change management procedures
- Incident response procedures
- Regular security assessments

## Troubleshooting

### Common Issues

1. **Model Loading Failures**
   ```bash
   kubectl logs deployment/pneumonia-detection-api -n pneumonia-detection
   ```

2. **High Latency**
   ```bash
   # Check resource utilization
   kubectl top pods -n pneumonia-detection
   
   # Scale up if needed
   kubectl scale deployment pneumonia-detection-api --replicas=10
   ```

3. **Health Check Failures**
   ```bash
   # Test health endpoint
   curl http://api.pneumonia-detector.yourdomain.com/health
   
   # Check pod status
   kubectl describe pod <pod-name> -n pneumonia-detection
   ```

### Debug Mode

Enable debug logging:

```bash
kubectl set env deployment/pneumonia-detection-api LOG_LEVEL=DEBUG -n pneumonia-detection
```

## Advanced Features

### A/B Testing Framework

Deploy multiple model versions simultaneously:

```bash
# Deploy model A to 80% traffic
# Deploy model B to 20% traffic
python scripts/ab-test-deployment.py \
    --model-a v1.2.0 --traffic-a 80 \
    --model-b v1.3.0 --traffic-b 20
```

### Model Versioning

Semantic versioning with automatic promotion:

```bash
# Promote model from staging to production
python scripts/model-promotion.py \
    --source staging \
    --target production \
    --model-version v1.3.0
```

### Multi-Region Deployment

Deploy across multiple regions:

```bash
# Deploy to us-east-1, us-west-2, eu-west-1
python scripts/multi-region-deploy.py \
    --regions us-east-1,us-west-2,eu-west-1 \
    --model-version v1.3.0
```

This advanced deployment system provides enterprise-grade capabilities suitable for production healthcare ML systems, with comprehensive monitoring, security, and operational excellence built-in.