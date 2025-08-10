# Production Deployment Guide
## Pneumonia Detection AI System

This guide provides comprehensive instructions for deploying the Pneumonia Detection AI System to production environments with high availability, security, and monitoring.

## 🏗️ Architecture Overview

The production deployment includes:
- **API Services**: Load-balanced API instances with auto-scaling
- **Database**: PostgreSQL with backup and monitoring
- **Caching**: Redis for performance optimization
- **Monitoring**: Prometheus, Grafana, Jaeger for observability
- **Logging**: ELK stack for centralized log management
- **Security**: TLS encryption, network policies, secret management

## 📋 Prerequisites

### System Requirements
- Kubernetes cluster (v1.24+) with at least 3 nodes
- 16 GB RAM per node (minimum)
- 4 CPU cores per node (minimum)
- 500 GB storage for persistent volumes
- Load balancer support (AWS ALB, GCP LB, etc.)

### Required Tools
```bash
# Install required command-line tools
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh
```

### Environment Variables
Set these environment variables before deployment:
```bash
export DEPLOYMENT_ENV=production
export DOCKER_REGISTRY=your-registry.com
export IMAGE_TAG=v1.0.0
export NAMESPACE=pneumonia-detection
export DB_PASSWORD="secure_password_here"
export REDIS_PASSWORD="secure_redis_password"
export GRAFANA_PASSWORD="secure_grafana_password"
```

## 🚀 Deployment Methods

### Method 1: Kubernetes Native Deployment

1. **Prepare Kubernetes Cluster**
   ```bash
   # Create namespace
   kubectl create namespace pneumonia-detection
   
   # Create secrets
   kubectl create secret generic pneumonia-secrets \
     --from-literal=DB_PASSWORD="${DB_PASSWORD}" \
     --from-literal=REDIS_PASSWORD="${REDIS_PASSWORD}" \
     --namespace=pneumonia-detection
   ```

2. **Deploy Application**
   ```bash
   # Apply all Kubernetes manifests
   kubectl apply -f k8s/production-deployment.yaml
   
   # Wait for deployment to complete
   kubectl rollout status deployment/pneumonia-api -n pneumonia-detection
   ```

3. **Verify Deployment**
   ```bash
   # Check pod status
   kubectl get pods -n pneumonia-detection
   
   # Check services
   kubectl get services -n pneumonia-detection
   
   # Test health endpoints
   kubectl port-forward service/pneumonia-api-service 8080:80 -n pneumonia-detection
   curl http://localhost:8080/health
   ```

### Method 2: Docker Compose Deployment

1. **Prepare Environment**
   ```bash
   # Create required directories
   sudo mkdir -p /data/postgres /backup
   sudo chown -R 999:999 /data/postgres
   
   # Set environment variables
   cat > .env <<EOF
   DB_PASSWORD=${DB_PASSWORD}
   REDIS_PASSWORD=${REDIS_PASSWORD}
   GRAFANA_PASSWORD=${GRAFANA_PASSWORD}
   EOF
   ```

2. **Deploy Services**
   ```bash
   # Start all services
   docker-compose -f docker-compose.production.yml up -d
   
   # Check service status
   docker-compose -f docker-compose.production.yml ps
   ```

### Method 3: Automated Deployment Script

```bash
# Run automated deployment
./scripts/deploy-production.sh deploy

# For rollback if needed
./scripts/deploy-production.sh rollback

# For health checks only
./scripts/deploy-production.sh health-check
```

## 🔧 Configuration

### Application Configuration
Key configuration files:
- `config/production.yaml`: Main application settings
- `k8s/production-deployment.yaml`: Kubernetes deployment configuration
- `docker-compose.production.yml`: Docker Compose setup

### Resource Limits
Default resource allocation per component:

| Component | CPU Request | CPU Limit | Memory Request | Memory Limit |
|-----------|-------------|-----------|----------------|--------------|
| API       | 500m        | 1000m     | 1Gi            | 2Gi          |
| PostgreSQL| 500m        | 1000m     | 1Gi            | 2Gi          |
| Redis     | 250m        | 500m      | 256Mi          | 512Mi        |
| Worker    | 250m        | 500m      | 512Mi          | 1Gi          |

### Auto-Scaling Configuration
- **Minimum Replicas**: 3
- **Maximum Replicas**: 20
- **CPU Threshold**: 70%
- **Memory Threshold**: 80%
- **Scale-up Policy**: 50% increase every 60s
- **Scale-down Policy**: 10% decrease every 300s

## 📊 Monitoring and Observability

### Prometheus Metrics
Access Prometheus at: `http://<your-domain>:9090`

Key metrics to monitor:
- `http_requests_total`: Total HTTP requests
- `http_request_duration_seconds`: Request latency
- `pneumonia_predictions_total`: Total predictions made
- `model_inference_duration_seconds`: Model inference time
- `postgres_connections`: Database connections
- `redis_memory_usage_bytes`: Redis memory usage

### Grafana Dashboards
Access Grafana at: `http://<your-domain>:3000`
- Username: `admin`
- Password: Set via `GRAFANA_PASSWORD`

Pre-configured dashboards:
1. **Application Overview**: API metrics, response times, error rates
2. **Infrastructure**: CPU, memory, disk usage across all nodes
3. **Database**: PostgreSQL performance and connection metrics
4. **ML Model**: Prediction accuracy, inference times, model performance

### Distributed Tracing
Access Jaeger at: `http://<your-domain>:16686`

Traces include:
- Complete request lifecycle
- Database query performance
- Model inference timing
- External API calls

### Centralized Logging
Access Kibana at: `http://<your-domain>:5601`

Log aggregation includes:
- Application logs with structured JSON format
- Database query logs
- Security audit logs
- Performance metrics logs

## 🔒 Security Considerations

### TLS/SSL Configuration
```yaml
# Example ingress with TLS
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: pneumonia-api-ingress
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.pneumonia-detection.com
    secretName: pneumonia-api-tls
  rules:
  - host: api.pneumonia-detection.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: pneumonia-api-service
            port:
              number: 80
```

### Network Security
- **Network Policies**: Restrict inter-pod communication
- **Service Mesh**: Consider Istio for advanced traffic management
- **WAF**: Web Application Firewall for API protection
- **DDoS Protection**: Rate limiting and traffic shaping

### Data Security
- **Encryption at Rest**: All persistent volumes encrypted
- **Encryption in Transit**: TLS 1.3 for all communications
- **Secret Management**: Kubernetes secrets or external secret managers
- **Audit Logging**: All API access and data modifications logged

## 🔄 Backup and Recovery

### Database Backup
Automated daily backups:
```bash
# Manual backup
kubectl exec -n pneumonia-detection postgres-0 -- pg_dump -U pneumonia pneumonia_db | gzip > backup_$(date +%Y%m%d).sql.gz

# Restore from backup
gunzip -c backup_20231201.sql.gz | kubectl exec -i -n pneumonia-detection postgres-0 -- psql -U pneumonia pneumonia_db
```

### Model Backup
```bash
# Backup trained models
kubectl cp pneumonia-detection/pneumonia-api-pod:/app/models ./model-backup/

# Restore models
kubectl cp ./model-backup/ pneumonia-detection/pneumonia-api-pod:/app/models
```

## 📈 Performance Optimization

### Database Optimization
```sql
-- Recommended PostgreSQL settings for production
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
SELECT pg_reload_conf();
```

### Cache Configuration
```yaml
# Redis optimization
redis:
  config: |
    maxmemory 512mb
    maxmemory-policy allkeys-lru
    save 900 1
    save 300 10
    save 60 10000
```

### CDN Setup
For serving model artifacts and static assets:
```bash
# Configure CloudFront (AWS) or equivalent
aws cloudfront create-distribution \
  --distribution-config file://cdn-config.json
```

## 🚨 Troubleshooting

### Common Issues

1. **Pod Stuck in Pending**
   ```bash
   kubectl describe pod <pod-name> -n pneumonia-detection
   # Check resource constraints and node capacity
   ```

2. **Database Connection Issues**
   ```bash
   kubectl logs deployment/pneumonia-api -n pneumonia-detection
   kubectl exec -it postgres-0 -n pneumonia-detection -- psql -U pneumonia
   ```

3. **High Memory Usage**
   ```bash
   kubectl top pods -n pneumonia-detection
   kubectl describe hpa pneumonia-api-hpa -n pneumonia-detection
   ```

4. **Model Loading Failures**
   ```bash
   kubectl exec -it deployment/pneumonia-api -n pneumonia-detection -- ls -la /app/models
   kubectl logs deployment/pneumonia-api -n pneumonia-detection | grep -i "model"
   ```

### Performance Issues
```bash
# Check resource utilization
kubectl top nodes
kubectl top pods -n pneumonia-detection

# Analyze slow queries
kubectl exec -it postgres-0 -n pneumonia-detection -- psql -U pneumonia -c "SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;"

# Check API response times
curl -w "@curl-format.txt" -s http://your-api-endpoint/health
```

## 📝 Maintenance

### Regular Maintenance Tasks

1. **Weekly**
   - Review monitoring dashboards
   - Check backup integrity
   - Update security patches

2. **Monthly**
   - Rotate secrets and certificates
   - Review and optimize resource usage
   - Update dependencies

3. **Quarterly**
   - Disaster recovery testing
   - Security audit
   - Performance optimization review

### Updating the Application
```bash
# Build new image
docker build -f Dockerfile.production -t your-registry.com/pneumonia-detector:v1.1.0 .

# Deploy update
kubectl set image deployment/pneumonia-api pneumonia-api=your-registry.com/pneumonia-detector:v1.1.0 -n pneumonia-detection

# Monitor rollout
kubectl rollout status deployment/pneumonia-api -n pneumonia-detection
```

## 🆘 Emergency Procedures

### Rollback Deployment
```bash
# Quick rollback to previous version
kubectl rollout undo deployment/pneumonia-api -n pneumonia-detection

# Rollback to specific revision
kubectl rollout undo deployment/pneumonia-api --to-revision=2 -n pneumonia-detection
```

### Scale Down for Maintenance
```bash
# Scale down to zero replicas
kubectl scale deployment/pneumonia-api --replicas=0 -n pneumonia-detection

# Scale back up
kubectl scale deployment/pneumonia-api --replicas=3 -n pneumonia-detection
```

### Emergency Database Maintenance
```bash
# Create immediate backup
kubectl exec postgres-0 -n pneumonia-detection -- pg_dump -U pneumonia pneumonia_db > emergency_backup.sql

# Put application in maintenance mode
kubectl patch deployment/pneumonia-api -n pneumonia-detection -p '{"spec":{"replicas":0}}'
```

## 📞 Support and Contacts

- **Technical Issues**: Create issue at GitHub repository
- **Security Incidents**: security@terragon-labs.com
- **Emergency**: Use emergency escalation procedure

## 📚 Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [PostgreSQL Performance Tuning](https://wiki.postgresql.org/wiki/Performance_Optimization)
- [Redis Best Practices](https://redis.io/docs/manual/admin/)
- [Prometheus Monitoring](https://prometheus.io/docs/)
- [Security Best Practices](https://kubernetes.io/docs/concepts/security/)

---

**Last Updated**: December 2024  
**Version**: 1.0.0  
**Maintainer**: Terragon Labs Engineering Team