#!/bin/bash
# Production Deployment Script for Pneumonia Detection AI System
# Comprehensive deployment with security, monitoring, and validation

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEPLOYMENT_ENV=${DEPLOYMENT_ENV:-production}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-"your-registry.com"}
IMAGE_TAG=${IMAGE_TAG:-"latest"}
NAMESPACE=${NAMESPACE:-"pneumonia-detection"}
BACKUP_RETENTION_DAYS=${BACKUP_RETENTION_DAYS:-30}

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check required tools
    local tools=("docker" "kubectl" "helm" "jq" "curl")
    for tool in "${tools[@]}"; do
        if ! command -v $tool &> /dev/null; then
            error "$tool is required but not installed"
        fi
    done
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
    fi
    
    # Check Kubernetes connection
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
    fi
    
    # Check required environment variables
    local env_vars=("DB_PASSWORD" "REDIS_PASSWORD" "GRAFANA_PASSWORD")
    for var in "${env_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            error "Environment variable $var is required but not set"
        fi
    done
    
    log "Prerequisites check passed"
}

# Build and push Docker images
build_and_push_images() {
    log "Building and pushing Docker images..."
    
    # Build production image
    docker build -f Dockerfile.production -t ${DOCKER_REGISTRY}/pneumonia-detector:${IMAGE_TAG} .
    
    # Security scan
    info "Running security scan on Docker image..."
    if command -v trivy &> /dev/null; then
        trivy image --exit-code 1 --severity HIGH,CRITICAL ${DOCKER_REGISTRY}/pneumonia-detector:${IMAGE_TAG}
    else
        warn "Trivy not found, skipping security scan"
    fi
    
    # Push to registry
    docker push ${DOCKER_REGISTRY}/pneumonia-detector:${IMAGE_TAG}
    
    log "Docker images built and pushed successfully"
}

# Create Kubernetes secrets
create_secrets() {
    log "Creating Kubernetes secrets..."
    
    # Create namespace if it doesn't exist
    kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
    
    # Create secrets
    kubectl create secret generic pneumonia-secrets \
        --from-literal=DB_PASSWORD="${DB_PASSWORD}" \
        --from-literal=REDIS_PASSWORD="${REDIS_PASSWORD}" \
        --from-literal=JWT_SECRET="$(openssl rand -base64 32)" \
        --namespace=${NAMESPACE} \
        --dry-run=client -o yaml | kubectl apply -f -
    
    log "Kubernetes secrets created"
}

# Setup monitoring
setup_monitoring() {
    log "Setting up monitoring stack..."
    
    # Add Helm repositories
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo add jaegertracing https://jaegertracing.github.io/helm-charts
    helm repo update
    
    # Install Prometheus
    if ! helm list -n monitoring | grep -q prometheus; then
        helm install prometheus prometheus-community/kube-prometheus-stack \
            --namespace monitoring \
            --create-namespace \
            --set grafana.adminPassword="${GRAFANA_PASSWORD}" \
            --set prometheus.prometheusSpec.retention=30d \
            --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=50Gi
    else
        info "Prometheus already installed"
    fi
    
    # Install Jaeger
    if ! helm list -n monitoring | grep -q jaeger; then
        helm install jaeger jaegertracing/jaeger \
            --namespace monitoring \
            --set storage.type=badger \
            --set storage.badger.ephemeral=false
    else
        info "Jaeger already installed"
    fi
    
    log "Monitoring stack setup complete"
}

# Deploy application
deploy_application() {
    log "Deploying application..."
    
    # Update image in deployment
    sed -i.bak "s|pneumonia-detector:production|${DOCKER_REGISTRY}/pneumonia-detector:${IMAGE_TAG}|g" k8s/production-deployment.yaml
    
    # Apply Kubernetes manifests
    kubectl apply -f k8s/production-deployment.yaml
    
    # Wait for rollout to complete
    kubectl rollout status deployment/pneumonia-api -n ${NAMESPACE} --timeout=600s
    
    log "Application deployed successfully"
}

# Run health checks
health_checks() {
    log "Running health checks..."
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod -l app=pneumonia-api -n ${NAMESPACE} --timeout=300s
    
    # Get service endpoint
    local service_ip
    service_ip=$(kubectl get service pneumonia-api-service -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [[ -z "$service_ip" ]]; then
        service_ip=$(kubectl get service pneumonia-api-service -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    fi
    
    if [[ -z "$service_ip" ]]; then
        warn "Load balancer IP/hostname not available yet, using port-forward for health check"
        kubectl port-forward service/pneumonia-api-service 8080:80 -n ${NAMESPACE} &
        local port_forward_pid=$!
        sleep 5
        service_ip="localhost:8080"
    fi
    
    # Health check endpoints
    local endpoints=("/health" "/ready" "/metrics")
    for endpoint in "${endpoints[@]}"; do
        info "Checking endpoint: $endpoint"
        if curl -f --max-time 30 "http://${service_ip}${endpoint}" &> /dev/null; then
            log "✓ $endpoint is healthy"
        else
            error "✗ $endpoint health check failed"
        fi
    done
    
    # Kill port-forward if used
    if [[ -n "${port_forward_pid:-}" ]]; then
        kill $port_forward_pid 2>/dev/null || true
    fi
    
    log "Health checks passed"
}

# Setup backup
setup_backup() {
    log "Setting up backup system..."
    
    # Create backup CronJob
    cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-backup
  namespace: ${NAMESPACE}
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: postgres-backup
            image: postgres:15-alpine
            env:
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: pneumonia-secrets
                  key: DB_PASSWORD
            command:
            - /bin/bash
            - -c
            - |
              DATE=\$(date +%Y%m%d_%H%M%S)
              pg_dump -h postgres -U pneumonia pneumonia_db | gzip > /backup/backup_\${DATE}.sql.gz
              # Clean up old backups
              find /backup -name "backup_*.sql.gz" -mtime +${BACKUP_RETENTION_DAYS} -delete
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: backup-pvc
          restartPolicy: OnFailure
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: backup-pvc
  namespace: ${NAMESPACE}
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: gp2
  resources:
    requests:
      storage: 100Gi
EOF
    
    log "Backup system configured"
}

# Performance testing
performance_test() {
    log "Running performance tests..."
    
    # Get service endpoint
    local service_ip
    service_ip=$(kubectl get service pneumonia-api-service -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [[ -z "$service_ip" ]]; then
        service_ip=$(kubectl get service pneumonia-api-service -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    fi
    
    if [[ -z "$service_ip" ]]; then
        kubectl port-forward service/pneumonia-api-service 8080:80 -n ${NAMESPACE} &
        local port_forward_pid=$!
        sleep 5
        service_ip="localhost:8080"
    fi
    
    # Simple load test using curl
    info "Running basic load test..."
    for i in {1..10}; do
        response_time=$(curl -w "%{time_total}" -s -o /dev/null "http://${service_ip}/health")
        echo "Request $i: ${response_time}s"
    done
    
    # Kill port-forward if used
    if [[ -n "${port_forward_pid:-}" ]]; then
        kill $port_forward_pid 2>/dev/null || true
    fi
    
    log "Performance test completed"
}

# Generate deployment report
generate_report() {
    log "Generating deployment report..."
    
    local report_file="deployment-report-$(date +%Y%m%d_%H%M%S).md"
    
    cat > $report_file <<EOF
# Production Deployment Report

**Date:** $(date)
**Environment:** ${DEPLOYMENT_ENV}
**Image Tag:** ${IMAGE_TAG}
**Namespace:** ${NAMESPACE}

## Deployment Summary

### Application Status
\`\`\`
$(kubectl get pods -n ${NAMESPACE} -l app=pneumonia-api)
\`\`\`

### Service Status
\`\`\`
$(kubectl get services -n ${NAMESPACE})
\`\`\`

### Resource Usage
\`\`\`
$(kubectl top pods -n ${NAMESPACE} 2>/dev/null || echo "Metrics server not available")
\`\`\`

### Scaling Configuration
- Minimum Replicas: 3
- Maximum Replicas: 20
- CPU Target: 70%
- Memory Target: 80%

### Monitoring
- Prometheus: http://$(kubectl get service prometheus-kube-prometheus-prometheus -n monitoring -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):9090
- Grafana: http://$(kubectl get service prometheus-grafana -n monitoring -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):80
- Jaeger: http://$(kubectl get service jaeger-query -n monitoring -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):16686

### Backup
- Scheduled: Daily at 2 AM UTC
- Retention: ${BACKUP_RETENTION_DAYS} days
- Location: PVC backup-pvc

## Next Steps
1. Monitor application metrics for 24 hours
2. Run comprehensive load testing
3. Verify backup and restore procedures
4. Update documentation and runbooks

## Deployment Notes
$(git log --oneline -5)
EOF
    
    info "Deployment report generated: $report_file"
}

# Rollback function
rollback() {
    warn "Initiating rollback..."
    kubectl rollout undo deployment/pneumonia-api -n ${NAMESPACE}
    kubectl rollout status deployment/pneumonia-api -n ${NAMESPACE} --timeout=300s
    log "Rollback completed"
}

# Main deployment function
main() {
    log "Starting production deployment..."
    log "Environment: $DEPLOYMENT_ENV"
    log "Image tag: $IMAGE_TAG"
    log "Namespace: $NAMESPACE"
    
    # Trap errors and rollback if needed
    trap 'error "Deployment failed. Consider running rollback."; exit 1' ERR
    
    check_prerequisites
    build_and_push_images
    create_secrets
    setup_monitoring
    deploy_application
    health_checks
    setup_backup
    performance_test
    generate_report
    
    log "🎉 Production deployment completed successfully!"
    log "Application URL: http://$(kubectl get service pneumonia-api-service -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}')"
    log "Monitoring dashboards available in the monitoring namespace"
}

# Handle command line arguments
case "${1:-deploy}" in
    deploy)
        main
        ;;
    rollback)
        rollback
        ;;
    health-check)
        health_checks
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|health-check}"
        exit 1
        ;;
esac