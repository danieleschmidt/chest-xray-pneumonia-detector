# Incident Response Runbook

## Overview

This runbook provides step-by-step procedures for responding to incidents in the Chest X-Ray Pneumonia Detector system. Follow these procedures to quickly identify, escalate, and resolve issues.

## Incident Severity Levels

### 🔴 SEV-1 (Critical)
- **Impact**: Complete service outage, data loss, or security breach
- **Response Time**: Immediate (< 15 minutes)
- **Escalation**: Immediate to on-call engineer and management
- **Examples**: Service completely down, model returning all incorrect predictions, data breach

### 🟡 SEV-2 (High)
- **Impact**: Significant performance degradation or partial outage
- **Response Time**: < 1 hour
- **Escalation**: To on-call engineer within 30 minutes
- **Examples**: High error rate (>10%), severe latency issues (>5s), model accuracy drop

### 🟠 SEV-3 (Medium)
- **Impact**: Minor performance issues or non-critical functionality affected
- **Response Time**: < 4 hours (during business hours)
- **Escalation**: Standard team notification
- **Examples**: Elevated error rate (1-10%), minor latency increase, monitoring alerts

### 🟢 SEV-4 (Low)
- **Impact**: Cosmetic issues or minor improvements needed
- **Response Time**: Next business day or sprint planning
- **Escalation**: None required
- **Examples**: Documentation updates, minor UI issues, optimization opportunities

## Alert Response Procedures

### Service Down Alert

#### Immediate Actions (< 5 minutes)
1. **Acknowledge the alert** in your monitoring system
2. **Check service status** using health endpoints:
   ```bash
   curl -f http://pneumonia-detector:8000/health
   ```
3. **Verify container status**:
   ```bash
   docker ps | grep pneumonia-detector
   kubectl get pods -l app=pneumonia-detector  # if using Kubernetes
   ```

#### Investigation Steps (5-15 minutes)
1. **Check recent deployments**:
   ```bash
   git log --oneline -10  # Recent commits
   kubectl rollout history deployment/pneumonia-detector  # if K8s
   ```

2. **Review application logs**:
   ```bash
   docker logs pneumonia-detector-api --tail=100
   kubectl logs -l app=pneumonia-detector --tail=100
   ```

3. **Check resource usage**:
   ```bash
   docker stats pneumonia-detector-api
   # Or check Grafana dashboard for resource metrics
   ```

4. **Test dependencies**:
   ```bash
   # Check MLflow
   curl -f http://mlflow:5000/health
   
   # Check PostgreSQL
   docker exec postgres pg_isready
   ```

#### Resolution Steps
1. **If resource exhaustion**: Scale up resources or restart service
2. **If dependency failure**: Restart failed dependencies
3. **If recent deployment issue**: Rollback to previous version
4. **If configuration issue**: Fix configuration and redeploy

#### Communication
- **Internal**: Update team Slack/Teams channel with status
- **External**: Update status page if customer-facing
- **Timeline**: Provide updates every 15 minutes during SEV-1

### High Error Rate Alert

#### Immediate Actions
1. **Check error patterns** in logs:
   ```bash
   docker logs pneumonia-detector-api | grep -i error | tail -20
   ```

2. **Analyze error types**:
   ```bash
   # Check metrics endpoint for error breakdown
   curl http://pneumonia-detector:9090/metrics | grep error
   ```

3. **Verify model status**:
   ```bash
   # Test inference endpoint
   curl -X POST http://pneumonia-detector:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"test": true}'
   ```

#### Common Error Scenarios

**Model Loading Errors**
- Check model file integrity
- Verify model path configuration
- Check available memory and disk space
- Review model version compatibility

**Input Validation Errors**
- Check recent input data patterns
- Verify image format support
- Review preprocessing pipeline

**External Service Errors**
- Check MLflow connectivity
- Verify database connections
- Test third-party API availability

### High Latency Alert

#### Investigation Priority
1. **Resource bottlenecks** (CPU, Memory, I/O)
2. **Model performance** degradation
3. **Network connectivity** issues
4. **Database query** performance

#### Troubleshooting Steps
1. **Check system resources**:
   ```bash
   # CPU and memory usage
   top -p $(pgrep -f pneumonia-detector)
   
   # I/O statistics
   iotop -p $(pgrep -f pneumonia-detector)
   ```

2. **Analyze inference timing**:
   ```bash
   # Review metrics for timing breakdown
   curl http://pneumonia-detector:9090/metrics | grep inference_duration
   ```

3. **Check for concurrent requests**:
   ```bash
   # Monitor active connections
   ss -tulpn | grep :8000
   ```

### Memory/CPU Alerts

#### High Memory Usage
1. **Check memory breakdown**:
   ```bash
   # Process memory usage
   ps aux | grep pneumonia-detector
   
   # Container memory usage
   docker stats pneumonia-detector-api --no-stream
   ```

2. **Identify memory leaks**:
   ```bash
   # Monitor memory over time
   while true; do
     ps -o pid,vsz,rss,comm -p $(pgrep -f pneumonia-detector)
     sleep 30
   done
   ```

3. **Immediate mitigation**:
   - Restart service if memory leak suspected
   - Scale up resources if legitimate usage
   - Implement memory limits if unbounded growth

#### High CPU Usage
1. **Profile CPU usage**:
   ```bash
   # CPU breakdown by thread
   top -H -p $(pgrep -f pneumonia-detector)
   
   # System-wide CPU usage
   htop
   ```

2. **Check for CPU-intensive operations**:
   - Model inference spikes
   - Background processing
   - Garbage collection issues

## Escalation Procedures

### When to Escalate
- **SEV-1**: Immediately to on-call engineer and manager
- **SEV-2**: Within 30 minutes if no resolution progress
- **SEV-3**: After initial investigation if expert help needed
- **Complex Issues**: When multiple systems affected

### Escalation Contacts
```
Primary On-Call: [CONTACT_INFO]
Backup On-Call: [CONTACT_INFO]
Team Lead: [CONTACT_INFO]
Manager: [CONTACT_INFO]
```

### Escalation Communication Template
```
Subject: [SEV-X] Brief description of incident

INCIDENT SUMMARY:
- Start Time: [TIMESTAMP]
- Severity: [SEV-X]
- Impact: [Description of user/business impact]
- Status: [Investigating/Mitigating/Resolved]

SYMPTOMS:
- [What we're seeing]
- [Affected components]

INVESTIGATION:
- [What we've checked]
- [Current hypothesis]

NEXT STEPS:
- [Immediate actions planned]
- [ETA for next update]

RESPONSE TEAM:
- [People involved]
```

## Post-Incident Procedures

### Immediate Post-Resolution (< 1 hour)
1. **Verify full resolution**:
   - All metrics back to baseline
   - No related alerts firing
   - Customer impact resolved

2. **Document timeline**:
   - Incident start/end times
   - Detection method
   - Resolution steps taken

3. **Preserve evidence**:
   - Save relevant logs
   - Export monitoring data
   - Document configuration changes

### Post-Incident Review (within 48 hours)
1. **Schedule blameless post-mortem**
2. **Gather stakeholders**:
   - Response team members
   - Service owners
   - Customer success (if customer impact)

3. **Document lessons learned**:
   - Root cause analysis
   - Timeline of events
   - What went well/poorly
   - Action items for improvement

### Follow-up Actions
1. **Implement preventive measures**
2. **Update runbooks** based on learnings
3. **Add/improve monitoring** if gaps identified
4. **Schedule fixes** for underlying issues

## Tools and Resources

### Monitoring and Alerting
- **Grafana Dashboard**: http://grafana:3000/d/pneumonia-detector
- **Prometheus**: http://prometheus:9090
- **AlertManager**: http://alertmanager:9093

### Logs and Metrics
```bash
# Application logs
docker logs pneumonia-detector-api

# System metrics
curl http://pneumonia-detector:9090/metrics

# Health checks
curl http://pneumonia-detector:8000/health
```

### Common Commands
```bash
# Service status
docker ps | grep pneumonia-detector
docker-compose ps

# Restart services
docker-compose restart api
docker-compose restart mlflow

# Scale services
docker-compose up --scale api=3

# Check configurations
docker-compose config
```

### Emergency Contacts
- **Infrastructure Team**: [CONTACT]
- **Security Team**: [CONTACT]
- **Customer Success**: [CONTACT]
- **External Vendor Support**: [CONTACT]

## Testing and Validation

### Health Check Endpoints
```bash
# Application health
curl -f http://pneumonia-detector:8000/health

# Detailed health with dependencies
curl -f http://pneumonia-detector:8000/health/detailed

# Readiness check
curl -f http://pneumonia-detector:8000/ready
```

### Performance Testing
```bash
# Load test inference endpoint
ab -n 100 -c 10 -T application/json \
   -p test_payload.json \
   http://pneumonia-detector:8000/predict

# Monitor during test
watch -n 1 curl -s http://pneumonia-detector:9090/metrics | grep inference
```

### Rollback Procedures
```bash
# Docker Compose rollback
docker-compose down
git checkout previous-stable-commit
docker-compose up -d

# Kubernetes rollback
kubectl rollout undo deployment/pneumonia-detector
kubectl rollout status deployment/pneumonia-detector
```

---

**Last Updated**: 2025-07-28  
**Next Review**: 2025-08-28  
**Owner**: DevOps Team  
**Reviewers**: Engineering Team, SRE Team