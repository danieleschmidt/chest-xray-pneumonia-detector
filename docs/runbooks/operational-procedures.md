# Operational Procedures

## Overview

This document outlines standard operational procedures for maintaining, monitoring, and supporting the Chest X-Ray Pneumonia Detector system in production.

## Daily Operations

### Morning Health Check (Start of Business Day)

#### System Status Review
1. **Check overall system health**:
   ```bash
   # Verify all services are running
   docker-compose ps
   
   # Check service health endpoints
   curl -f http://pneumonia-detector:8000/health
   curl -f http://mlflow:5000
   ```

2. **Review overnight alerts**:
   - Check Grafana dashboard for any alerts
   - Review alert history in AlertManager
   - Identify any resolved incidents

3. **Performance baseline verification**:
   - Inference latency within acceptable range (<2s)
   - Error rate below threshold (<1%)
   - Resource utilization normal levels

4. **Review system metrics**:
   ```bash
   # Check key metrics
   curl -s http://pneumonia-detector:9090/metrics | grep -E "(inference_total|error_total|cpu_percent|memory_percent)"
   ```

#### Daily Checklist
- [ ] All services healthy and responding
- [ ] No critical alerts from previous 24 hours
- [ ] Performance metrics within normal ranges
- [ ] Disk space above 20% free
- [ ] Log rotation functioning properly
- [ ] Backup completion status verified

### Weekly Maintenance

#### System Maintenance (Every Monday)
1. **Update dependency security scan**:
   ```bash
   cd /path/to/repo
   ./scripts/security-scan.sh
   ```

2. **Review and rotate logs**:
   ```bash
   # Check log sizes
   du -sh logs/
   
   # Archive old logs if needed
   tar -czf logs/archive/logs-$(date +%Y%m%d).tar.gz logs/*.log
   rm logs/*.log.old
   ```

3. **Performance optimization review**:
   - Analyze inference time trends
   - Review resource utilization patterns
   - Identify optimization opportunities

#### Health Check Deep Dive (Every Wednesday)
1. **Comprehensive system analysis**:
   ```bash
   # Run performance benchmark
   python -m src.performance_benchmark
   
   # Generate comprehensive metrics report
   python -m src.monitoring.metrics --serve --port 9090 &
   curl -s http://localhost:9090/metrics > weekly-metrics-$(date +%Y%m%d).txt
   ```

2. **Model performance review**:
   - Check inference accuracy trends
   - Review model drift indicators
   - Validate model performance against benchmarks

3. **Capacity planning**:
   - Analyze usage trends
   - Forecast resource needs
   - Plan for scaling requirements

#### Security Review (Every Friday)
1. **Run security scans**:
   ```bash
   # Comprehensive security scan
   ./scripts/security-scan.sh
   
   # Generate SBOM for compliance
   ./scripts/generate-sbom.sh
   ```

2. **Review access logs**:
   - Check for unusual access patterns
   - Verify authentication logs
   - Review API usage statistics

3. **Update security configurations**:
   - Rotate API keys if applicable
   - Review firewall rules
   - Update security policies

## Monthly Operations

### First Monday of Month - Infrastructure Review
1. **Capacity planning assessment**
2. **Security patch review and application**
3. **Backup and disaster recovery testing**
4. **Documentation updates**

### Third Monday of Month - Performance Optimization
1. **Model performance analysis**
2. **Infrastructure optimization review**
3. **Cost optimization assessment**
4. **Monitoring and alerting refinement**

## Deployment Procedures

### Standard Deployment Process

#### Pre-Deployment Checklist
- [ ] Code changes reviewed and approved
- [ ] Unit tests passing (>85% coverage)
- [ ] Integration tests passing
- [ ] Security scans clean
- [ ] Performance benchmarks acceptable
- [ ] Rollback plan documented

#### Deployment Steps
1. **Prepare deployment**:
   ```bash
   # Build and tag new version
   git tag v$(date +%Y%m%d)-$(git rev-parse --short HEAD)
   ./scripts/build-security.sh production
   ```

2. **Deploy to staging**:
   ```bash
   # Deploy to staging environment
   docker-compose -f docker-compose.staging.yml up -d
   
   # Run smoke tests
   ./scripts/test-deployment.sh staging
   ```

3. **Validate staging deployment**:
   ```bash
   # Health check
   curl -f http://staging-pneumonia-detector:8000/health
   
   # Performance test
   ab -n 100 -c 5 http://staging-pneumonia-detector:8000/predict
   ```

4. **Deploy to production**:
   ```bash
   # Rolling deployment
   docker-compose up -d --no-deps api
   
   # Wait for health check
   while ! curl -f http://pneumonia-detector:8000/health; do
     echo "Waiting for service to be healthy..."
     sleep 10
   done
   ```

5. **Post-deployment validation**:
   ```bash
   # Verify deployment
   curl -f http://pneumonia-detector:8000/health
   
   # Check metrics
   curl -s http://pneumonia-detector:9090/metrics | grep uptime
   
   # Monitor for 15 minutes
   watch -n 30 'curl -s http://pneumonia-detector:9090/metrics | grep -E "(error_total|inference_total)"'
   ```

#### Rollback Procedure
```bash
# Immediate rollback if issues detected
docker-compose down
git checkout previous-stable-tag
docker-compose up -d

# Verify rollback successful
curl -f http://pneumonia-detector:8000/health
```

### Emergency Deployment Process

#### Hotfix Deployment
1. **Create hotfix branch** from stable release
2. **Implement minimal fix** with focused testing
3. **Fast-track review** with senior engineer approval
4. **Deploy with enhanced monitoring**:
   ```bash
   # Enhanced monitoring during hotfix deployment
   ./scripts/deploy-with-monitoring.sh --hotfix --monitor-duration=30
   ```

## Monitoring and Alerting

### Key Metrics to Monitor

#### Business Metrics
- **Inference Success Rate**: >99%
- **Average Inference Latency**: <2 seconds
- **Daily Inference Volume**: Track trends
- **Model Accuracy**: Baseline performance maintenance

#### Technical Metrics  
- **Service Uptime**: >99.9%
- **Error Rate**: <1%
- **Resource Utilization**: CPU <80%, Memory <85%
- **Disk Space**: >20% free

#### Infrastructure Metrics
- **Container Health**: All containers running
- **Database Connectivity**: MLflow backend accessible
- **External Dependencies**: All external services responding

### Alert Configuration Review

#### Monthly Alert Tuning
1. **Review alert frequency** and adjust thresholds
2. **Analyze false positive** patterns
3. **Update alert descriptions** and runbooks
4. **Test alert delivery** mechanisms

#### Alert Response Time Targets
- **Critical (SEV-1)**: 15 minutes
- **High (SEV-2)**: 1 hour
- **Medium (SEV-3)**: 4 hours
- **Low (SEV-4)**: Next business day

## Backup and Recovery

### Data Backup Procedures

#### Daily Backups
1. **Model artifacts**:
   ```bash
   # Backup model files
   tar -czf backups/models-$(date +%Y%m%d).tar.gz saved_models/
   
   # Upload to cloud storage
   aws s3 cp backups/models-$(date +%Y%m%d).tar.gz s3://backup-bucket/models/
   ```

2. **Configuration backup**:
   ```bash
   # Backup configurations
   tar -czf backups/config-$(date +%Y%m%d).tar.gz \
     docker-compose.yml \
     monitoring/ \
     .env.example
   ```

3. **Database backup** (if applicable):
   ```bash
   # Backup MLflow database
   docker exec postgres pg_dump -U mlflow mlflow > backups/mlflow-$(date +%Y%m%d).sql
   ```

#### Weekly Full Backup
```bash
# Complete system backup
./scripts/full-backup.sh --destination /backup/weekly/
```

### Disaster Recovery Testing

#### Quarterly DR Test
1. **Simulate complete failure**
2. **Restore from backups**
3. **Verify functionality**
4. **Document recovery time**
5. **Update DR procedures**

#### Recovery Time Objectives (RTO)
- **Model Service**: 30 minutes
- **Complete System**: 2 hours
- **Data Recovery**: 4 hours

#### Recovery Point Objectives (RPO)
- **Configuration**: 24 hours
- **Model Artifacts**: 24 hours
- **Metrics Data**: 1 hour

## Performance Optimization

### Regular Performance Reviews

#### Weekly Performance Analysis
```bash
# Generate performance report
python -m src.performance_benchmark --report --output weekly-perf-$(date +%Y%m%d).json

# Analyze trends
python -m scripts.analyze_performance_trends --weeks 4
```

#### Optimization Opportunities
1. **Model Optimization**:
   - Model quantization for faster inference
   - Batch processing optimization
   - Caching frequently used models

2. **Infrastructure Optimization**:
   - Container resource tuning
   - Load balancer configuration
   - Database query optimization

3. **Application Optimization**:
   - Code profiling and optimization
   - Memory usage optimization
   - I/O operation efficiency

### Capacity Planning

#### Growth Planning Process
1. **Analyze usage trends** (monthly)
2. **Forecast capacity needs** (quarterly)
3. **Plan infrastructure scaling** (bi-annually)
4. **Budget planning** for resources (annually)

#### Scaling Triggers
- **CPU Usage**: >70% average over 1 week
- **Memory Usage**: >75% average over 1 week
- **Response Time**: >1.5s average over 1 day
- **Error Rate**: >0.5% over 1 day

## Security Operations

### Regular Security Tasks

#### Daily Security Monitoring
- Review access logs for anomalies
- Check security alert feeds
- Verify SSL certificate status
- Monitor for unauthorized access attempts

#### Weekly Security Review
```bash
# Run comprehensive security scan
FAIL_ON_MEDIUM=true ./scripts/security-scan.sh

# Review security metrics
curl -s http://pneumonia-detector:9090/metrics | grep security
```

#### Monthly Security Audit
1. **Dependency vulnerability scan**
2. **Infrastructure security assessment**
3. **Access control review**
4. **Security configuration validation**

### Incident Response Integration

#### Security Incident Escalation
1. **Immediate isolation** of affected systems
2. **Evidence preservation** procedures
3. **Stakeholder notification** protocol
4. **Forensic analysis** coordination

## Documentation Maintenance

### Documentation Review Schedule

#### Monthly Reviews
- Update operational procedures
- Review and update runbooks
- Validate troubleshooting guides
- Update emergency contact information

#### Quarterly Reviews
- Comprehensive documentation audit
- User feedback integration
- Process improvement documentation
- Training material updates

### Documentation Standards
- **Clarity**: Step-by-step procedures
- **Accuracy**: Regular testing and validation
- **Completeness**: Cover all operational scenarios
- **Accessibility**: Easy to find and understand

## Training and Knowledge Management

### Team Training Requirements

#### New Team Member Onboarding
1. **System architecture overview**
2. **Operational procedures training**
3. **Emergency response procedures**
4. **Tool and access setup**

#### Ongoing Training
- **Monthly tech talks** on system updates
- **Quarterly incident response drills**
- **Annual security training**
- **Continuous learning** on ML operations

### Knowledge Sharing
- **Weekly team standups** with operational updates
- **Monthly retrospectives** on operational efficiency
- **Quarterly all-hands** presentations on system health
- **Annual operational review** and planning

---

**Document Version**: 1.0  
**Last Updated**: 2025-07-28  
**Next Review**: 2025-08-28  
**Owner**: SRE Team  
**Reviewers**: Engineering Team, Security Team