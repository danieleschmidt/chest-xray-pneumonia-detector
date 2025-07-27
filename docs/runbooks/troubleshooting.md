# Troubleshooting Runbook

This runbook provides step-by-step procedures for diagnosing and resolving common issues with the Chest X-Ray Pneumonia Detector.

## Table of Contents

- [Quick Diagnostics](#quick-diagnostics)
- [Service Issues](#service-issues)
- [Performance Issues](#performance-issues)
- [Data Issues](#data-issues)
- [Model Issues](#model-issues)
- [Infrastructure Issues](#infrastructure-issues)
- [Security Issues](#security-issues)
- [Monitoring Issues](#monitoring-issues)
- [Emergency Procedures](#emergency-procedures)

## Quick Diagnostics

### Health Check Script

```bash
#!/bin/bash
# Quick health check script

echo "=== Pneumonia Detector Health Check ==="
echo "Timestamp: $(date)"
echo

# System resources
echo "📊 System Resources:"
echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "Memory: $(free | grep Mem | awk '{printf("%.1f%%"), $3/$2 * 100.0}')"
echo "Disk: $(df / | tail -1 | awk '{print $5}')"
echo

# Service status
echo "🔍 Service Status:"
if docker-compose ps > /dev/null 2>&1; then
    docker-compose ps
else
    echo "Docker Compose not available"
fi
echo

# Application health
echo "🏥 Application Health:"
if curl -s -f http://localhost:8000/health > /dev/null; then
    echo "✅ API is responding"
    curl -s http://localhost:8000/health | jq '.status' 2>/dev/null || echo "Health check returned non-JSON response"
else
    echo "❌ API is not responding"
fi
echo

# Database connectivity
echo "💾 Database Status:"
if docker-compose exec -T db pg_isready > /dev/null 2>&1; then
    echo "✅ Database is ready"
else
    echo "❌ Database is not ready"
fi
echo

# Model availability
echo "🤖 Model Status:"
if [ -f "saved_models/pneumonia_cnn_v1.keras" ]; then
    echo "✅ Model file exists"
    echo "Size: $(ls -lh saved_models/pneumonia_cnn_v1.keras | awk '{print $5}')"
else
    echo "❌ Model file not found"
fi
echo

echo "=== Health Check Complete ==="
```

### Log Analysis

```bash
# Check recent errors
docker-compose logs --tail=50 | grep -i error

# Check service startup
docker-compose logs api --tail=100 | grep -E "(started|error|exception)"

# Check resource usage
docker stats --no-stream
```

## Service Issues

### Issue: API Service Won't Start

**Symptoms:**
- Container exits immediately
- Health checks fail
- Connection refused errors

**Diagnosis Steps:**

1. **Check container logs:**
   ```bash
   docker-compose logs api
   ```

2. **Check container status:**
   ```bash
   docker-compose ps
   ```

3. **Verify configuration:**
   ```bash
   docker-compose config
   ```

4. **Check resource availability:**
   ```bash
   docker system df
   free -h
   df -h
   ```

**Resolution Steps:**

1. **Fix common issues:**
   ```bash
   # Port conflicts
   sudo lsof -i :8000
   
   # Permission issues
   sudo chown -R 1000:1000 saved_models/
   sudo chmod -R 755 saved_models/
   
   # Clean restart
   docker-compose down
   docker-compose pull
   docker-compose up -d
   ```

2. **Environment variables:**
   ```bash
   # Check required variables
   grep -E "(DATABASE_URL|MODEL_PATH)" .env
   
   # Validate database URL
   docker-compose exec api python -c "
   import os
   print('DATABASE_URL:', os.getenv('DATABASE_URL', 'Not set'))
   "
   ```

3. **Model file verification:**
   ```bash
   # Check model file
   ls -la saved_models/
   
   # Test model loading
   docker-compose exec api python -c "
   try:
       import tensorflow as tf
       model = tf.keras.models.load_model('/app/saved_models/pneumonia_cnn_v1.keras')
       print('✅ Model loads successfully')
   except Exception as e:
       print(f'❌ Model loading failed: {e}')
   "
   ```

### Issue: Database Connection Failures

**Symptoms:**
- Connection timeout errors
- Authentication failures
- Database not ready

**Diagnosis Steps:**

1. **Check database status:**
   ```bash
   docker-compose exec db pg_isready
   docker-compose logs db --tail=50
   ```

2. **Test connectivity:**
   ```bash
   docker-compose exec api python -c "
   import psycopg2
   import os
   try:
       conn = psycopg2.connect(os.getenv('DATABASE_URL'))
       print('✅ Database connection successful')
       conn.close()
   except Exception as e:
       print(f'❌ Database connection failed: {e}')
   "
   ```

**Resolution Steps:**

1. **Restart database:**
   ```bash
   docker-compose restart db
   sleep 10
   docker-compose exec db pg_isready
   ```

2. **Check database credentials:**
   ```bash
   # Verify credentials in environment
   grep POSTGRES .env
   
   # Reset database if needed
   docker-compose down
   docker volume rm $(docker volume ls -q | grep postgres)
   docker-compose up -d db
   ```

3. **Network connectivity:**
   ```bash
   # Test network connectivity
   docker-compose exec api ping db
   
   # Check network configuration
   docker network ls
   docker network inspect pneumonia-detector_default
   ```

### Issue: High Response Times

**Symptoms:**
- Slow API responses
- Timeout errors
- High latency

**Diagnosis Steps:**

1. **Check system resources:**
   ```bash
   top -bn1 | head -20
   iostat 1 5
   docker stats
   ```

2. **Analyze response times:**
   ```bash
   # Test API response time
   time curl -s http://localhost:8000/health
   
   # Check application metrics
   curl -s http://localhost:8000/metrics | grep duration
   ```

3. **Database performance:**
   ```bash
   docker-compose exec db psql -U pneumonia -d pneumonia_prod -c "
   SELECT query, mean_time, calls 
   FROM pg_stat_statements 
   ORDER BY mean_time DESC 
   LIMIT 10;
   "
   ```

**Resolution Steps:**

1. **Scale resources:**
   ```bash
   # Add more API instances
   docker-compose up -d --scale api=3
   
   # Increase memory limits
   # Edit docker-compose.yml:
   # deploy:
   #   resources:
   #     limits:
   #       memory: 4G
   ```

2. **Optimize database:**
   ```bash
   # Analyze slow queries
   docker-compose exec db psql -U pneumonia -d pneumonia_prod -c "
   SELECT query, total_time, mean_time, calls
   FROM pg_stat_statements
   WHERE mean_time > 100
   ORDER BY total_time DESC;
   "
   
   # Add indexes if needed
   docker-compose exec db psql -U pneumonia -d pneumonia_prod -c "
   CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_table_column ON table_name(column_name);
   "
   ```

3. **Application optimization:**
   ```bash
   # Enable caching (if implemented)
   docker-compose exec api python -c "
   import os
   os.environ['CACHE_ENABLED'] = 'true'
   "
   
   # Adjust batch sizes
   docker-compose exec api python -c "
   import os
   os.environ['BATCH_SIZE'] = '16'  # Reduce if memory constrained
   "
   ```

## Performance Issues

### Issue: High Memory Usage

**Symptoms:**
- Out of memory errors
- Container kills
- Swap usage

**Diagnosis Steps:**

1. **Memory analysis:**
   ```bash
   # System memory
   free -h
   cat /proc/meminfo | grep -E "(MemTotal|MemFree|MemAvailable)"
   
   # Container memory
   docker stats --no-stream
   
   # Process memory
   docker-compose exec api ps aux --sort=-%mem | head -10
   ```

2. **Application memory profiling:**
   ```bash
   # Memory usage by component
   docker-compose exec api python -c "
   import psutil
   import os
   process = psutil.Process(os.getpid())
   memory_info = process.memory_info()
   print(f'RSS: {memory_info.rss / 1024 / 1024:.1f} MB')
   print(f'VMS: {memory_info.vms / 1024 / 1024:.1f} MB')
   "
   ```

**Resolution Steps:**

1. **Immediate relief:**
   ```bash
   # Restart high-memory containers
   docker-compose restart api
   
   # Clear system cache
   sudo sync && sudo sysctl vm.drop_caches=3
   
   # Reduce batch sizes
   export BATCH_SIZE=8
   docker-compose restart api
   ```

2. **Long-term optimization:**
   ```bash
   # Set memory limits
   # In docker-compose.yml:
   deploy:
     resources:
       limits:
         memory: 2G
       reservations:
         memory: 1G
   
   # Enable memory monitoring
   docker-compose exec api python -c "
   from src.monitoring.metrics import get_metrics_collector
   collector = get_metrics_collector()
   collector.set_gauge('memory_limit_mb', 2048)
   "
   ```

### Issue: High CPU Usage

**Symptoms:**
- Slow response times
- High load average
- CPU throttling

**Diagnosis Steps:**

1. **CPU analysis:**
   ```bash
   # System CPU
   top -bn1 | grep "Cpu(s)"
   uptime
   
   # Container CPU
   docker stats --no-stream | grep -E "(CONTAINER|api)"
   
   # Process analysis
   docker-compose exec api top -bn1 | head -20
   ```

**Resolution Steps:**

1. **Optimize processing:**
   ```bash
   # Reduce inference batch size
   export BATCH_SIZE=4
   
   # Limit concurrent requests
   export API_WORKERS=2
   
   # Use CPU-optimized model (if available)
   export MODEL_PATH=/app/models/pneumonia_cnn_cpu_optimized.keras
   ```

2. **Scale horizontally:**
   ```bash
   # Add more API instances
   docker-compose up -d --scale api=2
   
   # Use load balancer
   # Configure nginx upstream with multiple backends
   ```

## Data Issues

### Issue: Image Processing Errors

**Symptoms:**
- Failed image loading
- Preprocessing errors
- Invalid image formats

**Diagnosis Steps:**

1. **Test image processing:**
   ```bash
   docker-compose exec api python -c "
   from PIL import Image
   import numpy as np
   
   try:
       # Test image loading
       img = Image.open('/path/to/test/image.jpg')
       img_array = np.array(img)
       print(f'✅ Image loaded: {img_array.shape}')
   except Exception as e:
       print(f'❌ Image loading failed: {e}')
   "
   ```

2. **Check file permissions:**
   ```bash
   ls -la data/
   docker-compose exec api ls -la /app/data/
   ```

**Resolution Steps:**

1. **Fix permissions:**
   ```bash
   sudo chown -R 1000:1000 data/
   sudo chmod -R 755 data/
   ```

2. **Validate image formats:**
   ```bash
   # Check supported formats
   docker-compose exec api python -c "
   from PIL import Image
   print('Supported formats:', Image.EXTENSION.keys())
   "
   
   # Convert unsupported formats
   find data/ -name "*.bmp" -exec convert {} {}.jpg \;
   ```

### Issue: Data Validation Failures

**Symptoms:**
- Input validation errors
- Schema validation failures
- Data quality issues

**Diagnosis Steps:**

1. **Run data validation:**
   ```bash
   docker-compose exec api python -m src.input_validation \
     --data_dir /app/data/test \
     --output_file validation_report.json
   ```

2. **Check data statistics:**
   ```bash
   docker-compose exec api python -m src.dataset_stats \
     --data_dir /app/data \
     --output_json stats.json
   ```

**Resolution Steps:**

1. **Clean invalid data:**
   ```bash
   # Remove corrupted files
   docker-compose exec api python -c "
   import os
   from PIL import Image
   
   data_dir = '/app/data'
   for root, dirs, files in os.walk(data_dir):
       for file in files:
           if file.lower().endswith(('.jpg', '.jpeg', '.png')):
               filepath = os.path.join(root, file)
               try:
                   with Image.open(filepath) as img:
                       img.verify()
               except Exception as e:
                   print(f'Removing corrupted file: {filepath}')
                   os.remove(filepath)
   "
   ```

## Model Issues

### Issue: Model Loading Failures

**Symptoms:**
- Model file not found
- Incompatible model format
- Corruption errors

**Diagnosis Steps:**

1. **Check model file:**
   ```bash
   ls -la saved_models/
   file saved_models/pneumonia_cnn_v1.keras
   ```

2. **Test model loading:**
   ```bash
   docker-compose exec api python -c "
   import tensorflow as tf
   try:
       model = tf.keras.models.load_model('/app/saved_models/pneumonia_cnn_v1.keras')
       print(f'✅ Model loaded: {model.summary()}')
   except Exception as e:
       print(f'❌ Model loading failed: {e}')
   "
   ```

**Resolution Steps:**

1. **Download fresh model:**
   ```bash
   # Backup current model
   mv saved_models/pneumonia_cnn_v1.keras saved_models/pneumonia_cnn_v1.keras.backup
   
   # Download from MLflow or model registry
   docker-compose exec api python -c "
   import mlflow
   model = mlflow.keras.load_model('models:/pneumonia-detector/production')
   model.save('/app/saved_models/pneumonia_cnn_v1.keras')
   "
   ```

2. **Verify model integrity:**
   ```bash
   # Calculate checksum
   md5sum saved_models/pneumonia_cnn_v1.keras
   
   # Compare with known good checksum
   echo "expected_checksum saved_models/pneumonia_cnn_v1.keras" | md5sum -c
   ```

### Issue: Poor Model Performance

**Symptoms:**
- Low accuracy
- High inference times
- Inconsistent predictions

**Diagnosis Steps:**

1. **Run model evaluation:**
   ```bash
   docker-compose exec api python -m src.evaluate \
     --model_path /app/saved_models/pneumonia_cnn_v1.keras \
     --data_dir /app/data/test \
     --output_json evaluation_results.json
   ```

2. **Check model metrics:**
   ```bash
   # Performance benchmarking
   docker-compose exec api python -m src.performance_benchmark \
     --model_path /app/saved_models/pneumonia_cnn_v1.keras
   ```

**Resolution Steps:**

1. **Model retraining:**
   ```bash
   # Trigger retraining pipeline
   docker-compose exec api python -m src.train_engine \
     --train_dir /app/data/train \
     --val_dir /app/data/val \
     --epochs 50 \
     --save_model_path /app/saved_models/pneumonia_cnn_retrained.keras
   ```

2. **Model optimization:**
   ```bash
   # Convert to optimized format
   docker-compose exec api python -c "
   import tensorflow as tf
   
   # Load and optimize model
   model = tf.keras.models.load_model('/app/saved_models/pneumonia_cnn_v1.keras')
   
   # Convert to TensorFlow Lite for mobile/edge deployment
   converter = tf.lite.TFLiteConverter.from_keras_model(model)
   converter.optimizations = [tf.lite.Optimize.DEFAULT]
   tflite_model = converter.convert()
   
   with open('/app/saved_models/pneumonia_cnn_v1.tflite', 'wb') as f:
       f.write(tflite_model)
   "
   ```

## Infrastructure Issues

### Issue: Disk Space Problems

**Symptoms:**
- No space left on device
- Container startup failures
- Log rotation issues

**Diagnosis Steps:**

1. **Check disk usage:**
   ```bash
   df -h
   du -sh /* | sort -hr | head -10
   docker system df
   ```

2. **Find large files:**
   ```bash
   find / -type f -size +100M 2>/dev/null | head -20
   find /var/lib/docker -type f -size +100M 2>/dev/null
   ```

**Resolution Steps:**

1. **Clean up Docker:**
   ```bash
   # Remove unused containers, networks, images
   docker system prune -af
   
   # Clean up volumes (be careful!)
   docker volume prune -f
   
   # Remove old images
   docker image prune -af --filter "until=24h"
   ```

2. **Clean up logs:**
   ```bash
   # Truncate large log files
   sudo truncate -s 0 /var/log/syslog
   sudo truncate -s 0 /var/log/auth.log
   
   # Clean up application logs
   find logs/ -name "*.log" -mtime +7 -delete
   
   # Configure log rotation
   sudo logrotate -f /etc/logrotate.conf
   ```

3. **Clean up data:**
   ```bash
   # Remove old model versions
   find saved_models/ -name "*.keras" -mtime +30 -delete
   
   # Clean up temporary files
   find /tmp -type f -mtime +7 -delete
   
   # Clean up MLflow artifacts
   find mlruns/ -type f -mtime +30 -delete
   ```

### Issue: Network Connectivity Problems

**Symptoms:**
- Service unreachable
- DNS resolution failures
- Timeout errors

**Diagnosis Steps:**

1. **Test connectivity:**
   ```bash
   # Test external connectivity
   docker-compose exec api ping -c 3 8.8.8.8
   
   # Test DNS resolution
   docker-compose exec api nslookup google.com
   
   # Test internal connectivity
   docker-compose exec api ping db
   ```

2. **Check network configuration:**
   ```bash
   docker network ls
   docker network inspect pneumonia-detector_default
   
   # Check port bindings
   docker-compose ps
   netstat -tlnp | grep :8000
   ```

**Resolution Steps:**

1. **Restart networking:**
   ```bash
   # Restart Docker networking
   sudo systemctl restart docker
   
   # Recreate networks
   docker-compose down
   docker network prune -f
   docker-compose up -d
   ```

2. **Fix DNS issues:**
   ```bash
   # Configure DNS in docker-compose.yml
   dns:
     - 8.8.8.8
     - 8.8.4.4
   
   # Or set system DNS
   echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf
   ```

## Emergency Procedures

### Service Outage Response

1. **Immediate Assessment:**
   ```bash
   # Quick health check
   curl -f http://localhost:8000/health || echo "Service is down"
   
   # Check all services
   docker-compose ps
   
   # Check system resources
   uptime && free -h && df -h
   ```

2. **Emergency Restart:**
   ```bash
   # Quick restart
   docker-compose restart
   
   # Full restart if needed
   docker-compose down && docker-compose up -d
   
   # Verify recovery
   sleep 30 && curl -f http://localhost:8000/health
   ```

3. **Incident Documentation:**
   ```bash
   # Collect logs
   mkdir -p incident-$(date +%Y%m%d_%H%M%S)
   docker-compose logs > incident-$(date +%Y%m%d_%H%M%S)/docker-logs.txt
   dmesg > incident-$(date +%Y%m%d_%H%M%S)/system-logs.txt
   docker system df > incident-$(date +%Y%m%d_%H%M%S)/docker-usage.txt
   ```

### Data Recovery

1. **Backup Recovery:**
   ```bash
   # Stop services
   docker-compose down
   
   # Restore database
   docker-compose up -d db
   sleep 10
   cat /backups/db_backup_latest.sql | docker-compose exec -T db psql -U pneumonia -d pneumonia_prod
   
   # Restore models
   tar -xzf /backups/models_backup_latest.tar.gz -C saved_models/
   
   # Restart services
   docker-compose up -d
   ```

### Security Incident Response

1. **Immediate Actions:**
   ```bash
   # Isolate affected services
   docker-compose stop api
   
   # Check for suspicious activity
   grep -i "error\|failed\|unauthorized" logs/application.log | tail -50
   
   # Change access credentials
   # Update database passwords
   # Rotate API keys
   # Update SSL certificates
   ```

2. **Investigation:**
   ```bash
   # Collect security logs
   mkdir -p security-incident-$(date +%Y%m%d_%H%M%S)
   
   # Application logs
   cp logs/application.log security-incident-*/
   
   # System logs
   sudo journalctl --since "1 hour ago" > security-incident-*/system-logs.txt
   
   # Network connections
   netstat -tulpn > security-incident-*/network-connections.txt
   ```

### Contact Information

**Emergency Contacts:**
- On-call Engineer: [Phone/Email]
- Security Team: [Email]
- Infrastructure Team: [Email]

**Escalation Procedures:**
1. Level 1: On-call engineer (< 15 minutes)
2. Level 2: Team lead (< 30 minutes)
3. Level 3: Management (< 1 hour)

For additional support or to report issues not covered in this runbook, create an incident ticket or contact the on-call engineer immediately.