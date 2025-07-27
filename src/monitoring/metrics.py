"""
Prometheus metrics for the Chest X-Ray Pneumonia Detector API.
"""

import time
from typing import Dict, Any
import psutil
from prometheus_client import (
    Counter, Histogram, Gauge, Info, CollectorRegistry, 
    generate_latest, CONTENT_TYPE_LATEST
)


class MetricsRegistry:
    """Registry for all application metrics."""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        self._start_time = time.time()
        self._initialized = False
        
    def init_metrics(self):
        """Initialize all metrics."""
        if self._initialized:
            return
            
        # Request metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total number of HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.http_request_duration_seconds = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0],
            registry=self.registry
        )
        
        # Prediction metrics
        self.prediction_counter = Counter(
            'predictions_total',
            'Total number of predictions made',
            registry=self.registry
        )
        
        self.batch_prediction_counter = Counter(
            'batch_predictions_total',
            'Total number of batch predictions made',
            registry=self.registry
        )
        
        self.prediction_duration = Histogram(
            'prediction_duration_seconds',
            'Time spent on predictions in seconds',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )
        
        self.batch_size_histogram = Histogram(
            'batch_size',
            'Distribution of batch sizes',
            buckets=[1, 2, 4, 8, 16, 32],
            registry=self.registry
        )
        
        # Model metrics
        self.model_load_duration = Histogram(
            'model_load_duration_seconds',
            'Time spent loading models in seconds',
            registry=self.registry
        )
        
        self.model_info = Info(
            'model_info',
            'Information about the current model',
            registry=self.registry
        )
        
        # Error metrics
        self.error_counter = Counter(
            'errors_total',
            'Total number of errors',
            ['error_type'],
            registry=self.registry
        )
        
        # System metrics
        self.memory_usage_bytes = Gauge(
            'memory_usage_bytes',
            'Current memory usage in bytes',
            registry=self.registry
        )
        
        self.cpu_usage_percent = Gauge(
            'cpu_usage_percent',
            'Current CPU usage percentage',
            registry=self.registry
        )
        
        self.disk_usage_bytes = Gauge(
            'disk_usage_bytes',
            'Current disk usage in bytes',
            ['mount_point'],
            registry=self.registry
        )
        
        self.uptime_seconds = Gauge(
            'uptime_seconds',
            'Application uptime in seconds',
            registry=self.registry
        )
        
        # Health metrics
        self.health_check_duration = Histogram(
            'health_check_duration_seconds',
            'Time spent on health checks in seconds',
            ['check_name'],
            registry=self.registry
        )
        
        self.health_check_status = Gauge(
            'health_check_status',
            'Health check status (1=healthy, 0=unhealthy)',
            ['check_name'],
            registry=self.registry
        )
        
        # Business metrics
        self.prediction_confidence = Histogram(
            'prediction_confidence',
            'Distribution of prediction confidence scores',
            buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
            registry=self.registry
        )
        
        self.class_predictions = Counter(
            'class_predictions_total',
            'Total predictions by class',
            ['class_name'],
            registry=self.registry
        )
        
        # Set application info
        self.app_info = Info(
            'app_info',
            'Application information',
            registry=self.registry
        )
        
        self.app_info.info({
            'version': '0.2.0',
            'name': 'chest_xray_pneumonia_detector',
            'description': 'AI-powered pneumonia detection from chest X-ray images'
        })
        
        self._initialized = True
        
    def update_system_metrics(self):
        """Update system metrics."""
        if not self._initialized:
            return
            
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage_bytes.set(memory.used)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            self.cpu_usage_percent.set(cpu_percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.disk_usage_bytes.labels(mount_point='/').set(disk.used)
            
            # Uptime
            uptime = time.time() - self._start_time
            self.uptime_seconds.set(uptime)
            
        except Exception:
            # Silently ignore system metrics errors
            pass
    
    def record_prediction(self, confidence: float, class_name: str, duration: float):
        """Record a prediction with associated metrics."""
        if not self._initialized:
            return
            
        self.prediction_counter.inc()
        self.prediction_duration.observe(duration)
        self.prediction_confidence.observe(confidence)
        self.class_predictions.labels(class_name=class_name).inc()
        
    def record_batch_prediction(self, batch_size: int, duration: float):
        """Record a batch prediction."""
        if not self._initialized:
            return
            
        self.batch_prediction_counter.inc()
        self.batch_size_histogram.observe(batch_size)
        self.prediction_duration.observe(duration)
        
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record an HTTP request."""
        if not self._initialized:
            return
            
        self.http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).inc()
        
        self.http_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
    def record_health_check(self, check_name: str, is_healthy: bool, duration: float):
        """Record a health check result."""
        if not self._initialized:
            return
            
        self.health_check_duration.labels(check_name=check_name).observe(duration)
        self.health_check_status.labels(check_name=check_name).set(1 if is_healthy else 0)
        
    def set_model_info(self, model_info: Dict[str, Any]):
        """Set model information."""
        if not self._initialized:
            return
            
        self.model_info.info({
            'name': str(model_info.get('name', 'unknown')),
            'version': str(model_info.get('version', 'unknown')),
            'type': str(model_info.get('type', 'unknown')),
            'parameters': str(model_info.get('parameters', 0)),
            'size_mb': str(model_info.get('size_mb', 0))
        })
        
    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format."""
        if not self._initialized:
            self.init_metrics()
            
        # Update system metrics before generating output
        self.update_system_metrics()
        
        return generate_latest(self.registry)


# Global metrics registry instance
metrics_registry = MetricsRegistry()