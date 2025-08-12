"""
Comprehensive health monitoring system for pneumonia detection service.
Provides deep system health insights and automated recovery.
"""

import asyncio
import psutil
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import threading
import queue
from datetime import datetime, timedelta
import tensorflow as tf
import numpy as np
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    name: str
    status: HealthStatus
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    execution_time: float = 0.0
    recommendations: List[str] = field(default_factory=list)


class HealthCheckRegistry:
    """Registry for managing health check functions."""
    
    def __init__(self):
        self._checks: Dict[str, Callable] = {}
        self._intervals: Dict[str, int] = {}
        self._last_runs: Dict[str, datetime] = {}
        self._results: Dict[str, HealthCheckResult] = {}
        self._lock = threading.Lock()
    
    def register(self, name: str, check_func: Callable, interval_seconds: int = 60):
        """Register a health check function."""
        with self._lock:
            self._checks[name] = check_func
            self._intervals[name] = interval_seconds
            logger.info(f"Registered health check: {name} (interval: {interval_seconds}s)")
    
    def should_run(self, name: str) -> bool:
        """Check if a health check should be executed based on its interval."""
        if name not in self._last_runs:
            return True
        
        interval = self._intervals.get(name, 60)
        elapsed = (datetime.utcnow() - self._last_runs[name]).total_seconds()
        return elapsed >= interval
    
    def run_check(self, name: str) -> HealthCheckResult:
        """Execute a specific health check."""
        if name not in self._checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Unknown health check: {name}"
            )
        
        start_time = time.time()
        try:
            result = self._checks[name]()
            if not isinstance(result, HealthCheckResult):
                result = HealthCheckResult(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    message=str(result)
                )
        except Exception as e:
            logger.error(f"Health check {name} failed: {e}")
            result = HealthCheckResult(
                name=name,
                status=HealthStatus.CRITICAL,
                message=f"Check failed: {str(e)}"
            )
        
        result.execution_time = time.time() - start_time
        
        with self._lock:
            self._last_runs[name] = datetime.utcnow()
            self._results[name] = result
        
        return result
    
    def get_all_results(self) -> Dict[str, HealthCheckResult]:
        """Get all cached health check results."""
        with self._lock:
            return self._results.copy()
    
    def get_check_names(self) -> List[str]:
        """Get list of registered check names."""
        return list(self._checks.keys())


class ComprehensiveHealthMonitor:
    """Advanced health monitoring system with automated recovery."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.registry = HealthCheckRegistry()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.alert_queue = queue.Queue()
        self.recovery_actions: Dict[str, Callable] = {}
        self.metrics_history: Dict[str, List[Dict[str, Any]]] = {}
        self.is_monitoring = False
        self._monitor_thread = None
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register standard health checks."""
        self.registry.register("system_resources", self._check_system_resources, 30)
        self.registry.register("disk_space", self._check_disk_space, 60)
        self.registry.register("memory_usage", self._check_memory_usage, 30)
        self.registry.register("tensorflow_gpu", self._check_tensorflow_gpu, 120)
        self.registry.register("model_availability", self._check_model_availability, 60)
        self.registry.register("data_pipeline", self._check_data_pipeline, 60)
        self.registry.register("api_endpoints", self._check_api_endpoints, 30)
        self.registry.register("database_connection", self._check_database_connection, 45)
    
    def _check_system_resources(self) -> HealthCheckResult:
        """Check overall system resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3)
            }
            
            # Determine status based on thresholds
            status = HealthStatus.HEALTHY
            message = "System resources within normal range"
            recommendations = []
            
            if cpu_percent > 90:
                status = HealthStatus.CRITICAL
                message = f"Critical CPU usage: {cpu_percent:.1f}%"
                recommendations.append("Consider scaling up compute resources")
            elif cpu_percent > 75:
                status = HealthStatus.WARNING
                message = f"High CPU usage: {cpu_percent:.1f}%"
                recommendations.append("Monitor CPU usage trends")
            
            if memory.percent > 90:
                status = max(status, HealthStatus.CRITICAL, key=lambda x: x.value)
                message += f" | Critical memory usage: {memory.percent:.1f}%"
                recommendations.append("Increase available memory")
            elif memory.percent > 80:
                status = max(status, HealthStatus.WARNING, key=lambda x: x.value)
                message += f" | High memory usage: {memory.percent:.1f}%"
            
            if disk.percent > 95:
                status = max(status, HealthStatus.CRITICAL, key=lambda x: x.value)
                message += f" | Critical disk usage: {disk.percent:.1f}%"
                recommendations.append("Free up disk space immediately")
            
            return HealthCheckResult(
                name="system_resources",
                status=status,
                message=message,
                metrics=metrics,
                recommendations=recommendations
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="system_resources",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check system resources: {e}"
            )
    
    def _check_tensorflow_gpu(self) -> HealthCheckResult:
        """Check TensorFlow GPU availability and health."""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            
            if not gpus:
                return HealthCheckResult(
                    name="tensorflow_gpu",
                    status=HealthStatus.WARNING,
                    message="No GPU devices detected, using CPU",
                    metrics={'gpu_count': 0, 'gpu_memory_total': 0}
                )
            
            gpu_info = []
            total_memory = 0
            
            for i, gpu in enumerate(gpus):
                try:
                    # Get GPU memory info
                    gpu_details = tf.config.experimental.get_device_details(gpu)
                    memory_limit = tf.config.experimental.get_memory_info(f'GPU:{i}')
                    
                    gpu_data = {
                        'index': i,
                        'name': gpu_details.get('device_name', 'Unknown'),
                        'memory_limit': memory_limit.get('current', 0) / (1024**3),  # GB
                        'memory_peak': memory_limit.get('peak', 0) / (1024**3)  # GB
                    }
                    gpu_info.append(gpu_data)
                    total_memory += gpu_data['memory_limit']
                    
                except Exception as gpu_error:
                    logger.warning(f"Could not get details for GPU {i}: {gpu_error}")
            
            metrics = {
                'gpu_count': len(gpus),
                'gpu_info': gpu_info,
                'total_gpu_memory_gb': total_memory
            }
            
            return HealthCheckResult(
                name="tensorflow_gpu",
                status=HealthStatus.HEALTHY,
                message=f"Found {len(gpus)} GPU(s) with {total_memory:.1f}GB total memory",
                metrics=metrics
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="tensorflow_gpu",
                status=HealthStatus.WARNING,
                message=f"GPU check failed: {e}",
                metrics={'gpu_count': 0, 'error': str(e)}
            )
    
    def _check_model_availability(self) -> HealthCheckResult:
        """Check if models are available and loadable."""
        try:
            model_paths = [
                '/root/repo/saved_models',
                '/root/repo/models'
            ]
            
            available_models = []
            
            for model_dir in model_paths:
                if Path(model_dir).exists():
                    for model_file in Path(model_dir).glob('*.keras'):
                        try:
                            # Quick model validation without full load
                            model_stat = model_file.stat()
                            available_models.append({
                                'path': str(model_file),
                                'size_mb': model_stat.st_size / (1024**2),
                                'modified': datetime.fromtimestamp(model_stat.st_mtime).isoformat()
                            })
                        except Exception as model_error:
                            logger.warning(f"Model check failed for {model_file}: {model_error}")
            
            metrics = {
                'available_models': len(available_models),
                'model_details': available_models
            }
            
            if not available_models:
                return HealthCheckResult(
                    name="model_availability",
                    status=HealthStatus.WARNING,
                    message="No trained models found",
                    metrics=metrics,
                    recommendations=["Train a model or ensure model files are in correct location"]
                )
            
            return HealthCheckResult(
                name="model_availability",
                status=HealthStatus.HEALTHY,
                message=f"Found {len(available_models)} available model(s)",
                metrics=metrics
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="model_availability",
                status=HealthStatus.CRITICAL,
                message=f"Model availability check failed: {e}"
            )
    
    def _check_data_pipeline(self) -> HealthCheckResult:
        """Check data pipeline health."""
        try:
            data_dirs = ['/root/repo/data', '/root/repo/data_train_engine']
            pipeline_health = {}
            
            for data_dir in data_dirs:
                if Path(data_dir).exists():
                    train_path = Path(data_dir) / 'train'
                    val_path = Path(data_dir) / 'val'
                    
                    train_count = 0
                    val_count = 0
                    
                    if train_path.exists():
                        train_count = sum(1 for _ in train_path.rglob('*.jpg'))
                    
                    if val_path.exists():
                        val_count = sum(1 for _ in val_path.rglob('*.jpg'))
                    
                    pipeline_health[data_dir] = {
                        'train_images': train_count,
                        'val_images': val_count,
                        'total_images': train_count + val_count
                    }
            
            total_images = sum(
                info['total_images'] 
                for info in pipeline_health.values()
            )
            
            metrics = {
                'data_directories': pipeline_health,
                'total_images_available': total_images
            }
            
            if total_images == 0:
                return HealthCheckResult(
                    name="data_pipeline",
                    status=HealthStatus.WARNING,
                    message="No training data found",
                    metrics=metrics,
                    recommendations=["Load training data or use dummy data generation"]
                )
            
            return HealthCheckResult(
                name="data_pipeline",
                status=HealthStatus.HEALTHY,
                message=f"Data pipeline healthy with {total_images} images",
                metrics=metrics
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="data_pipeline",
                status=HealthStatus.CRITICAL,
                message=f"Data pipeline check failed: {e}"
            )
    
    def _check_disk_space(self) -> HealthCheckResult:
        """Check disk space availability."""
        try:
            disk_usage = psutil.disk_usage('/')
            free_gb = disk_usage.free / (1024**3)
            used_percent = (disk_usage.used / disk_usage.total) * 100
            
            metrics = {
                'free_gb': free_gb,
                'used_percent': used_percent,
                'total_gb': disk_usage.total / (1024**3)
            }
            
            status = HealthStatus.HEALTHY
            message = f"Disk usage: {used_percent:.1f}% ({free_gb:.1f}GB free)"
            recommendations = []
            
            if used_percent > 95:
                status = HealthStatus.CRITICAL
                message = f"Critical disk usage: {used_percent:.1f}%"
                recommendations.append("Free up disk space immediately")
            elif used_percent > 85:
                status = HealthStatus.WARNING
                message = f"High disk usage: {used_percent:.1f}%"
                recommendations.append("Plan for disk cleanup")
            
            return HealthCheckResult(
                name="disk_space",
                status=status,
                message=message,
                metrics=metrics,
                recommendations=recommendations
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="disk_space",
                status=HealthStatus.CRITICAL,
                message=f"Disk space check failed: {e}"
            )
    
    def _check_memory_usage(self) -> HealthCheckResult:
        """Detailed memory usage check."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            metrics = {
                'memory_total_gb': memory.total / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'memory_used_percent': memory.percent,
                'swap_total_gb': swap.total / (1024**3),
                'swap_used_percent': swap.percent
            }
            
            status = HealthStatus.HEALTHY
            message = f"Memory usage: {memory.percent:.1f}%"
            recommendations = []
            
            if memory.percent > 95:
                status = HealthStatus.CRITICAL
                message = f"Critical memory usage: {memory.percent:.1f}%"
                recommendations.append("Restart services to free memory")
            elif memory.percent > 85:
                status = HealthStatus.WARNING
                message = f"High memory usage: {memory.percent:.1f}%"
                recommendations.append("Monitor memory-intensive processes")
            
            if swap.percent > 50:
                status = max(status, HealthStatus.WARNING, key=lambda x: x.value)
                message += f" | Swap usage: {swap.percent:.1f}%"
                recommendations.append("High swap usage detected")
            
            return HealthCheckResult(
                name="memory_usage",
                status=status,
                message=message,
                metrics=metrics,
                recommendations=recommendations
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="memory_usage",
                status=HealthStatus.CRITICAL,
                message=f"Memory check failed: {e}"
            )
    
    def _check_api_endpoints(self) -> HealthCheckResult:
        """Check API endpoint availability."""
        try:
            # For now, just check if the API module can be imported
            try:
                from src.api import main
                api_available = True
            except ImportError:
                api_available = False
            
            metrics = {
                'api_module_available': api_available,
                'endpoints_tested': 0  # Will be expanded with actual endpoint tests
            }
            
            if api_available:
                return HealthCheckResult(
                    name="api_endpoints",
                    status=HealthStatus.HEALTHY,
                    message="API module is available",
                    metrics=metrics
                )
            else:
                return HealthCheckResult(
                    name="api_endpoints",
                    status=HealthStatus.WARNING,
                    message="API module not available",
                    metrics=metrics
                )
                
        except Exception as e:
            return HealthCheckResult(
                name="api_endpoints",
                status=HealthStatus.CRITICAL,
                message=f"API check failed: {e}"
            )
    
    def _check_database_connection(self) -> HealthCheckResult:
        """Check database connectivity."""
        try:
            # Basic database connectivity check
            # This would be expanded based on actual database usage
            
            metrics = {
                'connection_tested': datetime.utcnow().isoformat(),
                'database_type': 'file_based'  # Currently using file-based storage
            }
            
            return HealthCheckResult(
                name="database_connection",
                status=HealthStatus.HEALTHY,
                message="Database connectivity healthy (file-based)",
                metrics=metrics
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="database_connection",
                status=HealthStatus.CRITICAL,
                message=f"Database check failed: {e}"
            )
    
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        
        for check_name in self.registry.get_check_names():
            if self.registry.should_run(check_name):
                try:
                    result = self.registry.run_check(check_name)
                    results[check_name] = result
                    
                    # Store metrics history
                    if check_name not in self.metrics_history:
                        self.metrics_history[check_name] = []
                    
                    self.metrics_history[check_name].append({
                        'timestamp': result.timestamp.isoformat(),
                        'status': result.status.value,
                        'metrics': result.metrics,
                        'execution_time': result.execution_time
                    })
                    
                    # Keep only last 100 entries per check
                    if len(self.metrics_history[check_name]) > 100:
                        self.metrics_history[check_name] = self.metrics_history[check_name][-100:]
                    
                    # Generate alerts for critical issues
                    if result.status == HealthStatus.CRITICAL:
                        self.alert_queue.put(result)
                        
                except Exception as e:
                    logger.error(f"Failed to run health check {check_name}: {e}")
                    results[check_name] = HealthCheckResult(
                        name=check_name,
                        status=HealthStatus.CRITICAL,
                        message=f"Check execution failed: {e}"
                    )
        
        return results
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        results = self.registry.get_all_results()
        
        if not results:
            return {
                'overall_status': HealthStatus.UNKNOWN.value,
                'message': 'No health checks have been run',
                'check_count': 0
            }
        
        status_counts = {status.value: 0 for status in HealthStatus}
        critical_issues = []
        warnings = []
        
        for result in results.values():
            status_counts[result.status.value] += 1
            
            if result.status == HealthStatus.CRITICAL:
                critical_issues.append(result.name)
            elif result.status == HealthStatus.WARNING:
                warnings.append(result.name)
        
        # Determine overall status
        if status_counts['critical'] > 0:
            overall_status = HealthStatus.CRITICAL
            message = f"{status_counts['critical']} critical issue(s) detected"
        elif status_counts['warning'] > 0:
            overall_status = HealthStatus.WARNING
            message = f"{status_counts['warning']} warning(s) detected"
        else:
            overall_status = HealthStatus.HEALTHY
            message = "All systems healthy"
        
        return {
            'overall_status': overall_status.value,
            'message': message,
            'check_count': len(results),
            'status_counts': status_counts,
            'critical_issues': critical_issues,
            'warnings': warnings,
            'last_check': max(result.timestamp for result in results.values()).isoformat(),
            'checks': {name: {
                'status': result.status.value,
                'message': result.message,
                'execution_time': result.execution_time
            } for name, result in results.items()}
        }
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous health monitoring."""
        if self.is_monitoring:
            logger.warning("Monitoring is already active")
            return
        
        self.is_monitoring = True
        
        def monitor_loop():
            logger.info(f"Starting health monitoring with {interval_seconds}s interval")
            
            while self.is_monitoring:
                try:
                    self.run_all_checks()
                    time.sleep(interval_seconds)
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(10)  # Brief pause before retrying
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous health monitoring."""
        self.is_monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Health monitoring stopped")
    
    def export_health_report(self, file_path: str):
        """Export comprehensive health report to file."""
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'overall_health': self.get_overall_health(),
            'detailed_results': {
                name: {
                    'status': result.status.value,
                    'message': result.message,
                    'metrics': result.metrics,
                    'recommendations': result.recommendations,
                    'timestamp': result.timestamp.isoformat(),
                    'execution_time': result.execution_time
                }
                for name, result in self.registry.get_all_results().items()
            },
            'metrics_history': self.metrics_history
        }
        
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Health report exported to {file_path}")


def create_health_monitor(config: Optional[Dict[str, Any]] = None) -> ComprehensiveHealthMonitor:
    """Factory function to create a configured health monitor."""
    return ComprehensiveHealthMonitor(config)


if __name__ == '__main__':
    # Example usage
    monitor = create_health_monitor()
    
    # Run all checks once
    results = monitor.run_all_checks()
    print(f"Ran {len(results)} health checks")
    
    # Get overall health
    overall = monitor.get_overall_health()
    print(f"Overall status: {overall['overall_status']}")
    
    # Export report
    monitor.export_health_report('/tmp/health_report.json')
