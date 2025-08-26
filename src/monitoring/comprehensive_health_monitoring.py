"""Comprehensive Health Monitoring System for Medical AI Applications.

Implements advanced monitoring, alerting, and observability for medical AI systems
with focus on model performance, data quality, and system reliability.
"""

import json
import logging
import time
import threading
from collections import deque, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from queue import Queue
import statistics
import asyncio

import numpy as np
import psutil

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class HealthStatus(Enum):
    """System health status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNAVAILABLE = "unavailable"


@dataclass
class HealthMetric:
    """Health metric data structure."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None


@dataclass
class Alert:
    """Alert data structure."""
    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    timestamp: datetime
    source: str
    metric_name: Optional[str] = None
    current_value: Optional[float] = None
    threshold: Optional[float] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    tags: Dict[str, str] = field(default_factory=dict)


class MetricCollector:
    """Collects various system and application metrics."""
    
    def __init__(self, collection_interval: float = 60.0):
        self.collection_interval = collection_interval
        self.metrics = deque(maxlen=1000)  # Keep last 1000 metrics
        self.is_running = False
        self.collection_thread = None
    
    def start_collection(self):
        """Start metric collection."""
        if self.is_running:
            return
        
        self.is_running = True
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        logger.info("Metric collection started")
    
    def stop_collection(self):
        """Stop metric collection."""
        self.is_running = False
        if self.collection_thread:
            self.collection_thread.join()
        logger.info("Metric collection stopped")
    
    def _collection_loop(self):
        """Main collection loop."""
        while self.is_running:
            try:
                metrics = self._collect_metrics()
                timestamp = datetime.now()
                
                for metric_name, value in metrics.items():
                    metric = HealthMetric(
                        name=metric_name,
                        value=value,
                        unit=self._get_metric_unit(metric_name),
                        timestamp=timestamp
                    )
                    self.metrics.append(metric)
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _collect_metrics(self) -> Dict[str, float]:
        """Collect current metrics."""
        metrics = {}
        
        # System metrics
        metrics['cpu_usage_percent'] = psutil.cpu_percent(interval=1)
        metrics['memory_usage_percent'] = psutil.virtual_memory().percent
        metrics['disk_usage_percent'] = psutil.disk_usage('/').percent
        
        # Network metrics
        net_io = psutil.net_io_counters()
        metrics['network_bytes_sent'] = float(net_io.bytes_sent)
        metrics['network_bytes_recv'] = float(net_io.bytes_recv)
        
        # Process metrics
        process = psutil.Process()
        metrics['process_cpu_percent'] = process.cpu_percent()
        metrics['process_memory_mb'] = process.memory_info().rss / 1024 / 1024
        metrics['process_threads'] = process.num_threads()
        
        # Custom application metrics (placeholders)
        metrics['model_inference_time_ms'] = np.random.uniform(50, 200)
        metrics['model_accuracy'] = np.random.uniform(0.85, 0.95)
        metrics['data_quality_score'] = np.random.uniform(0.8, 1.0)
        metrics['active_sessions'] = np.random.randint(1, 50)
        
        return metrics
    
    def _get_metric_unit(self, metric_name: str) -> str:
        """Get unit for metric."""
        unit_map = {
            'cpu_usage_percent': '%',
            'memory_usage_percent': '%',
            'disk_usage_percent': '%',
            'network_bytes_sent': 'bytes',
            'network_bytes_recv': 'bytes',
            'process_cpu_percent': '%',
            'process_memory_mb': 'MB',
            'process_threads': 'count',
            'model_inference_time_ms': 'ms',
            'model_accuracy': 'score',
            'data_quality_score': 'score',
            'active_sessions': 'count'
        }
        return unit_map.get(metric_name, 'unit')
    
    def get_latest_metrics(self, n: int = 10) -> List[HealthMetric]:
        """Get latest n metrics."""
        return list(self.metrics)[-n:] if n <= len(self.metrics) else list(self.metrics)
    
    def get_metric_history(self, 
                          metric_name: str, 
                          time_window: timedelta = timedelta(hours=1)) -> List[HealthMetric]:
        """Get metric history within time window."""
        cutoff_time = datetime.now() - time_window
        return [m for m in self.metrics 
                if m.name == metric_name and m.timestamp > cutoff_time]


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self, max_alerts: int = 1000):
        self.alerts = deque(maxlen=max_alerts)
        self.alert_rules = {}
        self.notification_handlers = []
        self.alert_counter = 0
    
    def add_alert_rule(self, 
                      metric_name: str,
                      warning_threshold: Optional[float] = None,
                      critical_threshold: Optional[float] = None,
                      comparison: str = 'greater_than',
                      description: str = None):
        """Add alert rule for metric."""
        self.alert_rules[metric_name] = {
            'warning_threshold': warning_threshold,
            'critical_threshold': critical_threshold,
            'comparison': comparison,
            'description': description or f"Alert for {metric_name}"
        }
    
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add notification handler."""
        self.notification_handlers.append(handler)
    
    def check_metrics(self, metrics: List[HealthMetric]):
        """Check metrics against alert rules."""
        for metric in metrics:
            if metric.name in self.alert_rules:
                self._evaluate_alert_rule(metric)
    
    def _evaluate_alert_rule(self, metric: HealthMetric):
        """Evaluate alert rule for metric."""
        rule = self.alert_rules[metric.name]
        
        # Determine if alert should be triggered
        alert_severity = None
        threshold = None
        
        if rule['critical_threshold'] is not None:
            if self._compare_value(metric.value, rule['critical_threshold'], rule['comparison']):
                alert_severity = AlertSeverity.CRITICAL
                threshold = rule['critical_threshold']
        
        if alert_severity is None and rule['warning_threshold'] is not None:
            if self._compare_value(metric.value, rule['warning_threshold'], rule['comparison']):
                alert_severity = AlertSeverity.HIGH
                threshold = rule['warning_threshold']
        
        if alert_severity:
            self._create_alert(
                severity=alert_severity,
                title=f"{metric.name} threshold exceeded",
                description=rule['description'],
                source="metric_monitoring",
                metric_name=metric.name,
                current_value=metric.value,
                threshold=threshold
            )
    
    def _compare_value(self, value: float, threshold: float, comparison: str) -> bool:
        """Compare value against threshold."""
        if comparison == 'greater_than':
            return value > threshold
        elif comparison == 'less_than':
            return value < threshold
        elif comparison == 'equal':
            return abs(value - threshold) < 1e-6
        else:
            return False
    
    def _create_alert(self, 
                     severity: AlertSeverity,
                     title: str,
                     description: str,
                     source: str,
                     **kwargs):
        """Create new alert."""
        self.alert_counter += 1
        alert = Alert(
            alert_id=f"alert_{self.alert_counter}",
            severity=severity,
            title=title,
            description=description,
            timestamp=datetime.now(),
            source=source,
            **kwargs
        )
        
        self.alerts.append(alert)
        
        # Send notifications
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in notification handler: {e}")
        
        logger.warning(f"Alert created: {alert.title} (Severity: {alert.severity.value})")
    
    def resolve_alert(self, alert_id: str):
        """Resolve alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                logger.info(f"Alert resolved: {alert_id}")
                break
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts."""
        active_alerts = [a for a in self.alerts if not a.resolved]
        if severity:
            active_alerts = [a for a in active_alerts if a.severity == severity]
        return active_alerts
    
    def get_alert_summary(self) -> Dict[str, int]:
        """Get alert summary statistics."""
        active_alerts = self.get_active_alerts()
        summary = {
            'total_active': len(active_alerts),
            'critical': len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
            'high': len([a for a in active_alerts if a.severity == AlertSeverity.HIGH]),
            'medium': len([a for a in active_alerts if a.severity == AlertSeverity.MEDIUM]),
            'low': len([a for a in active_alerts if a.severity == AlertSeverity.LOW])
        }
        return summary


class HealthChecker:
    """Performs health checks on various system components."""
    
    def __init__(self):
        self.health_checks = {}
        self.last_check_results = {}
    
    def register_health_check(self, 
                            name: str, 
                            check_func: Callable[[], Tuple[HealthStatus, str]],
                            timeout: float = 30.0):
        """Register health check function."""
        self.health_checks[name] = {
            'func': check_func,
            'timeout': timeout
        }
    
    def run_health_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run all health checks."""
        results = {}
        
        for name, check_config in self.health_checks.items():
            try:
                start_time = time.time()
                status, message = check_config['func']()
                duration = time.time() - start_time
                
                results[name] = {
                    'status': status.value,
                    'message': message,
                    'duration_seconds': duration,
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                results[name] = {
                    'status': HealthStatus.CRITICAL.value,
                    'message': f"Health check failed: {str(e)}",
                    'duration_seconds': 0,
                    'timestamp': datetime.now().isoformat()
                }
        
        self.last_check_results = results
        return results
    
    def get_overall_health(self) -> Tuple[HealthStatus, str]:
        """Get overall system health status."""
        if not self.last_check_results:
            return HealthStatus.UNAVAILABLE, "No health checks run"
        
        status_counts = defaultdict(int)
        for result in self.last_check_results.values():
            status_counts[result['status']] += 1
        
        # Determine overall status
        if status_counts['critical'] > 0:
            return HealthStatus.CRITICAL, f"{status_counts['critical']} critical issues"
        elif status_counts['degraded'] > 0:
            return HealthStatus.DEGRADED, f"{status_counts['degraded']} degraded components"
        elif status_counts['warning'] > 0:
            return HealthStatus.WARNING, f"{status_counts['warning']} warnings"
        else:
            return HealthStatus.HEALTHY, "All systems operational"


class PerformanceTracker:
    """Tracks application performance metrics."""
    
    def __init__(self):
        self.operation_times = defaultdict(list)
        self.operation_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
    
    @contextmanager
    def track_operation(self, operation_name: str):
        """Context manager to track operation performance."""
        start_time = time.time()
        success = False
        
        try:
            yield
            success = True
        except Exception as e:
            self.error_counts[operation_name] += 1
            logger.error(f"Operation {operation_name} failed: {e}")
            raise
        finally:
            duration = time.time() - start_time
            self.operation_times[operation_name].append(duration)
            self.operation_counts[operation_name] += 1
            
            if success:
                logger.debug(f"Operation {operation_name} completed in {duration:.3f}s")
    
    def get_performance_stats(self, operation_name: str) -> Dict[str, float]:
        """Get performance statistics for operation."""
        times = self.operation_times[operation_name]
        if not times:
            return {}
        
        return {
            'count': len(times),
            'total_time': sum(times),
            'avg_time': statistics.mean(times),
            'min_time': min(times),
            'max_time': max(times),
            'median_time': statistics.median(times),
            'p95_time': np.percentile(times, 95),
            'error_count': self.error_counts[operation_name],
            'error_rate': self.error_counts[operation_name] / self.operation_counts[operation_name]
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for all operations."""
        return {op: self.get_performance_stats(op) 
                for op in self.operation_times.keys()}


class ComprehensiveHealthMonitor:
    """Main comprehensive health monitoring system."""
    
    def __init__(self, 
                 metrics_interval: float = 60.0,
                 health_check_interval: float = 300.0):
        self.metrics_collector = MetricCollector(metrics_interval)
        self.alert_manager = AlertManager()
        self.health_checker = HealthChecker()
        self.performance_tracker = PerformanceTracker()
        
        self.health_check_interval = health_check_interval
        self.monitoring_thread = None
        self.is_monitoring = False
        
        # Setup default alert rules
        self._setup_default_alert_rules()
        
        # Setup default health checks
        self._setup_default_health_checks()
        
        # Setup default notification handler
        self.alert_manager.add_notification_handler(self._default_notification_handler)
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules."""
        rules = [
            ('cpu_usage_percent', 80, 95, 'greater_than', 'High CPU usage'),
            ('memory_usage_percent', 85, 95, 'greater_than', 'High memory usage'),
            ('disk_usage_percent', 90, 98, 'greater_than', 'High disk usage'),
            ('model_accuracy', 0.8, 0.7, 'less_than', 'Low model accuracy'),
            ('data_quality_score', 0.85, 0.75, 'less_than', 'Low data quality'),
            ('model_inference_time_ms', 500, 1000, 'greater_than', 'Slow model inference')
        ]
        
        for metric_name, warning, critical, comparison, description in rules:
            self.alert_manager.add_alert_rule(
                metric_name=metric_name,
                warning_threshold=warning,
                critical_threshold=critical,
                comparison=comparison,
                description=description
            )
    
    def _setup_default_health_checks(self):
        """Setup default health checks."""
        
        def check_disk_space() -> Tuple[HealthStatus, str]:
            usage = psutil.disk_usage('/').percent
            if usage > 95:
                return HealthStatus.CRITICAL, f"Disk usage: {usage:.1f}%"
            elif usage > 85:
                return HealthStatus.WARNING, f"Disk usage: {usage:.1f}%"
            else:
                return HealthStatus.HEALTHY, f"Disk usage: {usage:.1f}%"
        
        def check_memory() -> Tuple[HealthStatus, str]:
            usage = psutil.virtual_memory().percent
            if usage > 95:
                return HealthStatus.CRITICAL, f"Memory usage: {usage:.1f}%"
            elif usage > 85:
                return HealthStatus.WARNING, f"Memory usage: {usage:.1f}%"
            else:
                return HealthStatus.HEALTHY, f"Memory usage: {usage:.1f}%"
        
        def check_model_service() -> Tuple[HealthStatus, str]:
            # Simulate model service check
            import random
            if random.random() > 0.1:  # 90% success rate
                return HealthStatus.HEALTHY, "Model service operational"
            else:
                return HealthStatus.DEGRADED, "Model service experiencing issues"
        
        self.health_checker.register_health_check("disk_space", check_disk_space)
        self.health_checker.register_health_check("memory", check_memory)
        self.health_checker.register_health_check("model_service", check_model_service)
    
    def _default_notification_handler(self, alert: Alert):
        """Default notification handler (logs alerts)."""
        logger.info(f"ALERT: {alert.severity.value.upper()} - {alert.title}")
        logger.info(f"Description: {alert.description}")
        if alert.current_value is not None and alert.threshold is not None:
            logger.info(f"Current value: {alert.current_value}, Threshold: {alert.threshold}")
    
    def start_monitoring(self):
        """Start comprehensive monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.metrics_collector.start_collection()
        
        # Start monitoring thread for health checks and alert evaluation
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Comprehensive health monitoring started")
    
    def stop_monitoring(self):
        """Stop comprehensive monitoring."""
        self.is_monitoring = False
        self.metrics_collector.stop_collection()
        
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        logger.info("Comprehensive health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        last_health_check = 0
        
        while self.is_monitoring:
            try:
                current_time = time.time()
                
                # Check metrics against alert rules
                recent_metrics = self.metrics_collector.get_latest_metrics(10)
                self.alert_manager.check_metrics(recent_metrics)
                
                # Run health checks periodically
                if current_time - last_health_check > self.health_check_interval:
                    self.health_checker.run_health_checks()
                    last_health_check = current_time
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        # Get latest metrics
        latest_metrics = self.metrics_collector.get_latest_metrics(20)
        metric_summary = {}
        
        for metric in latest_metrics:
            if metric.name not in metric_summary:
                metric_summary[metric.name] = []
            metric_summary[metric.name].append({
                'value': metric.value,
                'timestamp': metric.timestamp.isoformat(),
                'unit': metric.unit
            })
        
        # Get health status
        overall_health, health_message = self.health_checker.get_overall_health()
        
        # Get alert summary
        alert_summary = self.alert_manager.get_alert_summary()
        
        # Get performance stats
        performance_stats = self.performance_tracker.get_all_stats()
        
        return {
            'overall_health': {
                'status': overall_health.value,
                'message': health_message
            },
            'alerts': {
                'summary': alert_summary,
                'recent_alerts': [
                    {
                        'id': alert.alert_id,
                        'severity': alert.severity.value,
                        'title': alert.title,
                        'timestamp': alert.timestamp.isoformat(),
                        'resolved': alert.resolved
                    }
                    for alert in list(self.alert_manager.get_active_alerts())[-10:]
                ]
            },
            'metrics': metric_summary,
            'health_checks': self.health_checker.last_check_results,
            'performance': performance_stats,
            'monitoring_status': {
                'is_active': self.is_monitoring,
                'metrics_collected': len(self.metrics_collector.metrics),
                'uptime': 'N/A'  # Would track actual uptime in production
            }
        }
    
    @contextmanager
    def track_operation(self, operation_name: str):
        """Context manager for tracking operations."""
        with self.performance_tracker.track_operation(operation_name):
            yield


def demonstrate_health_monitoring():
    """Demonstrate comprehensive health monitoring system."""
    print("Comprehensive Health Monitoring System Demo")
    print("=" * 50)
    
    # Create monitoring system
    monitor = ComprehensiveHealthMonitor(metrics_interval=10.0, health_check_interval=30.0)
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        # Simulate some operations being tracked
        print("\n1. Tracking operations...")
        with monitor.track_operation("model_inference"):
            time.sleep(0.1)  # Simulate inference time
        
        with monitor.track_operation("data_preprocessing"):
            time.sleep(0.05)  # Simulate preprocessing time
        
        # Let monitoring run for a bit
        print("2. Collecting metrics and running health checks...")
        time.sleep(15)
        
        # Get dashboard data
        print("\n3. Monitoring Dashboard:")
        dashboard = monitor.get_monitoring_dashboard()
        
        print(f"Overall Health: {dashboard['overall_health']['status']}")
        print(f"Health Message: {dashboard['overall_health']['message']}")
        print(f"Active Alerts: {dashboard['alerts']['summary']['total_active']}")
        print(f"Metrics Collected: {dashboard['monitoring_status']['metrics_collected']}")
        
        # Show recent metrics
        print("\n4. Recent Metrics:")
        for metric_name, values in list(dashboard['metrics'].items())[:3]:
            if values:
                latest = values[-1]
                print(f"  {metric_name}: {latest['value']:.2f} {latest['unit']}")
        
        # Show performance stats
        print("\n5. Performance Statistics:")
        for op_name, stats in dashboard['performance'].items():
            if stats:
                print(f"  {op_name}: {stats['count']} calls, "
                     f"avg {stats['avg_time']:.3f}s")
        
    finally:
        # Stop monitoring
        monitor.stop_monitoring()
        print("\nMonitoring stopped.")


if __name__ == "__main__":
    demonstrate_health_monitoring()