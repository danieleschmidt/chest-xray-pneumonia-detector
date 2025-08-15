#!/usr/bin/env python3
"""
Advanced Monitoring System
Generation 2: Comprehensive monitoring, alerting, and observability
"""

import json
import os
import time
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging
from pathlib import Path

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    value: float
    labels: Dict[str, str] = None


@dataclass
class Alert:
    """Alert configuration and state."""
    name: str
    condition: str
    threshold: float
    severity: str
    description: str
    active: bool = False
    triggered_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None


class MetricsCollector:
    """Comprehensive metrics collection system."""
    
    def __init__(self, retention_hours: int = 24):
        self.metrics = defaultdict(lambda: deque(maxlen=retention_hours * 360))  # 10-second intervals
        self.retention_hours = retention_hours
        self.collection_interval = 10  # seconds
        self.running = False
        self.collector_thread = None
        
        # Performance counters
        self.prediction_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        self.last_prediction_time = None
        
        # Initialize logging
        self.logger = logging.getLogger("metrics_collector")
        
    def start_collection(self):
        """Start automatic metrics collection."""
        if self.running:
            return
        
        self.running = True
        self.collector_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collector_thread.start()
        self.logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection."""
        self.running = False
        if self.collector_thread:
            self.collector_thread.join(timeout=5)
        self.logger.info("Metrics collection stopped")
    
    def _collection_loop(self):
        """Main metrics collection loop."""
        while self.running:
            try:
                self._collect_system_metrics()
                self._collect_application_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        if not PSUTIL_AVAILABLE:
            return
            
        current_time = time.time()
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.record_metric("system_cpu_percent", cpu_percent, current_time)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.record_metric("system_memory_percent", memory.percent, current_time)
        self.record_metric("system_memory_available_gb", memory.available / (1024**3), current_time)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self.record_metric("system_disk_percent", disk_percent, current_time)
        self.record_metric("system_disk_free_gb", disk.free / (1024**3), current_time)
        
        # Network I/O
        network = psutil.net_io_counters()
        self.record_metric("system_network_bytes_sent", network.bytes_sent, current_time)
        self.record_metric("system_network_bytes_recv", network.bytes_recv, current_time)
        
        # Load average (Unix only)
        try:
            load_avg = os.getloadavg()
            self.record_metric("system_load_1min", load_avg[0], current_time)
            self.record_metric("system_load_5min", load_avg[1], current_time)
            self.record_metric("system_load_15min", load_avg[2], current_time)
        except (OSError, AttributeError):
            pass  # Not available on Windows
    
    def _collect_application_metrics(self):
        """Collect application-specific metrics."""
        current_time = time.time()
        
        # Prediction rate (predictions per minute)
        prediction_rate = self.prediction_count / max(1, (current_time - (current_time - 60)) / 60)
        self.record_metric("app_predictions_per_minute", prediction_rate, current_time)
        
        # Error rate
        total_requests = max(1, self.prediction_count)
        error_rate = (self.error_count / total_requests) * 100
        self.record_metric("app_error_rate_percent", error_rate, current_time)
        
        # Average response time
        if self.prediction_count > 0:
            avg_response_time = self.total_response_time / self.prediction_count
            self.record_metric("app_avg_response_time_ms", avg_response_time * 1000, current_time)
        
        # Time since last prediction
        if self.last_prediction_time:
            time_since_last = current_time - self.last_prediction_time
            self.record_metric("app_time_since_last_prediction_seconds", time_since_last, current_time)
        
        # Model memory usage (if models are loaded)
        self._collect_model_metrics(current_time)
    
    def _collect_model_metrics(self, current_time: float):
        """Collect model-specific metrics."""
        # Check for model files and their sizes
        model_dirs = ["saved_models", "models"]
        total_model_size = 0
        model_count = 0
        
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                for item in os.listdir(model_dir):
                    item_path = os.path.join(model_dir, item)
                    if os.path.isfile(item_path) and item.endswith(('.keras', '.h5', '.pb')):
                        total_model_size += os.path.getsize(item_path)
                        model_count += 1
        
        self.record_metric("model_total_size_mb", total_model_size / (1024**2), current_time)
        self.record_metric("model_count", model_count, current_time)
    
    def record_metric(self, name: str, value: float, timestamp: float = None, 
                     labels: Dict[str, str] = None):
        """Record a metric data point."""
        if timestamp is None:
            timestamp = time.time()
        
        metric_point = MetricPoint(timestamp=timestamp, value=value, labels=labels)
        self.metrics[name].append(metric_point)
    
    def record_prediction(self, response_time: float, success: bool = True):
        """Record a prediction event."""
        self.prediction_count += 1
        self.total_response_time += response_time
        self.last_prediction_time = time.time()
        
        if not success:
            self.error_count += 1
        
        # Record specific metrics
        self.record_metric("app_prediction_response_time_ms", response_time * 1000)
        self.record_metric("app_prediction_success", 1.0 if success else 0.0)
    
    def get_metric_values(self, name: str, minutes_back: int = 60) -> List[MetricPoint]:
        """Get metric values for the specified time range."""
        if name not in self.metrics:
            return []
        
        cutoff_time = time.time() - (minutes_back * 60)
        return [point for point in self.metrics[name] if point.timestamp >= cutoff_time]
    
    def get_metric_summary(self, name: str, minutes_back: int = 60) -> Dict[str, float]:
        """Get summary statistics for a metric."""
        values = self.get_metric_values(name, minutes_back)
        
        if not values:
            return {"count": 0, "avg": 0, "min": 0, "max": 0, "latest": 0}
        
        metric_values = [point.value for point in values]
        
        return {
            "count": len(metric_values),
            "avg": sum(metric_values) / len(metric_values),
            "min": min(metric_values),
            "max": max(metric_values),
            "latest": metric_values[-1] if metric_values else 0
        }


class AlertManager:
    """Advanced alerting system with multiple notification channels."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alerts = {}
        self.alert_history = []
        self.notification_channels = []
        self.evaluation_interval = 30  # seconds
        self.running = False
        self.alert_thread = None
        
        # Setup default alerts
        self._setup_default_alerts()
        
        self.logger = logging.getLogger("alert_manager")
    
    def _setup_default_alerts(self):
        """Configure default system alerts."""
        default_alerts = [
            Alert(
                name="high_cpu_usage",
                condition="system_cpu_percent > threshold",
                threshold=80.0,
                severity="warning",
                description="CPU usage is above 80%"
            ),
            Alert(
                name="high_memory_usage",
                condition="system_memory_percent > threshold",
                threshold=85.0,
                severity="warning",
                description="Memory usage is above 85%"
            ),
            Alert(
                name="low_disk_space",
                condition="system_disk_percent > threshold",
                threshold=90.0,
                severity="critical",
                description="Disk space is above 90%"
            ),
            Alert(
                name="high_error_rate",
                condition="app_error_rate_percent > threshold",
                threshold=10.0,
                severity="critical",
                description="Application error rate is above 10%"
            ),
            Alert(
                name="slow_response_time",
                condition="app_avg_response_time_ms > threshold",
                threshold=5000.0,
                severity="warning",
                description="Average response time is above 5 seconds"
            ),
            Alert(
                name="no_recent_predictions",
                condition="app_time_since_last_prediction_seconds > threshold",
                threshold=300.0,
                severity="warning",
                description="No predictions in the last 5 minutes"
            )
        ]
        
        for alert in default_alerts:
            self.alerts[alert.name] = alert
    
    def start_monitoring(self):
        """Start alert monitoring."""
        if self.running:
            return
        
        self.running = True
        self.alert_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.alert_thread.start()
        self.logger.info("Alert monitoring started")
    
    def stop_monitoring(self):
        """Stop alert monitoring."""
        self.running = False
        if self.alert_thread:
            self.alert_thread.join(timeout=5)
        self.logger.info("Alert monitoring stopped")
    
    def _monitoring_loop(self):
        """Main alert evaluation loop."""
        while self.running:
            try:
                self._evaluate_alerts()
                time.sleep(self.evaluation_interval)
            except Exception as e:
                self.logger.error(f"Error in alert evaluation: {e}")
                time.sleep(self.evaluation_interval)
    
    def _evaluate_alerts(self):
        """Evaluate all configured alerts."""
        for alert_name, alert in self.alerts.items():
            try:
                should_trigger = self._evaluate_alert_condition(alert)
                
                if should_trigger and not alert.active:
                    self._trigger_alert(alert)
                elif not should_trigger and alert.active:
                    self._resolve_alert(alert)
            
            except Exception as e:
                self.logger.error(f"Error evaluating alert {alert_name}: {e}")
    
    def _evaluate_alert_condition(self, alert: Alert) -> bool:
        """Evaluate if an alert condition is met."""
        # Extract metric name from condition
        if "system_cpu_percent" in alert.condition:
            metric_name = "system_cpu_percent"
        elif "system_memory_percent" in alert.condition:
            metric_name = "system_memory_percent"
        elif "system_disk_percent" in alert.condition:
            metric_name = "system_disk_percent"
        elif "app_error_rate_percent" in alert.condition:
            metric_name = "app_error_rate_percent"
        elif "app_avg_response_time_ms" in alert.condition:
            metric_name = "app_avg_response_time_ms"
        elif "app_time_since_last_prediction_seconds" in alert.condition:
            metric_name = "app_time_since_last_prediction_seconds"
        else:
            return False
        
        # Get recent metric values
        summary = self.metrics_collector.get_metric_summary(metric_name, minutes_back=5)
        
        if summary["count"] == 0:
            return False
        
        # Use latest value for evaluation
        current_value = summary["latest"]
        
        # Evaluate condition (simple threshold comparison for now)
        if ">" in alert.condition:
            return current_value > alert.threshold
        elif "<" in alert.condition:
            return current_value < alert.threshold
        
        return False
    
    def _trigger_alert(self, alert: Alert):
        """Trigger an alert."""
        alert.active = True
        alert.triggered_at = datetime.now()
        
        alert_event = {
            "alert_name": alert.name,
            "severity": alert.severity,
            "description": alert.description,
            "triggered_at": alert.triggered_at.isoformat(),
            "threshold": alert.threshold
        }
        
        self.alert_history.append(alert_event)
        self.logger.warning(f"ALERT TRIGGERED: {alert.name} - {alert.description}")
        
        # Send notifications
        self._send_notifications(alert_event)
    
    def _resolve_alert(self, alert: Alert):
        """Resolve an active alert."""
        alert.active = False
        alert.resolved_at = datetime.now()
        
        resolution_event = {
            "alert_name": alert.name,
            "resolved_at": alert.resolved_at.isoformat(),
            "duration_minutes": (alert.resolved_at - alert.triggered_at).total_seconds() / 60
        }
        
        self.logger.info(f"ALERT RESOLVED: {alert.name}")
        self._send_notifications(resolution_event, event_type="resolution")
    
    def _send_notifications(self, alert_data: Dict[str, Any], event_type: str = "alert"):
        """Send alert notifications through configured channels."""
        for channel in self.notification_channels:
            try:
                channel(alert_data, event_type)
            except Exception as e:
                self.logger.error(f"Failed to send notification via {channel}: {e}")
    
    def add_notification_channel(self, channel_func: Callable):
        """Add a notification channel function."""
        self.notification_channels.append(channel_func)
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of currently active alerts."""
        return [
            {
                "name": alert.name,
                "severity": alert.severity,
                "description": alert.description,
                "triggered_at": alert.triggered_at.isoformat() if alert.triggered_at else None
            }
            for alert in self.alerts.values() if alert.active
        ]


class PerformanceProfiler:
    """Application performance profiling and analysis."""
    
    def __init__(self):
        self.performance_data = defaultdict(list)
        self.active_traces = {}
        self.logger = logging.getLogger("performance_profiler")
    
    def start_trace(self, operation_name: str) -> str:
        """Start performance trace for an operation."""
        trace_id = f"{operation_name}_{int(time.time() * 1000)}"
        
        self.active_traces[trace_id] = {
            "operation": operation_name,
            "start_time": time.time(),
            "memory_start": self._get_memory_usage()
        }
        
        return trace_id
    
    def end_trace(self, trace_id: str, metadata: Dict[str, Any] = None):
        """End performance trace and record metrics."""
        if trace_id not in self.active_traces:
            self.logger.warning(f"Unknown trace ID: {trace_id}")
            return
        
        trace_data = self.active_traces.pop(trace_id)
        end_time = time.time()
        
        performance_record = {
            "operation": trace_data["operation"],
            "duration_ms": (end_time - trace_data["start_time"]) * 1000,
            "memory_delta_mb": (self._get_memory_usage() - trace_data["memory_start"]) / (1024 * 1024),
            "timestamp": end_time,
            "metadata": metadata or {}
        }
        
        self.performance_data[trace_data["operation"]].append(performance_record)
        
        # Log slow operations
        if performance_record["duration_ms"] > 1000:  # 1 second
            self.logger.warning(
                f"Slow operation: {trace_data['operation']} took "
                f"{performance_record['duration_ms']:.2f}ms"
            )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in bytes."""
        if not PSUTIL_AVAILABLE:
            return 0.0
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        except Exception:
            return 0.0
    
    def get_performance_summary(self, operation: str = None) -> Dict[str, Any]:
        """Get performance summary for operations."""
        if operation:
            data = self.performance_data.get(operation, [])
        else:
            data = []
            for op_data in self.performance_data.values():
                data.extend(op_data)
        
        if not data:
            return {"operation": operation, "count": 0}
        
        durations = [record["duration_ms"] for record in data]
        memory_deltas = [record["memory_delta_mb"] for record in data]
        
        return {
            "operation": operation or "all",
            "count": len(data),
            "avg_duration_ms": sum(durations) / len(durations),
            "min_duration_ms": min(durations),
            "max_duration_ms": max(durations),
            "avg_memory_delta_mb": sum(memory_deltas) / len(memory_deltas),
            "total_duration_ms": sum(durations)
        }


def create_monitoring_dashboard() -> Dict[str, Any]:
    """Create a comprehensive monitoring dashboard."""
    metrics_collector = MetricsCollector()
    alert_manager = AlertManager(metrics_collector)
    profiler = PerformanceProfiler()
    
    # Start monitoring
    metrics_collector.start_collection()
    alert_manager.start_monitoring()
    
    # Add console notification channel
    def console_notifier(alert_data: Dict[str, Any], event_type: str = "alert"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if event_type == "alert":
            print(f"[{timestamp}] ALERT: {alert_data['alert_name']} - {alert_data['description']}")
        else:
            print(f"[{timestamp}] RESOLVED: {alert_data['alert_name']}")
    
    alert_manager.add_notification_channel(console_notifier)
    
    return {
        "metrics_collector": metrics_collector,
        "alert_manager": alert_manager,
        "profiler": profiler,
        "status": "operational"
    }


if __name__ == "__main__":
    # Example usage and testing
    dashboard = create_monitoring_dashboard()
    
    metrics_collector = dashboard["metrics_collector"]
    profiler = dashboard["profiler"]
    
    # Simulate some application activity
    for i in range(5):
        trace_id = profiler.start_trace("model_prediction")
        time.sleep(0.1)  # Simulate processing time
        profiler.end_trace(trace_id, {"input_size": 224})
        
        metrics_collector.record_prediction(0.1, success=True)
    
    # Display some metrics
    time.sleep(2)
    
    cpu_summary = metrics_collector.get_metric_summary("system_cpu_percent")
    print(f"CPU Usage Summary: {cpu_summary}")
    
    perf_summary = profiler.get_performance_summary("model_prediction")
    print(f"Prediction Performance: {perf_summary}")
    
    # Clean shutdown
    dashboard["metrics_collector"].stop_collection()
    dashboard["alert_manager"].stop_monitoring()