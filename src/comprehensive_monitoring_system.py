#!/usr/bin/env python3
"""Comprehensive Monitoring System for Medical AI Applications"""

import asyncio
import time
import json
import logging
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import subprocess
import sqlite3
from datetime import datetime, timedelta
import statistics


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: float
    metric_name: str
    value: float
    tags: Dict[str, str]
    unit: str


@dataclass
class HealthCheck:
    """Health check definition and result"""
    name: str
    check_function: Callable
    interval: int  # seconds
    timeout: int   # seconds
    critical: bool = False
    last_run: Optional[float] = None
    last_result: Optional[bool] = None
    last_error: Optional[str] = None


@dataclass
class Alert:
    """Alert definition"""
    alert_id: str
    timestamp: float
    severity: str  # INFO, WARNING, CRITICAL
    component: str
    message: str
    metric_name: str
    current_value: float
    threshold: float
    tags: Dict[str, str]


class MetricsCollector:
    """Collects and stores system and application metrics"""
    
    def __init__(self, db_path: str = "metrics.db"):
        self.db_path = db_path
        self.metrics_buffer: List[MetricPoint] = []
        self.buffer_lock = threading.Lock()
        self._init_database()
        
    def _init_database(self):
        """Initialize metrics database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                metric_name TEXT,
                value REAL,
                tags TEXT,
                unit TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON metrics(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_metric_name ON metrics(metric_name)
        """)
        
        conn.commit()
        conn.close()
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None, unit: str = ""):
        """Record a metric point"""
        metric = MetricPoint(
            timestamp=time.time(),
            metric_name=name,
            value=value,
            tags=tags or {},
            unit=unit
        )
        
        with self.buffer_lock:
            self.metrics_buffer.append(metric)
    
    def flush_metrics(self):
        """Flush metrics buffer to database"""
        with self.buffer_lock:
            if not self.metrics_buffer:
                return
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for metric in self.metrics_buffer:
                cursor.execute("""
                    INSERT INTO metrics (timestamp, metric_name, value, tags, unit)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    metric.timestamp,
                    metric.metric_name,
                    metric.value,
                    json.dumps(metric.tags),
                    metric.unit
                ))
            
            conn.commit()
            conn.close()
            
            self.metrics_buffer.clear()
    
    def get_metrics(self, metric_name: str, start_time: float, end_time: float) -> List[MetricPoint]:
        """Retrieve metrics from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, metric_name, value, tags, unit
            FROM metrics
            WHERE metric_name = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """, (metric_name, start_time, end_time))
        
        results = cursor.fetchall()
        conn.close()
        
        metrics = []
        for row in results:
            metrics.append(MetricPoint(
                timestamp=row[0],
                metric_name=row[1],
                value=row[2],
                tags=json.loads(row[3]),
                unit=row[4]
            ))
        
        return metrics


class SystemMonitor:
    """Monitors system resources and health"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.monitoring_active = False
        self.monitor_thread = None
        
    def start_monitoring(self, interval: int = 30):
        """Start system monitoring"""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitoring_loop(self, interval: int):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._collect_system_metrics()
                time.sleep(interval)
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def _collect_system_metrics(self):
        """Collect system metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics.record_metric("system.cpu.usage", cpu_percent, unit="percent")
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.metrics.record_metric("system.memory.usage", memory.percent, unit="percent")
        self.metrics.record_metric("system.memory.available", memory.available, unit="bytes")
        self.metrics.record_metric("system.memory.total", memory.total, unit="bytes")
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self.metrics.record_metric("system.disk.usage", disk_percent, unit="percent")
        self.metrics.record_metric("system.disk.free", disk.free, unit="bytes")
        
        # Network I/O
        network = psutil.net_io_counters()
        self.metrics.record_metric("system.network.bytes_sent", network.bytes_sent, unit="bytes")
        self.metrics.record_metric("system.network.bytes_recv", network.bytes_recv, unit="bytes")
        
        # Load average (Unix systems)
        try:
            load_avg = psutil.getloadavg()
            self.metrics.record_metric("system.load.1min", load_avg[0])
            self.metrics.record_metric("system.load.5min", load_avg[1])
            self.metrics.record_metric("system.load.15min", load_avg[2])
        except AttributeError:
            # Windows doesn't have load average
            pass


class ApplicationMonitor:
    """Monitors application-specific metrics"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.request_times: List[float] = []
        self.error_count = 0
        self.prediction_count = 0
        self.prediction_confidences: List[float] = []
        
    def record_request(self, duration: float, status_code: int = 200):
        """Record API request metrics"""
        self.request_times.append(duration)
        
        # Keep only recent request times (last 1000)
        if len(self.request_times) > 1000:
            self.request_times = self.request_times[-1000:]
        
        self.metrics.record_metric("app.request.duration", duration, unit="seconds")
        self.metrics.record_metric("app.request.count", 1, tags={"status": str(status_code)})
        
        if status_code >= 400:
            self.error_count += 1
            self.metrics.record_metric("app.error.count", 1)
    
    def record_prediction(self, confidence: float, processing_time: float):
        """Record prediction metrics"""
        self.prediction_count += 1
        self.prediction_confidences.append(confidence)
        
        # Keep only recent predictions (last 1000)
        if len(self.prediction_confidences) > 1000:
            self.prediction_confidences = self.prediction_confidences[-1000:]
        
        self.metrics.record_metric("app.prediction.count", 1)
        self.metrics.record_metric("app.prediction.confidence", confidence, unit="percent")
        self.metrics.record_metric("app.prediction.processing_time", processing_time, unit="seconds")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get application performance summary"""
        if not self.request_times:
            return {
                "request_count": 0,
                "avg_response_time": 0,
                "p95_response_time": 0,
                "error_rate": 0,
                "prediction_count": self.prediction_count,
                "avg_confidence": 0
            }
        
        avg_response_time = statistics.mean(self.request_times)
        p95_response_time = statistics.quantiles(self.request_times, n=20)[18]  # 95th percentile
        error_rate = self.error_count / len(self.request_times) if self.request_times else 0
        avg_confidence = statistics.mean(self.prediction_confidences) if self.prediction_confidences else 0
        
        return {
            "request_count": len(self.request_times),
            "avg_response_time": avg_response_time,
            "p95_response_time": p95_response_time,
            "error_rate": error_rate,
            "prediction_count": self.prediction_count,
            "avg_confidence": avg_confidence
        }


class HealthChecker:
    """Performs health checks on system components"""
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.check_results: Dict[str, bool] = {}
        self.logger = logging.getLogger("health_checker")
        
    def register_health_check(self, 
                            name: str,
                            check_function: Callable,
                            interval: int = 60,
                            timeout: int = 10,
                            critical: bool = False):
        """Register a health check"""
        self.health_checks[name] = HealthCheck(
            name=name,
            check_function=check_function,
            interval=interval,
            timeout=timeout,
            critical=critical
        )
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {}
        overall_healthy = True
        
        for name, health_check in self.health_checks.items():
            try:
                # Check if it's time to run this health check
                current_time = time.time()
                if (health_check.last_run is not None and 
                    current_time - health_check.last_run < health_check.interval):
                    # Use cached result
                    results[name] = {
                        "healthy": health_check.last_result,
                        "last_run": health_check.last_run,
                        "error": health_check.last_error
                    }
                    if health_check.critical and not health_check.last_result:
                        overall_healthy = False
                    continue
                
                # Run the health check
                start_time = time.time()
                
                try:
                    # Run with timeout
                    if asyncio.iscoroutinefunction(health_check.check_function):
                        result = await asyncio.wait_for(
                            health_check.check_function(),
                            timeout=health_check.timeout
                        )
                    else:
                        result = health_check.check_function()
                    
                    health_check.last_result = bool(result)
                    health_check.last_error = None
                    
                except asyncio.TimeoutError:
                    health_check.last_result = False
                    health_check.last_error = "Timeout"
                    
                except Exception as e:
                    health_check.last_result = False
                    health_check.last_error = str(e)
                
                health_check.last_run = current_time
                
                results[name] = {
                    "healthy": health_check.last_result,
                    "last_run": health_check.last_run,
                    "error": health_check.last_error,
                    "duration": time.time() - start_time
                }
                
                if health_check.critical and not health_check.last_result:
                    overall_healthy = False
                    
            except Exception as e:
                self.logger.error(f"Error running health check {name}: {e}")
                results[name] = {
                    "healthy": False,
                    "error": str(e),
                    "last_run": time.time()
                }
                if health_check.critical:
                    overall_healthy = False
        
        return {
            "overall_healthy": overall_healthy,
            "checks": results,
            "timestamp": time.time()
        }


class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.notification_channels: List[Callable] = []
        
    def add_alert_rule(self, 
                      metric_name: str,
                      threshold: float,
                      operator: str = "greater",  # greater, less, equal
                      severity: str = "WARNING"):
        """Add alert rule for metric"""
        self.alert_rules[metric_name] = {
            "threshold": threshold,
            "operator": operator,
            "severity": severity
        }
    
    def add_notification_channel(self, channel: Callable):
        """Add notification channel"""
        self.notification_channels.append(channel)
    
    def check_alerts(self, metrics: List[MetricPoint]):
        """Check metrics against alert rules"""
        for metric in metrics:
            if metric.metric_name in self.alert_rules:
                rule = self.alert_rules[metric.metric_name]
                
                triggered = False
                if rule["operator"] == "greater" and metric.value > rule["threshold"]:
                    triggered = True
                elif rule["operator"] == "less" and metric.value < rule["threshold"]:
                    triggered = True
                elif rule["operator"] == "equal" and metric.value == rule["threshold"]:
                    triggered = True
                
                if triggered:
                    alert = Alert(
                        alert_id=f"{metric.metric_name}_{int(metric.timestamp)}",
                        timestamp=metric.timestamp,
                        severity=rule["severity"],
                        component=metric.tags.get("component", "unknown"),
                        message=f"{metric.metric_name} {rule['operator']} {rule['threshold']}",
                        metric_name=metric.metric_name,
                        current_value=metric.value,
                        threshold=rule["threshold"],
                        tags=metric.tags
                    )
                    
                    self.alerts.append(alert)
                    self._send_notifications(alert)
    
    def _send_notifications(self, alert: Alert):
        """Send alert notifications"""
        for channel in self.notification_channels:
            try:
                channel(alert)
            except Exception as e:
                logging.error(f"Failed to send notification: {e}")


class ComprehensiveMonitoringSystem:
    """Main monitoring system orchestrator"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.system_monitor = SystemMonitor(self.metrics_collector)
        self.app_monitor = ApplicationMonitor(self.metrics_collector)
        self.health_checker = HealthChecker()
        self.alert_manager = AlertManager()
        
        self.monitoring_active = False
        self.flush_thread = None
        
        # Setup default health checks
        self._setup_default_health_checks()
        
        # Setup default alert rules
        self._setup_default_alert_rules()
        
        # Setup logging
        self.logger = logging.getLogger("monitoring_system")
    
    def _setup_default_health_checks(self):
        """Setup default health checks"""
        
        def check_disk_space():
            disk = psutil.disk_usage('/')
            return (disk.free / disk.total) > 0.1  # More than 10% free
        
        def check_memory():
            memory = psutil.virtual_memory()
            return memory.percent < 90  # Less than 90% used
        
        def check_cpu():
            cpu_percent = psutil.cpu_percent(interval=1)
            return cpu_percent < 80  # Less than 80% used
        
        async def check_model_health():
            # Simulate model health check
            return True
        
        self.health_checker.register_health_check("disk_space", check_disk_space, critical=True)
        self.health_checker.register_health_check("memory_usage", check_memory, critical=True)
        self.health_checker.register_health_check("cpu_usage", check_cpu, critical=False)
        self.health_checker.register_health_check("model_health", check_model_health, critical=True)
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules"""
        self.alert_manager.add_alert_rule("system.cpu.usage", 80, "greater", "WARNING")
        self.alert_manager.add_alert_rule("system.memory.usage", 90, "greater", "CRITICAL")
        self.alert_manager.add_alert_rule("system.disk.usage", 90, "greater", "CRITICAL")
        self.alert_manager.add_alert_rule("app.error.count", 10, "greater", "WARNING")
        self.alert_manager.add_alert_rule("app.prediction.confidence", 0.7, "less", "WARNING")
    
    def start_monitoring(self):
        """Start comprehensive monitoring"""
        self.monitoring_active = True
        
        # Start system monitoring
        self.system_monitor.start_monitoring()
        
        # Start metrics flushing
        self.flush_thread = threading.Thread(
            target=self._flush_loop,
            daemon=True
        )
        self.flush_thread.start()
        
        self.logger.info("Comprehensive monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        self.system_monitor.stop_monitoring()
        
        if self.flush_thread:
            self.flush_thread.join()
        
        # Final flush
        self.metrics_collector.flush_metrics()
        
        self.logger.info("Comprehensive monitoring stopped")
    
    def _flush_loop(self):
        """Metrics flushing loop"""
        while self.monitoring_active:
            try:
                self.metrics_collector.flush_metrics()
                time.sleep(10)  # Flush every 10 seconds
            except Exception as e:
                self.logger.error(f"Error flushing metrics: {e}")
                time.sleep(10)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        # Get health check results
        health_results = await self.health_checker.run_health_checks()
        
        # Get application performance summary
        app_performance = self.app_monitor.get_performance_summary()
        
        # Get recent alerts
        recent_alerts = sorted(
            self.alert_manager.alerts[-10:],
            key=lambda x: x.timestamp,
            reverse=True
        )
        
        # Get system metrics summary
        current_time = time.time()
        start_time = current_time - 300  # Last 5 minutes
        
        cpu_metrics = self.metrics_collector.get_metrics("system.cpu.usage", start_time, current_time)
        memory_metrics = self.metrics_collector.get_metrics("system.memory.usage", start_time, current_time)
        
        avg_cpu = statistics.mean([m.value for m in cpu_metrics]) if cpu_metrics else 0
        avg_memory = statistics.mean([m.value for m in memory_metrics]) if memory_metrics else 0
        
        return {
            "timestamp": current_time,
            "overall_status": "healthy" if health_results["overall_healthy"] else "unhealthy",
            "health_checks": health_results,
            "system_metrics": {
                "avg_cpu_usage": avg_cpu,
                "avg_memory_usage": avg_memory
            },
            "application_performance": app_performance,
            "recent_alerts": [asdict(alert) for alert in recent_alerts],
            "monitoring_active": self.monitoring_active
        }
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        current_time = time.time()
        
        # Get metrics for the last hour
        start_time = current_time - 3600
        
        report = {
            "timestamp": current_time,
            "period": "last_hour",
            "system_summary": {},
            "application_summary": self.app_monitor.get_performance_summary(),
            "alert_summary": {
                "total_alerts": len(self.alert_manager.alerts),
                "critical_alerts": len([a for a in self.alert_manager.alerts if a.severity == "CRITICAL"]),
                "warning_alerts": len([a for a in self.alert_manager.alerts if a.severity == "WARNING"])
            },
            "recommendations": []
        }
        
        # Add system metrics summary
        for metric_name in ["system.cpu.usage", "system.memory.usage", "system.disk.usage"]:
            metrics = self.metrics_collector.get_metrics(metric_name, start_time, current_time)
            if metrics:
                values = [m.value for m in metrics]
                report["system_summary"][metric_name] = {
                    "avg": statistics.mean(values),
                    "max": max(values),
                    "min": min(values),
                    "count": len(values)
                }
        
        # Generate recommendations
        if report["application_summary"]["error_rate"] > 0.05:
            report["recommendations"].append("High error rate detected - investigate application errors")
        
        if report["system_summary"].get("system.cpu.usage", {}).get("avg", 0) > 70:
            report["recommendations"].append("High CPU usage - consider scaling resources")
        
        if report["system_summary"].get("system.memory.usage", {}).get("avg", 0) > 80:
            report["recommendations"].append("High memory usage - investigate memory leaks")
        
        return report


# Example notification channels
def email_notification(alert: Alert):
    """Email notification channel (mock implementation)"""
    print(f"EMAIL ALERT: {alert.severity} - {alert.message}")


def console_notification(alert: Alert):
    """Console notification channel"""
    print(f"ALERT [{alert.severity}] {alert.component}: {alert.message} (Current: {alert.current_value}, Threshold: {alert.threshold})")


# Example usage
async def main():
    """Example usage of the monitoring system"""
    
    # Create monitoring system
    monitoring = ComprehensiveMonitoringSystem()
    
    # Add notification channels
    monitoring.alert_manager.add_notification_channel(console_notification)
    
    # Start monitoring
    monitoring.start_monitoring()
    
    try:
        # Simulate some application activity
        for i in range(10):
            # Record some requests
            monitoring.app_monitor.record_request(0.1 + i * 0.01, 200)
            
            # Record some predictions
            monitoring.app_monitor.record_prediction(0.85 + i * 0.01, 0.05)
            
            await asyncio.sleep(1)
        
        # Get system status
        status = await monitoring.get_system_status()
        print(json.dumps(status, indent=2, default=str))
        
        # Generate report
        report = monitoring.generate_monitoring_report()
        print(json.dumps(report, indent=2, default=str))
        
    finally:
        # Stop monitoring
        monitoring.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())