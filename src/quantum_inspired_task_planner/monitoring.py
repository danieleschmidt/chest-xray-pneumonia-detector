"""Monitoring and observability for quantum task planner.

Provides comprehensive monitoring, metrics collection, and health checking
for the quantum-inspired task scheduling system.
"""

import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import json

from .quantum_scheduler import QuantumTask, TaskStatus, TaskPriority

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for quantum task operations."""
    operation_count: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    error_count: int = 0
    last_operation_time: Optional[datetime] = None
    
    def record_operation(self, duration: float, success: bool = True) -> None:
        """Record a completed operation."""
        self.operation_count += 1
        self.total_duration += duration
        self.min_duration = min(self.min_duration, duration)
        self.max_duration = max(self.max_duration, duration)
        if not success:
            self.error_count += 1
        self.last_operation_time = datetime.now()
    
    @property
    def average_duration(self) -> float:
        """Calculate average operation duration."""
        return self.total_duration / self.operation_count if self.operation_count > 0 else 0.0
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate percentage."""
        return (self.error_count / self.operation_count * 100) if self.operation_count > 0 else 0.0


class QuantumMetricsCollector:
    """Collects and aggregates metrics for quantum scheduling operations."""
    
    def __init__(self, metrics_retention_hours: int = 24):
        self.metrics_retention = timedelta(hours=metrics_retention_hours)
        self.operation_metrics: Dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
        self.task_lifecycle_metrics: Dict[str, Dict] = {}
        self.quantum_state_metrics: List[Dict] = []
        self.resource_usage_history: deque = deque(maxlen=1000)
        self._lock = threading.Lock()
        
    def record_task_creation(self, task: QuantumTask, duration: float) -> None:
        """Record task creation metrics."""
        with self._lock:
            self.operation_metrics["task_creation"].record_operation(duration)
            self.task_lifecycle_metrics[task.id] = {
                "created_at": task.created_at,
                "priority": task.priority.value,
                "estimated_duration": task.estimated_duration.total_seconds(),
                "dependencies_count": len(task.dependencies),
                "resource_requirements": task.resource_requirements.copy()
            }
    
    def record_task_completion(self, task: QuantumTask, actual_duration: timedelta) -> None:
        """Record task completion metrics."""
        with self._lock:
            completion_time = (datetime.now() - task.started_at).total_seconds() if task.started_at else 0.0
            self.operation_metrics["task_completion"].record_operation(completion_time)
            
            if task.id in self.task_lifecycle_metrics:
                self.task_lifecycle_metrics[task.id].update({
                    "completed_at": task.completed_at,
                    "actual_duration": actual_duration.total_seconds(),
                    "duration_accuracy": self._calculate_duration_accuracy(task, actual_duration)
                })
    
    def record_schedule_optimization(self, algorithm: str, duration: float, 
                                   iterations: int, energy: float, success: bool) -> None:
        """Record schedule optimization metrics."""
        with self._lock:
            self.operation_metrics[f"optimization_{algorithm}"].record_operation(duration, success)
            
            optimization_record = {
                "timestamp": datetime.now(),
                "algorithm": algorithm,
                "duration": duration,
                "iterations": iterations,
                "final_energy": energy,
                "success": success
            }
            self.quantum_state_metrics.append(optimization_record)
    
    def record_resource_utilization(self, utilization_data: Dict[str, Dict]) -> None:
        """Record resource utilization snapshot."""
        with self._lock:
            timestamp = datetime.now()
            self.resource_usage_history.append({
                "timestamp": timestamp,
                "utilization": utilization_data.copy()
            })
    
    def _calculate_duration_accuracy(self, task: QuantumTask, actual_duration: timedelta) -> float:
        """Calculate how accurate the duration estimate was."""
        estimated_seconds = task.estimated_duration.total_seconds()
        actual_seconds = actual_duration.total_seconds()
        
        if estimated_seconds == 0:
            return 0.0
        
        accuracy = 1.0 - abs(estimated_seconds - actual_seconds) / estimated_seconds
        return max(0.0, accuracy)  # Clamp to non-negative
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        with self._lock:
            summary = {
                "operation_metrics": {},
                "task_lifecycle_summary": self._get_task_lifecycle_summary(),
                "quantum_optimization_summary": self._get_optimization_summary(),
                "resource_utilization_summary": self._get_resource_summary(),
                "collection_timestamp": datetime.now()
            }
            
            # Summarize operation metrics
            for operation, metrics in self.operation_metrics.items():
                summary["operation_metrics"][operation] = {
                    "count": metrics.operation_count,
                    "average_duration": metrics.average_duration,
                    "min_duration": metrics.min_duration if metrics.min_duration != float('inf') else None,
                    "max_duration": metrics.max_duration,
                    "error_rate": metrics.error_rate,
                    "last_operation": metrics.last_operation_time
                }
            
            return summary
    
    def _get_task_lifecycle_summary(self) -> Dict[str, Any]:
        """Summarize task lifecycle metrics."""
        if not self.task_lifecycle_metrics:
            return {}
        
        total_tasks = len(self.task_lifecycle_metrics)
        completed_tasks = [
            metrics for metrics in self.task_lifecycle_metrics.values()
            if "completed_at" in metrics
        ]
        
        if completed_tasks:
            duration_accuracies = [metrics.get("duration_accuracy", 0) for metrics in completed_tasks]
            avg_accuracy = sum(duration_accuracies) / len(duration_accuracies)
        else:
            avg_accuracy = 0.0
        
        return {
            "total_tasks": total_tasks,
            "completed_tasks": len(completed_tasks),
            "completion_rate": len(completed_tasks) / total_tasks if total_tasks > 0 else 0.0,
            "average_duration_accuracy": avg_accuracy
        }
    
    def _get_optimization_summary(self) -> Dict[str, Any]:
        """Summarize quantum optimization metrics."""
        if not self.quantum_state_metrics:
            return {}
        
        recent_optimizations = [
            record for record in self.quantum_state_metrics
            if (datetime.now() - record["timestamp"]).hours < 24
        ]
        
        if recent_optimizations:
            avg_duration = sum(record["duration"] for record in recent_optimizations) / len(recent_optimizations)
            success_rate = sum(1 for record in recent_optimizations if record["success"]) / len(recent_optimizations)
        else:
            avg_duration = 0.0
            success_rate = 0.0
        
        return {
            "total_optimizations": len(self.quantum_state_metrics),
            "recent_optimizations_24h": len(recent_optimizations),
            "average_optimization_duration": avg_duration,
            "optimization_success_rate": success_rate
        }
    
    def _get_resource_summary(self) -> Dict[str, Any]:
        """Summarize resource utilization metrics."""
        if not self.resource_usage_history:
            return {}
        
        latest_utilization = self.resource_usage_history[-1]["utilization"]
        
        # Calculate average utilization over time
        total_utilization = defaultdict(list)
        for record in self.resource_usage_history:
            for resource_id, data in record["utilization"].items():
                total_utilization[resource_id].append(data["utilization_percent"])
        
        avg_utilization = {}
        for resource_id, utilizations in total_utilization.items():
            avg_utilization[resource_id] = sum(utilizations) / len(utilizations)
        
        return {
            "latest_utilization": latest_utilization,
            "average_utilization": avg_utilization,
            "history_points": len(self.resource_usage_history)
        }
    
    def cleanup_old_metrics(self) -> None:
        """Remove metrics older than retention period."""
        cutoff_time = datetime.now() - self.metrics_retention
        
        with self._lock:
            # Clean quantum state metrics
            self.quantum_state_metrics = [
                record for record in self.quantum_state_metrics
                if record["timestamp"] > cutoff_time
            ]
            
            # Clean resource usage history
            while (self.resource_usage_history and 
                   self.resource_usage_history[0]["timestamp"] < cutoff_time):
                self.resource_usage_history.popleft()


class HealthChecker:
    """Health monitoring for quantum task planner system."""
    
    def __init__(self, scheduler: Optional[Any] = None, 
                 resource_allocator: Optional[Any] = None):
        self.scheduler = scheduler
        self.resource_allocator = resource_allocator
        self.start_time = datetime.now()
        
    def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        health_status = {
            "healthy": True,
            "timestamp": datetime.now(),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "components": {}
        }
        
        # Check scheduler health
        scheduler_health = self._check_scheduler_health()
        health_status["components"]["scheduler"] = scheduler_health
        if not scheduler_health["healthy"]:
            health_status["healthy"] = False
        
        # Check resource allocator health
        if self.resource_allocator:
            allocator_health = self._check_allocator_health()
            health_status["components"]["resource_allocator"] = allocator_health
            if not allocator_health["healthy"]:
                health_status["healthy"] = False
        
        # Check system resources
        system_health = self._check_system_resources()
        health_status["components"]["system"] = system_health
        if not system_health["healthy"]:
            health_status["healthy"] = False
        
        return health_status
    
    def _check_scheduler_health(self) -> Dict[str, Any]:
        """Check quantum scheduler health."""
        if not self.scheduler:
            return {"healthy": False, "error": "Scheduler not initialized"}
        
        try:
            # Basic functionality test
            test_stats = self.scheduler.get_task_statistics()
            
            # Check for concerning conditions
            issues = []
            if test_stats.get("running", 0) > 1000:
                issues.append("Too many running tasks")
            
            if test_stats.get("blocked", 0) > test_stats.get("pending", 0):
                issues.append("More blocked than pending tasks")
            
            return {
                "healthy": len(issues) == 0,
                "task_statistics": test_stats,
                "issues": issues
            }
            
        except Exception as e:
            logger.error(f"Scheduler health check failed: {e}")
            return {"healthy": False, "error": str(e)}
    
    def _check_allocator_health(self) -> Dict[str, Any]:
        """Check resource allocator health."""
        try:
            utilization = self.resource_allocator.get_resource_utilization()
            
            # Check for resource exhaustion
            issues = []
            for resource_id, data in utilization.items():
                if data["utilization_percent"] > 95:
                    issues.append(f"Resource {resource_id} critically low: {data['utilization_percent']:.1f}%")
                elif data["utilization_percent"] > 85:
                    issues.append(f"Resource {resource_id} high utilization: {data['utilization_percent']:.1f}%")
            
            return {
                "healthy": len(issues) == 0,
                "resource_count": len(utilization),
                "issues": issues
            }
            
        except Exception as e:
            logger.error(f"Resource allocator health check failed: {e}")
            return {"healthy": False, "error": str(e)}
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system-level resource health."""
        try:
            import psutil
            
            # Check memory usage
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            disk = psutil.disk_usage('/')
            
            issues = []
            if memory.percent > 90:
                issues.append(f"High memory usage: {memory.percent:.1f}%")
            if cpu_percent > 90:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            if disk.percent > 90:
                issues.append(f"High disk usage: {disk.percent:.1f}%")
            
            return {
                "healthy": len(issues) == 0,
                "memory_percent": memory.percent,
                "cpu_percent": cpu_percent,
                "disk_percent": disk.percent,
                "issues": issues
            }
            
        except ImportError:
            # psutil not available
            return {
                "healthy": True,
                "note": "System resource monitoring requires psutil package"
            }
        except Exception as e:
            logger.error(f"System health check failed: {e}")
            return {"healthy": False, "error": str(e)}


class AlertManager:
    """Manages alerts and notifications for quantum task planner."""
    
    def __init__(self):
        self.alert_rules: Dict[str, Dict] = {}
        self.active_alerts: Dict[str, Dict] = {}
        self.alert_history: List[Dict] = []
        self._setup_default_rules()
    
    def _setup_default_rules(self) -> None:
        """Setup default alerting rules."""
        self.alert_rules = {
            "high_task_failure_rate": {
                "condition": lambda metrics: metrics.get("error_rate", 0) > 10,
                "severity": "warning",
                "message": "High task failure rate detected"
            },
            "resource_exhaustion": {
                "condition": lambda util: any(
                    data.get("utilization_percent", 0) > 95 
                    for data in util.values()
                ),
                "severity": "critical",
                "message": "Critical resource exhaustion"
            },
            "optimization_failures": {
                "condition": lambda opt_metrics: opt_metrics.get("optimization_success_rate", 1.0) < 0.8,
                "severity": "warning",
                "message": "Schedule optimization frequently failing"
            },
            "long_pending_tasks": {
                "condition": lambda task_stats: task_stats.get("pending", 0) > 100,
                "severity": "info",
                "message": "Many tasks pending execution"
            }
        }
    
    def check_alerts(self, system_metrics: Dict[str, Any]) -> List[Dict]:
        """Check all alert rules against current metrics."""
        triggered_alerts = []
        current_time = datetime.now()
        
        for rule_name, rule in self.alert_rules.items():
            try:
                # Extract relevant metrics for this rule
                if rule_name == "high_task_failure_rate":
                    relevant_metrics = system_metrics.get("operation_metrics", {}).get("task_completion", {})
                elif rule_name == "resource_exhaustion":
                    relevant_metrics = system_metrics.get("resource_utilization_summary", {}).get("latest_utilization", {})
                elif rule_name == "optimization_failures":
                    relevant_metrics = system_metrics.get("quantum_optimization_summary", {})
                elif rule_name == "long_pending_tasks":
                    relevant_metrics = system_metrics.get("task_lifecycle_summary", {})
                else:
                    relevant_metrics = {}
                
                # Check condition
                if rule["condition"](relevant_metrics):
                    alert = {
                        "rule_name": rule_name,
                        "severity": rule["severity"],
                        "message": rule["message"],
                        "timestamp": current_time,
                        "metrics_snapshot": relevant_metrics
                    }
                    
                    triggered_alerts.append(alert)
                    
                    # Add to active alerts if not already present
                    if rule_name not in self.active_alerts:
                        self.active_alerts[rule_name] = alert
                        self.alert_history.append(alert)
                        logger.warning(f"Alert triggered: {rule_name} - {rule['message']}")
                
                else:
                    # Clear alert if condition no longer met
                    if rule_name in self.active_alerts:
                        del self.active_alerts[rule_name]
                        logger.info(f"Alert cleared: {rule_name}")
            
            except Exception as e:
                logger.error(f"Error checking alert rule {rule_name}: {e}")
        
        return triggered_alerts
    
    def get_active_alerts(self) -> Dict[str, Dict]:
        """Get currently active alerts."""
        return self.active_alerts.copy()
    
    def get_alert_history(self, hours: int = 24) -> List[Dict]:
        """Get alert history for specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history
            if alert["timestamp"] > cutoff_time
        ]
    
    def add_custom_rule(self, rule_name: str, condition_func: callable,
                       severity: str = "info", message: str = "") -> None:
        """Add a custom alert rule."""
        self.alert_rules[rule_name] = {
            "condition": condition_func,
            "severity": severity,
            "message": message or f"Custom rule {rule_name} triggered"
        }


class QuantumMonitor:
    """Comprehensive monitoring system for quantum task planner."""
    
    def __init__(self, scheduler=None, resource_allocator=None):
        self.metrics_collector = QuantumMetricsCollector()
        self.health_checker = HealthChecker(scheduler, resource_allocator)
        self.alert_manager = AlertManager()
        self.monitoring_active = True
        self._monitoring_thread = None
        
    def start_monitoring(self, interval_seconds: int = 60) -> None:
        """Start background monitoring."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    self._collect_periodic_metrics()
                    time.sleep(interval_seconds)
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(interval_seconds)
        
        self._monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        logger.info("Quantum monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self.monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("Quantum monitoring stopped")
    
    def _collect_periodic_metrics(self) -> None:
        """Collect metrics on a periodic basis."""
        try:
            # Get current system metrics
            system_metrics = self.metrics_collector.get_performance_summary()
            
            # Check for alerts
            self.alert_manager.check_alerts(system_metrics)
            
            # Clean up old metrics
            self.metrics_collector.cleanup_old_metrics()
            
        except Exception as e:
            logger.error(f"Periodic metrics collection failed: {e}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            "health": self.health_checker.check_system_health(),
            "metrics": self.metrics_collector.get_performance_summary(),
            "active_alerts": self.alert_manager.get_active_alerts(),
            "recent_alerts": self.alert_manager.get_alert_history(hours=24)
        }