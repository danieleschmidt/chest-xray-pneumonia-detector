"""Comprehensive health monitoring and observability for quantum task scheduler."""

import logging
import time
import threading
import psutil
import os
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class HealthMetric:
    """Individual health metric with metadata."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    status: str  # 'healthy', 'warning', 'critical'
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: str  # 'healthy', 'degraded', 'critical'
    score: float  # 0-100
    metrics: Dict[str, HealthMetric] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


class QuantumSchedulerHealthMonitor:
    """Comprehensive health monitoring for quantum task scheduler."""
    
    def __init__(self, scheduler, check_interval: int = 30):
        self.scheduler = scheduler
        self.check_interval = check_interval
        self.is_running = False
        self.monitor_thread = None
        
        # Health metrics history
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.alert_history: List[Dict] = []
        
        # Health thresholds
        self.thresholds = {
            'cpu_usage': {'warning': 70.0, 'critical': 90.0},
            'memory_usage': {'warning': 80.0, 'critical': 95.0},
            'task_completion_rate': {'warning': 0.7, 'critical': 0.5},
            'avg_task_duration': {'warning': 300.0, 'critical': 600.0},  # seconds
            'error_rate': {'warning': 0.05, 'critical': 0.10},
            'queue_length': {'warning': 50, 'critical': 100}
        }
        
        # Performance baselines
        self.performance_baselines = {
            'task_throughput': 10.0,  # tasks per minute
            'resource_efficiency': 0.8,  # utilization efficiency
            'response_time': 1.0  # seconds
        }
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[str, str], None]] = []
        
    def start_monitoring(self) -> None:
        """Start health monitoring in background thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Health monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_running:
            try:
                self._collect_metrics()
                self._evaluate_health()
                self._check_alerts()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(self.check_interval)
    
    def _collect_metrics(self) -> None:
        """Collect comprehensive health metrics."""
        timestamp = datetime.now()
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        self._record_metric('cpu_usage', cpu_percent, '%', timestamp)
        self._record_metric('memory_usage', memory.percent, '%', timestamp)
        self._record_metric('available_memory', memory.available / (1024**3), 'GB', timestamp)
        
        # Scheduler-specific metrics
        if hasattr(self.scheduler, 'tasks'):
            total_tasks = len(self.scheduler.tasks)
            completed_tasks = len([t for t in self.scheduler.tasks.values() 
                                 if hasattr(t, 'status') and str(t.status).endswith('COMPLETED')])
            running_tasks = len(getattr(self.scheduler, 'running_tasks', []))
            pending_tasks = total_tasks - completed_tasks - running_tasks
            
            self._record_metric('total_tasks', total_tasks, 'count', timestamp)
            self._record_metric('completed_tasks', completed_tasks, 'count', timestamp)
            self._record_metric('running_tasks', running_tasks, 'count', timestamp)
            self._record_metric('pending_tasks', pending_tasks, 'count', timestamp)
            
            # Task completion rate
            completion_rate = completed_tasks / max(1, total_tasks)
            self._record_metric('task_completion_rate', completion_rate, 'ratio', timestamp)
            
            # Average task duration
            if completed_tasks > 0:
                total_duration = 0
                duration_count = 0
                
                for task in self.scheduler.tasks.values():
                    if (hasattr(task, 'started_at') and hasattr(task, 'completed_at') and
                        task.started_at and task.completed_at):
                        duration = (task.completed_at - task.started_at).total_seconds()
                        total_duration += duration
                        duration_count += 1
                
                if duration_count > 0:
                    avg_duration = total_duration / duration_count
                    self._record_metric('avg_task_duration', avg_duration, 'seconds', timestamp)
        
        # Error tracking
        if hasattr(self.scheduler, 'error_count'):
            error_rate = getattr(self.scheduler, 'error_count', 0) / max(1, getattr(self.scheduler, 'total_operations', 1))
            self._record_metric('error_rate', error_rate, 'ratio', timestamp)
        
        # Queue metrics
        queue_length = len(getattr(self.scheduler, 'running_tasks', [])) + len([
            t for t in getattr(self.scheduler, 'tasks', {}).values()
            if hasattr(t, 'status') and str(t.status).endswith('PENDING')
        ])
        self._record_metric('queue_length', queue_length, 'count', timestamp)
        
        # Performance metrics
        self._calculate_performance_metrics(timestamp)
    
    def _record_metric(self, name: str, value: float, unit: str, timestamp: datetime) -> None:
        """Record a single metric value."""
        # Determine status based on thresholds
        status = 'healthy'
        if name in self.thresholds:
            thresholds = self.thresholds[name]
            if value >= thresholds.get('critical', float('inf')):
                status = 'critical'
            elif value >= thresholds.get('warning', float('inf')):
                status = 'warning'
        
        # For inverse metrics (higher is worse)
        if name in ['error_rate'] and name in self.thresholds:
            thresholds = self.thresholds[name]
            if value >= thresholds.get('critical', 1.0):
                status = 'critical'
            elif value >= thresholds.get('warning', 1.0):
                status = 'warning'
        
        metric = HealthMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=timestamp,
            status=status,
            threshold_warning=self.thresholds.get(name, {}).get('warning'),
            threshold_critical=self.thresholds.get(name, {}).get('critical')
        )
        
        self.metrics_history[name].append(metric)
    
    def _calculate_performance_metrics(self, timestamp: datetime) -> None:
        """Calculate derived performance metrics."""
        # Task throughput (tasks completed per minute)
        if len(self.metrics_history['completed_tasks']) >= 2:
            recent_metrics = list(self.metrics_history['completed_tasks'])[-2:]
            time_diff = (recent_metrics[-1].timestamp - recent_metrics[0].timestamp).total_seconds() / 60.0
            
            if time_diff > 0:
                task_diff = recent_metrics[-1].value - recent_metrics[0].value
                throughput = task_diff / time_diff
                self._record_metric('task_throughput', throughput, 'tasks/min', timestamp)
        
        # Resource efficiency
        if ('cpu_usage' in self.metrics_history and 'memory_usage' in self.metrics_history and
            'task_completion_rate' in self.metrics_history):
            
            cpu_efficiency = (100 - self.metrics_history['cpu_usage'][-1].value) / 100
            memory_efficiency = (100 - self.metrics_history['memory_usage'][-1].value) / 100
            task_efficiency = self.metrics_history['task_completion_rate'][-1].value
            
            overall_efficiency = (cpu_efficiency + memory_efficiency + task_efficiency) / 3
            self._record_metric('resource_efficiency', overall_efficiency, 'ratio', timestamp)
    
    def _evaluate_health(self) -> SystemHealth:
        """Evaluate overall system health."""
        if not self.metrics_history:
            return SystemHealth(status='unknown', score=0.0)
        
        # Calculate health score
        total_score = 0.0
        metric_count = 0
        alerts = []
        
        current_metrics = {}
        
        for metric_name, history in self.metrics_history.items():
            if not history:
                continue
            
            latest_metric = history[-1]
            current_metrics[metric_name] = latest_metric
            
            # Calculate individual metric health score
            if latest_metric.status == 'critical':
                score = 0.0
                alerts.append(f"CRITICAL: {metric_name} = {latest_metric.value:.2f} {latest_metric.unit}")
            elif latest_metric.status == 'warning':
                score = 50.0
                alerts.append(f"WARNING: {metric_name} = {latest_metric.value:.2f} {latest_metric.unit}")
            else:
                score = 100.0
            
            total_score += score
            metric_count += 1
        
        overall_score = total_score / max(1, metric_count)
        
        # Determine overall status
        if overall_score >= 80:
            status = 'healthy'
        elif overall_score >= 60:
            status = 'degraded'
        else:
            status = 'critical'
        
        health = SystemHealth(
            status=status,
            score=overall_score,
            metrics=current_metrics,
            alerts=alerts,
            last_updated=datetime.now()
        )
        
        return health
    
    def _check_alerts(self) -> None:
        """Check for alert conditions and trigger notifications."""
        health = self._evaluate_health()
        
        # Check for new alerts
        current_time = datetime.now()
        for alert in health.alerts:
            # Avoid duplicate alerts within 5 minutes
            recent_alerts = [a for a in self.alert_history 
                           if (current_time - datetime.fromisoformat(a['timestamp'])).total_seconds() < 300]
            
            if not any(alert in a['message'] for a in recent_alerts):
                alert_record = {
                    'timestamp': current_time.isoformat(),
                    'level': 'critical' if 'CRITICAL' in alert else 'warning',
                    'message': alert,
                    'health_score': health.score
                }
                
                self.alert_history.append(alert_record)
                
                # Trigger alert callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert_record['level'], alert_record['message'])
                    except Exception as e:
                        logger.error(f"Alert callback failed: {e}")
    
    def get_current_health(self) -> SystemHealth:
        """Get current system health status."""
        return self._evaluate_health()
    
    def get_metric_history(self, metric_name: str, duration_minutes: int = 60) -> List[HealthMetric]:
        """Get metric history for specified duration."""
        if metric_name not in self.metrics_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        return [m for m in self.metrics_history[metric_name] if m.timestamp >= cutoff_time]
    
    def add_alert_callback(self, callback: Callable[[str, str], None]) -> None:
        """Add callback function for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def export_health_report(self) -> Dict:
        """Export comprehensive health report."""
        health = self.get_current_health()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_health': {
                'status': health.status,
                'score': health.score,
                'alerts_count': len(health.alerts)
            },
            'current_metrics': {},
            'trends': {},
            'recommendations': self._generate_recommendations(health)
        }
        
        # Current metrics
        for name, metric in health.metrics.items():
            report['current_metrics'][name] = {
                'value': metric.value,
                'unit': metric.unit,
                'status': metric.status,
                'timestamp': metric.timestamp.isoformat()
            }
        
        # Metric trends (last hour)
        for metric_name in self.metrics_history.keys():
            history = self.get_metric_history(metric_name, 60)
            if len(history) >= 2:
                # Calculate trend
                start_value = history[0].value
                end_value = history[-1].value
                trend_direction = 'increasing' if end_value > start_value else 'decreasing' if end_value < start_value else 'stable'
                trend_rate = abs(end_value - start_value) / max(1, len(history))
                
                report['trends'][metric_name] = {
                    'direction': trend_direction,
                    'rate': trend_rate,
                    'start_value': start_value,
                    'end_value': end_value,
                    'data_points': len(history)
                }
        
        return report
    
    def _generate_recommendations(self, health: SystemHealth) -> List[str]:
        """Generate actionable recommendations based on health status."""
        recommendations = []
        
        # CPU recommendations
        if 'cpu_usage' in health.metrics:
            cpu_metric = health.metrics['cpu_usage']
            if cpu_metric.status == 'critical':
                recommendations.append("Reduce parallel task limit or optimize task algorithms")
            elif cpu_metric.status == 'warning':
                recommendations.append("Monitor CPU usage and consider task scheduling optimization")
        
        # Memory recommendations
        if 'memory_usage' in health.metrics:
            memory_metric = health.metrics['memory_usage']
            if memory_metric.status == 'critical':
                recommendations.append("Implement task result cleanup and memory optimization")
            elif memory_metric.status == 'warning':
                recommendations.append("Review memory usage patterns and implement garbage collection")
        
        # Task performance recommendations
        if 'task_completion_rate' in health.metrics:
            completion_metric = health.metrics['task_completion_rate']
            if completion_metric.value < 0.7:
                recommendations.append("Review failed tasks and improve error handling")
        
        if 'avg_task_duration' in health.metrics:
            duration_metric = health.metrics['avg_task_duration']
            if duration_metric.value > 300:  # 5 minutes
                recommendations.append("Consider breaking down large tasks or optimizing task algorithms")
        
        # Queue management recommendations
        if 'queue_length' in health.metrics:
            queue_metric = health.metrics['queue_length']
            if queue_metric.value > 50:
                recommendations.append("Increase parallel processing capacity or optimize task prioritization")
        
        # Performance recommendations
        if 'task_throughput' in health.metrics:
            throughput_metric = health.metrics['task_throughput']
            if throughput_metric.value < self.performance_baselines['task_throughput']:
                recommendations.append("Optimize scheduling algorithms or increase parallel task limit")
        
        return recommendations
    
    def create_health_dashboard_data(self) -> Dict:
        """Create data structure for health monitoring dashboard."""
        health = self.get_current_health()
        
        dashboard_data = {
            'status': {
                'overall': health.status,
                'score': health.score,
                'last_updated': health.last_updated.isoformat()
            },
            'metrics': {},
            'alerts': health.alerts,
            'charts': {}
        }
        
        # Key metrics for dashboard
        key_metrics = ['cpu_usage', 'memory_usage', 'task_completion_rate', 'queue_length', 'task_throughput']
        
        for metric_name in key_metrics:
            if metric_name in health.metrics:
                metric = health.metrics[metric_name]
                dashboard_data['metrics'][metric_name] = {
                    'current': metric.value,
                    'unit': metric.unit,
                    'status': metric.status,
                    'warning_threshold': metric.threshold_warning,
                    'critical_threshold': metric.threshold_critical
                }
        
        # Time series data for charts
        for metric_name in key_metrics:
            history = self.get_metric_history(metric_name, 30)  # Last 30 minutes
            if history:
                dashboard_data['charts'][metric_name] = [
                    {
                        'timestamp': m.timestamp.isoformat(),
                        'value': m.value,
                        'status': m.status
                    }
                    for m in history
                ]
        
        return dashboard_data


def create_sample_alert_handler():
    """Create sample alert handler for demonstration."""
    def alert_handler(level: str, message: str):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"🚨 [{timestamp}] {level.upper()}: {message}")
        
        # In production, this could:
        # - Send emails/Slack notifications
        # - Write to logging system
        # - Trigger automated remediation
        # - Update external monitoring systems
    
    return alert_handler