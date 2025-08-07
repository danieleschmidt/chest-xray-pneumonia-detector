"""Simplified health monitoring without external dependencies."""

import logging
import time
import threading
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


@dataclass
class HealthMetric:
    """Individual health metric with metadata."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    status: str  # 'healthy', 'warning', 'critical'


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: str  # 'healthy', 'degraded', 'critical'
    score: float  # 0-100
    metrics: Dict[str, HealthMetric] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


class SimpleHealthMonitor:
    """Simplified health monitoring for quantum task scheduler."""
    
    def __init__(self, scheduler, check_interval: int = 30):
        self.scheduler = scheduler
        self.check_interval = check_interval
        self.is_running = False
        self.monitor_thread = None
        
        # Health metrics history
        self.metrics_history = defaultdict(lambda: deque(maxlen=100))
        
        # Health thresholds
        self.thresholds = {
            'task_completion_rate': {'warning': 0.7, 'critical': 0.5},
            'avg_task_duration': {'warning': 300.0, 'critical': 600.0},
            'error_rate': {'warning': 0.05, 'critical': 0.10},
            'queue_length': {'warning': 50, 'critical': 100}
        }
        
    def start_monitoring(self) -> None:
        """Start health monitoring in background thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Simple health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Simple health monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_running:
            try:
                self._collect_metrics()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(self.check_interval)
    
    def _collect_metrics(self) -> None:
        """Collect basic health metrics."""
        timestamp = datetime.now()
        
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
        
        # Queue metrics
        queue_length = len(getattr(self.scheduler, 'running_tasks', [])) + len([
            t for t in getattr(self.scheduler, 'tasks', {}).values()
            if hasattr(t, 'status') and str(t.status).endswith('PENDING')
        ])
        self._record_metric('queue_length', queue_length, 'count', timestamp)
    
    def _record_metric(self, name: str, value: float, unit: str, timestamp: datetime) -> None:
        """Record a single metric value."""
        # Determine status based on thresholds
        status = 'healthy'
        if name in self.thresholds:
            thresholds = self.thresholds[name]
            if name == 'error_rate':
                # Higher values are worse for error rate
                if value >= thresholds.get('critical', 1.0):
                    status = 'critical'
                elif value >= thresholds.get('warning', 1.0):
                    status = 'warning'
            else:
                # Standard threshold checking
                if value >= thresholds.get('critical', float('inf')):
                    status = 'critical'
                elif value >= thresholds.get('warning', float('inf')):
                    status = 'warning'
        
        metric = HealthMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=timestamp,
            status=status
        )
        
        self.metrics_history[name].append(metric)
    
    def get_current_health(self) -> SystemHealth:
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
        
        return SystemHealth(
            status=status,
            score=overall_score,
            metrics=current_metrics,
            alerts=alerts,
            last_updated=datetime.now()
        )
    
    def export_health_report(self) -> Dict:
        """Export simplified health report."""
        health = self.get_current_health()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_health': {
                'status': health.status,
                'score': health.score,
                'alerts_count': len(health.alerts)
            },
            'current_metrics': {
                name: {
                    'value': metric.value,
                    'unit': metric.unit,
                    'status': metric.status,
                    'timestamp': metric.timestamp.isoformat()
                }
                for name, metric in health.metrics.items()
            },
            'alerts': health.alerts
        }