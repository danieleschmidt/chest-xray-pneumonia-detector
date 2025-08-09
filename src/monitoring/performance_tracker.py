"""Performance monitoring and tracking for the pneumonia detection system."""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""
    name: str
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceSnapshot:
    """Snapshot of system performance at a point in time."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    active_requests: int
    avg_response_time_ms: float
    error_rate: float
    throughput_rps: float  # requests per second


class PerformanceTracker:
    """Tracks and analyzes system performance metrics."""
    
    def __init__(self, history_window_minutes: int = 60):
        """Initialize performance tracker.
        
        Args:
            history_window_minutes: How long to keep metrics history
        """
        self.history_window = timedelta(minutes=history_window_minutes)
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque())
        self._snapshots: deque = deque()
        self._active_requests = 0
        self._lock = threading.RLock()
        
        # Request tracking
        self._request_times: deque = deque()
        self._error_count: deque = deque()
        self._request_count: deque = deque()
    
    def record_metric(self, name: str, value: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a performance metric.
        
        Args:
            name: Metric name
            value: Metric value
            metadata: Additional metadata
        """
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        with self._lock:
            self._metrics[name].append(metric)
            self._cleanup_old_metrics()
    
    def start_request(self) -> str:
        """Mark the start of a request and return request ID.
        
        Returns:
            Request identifier for tracking
        """
        request_id = f"req_{int(time.time() * 1000)}_{threading.get_ident()}"
        
        with self._lock:
            self._active_requests += 1
            self._request_count.append((datetime.utcnow(), request_id))
        
        return request_id
    
    def end_request(self, request_id: str, error: bool = False) -> float:
        """Mark the end of a request.
        
        Args:
            request_id: Request identifier from start_request
            error: Whether the request resulted in an error
            
        Returns:
            Request duration in milliseconds
        """
        end_time = datetime.utcnow()
        
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
            
            # Find start time from request_id
            start_time = None
            request_timestamp = int(request_id.split('_')[1])
            start_time = datetime.utcfromtimestamp(request_timestamp / 1000)
            
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            self._request_times.append((end_time, duration_ms))
            
            if error:
                self._error_count.append((end_time, 1))
            else:
                self._error_count.append((end_time, 0))
            
            self._cleanup_old_request_data()
            
            return duration_ms
    
    def get_current_snapshot(self) -> PerformanceSnapshot:
        """Get current performance snapshot.
        
        Returns:
            Current performance metrics snapshot
        """
        now = datetime.utcnow()
        
        with self._lock:
            # Calculate metrics from recent data
            recent_cutoff = now - timedelta(minutes=5)
            
            # Response time metrics
            recent_times = [
                duration for timestamp, duration in self._request_times
                if timestamp >= recent_cutoff
            ]
            avg_response_time = statistics.mean(recent_times) if recent_times else 0.0
            
            # Error rate
            recent_errors = [
                error for timestamp, error in self._error_count
                if timestamp >= recent_cutoff
            ]
            error_rate = sum(recent_errors) / len(recent_errors) if recent_errors else 0.0
            
            # Throughput (requests per second)
            recent_requests = [
                1 for timestamp, _ in self._request_count
                if timestamp >= recent_cutoff
            ]
            time_window_seconds = 300  # 5 minutes
            throughput = len(recent_requests) / time_window_seconds if recent_requests else 0.0
            
            # System metrics (simplified - would use psutil in production)
            cpu_percent = 0.0  # Placeholder
            memory_percent = 0.0  # Placeholder
            
            snapshot = PerformanceSnapshot(
                timestamp=now,
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                active_requests=self._active_requests,
                avg_response_time_ms=avg_response_time,
                error_rate=error_rate,
                throughput_rps=throughput
            )
            
            self._snapshots.append(snapshot)
            self._cleanup_old_snapshots()
            
            return snapshot
    
    def get_metric_history(self, name: str, minutes: int = 60) -> List[PerformanceMetric]:
        """Get historical data for a specific metric.
        
        Args:
            name: Metric name
            minutes: How many minutes of history to return
            
        Returns:
            List of performance metrics
        """
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        
        with self._lock:
            if name not in self._metrics:
                return []
            
            return [
                metric for metric in self._metrics[name]
                if metric.timestamp >= cutoff
            ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary.
        
        Returns:
            Dictionary containing performance statistics
        """
        snapshot = self.get_current_snapshot()
        
        with self._lock:
            # Calculate percentiles for response times
            recent_times = [
                duration for timestamp, duration in self._request_times
                if timestamp >= datetime.utcnow() - timedelta(minutes=60)
            ]
            
            if recent_times:
                p50 = statistics.median(recent_times)
                p95 = statistics.quantiles(recent_times, n=20)[18]  # 95th percentile
                p99 = statistics.quantiles(recent_times, n=100)[98]  # 99th percentile
            else:
                p50 = p95 = p99 = 0.0
            
            return {
                'current': {
                    'active_requests': snapshot.active_requests,
                    'avg_response_time_ms': snapshot.avg_response_time_ms,
                    'error_rate': snapshot.error_rate,
                    'throughput_rps': snapshot.throughput_rps
                },
                'response_times': {
                    'p50_ms': p50,
                    'p95_ms': p95,
                    'p99_ms': p99
                },
                'totals': {
                    'total_requests': len(self._request_count),
                    'total_errors': sum(error for _, error in self._error_count)
                },
                'timestamp': snapshot.timestamp.isoformat()
            }
    
    def _cleanup_old_metrics(self) -> None:
        """Remove old metrics beyond the history window."""
        cutoff = datetime.utcnow() - self.history_window
        
        for name, metrics in self._metrics.items():
            while metrics and metrics[0].timestamp < cutoff:
                metrics.popleft()
    
    def _cleanup_old_request_data(self) -> None:
        """Remove old request tracking data."""
        cutoff = datetime.utcnow() - self.history_window
        
        while self._request_times and self._request_times[0][0] < cutoff:
            self._request_times.popleft()
        
        while self._error_count and self._error_count[0][0] < cutoff:
            self._error_count.popleft()
        
        while self._request_count and self._request_count[0][0] < cutoff:
            self._request_count.popleft()
    
    def _cleanup_old_snapshots(self) -> None:
        """Remove old performance snapshots."""
        cutoff = datetime.utcnow() - self.history_window
        
        while self._snapshots and self._snapshots[0].timestamp < cutoff:
            self._snapshots.popleft()


# Global performance tracker instance
performance_tracker = PerformanceTracker()


class PerformanceMonitor:
    """Context manager for monitoring request performance."""
    
    def __init__(self, operation_name: str = "request"):
        self.operation_name = operation_name
        self.request_id = None
        self.start_time = None
    
    def __enter__(self):
        self.request_id = performance_tracker.start_request()
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        error_occurred = exc_type is not None
        duration_ms = performance_tracker.end_request(self.request_id, error_occurred)
        
        # Record operation-specific metrics
        performance_tracker.record_metric(
            f"{self.operation_name}_duration_ms",
            duration_ms,
            {'error': error_occurred}
        )