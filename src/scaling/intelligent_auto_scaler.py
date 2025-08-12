"""
Intelligent auto-scaling system for pneumonia detection service.
Provides dynamic resource allocation based on demand patterns and system metrics.
"""

import time
import threading
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import psutil
import numpy as np
from collections import deque, defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling direction options."""
    UP = "up"
    DOWN = "down"
    MAINTAIN = "maintain"


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    CPU_WORKERS = "cpu_workers"
    GPU_WORKERS = "gpu_workers"
    MEMORY_ALLOCATION = "memory_allocation"
    BATCH_SIZE = "batch_size"
    CACHE_SIZE = "cache_size"
    CONNECTION_POOL = "connection_pool"


class WorkloadPattern(Enum):
    """Detected workload patterns."""
    STEADY = "steady"
    BURSTY = "bursty"
    PERIODIC = "periodic"
    DECLINING = "declining"
    RAMPING_UP = "ramping_up"
    UNPREDICTABLE = "unpredictable"


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    queue_length: int = 0
    active_requests: int = 0
    response_time_p95: float = 0.0
    response_time_avg: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    prediction_latency: float = 0.0
    model_loading_time: float = 0.0
    disk_io_usage: float = 0.0
    network_io_usage: float = 0.0


@dataclass
class ScalingAction:
    """A scaling action to be executed."""
    resource_type: ResourceType
    direction: ScalingDirection
    current_value: Any
    target_value: Any
    confidence: float
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    executed: bool = False
    execution_time: Optional[datetime] = None
    success: bool = False
    error_message: Optional[str] = None


class PredictiveScaler:
    """Predictive scaling based on historical patterns and ML."""
    
    def __init__(self, lookback_window: int = 1440):  # 24 hours in minutes
        self.lookback_window = lookback_window
        self.metric_history: deque = deque(maxlen=lookback_window)
        self.pattern_detector = WorkloadPatternDetector()
        self.demand_predictor = DemandPredictor()
    
    def add_metrics(self, metrics: ScalingMetrics):
        """Add new metrics to history."""
        self.metric_history.append(metrics)
        self.pattern_detector.analyze_pattern(list(self.metric_history))
    
    def predict_future_demand(self, horizon_minutes: int = 30) -> Dict[str, float]:
        """Predict future resource demand."""
        if len(self.metric_history) < 10:
            return {'cpu': 50.0, 'memory': 50.0, 'requests': 10.0}
        
        return self.demand_predictor.predict(
            list(self.metric_history), 
            horizon_minutes
        )
    
    def get_scaling_recommendations(self) -> List[ScalingAction]:
        """Get scaling recommendations based on predictions."""
        if len(self.metric_history) < 5:
            return []
        
        current_metrics = self.metric_history[-1]
        predicted_demand = self.predict_future_demand()
        detected_pattern = self.pattern_detector.get_current_pattern()
        
        recommendations = []
        
        # CPU scaling recommendations
        if predicted_demand['cpu'] > 80:
            recommendations.append(ScalingAction(
                resource_type=ResourceType.CPU_WORKERS,
                direction=ScalingDirection.UP,
                current_value=self._get_current_cpu_workers(),
                target_value=self._calculate_optimal_cpu_workers(predicted_demand['cpu']),
                confidence=0.8,
                reason=f"Predicted CPU usage: {predicted_demand['cpu']:.1f}%"
            ))
        elif predicted_demand['cpu'] < 30 and detected_pattern != WorkloadPattern.BURSTY:
            recommendations.append(ScalingAction(
                resource_type=ResourceType.CPU_WORKERS,
                direction=ScalingDirection.DOWN,
                current_value=self._get_current_cpu_workers(),
                target_value=max(1, self._get_current_cpu_workers() - 1),
                confidence=0.6,
                reason=f"Predicted low CPU usage: {predicted_demand['cpu']:.1f}%"
            ))
        
        # Memory scaling recommendations
        if predicted_demand['memory'] > 85:
            recommendations.append(ScalingAction(
                resource_type=ResourceType.MEMORY_ALLOCATION,
                direction=ScalingDirection.UP,
                current_value=self._get_current_memory_allocation(),
                target_value=self._calculate_optimal_memory_allocation(predicted_demand['memory']),
                confidence=0.9,
                reason=f"Predicted memory usage: {predicted_demand['memory']:.1f}%"
            ))
        
        # Batch size optimization
        if current_metrics.response_time_p95 > 5000:  # 5 seconds
            recommendations.append(ScalingAction(
                resource_type=ResourceType.BATCH_SIZE,
                direction=ScalingDirection.DOWN,
                current_value=self._get_current_batch_size(),
                target_value=max(1, self._get_current_batch_size() // 2),
                confidence=0.7,
                reason=f"High response time: {current_metrics.response_time_p95:.0f}ms"
            ))
        elif current_metrics.cpu_usage < 50 and current_metrics.memory_usage < 60:
            recommendations.append(ScalingAction(
                resource_type=ResourceType.BATCH_SIZE,
                direction=ScalingDirection.UP,
                current_value=self._get_current_batch_size(),
                target_value=min(128, self._get_current_batch_size() * 2),
                confidence=0.6,
                reason="Low resource utilization, can increase batch size"
            ))
        
        return recommendations
    
    def _get_current_cpu_workers(self) -> int:
        """Get current number of CPU workers."""
        return getattr(self, '_cpu_workers', psutil.cpu_count())
    
    def _get_current_memory_allocation(self) -> float:
        """Get current memory allocation in GB."""
        return getattr(self, '_memory_allocation', 4.0)
    
    def _get_current_batch_size(self) -> int:
        """Get current batch size."""
        return getattr(self, '_batch_size', 32)
    
    def _calculate_optimal_cpu_workers(self, predicted_cpu: float) -> int:
        """Calculate optimal number of CPU workers."""
        current_workers = self._get_current_cpu_workers()
        max_workers = psutil.cpu_count() * 2
        
        if predicted_cpu > 90:
            return min(max_workers, current_workers + 2)
        elif predicted_cpu > 70:
            return min(max_workers, current_workers + 1)
        else:
            return current_workers
    
    def _calculate_optimal_memory_allocation(self, predicted_memory: float) -> float:
        """Calculate optimal memory allocation."""
        current_allocation = self._get_current_memory_allocation()
        total_memory = psutil.virtual_memory().total / (1024**3)  # GB
        
        if predicted_memory > 90:
            return min(total_memory * 0.8, current_allocation * 1.5)
        elif predicted_memory > 75:
            return min(total_memory * 0.7, current_allocation * 1.2)
        else:
            return current_allocation


class WorkloadPatternDetector:
    """Detects workload patterns for informed scaling decisions."""
    
    def __init__(self):
        self.current_pattern = WorkloadPattern.STEADY
        self.pattern_confidence = 0.5
        self.pattern_history: List[WorkloadPattern] = []
    
    def analyze_pattern(self, metrics_history: List[ScalingMetrics]) -> WorkloadPattern:
        """Analyze metrics to detect workload patterns."""
        if len(metrics_history) < 10:
            return WorkloadPattern.STEADY
        
        # Extract time series data
        timestamps = [m.timestamp for m in metrics_history[-60:]]  # Last hour
        cpu_values = [m.cpu_usage for m in metrics_history[-60:]]
        request_counts = [m.active_requests for m in metrics_history[-60:]]
        
        # Detect patterns
        pattern = self._detect_pattern_type(cpu_values, request_counts, timestamps)
        
        self.pattern_history.append(pattern)
        if len(self.pattern_history) > 100:
            self.pattern_history = self.pattern_history[-100:]
        
        # Update current pattern with smoothing
        self.current_pattern = self._smooth_pattern_detection()
        
        return self.current_pattern
    
    def _detect_pattern_type(self, cpu_values: List[float], 
                           request_counts: List[int], 
                           timestamps: List[datetime]) -> WorkloadPattern:
        """Detect the type of workload pattern."""
        if len(cpu_values) < 10:
            return WorkloadPattern.STEADY
        
        # Calculate statistics
        cpu_mean = np.mean(cpu_values)
        cpu_std = np.std(cpu_values)
        cpu_trend = self._calculate_trend(cpu_values)
        
        request_mean = np.mean(request_counts)
        request_std = np.std(request_counts)
        request_trend = self._calculate_trend(request_counts)
        
        # Pattern detection logic
        if cpu_std / max(cpu_mean, 1) > 0.5:  # High variability
            if self._detect_periodicity(cpu_values):
                return WorkloadPattern.PERIODIC
            else:
                return WorkloadPattern.BURSTY
        
        elif cpu_trend > 0.1:  # Increasing trend
            return WorkloadPattern.RAMPING_UP
        
        elif cpu_trend < -0.1:  # Decreasing trend
            return WorkloadPattern.DECLINING
        
        elif cpu_std / max(cpu_mean, 1) < 0.2:  # Low variability
            return WorkloadPattern.STEADY
        
        else:
            return WorkloadPattern.UNPREDICTABLE
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend using linear regression slope."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression
        slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
        return slope if not np.isnan(slope) else 0.0
    
    def _detect_periodicity(self, values: List[float]) -> bool:
        """Detect if there's periodicity in the values."""
        if len(values) < 20:
            return False
        
        # Simple autocorrelation check
        try:
            autocorr = np.correlate(values, values, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Look for peaks indicating periodicity
            for i in range(5, min(30, len(autocorr))):
                if autocorr[i] > 0.7 * autocorr[0]:  # Strong correlation
                    return True
            
            return False
        except Exception:
            return False
    
    def _smooth_pattern_detection(self) -> WorkloadPattern:
        """Smooth pattern detection using recent history."""
        if len(self.pattern_history) < 3:
            return self.current_pattern
        
        # Count occurrences of each pattern in recent history
        recent_patterns = self.pattern_history[-10:]
        pattern_counts = defaultdict(int)
        
        for pattern in recent_patterns:
            pattern_counts[pattern] += 1
        
        # Return most common pattern
        return max(pattern_counts.items(), key=lambda x: x[1])[0]
    
    def get_current_pattern(self) -> WorkloadPattern:
        """Get current detected pattern."""
        return self.current_pattern


class DemandPredictor:
    """Predicts future resource demand using simple time series analysis."""
    
    def predict(self, metrics_history: List[ScalingMetrics], 
               horizon_minutes: int) -> Dict[str, float]:
        """Predict future demand for various resources."""
        if len(metrics_history) < 5:
            return {'cpu': 50.0, 'memory': 50.0, 'requests': 10.0}
        
        # Extract recent trends
        recent_metrics = metrics_history[-min(30, len(metrics_history)):]
        
        cpu_values = [m.cpu_usage for m in recent_metrics]
        memory_values = [m.memory_usage for m in recent_metrics]
        request_values = [m.active_requests for m in recent_metrics]
        
        # Simple prediction using moving average + trend
        predicted_cpu = self._predict_value(cpu_values, horizon_minutes)
        predicted_memory = self._predict_value(memory_values, horizon_minutes)
        predicted_requests = self._predict_value(request_values, horizon_minutes)
        
        return {
            'cpu': max(0, min(100, predicted_cpu)),
            'memory': max(0, min(100, predicted_memory)),
            'requests': max(0, predicted_requests)
        }
    
    def _predict_value(self, values: List[float], horizon_minutes: int) -> float:
        """Predict future value using exponential smoothing."""
        if len(values) < 2:
            return values[0] if values else 50.0
        
        # Exponential smoothing
        alpha = 0.3  # Smoothing factor
        beta = 0.1   # Trend factor
        
        # Initialize
        s = values[0]  # Smoothed value
        b = values[1] - values[0]  # Initial trend
        
        # Apply exponential smoothing
        for i in range(1, len(values)):
            prev_s = s
            s = alpha * values[i] + (1 - alpha) * (s + b)
            b = beta * (s - prev_s) + (1 - beta) * b
        
        # Project into future
        prediction = s + b * (horizon_minutes / 5)  # Assuming 5-minute intervals
        
        return prediction


class ResourceController:
    """Controls actual resource allocation and scaling operations."""
    
    def __init__(self):
        self.current_resources = {
            ResourceType.CPU_WORKERS: psutil.cpu_count(),
            ResourceType.GPU_WORKERS: 0,
            ResourceType.MEMORY_ALLOCATION: 4.0,  # GB
            ResourceType.BATCH_SIZE: 32,
            ResourceType.CACHE_SIZE: 1000,  # Number of cached items
            ResourceType.CONNECTION_POOL: 10
        }
        
        self.resource_locks = {
            resource_type: threading.Lock() 
            for resource_type in ResourceType
        }
        
        self.scaling_history: List[ScalingAction] = []
    
    def execute_scaling_action(self, action: ScalingAction) -> bool:
        """Execute a scaling action."""
        logger.info(
            f"Executing scaling action: {action.resource_type.value} "
            f"{action.direction.value} from {action.current_value} to {action.target_value}"
        )
        
        with self.resource_locks[action.resource_type]:
            try:
                success = self._perform_scaling(action)
                
                action.executed = True
                action.execution_time = datetime.utcnow()
                action.success = success
                
                if success:
                    self.current_resources[action.resource_type] = action.target_value
                    logger.info(f"Successfully scaled {action.resource_type.value}")
                else:
                    logger.warning(f"Failed to scale {action.resource_type.value}")
                
                self.scaling_history.append(action)
                if len(self.scaling_history) > 1000:
                    self.scaling_history = self.scaling_history[-1000:]
                
                return success
                
            except Exception as e:
                action.executed = True
                action.execution_time = datetime.utcnow()
                action.success = False
                action.error_message = str(e)
                
                logger.error(f"Error executing scaling action: {e}")
                return False
    
    def _perform_scaling(self, action: ScalingAction) -> bool:
        """Perform the actual scaling operation."""
        resource_type = action.resource_type
        target_value = action.target_value
        
        if resource_type == ResourceType.CPU_WORKERS:
            return self._scale_cpu_workers(target_value)
        
        elif resource_type == ResourceType.GPU_WORKERS:
            return self._scale_gpu_workers(target_value)
        
        elif resource_type == ResourceType.MEMORY_ALLOCATION:
            return self._scale_memory_allocation(target_value)
        
        elif resource_type == ResourceType.BATCH_SIZE:
            return self._scale_batch_size(target_value)
        
        elif resource_type == ResourceType.CACHE_SIZE:
            return self._scale_cache_size(target_value)
        
        elif resource_type == ResourceType.CONNECTION_POOL:
            return self._scale_connection_pool(target_value)
        
        else:
            logger.warning(f"Unknown resource type: {resource_type}")
            return False
    
    def _scale_cpu_workers(self, target_count: int) -> bool:
        """Scale CPU worker processes."""
        # In a real implementation, this would manage worker processes
        # For now, we'll just update the configuration
        max_workers = psutil.cpu_count() * 2
        if 1 <= target_count <= max_workers:
            logger.info(f"Would scale CPU workers to {target_count}")
            return True
        return False
    
    def _scale_gpu_workers(self, target_count: int) -> bool:
        """Scale GPU worker processes."""
        # Check GPU availability
        try:
            import tensorflow as tf
            available_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
            if 0 <= target_count <= available_gpus:
                logger.info(f"Would scale GPU workers to {target_count}")
                return True
        except ImportError:
            logger.warning("TensorFlow not available for GPU scaling")
        return False
    
    def _scale_memory_allocation(self, target_gb: float) -> bool:
        """Scale memory allocation."""
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        max_allocation = total_memory_gb * 0.8  # Use max 80% of total memory
        
        if 0.5 <= target_gb <= max_allocation:
            logger.info(f"Would scale memory allocation to {target_gb:.1f}GB")
            return True
        return False
    
    def _scale_batch_size(self, target_size: int) -> bool:
        """Scale batch size for model inference."""
        if 1 <= target_size <= 256:
            logger.info(f"Would scale batch size to {target_size}")
            return True
        return False
    
    def _scale_cache_size(self, target_size: int) -> bool:
        """Scale cache size."""
        if 100 <= target_size <= 10000:
            logger.info(f"Would scale cache size to {target_size}")
            return True
        return False
    
    def _scale_connection_pool(self, target_size: int) -> bool:
        """Scale database connection pool size."""
        if 1 <= target_size <= 100:
            logger.info(f"Would scale connection pool to {target_size}")
            return True
        return False
    
    def get_current_resources(self) -> Dict[ResourceType, Any]:
        """Get current resource allocations."""
        return self.current_resources.copy()
    
    def get_scaling_history(self) -> List[ScalingAction]:
        """Get scaling action history."""
        return self.scaling_history.copy()


class IntelligentAutoScaler:
    """Main auto-scaler orchestrating all components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Components
        self.metrics_collector = MetricsCollector()
        self.predictive_scaler = PredictiveScaler()
        self.resource_controller = ResourceController()
        
        # Configuration
        self.scaling_interval = self.config.get('scaling_interval', 60)  # seconds
        self.min_scaling_interval = self.config.get('min_scaling_interval', 30)
        self.max_scaling_actions_per_interval = self.config.get('max_scaling_actions', 3)
        
        # State
        self.is_running = False
        self.last_scaling_time = datetime.utcnow()
        self.scaling_thread = None
        self.metrics_thread = None
        
        # Metrics
        self.scaling_statistics = {
            'total_actions': 0,
            'successful_actions': 0,
            'failed_actions': 0,
            'scale_up_actions': 0,
            'scale_down_actions': 0
        }
    
    def start(self):
        """Start the auto-scaling system."""
        if self.is_running:
            logger.warning("Auto-scaler is already running")
            return
        
        self.is_running = True
        
        # Start metrics collection
        self.metrics_thread = threading.Thread(target=self._metrics_collection_loop, daemon=True)
        self.metrics_thread.start()
        
        # Start scaling loop
        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scaling_thread.start()
        
        logger.info("Intelligent auto-scaler started")
    
    def stop(self):
        """Stop the auto-scaling system."""
        self.is_running = False
        
        if self.metrics_thread:
            self.metrics_thread.join(timeout=5)
        
        if self.scaling_thread:
            self.scaling_thread.join(timeout=5)
        
        logger.info("Intelligent auto-scaler stopped")
    
    def _metrics_collection_loop(self):
        """Continuous metrics collection loop."""
        while self.is_running:
            try:
                metrics = self.metrics_collector.collect_metrics()
                self.predictive_scaler.add_metrics(metrics)
                time.sleep(30)  # Collect metrics every 30 seconds
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(10)
    
    def _scaling_loop(self):
        """Main scaling decision and execution loop."""
        while self.is_running:
            try:
                # Check if enough time has passed since last scaling
                time_since_last_scaling = (datetime.utcnow() - self.last_scaling_time).total_seconds()
                
                if time_since_last_scaling >= self.min_scaling_interval:
                    self._perform_scaling_cycle()
                    self.last_scaling_time = datetime.utcnow()
                
                time.sleep(self.scaling_interval)
                
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                time.sleep(10)
    
    def _perform_scaling_cycle(self):
        """Perform one cycle of scaling decisions and actions."""
        # Get scaling recommendations
        recommendations = self.predictive_scaler.get_scaling_recommendations()
        
        if not recommendations:
            logger.debug("No scaling recommendations")
            return
        
        # Sort recommendations by confidence (highest first)
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        
        # Execute top recommendations (limited by max actions per interval)
        actions_executed = 0
        
        for recommendation in recommendations[:self.max_scaling_actions_per_interval]:
            if recommendation.confidence > 0.5:  # Only execute high-confidence actions
                success = self.resource_controller.execute_scaling_action(recommendation)
                
                # Update statistics
                self.scaling_statistics['total_actions'] += 1
                if success:
                    self.scaling_statistics['successful_actions'] += 1
                    if recommendation.direction == ScalingDirection.UP:
                        self.scaling_statistics['scale_up_actions'] += 1
                    elif recommendation.direction == ScalingDirection.DOWN:
                        self.scaling_statistics['scale_down_actions'] += 1
                else:
                    self.scaling_statistics['failed_actions'] += 1
                
                actions_executed += 1
            else:
                logger.debug(f"Skipping low-confidence recommendation: {recommendation.confidence:.2f}")
        
        if actions_executed > 0:
            logger.info(f"Executed {actions_executed} scaling actions")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current auto-scaler status."""
        current_resources = self.resource_controller.get_current_resources()
        recent_metrics = list(self.predictive_scaler.metric_history)[-5:] if self.predictive_scaler.metric_history else []
        
        return {
            'is_running': self.is_running,
            'current_resources': {
                resource_type.value: value 
                for resource_type, value in current_resources.items()
            },
            'detected_pattern': self.predictive_scaler.pattern_detector.get_current_pattern().value,
            'scaling_statistics': self.scaling_statistics.copy(),
            'recent_metrics_count': len(recent_metrics),
            'last_scaling_time': self.last_scaling_time.isoformat(),
            'next_scaling_check': (self.last_scaling_time + timedelta(seconds=self.scaling_interval)).isoformat()
        }
    
    def force_scaling_check(self) -> List[ScalingAction]:
        """Force an immediate scaling check and return recommendations."""
        recommendations = self.predictive_scaler.get_scaling_recommendations()
        logger.info(f"Force scaling check generated {len(recommendations)} recommendations")
        return recommendations
    
    def set_resource_limits(self, resource_type: ResourceType, min_value: Any, max_value: Any):
        """Set limits for a specific resource type."""
        # This would be used to configure resource limits
        logger.info(f"Setting limits for {resource_type.value}: min={min_value}, max={max_value}")
    
    def export_scaling_report(self, file_path: str):
        """Export comprehensive scaling report."""
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'autoscaler_status': self.get_current_status(),
            'scaling_history': [
                {
                    'timestamp': action.timestamp.isoformat(),
                    'resource_type': action.resource_type.value,
                    'direction': action.direction.value,
                    'current_value': action.current_value,
                    'target_value': action.target_value,
                    'confidence': action.confidence,
                    'reason': action.reason,
                    'success': action.success
                }
                for action in self.resource_controller.get_scaling_history()[-100:]  # Last 100 actions
            ],
            'metrics_history': [
                {
                    'timestamp': metrics.timestamp.isoformat(),
                    'cpu_usage': metrics.cpu_usage,
                    'memory_usage': metrics.memory_usage,
                    'active_requests': metrics.active_requests,
                    'response_time_p95': metrics.response_time_p95,
                    'throughput': metrics.throughput
                }
                for metrics in list(self.predictive_scaler.metric_history)[-100:]  # Last 100 metrics
            ]
        }
        
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Scaling report exported to {file_path}")


class MetricsCollector:
    """Collects system and application metrics for scaling decisions."""
    
    def __init__(self):
        self.request_queue = queue.Queue()
        self.response_times = deque(maxlen=1000)
        self.active_requests = 0
        self.total_requests = 0
        self.failed_requests = 0
    
    def collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Disk and network I/O
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()
        
        # Application metrics
        queue_length = self.request_queue.qsize()
        active_requests = self.active_requests
        
        # Response time metrics
        if self.response_times:
            response_times_array = np.array(list(self.response_times))
            response_time_avg = np.mean(response_times_array)
            response_time_p95 = np.percentile(response_times_array, 95)
        else:
            response_time_avg = 0.0
            response_time_p95 = 0.0
        
        # Error rate
        error_rate = (self.failed_requests / max(1, self.total_requests)) * 100
        
        # Throughput (requests per second)
        throughput = self._calculate_throughput()
        
        # GPU metrics (if available)
        gpu_usage = self._get_gpu_usage()
        
        return ScalingMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            queue_length=queue_length,
            active_requests=active_requests,
            response_time_p95=response_time_p95,
            response_time_avg=response_time_avg,
            error_rate=error_rate,
            throughput=throughput,
            disk_io_usage=0.0,  # Would calculate from disk_io
            network_io_usage=0.0  # Would calculate from network_io
        )
    
    def record_request_start(self):
        """Record the start of a request."""
        self.active_requests += 1
        self.total_requests += 1
    
    def record_request_end(self, response_time_ms: float, success: bool = True):
        """Record the completion of a request."""
        self.active_requests = max(0, self.active_requests - 1)
        self.response_times.append(response_time_ms)
        
        if not success:
            self.failed_requests += 1
    
    def _calculate_throughput(self) -> float:
        """Calculate current throughput in requests per second."""
        # Simple implementation - would be more sophisticated in practice
        if len(self.response_times) < 10:
            return 0.0
        
        # Estimate based on recent response times
        recent_response_times = list(self.response_times)[-60:]  # Last 60 requests
        if recent_response_times:
            avg_response_time_seconds = np.mean(recent_response_times) / 1000.0
            return 1.0 / max(avg_response_time_seconds, 0.001)
        
        return 0.0
    
    def _get_gpu_usage(self) -> float:
        """Get GPU usage if available."""
        try:
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                # This is a simplified implementation
                # Real GPU monitoring would use nvidia-ml-py or similar
                return 0.0  # Placeholder
        except ImportError:
            pass
        
        return 0.0


def create_intelligent_autoscaler(config: Optional[Dict[str, Any]] = None) -> IntelligentAutoScaler:
    """Factory function to create a configured auto-scaler."""
    return IntelligentAutoScaler(config)


if __name__ == '__main__':
    # Example usage
    autoscaler = create_intelligent_autoscaler({
        'scaling_interval': 60,
        'min_scaling_interval': 30,
        'max_scaling_actions': 2
    })
    
    # Start auto-scaling
    autoscaler.start()
    
    try:
        # Run for a short time as demonstration
        time.sleep(120)
        
        # Get status
        status = autoscaler.get_current_status()
        print(f"Auto-scaler status: {status}")
        
        # Export report
        autoscaler.export_scaling_report('/tmp/scaling_report.json')
        
    finally:
        autoscaler.stop()
