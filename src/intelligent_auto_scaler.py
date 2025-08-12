"""
Intelligent Auto-Scaling System for Medical AI Workloads
Implements ML-driven auto-scaling with predictive analytics and resource optimization.
"""

import logging
import time
import threading
import queue
import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import psutil
import tensorflow as tf
from collections import deque
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib

logger = logging.getLogger(__name__)


class ScalingAction(Enum):
    """Types of scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down" 
    MAINTAIN = "maintain"
    OPTIMIZE = "optimize"


@dataclass
class ResourceMetrics:
    """System resource metrics snapshot."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    gpu_memory_percent: float
    disk_io_read: float
    disk_io_write: float
    network_sent: float
    network_recv: float
    active_requests: int
    queue_length: int
    response_time: float
    error_rate: float


@dataclass
class ScalingEvent:
    """Record of scaling event."""
    timestamp: float
    action: ScalingAction
    trigger_metrics: ResourceMetrics
    old_config: Dict[str, Any]
    new_config: Dict[str, Any]
    reason: str
    success: bool
    impact_score: float = 0.0


class MetricsCollector:
    """Collects and aggregates system metrics."""
    
    def __init__(self, collection_interval: float = 1.0, history_size: int = 3600):
        self.collection_interval = collection_interval
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.active_requests = 0
        self.queue_length = 0
        self.response_times = deque(maxlen=100)
        self.error_count = 0
        self.total_requests = 0
        self._collecting = False
        self._collection_thread = None
        
    def start_collection(self):
        """Start metrics collection."""
        if self._collecting:
            return
            
        self._collecting = True
        self._collection_thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._collection_thread.start()
        logger.info("Started metrics collection")
        
    def stop_collection(self):
        """Stop metrics collection."""
        self._collecting = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5)
        logger.info("Stopped metrics collection")
        
    def _collect_loop(self):
        """Main collection loop."""
        while self._collecting:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(self.collection_interval)
                
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # GPU metrics (if available)
        gpu_memory_percent = 0.0
        try:
            if tf.config.list_physical_devices('GPU'):
                gpu_details = tf.config.experimental.get_device_details(
                    tf.config.list_physical_devices('GPU')[0]
                )
                # This is a simplified GPU memory check
                gpu_memory_percent = 50.0  # Placeholder
        except:
            pass
            
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_read = disk_io.read_bytes if disk_io else 0
        disk_write = disk_io.write_bytes if disk_io else 0
        
        # Network I/O
        network_io = psutil.net_io_counters()
        network_sent = network_io.bytes_sent if network_io else 0
        network_recv = network_io.bytes_recv if network_io else 0
        
        # Application metrics
        avg_response_time = np.mean(list(self.response_times)) if self.response_times else 0.0
        error_rate = self.error_count / max(1, self.total_requests)
        
        return ResourceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            gpu_memory_percent=gpu_memory_percent,
            disk_io_read=disk_read,
            disk_io_write=disk_write,
            network_sent=network_sent,
            network_recv=network_recv,
            active_requests=self.active_requests,
            queue_length=self.queue_length,
            response_time=avg_response_time,
            error_rate=error_rate
        )
        
    def record_request(self, response_time: float, is_error: bool = False):
        """Record request metrics."""
        self.response_times.append(response_time)
        self.total_requests += 1
        if is_error:
            self.error_count += 1
            
    def get_recent_metrics(self, minutes: int = 5) -> List[ResourceMetrics]:
        """Get metrics from recent time period."""
        cutoff_time = time.time() - (minutes * 60)
        return [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
    def get_average_metrics(self, minutes: int = 5) -> Optional[ResourceMetrics]:
        """Get average metrics over time period."""
        recent_metrics = self.get_recent_metrics(minutes)
        if not recent_metrics:
            return None
            
        # Calculate averages
        avg_metrics = ResourceMetrics(
            timestamp=time.time(),
            cpu_percent=np.mean([m.cpu_percent for m in recent_metrics]),
            memory_percent=np.mean([m.memory_percent for m in recent_metrics]),
            gpu_memory_percent=np.mean([m.gpu_memory_percent for m in recent_metrics]),
            disk_io_read=np.mean([m.disk_io_read for m in recent_metrics]),
            disk_io_write=np.mean([m.disk_io_write for m in recent_metrics]),
            network_sent=np.mean([m.network_sent for m in recent_metrics]),
            network_recv=np.mean([m.network_recv for m in recent_metrics]),
            active_requests=int(np.mean([m.active_requests for m in recent_metrics])),
            queue_length=int(np.mean([m.queue_length for m in recent_metrics])),
            response_time=np.mean([m.response_time for m in recent_metrics]),
            error_rate=np.mean([m.error_rate for m in recent_metrics])
        )
        
        return avg_metrics


class PredictiveScaler:
    """ML-based predictive scaling using historical patterns."""
    
    def __init__(self, prediction_horizon: int = 300):  # 5 minutes
        self.prediction_horizon = prediction_horizon
        self.models = {}
        self.training_data = deque(maxlen=10000)
        self.last_training = 0
        self.training_interval = 3600  # Retrain every hour
        
    def add_training_data(self, metrics: ResourceMetrics, future_metrics: ResourceMetrics):
        """Add training data point."""
        features = self._extract_features(metrics)
        targets = self._extract_targets(future_metrics)
        self.training_data.append((features, targets))
        
    def _extract_features(self, metrics: ResourceMetrics) -> np.ndarray:
        """Extract features from metrics."""
        return np.array([
            metrics.cpu_percent,
            metrics.memory_percent,
            metrics.gpu_memory_percent,
            metrics.active_requests,
            metrics.queue_length,
            metrics.response_time,
            metrics.error_rate,
            metrics.timestamp % 86400,  # Time of day
            (metrics.timestamp % 604800) / 86400  # Day of week
        ])
        
    def _extract_targets(self, metrics: ResourceMetrics) -> np.ndarray:
        """Extract target values from metrics."""
        return np.array([
            metrics.cpu_percent,
            metrics.memory_percent,
            metrics.active_requests,
            metrics.response_time
        ])
        
    def train_models(self):
        """Train predictive models."""
        if len(self.training_data) < 100:
            logger.info("Insufficient training data for predictive models")
            return
            
        try:
            # Prepare training data
            X = np.array([features for features, _ in self.training_data])
            y = np.array([targets for _, targets in self.training_data])
            
            # Train separate models for each target
            target_names = ['cpu', 'memory', 'requests', 'response_time']
            
            for i, name in enumerate(target_names):
                # Use RandomForest for non-linear patterns
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X, y[:, i])
                self.models[name] = model
                
            self.last_training = time.time()
            logger.info(f"Trained predictive models with {len(self.training_data)} samples")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            
    def predict(self, current_metrics: ResourceMetrics) -> Optional[Dict[str, float]]:
        """Predict future resource usage."""
        if not self.models:
            return None
            
        try:
            features = self._extract_features(current_metrics).reshape(1, -1)
            predictions = {}
            
            for name, model in self.models.items():
                pred = model.predict(features)[0]
                predictions[name] = max(0, pred)  # Ensure non-negative
                
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
            
    def should_retrain(self) -> bool:
        """Check if models should be retrained."""
        return (time.time() - self.last_training) > self.training_interval


class ScalingStrategy:
    """Base class for scaling strategies."""
    
    def __init__(self, name: str):
        self.name = name
        
    def should_scale(
        self, 
        current_metrics: ResourceMetrics,
        predicted_metrics: Optional[Dict[str, float]] = None,
        config: Dict[str, Any] = None
    ) -> Tuple[ScalingAction, str]:
        """Determine if scaling is needed."""
        raise NotImplementedError
        
    def calculate_new_config(
        self,
        current_config: Dict[str, Any],
        action: ScalingAction,
        metrics: ResourceMetrics
    ) -> Dict[str, Any]:
        """Calculate new configuration."""
        raise NotImplementedError


class AdaptiveScalingStrategy(ScalingStrategy):
    """Adaptive scaling strategy that learns from system behavior."""
    
    def __init__(
        self,
        cpu_threshold_high: float = 80.0,
        cpu_threshold_low: float = 30.0,
        memory_threshold_high: float = 85.0,
        memory_threshold_low: float = 40.0,
        response_time_threshold: float = 2.0,
        error_rate_threshold: float = 0.05
    ):
        super().__init__("AdaptiveScalingStrategy")
        self.cpu_threshold_high = cpu_threshold_high
        self.cpu_threshold_low = cpu_threshold_low
        self.memory_threshold_high = memory_threshold_high
        self.memory_threshold_low = memory_threshold_low
        self.response_time_threshold = response_time_threshold
        self.error_rate_threshold = error_rate_threshold
        
    def should_scale(
        self,
        current_metrics: ResourceMetrics,
        predicted_metrics: Optional[Dict[str, float]] = None,
        config: Dict[str, Any] = None
    ) -> Tuple[ScalingAction, str]:
        """Determine scaling action based on current and predicted metrics."""
        
        # Check for immediate scale-up conditions
        if current_metrics.cpu_percent > self.cpu_threshold_high:
            return ScalingAction.SCALE_UP, f"High CPU usage: {current_metrics.cpu_percent:.1f}%"
            
        if current_metrics.memory_percent > self.memory_threshold_high:
            return ScalingAction.SCALE_UP, f"High memory usage: {current_metrics.memory_percent:.1f}%"
            
        if current_metrics.response_time > self.response_time_threshold:
            return ScalingAction.SCALE_UP, f"High response time: {current_metrics.response_time:.2f}s"
            
        if current_metrics.error_rate > self.error_rate_threshold:
            return ScalingAction.SCALE_UP, f"High error rate: {current_metrics.error_rate:.3f}"
            
        # Check predictive conditions if available
        if predicted_metrics:
            pred_cpu = predicted_metrics.get('cpu', current_metrics.cpu_percent)
            pred_memory = predicted_metrics.get('memory', current_metrics.memory_percent)
            pred_response = predicted_metrics.get('response_time', current_metrics.response_time)
            
            if pred_cpu > self.cpu_threshold_high * 0.9:
                return ScalingAction.SCALE_UP, f"Predicted high CPU: {pred_cpu:.1f}%"
                
            if pred_memory > self.memory_threshold_high * 0.9:
                return ScalingAction.SCALE_UP, f"Predicted high memory: {pred_memory:.1f}%"
                
        # Check for scale-down conditions
        if (current_metrics.cpu_percent < self.cpu_threshold_low and
            current_metrics.memory_percent < self.memory_threshold_low and
            current_metrics.response_time < self.response_time_threshold * 0.5):
            return ScalingAction.SCALE_DOWN, "Low resource utilization"
            
        return ScalingAction.MAINTAIN, "Metrics within normal ranges"
        
    def calculate_new_config(
        self,
        current_config: Dict[str, Any],
        action: ScalingAction,
        metrics: ResourceMetrics
    ) -> Dict[str, Any]:
        """Calculate new configuration based on scaling action."""
        new_config = current_config.copy()
        
        if action == ScalingAction.SCALE_UP:
            # Increase resources intelligently
            if metrics.cpu_percent > self.cpu_threshold_high:
                # Increase worker threads or processes
                current_workers = current_config.get('workers', 1)
                new_config['workers'] = min(current_workers * 2, psutil.cpu_count())
                
            if metrics.memory_percent > self.memory_threshold_high:
                # Reduce batch size to conserve memory
                current_batch = current_config.get('batch_size', 32)
                new_config['batch_size'] = max(1, current_batch // 2)
                
            # Increase timeout limits
            current_timeout = current_config.get('request_timeout', 30)
            new_config['request_timeout'] = current_timeout * 1.5
            
        elif action == ScalingAction.SCALE_DOWN:
            # Decrease resources to save costs
            current_workers = current_config.get('workers', 1)
            new_config['workers'] = max(1, current_workers // 2)
            
            # Increase batch size if memory allows
            current_batch = current_config.get('batch_size', 32)
            if metrics.memory_percent < self.memory_threshold_low:
                new_config['batch_size'] = min(current_batch * 2, 128)
                
        return new_config


class IntelligentAutoScaler:
    """
    Intelligent auto-scaling system with ML-based prediction and adaptive strategies.
    
    Features:
    - Real-time metrics collection and analysis
    - Predictive scaling using machine learning
    - Multiple scaling strategies
    - Cost-aware scaling decisions
    - Performance impact analysis
    - Historical scaling analytics
    """
    
    def __init__(
        self,
        strategy: ScalingStrategy = None,
        metrics_collector: MetricsCollector = None,
        enable_prediction: bool = True,
        scaling_cooldown: float = 300.0,  # 5 minutes
        config_callback: Optional[Callable] = None,
        save_history: bool = True,
        history_file: Optional[Path] = None
    ):
        """
        Initialize intelligent auto-scaler.
        
        Args:
            strategy: Scaling strategy to use
            metrics_collector: Metrics collection system
            enable_prediction: Enable predictive scaling
            scaling_cooldown: Minimum time between scaling actions
            config_callback: Callback to apply configuration changes
            save_history: Save scaling history
            history_file: File to save scaling history
        """
        self.strategy = strategy or AdaptiveScalingStrategy()
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.enable_prediction = enable_prediction
        self.scaling_cooldown = scaling_cooldown
        self.config_callback = config_callback
        self.save_history = save_history
        self.history_file = Path(history_file) if history_file else Path("./scaling_history.json")
        
        self.current_config = {}
        self.scaling_history = []
        self.last_scaling_time = 0
        self.is_running = False
        self.scaling_thread = None
        
        # Initialize predictive scaler
        if self.enable_prediction:
            self.predictor = PredictiveScaler()
        else:
            self.predictor = None
            
        logger.info("Initialized IntelligentAutoScaler")
        
    def start(self, initial_config: Dict[str, Any]):
        """Start the auto-scaling system."""
        if self.is_running:
            return
            
        self.current_config = initial_config.copy()
        self.is_running = True
        
        # Start metrics collection
        self.metrics_collector.start_collection()
        
        # Start scaling thread
        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scaling_thread.start()
        
        logger.info("Started intelligent auto-scaling")
        
    def stop(self):
        """Stop the auto-scaling system."""
        self.is_running = False
        
        # Stop metrics collection
        self.metrics_collector.stop_collection()
        
        # Wait for scaling thread
        if self.scaling_thread:
            self.scaling_thread.join(timeout=10)
            
        # Save history
        if self.save_history and self.scaling_history:
            self._save_history()
            
        logger.info("Stopped intelligent auto-scaling")
        
    def _scaling_loop(self):
        """Main scaling decision loop."""
        while self.is_running:
            try:
                # Get current metrics
                current_metrics = self.metrics_collector._collect_metrics()
                
                # Check cooldown period
                if time.time() - self.last_scaling_time < self.scaling_cooldown:
                    time.sleep(30)  # Check every 30 seconds during cooldown
                    continue
                    
                # Get predictions if enabled
                predictions = None
                if self.predictor:
                    # Train models if needed
                    if self.predictor.should_retrain():
                        self.predictor.train_models()
                        
                    predictions = self.predictor.predict(current_metrics)
                    
                # Determine scaling action
                action, reason = self.strategy.should_scale(
                    current_metrics, predictions, self.current_config
                )
                
                # Execute scaling if needed
                if action != ScalingAction.MAINTAIN:
                    success = self._execute_scaling(action, current_metrics, reason)
                    
                    if success:
                        self.last_scaling_time = time.time()
                        
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Scaling loop error: {e}")
                time.sleep(60)
                
    def _execute_scaling(
        self, 
        action: ScalingAction, 
        metrics: ResourceMetrics, 
        reason: str
    ) -> bool:
        """Execute scaling action."""
        try:
            old_config = self.current_config.copy()
            
            # Calculate new configuration
            new_config = self.strategy.calculate_new_config(
                self.current_config, action, metrics
            )
            
            # Apply configuration if callback provided
            if self.config_callback:
                success = self.config_callback(new_config, old_config)
                if not success:
                    logger.error("Configuration callback failed")
                    return False
                    
            # Update current configuration
            self.current_config = new_config
            
            # Record scaling event
            scaling_event = ScalingEvent(
                timestamp=time.time(),
                action=action,
                trigger_metrics=metrics,
                old_config=old_config,
                new_config=new_config,
                reason=reason,
                success=True
            )
            
            self.scaling_history.append(scaling_event)
            
            logger.info(f"Scaling {action.value}: {reason}")
            logger.info(f"Config change: {self._config_diff(old_config, new_config)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Scaling execution failed: {e}")
            return False
            
    def _config_diff(self, old_config: Dict, new_config: Dict) -> Dict:
        """Calculate configuration differences."""
        diff = {}
        all_keys = set(old_config.keys()) | set(new_config.keys())
        
        for key in all_keys:
            old_val = old_config.get(key, "N/A")
            new_val = new_config.get(key, "N/A")
            if old_val != new_val:
                diff[key] = f"{old_val} -> {new_val}"
                
        return diff
        
    def _save_history(self):
        """Save scaling history to file."""
        try:
            history_data = [asdict(event) for event in self.scaling_history]
            with open(self.history_file, 'w') as f:
                json.dump(history_data, f, indent=2, default=str)
                
            logger.info(f"Saved scaling history: {self.history_file}")
            
        except Exception as e:
            logger.error(f"Failed to save scaling history: {e}")
            
    def get_scaling_analytics(self) -> Dict[str, Any]:
        """Generate scaling analytics and insights."""
        if not self.scaling_history:
            return {"message": "No scaling history available"}
            
        # Basic statistics
        total_events = len(self.scaling_history)
        scale_ups = sum(1 for event in self.scaling_history if event.action == ScalingAction.SCALE_UP)
        scale_downs = sum(1 for event in self.scaling_history if event.action == ScalingAction.SCALE_DOWN)
        
        # Time analysis
        timestamps = [event.timestamp for event in self.scaling_history]
        time_span = max(timestamps) - min(timestamps)
        
        # Most common triggers
        triggers = [event.reason for event in self.scaling_history]
        trigger_counts = {reason: triggers.count(reason) for reason in set(triggers)}
        
        return {
            "total_scaling_events": total_events,
            "scale_up_events": scale_ups,
            "scale_down_events": scale_downs,
            "time_span_hours": time_span / 3600,
            "events_per_hour": total_events / (time_span / 3600) if time_span > 0 else 0,
            "most_common_triggers": sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True),
            "success_rate": sum(1 for e in self.scaling_history if e.success) / total_events
        }
        
    def force_scaling_evaluation(self) -> Dict[str, Any]:
        """Force immediate scaling evaluation (for testing)."""
        try:
            current_metrics = self.metrics_collector._collect_metrics()
            predictions = None
            
            if self.predictor:
                predictions = self.predictor.predict(current_metrics)
                
            action, reason = self.strategy.should_scale(
                current_metrics, predictions, self.current_config
            )
            
            return {
                "current_metrics": asdict(current_metrics),
                "predictions": predictions,
                "recommended_action": action.value,
                "reason": reason,
                "current_config": self.current_config
            }
            
        except Exception as e:
            return {"error": str(e)}


# Example usage and testing
if __name__ == "__main__":
    import time
    
    def dummy_config_callback(new_config: Dict, old_config: Dict) -> bool:
        """Dummy configuration callback for testing."""
        print(f"Applying config change: {new_config}")
        return True
    
    # Initialize auto-scaler
    scaler = IntelligentAutoScaler(
        strategy=AdaptiveScalingStrategy(),
        enable_prediction=True,
        scaling_cooldown=30.0,  # Short cooldown for testing
        config_callback=dummy_config_callback
    )
    
    # Start with initial configuration
    initial_config = {
        "workers": 2,
        "batch_size": 32,
        "request_timeout": 30
    }
    
    scaler.start(initial_config)
    
    # Run for a short time
    print("Auto-scaler running... (testing for 30 seconds)")
    time.sleep(30)
    
    # Get analytics
    analytics = scaler.get_scaling_analytics()
    print(f"Scaling analytics: {analytics}")
    
    # Force evaluation
    evaluation = scaler.force_scaling_evaluation()
    print(f"Current evaluation: {evaluation}")
    
    # Stop scaler
    scaler.stop()
    print("Intelligent Auto-Scaling system test completed!")