"""Auto-scaling system for dynamic resource management."""

import time
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricThreshold:
    """Threshold configuration for auto-scaling metrics."""
    scale_up_threshold: float
    scale_down_threshold: float
    min_duration_seconds: int = 60
    cooldown_seconds: int = 300


@dataclass
class ScalingEvent:
    """Record of a scaling event."""
    timestamp: datetime
    action: str  # 'scale_up' or 'scale_down'
    trigger_metric: str
    trigger_value: float
    old_capacity: int
    new_capacity: int
    success: bool


class ScalingStrategy(ABC):
    """Abstract base class for scaling strategies."""
    
    @abstractmethod
    def calculate_target_capacity(
        self,
        current_capacity: int,
        metrics: Dict[str, float]
    ) -> int:
        """Calculate target capacity based on metrics.
        
        Args:
            current_capacity: Current resource capacity
            metrics: Current system metrics
            
        Returns:
            Target capacity
        """
        pass


class LinearScalingStrategy(ScalingStrategy):
    """Linear scaling strategy based on CPU and memory utilization."""
    
    def __init__(
        self,
        min_capacity: int = 1,
        max_capacity: int = 10,
        scale_factor: float = 1.5
    ):
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.scale_factor = scale_factor
    
    def calculate_target_capacity(
        self,
        current_capacity: int,
        metrics: Dict[str, float]
    ) -> int:
        """Calculate target capacity using linear scaling."""
        cpu_utilization = metrics.get('cpu_percent', 0.0) / 100.0
        memory_utilization = metrics.get('memory_percent', 0.0) / 100.0
        
        # Use the higher of CPU or memory utilization
        max_utilization = max(cpu_utilization, memory_utilization)
        
        if max_utilization > 0.8:  # Scale up at 80%
            target_capacity = int(current_capacity * self.scale_factor)
        elif max_utilization < 0.3:  # Scale down at 30%
            target_capacity = max(1, int(current_capacity / self.scale_factor))
        else:
            target_capacity = current_capacity
        
        return max(
            self.min_capacity,
            min(self.max_capacity, target_capacity)
        )


class PredictiveScalingStrategy(ScalingStrategy):
    """Predictive scaling strategy based on historical patterns."""
    
    def __init__(
        self,
        min_capacity: int = 1,
        max_capacity: int = 10,
        history_window_hours: int = 24
    ):
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.history_window = timedelta(hours=history_window_hours)
        self.metric_history: deque = deque()
    
    def add_metric_sample(self, metrics: Dict[str, float]) -> None:
        """Add metric sample to history."""
        self.metric_history.append((datetime.utcnow(), metrics))
        
        # Clean old samples
        cutoff = datetime.utcnow() - self.history_window
        while self.metric_history and self.metric_history[0][0] < cutoff:
            self.metric_history.popleft()
    
    def calculate_target_capacity(
        self,
        current_capacity: int,
        metrics: Dict[str, float]
    ) -> int:
        """Calculate target capacity using predictive analysis."""
        self.add_metric_sample(metrics)
        
        if len(self.metric_history) < 10:
            # Fallback to simple scaling if insufficient history
            cpu_utilization = metrics.get('cpu_percent', 0.0) / 100.0
            if cpu_utilization > 0.8:
                return min(self.max_capacity, current_capacity + 1)
            elif cpu_utilization < 0.3:
                return max(self.min_capacity, current_capacity - 1)
            else:
                return current_capacity
        
        # Analyze trends in recent history
        recent_samples = list(self.metric_history)[-10:]  # Last 10 samples
        
        # Calculate trend in CPU utilization
        cpu_values = [sample[1].get('cpu_percent', 0.0) for sample in recent_samples]
        if len(cpu_values) >= 2:
            cpu_trend = cpu_values[-1] - cpu_values[0]  # Simple trend
        else:
            cpu_trend = 0.0
        
        current_cpu = metrics.get('cpu_percent', 0.0)
        
        # Predictive scaling decisions
        if current_cpu > 70 or cpu_trend > 20:  # Rising CPU or high current
            target_capacity = min(self.max_capacity, current_capacity + 1)
        elif current_cpu < 30 and cpu_trend < -10:  # Low CPU and decreasing
            target_capacity = max(self.min_capacity, current_capacity - 1)
        else:
            target_capacity = current_capacity
        
        return target_capacity


class AutoScaler:
    """Automatic resource scaling system."""
    
    def __init__(
        self,
        scaling_strategy: Optional[ScalingStrategy] = None,
        check_interval_seconds: int = 30,
        cooldown_seconds: int = 300
    ):
        """Initialize auto-scaler.
        
        Args:
            scaling_strategy: Strategy for scaling decisions
            check_interval_seconds: How often to check metrics
            cooldown_seconds: Minimum time between scaling actions
        """
        self.strategy = scaling_strategy or LinearScalingStrategy()
        self.check_interval = check_interval_seconds
        self.cooldown_seconds = cooldown_seconds
        
        # State
        self.current_capacity = 1
        self.target_capacity = 1
        self.last_scaling_event: Optional[datetime] = None
        self.scaling_history: deque = deque(maxlen=100)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Callbacks for scaling actions
        self.scale_up_callback: Optional[Callable[[int, int], bool]] = None
        self.scale_down_callback: Optional[Callable[[int, int], bool]] = None
        self.get_metrics_callback: Optional[Callable[[], Dict[str, float]]] = None
        
        self._lock = threading.RLock()
    
    def set_callbacks(
        self,
        scale_up: Optional[Callable[[int, int], bool]] = None,
        scale_down: Optional[Callable[[int, int], bool]] = None,
        get_metrics: Optional[Callable[[], Dict[str, float]]] = None
    ) -> None:
        """Set callbacks for scaling operations.
        
        Args:
            scale_up: Callback for scaling up (old_capacity, new_capacity) -> bool
            scale_down: Callback for scaling down (old_capacity, new_capacity) -> bool
            get_metrics: Callback for getting current metrics () -> Dict[str, float]
        """
        self.scale_up_callback = scale_up
        self.scale_down_callback = scale_down
        self.get_metrics_callback = get_metrics
    
    def start(self) -> None:
        """Start the auto-scaling loop."""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._thread = threading.Thread(target=self._scaling_loop, daemon=True)
            self._thread.start()
            
            logger.info("Auto-scaler started")
    
    def stop(self) -> None:
        """Stop the auto-scaling loop."""
        with self._lock:
            self._running = False
            
            if self._thread:
                self._thread.join(timeout=5.0)
                self._thread = None
            
            logger.info("Auto-scaler stopped")
    
    def force_scale_to(self, target_capacity: int) -> bool:
        """Force scaling to specific capacity.
        
        Args:
            target_capacity: Target capacity to scale to
            
        Returns:
            True if scaling was successful
        """
        with self._lock:
            return self._execute_scaling(target_capacity, 'manual')
    
    def get_status(self) -> Dict[str, Any]:
        """Get current auto-scaler status.
        
        Returns:
            Dictionary containing scaler status
        """
        with self._lock:
            recent_events = [
                {
                    'timestamp': event.timestamp.isoformat(),
                    'action': event.action,
                    'trigger_metric': event.trigger_metric,
                    'trigger_value': event.trigger_value,
                    'capacity_change': f"{event.old_capacity} -> {event.new_capacity}",
                    'success': event.success
                }
                for event in list(self.scaling_history)[-10:]  # Last 10 events
            ]
            
            return {
                'running': self._running,
                'current_capacity': self.current_capacity,
                'target_capacity': self.target_capacity,
                'last_scaling_event': self.last_scaling_event.isoformat() if self.last_scaling_event else None,
                'recent_events': recent_events,
                'total_scaling_events': len(self.scaling_history)
            }
    
    def _scaling_loop(self) -> None:
        """Main scaling loop."""
        while self._running:
            try:
                self._check_and_scale()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                time.sleep(self.check_interval)
    
    def _check_and_scale(self) -> None:
        """Check metrics and scale if necessary."""
        if not self.get_metrics_callback:
            return
        
        # Get current metrics
        try:
            metrics = self.get_metrics_callback()
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return
        
        # Calculate target capacity
        target_capacity = self.strategy.calculate_target_capacity(
            self.current_capacity,
            metrics
        )
        
        # Check if scaling is needed
        if target_capacity == self.current_capacity:
            return
        
        # Check cooldown period
        if self.last_scaling_event:
            time_since_last = datetime.utcnow() - self.last_scaling_event
            if time_since_last.total_seconds() < self.cooldown_seconds:
                logger.debug(f"Scaling on cooldown, {time_since_last.total_seconds()}s remaining")
                return
        
        # Determine trigger metric
        trigger_metric = 'cpu_percent'
        trigger_value = metrics.get(trigger_metric, 0.0)
        
        # Execute scaling
        self._execute_scaling(target_capacity, trigger_metric, trigger_value)
    
    def _execute_scaling(
        self,
        target_capacity: int,
        trigger_metric: str = 'manual',
        trigger_value: float = 0.0
    ) -> bool:
        """Execute scaling operation.
        
        Args:
            target_capacity: Target capacity to scale to
            trigger_metric: Metric that triggered scaling
            trigger_value: Value of trigger metric
            
        Returns:
            True if scaling was successful
        """
        old_capacity = self.current_capacity
        
        if target_capacity > old_capacity:
            # Scale up
            if self.scale_up_callback:
                success = self.scale_up_callback(old_capacity, target_capacity)
            else:
                success = True  # Default success if no callback
            
            action = 'scale_up'
            
        elif target_capacity < old_capacity:
            # Scale down
            if self.scale_down_callback:
                success = self.scale_down_callback(old_capacity, target_capacity)
            else:
                success = True  # Default success if no callback
            
            action = 'scale_down'
        else:
            return True  # No scaling needed
        
        # Record scaling event
        event = ScalingEvent(
            timestamp=datetime.utcnow(),
            action=action,
            trigger_metric=trigger_metric,
            trigger_value=trigger_value,
            old_capacity=old_capacity,
            new_capacity=target_capacity,
            success=success
        )
        
        self.scaling_history.append(event)
        
        if success:
            self.current_capacity = target_capacity
            self.last_scaling_event = event.timestamp
            logger.info(f"Scaled {action}: {old_capacity} -> {target_capacity}")
        else:
            logger.error(f"Scaling failed: {action} from {old_capacity} to {target_capacity}")
        
        return success


# Global auto-scaler instance
auto_scaler = AutoScaler()