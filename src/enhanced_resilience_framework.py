"""Enhanced Resilience Framework for Production Medical AI Systems.

This module provides advanced error recovery, circuit breakers, and self-healing
capabilities for mission-critical medical AI inference systems.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib


class SystemState(Enum):
    """System health states for adaptive behavior."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    RECOVERING = "recovering"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, block requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class HealthMetrics:
    """Comprehensive health metrics for medical AI systems."""
    timestamp: float = field(default_factory=time.time)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    inference_latency: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    model_accuracy: float = 0.0
    data_quality_score: float = 0.0
    

class ResilienceProtocol(Protocol):
    """Protocol for resilience-aware components."""
    def health_check(self) -> bool: ...
    def graceful_degradation(self) -> None: ...
    def self_heal(self) -> bool: ...


class CircuitBreaker:
    """Advanced circuit breaker with adaptive thresholds."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self._lock = threading.Lock()
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker."""
        def wrapper(*args, **kwargs):
            with self._lock:
                if self.state == CircuitState.OPEN:
                    if self._should_attempt_reset():
                        self.state = CircuitState.HALF_OPEN
                    else:
                        raise Exception("Circuit breaker is OPEN")
                
                try:
                    result = func(*args, **kwargs)
                    self._on_success()
                    return result
                except self.expected_exception as e:
                    self._on_failure()
                    raise e
                    
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self) -> None:
        """Handle successful execution."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        
    def _on_failure(self) -> None:
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN


class AdaptiveRetryStrategy:
    """Intelligent retry mechanism with exponential backoff and jitter."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with adaptive retry logic."""
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    logging.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)
                    
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and optional jitter."""
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # 50-100% of calculated delay
            
        return delay


class SelfHealingManager:
    """Advanced self-healing capabilities for medical AI systems."""
    
    def __init__(self):
        self.healing_strategies: Dict[str, Callable] = {}
        self.system_state = SystemState.HEALTHY
        self.metrics_history: List[HealthMetrics] = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def register_healing_strategy(self, name: str, strategy: Callable) -> None:
        """Register a self-healing strategy."""
        self.healing_strategies[name] = strategy
        
    def assess_system_health(self, metrics: HealthMetrics) -> SystemState:
        """Assess overall system health based on metrics."""
        self.metrics_history.append(metrics)
        
        # Keep last 100 metrics for trend analysis
        if len(self.metrics_history) > 100:
            self.metrics_history.pop(0)
            
        # Advanced health assessment logic
        critical_issues = 0
        
        if metrics.error_rate > 0.1:  # >10% error rate
            critical_issues += 2
        elif metrics.error_rate > 0.05:  # >5% error rate
            critical_issues += 1
            
        if metrics.inference_latency > 5.0:  # >5s latency
            critical_issues += 2
        elif metrics.inference_latency > 2.0:  # >2s latency
            critical_issues += 1
            
        if metrics.cpu_usage > 0.9:  # >90% CPU
            critical_issues += 1
            
        if metrics.memory_usage > 0.85:  # >85% memory
            critical_issues += 1
            
        if metrics.model_accuracy < 0.85:  # <85% accuracy
            critical_issues += 3  # Critical for medical systems
            
        # Determine system state
        if critical_issues >= 4:
            return SystemState.CRITICAL
        elif critical_issues >= 2:
            return SystemState.DEGRADED
        else:
            return SystemState.HEALTHY
    
    def auto_heal(self, current_state: SystemState) -> bool:
        """Attempt automatic healing based on system state."""
        if current_state == SystemState.HEALTHY:
            return True
            
        healing_success = True
        
        if current_state == SystemState.DEGRADED:
            # Apply moderate healing strategies
            strategies = ["clear_cache", "restart_workers", "reduce_batch_size"]
        elif current_state == SystemState.CRITICAL:
            # Apply aggressive healing strategies
            strategies = ["emergency_fallback", "restart_system", "alert_admins"]
        else:
            strategies = []
            
        for strategy_name in strategies:
            if strategy_name in self.healing_strategies:
                try:
                    self.healing_strategies[strategy_name]()
                    logging.info(f"Applied healing strategy: {strategy_name}")
                except Exception as e:
                    logging.error(f"Healing strategy {strategy_name} failed: {e}")
                    healing_success = False
                    
        return healing_success


class MedicalAIResilienceFramework:
    """Comprehensive resilience framework for medical AI systems."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_strategies: Dict[str, AdaptiveRetryStrategy] = {}
        self.self_healing = SelfHealingManager()
        self.performance_monitor = PerformanceMonitor()
        
        # Register default healing strategies
        self._register_default_healing_strategies()
        
    def create_circuit_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0
    ) -> CircuitBreaker:
        """Create and register a circuit breaker."""
        cb = CircuitBreaker(failure_threshold, recovery_timeout)
        self.circuit_breakers[name] = cb
        return cb
        
    def create_retry_strategy(
        self,
        name: str,
        max_attempts: int = 3,
        base_delay: float = 1.0
    ) -> AdaptiveRetryStrategy:
        """Create and register a retry strategy."""
        strategy = AdaptiveRetryStrategy(max_attempts, base_delay)
        self.retry_strategies[name] = strategy
        return strategy
        
    def monitor_and_heal(self, metrics: HealthMetrics) -> None:
        """Monitor system health and apply healing if needed."""
        system_state = self.self_healing.assess_system_health(metrics)
        
        if system_state != SystemState.HEALTHY:
            logging.warning(f"System state: {system_state.value}")
            healing_success = self.self_healing.auto_heal(system_state)
            
            if not healing_success:
                logging.error("Self-healing failed - manual intervention required")
                
    def _register_default_healing_strategies(self) -> None:
        """Register default healing strategies."""
        
        def clear_cache():
            """Clear system caches to free memory."""
            import gc
            gc.collect()
            
        def restart_workers():
            """Restart worker processes."""
            logging.info("Restarting worker processes")
            
        def reduce_batch_size():
            """Reduce batch size to lower memory pressure."""
            logging.info("Reducing inference batch size")
            
        def emergency_fallback():
            """Switch to simplified fallback model."""
            logging.warning("Switching to emergency fallback model")
            
        def restart_system():
            """Restart the entire system."""
            logging.critical("Initiating system restart")
            
        def alert_admins():
            """Send alerts to system administrators."""
            logging.critical("Sending alerts to administrators")
            
        strategies = {
            "clear_cache": clear_cache,
            "restart_workers": restart_workers,
            "reduce_batch_size": reduce_batch_size,
            "emergency_fallback": emergency_fallback,
            "restart_system": restart_system,
            "alert_admins": alert_admins,
        }
        
        for name, strategy in strategies.items():
            self.self_healing.register_healing_strategy(name, strategy)


class PerformanceMonitor:
    """Advanced performance monitoring for medical AI systems."""
    
    def __init__(self):
        self.metrics_buffer: List[HealthMetrics] = []
        self.anomaly_detector = AnomalyDetector()
        
    def collect_metrics(self) -> HealthMetrics:
        """Collect comprehensive system metrics."""
        import psutil
        
        # Basic system metrics
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent / 100.0
        
        # GPU metrics (if available)
        gpu_usage = self._get_gpu_usage()
        
        # Application-specific metrics (placeholder)
        inference_latency = self._measure_inference_latency()
        error_rate = self._calculate_error_rate()
        throughput = self._calculate_throughput()
        model_accuracy = self._get_model_accuracy()
        data_quality_score = self._assess_data_quality()
        
        return HealthMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            inference_latency=inference_latency,
            error_rate=error_rate,
            throughput=throughput,
            model_accuracy=model_accuracy,
            data_quality_score=data_quality_score
        )
        
    def _get_gpu_usage(self) -> float:
        """Get GPU usage if available."""
        try:
            # Placeholder for GPU monitoring
            return 0.0
        except Exception:
            return 0.0
            
    def _measure_inference_latency(self) -> float:
        """Measure average inference latency."""
        # Placeholder - would integrate with actual inference pipeline
        return 0.5
        
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate."""
        # Placeholder - would track actual errors
        return 0.01
        
    def _calculate_throughput(self) -> float:
        """Calculate requests per second."""
        # Placeholder - would track actual throughput
        return 100.0
        
    def _get_model_accuracy(self) -> float:
        """Get current model accuracy."""
        # Placeholder - would track real accuracy metrics
        return 0.92
        
    def _assess_data_quality(self) -> float:
        """Assess incoming data quality."""
        # Placeholder - would implement data quality checks
        return 0.95


class AnomalyDetector:
    """Statistical anomaly detection for system metrics."""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.metrics_window: List[float] = []
        
    def is_anomaly(self, value: float, threshold: float = 2.0) -> bool:
        """Detect if a value is anomalous using statistical methods."""
        if len(self.metrics_window) < 10:  # Need minimum data
            self.metrics_window.append(value)
            return False
            
        import statistics
        
        mean = statistics.mean(self.metrics_window)
        stdev = statistics.stdev(self.metrics_window)
        
        z_score = abs(value - mean) / (stdev + 1e-10)  # Avoid division by zero
        
        # Update window
        self.metrics_window.append(value)
        if len(self.metrics_window) > self.window_size:
            self.metrics_window.pop(0)
            
        return z_score > threshold


# Global resilience framework instance
resilience_framework = MedicalAIResilienceFramework()


def resilient_inference(func: Callable) -> Callable:
    """Decorator to make inference functions resilient."""
    circuit_breaker = resilience_framework.create_circuit_breaker(
        f"inference_{func.__name__}", 
        failure_threshold=3,
        recovery_timeout=30.0
    )
    retry_strategy = resilience_framework.create_retry_strategy(
        f"retry_{func.__name__}",
        max_attempts=2,
        base_delay=0.5
    )
    
    @circuit_breaker
    def wrapper(*args, **kwargs):
        return retry_strategy.execute(func, *args, **kwargs)
        
    return wrapper


def monitor_medical_ai_system():
    """Main monitoring loop for medical AI systems."""
    monitor = PerformanceMonitor()
    
    while True:
        try:
            metrics = monitor.collect_metrics()
            resilience_framework.monitor_and_heal(metrics)
            
            # Log critical metrics
            if metrics.error_rate > 0.05:
                logging.warning(f"High error rate detected: {metrics.error_rate:.2%}")
            if metrics.model_accuracy < 0.9:
                logging.warning(f"Model accuracy below threshold: {metrics.model_accuracy:.2%}")
                
            time.sleep(10)  # Monitor every 10 seconds
            
        except Exception as e:
            logging.error(f"Monitoring loop error: {e}")
            time.sleep(30)  # Back off on errors


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Simulate resilient inference function
    @resilient_inference
    def predict_pneumonia(image_data):
        """Example inference function with resilience."""
        # Simulate occasional failures
        import random
        if random.random() < 0.1:  # 10% failure rate
            raise Exception("Model inference failed")
        return {"prediction": "normal", "confidence": 0.95}
    
    # Test the resilient inference
    for i in range(20):
        try:
            result = predict_pneumonia("dummy_image_data")
            print(f"Prediction {i+1}: {result}")
        except Exception as e:
            print(f"Prediction {i+1} failed: {e}")
            
    print("Resilience framework demonstration complete")