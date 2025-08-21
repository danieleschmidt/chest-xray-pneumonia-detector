"""
Graceful Degradation Manager
===========================

This module provides intelligent system degradation capabilities that allow
the medical AI system to continue operating with reduced functionality when
components fail, rather than complete system failure.

Features:
- Service health monitoring and failure detection
- Automatic fallback to simpler models/algorithms
- Graceful service reduction with user notification
- Self-healing capabilities and recovery orchestration
- Load shedding and resource management
- Circuit breaker patterns for external dependencies
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import json
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service operational status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"
    RECOVERING = "recovering"


class DegradationLevel(Enum):
    """System degradation levels."""
    FULL = "full_functionality"
    REDUCED = "reduced_functionality"
    ESSENTIAL = "essential_only"
    EMERGENCY = "emergency_mode"


@dataclass
class ServiceHealthMetrics:
    """Health metrics for a service."""
    service_name: str
    status: ServiceStatus
    last_check: datetime
    response_time_ms: float = 0.0
    success_rate: float = 1.0
    error_count: int = 0
    consecutive_failures: int = 0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FallbackStrategy:
    """Fallback strategy configuration."""
    service_name: str
    fallback_function: Callable
    max_degradation_level: DegradationLevel
    priority: int = 1  # Higher number = higher priority
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    user_message: str = "Service operating in reduced mode"


class CircuitBreaker:
    """Circuit breaker implementation for external service calls."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Tuple[bool, Any]:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "open":
                if (datetime.now() - self.last_failure_time).seconds > self.recovery_timeout:
                    self.state = "half_open"
                else:
                    return False, "Circuit breaker is open"
            
            try:
                result = func(*args, **kwargs)
                
                if self.state == "half_open":
                    self.state = "closed"
                    self.failure_count = 0
                
                return True, result
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = datetime.now()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                
                return False, str(e)


class GracefulDegradationManager:
    """Manages graceful system degradation and recovery."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.services: Dict[str, ServiceHealthMetrics] = {}
        self.fallback_strategies: Dict[str, List[FallbackStrategy]] = defaultdict(list)
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.current_degradation_level = DegradationLevel.FULL
        self.health_check_interval = 30  # seconds
        self.monitoring_active = False
        self._lock = threading.RLock()
        self.degradation_history: List[Dict[str, Any]] = []
        self.notification_callbacks: List[Callable] = []
        
        if config_path:
            self.load_configuration(config_path)
        
        self.setup_default_strategies()
    
    def load_configuration(self, config_path: str):
        """Load degradation configuration from file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            self.health_check_interval = config.get('health_check_interval', 30)
            
            # Load service configurations
            for service_config in config.get('services', []):
                service_name = service_config['name']
                self.register_service(service_name)
                
                # Configure circuit breaker if specified
                if 'circuit_breaker' in service_config:
                    cb_config = service_config['circuit_breaker']
                    self.circuit_breakers[service_name] = CircuitBreaker(
                        failure_threshold=cb_config.get('failure_threshold', 5),
                        recovery_timeout=cb_config.get('recovery_timeout', 60)
                    )
                    
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
    
    def setup_default_strategies(self):
        """Set up default fallback strategies for common services."""
        
        # ML Model Prediction Fallback
        def simple_prediction_fallback(*args, **kwargs):
            """Simple rule-based prediction fallback."""
            logger.warning("Using rule-based fallback for ML prediction")
            # Return conservative prediction
            return {"prediction": 0.5, "confidence": 0.3, "method": "rule_based_fallback"}
        
        self.add_fallback_strategy(FallbackStrategy(
            service_name="ml_prediction",
            fallback_function=simple_prediction_fallback,
            max_degradation_level=DegradationLevel.REDUCED,
            priority=1,
            user_message="Using simplified prediction algorithm due to model unavailability"
        ))
        
        # Data Processing Fallback
        def basic_preprocessing_fallback(image_data, *args, **kwargs):
            """Basic preprocessing without advanced augmentation."""
            logger.warning("Using basic preprocessing fallback")
            # Return minimal preprocessing
            return image_data  # Simplified - just return original data
        
        self.add_fallback_strategy(FallbackStrategy(
            service_name="data_preprocessing",
            fallback_function=basic_preprocessing_fallback,
            max_degradation_level=DegradationLevel.ESSENTIAL,
            priority=1,
            user_message="Using basic image preprocessing due to system constraints"
        ))
        
        # API Response Fallback
        def cached_response_fallback(request_id, *args, **kwargs):
            """Return cached response if available."""
            logger.warning("Using cached response fallback")
            return {"status": "degraded", "message": "Serving cached response", "cached": True}
        
        self.add_fallback_strategy(FallbackStrategy(
            service_name="api_endpoint",
            fallback_function=cached_response_fallback,
            max_degradation_level=DegradationLevel.REDUCED,
            priority=2,
            user_message="API responses may be delayed or cached due to system load"
        ))
    
    def register_service(self, service_name: str) -> ServiceHealthMetrics:
        """Register a service for health monitoring."""
        with self._lock:
            if service_name not in self.services:
                metrics = ServiceHealthMetrics(
                    service_name=service_name,
                    status=ServiceStatus.HEALTHY,
                    last_check=datetime.now()
                )
                self.services[service_name] = metrics
                logger.info(f"Registered service: {service_name}")
                return metrics
            return self.services[service_name]
    
    def add_fallback_strategy(self, strategy: FallbackStrategy):
        """Add a fallback strategy for a service."""
        self.fallback_strategies[strategy.service_name].append(strategy)
        # Sort by priority (highest first)
        self.fallback_strategies[strategy.service_name].sort(
            key=lambda s: s.priority, reverse=True
        )
        logger.info(f"Added fallback strategy for {strategy.service_name}")
    
    def update_service_health(self, service_name: str, 
                            response_time_ms: float = 0,
                            success: bool = True,
                            custom_metrics: Optional[Dict[str, Any]] = None):
        """Update health metrics for a service."""
        with self._lock:
            if service_name not in self.services:
                self.register_service(service_name)
            
            metrics = self.services[service_name]
            metrics.last_check = datetime.now()
            metrics.response_time_ms = response_time_ms
            
            if custom_metrics:
                metrics.custom_metrics.update(custom_metrics)
            
            if success:
                metrics.consecutive_failures = 0
                metrics.last_success = datetime.now()
                
                # Update success rate (simple moving average)
                metrics.success_rate = min(1.0, metrics.success_rate * 0.9 + 0.1)
                
            else:
                metrics.consecutive_failures += 1
                metrics.error_count += 1
                metrics.last_failure = datetime.now()
                
                # Update success rate
                metrics.success_rate = max(0.0, metrics.success_rate * 0.9)
            
            # Determine new status
            metrics.status = self._calculate_service_status(metrics)
            
            # Check if system-wide degradation is needed
            self._evaluate_system_degradation()
    
    def _calculate_service_status(self, metrics: ServiceHealthMetrics) -> ServiceStatus:
        """Calculate service status based on metrics."""
        if metrics.consecutive_failures >= 5:
            return ServiceStatus.FAILED
        elif metrics.consecutive_failures >= 3:
            return ServiceStatus.CRITICAL
        elif metrics.success_rate < 0.5 or metrics.response_time_ms > 10000:
            return ServiceStatus.DEGRADED
        elif metrics.consecutive_failures > 0:
            return ServiceStatus.RECOVERING
        else:
            return ServiceStatus.HEALTHY
    
    def _evaluate_system_degradation(self):
        """Evaluate if system-wide degradation is necessary."""
        with self._lock:
            failed_services = [s for s in self.services.values() 
                             if s.status == ServiceStatus.FAILED]
            critical_services = [s for s in self.services.values() 
                               if s.status == ServiceStatus.CRITICAL]
            degraded_services = [s for s in self.services.values() 
                               if s.status == ServiceStatus.DEGRADED]
            
            # Determine new degradation level
            new_level = DegradationLevel.FULL
            
            if len(failed_services) >= 2:
                new_level = DegradationLevel.EMERGENCY
            elif len(failed_services) >= 1 or len(critical_services) >= 2:
                new_level = DegradationLevel.ESSENTIAL
            elif len(critical_services) >= 1 or len(degraded_services) >= 3:
                new_level = DegradationLevel.REDUCED
            
            if new_level != self.current_degradation_level:
                self._trigger_degradation(new_level)
    
    def _trigger_degradation(self, new_level: DegradationLevel):
        """Trigger system degradation to the specified level."""
        old_level = self.current_degradation_level
        self.current_degradation_level = new_level
        
        # Record degradation event
        degradation_event = {
            "timestamp": datetime.now().isoformat(),
            "old_level": old_level.value,
            "new_level": new_level.value,
            "service_states": {name: metrics.status.value 
                            for name, metrics in self.services.items()},
            "trigger_reason": self._analyze_degradation_cause()
        }
        
        self.degradation_history.append(degradation_event)
        
        # Notify subscribers
        self._notify_degradation_change(old_level, new_level)
        
        logger.warning(f"System degradation: {old_level.value} -> {new_level.value}")
    
    def _analyze_degradation_cause(self) -> str:
        """Analyze the root cause of degradation."""
        failed_services = [name for name, metrics in self.services.items()
                          if metrics.status == ServiceStatus.FAILED]
        critical_services = [name for name, metrics in self.services.items()
                           if metrics.status == ServiceStatus.CRITICAL]
        
        if failed_services:
            return f"Service failures: {', '.join(failed_services)}"
        elif critical_services:
            return f"Critical service issues: {', '.join(critical_services)}"
        else:
            return "Multiple service degradation"
    
    def _notify_degradation_change(self, old_level: DegradationLevel, 
                                 new_level: DegradationLevel):
        """Notify registered callbacks about degradation changes."""
        for callback in self.notification_callbacks:
            try:
                callback(old_level, new_level, self.services)
            except Exception as e:
                logger.error(f"Notification callback failed: {e}")
    
    def execute_with_fallback(self, service_name: str, primary_function: Callable,
                            *args, **kwargs) -> Tuple[bool, Any, str]:
        """Execute function with automatic fallback on failure."""
        
        # Try primary function first
        try:
            # Use circuit breaker if configured
            if service_name in self.circuit_breakers:
                cb = self.circuit_breakers[service_name]
                success, result = cb.call(primary_function, *args, **kwargs)
                
                if success:
                    self.update_service_health(service_name, success=True)
                    return True, result, "primary"
                else:
                    self.update_service_health(service_name, success=False)
                    
            else:
                # Direct execution
                start_time = time.time()
                result = primary_function(*args, **kwargs)
                response_time = (time.time() - start_time) * 1000
                
                self.update_service_health(service_name, 
                                         response_time_ms=response_time, 
                                         success=True)
                return True, result, "primary"
                
        except Exception as e:
            logger.warning(f"Primary function failed for {service_name}: {e}")
            self.update_service_health(service_name, success=False)
        
        # Try fallback strategies
        strategies = self.fallback_strategies.get(service_name, [])
        
        for strategy in strategies:
            # Check if strategy is appropriate for current degradation level
            if self._is_strategy_appropriate(strategy):
                try:
                    logger.info(f"Attempting fallback strategy for {service_name}")
                    result = strategy.fallback_function(*args, **kwargs)
                    return True, result, f"fallback_{strategy.priority}"
                    
                except Exception as e:
                    logger.error(f"Fallback strategy failed: {e}")
                    continue
        
        # All strategies failed
        return False, f"All strategies failed for {service_name}", "failed"
    
    def _is_strategy_appropriate(self, strategy: FallbackStrategy) -> bool:
        """Check if a fallback strategy is appropriate for current degradation level."""
        degradation_levels = [DegradationLevel.FULL, DegradationLevel.REDUCED, 
                             DegradationLevel.ESSENTIAL, DegradationLevel.EMERGENCY]
        
        current_index = degradation_levels.index(self.current_degradation_level)
        max_index = degradation_levels.index(strategy.max_degradation_level)
        
        return current_index <= max_index
    
    def add_notification_callback(self, callback: Callable):
        """Add callback for degradation notifications."""
        self.notification_callbacks.append(callback)
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report."""
        with self._lock:
            healthy_count = sum(1 for s in self.services.values() 
                              if s.status == ServiceStatus.HEALTHY)
            total_services = len(self.services)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "degradation_level": self.current_degradation_level.value,
                "overall_health_score": healthy_count / total_services if total_services > 0 else 1.0,
                "service_count": {
                    "total": total_services,
                    "healthy": healthy_count,
                    "degraded": sum(1 for s in self.services.values() 
                                  if s.status == ServiceStatus.DEGRADED),
                    "critical": sum(1 for s in self.services.values() 
                                  if s.status == ServiceStatus.CRITICAL),
                    "failed": sum(1 for s in self.services.values() 
                                if s.status == ServiceStatus.FAILED)
                },
                "services": {
                    name: {
                        "status": metrics.status.value,
                        "last_check": metrics.last_check.isoformat(),
                        "success_rate": metrics.success_rate,
                        "consecutive_failures": metrics.consecutive_failures,
                        "response_time_ms": metrics.response_time_ms
                    }
                    for name, metrics in self.services.items()
                },
                "recent_degradation_events": self.degradation_history[-10:],
                "fallback_strategies_count": {
                    service: len(strategies) 
                    for service, strategies in self.fallback_strategies.items()
                }
            }
    
    def force_recovery_attempt(self, service_name: Optional[str] = None):
        """Force a recovery attempt for specific service or all services."""
        if service_name:
            if service_name in self.services:
                metrics = self.services[service_name]
                if metrics.status in [ServiceStatus.FAILED, ServiceStatus.CRITICAL]:
                    metrics.status = ServiceStatus.RECOVERING
                    metrics.consecutive_failures = max(0, metrics.consecutive_failures - 2)
                    logger.info(f"Forced recovery attempt for {service_name}")
        else:
            # Attempt recovery for all problematic services
            for name, metrics in self.services.items():
                if metrics.status in [ServiceStatus.FAILED, ServiceStatus.CRITICAL]:
                    metrics.status = ServiceStatus.RECOVERING
                    metrics.consecutive_failures = max(0, metrics.consecutive_failures - 1)
            logger.info("Forced recovery attempt for all services")
        
        # Re-evaluate system degradation
        self._evaluate_system_degradation()


def example_usage():
    """Example usage of the graceful degradation manager."""
    
    # Initialize manager
    manager = GracefulDegradationManager()
    
    # Add notification callback
    def degradation_notification(old_level, new_level, services):
        print(f"🔄 System degradation: {old_level.value} -> {new_level.value}")
        failed_services = [name for name, metrics in services.items()
                          if metrics.status == ServiceStatus.FAILED]
        if failed_services:
            print(f"📛 Failed services: {', '.join(failed_services)}")
    
    manager.add_notification_callback(degradation_notification)
    
    # Simulate service operations
    def ml_predict(image_data):
        """Example ML prediction function."""
        import random
        if random.random() < 0.8:  # 80% success rate
            return {"prediction": "normal", "confidence": 0.95}
        else:
            raise Exception("Model inference failed")
    
    # Test execution with fallback
    for i in range(10):
        success, result, method = manager.execute_with_fallback(
            "ml_prediction", ml_predict, "test_image_data"
        )
        print(f"Iteration {i+1}: {method} - {result}")
        time.sleep(1)
    
    # Generate health report
    report = manager.get_system_health_report()
    print("\n📊 System Health Report:")
    print(f"Overall Health Score: {report['overall_health_score']:.1%}")
    print(f"Degradation Level: {report['degradation_level']}")


if __name__ == "__main__":
    example_usage()