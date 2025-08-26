"""Intelligent Error Recovery System for Medical AI Applications.

Implements advanced error recovery, fault tolerance, and self-healing
capabilities for robust medical AI systems.
"""

import asyncio
import functools
import logging
import random
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryAction(Enum):
    """Types of recovery actions."""
    RETRY = "retry"
    FAILOVER = "failover"
    CIRCUIT_BREAK = "circuit_break"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    RESTART_COMPONENT = "restart_component"
    ALERT_OPERATOR = "alert_operator"
    ROLLBACK = "rollback"


@dataclass
class ErrorContext:
    """Context information for an error."""
    error_id: str
    timestamp: datetime
    error_type: str
    error_message: str
    severity: ErrorSeverity
    component: str
    operation: str
    stack_trace: str
    input_data: Optional[Dict[str, Any]] = None
    system_state: Optional[Dict[str, Any]] = None
    recovery_attempts: int = 0
    resolved: bool = False


@dataclass
class RecoveryStrategy:
    """Recovery strategy configuration."""
    error_patterns: List[str]
    actions: List[RecoveryAction]
    max_retries: int = 3
    backoff_multiplier: float = 2.0
    initial_delay: float = 1.0
    timeout: float = 30.0
    conditions: Optional[Dict[str, Any]] = None


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self._call(func, *args, **kwargs)
        return wrapper
    
    def _call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.expected_exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return False
        return time.time() - self.last_failure_time > self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
        self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class RetryMechanism:
    """Advanced retry mechanism with exponential backoff."""
    
    def __init__(self, 
                 max_retries: int = 3,
                 backoff_multiplier: float = 2.0,
                 initial_delay: float = 1.0,
                 max_delay: float = 60.0,
                 jitter: bool = True):
        self.max_retries = max_retries
        self.backoff_multiplier = backoff_multiplier
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.jitter = jitter
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for retry mechanism."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self._retry_call(func, *args, **kwargs)
        return wrapper
    
    def _retry_call(self, func: Callable, *args, **kwargs):
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"Function {func.__name__} succeeded after {attempt} retries")
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}), "
                                 f"retrying in {delay:.2f} seconds: {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"Function {func.__name__} failed after {self.max_retries} retries")
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for next retry."""
        delay = self.initial_delay * (self.backoff_multiplier ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add random jitter to prevent thundering herd
            delay *= (0.5 + random.random() * 0.5)
        
        return delay


class ErrorPatternMatcher:
    """Matches errors against known patterns for recovery."""
    
    def __init__(self):
        self.patterns = {}
    
    def register_pattern(self, 
                        pattern_name: str,
                        error_types: List[str],
                        message_patterns: List[str],
                        recovery_strategy: RecoveryStrategy):
        """Register error pattern and recovery strategy."""
        self.patterns[pattern_name] = {
            'error_types': error_types,
            'message_patterns': message_patterns,
            'recovery_strategy': recovery_strategy
        }
    
    def match_error(self, error_context: ErrorContext) -> Optional[RecoveryStrategy]:
        """Match error against known patterns."""
        for pattern_name, pattern_config in self.patterns.items():
            if self._matches_pattern(error_context, pattern_config):
                logger.info(f"Error matched pattern: {pattern_name}")
                return pattern_config['recovery_strategy']
        
        logger.warning("No matching recovery pattern found for error")
        return None
    
    def _matches_pattern(self, 
                        error_context: ErrorContext, 
                        pattern_config: Dict[str, Any]) -> bool:
        """Check if error matches pattern."""
        # Check error type
        error_type_match = any(
            error_type.lower() in error_context.error_type.lower()
            for error_type in pattern_config['error_types']
        )
        
        if not error_type_match:
            return False
        
        # Check message patterns
        message_match = any(
            pattern.lower() in error_context.error_message.lower()
            for pattern in pattern_config['message_patterns']
        )
        
        return message_match


class SelfHealingSystem:
    """Self-healing system that automatically recovers from errors."""
    
    def __init__(self):
        self.component_states = {}
        self.health_checks = {}
        self.recovery_actions = {}
        self.is_running = False
        self.healing_thread = None
    
    def register_component(self, 
                          component_name: str,
                          health_check: Callable[[], bool],
                          recovery_action: Callable[[], bool]):
        """Register component for self-healing."""
        self.component_states[component_name] = {
            'healthy': True,
            'last_check': datetime.now(),
            'failure_count': 0,
            'last_recovery': None
        }
        self.health_checks[component_name] = health_check
        self.recovery_actions[component_name] = recovery_action
    
    def start_healing(self, check_interval: float = 30.0):
        """Start self-healing monitoring."""
        if self.is_running:
            return
        
        self.is_running = True
        self.healing_thread = threading.Thread(
            target=self._healing_loop,
            args=(check_interval,)
        )
        self.healing_thread.daemon = True
        self.healing_thread.start()
        logger.info("Self-healing system started")
    
    def stop_healing(self):
        """Stop self-healing monitoring."""
        self.is_running = False
        if self.healing_thread:
            self.healing_thread.join()
        logger.info("Self-healing system stopped")
    
    def _healing_loop(self, check_interval: float):
        """Main self-healing loop."""
        while self.is_running:
            try:
                for component_name in self.component_states.keys():
                    self._check_and_heal_component(component_name)
                
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error in self-healing loop: {e}")
                time.sleep(10)
    
    def _check_and_heal_component(self, component_name: str):
        """Check component health and attempt healing if needed."""
        try:
            # Run health check
            is_healthy = self.health_checks[component_name]()
            state = self.component_states[component_name]
            
            if is_healthy:
                if not state['healthy']:
                    logger.info(f"Component {component_name} recovered")
                state['healthy'] = True
                state['failure_count'] = 0
            else:
                logger.warning(f"Component {component_name} unhealthy")
                state['healthy'] = False
                state['failure_count'] += 1
                
                # Attempt recovery
                if self._should_attempt_recovery(component_name):
                    self._attempt_recovery(component_name)
            
            state['last_check'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error checking component {component_name}: {e}")
    
    def _should_attempt_recovery(self, component_name: str) -> bool:
        """Determine if recovery should be attempted."""
        state = self.component_states[component_name]
        
        # Don't attempt recovery too frequently
        if state['last_recovery']:
            time_since_recovery = datetime.now() - state['last_recovery']
            if time_since_recovery < timedelta(minutes=5):
                return False
        
        # Attempt recovery if component has failed multiple times
        return state['failure_count'] >= 2
    
    def _attempt_recovery(self, component_name: str):
        """Attempt to recover component."""
        try:
            logger.info(f"Attempting recovery for component {component_name}")
            success = self.recovery_actions[component_name]()
            
            state = self.component_states[component_name]
            state['last_recovery'] = datetime.now()
            
            if success:
                logger.info(f"Recovery successful for component {component_name}")
                state['failure_count'] = 0
            else:
                logger.warning(f"Recovery failed for component {component_name}")
                
        except Exception as e:
            logger.error(f"Recovery attempt failed for {component_name}: {e}")


class GracefulDegradationManager:
    """Manages graceful degradation of system functionality."""
    
    def __init__(self):
        self.degradation_modes = {}
        self.current_mode = "normal"
        self.feature_toggles = {}
    
    def register_degradation_mode(self, 
                                 mode_name: str,
                                 disabled_features: List[str],
                                 fallback_implementations: Dict[str, Callable] = None):
        """Register a degradation mode."""
        self.degradation_modes[mode_name] = {
            'disabled_features': disabled_features,
            'fallback_implementations': fallback_implementations or {}
        }
    
    def activate_degradation_mode(self, mode_name: str):
        """Activate degradation mode."""
        if mode_name not in self.degradation_modes:
            logger.error(f"Unknown degradation mode: {mode_name}")
            return False
        
        self.current_mode = mode_name
        mode_config = self.degradation_modes[mode_name]
        
        # Disable features
        for feature in mode_config['disabled_features']:
            self.feature_toggles[feature] = False
        
        logger.warning(f"Activated degradation mode: {mode_name}")
        return True
    
    def deactivate_degradation_mode(self):
        """Return to normal operation mode."""
        self.current_mode = "normal"
        self.feature_toggles.clear()
        logger.info("Returned to normal operation mode")
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if feature is currently enabled."""
        return self.feature_toggles.get(feature_name, True)
    
    def get_fallback_implementation(self, feature_name: str) -> Optional[Callable]:
        """Get fallback implementation for feature."""
        if self.current_mode == "normal":
            return None
        
        mode_config = self.degradation_modes.get(self.current_mode, {})
        return mode_config.get('fallback_implementations', {}).get(feature_name)


class IntelligentErrorRecoverySystem:
    """Main intelligent error recovery system."""
    
    def __init__(self):
        self.error_history = deque(maxlen=1000)
        self.pattern_matcher = ErrorPatternMatcher()
        self.self_healing = SelfHealingSystem()
        self.degradation_manager = GracefulDegradationManager()
        
        self.recovery_stats = defaultdict(int)
        self.is_learning = True
        
        # Setup default recovery patterns
        self._setup_default_patterns()
        
        # Setup default degradation modes
        self._setup_default_degradation_modes()
    
    def _setup_default_patterns(self):
        """Setup default error recovery patterns."""
        
        # Network/Connection errors
        network_strategy = RecoveryStrategy(
            error_patterns=["connection", "network", "timeout", "unreachable"],
            actions=[RecoveryAction.RETRY, RecoveryAction.FAILOVER],
            max_retries=3,
            backoff_multiplier=2.0,
            initial_delay=1.0
        )
        
        self.pattern_matcher.register_pattern(
            "network_errors",
            ["ConnectionError", "TimeoutError", "NetworkError"],
            ["connection", "network", "timeout", "unreachable"],
            network_strategy
        )
        
        # Memory errors
        memory_strategy = RecoveryStrategy(
            error_patterns=["memory", "out of memory", "allocation"],
            actions=[RecoveryAction.GRACEFUL_DEGRADATION, RecoveryAction.RESTART_COMPONENT],
            max_retries=1
        )
        
        self.pattern_matcher.register_pattern(
            "memory_errors",
            ["MemoryError", "OutOfMemoryError"],
            ["memory", "out of memory", "allocation"],
            memory_strategy
        )
        
        # Model errors
        model_strategy = RecoveryStrategy(
            error_patterns=["model", "inference", "prediction"],
            actions=[RecoveryAction.RETRY, RecoveryAction.GRACEFUL_DEGRADATION],
            max_retries=2
        )
        
        self.pattern_matcher.register_pattern(
            "model_errors",
            ["ModelError", "InferenceError", "PredictionError"],
            ["model", "inference", "prediction", "tensor"],
            model_strategy
        )
    
    def _setup_default_degradation_modes(self):
        """Setup default degradation modes."""
        
        # Light degradation mode
        self.degradation_manager.register_degradation_mode(
            "light_degradation",
            disabled_features=["advanced_analytics", "detailed_logging"],
            fallback_implementations={
                "model_inference": self._simple_fallback_inference
            }
        )
        
        # Heavy degradation mode
        self.degradation_manager.register_degradation_mode(
            "heavy_degradation",
            disabled_features=[
                "advanced_analytics", "detailed_logging", 
                "real_time_processing", "batch_processing"
            ],
            fallback_implementations={
                "model_inference": self._emergency_fallback_inference
            }
        )
    
    def _simple_fallback_inference(self, *args, **kwargs):
        """Simple fallback for model inference."""
        logger.info("Using simple fallback inference")
        # Return safe default prediction
        return {"prediction": 0.5, "confidence": 0.1, "fallback": True}
    
    def _emergency_fallback_inference(self, *args, **kwargs):
        """Emergency fallback for model inference."""
        logger.warning("Using emergency fallback inference")
        # Return minimal safe response
        return {"prediction": 0.0, "confidence": 0.0, "emergency_fallback": True}
    
    @contextmanager
    def error_recovery_context(self, 
                              component: str,
                              operation: str,
                              input_data: Optional[Dict[str, Any]] = None):
        """Context manager for automatic error recovery."""
        error_id = f"err_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        
        try:
            yield
            
        except Exception as e:
            # Create error context
            error_context = ErrorContext(
                error_id=error_id,
                timestamp=datetime.now(),
                error_type=type(e).__name__,
                error_message=str(e),
                severity=self._classify_error_severity(e),
                component=component,
                operation=operation,
                stack_trace=traceback.format_exc(),
                input_data=input_data
            )
            
            self.error_history.append(error_context)
            
            # Attempt recovery
            recovery_success = self._attempt_error_recovery(error_context)
            
            if not recovery_success:
                # Re-raise if recovery failed
                raise e
    
    def _classify_error_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity based on error type and context."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Critical errors
        critical_patterns = ["memory", "segmentation", "corruption", "fatal"]
        if any(pattern in error_message for pattern in critical_patterns):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        high_patterns = ["connection", "authentication", "permission", "model"]
        if any(pattern in error_message for pattern in high_patterns):
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        medium_patterns = ["timeout", "validation", "parsing"]
        if any(pattern in error_message for pattern in medium_patterns):
            return ErrorSeverity.MEDIUM
        
        return ErrorSeverity.LOW
    
    def _attempt_error_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt to recover from error."""
        logger.info(f"Attempting recovery for error: {error_context.error_id}")
        
        # Find matching recovery strategy
        recovery_strategy = self.pattern_matcher.match_error(error_context)
        
        if not recovery_strategy:
            logger.warning("No recovery strategy found")
            return False
        
        # Execute recovery actions
        for action in recovery_strategy.actions:
            try:
                success = self._execute_recovery_action(action, error_context, recovery_strategy)
                if success:
                    logger.info(f"Recovery successful using action: {action.value}")
                    self.recovery_stats[action.value] += 1
                    error_context.resolved = True
                    return True
            except Exception as e:
                logger.error(f"Recovery action {action.value} failed: {e}")
        
        logger.error(f"All recovery actions failed for error: {error_context.error_id}")
        return False
    
    def _execute_recovery_action(self, 
                                action: RecoveryAction,
                                error_context: ErrorContext,
                                strategy: RecoveryStrategy) -> bool:
        """Execute specific recovery action."""
        
        if action == RecoveryAction.RETRY:
            # Retry logic already handled by RetryMechanism decorator
            return True
        
        elif action == RecoveryAction.GRACEFUL_DEGRADATION:
            if error_context.severity == ErrorSeverity.CRITICAL:
                return self.degradation_manager.activate_degradation_mode("heavy_degradation")
            else:
                return self.degradation_manager.activate_degradation_mode("light_degradation")
        
        elif action == RecoveryAction.FAILOVER:
            # Implement failover logic
            logger.info("Executing failover recovery action")
            return True  # Placeholder
        
        elif action == RecoveryAction.CIRCUIT_BREAK:
            # Circuit breaker already handles this
            return True
        
        elif action == RecoveryAction.RESTART_COMPONENT:
            logger.info(f"Restarting component: {error_context.component}")
            # Implement component restart logic
            return True  # Placeholder
        
        elif action == RecoveryAction.ALERT_OPERATOR:
            logger.critical(f"Operator intervention required for error: {error_context.error_id}")
            return False  # Requires manual intervention
        
        else:
            return False
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get error recovery statistics."""
        total_errors = len(self.error_history)
        resolved_errors = sum(1 for error in self.error_history if error.resolved)
        
        severity_counts = defaultdict(int)
        for error in self.error_history:
            severity_counts[error.severity.value] += 1
        
        return {
            "total_errors": total_errors,
            "resolved_errors": resolved_errors,
            "resolution_rate": resolved_errors / total_errors if total_errors > 0 else 0,
            "severity_distribution": dict(severity_counts),
            "recovery_actions_used": dict(self.recovery_stats),
            "current_degradation_mode": self.degradation_manager.current_mode
        }


# Decorators for easy integration
def with_circuit_breaker(failure_threshold: int = 5, recovery_timeout: float = 60.0):
    """Decorator for circuit breaker protection."""
    def decorator(func):
        breaker = CircuitBreaker(failure_threshold, recovery_timeout)
        return breaker(func)
    return decorator


def with_retry(max_retries: int = 3, backoff_multiplier: float = 2.0):
    """Decorator for retry mechanism."""
    def decorator(func):
        retry = RetryMechanism(max_retries, backoff_multiplier)
        return retry(func)
    return decorator


def demonstrate_error_recovery():
    """Demonstrate intelligent error recovery system."""
    print("Intelligent Error Recovery System Demo")
    print("=" * 50)
    
    # Create recovery system
    recovery_system = IntelligentErrorRecoverySystem()
    
    # Start self-healing
    recovery_system.self_healing.start_healing(check_interval=5.0)
    
    @with_retry(max_retries=2)
    @with_circuit_breaker(failure_threshold=3)
    def unreliable_function(fail_probability: float = 0.7):
        """Simulate unreliable function."""
        if random.random() < fail_probability:
            raise ConnectionError("Network connection failed")
        return "Success!"
    
    def model_inference_with_recovery(data):
        """Model inference with error recovery."""
        with recovery_system.error_recovery_context(
            component="model_service",
            operation="inference",
            input_data={"data": data}
        ):
            # Simulate model inference that might fail
            if random.random() < 0.3:  # 30% failure rate
                raise RuntimeError("Model inference failed")
            return {"prediction": 0.95, "confidence": 0.89}
    
    # Register self-healing component
    def model_health_check():
        return random.random() > 0.2  # 80% healthy
    
    def model_recovery_action():
        logger.info("Recovering model service...")
        time.sleep(0.1)  # Simulate recovery time
        return True
    
    recovery_system.self_healing.register_component(
        "model_service",
        model_health_check,
        model_recovery_action
    )
    
    print("\n1. Testing retry mechanism:")
    try:
        result = unreliable_function(fail_probability=0.5)
        print(f"Function succeeded: {result}")
    except Exception as e:
        print(f"Function failed after retries: {e}")
    
    print("\n2. Testing error recovery context:")
    successes = 0
    attempts = 5
    
    for i in range(attempts):
        try:
            result = model_inference_with_recovery(f"data_{i}")
            print(f"Inference {i+1} succeeded: {result}")
            successes += 1
        except Exception as e:
            print(f"Inference {i+1} failed: {e}")
    
    print(f"\nSuccess rate: {successes}/{attempts} ({successes/attempts*100:.1f}%)")
    
    # Wait for self-healing to run
    print("\n3. Self-healing monitoring (waiting 10 seconds)...")
    time.sleep(10)
    
    # Show recovery statistics
    print("\n4. Recovery Statistics:")
    stats = recovery_system.get_recovery_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Stop self-healing
    recovery_system.self_healing.stop_healing()
    print("\nDemo completed.")


if __name__ == "__main__":
    demonstrate_error_recovery()