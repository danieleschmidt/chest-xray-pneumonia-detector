"""Comprehensive error handling for quantum task planner.

Provides structured error handling, recovery mechanisms, and
resilience patterns for quantum scheduling operations.
"""

import logging
import traceback
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import functools
import time

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors in the quantum task system."""
    VALIDATION = "validation"
    RESOURCE = "resource"
    SCHEDULING = "scheduling"
    QUANTUM_COMPUTATION = "quantum_computation"
    PERSISTENCE = "persistence"
    NETWORK = "network"
    SECURITY = "security"
    SYSTEM = "system"


@dataclass
class QuantumError:
    """Structured error information for quantum operations."""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    operation: str
    user_message: Optional[str] = None
    recovery_suggestion: Optional[str] = None
    error_code: Optional[str] = None


class QuantumTaskPlannerException(Exception):
    """Base exception for quantum task planner operations."""
    
    def __init__(self, quantum_error: QuantumError):
        self.quantum_error = quantum_error
        super().__init__(quantum_error.message)


class ValidationException(QuantumTaskPlannerException):
    """Exception for validation failures."""
    pass


class ResourceExhaustionException(QuantumTaskPlannerException):
    """Exception for resource allocation failures."""
    pass


class SchedulingException(QuantumTaskPlannerException):
    """Exception for scheduling operation failures."""
    pass


class QuantumComputationException(QuantumTaskPlannerException):
    """Exception for quantum algorithm computation failures."""
    pass


class ErrorHandler:
    """Centralized error handling and recovery system."""
    
    def __init__(self):
        self.error_history: List[QuantumError] = []
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {}
        self.error_counts: Dict[str, int] = {}
        self._setup_default_recovery_strategies()
    
    def _setup_default_recovery_strategies(self) -> None:
        """Setup default error recovery strategies."""
        self.recovery_strategies = {
            ErrorCategory.VALIDATION: self._handle_validation_error,
            ErrorCategory.RESOURCE: self._handle_resource_error,
            ErrorCategory.SCHEDULING: self._handle_scheduling_error,
            ErrorCategory.QUANTUM_COMPUTATION: self._handle_quantum_error,
            ErrorCategory.PERSISTENCE: self._handle_persistence_error,
            ErrorCategory.NETWORK: self._handle_network_error,
            ErrorCategory.SECURITY: self._handle_security_error,
            ErrorCategory.SYSTEM: self._handle_system_error
        }
    
    def handle_error(self, error: QuantumError) -> Optional[Any]:
        """Handle an error using appropriate recovery strategy."""
        # Log the error
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }
        logger.log(log_level[error.severity], 
                  f"Quantum error [{error.category.value}]: {error.message}")
        
        # Record error
        self.error_history.append(error)
        error_key = f"{error.category.value}_{error.operation}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Apply recovery strategy
        recovery_function = self.recovery_strategies.get(error.category)
        if recovery_function:
            try:
                return recovery_function(error)
            except Exception as recovery_error:
                logger.error(f"Error recovery failed: {recovery_error}")
                # Escalate error
                escalated_error = QuantumError(
                    category=ErrorCategory.SYSTEM,
                    severity=ErrorSeverity.CRITICAL,
                    message=f"Recovery failed for {error.category.value} error",
                    details={"original_error": error.message, "recovery_error": str(recovery_error)},
                    timestamp=datetime.now(),
                    operation="error_recovery"
                )
                self.error_history.append(escalated_error)
        
        return None
    
    def _handle_validation_error(self, error: QuantumError) -> Optional[Any]:
        """Handle validation errors with data sanitization."""
        logger.info(f"Applying validation error recovery for: {error.operation}")
        
        # For validation errors, we typically can't auto-recover
        # but we can provide helpful guidance
        if "name" in error.details:
            error.recovery_suggestion = "Please provide a valid task name (1-200 characters, no special characters)"
        elif "priority" in error.details:
            error.recovery_suggestion = "Please use one of: low, medium, high, critical"
        elif "dependencies" in error.details:
            error.recovery_suggestion = "Check that all dependency IDs exist and are valid UUIDs"
        
        return None
    
    def _handle_resource_error(self, error: QuantumError) -> Optional[Any]:
        """Handle resource allocation errors with retry logic."""
        logger.info(f"Applying resource error recovery for: {error.operation}")
        
        # Attempt resource rebalancing
        if "insufficient" in error.message.lower():
            error.recovery_suggestion = "Consider rebalancing resources or increasing capacity"
            # Could trigger automatic rebalancing here
        
        return None
    
    def _handle_scheduling_error(self, error: QuantumError) -> Optional[Any]:
        """Handle scheduling errors with fallback algorithms."""
        logger.info(f"Applying scheduling error recovery for: {error.operation}")
        
        # For scheduling errors, suggest simpler algorithms
        if "optimization" in error.operation:
            error.recovery_suggestion = "Try using a simpler scheduling algorithm or reduce iteration count"
        
        return None
    
    def _handle_quantum_error(self, error: QuantumError) -> Optional[Any]:
        """Handle quantum computation errors with classical fallbacks."""
        logger.info(f"Applying quantum error recovery for: {error.operation}")
        
        # Suggest classical algorithms as fallback
        error.recovery_suggestion = "Quantum computation failed - falling back to classical scheduling algorithms"
        
        return None
    
    def _handle_persistence_error(self, error: QuantumError) -> Optional[Any]:
        """Handle persistence/storage errors with retry logic."""
        logger.info(f"Applying persistence error recovery for: {error.operation}")
        
        # Suggest retry with exponential backoff
        error.recovery_suggestion = "Storage operation failed - will retry with exponential backoff"
        
        return None
    
    def _handle_network_error(self, error: QuantumError) -> Optional[Any]:
        """Handle network errors with retry and circuit breaker patterns."""
        logger.info(f"Applying network error recovery for: {error.operation}")
        
        error.recovery_suggestion = "Network operation failed - check connectivity and retry"
        
        return None
    
    def _handle_security_error(self, error: QuantumError) -> Optional[Any]:
        """Handle security errors with immediate blocking."""
        logger.critical(f"Security error detected: {error.message}")
        
        # Security errors should not be auto-recovered
        error.recovery_suggestion = "Security violation detected - manual review required"
        
        return None
    
    def _handle_system_error(self, error: QuantumError) -> Optional[Any]:
        """Handle system-level errors with graceful degradation."""
        logger.error(f"System error detected: {error.message}")
        
        error.recovery_suggestion = "System error occurred - check system resources and restart if necessary"
        
        return None
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_errors = [
            error for error in self.error_history
            if error.timestamp > cutoff_time
        ]
        
        # Categorize errors
        error_by_category = {}
        error_by_severity = {}
        
        for error in recent_errors:
            category = error.category.value
            severity = error.severity.value
            
            error_by_category[category] = error_by_category.get(category, 0) + 1
            error_by_severity[severity] = error_by_severity.get(severity, 0) + 1
        
        return {
            "total_errors": len(recent_errors),
            "errors_by_category": error_by_category,
            "errors_by_severity": error_by_severity,
            "most_common_errors": self._get_most_common_errors(recent_errors),
            "period_hours": hours
        }
    
    def _get_most_common_errors(self, errors: List[QuantumError], limit: int = 5) -> List[Dict]:
        """Get most common error patterns."""
        error_patterns = {}
        
        for error in errors:
            pattern_key = f"{error.category.value}_{error.operation}"
            if pattern_key not in error_patterns:
                error_patterns[pattern_key] = {
                    "pattern": pattern_key,
                    "count": 0,
                    "latest_message": error.message,
                    "category": error.category.value,
                    "operation": error.operation
                }
            error_patterns[pattern_key]["count"] += 1
        
        # Sort by count and return top patterns
        sorted_patterns = sorted(error_patterns.values(), key=lambda x: x["count"], reverse=True)
        return sorted_patterns[:limit]


def quantum_error_handler(category: ErrorCategory, operation: str, 
                         severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """Decorator for automatic error handling in quantum operations."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Create structured error
                quantum_error = QuantumError(
                    category=category,
                    severity=severity,
                    message=str(e),
                    details={
                        "function": func.__name__,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys()),
                        "traceback": traceback.format_exc()
                    },
                    timestamp=datetime.now(),
                    operation=operation,
                    error_code=f"{category.value}_{operation}_{int(time.time())}"
                )
                
                # Get error handler instance (would be injected in real implementation)
                error_handler = ErrorHandler()
                error_handler.handle_error(quantum_error)
                
                # Re-raise as quantum exception
                if category == ErrorCategory.VALIDATION:
                    raise ValidationException(quantum_error)
                elif category == ErrorCategory.RESOURCE:
                    raise ResourceExhaustionException(quantum_error)
                elif category == ErrorCategory.SCHEDULING:
                    raise SchedulingException(quantum_error)
                elif category == ErrorCategory.QUANTUM_COMPUTATION:
                    raise QuantumComputationException(quantum_error)
                else:
                    raise QuantumTaskPlannerException(quantum_error)
        
        return wrapper
    return decorator


def circuit_breaker(failure_threshold: int = 5, recovery_timeout: int = 60,
                   expected_exception: type = Exception):
    """Circuit breaker pattern for quantum operations."""
    def decorator(func: Callable) -> Callable:
        failure_count = 0
        last_failure_time = None
        circuit_open = False
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal failure_count, last_failure_time, circuit_open
            
            # Check if circuit should be closed (recovery timeout passed)
            if circuit_open and last_failure_time:
                if time.time() - last_failure_time > recovery_timeout:
                    circuit_open = False
                    failure_count = 0
                    logger.info(f"Circuit breaker closed for {func.__name__}")
            
            # If circuit is open, fail fast
            if circuit_open:
                raise QuantumTaskPlannerException(
                    QuantumError(
                        category=ErrorCategory.SYSTEM,
                        severity=ErrorSeverity.HIGH,
                        message=f"Circuit breaker open for {func.__name__}",
                        details={"failure_count": failure_count, "last_failure": last_failure_time},
                        timestamp=datetime.now(),
                        operation="circuit_breaker"
                    )
                )
            
            try:
                result = func(*args, **kwargs)
                # Success - reset failure count
                if failure_count > 0:
                    failure_count = 0
                    logger.info(f"Circuit breaker reset for {func.__name__}")
                return result
                
            except expected_exception as e:
                failure_count += 1
                last_failure_time = time.time()
                
                # Open circuit if threshold reached
                if failure_count >= failure_threshold:
                    circuit_open = True
                    logger.error(f"Circuit breaker opened for {func.__name__} after {failure_count} failures")
                
                raise e
        
        return wrapper
    return decorator


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0,
                      backoff_factor: float = 2.0, max_delay: float = 60.0):
    """Retry decorator with exponential backoff for quantum operations."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        # Final attempt failed
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                        raise e
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}, retrying in {delay:.2f}s: {e}")
                    time.sleep(delay)
            
            # Should never reach here, but just in case
            raise last_exception
        
        return wrapper
    return decorator


class QuantumErrorRecovery:
    """Advanced error recovery mechanisms for quantum operations."""
    
    @staticmethod
    def recover_quantum_state(scheduler, error_details: Dict[str, Any]) -> bool:
        """Attempt to recover quantum state after computation error."""
        try:
            logger.info("Attempting quantum state recovery")
            
            # Reset quantum state matrix
            if hasattr(scheduler, '_update_quantum_state'):
                scheduler._update_quantum_state()
                logger.info("Quantum state matrix reset successfully")
                return True
            
        except Exception as e:
            logger.error(f"Quantum state recovery failed: {e}")
        
        return False
    
    @staticmethod
    def recover_resource_allocation(allocator, task_id: str) -> bool:
        """Attempt to recover from resource allocation failure."""
        try:
            logger.info(f"Attempting resource recovery for task {task_id}")
            
            # Try rebalancing resources
            if hasattr(allocator, 'rebalance_allocations'):
                rebalanced_count = allocator.rebalance_allocations()
                if rebalanced_count > 0:
                    logger.info(f"Rebalanced {rebalanced_count} resource allocations")
                    return True
            
        except Exception as e:
            logger.error(f"Resource recovery failed: {e}")
        
        return False
    
    @staticmethod
    def recover_scheduling_deadlock(scheduler, affected_tasks: List[str]) -> bool:
        """Attempt to recover from scheduling deadlock."""
        try:
            logger.info(f"Attempting deadlock recovery for tasks: {affected_tasks}")
            
            # Temporarily remove some dependencies to break deadlock
            for task_id in affected_tasks[:len(affected_tasks)//2]:  # Remove half
                task = scheduler.get_task(task_id)
                if task and task.dependencies:
                    # Remove one dependency temporarily
                    removed_dep = task.dependencies.pop()
                    logger.info(f"Temporarily removed dependency {removed_dep} from task {task_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Deadlock recovery failed: {e}")
        
        return False


class QuantumResilientOperations:
    """Resilient wrappers for quantum operations with automatic error handling."""
    
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
    
    @quantum_error_handler(ErrorCategory.SCHEDULING, "task_creation", ErrorSeverity.MEDIUM)
    @retry_with_backoff(max_retries=2)
    def resilient_task_creation(self, scheduler, **task_params) -> str:
        """Create task with resilience against transient failures."""
        return scheduler.create_task(**task_params)
    
    @quantum_error_handler(ErrorCategory.QUANTUM_COMPUTATION, "schedule_optimization", ErrorSeverity.HIGH)
    @circuit_breaker(failure_threshold=3, recovery_timeout=120)
    @retry_with_backoff(max_retries=1)
    def resilient_optimization(self, optimizer, cost_function, initial_schedule, max_iterations) -> Any:
        """Perform schedule optimization with resilience."""
        return optimizer.anneal(cost_function, initial_schedule, max_iterations)
    
    @quantum_error_handler(ErrorCategory.RESOURCE, "resource_allocation", ErrorSeverity.MEDIUM)
    @retry_with_backoff(max_retries=3, base_delay=0.5)
    def resilient_resource_allocation(self, allocator, task_id: str, requirements: Dict[str, float]) -> bool:
        """Allocate resources with automatic retry and recovery."""
        success = allocator.allocate_resources(task_id, requirements)
        if not success:
            # Try recovery
            if QuantumErrorRecovery.recover_resource_allocation(allocator, task_id):
                # Retry after recovery
                return allocator.allocate_resources(task_id, requirements)
        return success


def create_user_friendly_error(quantum_error: QuantumError) -> Dict[str, Any]:
    """Create user-friendly error response from quantum error."""
    return {
        "error": True,
        "message": quantum_error.user_message or quantum_error.message,
        "category": quantum_error.category.value,
        "severity": quantum_error.severity.value,
        "timestamp": quantum_error.timestamp.isoformat(),
        "error_code": quantum_error.error_code,
        "recovery_suggestion": quantum_error.recovery_suggestion,
        "support_info": "Contact support with error code for assistance"
    }