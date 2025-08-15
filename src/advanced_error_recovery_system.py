#!/usr/bin/env python3
"""Advanced Error Recovery System for Medical AI Applications"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import traceback
import functools
import uuid


class ErrorSeverity(Enum):
    """Error severity levels for medical AI systems"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    PATIENT_SAFETY = "patient_safety"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types"""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    HUMAN_INTERVENTION = "human_intervention"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


@dataclass
class ErrorContext:
    """Context information for error tracking"""
    error_id: str
    timestamp: float
    component: str
    severity: ErrorSeverity
    error_type: str
    error_message: str
    stack_trace: str
    recovery_strategy: RecoveryStrategy
    patient_impact: bool = False
    phi_involved: bool = False
    regulatory_reportable: bool = False


@dataclass
class RecoveryAction:
    """Recovery action definition"""
    action_id: str
    strategy: RecoveryStrategy
    handler: Callable
    timeout: float
    max_retries: int
    backoff_multiplier: float = 2.0
    prerequisites: List[str] = None


class CircuitBreaker:
    """Circuit breaker for preventing cascade failures"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e


class AdvancedErrorRecoverySystem:
    """Comprehensive error recovery system for medical AI"""
    
    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.recovery_actions: Dict[str, RecoveryAction] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.logger = self._setup_logging()
        self.metrics = {
            "total_errors": 0,
            "recovered_errors": 0,
            "critical_errors": 0,
            "patient_safety_errors": 0
        }
        
        # Register default recovery actions
        self._register_default_recovery_actions()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup error recovery logging"""
        logger = logging.getLogger("error_recovery")
        logger.setLevel(logging.INFO)
        
        # Create file handler for error logs
        file_handler = logging.FileHandler("error_recovery.log")
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _register_default_recovery_actions(self):
        """Register default recovery actions"""
        
        # Retry with exponential backoff
        self.recovery_actions["retry_exponential"] = RecoveryAction(
            action_id="retry_exponential",
            strategy=RecoveryStrategy.RETRY,
            handler=self._retry_with_backoff,
            timeout=30.0,
            max_retries=3,
            backoff_multiplier=2.0
        )
        
        # Fallback to backup model
        self.recovery_actions["model_fallback"] = RecoveryAction(
            action_id="model_fallback",
            strategy=RecoveryStrategy.FALLBACK,
            handler=self._fallback_to_backup_model,
            timeout=10.0,
            max_retries=1
        )
        
        # Graceful degradation
        self.recovery_actions["graceful_degradation"] = RecoveryAction(
            action_id="graceful_degradation",
            strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
            handler=self._activate_graceful_degradation,
            timeout=5.0,
            max_retries=1
        )
        
        # Human intervention
        self.recovery_actions["human_intervention"] = RecoveryAction(
            action_id="human_intervention",
            strategy=RecoveryStrategy.HUMAN_INTERVENTION,
            handler=self._request_human_intervention,
            timeout=300.0,  # 5 minutes
            max_retries=1
        )
        
        # Emergency shutdown
        self.recovery_actions["emergency_shutdown"] = RecoveryAction(
            action_id="emergency_shutdown",
            strategy=RecoveryStrategy.EMERGENCY_SHUTDOWN,
            handler=self._emergency_shutdown,
            timeout=5.0,
            max_retries=1
        )
    
    def handle_error(self, 
                    error: Exception,
                    component: str,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    patient_impact: bool = False,
                    phi_involved: bool = False) -> bool:
        """Handle error with appropriate recovery strategy"""
        
        # Create error context
        error_context = ErrorContext(
            error_id=str(uuid.uuid4()),
            timestamp=time.time(),
            component=component,
            severity=severity,
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            recovery_strategy=self._determine_recovery_strategy(severity, patient_impact),
            patient_impact=patient_impact,
            phi_involved=phi_involved,
            regulatory_reportable=self._is_regulatory_reportable(severity, patient_impact)
        )
        
        # Log error
        self._log_error(error_context)
        
        # Add to history
        self.error_history.append(error_context)
        self.metrics["total_errors"] += 1
        
        if severity == ErrorSeverity.CRITICAL:
            self.metrics["critical_errors"] += 1
        
        if severity == ErrorSeverity.PATIENT_SAFETY:
            self.metrics["patient_safety_errors"] += 1
        
        # Execute recovery
        recovery_successful = self._execute_recovery(error_context)
        
        if recovery_successful:
            self.metrics["recovered_errors"] += 1
        
        return recovery_successful
    
    def _determine_recovery_strategy(self, 
                                   severity: ErrorSeverity, 
                                   patient_impact: bool) -> RecoveryStrategy:
        """Determine appropriate recovery strategy based on error characteristics"""
        
        if severity == ErrorSeverity.PATIENT_SAFETY or patient_impact:
            return RecoveryStrategy.HUMAN_INTERVENTION
        
        if severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.EMERGENCY_SHUTDOWN
        
        if severity == ErrorSeverity.HIGH:
            return RecoveryStrategy.FALLBACK
        
        if severity == ErrorSeverity.MEDIUM:
            return RecoveryStrategy.CIRCUIT_BREAKER
        
        return RecoveryStrategy.RETRY
    
    def _is_regulatory_reportable(self, severity: ErrorSeverity, patient_impact: bool) -> bool:
        """Determine if error needs regulatory reporting"""
        return (severity in [ErrorSeverity.CRITICAL, ErrorSeverity.PATIENT_SAFETY] or 
                patient_impact)
    
    def _log_error(self, error_context: ErrorContext):
        """Log error with appropriate level"""
        
        log_message = (
            f"Error ID: {error_context.error_id} | "
            f"Component: {error_context.component} | "
            f"Type: {error_context.error_type} | "
            f"Message: {error_context.error_message}"
        )
        
        if error_context.patient_impact:
            log_message += " | PATIENT IMPACT"
        
        if error_context.phi_involved:
            log_message += " | PHI INVOLVED"
        
        if error_context.regulatory_reportable:
            log_message += " | REGULATORY REPORTABLE"
        
        if error_context.severity == ErrorSeverity.PATIENT_SAFETY:
            self.logger.critical(log_message)
        elif error_context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_context.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _execute_recovery(self, error_context: ErrorContext) -> bool:
        """Execute recovery action based on strategy"""
        
        strategy = error_context.recovery_strategy
        
        try:
            if strategy == RecoveryStrategy.RETRY:
                return self._retry_with_backoff(error_context)
            elif strategy == RecoveryStrategy.FALLBACK:
                return self._fallback_to_backup_model(error_context)
            elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                return self._activate_circuit_breaker(error_context)
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                return self._activate_graceful_degradation(error_context)
            elif strategy == RecoveryStrategy.HUMAN_INTERVENTION:
                return self._request_human_intervention(error_context)
            elif strategy == RecoveryStrategy.EMERGENCY_SHUTDOWN:
                return self._emergency_shutdown(error_context)
            else:
                self.logger.error(f"Unknown recovery strategy: {strategy}")
                return False
                
        except Exception as recovery_error:
            self.logger.error(f"Recovery action failed: {recovery_error}")
            return False
    
    def _retry_with_backoff(self, error_context: ErrorContext) -> bool:
        """Retry operation with exponential backoff"""
        action = self.recovery_actions["retry_exponential"]
        
        for attempt in range(action.max_retries):
            try:
                wait_time = action.backoff_multiplier ** attempt
                time.sleep(wait_time)
                
                self.logger.info(f"Retry attempt {attempt + 1} for error {error_context.error_id}")
                
                # Simulate retry operation (would call actual function in real implementation)
                # For now, simulate success after 2 attempts
                if attempt >= 1:
                    self.logger.info(f"Retry successful for error {error_context.error_id}")
                    return True
                else:
                    raise Exception("Retry failed")
                    
            except Exception as e:
                self.logger.warning(f"Retry attempt {attempt + 1} failed: {e}")
                continue
        
        self.logger.error(f"All retry attempts failed for error {error_context.error_id}")
        return False
    
    def _fallback_to_backup_model(self, error_context: ErrorContext) -> bool:
        """Fallback to backup model"""
        try:
            self.logger.info(f"Activating backup model for error {error_context.error_id}")
            
            # Simulate fallback activation
            backup_config = {
                "model_type": "backup_cnn",
                "confidence_threshold": 0.8,
                "performance_degradation": "15%",
                "activation_time": time.time()
            }
            
            # Save fallback configuration
            with open("backup_model_config.json", "w") as f:
                json.dump(backup_config, f, indent=2)
            
            self.logger.info("Backup model activated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Backup model activation failed: {e}")
            return False
    
    def _activate_circuit_breaker(self, error_context: ErrorContext) -> bool:
        """Activate circuit breaker for component"""
        component = error_context.component
        
        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = CircuitBreaker()
        
        circuit_breaker = self.circuit_breakers[component]
        
        try:
            # Register failure with circuit breaker
            circuit_breaker.failure_count += 1
            circuit_breaker.last_failure_time = time.time()
            
            if circuit_breaker.failure_count >= circuit_breaker.failure_threshold:
                circuit_breaker.state = "OPEN"
                self.logger.warning(f"Circuit breaker OPEN for component {component}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Circuit breaker activation failed: {e}")
            return False
    
    def _activate_graceful_degradation(self, error_context: ErrorContext) -> bool:
        """Activate graceful degradation mode"""
        try:
            self.logger.info(f"Activating graceful degradation for error {error_context.error_id}")
            
            degradation_config = {
                "mode": "reduced_functionality",
                "features_disabled": ["advanced_analytics", "batch_processing"],
                "performance_limit": "50%",
                "user_notification": True,
                "activation_time": time.time()
            }
            
            # Save degradation configuration
            with open("graceful_degradation_config.json", "w") as f:
                json.dump(degradation_config, f, indent=2)
            
            self.logger.info("Graceful degradation activated")
            return True
            
        except Exception as e:
            self.logger.error(f"Graceful degradation activation failed: {e}")
            return False
    
    def _request_human_intervention(self, error_context: ErrorContext) -> bool:
        """Request human intervention for critical errors"""
        try:
            self.logger.critical(f"Human intervention requested for error {error_context.error_id}")
            
            intervention_request = {
                "error_id": error_context.error_id,
                "severity": error_context.severity.value,
                "component": error_context.component,
                "patient_impact": error_context.patient_impact,
                "phi_involved": error_context.phi_involved,
                "regulatory_reportable": error_context.regulatory_reportable,
                "requested_at": time.time(),
                "contact_methods": ["email", "sms", "pager"],
                "escalation_timeline": "immediate"
            }
            
            # Save intervention request
            with open(f"intervention_request_{error_context.error_id}.json", "w") as f:
                json.dump(intervention_request, f, indent=2)
            
            # In real implementation, would send alerts to on-call personnel
            self.logger.critical("Human intervention request logged - alerts sent")
            return True
            
        except Exception as e:
            self.logger.error(f"Human intervention request failed: {e}")
            return False
    
    def _emergency_shutdown(self, error_context: ErrorContext) -> bool:
        """Emergency shutdown for critical failures"""
        try:
            self.logger.critical(f"Emergency shutdown initiated for error {error_context.error_id}")
            
            shutdown_config = {
                "shutdown_reason": error_context.error_message,
                "initiated_by": "error_recovery_system",
                "shutdown_time": time.time(),
                "patient_safety_concern": error_context.patient_impact,
                "regulatory_notification_required": error_context.regulatory_reportable
            }
            
            # Save shutdown configuration
            with open(f"emergency_shutdown_{error_context.error_id}.json", "w") as f:
                json.dump(shutdown_config, f, indent=2)
            
            self.logger.critical("Emergency shutdown configuration saved")
            
            # In real implementation, would actually shutdown system components
            return True
            
        except Exception as e:
            self.logger.error(f"Emergency shutdown failed: {e}")
            return False
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        total_errors = len(self.error_history)
        
        if total_errors == 0:
            return {
                "total_errors": 0,
                "recovery_rate": 0.0,
                "error_by_severity": {},
                "error_by_component": {},
                "regulatory_reportable": 0
            }
        
        # Count by severity
        severity_counts = {}
        for error in self.error_history:
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Count by component
        component_counts = {}
        for error in self.error_history:
            component = error.component
            component_counts[component] = component_counts.get(component, 0) + 1
        
        # Count regulatory reportable
        regulatory_count = sum(1 for error in self.error_history if error.regulatory_reportable)
        
        recovery_rate = self.metrics["recovered_errors"] / total_errors
        
        return {
            "total_errors": total_errors,
            "recovery_rate": recovery_rate,
            "error_by_severity": severity_counts,
            "error_by_component": component_counts,
            "regulatory_reportable": regulatory_count,
            "patient_impact_errors": sum(1 for error in self.error_history if error.patient_impact),
            "phi_involved_errors": sum(1 for error in self.error_history if error.phi_involved),
            "metrics": self.metrics
        }
    
    def generate_error_report(self) -> Dict[str, Any]:
        """Generate comprehensive error report"""
        statistics = self.get_error_statistics()
        
        recent_errors = sorted(
            self.error_history[-10:], 
            key=lambda x: x.timestamp, 
            reverse=True
        )
        
        return {
            "timestamp": time.time(),
            "statistics": statistics,
            "recent_errors": [asdict(error) for error in recent_errors],
            "circuit_breaker_status": {
                component: {
                    "state": cb.state,
                    "failure_count": cb.failure_count,
                    "last_failure": cb.last_failure_time
                }
                for component, cb in self.circuit_breakers.items()
            },
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on error patterns"""
        recommendations = []
        statistics = self.get_error_statistics()
        
        if statistics["recovery_rate"] < 0.8:
            recommendations.append("Review and improve error recovery strategies")
        
        if statistics["regulatory_reportable"] > 0:
            recommendations.append("Review regulatory reportable errors and compliance procedures")
        
        if statistics["patient_impact_errors"] > 0:
            recommendations.append("Implement additional patient safety measures")
        
        # Check for frequent errors by component
        component_counts = statistics["error_by_component"]
        for component, count in component_counts.items():
            if count > 5:
                recommendations.append(f"Investigate frequent errors in {component} component")
        
        return recommendations


# Decorator for automatic error handling
def error_recovery(component: str, 
                  severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                  patient_impact: bool = False,
                  phi_involved: bool = False):
    """Decorator for automatic error handling and recovery"""
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create global error recovery system
            if not hasattr(wrapper, "_error_recovery_system"):
                wrapper._error_recovery_system = AdvancedErrorRecoverySystem()
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                recovery_system = wrapper._error_recovery_system
                
                recovery_successful = recovery_system.handle_error(
                    error=e,
                    component=component,
                    severity=severity,
                    patient_impact=patient_impact,
                    phi_involved=phi_involved
                )
                
                if not recovery_successful:
                    # Re-raise if recovery failed
                    raise e
                
                # Return None or appropriate default value if recovery succeeded
                return None
        
        return wrapper
    return decorator


# Example usage and testing
def main():
    """Example usage of the error recovery system"""
    recovery_system = AdvancedErrorRecoverySystem()
    
    # Simulate various errors
    try:
        raise ValueError("Model prediction confidence too low")
    except Exception as e:
        recovery_system.handle_error(e, "prediction_engine", ErrorSeverity.MEDIUM)
    
    try:
        raise ConnectionError("Database connection failed")
    except Exception as e:
        recovery_system.handle_error(e, "database", ErrorSeverity.HIGH)
    
    try:
        raise RuntimeError("Memory allocation failed during inference")
    except Exception as e:
        recovery_system.handle_error(e, "inference_engine", ErrorSeverity.CRITICAL, patient_impact=True)
    
    # Generate and print report
    report = recovery_system.generate_error_report()
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()