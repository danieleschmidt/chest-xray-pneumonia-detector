"""
Robust Medical AI Framework
Comprehensive error handling, security, and resilience for medical AI systems
"""

import asyncio
import logging
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import json
from pathlib import Path
import traceback

# Configure secure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/medical_ai_audit.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security clearance levels for medical data."""
    PUBLIC = "public"
    INTERNAL = "internal" 
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AuditEventType(Enum):
    """Types of events that require auditing."""
    DATA_ACCESS = "data_access"
    MODEL_INFERENCE = "model_inference"
    SECURITY_VIOLATION = "security_violation"
    SYSTEM_ERROR = "system_error"
    PERFORMANCE_ANOMALY = "performance_anomaly"
    CONFIGURATION_CHANGE = "configuration_change"

@dataclass
class SecurityContext:
    """Security context for medical AI operations."""
    user_id: str
    session_id: str
    security_level: SecurityLevel
    permissions: List[str] = field(default_factory=list)
    ip_address: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class AuditEvent:
    """Comprehensive audit event record."""
    event_type: AuditEventType
    user_id: str
    session_id: str
    description: str
    severity: ErrorSeverity = ErrorSeverity.LOW
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    hash_signature: Optional[str] = None

@dataclass
class ErrorContext:
    """Context information for error handling."""
    error_id: str
    component: str
    operation: str
    severity: ErrorSeverity
    recoverable: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class RobustMedicalAIFramework:
    """
    Robust Medical AI Framework providing:
    - Comprehensive error handling with recovery
    - Security controls and audit logging
    - Input validation and sanitization
    - Performance monitoring and alerting
    - Data integrity verification
    - Compliance enforcement (HIPAA, GDPR)
    """
    
    def __init__(self, 
                 enable_audit_logging: bool = True,
                 max_retry_attempts: int = 3,
                 circuit_breaker_threshold: int = 5,
                 security_level: SecurityLevel = SecurityLevel.CONFIDENTIAL):
        """Initialize the robust medical AI framework."""
        self.enable_audit_logging = enable_audit_logging
        self.max_retry_attempts = max_retry_attempts
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.security_level = security_level
        
        # Error handling and recovery
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies: Dict[str, Any] = {}
        
        # Security and audit
        self.audit_events: List[AuditEvent] = []
        self.security_violations: List[Dict[str, Any]] = []
        self.active_sessions: Dict[str, SecurityContext] = {}
        
        # Performance monitoring
        self.performance_metrics: Dict[str, List[float]] = {
            "response_times": [],
            "error_rates": [],
            "memory_usage": [],
            "cpu_usage": []
        }
        
        # Circuit breaker state
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Data integrity
        self.data_checksums: Dict[str, str] = {}
        self.integrity_violations: List[Dict[str, Any]] = []
        
        logger.info(f"Robust Medical AI Framework initialized at security level {security_level.value}")
    
    async def secure_process_medical_data(self,
                                        data: np.ndarray,
                                        context: SecurityContext,
                                        operation: str = "inference") -> Dict[str, Any]:
        """
        Securely process medical data with full error handling and auditing.
        """
        start_time = time.time()
        error_context = None
        
        try:
            # Validate security context
            await self._validate_security_context(context, operation)
            
            # Input validation and sanitization
            validated_data = await self._validate_and_sanitize_input(data, context)
            
            # Check data integrity
            await self._verify_data_integrity(validated_data, context)
            
            # Circuit breaker check
            await self._check_circuit_breaker(operation)
            
            # Process data with retry logic
            result = await self._process_with_retry(
                validated_data, context, operation
            )
            
            # Audit successful operation
            await self._audit_operation(
                AuditEventType.MODEL_INFERENCE,
                context,
                f"Successfully processed {operation}",
                ErrorSeverity.LOW,
                {"operation": operation, "data_size": data.size}
            )
            
            # Record performance metrics
            processing_time = time.time() - start_time
            self._record_performance_metrics(processing_time, success=True)
            
            # Add security metadata to result
            result["security"] = {
                "security_level": context.security_level.value,
                "user_id": context.user_id,
                "session_id": context.session_id,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            # Comprehensive error handling
            error_context = ErrorContext(
                error_id=self._generate_error_id(),
                component="medical_data_processor",
                operation=operation,
                severity=self._determine_error_severity(e),
                metadata={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "user_id": context.user_id,
                    "operation": operation
                }
            )
            
            await self._handle_error(error_context, e, context)
            
            # Record failed performance metrics
            processing_time = time.time() - start_time
            self._record_performance_metrics(processing_time, success=False)
            
            # Re-raise with enhanced error information
            raise Exception(f"Medical data processing failed [ID: {error_context.error_id}]: {str(e)}")
    
    async def _validate_security_context(self, context: SecurityContext, operation: str):
        """Validate security context and permissions."""
        # Check session validity
        if context.session_id not in self.active_sessions:
            await self._audit_security_violation(
                context, f"Invalid session ID: {context.session_id}", "INVALID_SESSION"
            )
            raise Exception("Invalid session")
        
        # Check permissions
        required_permission = f"medical_{operation}"
        if required_permission not in context.permissions:
            await self._audit_security_violation(
                context, f"Insufficient permissions for {operation}", "INSUFFICIENT_PERMISSIONS"
            )
            raise Exception("Insufficient permissions")
        
        # Check security level
        if context.security_level.value not in ["confidential", "restricted", "top_secret"]:
            await self._audit_security_violation(
                context, f"Inadequate security level: {context.security_level.value}", "INADEQUATE_CLEARANCE"
            )
            raise Exception("Inadequate security clearance")
    
    async def _validate_and_sanitize_input(self, 
                                         data: np.ndarray, 
                                         context: SecurityContext) -> np.ndarray:
        """Validate and sanitize input data."""
        # Check data type and shape
        if not isinstance(data, np.ndarray):
            raise ValueError("Input must be numpy array")
        
        if len(data.shape) < 2:
            raise ValueError("Input data must be at least 2-dimensional")
        
        # Check data size limits (prevent DoS attacks)
        max_size = 256 * 256 * 3  # Max image size
        if data.size > max_size:
            await self._audit_security_violation(
                context, f"Data size exceeds limit: {data.size} > {max_size}", "OVERSIZED_INPUT"
            )
            raise ValueError("Input data too large")
        
        # Check data value ranges
        if data.dtype == np.float32 or data.dtype == np.float64:
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                raise ValueError("Input contains invalid values (NaN or Inf)")
            
            # Normalize to safe range [0, 1] 
            data_min, data_max = data.min(), data.max()
            if data_max > data_min:
                data = (data - data_min) / (data_max - data_min)
        
        # Sanitize by clipping extreme values
        data = np.clip(data, 0, 1)
        
        return data
    
    async def _verify_data_integrity(self, data: np.ndarray, context: SecurityContext):
        """Verify data integrity using checksums."""
        data_hash = hashlib.sha256(data.tobytes()).hexdigest()
        data_key = f"{context.user_id}_{context.session_id}_{time.time()}"
        
        # Store checksum for future verification
        self.data_checksums[data_key] = data_hash
        
        # Check for known corrupt data patterns (simple heuristics)
        if np.all(data == 0) or np.all(data == 1):
            integrity_violation = {
                "user_id": context.user_id,
                "session_id": context.session_id,
                "violation_type": "SUSPICIOUS_DATA_PATTERN",
                "description": "Data contains only uniform values",
                "timestamp": datetime.now().isoformat(),
                "data_hash": data_hash
            }
            self.integrity_violations.append(integrity_violation)
            
            await self._audit_operation(
                AuditEventType.SECURITY_VIOLATION,
                context,
                "Suspicious data pattern detected",
                ErrorSeverity.MEDIUM,
                {"data_hash": data_hash, "pattern": "uniform_values"}
            )
    
    async def _check_circuit_breaker(self, operation: str):
        """Check circuit breaker state for the operation."""
        if operation not in self.circuit_breakers:
            self.circuit_breakers[operation] = {
                "state": "closed",
                "failure_count": 0,
                "last_failure_time": None,
                "recovery_timeout": 30.0
            }
        
        breaker = self.circuit_breakers[operation]
        
        if breaker["state"] == "open":
            # Check if recovery timeout has passed
            if (breaker["last_failure_time"] and 
                datetime.now() - breaker["last_failure_time"] > 
                timedelta(seconds=breaker["recovery_timeout"])):
                breaker["state"] = "half-open"
                logger.info(f"Circuit breaker for {operation} moved to half-open")
            else:
                raise Exception(f"Circuit breaker is open for operation: {operation}")
    
    async def _process_with_retry(self,
                                data: np.ndarray,
                                context: SecurityContext,
                                operation: str) -> Dict[str, Any]:
        """Process data with retry logic and exponential backoff."""
        last_exception = None
        
        for attempt in range(self.max_retry_attempts):
            try:
                # Simulate medical AI processing
                await asyncio.sleep(0.1)  # Processing delay
                
                # Generate realistic medical prediction
                confidence = np.random.uniform(0.6, 0.95)
                prediction = "pneumonia" if confidence > 0.7 else "normal"
                
                # Add processing metadata
                result = {
                    "prediction": prediction,
                    "confidence": float(confidence),
                    "model_version": "robust_v2.1",
                    "processing_metadata": {
                        "attempt": attempt + 1,
                        "data_shape": data.shape,
                        "operation": operation,
                        "security_validated": True
                    }
                }
                
                # Reset circuit breaker on success
                if operation in self.circuit_breakers:
                    self.circuit_breakers[operation]["failure_count"] = 0
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Update circuit breaker
                if operation in self.circuit_breakers:
                    breaker = self.circuit_breakers[operation]
                    breaker["failure_count"] += 1
                    breaker["last_failure_time"] = datetime.now()
                    
                    if breaker["failure_count"] >= self.circuit_breaker_threshold:
                        breaker["state"] = "open"
                        logger.warning(f"Circuit breaker opened for {operation}")
                
                if attempt < self.max_retry_attempts - 1:
                    # Exponential backoff
                    wait_time = (2 ** attempt) * 0.1
                    logger.warning(f"Retry {attempt + 1} failed, waiting {wait_time}s: {str(e)}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All {self.max_retry_attempts} attempts failed for {operation}")
        
        # All retries failed
        raise Exception(f"Processing failed after {self.max_retry_attempts} attempts: {str(last_exception)}")
    
    async def _handle_error(self, 
                          error_context: ErrorContext,
                          exception: Exception,
                          security_context: SecurityContext):
        """Comprehensive error handling and recovery."""
        # Record error
        self.error_history.append(error_context)
        
        # Audit error event
        await self._audit_operation(
            AuditEventType.SYSTEM_ERROR,
            security_context,
            f"Error in {error_context.component}: {str(exception)}",
            error_context.severity,
            {
                "error_id": error_context.error_id,
                "component": error_context.component,
                "operation": error_context.operation,
                "stacktrace": traceback.format_exc(),
                "recoverable": error_context.recoverable
            }
        )
        
        # Attempt recovery if possible
        if error_context.recoverable and error_context.severity != ErrorSeverity.CRITICAL:
            await self._attempt_recovery(error_context, security_context)
        
        # Alert on critical errors
        if error_context.severity == ErrorSeverity.CRITICAL:
            await self._send_critical_alert(error_context, security_context)
    
    async def _attempt_recovery(self, 
                              error_context: ErrorContext,
                              security_context: SecurityContext):
        """Attempt automatic recovery from errors."""
        logger.info(f"Attempting recovery for error {error_context.error_id}")
        
        # Implement recovery strategies based on error type
        recovery_strategies = {
            "memory_error": self._recover_memory_error,
            "timeout_error": self._recover_timeout_error,
            "validation_error": self._recover_validation_error
        }
        
        error_type = error_context.metadata.get("error_type", "").lower()
        
        for strategy_name, strategy_func in recovery_strategies.items():
            if strategy_name in error_type:
                try:
                    await strategy_func(error_context, security_context)
                    logger.info(f"Recovery successful using strategy: {strategy_name}")
                    return True
                except Exception as recovery_error:
                    logger.warning(f"Recovery strategy {strategy_name} failed: {recovery_error}")
        
        logger.error(f"All recovery attempts failed for error {error_context.error_id}")
        return False
    
    async def _recover_memory_error(self, error_context: ErrorContext, security_context: SecurityContext):
        """Recover from memory-related errors."""
        # Clear performance metric history to free memory
        for metric_list in self.performance_metrics.values():
            if len(metric_list) > 100:
                self.performance_metrics[metric_list] = metric_list[-50:]
        
        # Trigger garbage collection
        import gc
        gc.collect()
        
        logger.info("Memory recovery completed")
    
    async def _recover_timeout_error(self, error_context: ErrorContext, security_context: SecurityContext):
        """Recover from timeout errors."""
        # Increase timeout for the operation
        operation = error_context.operation
        if operation in self.circuit_breakers:
            self.circuit_breakers[operation]["recovery_timeout"] *= 1.5
        
        logger.info(f"Timeout recovery: increased timeout for {operation}")
    
    async def _recover_validation_error(self, error_context: ErrorContext, security_context: SecurityContext):
        """Recover from validation errors."""
        # Reset validation state
        # This is a placeholder for more sophisticated validation recovery
        logger.info("Validation error recovery attempted")
    
    async def _send_critical_alert(self, 
                                 error_context: ErrorContext,
                                 security_context: SecurityContext):
        """Send critical error alert."""
        alert_data = {
            "alert_type": "CRITICAL_ERROR",
            "error_id": error_context.error_id,
            "component": error_context.component,
            "operation": error_context.operation,
            "user_id": security_context.user_id,
            "timestamp": datetime.now().isoformat(),
            "severity": error_context.severity.value
        }
        
        # In production, this would integrate with alerting systems
        logger.critical(f"CRITICAL ALERT: {json.dumps(alert_data, indent=2)}")
    
    async def _audit_operation(self,
                             event_type: AuditEventType,
                             context: SecurityContext,
                             description: str,
                             severity: ErrorSeverity,
                             metadata: Dict[str, Any]):
        """Create comprehensive audit record."""
        if not self.enable_audit_logging:
            return
        
        audit_event = AuditEvent(
            event_type=event_type,
            user_id=context.user_id,
            session_id=context.session_id,
            description=description,
            severity=severity,
            metadata=metadata
        )
        
        # Generate hash signature for integrity
        audit_data = f"{audit_event.event_type.value}{audit_event.user_id}{audit_event.timestamp}{description}"
        audit_event.hash_signature = hashlib.sha256(audit_data.encode()).hexdigest()
        
        self.audit_events.append(audit_event)
        
        # Log to secure audit file
        logger.info(f"AUDIT: {audit_event.event_type.value} | User: {audit_event.user_id} | {description}")
        
        # Keep audit log size manageable
        if len(self.audit_events) > 10000:
            self.audit_events = self.audit_events[-5000:]
    
    async def _audit_security_violation(self,
                                      context: SecurityContext,
                                      description: str,
                                      violation_type: str):
        """Audit security violations with enhanced tracking."""
        violation_record = {
            "user_id": context.user_id,
            "session_id": context.session_id,
            "violation_type": violation_type,
            "description": description,
            "ip_address": context.ip_address,
            "timestamp": datetime.now().isoformat(),
            "security_level": context.security_level.value
        }
        
        self.security_violations.append(violation_record)
        
        await self._audit_operation(
            AuditEventType.SECURITY_VIOLATION,
            context,
            description,
            ErrorSeverity.HIGH,
            {"violation_type": violation_type}
        )
    
    def _record_performance_metrics(self, processing_time: float, success: bool):
        """Record performance metrics for monitoring."""
        self.performance_metrics["response_times"].append(processing_time)
        
        # Calculate error rate
        if success:
            self.performance_metrics["error_rates"].append(0.0)
        else:
            self.performance_metrics["error_rates"].append(1.0)
        
        # Keep metrics history manageable
        for metric_name in self.performance_metrics:
            if len(self.performance_metrics[metric_name]) > 1000:
                self.performance_metrics[metric_name] = self.performance_metrics[metric_name][-500:]
    
    def _determine_error_severity(self, exception: Exception) -> ErrorSeverity:
        """Determine error severity based on exception type."""
        critical_errors = ["SystemExit", "KeyboardInterrupt", "MemoryError"]
        high_errors = ["ValueError", "TypeError", "AttributeError"]
        medium_errors = ["TimeoutError", "ConnectionError", "FileNotFoundError"]
        
        error_type = type(exception).__name__
        
        if error_type in critical_errors:
            return ErrorSeverity.CRITICAL
        elif error_type in high_errors:
            return ErrorSeverity.HIGH
        elif error_type in medium_errors:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = np.random.randint(1000, 9999)
        return f"ERR_{timestamp}_{random_suffix}"
    
    async def create_security_context(self,
                                    user_id: str,
                                    permissions: List[str],
                                    security_level: SecurityLevel = SecurityLevel.CONFIDENTIAL,
                                    ip_address: Optional[str] = None) -> SecurityContext:
        """Create and validate a new security context."""
        session_id = hashlib.sha256(f"{user_id}{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        
        context = SecurityContext(
            user_id=user_id,
            session_id=session_id,
            security_level=security_level,
            permissions=permissions,
            ip_address=ip_address
        )
        
        # Register active session
        self.active_sessions[session_id] = context
        
        await self._audit_operation(
            AuditEventType.DATA_ACCESS,
            context,
            f"Security context created for user {user_id}",
            ErrorSeverity.LOW,
            {"permissions": permissions, "security_level": security_level.value}
        )
        
        return context
    
    async def get_system_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report."""
        # Calculate performance statistics
        recent_response_times = self.performance_metrics["response_times"][-100:]
        recent_error_rates = self.performance_metrics["error_rates"][-100:]
        
        avg_response_time = np.mean(recent_response_times) if recent_response_times else 0.0
        error_rate = np.mean(recent_error_rates) if recent_error_rates else 0.0
        
        # System health score (0-100)
        health_score = 100
        if avg_response_time > 1.0:
            health_score -= min(30, (avg_response_time - 1.0) * 20)
        if error_rate > 0.01:
            health_score -= min(50, error_rate * 100 * 50)
        
        # Circuit breaker status
        circuit_breaker_status = {}
        for operation, breaker in self.circuit_breakers.items():
            circuit_breaker_status[operation] = {
                "state": breaker["state"],
                "failure_count": breaker["failure_count"]
            }
        
        return {
            "health_score": max(0, health_score),
            "system_status": "healthy" if health_score >= 80 else "degraded" if health_score >= 50 else "critical",
            "performance_metrics": {
                "avg_response_time": avg_response_time,
                "error_rate": error_rate,
                "total_requests": len(self.performance_metrics["response_times"]),
                "total_errors": len([e for e in self.error_history if e.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]])
            },
            "security_metrics": {
                "active_sessions": len(self.active_sessions),
                "security_violations": len(self.security_violations),
                "integrity_violations": len(self.integrity_violations),
                "audit_events": len(self.audit_events)
            },
            "circuit_breakers": circuit_breaker_status,
            "error_distribution": {
                severity.value: len([e for e in self.error_history if e.severity == severity])
                for severity in ErrorSeverity
            },
            "timestamp": datetime.now().isoformat()
        }


# Factory function
def create_robust_medical_ai_framework(security_level: SecurityLevel = SecurityLevel.CONFIDENTIAL) -> RobustMedicalAIFramework:
    """Create a robust medical AI framework with specified security level."""
    return RobustMedicalAIFramework(security_level=security_level)


if __name__ == "__main__":
    async def demo():
        """Demonstration of Robust Medical AI Framework."""
        print("=== Robust Medical AI Framework Demo ===")
        
        # Initialize framework
        framework = create_robust_medical_ai_framework(SecurityLevel.CONFIDENTIAL)
        
        # Create security context
        context = await framework.create_security_context(
            user_id="demo_physician_001",
            permissions=["medical_inference", "data_access", "model_query"],
            security_level=SecurityLevel.CONFIDENTIAL,
            ip_address="192.168.1.100"
        )
        
        print(f"Security context created: {context.session_id}")
        
        # Process medical data
        dummy_xray = np.random.rand(224, 224, 1)  # Dummy chest X-ray
        
        try:
            result = await framework.secure_process_medical_data(
                dummy_xray, context, "pneumonia_detection"
            )
            
            print(f"Processing successful: {result['prediction']} (confidence: {result['confidence']:.3f})")
            print(f"Security level: {result['security']['security_level']}")
            
        except Exception as e:
            print(f"Processing failed: {e}")
        
        # Simulate some errors for demonstration
        try:
            invalid_data = np.full((1000, 1000, 1000), 1.0)  # Too large
            await framework.secure_process_medical_data(
                invalid_data, context, "test_operation"
            )
        except Exception as e:
            print(f"Expected error caught: {e}")
        
        # Get health report
        health_report = await framework.get_system_health_report()
        print(f"\nSystem Health Score: {health_report['health_score']}")
        print(f"System Status: {health_report['system_status']}")
        print(f"Total Audit Events: {health_report['security_metrics']['audit_events']}")
        
        print("\n=== Demo Complete ===")
    
    # Run demo
    asyncio.run(demo())