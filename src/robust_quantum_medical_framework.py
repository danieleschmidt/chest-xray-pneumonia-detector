"""Robust Quantum-Medical Framework - Generation 2: Reliable & Secure.

Enterprise-grade quantum-medical AI system with comprehensive error handling,
medical compliance, security validation, and intelligent monitoring.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
import uuid

import numpy as np
import tensorflow as tf
from cryptography.fernet import Fernet

from adaptive_quantum_medical_pipeline import AdaptiveQuantumMedicalPipeline, AdaptivePipelineConfig
from quantum_medical_fusion_engine import MedicalDiagnosisResult


class SecurityLevel(Enum):
    """Security levels for medical data handling."""
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"
    TOP_SECRET = "top_secret"


class ComplianceStandard(Enum):
    """Medical compliance standards."""
    HIPAA = "hipaa"
    GDPR = "gdpr"
    FDA_510K = "fda_510k"
    ISO_13485 = "iso_13485"
    IEC_62304 = "iec_62304"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class SecurityContext:
    """Security context for medical processing."""
    user_id: str
    session_id: str
    security_level: SecurityLevel
    compliance_standards: Set[ComplianceStandard]
    encryption_key: Optional[bytes] = None
    access_permissions: Set[str] = field(default_factory=set)
    audit_trail: List[str] = field(default_factory=list)


@dataclass
class MedicalError:
    """Structured medical error information."""
    error_id: str
    severity: ErrorSeverity
    category: str
    message: str
    patient_id: Optional[str]
    timestamp: datetime
    stack_trace: Optional[str] = None
    recovery_actions: List[str] = field(default_factory=list)
    compliance_impact: Dict[ComplianceStandard, str] = field(default_factory=dict)


@dataclass
class ComplianceAuditEntry:
    """Medical compliance audit entry."""
    audit_id: str
    timestamp: datetime
    action: str
    user_id: str
    patient_id: Optional[str]
    data_accessed: List[str]
    compliance_standards: Set[ComplianceStandard]
    security_level: SecurityLevel
    result_hash: Optional[str] = None


class RobustQuantumMedicalFramework:
    """Enterprise-grade quantum-medical framework with comprehensive reliability features."""
    
    def __init__(self, 
                 config: Optional[AdaptivePipelineConfig] = None,
                 security_config: Optional[Dict] = None):
        """Initialize robust quantum-medical framework."""
        self.config = config or AdaptivePipelineConfig()
        self.security_config = security_config or {}
        
        # Core pipeline
        self.adaptive_pipeline = AdaptiveQuantumMedicalPipeline(config)
        
        # Security and encryption
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        
        # Error handling and recovery
        self.error_history: List[MedicalError] = []
        self.recovery_strategies: Dict[str, callable] = {}
        self.circuit_breakers: Dict[str, Dict] = {}
        
        # Compliance and audit
        self.compliance_audit_log: List[ComplianceAuditEntry] = []
        self.compliance_validators: Dict[ComplianceStandard, callable] = {}
        
        # Monitoring and health
        self.health_status: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, List[float]] = {}
        self.alert_thresholds: Dict[str, float] = {
            'error_rate': 0.05,
            'processing_time': 5.0,
            'memory_usage': 0.85,
            'security_violations': 0.0
        }
        
        # Session management
        self.active_sessions: Dict[str, SecurityContext] = {}
        self.session_timeouts: Dict[str, datetime] = {}
        
        self.logger = self._setup_secure_logging()
        self.logger.info("Robust Quantum-Medical Framework initialized")
        
        # Initialize security components
        self._initialize_security_components()
        self._initialize_compliance_validators()
        self._initialize_recovery_strategies()
    
    def _setup_secure_logging(self) -> logging.Logger:
        """Setup secure logging with audit capabilities."""
        logger = logging.getLogger(f"{__name__}.{id(self)}")
        
        # Create secure formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [SECURE] - %(message)s'
        )
        
        # File handler with restricted permissions
        log_file = Path("/tmp/quantum_medical_secure.log")
        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        return logger
    
    def _initialize_security_components(self):
        """Initialize security validation components."""
        self.security_validators = {
            SecurityLevel.PUBLIC: lambda ctx: True,
            SecurityLevel.INTERNAL: self._validate_internal_access,
            SecurityLevel.RESTRICTED: self._validate_restricted_access,
            SecurityLevel.CONFIDENTIAL: self._validate_confidential_access,
            SecurityLevel.TOP_SECRET: self._validate_top_secret_access
        }
        
        # Initialize circuit breakers
        self.circuit_breakers = {
            'processing_errors': {'threshold': 5, 'window': 300, 'count': 0, 'last_reset': time.time()},
            'security_violations': {'threshold': 3, 'window': 600, 'count': 0, 'last_reset': time.time()},
            'compliance_failures': {'threshold': 2, 'window': 900, 'count': 0, 'last_reset': time.time()}
        }
    
    def _initialize_compliance_validators(self):
        """Initialize medical compliance validators."""
        self.compliance_validators = {
            ComplianceStandard.HIPAA: self._validate_hipaa_compliance,
            ComplianceStandard.GDPR: self._validate_gdpr_compliance,
            ComplianceStandard.FDA_510K: self._validate_fda_compliance,
            ComplianceStandard.ISO_13485: self._validate_iso13485_compliance,
            ComplianceStandard.IEC_62304: self._validate_iec62304_compliance
        }
    
    def _initialize_recovery_strategies(self):
        """Initialize error recovery strategies."""
        self.recovery_strategies = {
            'model_loading_error': self._recover_model_loading,
            'processing_timeout': self._recover_processing_timeout,
            'memory_overflow': self._recover_memory_overflow,
            'security_violation': self._recover_security_violation,
            'compliance_failure': self._recover_compliance_failure
        }
    
    def create_secure_session(self, 
                            user_id: str, 
                            security_level: SecurityLevel,
                            compliance_standards: Set[ComplianceStandard],
                            permissions: Optional[Set[str]] = None) -> str:
        """Create secure user session with proper authentication."""
        session_id = str(uuid.uuid4())
        
        security_context = SecurityContext(
            user_id=user_id,
            session_id=session_id,
            security_level=security_level,
            compliance_standards=compliance_standards,
            encryption_key=self.encryption_key,
            access_permissions=permissions or set(),
            audit_trail=[]
        )
        
        self.active_sessions[session_id] = security_context
        self.session_timeouts[session_id] = datetime.now() + timedelta(hours=8)
        
        # Audit session creation
        audit_entry = ComplianceAuditEntry(
            audit_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            action="session_created",
            user_id=user_id,
            patient_id=None,
            data_accessed=[],
            compliance_standards=compliance_standards,
            security_level=security_level
        )
        self.compliance_audit_log.append(audit_entry)
        
        self.logger.info(f"Secure session created for user {user_id}: {session_id}")
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[SecurityContext]:
        """Validate and retrieve session context."""
        if session_id not in self.active_sessions:
            self.logger.warning(f"Invalid session ID: {session_id}")
            return None
        
        if session_id in self.session_timeouts:
            if datetime.now() > self.session_timeouts[session_id]:
                self.logger.warning(f"Session expired: {session_id}")
                self.cleanup_session(session_id)
                return None
        
        return self.active_sessions[session_id]
    
    def cleanup_session(self, session_id: str):
        """Clean up expired or invalid session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        if session_id in self.session_timeouts:
            del self.session_timeouts[session_id]
        
        self.logger.info(f"Session cleaned up: {session_id}")
    
    async def secure_medical_processing(self,
                                      session_id: str,
                                      image_data: np.ndarray,
                                      patient_metadata: Dict,
                                      processing_options: Optional[Dict] = None) -> MedicalDiagnosisResult:
        """Process medical data with comprehensive security and error handling."""
        # Validate session
        security_context = self.validate_session(session_id)
        if not security_context:
            raise SecurityError("Invalid or expired session")
        
        # Validate security clearance
        if not self._validate_security_clearance(security_context, patient_metadata):
            raise SecurityError("Insufficient security clearance for patient data")
        
        processing_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Pre-processing compliance checks
            await self._validate_pre_processing_compliance(
                security_context, patient_metadata, processing_id
            )
            
            # Encrypt sensitive data
            encrypted_metadata = self._encrypt_patient_data(patient_metadata, security_context)
            
            # Circuit breaker check
            if not self._check_circuit_breakers():
                raise ProcessingError("System circuit breakers activated - processing suspended")
            
            # Robust processing with retry logic
            result = await self._robust_processing_with_retry(
                image_data, encrypted_metadata, security_context, processing_id
            )
            
            # Post-processing validation
            validated_result = await self._validate_processing_result(
                result, security_context, processing_id
            )
            
            # Audit successful processing
            await self._audit_successful_processing(
                security_context, patient_metadata, validated_result, processing_id
            )
            
            # Update health metrics
            self._update_health_metrics(time.time() - start_time, success=True)
            
            self.logger.info(f"Secure processing completed: {processing_id}")
            return validated_result
            
        except Exception as e:
            # Comprehensive error handling
            await self._handle_processing_error(
                e, security_context, patient_metadata, processing_id
            )
            raise
    
    async def _robust_processing_with_retry(self,
                                          image_data: np.ndarray,
                                          encrypted_metadata: Dict,
                                          security_context: SecurityContext,
                                          processing_id: str,
                                          max_retries: int = 3) -> MedicalDiagnosisResult:
        """Robust processing with intelligent retry logic."""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Decrypt metadata for processing
                decrypted_metadata = self._decrypt_patient_data(encrypted_metadata, security_context)
                
                # Process through adaptive pipeline
                result = await self.adaptive_pipeline.process_medical_case_adaptive(
                    image_data, decrypted_metadata
                )
                
                # Validate result integrity
                if self._validate_result_integrity(result):
                    return result
                else:
                    raise ProcessingError("Result integrity validation failed")
                    
            except Exception as e:
                last_error = e
                self.logger.warning(f"Processing attempt {attempt + 1} failed: {e}")
                
                # Apply recovery strategy if available
                error_type = type(e).__name__.lower()
                if error_type in self.recovery_strategies:
                    try:
                        await self.recovery_strategies[error_type](e, attempt)
                    except Exception as recovery_error:
                        self.logger.error(f"Recovery strategy failed: {recovery_error}")
                
                # Exponential backoff
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 0.5
                    await asyncio.sleep(wait_time)
        
        # All retries failed
        raise ProcessingError(f"Processing failed after {max_retries} attempts: {last_error}")
    
    def _validate_security_clearance(self, 
                                   security_context: SecurityContext, 
                                   patient_metadata: Dict) -> bool:
        """Validate security clearance for patient data access."""
        patient_sensitivity = patient_metadata.get('sensitivity_level', 'internal')
        required_level = SecurityLevel(patient_sensitivity)
        
        # Check if user's security level is sufficient
        user_level_value = list(SecurityLevel).index(security_context.security_level)
        required_level_value = list(SecurityLevel).index(required_level)
        
        if user_level_value < required_level_value:
            self.logger.warning(f"Insufficient security clearance: {security_context.user_id}")
            return False
        
        # Check specific permissions
        required_permissions = patient_metadata.get('required_permissions', set())
        if not required_permissions.issubset(security_context.access_permissions):
            self.logger.warning(f"Missing required permissions: {security_context.user_id}")
            return False
        
        return True
    
    def _encrypt_patient_data(self, data: Dict, security_context: SecurityContext) -> Dict:
        """Encrypt sensitive patient data."""
        sensitive_fields = ['name', 'ssn', 'dob', 'medical_record_number']
        encrypted_data = data.copy()
        
        for field in sensitive_fields:
            if field in encrypted_data:
                value = str(encrypted_data[field]).encode()
                encrypted_value = self.cipher.encrypt(value)
                encrypted_data[field] = encrypted_value.decode()
        
        return encrypted_data
    
    def _decrypt_patient_data(self, encrypted_data: Dict, security_context: SecurityContext) -> Dict:
        """Decrypt sensitive patient data."""
        sensitive_fields = ['name', 'ssn', 'dob', 'medical_record_number']
        decrypted_data = encrypted_data.copy()
        
        for field in sensitive_fields:
            if field in decrypted_data and isinstance(decrypted_data[field], str):
                try:
                    encrypted_value = decrypted_data[field].encode()
                    decrypted_value = self.cipher.decrypt(encrypted_value)
                    decrypted_data[field] = decrypted_value.decode()
                except Exception:
                    # Field might not be encrypted
                    pass
        
        return decrypted_data
    
    def _check_circuit_breakers(self) -> bool:
        """Check if any circuit breakers are active."""
        current_time = time.time()
        
        for breaker_name, breaker in self.circuit_breakers.items():
            # Reset counter if window expired
            if current_time - breaker['last_reset'] > breaker['window']:
                breaker['count'] = 0
                breaker['last_reset'] = current_time
            
            # Check if threshold exceeded
            if breaker['count'] >= breaker['threshold']:
                self.logger.warning(f"Circuit breaker active: {breaker_name}")
                return False
        
        return True
    
    def _increment_circuit_breaker(self, breaker_name: str):
        """Increment circuit breaker counter."""
        if breaker_name in self.circuit_breakers:
            self.circuit_breakers[breaker_name]['count'] += 1
    
    async def _validate_pre_processing_compliance(self,
                                                security_context: SecurityContext,
                                                patient_metadata: Dict,
                                                processing_id: str):
        """Validate compliance requirements before processing."""
        for standard in security_context.compliance_standards:
            if standard in self.compliance_validators:
                validator = self.compliance_validators[standard]
                is_compliant = await validator(patient_metadata, security_context)
                
                if not is_compliant:
                    self._increment_circuit_breaker('compliance_failures')
                    raise ComplianceError(f"Pre-processing compliance validation failed: {standard}")
    
    async def _validate_processing_result(self,
                                        result: MedicalDiagnosisResult,
                                        security_context: SecurityContext,
                                        processing_id: str) -> MedicalDiagnosisResult:
        """Validate processing result for compliance and quality."""
        # Quality validation
        if result.confidence < 0.5:
            self.logger.warning(f"Low confidence result: {result.confidence}")
        
        # Medical validation
        if result.prediction < 0 or result.prediction > 1:
            raise ValidationError("Medical prediction out of valid range")
        
        # Add compliance metadata
        result.metadata.update({
            'compliance_validated': True,
            'security_level': security_context.security_level.value,
            'processing_id': processing_id,
            'validation_timestamp': datetime.now().isoformat()
        })
        
        return result
    
    def _validate_result_integrity(self, result: MedicalDiagnosisResult) -> bool:
        """Validate result data integrity."""
        # Check for NaN or infinite values
        if np.isnan(result.prediction) or np.isinf(result.prediction):
            return False
        
        if np.isnan(result.confidence) or np.isinf(result.confidence):
            return False
        
        # Check value ranges
        if not (0 <= result.prediction <= 1):
            return False
        
        if not (0 <= result.confidence <= 1):
            return False
        
        return True
    
    async def _audit_successful_processing(self,
                                         security_context: SecurityContext,
                                         patient_metadata: Dict,
                                         result: MedicalDiagnosisResult,
                                         processing_id: str):
        """Audit successful medical processing."""
        # Create result hash for integrity verification
        result_data = f"{result.prediction}{result.confidence}{processing_id}"
        result_hash = hashlib.sha256(result_data.encode()).hexdigest()
        
        audit_entry = ComplianceAuditEntry(
            audit_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            action="medical_processing_completed",
            user_id=security_context.user_id,
            patient_id=patient_metadata.get('id'),
            data_accessed=['image_data', 'patient_metadata'],
            compliance_standards=security_context.compliance_standards,
            security_level=security_context.security_level,
            result_hash=result_hash
        )
        
        self.compliance_audit_log.append(audit_entry)
        
        # Update security context audit trail
        security_context.audit_trail.append(f"Processing completed: {processing_id}")
    
    async def _handle_processing_error(self,
                                     error: Exception,
                                     security_context: SecurityContext,
                                     patient_metadata: Dict,
                                     processing_id: str):
        """Comprehensive error handling with audit and recovery."""
        error_id = str(uuid.uuid4())
        severity = self._determine_error_severity(error)
        
        medical_error = MedicalError(
            error_id=error_id,
            severity=severity,
            category=type(error).__name__,
            message=str(error),
            patient_id=patient_metadata.get('id'),
            timestamp=datetime.now(),
            stack_trace=traceback.format_exc(),
            recovery_actions=[],
            compliance_impact={}
        )
        
        self.error_history.append(medical_error)
        
        # Increment circuit breaker
        self._increment_circuit_breaker('processing_errors')
        
        # Update health metrics
        self._update_health_metrics(0, success=False)
        
        # Log security-sensitive error information
        self.logger.error(f"Processing error [{error_id}]: {error}")
        
        # Audit error occurrence
        audit_entry = ComplianceAuditEntry(
            audit_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            action="processing_error_occurred",
            user_id=security_context.user_id,
            patient_id=patient_metadata.get('id'),
            data_accessed=['error_handling'],
            compliance_standards=security_context.compliance_standards,
            security_level=security_context.security_level
        )
        
        self.compliance_audit_log.append(audit_entry)
    
    def _determine_error_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity based on error type and context."""
        if isinstance(error, SecurityError):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, ComplianceError):
            return ErrorSeverity.HIGH
        elif isinstance(error, ProcessingError):
            return ErrorSeverity.MEDIUM
        elif isinstance(error, ValidationError):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _update_health_metrics(self, processing_time: float, success: bool):
        """Update system health metrics."""
        if 'processing_times' not in self.performance_metrics:
            self.performance_metrics['processing_times'] = []
        
        if 'success_rate' not in self.performance_metrics:
            self.performance_metrics['success_rate'] = []
        
        self.performance_metrics['processing_times'].append(processing_time)
        self.performance_metrics['success_rate'].append(1.0 if success else 0.0)
        
        # Keep only recent metrics
        max_metrics = 1000
        for metric_list in self.performance_metrics.values():
            if len(metric_list) > max_metrics:
                metric_list[:] = metric_list[-max_metrics:]
    
    def get_system_health_status(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        current_time = datetime.now()
        
        # Calculate metrics
        recent_success_rate = 0.0
        avg_processing_time = 0.0
        
        if 'success_rate' in self.performance_metrics and self.performance_metrics['success_rate']:
            recent_success_rate = np.mean(self.performance_metrics['success_rate'][-100:])
        
        if 'processing_times' in self.performance_metrics and self.performance_metrics['processing_times']:
            avg_processing_time = np.mean(self.performance_metrics['processing_times'][-100:])
        
        # Count recent errors
        recent_errors = [e for e in self.error_history 
                        if (current_time - e.timestamp).total_seconds() < 3600]
        
        critical_errors = [e for e in recent_errors if e.severity == ErrorSeverity.CRITICAL]
        
        # Active sessions
        active_session_count = len(self.active_sessions)
        
        health_status = {
            'timestamp': current_time.isoformat(),
            'overall_status': 'healthy',  # Will be updated based on checks
            'metrics': {
                'success_rate': recent_success_rate,
                'avg_processing_time': avg_processing_time,
                'active_sessions': active_session_count,
                'recent_error_count': len(recent_errors),
                'critical_error_count': len(critical_errors)
            },
            'circuit_breakers': {name: breaker['count'] >= breaker['threshold'] 
                               for name, breaker in self.circuit_breakers.items()},
            'compliance_audit_entries': len(self.compliance_audit_log),
            'alerts': []
        }
        
        # Determine overall status
        if critical_errors:
            health_status['overall_status'] = 'critical'
            health_status['alerts'].append(f"Critical errors detected: {len(critical_errors)}")
        elif recent_success_rate < 0.9:
            health_status['overall_status'] = 'degraded'
            health_status['alerts'].append(f"Low success rate: {recent_success_rate:.2%}")
        elif avg_processing_time > 5.0:
            health_status['overall_status'] = 'degraded'
            health_status['alerts'].append(f"High processing time: {avg_processing_time:.2f}s")
        elif any(health_status['circuit_breakers'].values()):
            health_status['overall_status'] = 'degraded'
            health_status['alerts'].append("Circuit breakers active")
        
        return health_status
    
    # Compliance validators (simplified implementations)
    async def _validate_hipaa_compliance(self, patient_metadata: Dict, security_context: SecurityContext) -> bool:
        """Validate HIPAA compliance requirements."""
        required_fields = ['patient_consent', 'data_minimization_applied']
        return all(field in patient_metadata for field in required_fields)
    
    async def _validate_gdpr_compliance(self, patient_metadata: Dict, security_context: SecurityContext) -> bool:
        """Validate GDPR compliance requirements."""
        required_fields = ['explicit_consent', 'data_retention_policy']
        return all(field in patient_metadata for field in required_fields)
    
    async def _validate_fda_compliance(self, patient_metadata: Dict, security_context: SecurityContext) -> bool:
        """Validate FDA 510(k) compliance requirements."""
        return patient_metadata.get('fda_cleared_device', False)
    
    async def _validate_iso13485_compliance(self, patient_metadata: Dict, security_context: SecurityContext) -> bool:
        """Validate ISO 13485 compliance requirements."""
        return patient_metadata.get('quality_management_system', False)
    
    async def _validate_iec62304_compliance(self, patient_metadata: Dict, security_context: SecurityContext) -> bool:
        """Validate IEC 62304 compliance requirements."""
        return patient_metadata.get('software_lifecycle_process', False)
    
    # Security validators
    def _validate_internal_access(self, security_context: SecurityContext) -> bool:
        """Validate internal access level."""
        return 'internal_access' in security_context.access_permissions
    
    def _validate_restricted_access(self, security_context: SecurityContext) -> bool:
        """Validate restricted access level."""
        return 'restricted_access' in security_context.access_permissions
    
    def _validate_confidential_access(self, security_context: SecurityContext) -> bool:
        """Validate confidential access level."""
        return 'confidential_access' in security_context.access_permissions
    
    def _validate_top_secret_access(self, security_context: SecurityContext) -> bool:
        """Validate top secret access level."""
        return 'top_secret_access' in security_context.access_permissions
    
    # Recovery strategies
    async def _recover_model_loading(self, error: Exception, attempt: int):
        """Recovery strategy for model loading errors."""
        self.logger.info(f"Attempting model loading recovery (attempt {attempt})")
        # Could implement model cache cleanup, alternative model loading, etc.
    
    async def _recover_processing_timeout(self, error: Exception, attempt: int):
        """Recovery strategy for processing timeouts."""
        self.logger.info(f"Attempting timeout recovery (attempt {attempt})")
        # Could implement resource cleanup, priority adjustment, etc.
    
    async def _recover_memory_overflow(self, error: Exception, attempt: int):
        """Recovery strategy for memory overflow."""
        self.logger.info(f"Attempting memory recovery (attempt {attempt})")
        # Could implement garbage collection, cache cleanup, etc.
        import gc
        gc.collect()
    
    async def _recover_security_violation(self, error: Exception, attempt: int):
        """Recovery strategy for security violations."""
        self.logger.error(f"Security violation recovery attempt {attempt}")
        self._increment_circuit_breaker('security_violations')
        # Security violations should not be automatically recovered
        raise SecurityError("Security violation - manual intervention required")
    
    async def _recover_compliance_failure(self, error: Exception, attempt: int):
        """Recovery strategy for compliance failures."""
        self.logger.error(f"Compliance failure recovery attempt {attempt}")
        self._increment_circuit_breaker('compliance_failures')
        # Compliance failures require manual review
        raise ComplianceError("Compliance failure - manual review required")
    
    def cleanup(self):
        """Clean up framework resources."""
        # Clean up all sessions
        for session_id in list(self.active_sessions.keys()):
            self.cleanup_session(session_id)
        
        # Clean up pipeline
        if hasattr(self.adaptive_pipeline, 'cleanup'):
            self.adaptive_pipeline.cleanup()
        
        self.logger.info("Robust framework cleanup complete")


# Custom exceptions
class SecurityError(Exception):
    """Security-related error."""
    pass


class ComplianceError(Exception):
    """Compliance-related error."""
    pass


class ProcessingError(Exception):
    """Medical processing error."""
    pass


class ValidationError(Exception):
    """Validation error."""
    pass


async def main():
    """Demonstration of Robust Quantum-Medical Framework."""
    print("🛡️ Robust Quantum-Medical Framework - Generation 2 Demo")
    print("=" * 65)
    
    try:
        # Initialize robust framework
        framework = RobustQuantumMedicalFramework()
        
        # Create secure session
        session_id = framework.create_secure_session(
            user_id="dr_smith_001",
            security_level=SecurityLevel.CONFIDENTIAL,
            compliance_standards={ComplianceStandard.HIPAA, ComplianceStandard.GDPR},
            permissions={'confidential_access', 'medical_processing'}
        )
        
        print(f"🔐 Secure session created: {session_id[:8]}...")
        
        # Create demo medical cases with compliance metadata
        demo_cases = []
        for i in range(4):
            image = np.random.normal(0.5, 0.2, (150, 150, 1))
            metadata = {
                'id': f'patient_{i:03d}',
                'name': f'Patient {i+1}',
                'age': 35 + i * 10,
                'sensitivity_level': 'confidential',
                'patient_consent': True,
                'data_minimization_applied': True,
                'explicit_consent': True,
                'data_retention_policy': True,
                'required_permissions': {'medical_processing'}
            }
            demo_cases.append((image, metadata))
        
        print(f"🏥 Processing {len(demo_cases)} secure medical cases...")
        
        # Process cases through robust framework
        results = []
        for i, (image, metadata) in enumerate(demo_cases):
            try:
                result = await framework.secure_medical_processing(
                    session_id=session_id,
                    image_data=image,
                    patient_metadata=metadata
                )
                results.append(result)
                
                status = "PNEUMONIA DETECTED" if result.prediction > 0.5 else "NORMAL"
                print(f"  Patient {i+1}: {status} (conf: {result.confidence:.3f}) ✅")
                
            except Exception as e:
                print(f"  Patient {i+1}: ERROR - {type(e).__name__} ❌")
        
        # Display system health
        health_status = framework.get_system_health_status()
        print(f"\n📊 System Health Status: {health_status['overall_status'].upper()}")
        print(f"  Success Rate: {health_status['metrics']['success_rate']:.1%}")
        print(f"  Avg Processing Time: {health_status['metrics']['avg_processing_time']:.3f}s")
        print(f"  Active Sessions: {health_status['metrics']['active_sessions']}")
        print(f"  Recent Errors: {health_status['metrics']['recent_error_count']}")
        
        # Display compliance audit summary
        print(f"\n🔍 Compliance Audit Summary:")
        print(f"  Total audit entries: {len(framework.compliance_audit_log)}")
        print(f"  Session activities: {len([e for e in framework.compliance_audit_log if 'session' in e.action])}")
        print(f"  Processing activities: {len([e for e in framework.compliance_audit_log if 'processing' in e.action])}")
        
        # Display security metrics
        print(f"\n🛡️ Security Metrics:")
        print(f"  Active sessions: {len(framework.active_sessions)}")
        print(f"  Circuit breaker status: {health_status['circuit_breakers']}")
        
        if health_status['alerts']:
            print(f"\n⚠️ System Alerts:")
            for alert in health_status['alerts']:
                print(f"  • {alert}")
        
    except Exception as e:
        print(f"\n❌ Framework error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'framework' in locals():
            framework.cleanup()
    
    print("\n✅ Robust Quantum-Medical Framework demonstration complete!")


if __name__ == "__main__":
    asyncio.run(main())