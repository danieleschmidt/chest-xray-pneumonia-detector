#!/usr/bin/env python3
"""
Advanced Security Framework - Generation 2: MAKE IT ROBUST
Comprehensive security with threat detection, encryption, and compliance monitoring.
"""

import asyncio
import json
import logging
import time
import hashlib
import secrets
import jwt
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import re

class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SecurityEvent(Enum):
    """Types of security events."""
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_DENIED = "authz_denied"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_ACCESS = "data_access"
    ENCRYPTION_EVENT = "encryption_event"
    VULNERABILITY_DETECTED = "vulnerability_detected"
    COMPLIANCE_VIOLATION = "compliance_violation"

@dataclass
class SecurityAlert:
    """Security alert definition."""
    id: str
    event_type: SecurityEvent
    threat_level: ThreatLevel
    source_ip: str
    user_id: Optional[str]
    description: str
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False

@dataclass
class EncryptionKey:
    """Encryption key metadata."""
    key_id: str
    algorithm: str
    created_at: float
    expires_at: Optional[float]
    key_data: bytes
    active: bool = True

class ThreatDetector:
    """AI-powered threat detection system."""
    
    def __init__(self):
        self.failed_attempts: Dict[str, List[float]] = {}
        self.suspicious_patterns = [
            r'(\.\./){3,}',  # Directory traversal
            r'(union|select|insert|delete|drop|create|alter)\s+',  # SQL injection
            r'<script[^>]*>.*?</script>',  # XSS
            r'eval\s*\(',  # Code injection
            r'(curl|wget|nc|netcat)\s+',  # Command injection
        ]
        self.rate_limits: Dict[str, List[float]] = {}
        
    def detect_brute_force(self, ip_address: str, user_id: Optional[str] = None) -> bool:
        """Detect brute force attacks."""
        key = f"{ip_address}:{user_id or 'unknown'}"
        current_time = time.time()
        
        if key not in self.failed_attempts:
            self.failed_attempts[key] = []
            
        # Add current attempt
        self.failed_attempts[key].append(current_time)
        
        # Remove attempts older than 15 minutes
        cutoff_time = current_time - 900
        self.failed_attempts[key] = [
            attempt for attempt in self.failed_attempts[key] 
            if attempt > cutoff_time
        ]
        
        # Check if threshold exceeded (5 attempts in 15 minutes)
        return len(self.failed_attempts[key]) >= 5
        
    def detect_suspicious_patterns(self, input_data: str) -> List[str]:
        """Detect suspicious patterns in input data."""
        detected_patterns = []
        
        for pattern in self.suspicious_patterns:
            if re.search(pattern, input_data, re.IGNORECASE):
                detected_patterns.append(pattern)
                
        return detected_patterns
        
    def detect_rate_limit_violation(self, ip_address: str, limit: int = 100, 
                                  window_seconds: int = 60) -> bool:
        """Detect rate limiting violations."""
        current_time = time.time()
        
        if ip_address not in self.rate_limits:
            self.rate_limits[ip_address] = []
            
        # Add current request
        self.rate_limits[ip_address].append(current_time)
        
        # Remove requests outside window
        cutoff_time = current_time - window_seconds
        self.rate_limits[ip_address] = [
            req_time for req_time in self.rate_limits[ip_address]
            if req_time > cutoff_time
        ]
        
        # Check if limit exceeded
        return len(self.rate_limits[ip_address]) > limit
        
    def analyze_user_behavior(self, user_id: str, actions: List[Dict]) -> Dict[str, Any]:
        """Analyze user behavior for anomalies."""
        if not actions:
            return {'anomalous': False}
            
        # Analyze request patterns
        timestamps = [action['timestamp'] for action in actions]
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            
            # Check for automated behavior (very regular intervals)
            if avg_interval < 1.0 and len(set(intervals)) == 1:
                return {
                    'anomalous': True,
                    'reason': 'automated_behavior',
                    'confidence': 0.9
                }
                
            # Check for unusual access patterns
            unique_endpoints = set(action.get('endpoint', '') for action in actions)
            if len(unique_endpoints) > 20:  # Accessing many different endpoints
                return {
                    'anomalous': True,
                    'reason': 'endpoint_scanning',
                    'confidence': 0.7
                }
                
        return {'anomalous': False}

class EncryptionManager:
    """Advanced encryption and key management."""
    
    def __init__(self):
        self.keys: Dict[str, EncryptionKey] = {}
        self.master_key = self._generate_master_key()
        
    def _generate_master_key(self) -> bytes:
        """Generate master encryption key."""
        return Fernet.generate_key()
        
    def create_encryption_key(self, algorithm: str = "AES-256", 
                            expires_in_hours: Optional[int] = None) -> str:
        """Create new encryption key."""
        key_id = secrets.token_urlsafe(32)
        
        if algorithm == "AES-256":
            key_data = Fernet.generate_key()
        elif algorithm == "RSA-2048":
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            key_data = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
            
        expires_at = None
        if expires_in_hours:
            expires_at = time.time() + (expires_in_hours * 3600)
            
        encryption_key = EncryptionKey(
            key_id=key_id,
            algorithm=algorithm,
            created_at=time.time(),
            expires_at=expires_at,
            key_data=key_data
        )
        
        self.keys[key_id] = encryption_key
        return key_id
        
    def encrypt_data(self, data: bytes, key_id: Optional[str] = None) -> Dict[str, Any]:
        """Encrypt data with specified or default key."""
        if key_id and key_id in self.keys:
            key = self.keys[key_id]
        else:
            # Use master key
            key_id = "master"
            key = EncryptionKey(
                key_id="master",
                algorithm="AES-256",
                created_at=time.time(),
                expires_at=None,
                key_data=self.master_key
            )
            
        if key.algorithm == "AES-256":
            fernet = Fernet(key.key_data)
            encrypted_data = fernet.encrypt(data)
        else:
            raise ValueError(f"Encryption not implemented for {key.algorithm}")
            
        return {
            'encrypted_data': base64.b64encode(encrypted_data).decode(),
            'key_id': key_id,
            'algorithm': key.algorithm,
            'timestamp': time.time()
        }
        
    def decrypt_data(self, encrypted_data: str, key_id: str) -> bytes:
        """Decrypt data with specified key."""
        if key_id == "master":
            key_data = self.master_key
        elif key_id in self.keys:
            key = self.keys[key_id]
            if key.expires_at and time.time() > key.expires_at:
                raise ValueError("Encryption key has expired")
            key_data = key.key_data
        else:
            raise ValueError("Encryption key not found")
            
        fernet = Fernet(key_data)
        encrypted_bytes = base64.b64decode(encrypted_data.encode())
        return fernet.decrypt(encrypted_bytes)
        
    def rotate_keys(self):
        """Rotate expired encryption keys."""
        current_time = time.time()
        expired_keys = []
        
        for key_id, key in self.keys.items():
            if key.expires_at and current_time > key.expires_at:
                key.active = False
                expired_keys.append(key_id)
                
        logging.info(f"Rotated {len(expired_keys)} expired keys")
        return expired_keys

class ComplianceMonitor:
    """HIPAA/SOC2/GDPR compliance monitoring."""
    
    def __init__(self):
        self.compliance_rules = {
            'data_encryption': {
                'required': True,
                'description': 'All PHI must be encrypted at rest and in transit'
            },
            'access_logging': {
                'required': True,
                'description': 'All PHI access must be logged with user identification'
            },
            'data_retention': {
                'max_days': 2555,  # 7 years for HIPAA
                'description': 'PHI must be retained for regulatory period'
            },
            'user_authentication': {
                'mfa_required': True,
                'session_timeout': 3600,  # 1 hour
                'description': 'Strong authentication required for PHI access'
            }
        }
        self.violations: List[Dict] = []
        
    def check_data_access_compliance(self, access_event: Dict[str, Any]) -> List[str]:
        """Check data access for compliance violations."""
        violations = []
        
        # Check if access is logged
        required_fields = ['user_id', 'timestamp', 'data_type', 'action']
        missing_fields = [field for field in required_fields 
                         if field not in access_event]
        
        if missing_fields:
            violations.append(f"Missing required audit fields: {missing_fields}")
            
        # Check if PHI access has proper encryption
        if access_event.get('data_type') == 'PHI':
            if not access_event.get('encrypted', False):
                violations.append("PHI accessed without encryption")
                
            if not access_event.get('user_authenticated', False):
                violations.append("PHI accessed without proper authentication")
                
        # Check session validity
        if access_event.get('session_age', 0) > self.compliance_rules['user_authentication']['session_timeout']:
            violations.append("Session timeout exceeded for PHI access")
            
        return violations
        
    def check_data_retention_compliance(self, data_records: List[Dict]) -> List[str]:
        """Check data retention compliance."""
        violations = []
        current_time = time.time()
        max_age = self.compliance_rules['data_retention']['max_days'] * 24 * 3600
        
        for record in data_records:
            age = current_time - record.get('created_at', 0)
            if age > max_age:
                violations.append(f"Data record {record.get('id')} exceeds retention period")
                
        return violations
        
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        return {
            'timestamp': time.time(),
            'compliance_status': 'compliant' if not self.violations else 'violations_detected',
            'total_violations': len(self.violations),
            'violations_by_type': self._group_violations_by_type(),
            'recommendations': self._generate_recommendations(),
            'next_audit_date': time.time() + (30 * 24 * 3600)  # 30 days
        }
        
    def _group_violations_by_type(self) -> Dict[str, int]:
        """Group violations by type for reporting."""
        violation_types = {}
        for violation in self.violations:
            violation_type = violation.get('type', 'unknown')
            violation_types[violation_type] = violation_types.get(violation_type, 0) + 1
        return violation_types
        
    def _generate_recommendations(self) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []
        
        if self.violations:
            recommendations.append("Review and address all compliance violations")
            recommendations.append("Implement automated compliance monitoring")
            recommendations.append("Conduct staff training on compliance requirements")
            
        recommendations.extend([
            "Regular security audits and penetration testing",
            "Update incident response procedures",
            "Review data access controls quarterly"
        ])
        
        return recommendations

class SecurityOrchestrator:
    """Main security framework orchestrator."""
    
    def __init__(self):
        self.threat_detector = ThreatDetector()
        self.encryption_manager = EncryptionManager()
        self.compliance_monitor = ComplianceMonitor()
        self.security_alerts: List[SecurityAlert] = []
        self.blocked_ips: Set[str] = set()
        self.jwt_secret = secrets.token_urlsafe(64)
        
    async def validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive request validation."""
        ip_address = request_data.get('ip_address', 'unknown')
        user_id = request_data.get('user_id')
        input_data = request_data.get('input', '')
        
        validation_result = {
            'allowed': True,
            'threat_level': ThreatLevel.LOW,
            'warnings': [],
            'actions_taken': []
        }
        
        # Check if IP is blocked
        if ip_address in self.blocked_ips:
            validation_result['allowed'] = False
            validation_result['threat_level'] = ThreatLevel.HIGH
            validation_result['warnings'].append("IP address is blocked")
            return validation_result
            
        # Check for brute force attacks
        if self.threat_detector.detect_brute_force(ip_address, user_id):
            await self._handle_brute_force(ip_address, user_id)
            validation_result['threat_level'] = ThreatLevel.HIGH
            validation_result['warnings'].append("Brute force attack detected")
            
        # Check for suspicious patterns
        suspicious_patterns = self.threat_detector.detect_suspicious_patterns(input_data)
        if suspicious_patterns:
            validation_result['threat_level'] = ThreatLevel.MEDIUM
            validation_result['warnings'].append(f"Suspicious patterns detected: {suspicious_patterns}")
            
        # Check rate limits
        if self.threat_detector.detect_rate_limit_violation(ip_address):
            validation_result['threat_level'] = ThreatLevel.MEDIUM
            validation_result['warnings'].append("Rate limit exceeded")
            
        # Log security event
        if validation_result['warnings']:
            await self._log_security_event(request_data, validation_result)
            
        return validation_result
        
    async def _handle_brute_force(self, ip_address: str, user_id: Optional[str]):
        """Handle brute force attack detection."""
        # Block IP temporarily
        self.blocked_ips.add(ip_address)
        
        # Create security alert
        alert = SecurityAlert(
            id=secrets.token_urlsafe(16),
            event_type=SecurityEvent.AUTHENTICATION_FAILURE,
            threat_level=ThreatLevel.HIGH,
            source_ip=ip_address,
            user_id=user_id,
            description=f"Brute force attack detected from {ip_address}"
        )
        
        self.security_alerts.append(alert)
        logging.error(f"Brute force attack blocked: {ip_address}")
        
        # Schedule IP unblock after 1 hour
        asyncio.create_task(self._schedule_ip_unblock(ip_address, 3600))
        
    async def _schedule_ip_unblock(self, ip_address: str, delay_seconds: int):
        """Schedule IP address unblocking."""
        await asyncio.sleep(delay_seconds)
        if ip_address in self.blocked_ips:
            self.blocked_ips.remove(ip_address)
            logging.info(f"IP address unblocked: {ip_address}")
            
    async def _log_security_event(self, request_data: Dict, validation_result: Dict):
        """Log security events for monitoring."""
        event_data = {
            'timestamp': time.time(),
            'ip_address': request_data.get('ip_address'),
            'user_id': request_data.get('user_id'),
            'threat_level': validation_result['threat_level'].value,
            'warnings': validation_result['warnings'],
            'request_data': request_data
        }
        
        # In production, would send to SIEM system
        logging.warning(f"Security event: {json.dumps(event_data)}")
        
    def generate_jwt_token(self, user_id: str, permissions: List[str], 
                          expires_in_hours: int = 1) -> str:
        """Generate secure JWT token."""
        payload = {
            'user_id': user_id,
            'permissions': permissions,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(hours=expires_in_hours),
            'jti': secrets.token_urlsafe(16)  # Unique token ID
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
        
    def validate_jwt_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT token."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return {'valid': True, 'payload': payload}
        except jwt.ExpiredSignatureError:
            return {'valid': False, 'error': 'Token expired'}
        except jwt.InvalidTokenError:
            return {'valid': False, 'error': 'Invalid token'}
            
    async def encrypt_sensitive_data(self, data: Dict[str, Any], 
                                   phi_fields: List[str]) -> Dict[str, Any]:
        """Encrypt sensitive data fields."""
        encrypted_data = data.copy()
        
        for field in phi_fields:
            if field in data:
                # Encrypt the field
                field_data = json.dumps(data[field]).encode()
                encryption_result = self.encryption_manager.encrypt_data(field_data)
                
                encrypted_data[field] = encryption_result['encrypted_data']
                encrypted_data[f'{field}_key_id'] = encryption_result['key_id']
                encrypted_data[f'{field}_encrypted'] = True
                
                # Log encryption event
                await self._log_encryption_event(field, encryption_result['key_id'])
                
        return encrypted_data
        
    async def _log_encryption_event(self, field_name: str, key_id: str):
        """Log encryption events for compliance."""
        event = {
            'timestamp': time.time(),
            'event_type': 'data_encryption',
            'field_name': field_name,
            'key_id': key_id,
            'compliance_requirement': 'HIPAA'
        }
        
        logging.info(f"Encryption event: {json.dumps(event)}")
        
    async def run_security_monitoring(self):
        """Background security monitoring."""
        while True:
            try:
                # Rotate encryption keys
                expired_keys = self.encryption_manager.rotate_keys()
                if expired_keys:
                    logging.info(f"Rotated {len(expired_keys)} encryption keys")
                    
                # Clean up old security alerts
                cutoff_time = time.time() - (7 * 24 * 3600)  # 7 days
                self.security_alerts = [
                    alert for alert in self.security_alerts 
                    if alert.timestamp > cutoff_time
                ]
                
                # Generate compliance report
                compliance_report = self.compliance_monitor.generate_compliance_report()
                if compliance_report['total_violations'] > 0:
                    logging.warning(f"Compliance violations detected: {compliance_report['total_violations']}")
                    
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                logging.error(f"Security monitoring error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
                
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get security dashboard data."""
        recent_alerts = [
            alert for alert in self.security_alerts 
            if alert.timestamp > time.time() - 3600  # Last hour
        ]
        
        return {
            'active_threats': len(recent_alerts),
            'blocked_ips': len(self.blocked_ips),
            'encryption_keys': len(self.encryption_manager.keys),
            'compliance_status': self.compliance_monitor.generate_compliance_report()['compliance_status'],
            'recent_alerts': [
                {
                    'id': alert.id,
                    'type': alert.event_type.value,
                    'level': alert.threat_level.value,
                    'source': alert.source_ip,
                    'timestamp': alert.timestamp
                }
                for alert in recent_alerts
            ]
        }

async def main():
    """Main entry point for testing."""
    security = SecurityOrchestrator()
    
    print("Advanced Security Framework initialized")
    
    # Test security validation
    test_request = {
        'ip_address': '192.168.1.100',
        'user_id': 'test_user',
        'input': 'SELECT * FROM users WHERE id=1'
    }
    
    validation_result = await security.validate_request(test_request)
    print(f"Validation result: {validation_result}")
    
    # Test encryption
    sensitive_data = {
        'patient_name': 'John Doe',
        'ssn': '123-45-6789',
        'diagnosis': 'Pneumonia'
    }
    
    encrypted_data = await security.encrypt_sensitive_data(
        sensitive_data, 
        ['patient_name', 'ssn']
    )
    print(f"Encrypted data: {encrypted_data}")
    
    # Start monitoring
    await security.run_security_monitoring()

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Security framework stopped")