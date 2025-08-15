#!/usr/bin/env python3
"""
Comprehensive Security Framework
Generation 2: Advanced security measures for medical AI systems
"""

import hashlib
import hmac
import json
import os
import re
import secrets
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
import logging
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SecurityEvent:
    """Security event data structure."""
    timestamp: str
    event_type: str
    severity: str
    description: str
    source_ip: str = ""
    user_id: str = ""
    additional_data: Dict[str, Any] = None


class SecureAuditLogger:
    """Tamper-resistant audit logging for medical data compliance."""
    
    def __init__(self, audit_dir: str = "audit_logs", encryption_key: str = None):
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(exist_ok=True)
        self.encryption_key = encryption_key or self._generate_key()
        self.sequence_number = 0
        
        # Initialize audit log file
        self.audit_file = self.audit_dir / "medical_audit.log"
        self.integrity_file = self.audit_dir / "integrity.hash"
        
        # Create logger
        self.logger = logging.getLogger("security_audit")
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.audit_file)
        formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def _generate_key(self) -> str:
        """Generate a secure encryption key."""
        return secrets.token_hex(32)
    
    def log_access(self, user_id: str, resource: str, action: str, 
                   source_ip: str = "", success: bool = True):
        """Log access attempts to medical data."""
        self.sequence_number += 1
        
        audit_entry = {
            "sequence": self.sequence_number,
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "source_ip": source_ip,
            "success": success,
            "session_id": self._get_session_id(user_id)
        }
        
        # Create integrity hash
        entry_hash = self._create_integrity_hash(audit_entry)
        audit_entry["integrity_hash"] = entry_hash
        
        # Log the entry
        self.logger.info(json.dumps(audit_entry))
        
        # Update integrity file
        self._update_integrity_chain(entry_hash)
    
    def log_data_modification(self, user_id: str, data_type: str, 
                            operation: str, record_id: str = ""):
        """Log modifications to medical data."""
        self.log_access(
            user_id=user_id,
            resource=f"{data_type}:{record_id}",
            action=f"MODIFY:{operation}",
            success=True
        )
    
    def log_security_event(self, event: SecurityEvent):
        """Log security-related events."""
        self.sequence_number += 1
        
        security_entry = {
            "sequence": self.sequence_number,
            "timestamp": event.timestamp,
            "event_type": event.event_type,
            "severity": event.severity,
            "description": event.description,
            "source_ip": event.source_ip,
            "user_id": event.user_id,
            "additional_data": event.additional_data or {}
        }
        
        entry_hash = self._create_integrity_hash(security_entry)
        security_entry["integrity_hash"] = entry_hash
        
        self.logger.warning(f"SECURITY_EVENT: {json.dumps(security_entry)}")
        self._update_integrity_chain(entry_hash)
    
    def _create_integrity_hash(self, entry: Dict[str, Any]) -> str:
        """Create tamper-resistant hash for audit entry."""
        entry_copy = entry.copy()
        entry_copy.pop("integrity_hash", None)
        
        entry_string = json.dumps(entry_copy, sort_keys=True)
        return hmac.new(
            self.encryption_key.encode(),
            entry_string.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def _update_integrity_chain(self, entry_hash: str):
        """Update the integrity chain file."""
        chain_data = {"latest_hash": entry_hash, "updated_at": datetime.now().isoformat()}
        
        with open(self.integrity_file, "w") as f:
            json.dump(chain_data, f)
    
    def _get_session_id(self, user_id: str) -> str:
        """Generate session ID for user."""
        return hashlib.sha256(f"{user_id}:{time.time()}".encode()).hexdigest()[:16]
    
    def verify_integrity(self) -> bool:
        """Verify the integrity of audit logs."""
        try:
            with open(self.audit_file, "r") as f:
                for line in f:
                    if "AUDIT" in line and "{" in line:
                        entry_data = json.loads(line.split(" - AUDIT - INFO - ")[1])
                        
                        stored_hash = entry_data.pop("integrity_hash", "")
                        calculated_hash = self._create_integrity_hash(entry_data)
                        
                        if stored_hash != calculated_hash:
                            return False
            
            return True
        
        except Exception as e:
            logging.error(f"Integrity verification failed: {e}")
            return False


class MedicalDataProtector:
    """Advanced protection for medical data (PHI/PII)."""
    
    def __init__(self):
        self.sensitive_patterns = {
            "ssn": r"\\b\\d{3}-?\\d{2}-?\\d{4}\\b",
            "phone": r"\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b",
            "email": r"\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b",
            "medical_id": r"\\b[A-Z]{2}\\d{8}\\b",
            "credit_card": r"\\b\\d{4}[-\\s]?\\d{4}[-\\s]?\\d{4}[-\\s]?\\d{4}\\b"
        }
    
    def detect_sensitive_data(self, text: str) -> Dict[str, List[str]]:
        """Detect sensitive data patterns in text."""
        findings = {}
        
        for pattern_name, pattern in self.sensitive_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                findings[pattern_name] = matches
        
        return findings
    
    def anonymize_text(self, text: str) -> Tuple[str, Dict[str, int]]:
        """Anonymize sensitive data in text."""
        anonymized_text = text
        replacements = {}
        
        for pattern_name, pattern in self.sensitive_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            
            if matches:
                replacement_token = f"[{pattern_name.upper()}_REDACTED]"
                anonymized_text = re.sub(pattern, replacement_token, anonymized_text, flags=re.IGNORECASE)
                replacements[pattern_name] = len(matches)
        
        return anonymized_text, replacements
    
    def encrypt_sensitive_field(self, data: str, key: str) -> str:
        """Encrypt sensitive data fields."""
        try:
            from cryptography.fernet import Fernet
            
            # Use provided key or generate one
            if len(key) != 44:  # Fernet key should be 44 characters
                key = Fernet.generate_key().decode()
            
            fernet = Fernet(key.encode() if isinstance(key, str) else key)
            encrypted_data = fernet.encrypt(data.encode())
            
            return encrypted_data.decode()
        
        except ImportError:
            # Fallback to basic encoding if cryptography not available
            import base64
            return base64.b64encode(data.encode()).decode()
    
    def validate_hipaa_compliance(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data for HIPAA compliance requirements."""
        compliance_report = {
            "compliant": True,
            "violations": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Check for unencrypted sensitive data
        for key, value in data_dict.items():
            if isinstance(value, str):
                sensitive_findings = self.detect_sensitive_data(value)
                
                if sensitive_findings:
                    compliance_report["compliant"] = False
                    compliance_report["violations"].append({
                        "field": key,
                        "issue": "Unprotected sensitive data",
                        "types": list(sensitive_findings.keys())
                    })
        
        # Add recommendations
        compliance_report["recommendations"] = [
            "Encrypt all sensitive data fields",
            "Implement access controls",
            "Enable audit logging",
            "Regular compliance audits"
        ]
        
        return compliance_report


class AccessController:
    """Role-based access control for medical systems."""
    
    def __init__(self):
        self.roles = {
            "admin": {
                "permissions": ["read", "write", "delete", "admin"],
                "resources": ["*"]
            },
            "doctor": {
                "permissions": ["read", "write"],
                "resources": ["patient_data", "medical_images", "reports"]
            },
            "nurse": {
                "permissions": ["read", "write"],
                "resources": ["patient_data", "medical_images"]
            },
            "technician": {
                "permissions": ["read"],
                "resources": ["medical_images", "equipment_data"]
            },
            "researcher": {
                "permissions": ["read"],
                "resources": ["anonymized_data", "research_datasets"]
            }
        }
        
        self.active_sessions = {}
        self.failed_attempts = {}
        self.audit_logger = SecureAuditLogger()
    
    def authenticate_user(self, user_id: str, password: str, 
                         source_ip: str = "") -> Optional[str]:
        """Authenticate user and return session token."""
        # Check for brute force protection
        if self._is_blocked(user_id, source_ip):
            self.audit_logger.log_security_event(SecurityEvent(
                timestamp=datetime.now().isoformat(),
                event_type="AUTHENTICATION_BLOCKED",
                severity="HIGH",
                description=f"User {user_id} blocked due to repeated failures",
                source_ip=source_ip,
                user_id=user_id
            ))
            return None
        
        # Simulate password check (in real implementation, use proper hashing)
        if self._verify_password(user_id, password):
            session_token = self._create_session(user_id, source_ip)
            
            self.audit_logger.log_access(
                user_id=user_id,
                resource="authentication",
                action="LOGIN",
                source_ip=source_ip,
                success=True
            )
            
            # Reset failed attempts
            key = f"{user_id}:{source_ip}"
            if key in self.failed_attempts:
                del self.failed_attempts[key]
            
            return session_token
        
        else:
            self._record_failed_attempt(user_id, source_ip)
            
            self.audit_logger.log_access(
                user_id=user_id,
                resource="authentication",
                action="LOGIN",
                source_ip=source_ip,
                success=False
            )
            
            return None
    
    def authorize_action(self, session_token: str, resource: str, 
                        action: str) -> bool:
        """Check if user is authorized for specific action."""
        session = self.active_sessions.get(session_token)
        
        if not session:
            return False
        
        # Check session expiry
        if datetime.now() > session["expires_at"]:
            del self.active_sessions[session_token]
            return False
        
        user_role = session["role"]
        role_config = self.roles.get(user_role, {})
        
        # Check permissions
        if action not in role_config.get("permissions", []):
            self.audit_logger.log_security_event(SecurityEvent(
                timestamp=datetime.now().isoformat(),
                event_type="AUTHORIZATION_DENIED",
                severity="MEDIUM",
                description=f"User {session['user_id']} denied {action} on {resource}",
                user_id=session["user_id"]
            ))
            return False
        
        # Check resource access
        allowed_resources = role_config.get("resources", [])
        if "*" not in allowed_resources and resource not in allowed_resources:
            return False
        
        # Log successful authorization
        self.audit_logger.log_access(
            user_id=session["user_id"],
            resource=resource,
            action=action,
            success=True
        )
        
        return True
    
    def _verify_password(self, user_id: str, password: str) -> bool:
        """Verify user password (placeholder implementation)."""
        # In real implementation, use proper password hashing (bcrypt, scrypt, etc.)
        dummy_users = {
            "admin": "admin_secure_password",
            "doctor_smith": "doctor_password",
            "nurse_jones": "nurse_password"
        }
        
        return dummy_users.get(user_id) == password
    
    def _create_session(self, user_id: str, source_ip: str) -> str:
        """Create authenticated session."""
        session_token = secrets.token_urlsafe(32)
        
        # Determine user role (in real implementation, fetch from database)
        user_role = "doctor" if "doctor" in user_id else "admin"
        
        self.active_sessions[session_token] = {
            "user_id": user_id,
            "role": user_role,
            "source_ip": source_ip,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(hours=8),
            "last_activity": datetime.now()
        }
        
        return session_token
    
    def _record_failed_attempt(self, user_id: str, source_ip: str):
        """Record failed authentication attempt."""
        key = f"{user_id}:{source_ip}"
        current_time = datetime.now()
        
        if key not in self.failed_attempts:
            self.failed_attempts[key] = []
        
        self.failed_attempts[key].append(current_time)
        
        # Clean old attempts (older than 1 hour)
        cutoff_time = current_time - timedelta(hours=1)
        self.failed_attempts[key] = [
            attempt for attempt in self.failed_attempts[key]
            if attempt > cutoff_time
        ]
    
    def _is_blocked(self, user_id: str, source_ip: str) -> bool:
        """Check if user/IP is blocked due to failed attempts."""
        key = f"{user_id}:{source_ip}"
        
        if key not in self.failed_attempts:
            return False
        
        # Block if more than 5 failed attempts in the last hour
        return len(self.failed_attempts[key]) >= 5


class SecurityScanner:
    """Security vulnerability scanner for medical AI systems."""
    
    def __init__(self):
        self.vulnerability_patterns = {
            "sql_injection": [
                r"(union|select|insert|update|delete|drop)\\s+",
                r"'\\s*(or|and)\\s*'",
                r"--\\s*",
                r"/\\*.*\\*/"
            ],
            "xss": [
                r"<script[^>]*>",
                r"javascript:",
                r"on\\w+\\s*="
            ],
            "path_traversal": [
                r"\\.\\.[\\\\/]",
                r"\\.\\.%2f",
                r"\\.\\.%5c"
            ],
            "command_injection": [
                r"[;&|`]",
                r"\\$\\([^)]*\\)",
                r"`[^`]*`"
            ]
        }
    
    def scan_input(self, input_data: str) -> Dict[str, Any]:
        """Scan input for security vulnerabilities."""
        findings = {}
        risk_score = 0
        
        for vuln_type, patterns in self.vulnerability_patterns.items():
            matches = []
            
            for pattern in patterns:
                found = re.findall(pattern, input_data, re.IGNORECASE)
                if found:
                    matches.extend(found)
            
            if matches:
                findings[vuln_type] = {
                    "matches": matches,
                    "count": len(matches)
                }
                risk_score += len(matches) * 10
        
        return {
            "safe": len(findings) == 0,
            "risk_score": risk_score,
            "vulnerabilities": findings,
            "recommendation": self._get_security_recommendation(risk_score)
        }
    
    def _get_security_recommendation(self, risk_score: int) -> str:
        """Get security recommendation based on risk score."""
        if risk_score == 0:
            return "Input appears safe"
        elif risk_score < 20:
            return "Low risk - monitor and sanitize input"
        elif risk_score < 50:
            return "Medium risk - sanitize input and log event"
        else:
            return "High risk - block input and alert security team"


if __name__ == "__main__":
    # Example usage and testing
    
    # Test audit logging
    audit_logger = SecureAuditLogger()
    audit_logger.log_access("doctor_smith", "patient_record_123", "READ")
    
    # Test data protection
    protector = MedicalDataProtector()
    test_text = "Patient SSN: 123-45-6789, Phone: 555-123-4567"
    anonymized, replacements = protector.anonymize_text(test_text)
    print(f"Anonymized: {anonymized}")
    print(f"Replacements: {replacements}")
    
    # Test access control
    access_controller = AccessController()
    session_token = access_controller.authenticate_user("doctor_smith", "doctor_password")
    
    if session_token:
        authorized = access_controller.authorize_action(session_token, "patient_data", "read")
        print(f"Authorization result: {authorized}")
    
    # Test security scanner
    scanner = SecurityScanner()
    scan_result = scanner.scan_input("SELECT * FROM users WHERE id = 1; DROP TABLE users;")
    print(f"Security scan: {scan_result}")