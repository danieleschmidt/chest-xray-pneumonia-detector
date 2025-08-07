"""Comprehensive security framework for quantum task scheduler."""

import hashlib
import secrets
import logging
import time
import threading
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import re

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security access levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class Permission(Enum):
    """System permissions."""
    READ_TASKS = "read_tasks"
    CREATE_TASKS = "create_tasks"
    MODIFY_TASKS = "modify_tasks"
    DELETE_TASKS = "delete_tasks"
    START_TASKS = "start_tasks"
    STOP_TASKS = "stop_tasks"
    VIEW_METRICS = "view_metrics"
    ADMIN_ACCESS = "admin_access"
    EXPORT_DATA = "export_data"
    IMPORT_DATA = "import_data"


@dataclass
class SecurityEvent:
    """Security event for audit logging."""
    event_id: str = field(default_factory=lambda: secrets.token_hex(8))
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = ""
    user_id: Optional[str] = None
    resource: Optional[str] = None
    action: str = ""
    success: bool = True
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    risk_score: int = 0  # 0-100


@dataclass
class User:
    """User with security permissions."""
    user_id: str
    username: str
    email: str
    permissions: Set[Permission] = field(default_factory=set)
    security_level: SecurityLevel = SecurityLevel.PUBLIC
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    is_active: bool = True
    password_hash: Optional[str] = None
    api_tokens: Dict[str, str] = field(default_factory=dict)


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, max_tokens: int = 100, refill_rate: float = 10.0):
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self.tokens = max_tokens
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def allow_request(self, tokens_needed: int = 1) -> bool:
        """Check if request is allowed under rate limit."""
        with self.lock:
            now = time.time()
            time_passed = now - self.last_refill
            
            # Refill tokens
            self.tokens = min(self.max_tokens, self.tokens + time_passed * self.refill_rate)
            self.last_refill = now
            
            if self.tokens >= tokens_needed:
                self.tokens -= tokens_needed
                return True
            
            return False


class SecurityAuditLogger:
    """Comprehensive security audit logging."""
    
    def __init__(self, max_events: int = 10000):
        self.max_events = max_events
        self.events: List[SecurityEvent] = []
        self.lock = threading.Lock()
        
        # Event type risk scores
        self.risk_scores = {
            'login_failure': 30,
            'permission_denied': 40,
            'data_export': 20,
            'task_deletion': 25,
            'admin_access': 50,
            'suspicious_activity': 80,
            'security_breach': 100
        }
    
    def log_event(self, event: SecurityEvent) -> None:
        """Log security event with risk assessment."""
        # Assign risk score
        event.risk_score = self.risk_scores.get(event.event_type, 10)
        
        # Enhance risk score based on context
        if event.user_id and self._is_admin_user(event.user_id):
            event.risk_score += 10
        
        if not event.success:
            event.risk_score += 20
        
        with self.lock:
            self.events.append(event)
            
            # Maintain maximum events
            if len(self.events) > self.max_events:
                self.events.pop(0)
        
        # Log high-risk events
        if event.risk_score >= 50:
            logger.warning(f"High-risk security event: {event.event_type} by {event.user_id}")
        
        # Trigger alerts for critical events
        if event.risk_score >= 80:
            self._trigger_security_alert(event)
    
    def _is_admin_user(self, user_id: str) -> bool:
        """Check if user has admin privileges."""
        # This would integrate with user management system
        return False
    
    def _trigger_security_alert(self, event: SecurityEvent) -> None:
        """Trigger security alert for critical events."""
        alert_message = (f"SECURITY ALERT: {event.event_type} - "
                        f"User: {event.user_id}, Risk: {event.risk_score}")
        logger.critical(alert_message)
    
    def get_security_summary(self, hours: int = 24) -> Dict:
        """Get security summary for specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_events = [e for e in self.events if e.timestamp >= cutoff_time]
        
        # Calculate statistics
        total_events = len(recent_events)
        failed_events = len([e for e in recent_events if not e.success])
        high_risk_events = len([e for e in recent_events if e.risk_score >= 50])
        
        # Event type breakdown
        event_types = {}
        for event in recent_events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
        
        # Top risk events
        top_risk_events = sorted(recent_events, key=lambda e: e.risk_score, reverse=True)[:10]
        
        return {
            'period_hours': hours,
            'total_events': total_events,
            'failed_events': failed_events,
            'failure_rate': failed_events / max(1, total_events),
            'high_risk_events': high_risk_events,
            'event_types': event_types,
            'top_risk_events': [
                {
                    'event_id': e.event_id,
                    'type': e.event_type,
                    'user': e.user_id,
                    'risk_score': e.risk_score,
                    'timestamp': e.timestamp.isoformat()
                }
                for e in top_risk_events
            ]
        }


class InputSanitizer:
    """Input sanitization and validation."""
    
    @staticmethod
    def sanitize_string(input_str: str, max_length: int = 255, allow_html: bool = False) -> str:
        """Sanitize string input."""
        if not isinstance(input_str, str):
            raise ValueError("Input must be a string")
        
        # Limit length
        sanitized = input_str[:max_length]
        
        # Remove control characters
        sanitized = re.sub(r'[\x00-\x1F\x7F]', '', sanitized)
        
        # Remove HTML if not allowed
        if not allow_html:
            sanitized = re.sub(r'<[^>]+>', '', sanitized)
        
        # Escape dangerous characters
        sanitized = sanitized.replace('<script', '&lt;script')
        sanitized = sanitized.replace('javascript:', 'javascript-')
        
        return sanitized.strip()
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_task_name(name: str) -> bool:
        """Validate task name format."""
        if not name or len(name) > 100:
            return False
        
        # Allow alphanumeric, spaces, hyphens, underscores
        pattern = r'^[a-zA-Z0-9\s\-_]+$'
        return bool(re.match(pattern, name))
    
    @staticmethod
    def validate_json_input(json_str: str, max_size: int = 10000) -> bool:
        """Validate JSON input."""
        if len(json_str) > max_size:
            return False
        
        try:
            json.loads(json_str)
            return True
        except (json.JSONDecodeError, TypeError):
            return False


class QuantumSchedulerSecurity:
    """Comprehensive security framework for quantum task scheduler."""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.audit_logger = SecurityAuditLogger()
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.blocked_ips: Set[str] = set()
        self.session_tokens: Dict[str, Dict] = {}
        
        # Security policies
        self.password_policy = {
            'min_length': 8,
            'require_uppercase': True,
            'require_lowercase': True,
            'require_digits': True,
            'require_special': True
        }
        
        self.session_timeout = timedelta(hours=8)
        self.max_failed_logins = 5
        self.lockout_duration = timedelta(minutes=30)
    
    def create_user(self, user_id: str, username: str, email: str, 
                   password: str, permissions: List[Permission] = None,
                   security_level: SecurityLevel = SecurityLevel.PUBLIC) -> User:
        """Create new user with security validation."""
        
        # Validate input
        if not self._validate_password(password):
            raise ValueError("Password does not meet security requirements")
        
        if not InputSanitizer.validate_email(email):
            raise ValueError("Invalid email format")
        
        username = InputSanitizer.sanitize_string(username, 50)
        
        # Check for existing user
        if user_id in self.users:
            raise ValueError("User already exists")
        
        # Create user
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            permissions=set(permissions or []),
            security_level=security_level,
            password_hash=self._hash_password(password)
        )
        
        self.users[user_id] = user
        
        # Log event
        self.audit_logger.log_event(SecurityEvent(
            event_type='user_created',
            user_id=user_id,
            action='create_user',
            details={'security_level': security_level.value}
        ))
        
        return user
    
    def authenticate_user(self, username: str, password: str, 
                         ip_address: Optional[str] = None) -> Optional[str]:
        """Authenticate user and return session token."""
        
        # Check IP blocking
        if ip_address and ip_address in self.blocked_ips:
            self.audit_logger.log_event(SecurityEvent(
                event_type='blocked_ip_access',
                action='authenticate',
                success=False,
                ip_address=ip_address,
                details={'reason': 'IP blocked'}
            ))
            return None
        
        # Rate limiting
        rate_key = f"auth:{ip_address or 'unknown'}"
        if rate_key not in self.rate_limiters:
            self.rate_limiters[rate_key] = RateLimiter(max_tokens=10, refill_rate=1.0)
        
        if not self.rate_limiters[rate_key].allow_request():
            self.audit_logger.log_event(SecurityEvent(
                event_type='rate_limit_exceeded',
                action='authenticate',
                success=False,
                ip_address=ip_address
            ))
            return None
        
        # Find user by username
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break
        
        if not user or not user.is_active:
            self.audit_logger.log_event(SecurityEvent(
                event_type='login_failure',
                user_id=username,
                action='authenticate',
                success=False,
                ip_address=ip_address,
                details={'reason': 'user_not_found'}
            ))
            return None
        
        # Check password
        if not self._verify_password(password, user.password_hash):
            user.failed_login_attempts += 1
            
            # Lock account after max failed attempts
            if user.failed_login_attempts >= self.max_failed_logins:
                user.is_active = False
                
                self.audit_logger.log_event(SecurityEvent(
                    event_type='account_locked',
                    user_id=user.user_id,
                    action='lock_account',
                    ip_address=ip_address,
                    details={'failed_attempts': user.failed_login_attempts}
                ))
            
            self.audit_logger.log_event(SecurityEvent(
                event_type='login_failure',
                user_id=user.user_id,
                action='authenticate',
                success=False,
                ip_address=ip_address,
                details={'failed_attempts': user.failed_login_attempts}
            ))
            return None
        
        # Successful authentication
        user.failed_login_attempts = 0
        user.last_login = datetime.now()
        
        # Create session token
        session_token = secrets.token_urlsafe(32)
        self.session_tokens[session_token] = {
            'user_id': user.user_id,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + self.session_timeout,
            'ip_address': ip_address
        }
        
        self.audit_logger.log_event(SecurityEvent(
            event_type='login_success',
            user_id=user.user_id,
            action='authenticate',
            ip_address=ip_address
        ))
        
        return session_token
    
    def validate_session(self, session_token: str) -> Optional[User]:
        """Validate session token and return user."""
        if session_token not in self.session_tokens:
            return None
        
        session = self.session_tokens[session_token]
        
        # Check expiration
        if datetime.now() > session['expires_at']:
            del self.session_tokens[session_token]
            return None
        
        user_id = session['user_id']
        return self.users.get(user_id)
    
    def check_permission(self, session_token: str, permission: Permission, 
                        resource: str = None) -> bool:
        """Check if user has specific permission."""
        user = self.validate_session(session_token)
        if not user:
            self.audit_logger.log_event(SecurityEvent(
                event_type='permission_denied',
                action=f'check_{permission.value}',
                success=False,
                resource=resource,
                details={'reason': 'invalid_session'}
            ))
            return False
        
        has_permission = permission in user.permissions
        
        if not has_permission:
            self.audit_logger.log_event(SecurityEvent(
                event_type='permission_denied',
                user_id=user.user_id,
                action=f'check_{permission.value}',
                success=False,
                resource=resource
            ))
        
        return has_permission
    
    def secure_task_creation(self, session_token: str, task_data: Dict) -> Dict:
        """Securely create task with validation and sanitization."""
        user = self.validate_session(session_token)
        if not user:
            raise PermissionError("Invalid session")
        
        if not self.check_permission(session_token, Permission.CREATE_TASKS):
            raise PermissionError("Insufficient permissions")
        
        # Sanitize input data
        sanitized_data = {}
        
        if 'name' in task_data:
            name = InputSanitizer.sanitize_string(task_data['name'], 100)
            if not InputSanitizer.validate_task_name(name):
                raise ValueError("Invalid task name")
            sanitized_data['name'] = name
        
        if 'description' in task_data:
            sanitized_data['description'] = InputSanitizer.sanitize_string(
                task_data['description'], 1000
            )
        
        # Apply security level restrictions
        if user.security_level.value in ['public', 'internal']:
            # Restrict certain task types for lower security levels
            if 'type' in task_data and task_data['type'] in ['system', 'admin']:
                raise PermissionError("Cannot create system tasks with current security level")
        
        # Log successful task creation
        self.audit_logger.log_event(SecurityEvent(
            event_type='task_created',
            user_id=user.user_id,
            action='create_task',
            resource=sanitized_data.get('name', 'unnamed_task'),
            details={'task_type': task_data.get('type', 'standard')}
        ))
        
        return sanitized_data
    
    def secure_data_export(self, session_token: str, export_type: str) -> bool:
        """Securely handle data export with proper authorization."""
        user = self.validate_session(session_token)
        if not user:
            return False
        
        if not self.check_permission(session_token, Permission.EXPORT_DATA):
            return False
        
        # Additional restrictions for sensitive data
        if export_type in ['full_database', 'user_data'] and user.security_level.value not in ['confidential', 'restricted', 'top_secret']:
            self.audit_logger.log_event(SecurityEvent(
                event_type='permission_denied',
                user_id=user.user_id,
                action='export_data',
                success=False,
                details={'export_type': export_type, 'reason': 'insufficient_security_level'}
            ))
            return False
        
        # Log export attempt
        self.audit_logger.log_event(SecurityEvent(
            event_type='data_export',
            user_id=user.user_id,
            action='export_data',
            details={'export_type': export_type}
        ))
        
        return True
    
    def _validate_password(self, password: str) -> bool:
        """Validate password against security policy."""
        if len(password) < self.password_policy['min_length']:
            return False
        
        if self.password_policy['require_uppercase'] and not any(c.isupper() for c in password):
            return False
        
        if self.password_policy['require_lowercase'] and not any(c.islower() for c in password):
            return False
        
        if self.password_policy['require_digits'] and not any(c.isdigit() for c in password):
            return False
        
        if self.password_policy['require_special'] and not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password):
            return False
        
        return True
    
    def _hash_password(self, password: str) -> str:
        """Hash password using secure algorithm."""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{password_hash.hex()}"
    
    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify password against stored hash."""
        if not stored_hash:
            return False
        
        try:
            salt, hash_hex = stored_hash.split(':', 1)
            password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return hash_hex == password_hash.hex()
        except Exception:
            return False
    
    def cleanup_expired_sessions(self) -> None:
        """Clean up expired session tokens."""
        current_time = datetime.now()
        expired_tokens = [
            token for token, session in self.session_tokens.items()
            if current_time > session['expires_at']
        ]
        
        for token in expired_tokens:
            del self.session_tokens[token]
        
        if expired_tokens:
            logger.info(f"Cleaned up {len(expired_tokens)} expired sessions")
    
    def get_security_report(self) -> Dict:
        """Generate comprehensive security report."""
        audit_summary = self.audit_logger.get_security_summary()
        
        active_sessions = len([
            s for s in self.session_tokens.values()
            if datetime.now() <= s['expires_at']
        ])
        
        return {
            'timestamp': datetime.now().isoformat(),
            'user_statistics': {
                'total_users': len(self.users),
                'active_users': len([u for u in self.users.values() if u.is_active]),
                'locked_accounts': len([u for u in self.users.values() if not u.is_active])
            },
            'session_statistics': {
                'active_sessions': active_sessions,
                'total_tokens': len(self.session_tokens)
            },
            'security_events': audit_summary,
            'blocked_ips': len(self.blocked_ips),
            'rate_limiters': len(self.rate_limiters)
        }