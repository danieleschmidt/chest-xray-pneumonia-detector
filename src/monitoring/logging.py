"""
Structured logging configuration for HIPAA compliance and observability.
"""

import json
import logging
import logging.config
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
        
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if configured
        if self.include_extra:
            extra_fields = {
                k: v for k, v in record.__dict__.items()
                if k not in [
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                    'filename', 'module', 'lineno', 'funcName', 'created',
                    'msecs', 'relativeCreated', 'thread', 'threadName',
                    'processName', 'process', 'message', 'exc_info', 'exc_text',
                    'stack_info'
                ]
            }
            if extra_fields:
                log_entry['extra'] = extra_fields
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)


class HealthcareAuditLogger:
    """HIPAA-compliant audit logger for healthcare data access."""
    
    def __init__(self, log_file: Optional[str] = None):
        self.logger = logging.getLogger('healthcare_audit')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create file handler for audit logs
        if not log_file:
            log_file = os.getenv('AUDIT_LOG_FILE', 'logs/audit.log')
        
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(StructuredFormatter())
        
        self.logger.addHandler(file_handler)
        self.logger.propagate = False
    
    def log_data_access(self, 
                       user_id: str,
                       patient_id: Optional[str],
                       action: str,
                       resource: str,
                       ip_address: Optional[str] = None,
                       user_agent: Optional[str] = None,
                       result: str = "success",
                       additional_info: Optional[Dict[str, Any]] = None):
        """Log healthcare data access for HIPAA compliance."""
        audit_entry = {
            'audit_type': 'data_access',
            'user_id': user_id,
            'patient_id': patient_id,
            'action': action,
            'resource': resource,
            'ip_address': ip_address,
            'user_agent': user_agent,
            'result': result,
            'session_id': getattr(self, '_current_session_id', None),
        }
        
        if additional_info:
            audit_entry.update(additional_info)
        
        self.logger.info("Healthcare data access", extra=audit_entry)
    
    def log_model_inference(self,
                          user_id: str,
                          image_hash: str,
                          model_version: str,
                          prediction: int,
                          confidence: float,
                          processing_time_ms: int,
                          ip_address: Optional[str] = None):
        """Log model inference for audit trail."""
        audit_entry = {
            'audit_type': 'model_inference',
            'user_id': user_id,
            'image_hash': image_hash,
            'model_version': model_version,
            'prediction': prediction,
            'confidence': confidence,
            'processing_time_ms': processing_time_ms,
            'ip_address': ip_address,
        }
        
        self.logger.info("Model inference performed", extra=audit_entry)
    
    def log_security_event(self,
                          event_type: str,
                          severity: str,
                          description: str,
                          user_id: Optional[str] = None,
                          ip_address: Optional[str] = None,
                          additional_context: Optional[Dict[str, Any]] = None):
        """Log security events."""
        audit_entry = {
            'audit_type': 'security_event',
            'event_type': event_type,
            'severity': severity,
            'description': description,
            'user_id': user_id,
            'ip_address': ip_address,
        }
        
        if additional_context:
            audit_entry.update(additional_context)
        
        self.logger.warning("Security event", extra=audit_entry)


def setup_logging(log_level: str = "INFO",
                 log_format: str = "json",
                 log_file: Optional[str] = None,
                 enable_audit: bool = True) -> Dict[str, Any]:
    """Setup application logging configuration."""
    
    # Ensure logs directory exists
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    else:
        Path("logs").mkdir(exist_ok=True)
        log_file = "logs/app.log"
    
    # Choose formatter based on format preference
    if log_format.lower() == "json":
        formatter_class = StructuredFormatter
    else:
        formatter_class = logging.Formatter
    
    # Logging configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'structured': {
                '()': StructuredFormatter,
                'include_extra': True
            },
            'simple': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'structured' if log_format.lower() == 'json' else 'simple',
                'stream': sys.stdout
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': log_level,
                'formatter': 'structured',
                'filename': log_file,
                'maxBytes': 100 * 1024 * 1024,  # 100MB
                'backupCount': 5
            }
        },
        'loggers': {
            '': {  # Root logger
                'handlers': ['console', 'file'],
                'level': log_level,
                'propagate': False
            },
            'uvicorn': {
                'handlers': ['console', 'file'],
                'level': 'INFO',
                'propagate': False
            },
            'uvicorn.access': {
                'handlers': ['file'],
                'level': 'INFO',
                'propagate': False
            }
        }
    }
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Setup audit logging if enabled
    audit_logger = None
    if enable_audit:
        audit_log_file = os.getenv('AUDIT_LOG_FILE', 'logs/audit.log')
        audit_logger = HealthcareAuditLogger(audit_log_file)
    
    return {
        'config': config,
        'audit_logger': audit_logger
    }


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with consistent configuration."""
    return logging.getLogger(name)


def setup_audit_logging(log_file: str) -> HealthcareAuditLogger:
    """Setup audit logging for HIPAA compliance."""
    return HealthcareAuditLogger(log_file)


class RequestIDFilter(logging.Filter):
    """Add request ID to log records for tracing."""
    
    def __init__(self):
        super().__init__()
        self.request_id = None
    
    def set_request_id(self, request_id: str):
        """Set the current request ID."""
        self.request_id = request_id
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add request ID to log record."""
        if self.request_id:
            record.request_id = self.request_id
        return True


# Global instances
request_id_filter = RequestIDFilter()
audit_logger = None

# Initialize logging on module import
def init_logging():
    """Initialize logging system."""
    global audit_logger
    
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    log_format = os.getenv('LOG_FORMAT', 'json')
    log_file = os.getenv('LOG_FILE', 'logs/app.log')
    enable_audit = os.getenv('ENABLE_AUDIT_LOGGING', 'true').lower() == 'true'
    
    setup_result = setup_logging(
        log_level=log_level,
        log_format=log_format,
        log_file=log_file,
        enable_audit=enable_audit
    )
    
    audit_logger = setup_result['audit_logger']
    
    # Add request ID filter to root logger
    root_logger = logging.getLogger()
    root_logger.addFilter(request_id_filter)


# Initialize on import
init_logging()