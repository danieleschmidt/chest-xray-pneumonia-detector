"""Centralized error handling for the pneumonia detection system."""

import traceback
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from .exceptions import (
    PneumoniaDetectorError,
    ModelError,
    ValidationError,
    SecurityError,
    ResourceError,
    ConfigurationError,
    TrainingError,
    QuantumSchedulerError
)

logger = logging.getLogger(__name__)


class ErrorHandler:
    """Centralized error handling with logging and metrics."""
    
    ERROR_CATEGORIES = {
        'ModelError': 'model',
        'ValidationError': 'validation',
        'SecurityError': 'security',
        'ResourceError': 'resource',
        'ConfigurationError': 'configuration',
        'TrainingError': 'training',
        'QuantumSchedulerError': 'quantum',
        'PneumoniaDetectorError': 'application',
        'Exception': 'system'
    }
    
    @classmethod
    def handle_error(
        cls,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> Tuple[str, str, int]:
        """Handle errors with comprehensive logging and categorization.
        
        Args:
            error: The exception that occurred
            context: Additional context information
            user_id: User identifier for audit logging
            request_id: Request identifier for tracing
            
        Returns:
            Tuple of (error_message, error_code, http_status_code)
        """
        # Get error category
        error_type = type(error).__name__
        category = cls.ERROR_CATEGORIES.get(error_type, 'system')
        
        # Extract error details
        if isinstance(error, PneumoniaDetectorError):
            message = error.message
            error_code = error.error_code
            details = error.details
        else:
            message = str(error)
            error_code = f"{category.upper()}_ERROR"
            details = {}
        
        # Add context information
        error_context = {
            'error_type': error_type,
            'category': category,
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'request_id': request_id,
            'context': context or {},
            'details': details
        }
        
        # Log the error appropriately
        cls._log_error(error, error_context, message)
        
        # Determine HTTP status code
        status_code = cls._get_http_status_code(error)
        
        # Return sanitized error information
        user_message = cls._sanitize_error_message(message, category)
        
        return user_message, error_code, status_code
    
    @classmethod
    def _log_error(cls, error: Exception, context: Dict[str, Any], message: str) -> None:
        """Log error with appropriate level and detail."""
        
        # Create log entry
        log_data = {
            'message': message,
            'error_code': context.get('error_code', 'UNKNOWN'),
            'category': context['category'],
            'user_id': context.get('user_id'),
            'request_id': context.get('request_id'),
            'error_type': context['error_type']
        }
        
        # Log based on error severity
        if isinstance(error, SecurityError):
            logger.critical("Security violation detected", extra=log_data)
        elif isinstance(error, (ResourceError, ConfigurationError)):
            logger.error("System error occurred", extra=log_data)
        elif isinstance(error, (ValidationError, ModelError)):
            logger.warning("User/system input error", extra=log_data)
        else:
            logger.error("Unexpected error occurred", extra=log_data)
        
        # Log full traceback for debugging (but not in production logs)
        logger.debug(f"Full traceback: {traceback.format_exc()}")
    
    @classmethod
    def _get_http_status_code(cls, error: Exception) -> int:
        """Determine appropriate HTTP status code for error."""
        if isinstance(error, ValidationError):
            return 400  # Bad Request
        elif isinstance(error, SecurityError):
            return 403  # Forbidden
        elif isinstance(error, ResourceError):
            return 429  # Too Many Requests
        elif isinstance(error, ConfigurationError):
            return 503  # Service Unavailable
        elif isinstance(error, (ModelError, TrainingError)):
            return 422  # Unprocessable Entity
        elif isinstance(error, QuantumSchedulerError):
            return 500  # Internal Server Error
        else:
            return 500  # Internal Server Error
    
    @classmethod
    def _sanitize_error_message(cls, message: str, category: str) -> str:
        """Sanitize error message for user consumption."""
        
        # Security errors should be generic to prevent information leakage
        if category == 'security':
            return "Access denied or security violation detected"
        
        # System errors should be generic
        if category in ['system', 'configuration']:
            return "Internal system error occurred"
        
        # For user-facing errors, provide helpful but safe messages
        if category == 'validation':
            return f"Input validation failed: {message}"
        elif category == 'model':
            return "Model processing error occurred"
        elif category == 'resource':
            return "System resources temporarily unavailable"
        elif category == 'training':
            return "Model training error occurred"
        elif category == 'quantum':
            return "Task scheduling error occurred"
        else:
            return "An unexpected error occurred"
    
    @classmethod
    def create_error_response(cls, error: Exception, **kwargs) -> Dict[str, Any]:
        """Create standardized error response dictionary.
        
        Args:
            error: The exception that occurred
            **kwargs: Additional context for error handling
            
        Returns:
            Dictionary containing error response data
        """
        message, error_code, status_code = cls.handle_error(error, **kwargs)
        
        return {
            'success': False,
            'error': {
                'message': message,
                'code': error_code,
                'timestamp': datetime.utcnow().isoformat()
            },
            'status_code': status_code
        }