"""
Logging configuration and utilities.
"""

import logging
import logging.config
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
import traceback
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """JSON structured logging formatter."""
    
    def __init__(self, service_name: str = "pneumonia-detector"):
        super().__init__()
        self.service_name = service_name
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        # Add source location for debug/error logs
        if record.levelno >= logging.ERROR or record.levelno <= logging.DEBUG:
            log_entry["source"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName
            }
        
        return json.dumps(log_entry, default=str)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        level_color = self.COLORS.get(record.levelname, '')
        reset = self.RESET
        
        # Format: [TIMESTAMP] LEVEL LOGGER: MESSAGE
        formatted = f"{level_color}[{datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')}] {record.levelname:8} {record.name}: {record.getMessage()}{reset}"
        
        # Add exception traceback if present
        if record.exc_info:
            formatted += f"\n{reset}{traceback.format_exception(*record.exc_info)}"
        
        return formatted


def setup_logging(
    level: str = "INFO",
    format_type: str = "structured",
    log_file: Optional[str] = None,
    service_name: str = "pneumonia-detector",
    enable_console: bool = True
) -> None:
    """
    Setup comprehensive logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type ('structured' for JSON, 'colored' for console)
        log_file: Optional file path for file logging
        service_name: Service name for structured logging
        enable_console: Whether to enable console logging
    """
    
    # Ensure logs directory exists
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    handlers = []
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        
        if format_type == "structured":
            console_handler.setFormatter(StructuredFormatter(service_name))
        else:
            console_handler.setFormatter(ColoredFormatter())
        
        console_handler.setLevel(getattr(logging, level.upper()))
        handlers.append(console_handler)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(StructuredFormatter(service_name))
        file_handler.setLevel(logging.DEBUG)  # File gets all logs
        handlers.append(file_handler)
        root_logger.addHandler(file_handler)
    
    # Configure third-party loggers
    configure_third_party_loggers()
    
    # Log initial configuration
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging configured",
        extra={
            "level": level,
            "format_type": format_type,
            "log_file": log_file,
            "service_name": service_name,
            "console_enabled": enable_console
        }
    )


def configure_third_party_loggers():
    """Configure third-party library loggers."""
    
    # TensorFlow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF verbosity
    tf_logger = logging.getLogger('tensorflow')
    tf_logger.setLevel(logging.WARNING)
    
    # PIL/Pillow logging
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.WARNING)
    
    # urllib3 logging (often too verbose)
    urllib3_logger = logging.getLogger('urllib3')
    urllib3_logger.setLevel(logging.WARNING)
    
    # MLflow logging
    mlflow_logger = logging.getLogger('mlflow')
    mlflow_logger.setLevel(logging.INFO)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name. If None, uses the calling module's name.
    
    Returns:
        Configured logger instance.
    """
    if name is None:
        # Get the calling module's name
        frame = sys._getframe(1)
        name = frame.f_globals.get('__name__', 'unknown')
    
    return logging.getLogger(name)


class LogContext:
    """Context manager for adding structured logging context."""
    
    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.old_factory = logging.getLogRecordFactory()
    
    def __enter__(self):
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)


def log_function_call(func):
    """Decorator to log function calls with timing."""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        start_time = datetime.utcnow()
        
        try:
            logger.debug(
                f"Calling {func.__name__}",
                extra={
                    "function": func.__name__,
                    "module": func.__module__,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs)
                }
            )
            
            result = func(*args, **kwargs)
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            logger.debug(
                f"Completed {func.__name__}",
                extra={
                    "function": func.__name__,
                    "module": func.__module__,
                    "duration_seconds": duration,
                    "success": True
                }
            )
            
            return result
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            logger.error(
                f"Error in {func.__name__}: {str(e)}",
                extra={
                    "function": func.__name__,
                    "module": func.__module__,
                    "duration_seconds": duration,
                    "success": False,
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            
            raise
    
    return wrapper


class RequestLogger:
    """HTTP request logging middleware."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger(__name__)
    
    def log_request(self, method: str, path: str, status_code: int, 
                   duration: float, user_id: Optional[str] = None):
        """Log an HTTP request."""
        self.logger.info(
            f"{method} {path} {status_code}",
            extra={
                "request_method": method,
                "request_path": path,
                "response_status": status_code,
                "duration_seconds": duration,
                "user_id": user_id,
                "event_type": "http_request"
            }
        )


def setup_from_environment():
    """Setup logging from environment variables."""
    level = os.getenv("LOG_LEVEL", "INFO")
    format_type = os.getenv("LOG_FORMAT", "structured")
    log_file = os.getenv("LOG_FILE")
    service_name = os.getenv("SERVICE_NAME", "pneumonia-detector")
    
    # Check if running in container/production
    is_production = os.getenv("ENVIRONMENT") == "production"
    enable_console = not is_production or os.getenv("ENABLE_CONSOLE_LOGGING", "true").lower() == "true"
    
    setup_logging(
        level=level,
        format_type=format_type,
        log_file=log_file,
        service_name=service_name,
        enable_console=enable_console
    )


# Setup logging from environment on import
setup_from_environment()


if __name__ == "__main__":
    # CLI interface for logging
    import argparse
    
    parser = argparse.ArgumentParser(description="Logging configuration utility")
    parser.add_argument("--level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                       default="INFO", help="Logging level")
    parser.add_argument("--format", choices=["structured", "colored"], 
                       default="structured", help="Log format")
    parser.add_argument("--file", help="Log file path")
    parser.add_argument("--service", default="pneumonia-detector", help="Service name")
    parser.add_argument("--test", action="store_true", help="Test logging configuration")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        level=args.level,
        format_type=args.format,
        log_file=args.file,
        service_name=args.service
    )
    
    if args.test:
        # Test logging at different levels
        logger = get_logger(__name__)
        
        logger.debug("This is a debug message", extra={"test_field": "debug_value"})
        logger.info("This is an info message", extra={"test_field": "info_value"})
        logger.warning("This is a warning message", extra={"test_field": "warning_value"})
        logger.error("This is an error message", extra={"test_field": "error_value"})
        
        # Test exception logging
        try:
            raise ValueError("Test exception for logging")
        except Exception:
            logger.error("Caught test exception", exc_info=True)
        
        # Test context logging
        with LogContext(logger, request_id="test-123", user_id="test-user"):
            logger.info("Message with context")
        
        print("Logging test completed")
    else:
        print(f"Logging configured with level={args.level}, format={args.format}")
        if args.file:
            print(f"Logs will be written to: {args.file}")
        print("Use --test to test the configuration")