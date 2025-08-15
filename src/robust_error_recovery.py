#!/usr/bin/env python3
"""
Robust Error Recovery System
Generation 2: Comprehensive error handling and recovery mechanisms
"""

import functools
import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Type, Union
from datetime import datetime
import json
import os


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    self.failure_count = 0
                else:
                    raise Exception(f"Circuit breaker is OPEN. Service unavailable.")
            
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
        
        return wrapper


class RetryMechanism:
    """Configurable retry mechanism with exponential backoff."""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, backoff_factor: float = 2.0,
                 retriable_exceptions: tuple = (Exception,)):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.retriable_exceptions = retriable_exceptions
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_attempts):
                try:
                    return func(*args, **kwargs)
                
                except self.retriable_exceptions as e:
                    last_exception = e
                    
                    if attempt == self.max_attempts - 1:
                        break
                    
                    delay = min(
                        self.base_delay * (self.backoff_factor ** attempt),
                        self.max_delay
                    )
                    
                    logging.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper


class RobustErrorLogger:
    """Comprehensive error logging with structured data."""
    
    def __init__(self, log_dir: str = "logs", log_level: int = logging.INFO):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure main logger
        self.logger = logging.getLogger("robust_medical_ai")
        self.logger.setLevel(log_level)
        
        # File handler for general logs
        file_handler = logging.FileHandler(
            os.path.join(log_dir, "medical_ai.log")
        )
        file_handler.setLevel(log_level)
        
        # Error file handler
        error_handler = logging.FileHandler(
            os.path.join(log_dir, "errors.log")
        )
        error_handler.setLevel(logging.ERROR)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        file_handler.setFormatter(formatter)
        error_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(console_handler)
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with full context and stack trace."""
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "stack_trace": traceback.format_exc(),
            "context": context or {}
        }
        
        # Log structured error data
        error_file = os.path.join(self.log_dir, "structured_errors.jsonl")
        with open(error_file, "a") as f:
            f.write(json.dumps(error_data) + "\n")
        
        # Log to standard logger
        self.logger.error(
            f"Error in {context.get('function', 'unknown')}: {error}",
            extra={"error_data": error_data}
        )
    
    def log_performance_issue(self, operation: str, duration: float, 
                            threshold: float = 5.0):
        """Log performance issues when operations exceed thresholds."""
        if duration > threshold:
            self.logger.warning(
                f"Performance issue: {operation} took {duration:.2f}s "
                f"(threshold: {threshold}s)"
            )


class HealthMonitor:
    """System health monitoring and alerting."""
    
    def __init__(self, health_dir: str = "health_data"):
        self.health_dir = health_dir
        os.makedirs(health_dir, exist_ok=True)
        self.logger = RobustErrorLogger().logger
    
    def check_system_health(self) -> Dict[str, Any]:
        """Comprehensive system health check."""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "checks": {}
        }
        
        # Check disk space
        disk_usage = self._check_disk_space()
        health_status["checks"]["disk"] = disk_usage
        
        # Check memory usage
        memory_usage = self._check_memory_usage()
        health_status["checks"]["memory"] = memory_usage
        
        # Check log file sizes
        log_health = self._check_log_health()
        health_status["checks"]["logs"] = log_health
        
        # Check model files
        model_health = self._check_model_availability()
        health_status["checks"]["models"] = model_health
        
        # Determine overall status
        if any(check["status"] == "critical" for check in health_status["checks"].values()):
            health_status["overall_status"] = "critical"
        elif any(check["status"] == "warning" for check in health_status["checks"].values()):
            health_status["overall_status"] = "warning"
        
        # Save health report
        health_file = os.path.join(self.health_dir, "health_report.json")
        with open(health_file, "w") as f:
            json.dump(health_status, f, indent=2)
        
        return health_status
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            free_percent = (free / total) * 100
            
            status = "healthy"
            if free_percent < 5:
                status = "critical"
            elif free_percent < 15:
                status = "warning"
            
            return {
                "status": status,
                "free_percent": free_percent,
                "free_gb": free / (1024**3),
                "total_gb": total / (1024**3)
            }
        
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            status = "healthy"
            if memory.percent > 90:
                status = "critical"
            elif memory.percent > 80:
                status = "warning"
            
            return {
                "status": status,
                "used_percent": memory.percent,
                "available_gb": memory.available / (1024**3),
                "total_gb": memory.total / (1024**3)
            }
        
        except ImportError:
            return {"status": "unavailable", "error": "psutil not installed"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _check_log_health(self) -> Dict[str, Any]:
        """Check log file sizes and rotation needs."""
        try:
            log_dir = "logs"
            if not os.path.exists(log_dir):
                return {"status": "healthy", "message": "No log directory"}
            
            total_size = 0
            large_files = []
            
            for filename in os.listdir(log_dir):
                filepath = os.path.join(log_dir, filename)
                if os.path.isfile(filepath):
                    size = os.path.getsize(filepath)
                    total_size += size
                    
                    if size > 100 * 1024 * 1024:  # 100MB
                        large_files.append({"file": filename, "size_mb": size / (1024**2)})
            
            status = "healthy"
            if total_size > 1024**3:  # 1GB
                status = "warning"
            
            return {
                "status": status,
                "total_size_mb": total_size / (1024**2),
                "large_files": large_files
            }
        
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _check_model_availability(self) -> Dict[str, Any]:
        """Check if required model files are available."""
        try:
            model_dirs = ["saved_models", "models"]
            models_found = []
            
            for model_dir in model_dirs:
                if os.path.exists(model_dir):
                    for item in os.listdir(model_dir):
                        if item.endswith(('.keras', '.h5', '.pb')):
                            models_found.append(os.path.join(model_dir, item))
            
            status = "healthy" if models_found else "warning"
            
            return {
                "status": status,
                "models_found": len(models_found),
                "model_files": models_found[:5]  # Limit output
            }
        
        except Exception as e:
            return {"status": "error", "error": str(e)}


class RobustValidator:
    """Comprehensive input validation and sanitization."""
    
    @staticmethod
    def validate_image_input(image_data: Any, max_size_mb: int = 10) -> bool:
        """Validate image input data."""
        try:
            from PIL import Image
            import io
            
            if isinstance(image_data, str):
                # File path validation
                if not os.path.exists(image_data):
                    raise ValueError(f"Image file not found: {image_data}")
                
                # Check file size
                file_size = os.path.getsize(image_data)
                if file_size > max_size_mb * 1024 * 1024:
                    raise ValueError(f"Image file too large: {file_size / (1024**2):.2f}MB")
                
                # Try to open image
                with Image.open(image_data) as img:
                    # Verify it's a valid image
                    img.verify()
                
                return True
            
            elif isinstance(image_data, bytes):
                # Binary data validation
                if len(image_data) > max_size_mb * 1024 * 1024:
                    raise ValueError("Image data too large")
                
                # Try to open as image
                with Image.open(io.BytesIO(image_data)) as img:
                    img.verify()
                
                return True
            
            else:
                raise ValueError("Invalid image data type")
        
        except Exception as e:
            logging.error(f"Image validation failed: {e}")
            return False
    
    @staticmethod
    def validate_model_path(model_path: str) -> bool:
        """Validate model file path and format."""
        try:
            if not os.path.exists(model_path):
                raise ValueError(f"Model file not found: {model_path}")
            
            valid_extensions = ['.keras', '.h5', '.pb', '.tflite']
            if not any(model_path.lower().endswith(ext) for ext in valid_extensions):
                raise ValueError(f"Invalid model file format: {model_path}")
            
            # Check file size is reasonable
            file_size = os.path.getsize(model_path)
            if file_size < 1024:  # Less than 1KB
                raise ValueError("Model file suspiciously small")
            
            if file_size > 2 * 1024**3:  # More than 2GB
                logging.warning(f"Large model file: {file_size / (1024**3):.2f}GB")
            
            return True
        
        except Exception as e:
            logging.error(f"Model validation failed: {e}")
            return False
    
    @staticmethod
    def sanitize_file_path(file_path: str) -> str:
        """Sanitize file path to prevent directory traversal."""
        import re
        
        # Remove dangerous patterns
        sanitized = re.sub(r'\.\./', '', file_path)
        sanitized = re.sub(r'\.\.\\\\', '', sanitized)
        
        # Remove null bytes
        sanitized = sanitized.replace('\\x00', '')
        
        # Normalize path
        sanitized = os.path.normpath(sanitized)
        
        return sanitized


def robust_operation(retry_attempts: int = 3, circuit_breaker: bool = True):
    """Decorator to make operations robust with retry and circuit breaker."""
    def decorator(func: Callable) -> Callable:
        # Apply circuit breaker if requested
        if circuit_breaker:
            func = CircuitBreaker()(func)
        
        # Apply retry mechanism
        func = RetryMechanism(max_attempts=retry_attempts)(func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_logger = RobustErrorLogger()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Log performance if slow
                duration = time.time() - start_time
                error_logger.log_performance_issue(func.__name__, duration)
                
                return result
            
            except Exception as e:
                error_logger.log_error(e, {
                    "function": func.__name__,
                    "args": str(args)[:100],  # Truncate for privacy
                    "kwargs": str(kwargs)[:100]
                })
                raise
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage and testing
    health_monitor = HealthMonitor()
    health_report = health_monitor.check_system_health()
    
    print("System Health Report:")
    print(json.dumps(health_report, indent=2))