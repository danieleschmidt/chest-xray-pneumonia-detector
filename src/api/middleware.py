"""
Middleware components for the FastAPI application.
"""

import time
import uuid
import logging
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from ..monitoring.metrics import metrics_registry
from ..monitoring.logging import request_id_filter, audit_logger


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with logging."""
        # Generate request ID
        request_id = str(uuid.uuid4())
        request_id_filter.set_request_id(request_id)
        
        # Get logger
        logger = logging.getLogger(__name__)
        
        # Log request
        start_time = time.time()
        logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
            }
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log response
        logger.info(
            "Request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "status_code": response.status_code,
                "duration_ms": int(duration * 1000),
            }
        )
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting Prometheus metrics."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with metrics collection."""
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Extract endpoint pattern (simplified)
        endpoint = self._extract_endpoint(request.url.path)
        
        # Record metrics
        metrics_registry.record_http_request(
            method=request.method,
            endpoint=endpoint,
            status_code=response.status_code,
            duration=duration
        )
        
        return response
    
    def _extract_endpoint(self, path: str) -> str:
        """Extract endpoint pattern from path."""
        # Simplify path to endpoint pattern
        if path.startswith("/predict"):
            return "/predict"
        elif path.startswith("/health"):
            return "/health"
        elif path.startswith("/model"):
            return "/model"
        elif path.startswith("/metrics"):
            return "/metrics"
        elif path.startswith("/docs"):
            return "/docs"
        elif path == "/":
            return "/"
        else:
            return "/other"


class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for security headers and basic protection."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with security enhancements."""
        # Check for basic security issues
        if self._has_security_concerns(request):
            # Log security event
            if audit_logger:
                audit_logger.log_security_event(
                    event_type="suspicious_request",
                    severity="medium",
                    description="Request flagged for security review",
                    ip_address=request.client.host if request.client else None,
                    additional_context={
                        "method": request.method,
                        "path": request.url.path,
                        "user_agent": request.headers.get("user-agent")
                    }
                )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response
    
    def _has_security_concerns(self, request: Request) -> bool:
        """Check for basic security concerns."""
        # Check for suspicious patterns
        suspicious_patterns = [
            "../", "..\\", "<script", "javascript:", "data:text/html",
            "eval(", "document.cookie", "window.location"
        ]
        
        # Check URL and headers
        url_str = str(request.url).lower()
        user_agent = request.headers.get("user-agent", "").lower()
        
        for pattern in suspicious_patterns:
            if pattern in url_str or pattern in user_agent:
                return True
        
        return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""
    
    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.requests = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting."""
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean old entries
        self._cleanup_old_entries(current_time)
        
        # Check rate limit
        if self._is_rate_limited(client_ip, current_time):
            # Log rate limit violation
            logger = logging.getLogger(__name__)
            logger.warning(
                "Rate limit exceeded",
                extra={
                    "client_ip": client_ip,
                    "endpoint": request.url.path,
                    "method": request.method
                }
            )
            
            return Response(
                content="Rate limit exceeded",
                status_code=429,
                headers={"Retry-After": str(self.period)}
            )
        
        # Record request
        self._record_request(client_ip, current_time)
        
        return await call_next(request)
    
    def _cleanup_old_entries(self, current_time: float):
        """Remove old request entries."""
        cutoff_time = current_time - self.period
        for ip in list(self.requests.keys()):
            self.requests[ip] = [
                req_time for req_time in self.requests[ip]
                if req_time > cutoff_time
            ]
            if not self.requests[ip]:
                del self.requests[ip]
    
    def _is_rate_limited(self, client_ip: str, current_time: float) -> bool:
        """Check if client is rate limited."""
        if client_ip not in self.requests:
            return False
        
        return len(self.requests[client_ip]) >= self.calls
    
    def _record_request(self, client_ip: str, current_time: float):
        """Record a request for rate limiting."""
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        self.requests[client_ip].append(current_time)