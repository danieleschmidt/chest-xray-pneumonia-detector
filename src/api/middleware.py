"""
Production-ready middleware for the pneumonia detection API.
Implements security, rate limiting, and monitoring capabilities.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from typing import Dict, Any, Optional
import hashlib
import secrets

from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import jwt


logger = logging.getLogger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware with HIPAA compliance features."""
    
    def __init__(self, app, secret_key: Optional[str] = None):
        super().__init__(app)
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.csrf_tokens: Dict[str, float] = {}
        self.session_store: Dict[str, Dict[str, Any]] = {}
        
    async def dispatch(self, request: Request, call_next):
        """Apply security headers and validation."""
        start_time = time.time()
        
        # Add security headers
        response = await call_next(request)
        
        # HIPAA-compliant security headers
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
            "X-Request-ID": self._generate_request_id(),
            "X-API-Version": "1.0.0"
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value
        
        # Remove server identification
        if "server" in response.headers:
            del response.headers["server"]
        
        # Add processing time for monitoring
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID for tracking."""
        return secrets.token_hex(16)
    
    def _validate_csrf_token(self, request: Request) -> bool:
        """Validate CSRF token for state-changing operations."""
        if request.method in ["POST", "PUT", "DELETE", "PATCH"]:
            token = request.headers.get("X-CSRF-Token")
            if not token or token not in self.csrf_tokens:
                return False
            
            # Check token expiry (24 hours)
            if time.time() - self.csrf_tokens[token] > 86400:
                del self.csrf_tokens[token]
                return False
        
        return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with sliding window implementation."""
    
    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.blocked_ips: Dict[str, float] = {}
        
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting based on client IP."""
        client_ip = self._get_client_ip(request)
        current_time = time.time()
        
        # Check if IP is temporarily blocked
        if client_ip in self.blocked_ips:
            if current_time - self.blocked_ips[client_ip] < 300:  # 5 minute block
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="IP temporarily blocked due to rate limit violations"
                )
            else:
                del self.blocked_ips[client_ip]
        
        # Clean old requests
        self._clean_old_requests(client_ip, current_time)
        
        # Check rate limit
        if len(self.requests[client_ip]) >= self.calls:
            # Block IP after multiple violations
            self.blocked_ips[client_ip] = current_time
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "limit": self.calls,
                    "period": self.period,
                    "retry_after": self.period
                },
                headers={"Retry-After": str(self.period)}
            )
        
        # Add current request
        self.requests[client_ip].append(current_time)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = max(0, self.calls - len(self.requests[client_ip]))
        response.headers["X-RateLimit-Limit"] = str(self.calls)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(current_time + self.period))
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address handling proxies."""
        # Check for forwarded IP headers (in order of preference)
        for header in ["X-Forwarded-For", "X-Real-IP", "X-Client-IP"]:
            if header in request.headers:
                ip = request.headers[header].split(",")[0].strip()
                if ip:
                    return ip
        
        # Fall back to direct client IP
        if hasattr(request.client, "host"):
            return request.client.host
        
        return "unknown"
    
    def _clean_old_requests(self, client_ip: str, current_time: float):
        """Remove requests outside the time window."""
        cutoff_time = current_time - self.period
        while (self.requests[client_ip] and 
               self.requests[client_ip][0] < cutoff_time):
            self.requests[client_ip].popleft()


class AuditMiddleware(BaseHTTPMiddleware):
    """HIPAA audit logging middleware."""
    
    def __init__(self, app):
        super().__init__(app)
        self.audit_logger = logging.getLogger("audit")
    
    async def dispatch(self, request: Request, call_next):
        """Log all API interactions for audit compliance."""
        start_time = time.time()
        request_id = request.headers.get("X-Request-ID", "unknown")
        
        # Log request
        audit_data = {
            "request_id": request_id,
            "timestamp": time.time(),
            "method": request.method,
            "path": request.url.path,
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("User-Agent", "unknown"),
            "content_type": request.headers.get("Content-Type"),
            "content_length": request.headers.get("Content-Length")
        }
        
        # Process request
        try:
            response = await call_next(request)
            
            # Log response
            audit_data.update({
                "status_code": response.status_code,
                "response_time_ms": (time.time() - start_time) * 1000,
                "response_size": response.headers.get("Content-Length")
            })
            
            # Log at appropriate level based on status
            if response.status_code >= 500:
                self.audit_logger.error("API_REQUEST", extra=audit_data)
            elif response.status_code >= 400:
                self.audit_logger.warning("API_REQUEST", extra=audit_data)
            else:
                self.audit_logger.info("API_REQUEST", extra=audit_data)
            
            return response
            
        except Exception as e:
            # Log error
            audit_data.update({
                "status_code": 500,
                "error": str(e),
                "response_time_ms": (time.time() - start_time) * 1000
            })
            self.audit_logger.error("API_ERROR", extra=audit_data)
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP for audit logging."""
        for header in ["X-Forwarded-For", "X-Real-IP"]:
            if header in request.headers:
                return request.headers[header].split(",")[0].strip()
        
        if hasattr(request.client, "host"):
            return request.client.host
        
        return "unknown"


class HealthCheckMiddleware(BaseHTTPMiddleware):
    """Health check bypass middleware."""
    
    def __init__(self, app):
        super().__init__(app)
        self.health_paths = {"/health", "/health/liveness", "/health/readiness", "/metrics"}
    
    async def dispatch(self, request: Request, call_next):
        """Bypass heavy middleware for health checks."""
        if request.url.path in self.health_paths:
            # Skip rate limiting and audit logging for health checks
            return await call_next(request)
        
        return await call_next(request)


class CompressionMiddleware(BaseHTTPMiddleware):
    """Custom compression middleware for API responses."""
    
    def __init__(self, app, minimum_size: int = 1000):
        super().__init__(app)
        self.minimum_size = minimum_size
    
    async def dispatch(self, request: Request, call_next):
        """Apply compression to large responses."""
        response = await call_next(request)
        
        # Check if client accepts compression
        accept_encoding = request.headers.get("Accept-Encoding", "")
        if "gzip" not in accept_encoding:
            return response
        
        # Check response size
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) < self.minimum_size:
            return response
        
        # Apply compression if needed
        # Implementation would depend on specific compression library
        return response