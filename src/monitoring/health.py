"""
Health check components for the Chest X-Ray Pneumonia Detector API.
"""

import asyncio
import time
import psutil
import aiofiles
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..api.models import HealthResponse, HealthCheck


class HealthChecker:
    """Centralized health checking for the application."""
    
    def __init__(self):
        self.start_time = time.time()
        self.checks = {
            'model_availability': self._check_model_availability,
            'disk_space': self._check_disk_space,
            'memory_usage': self._check_memory_usage,
            'database_connection': self._check_database_connection,
            'redis_connection': self._check_redis_connection,
        }
    
    async def check_health(self) -> HealthResponse:
        """Run all health checks and return overall status."""
        check_results = []
        overall_healthy = True
        
        for check_name, check_func in self.checks.items():
            start_time = time.time()
            try:
                is_healthy, message = await check_func()
                duration_ms = int((time.time() - start_time) * 1000)
                
                check_results.append(HealthCheck(
                    name=check_name,
                    status="healthy" if is_healthy else "unhealthy",
                    message=message,
                    duration_ms=duration_ms
                ))
                
                if not is_healthy:
                    overall_healthy = False
                    
            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                check_results.append(HealthCheck(
                    name=check_name,
                    status="unhealthy",
                    message=f"Check failed: {str(e)}",
                    duration_ms=duration_ms
                ))
                overall_healthy = False
        
        uptime_seconds = int(time.time() - self.start_time)
        
        return HealthResponse(
            status="healthy" if overall_healthy else "unhealthy",
            timestamp=datetime.utcnow(),
            checks=check_results,
            uptime_seconds=uptime_seconds,
            version="0.2.0"
        )
    
    async def check_readiness(self) -> bool:
        """Check if the application is ready to serve requests."""
        # For readiness, we only check critical components
        critical_checks = ['model_availability', 'disk_space']
        
        for check_name in critical_checks:
            if check_name in self.checks:
                try:
                    is_healthy, _ = await self.checks[check_name]()
                    if not is_healthy:
                        return False
                except Exception:
                    return False
        
        return True
    
    async def _check_model_availability(self) -> tuple[bool, str]:
        """Check if the ML model is available and loaded."""
        try:
            # Check if model files exist
            model_paths = [
                Path("saved_models"),
                Path("/app/saved_models")
            ]
            
            model_found = False
            for model_path in model_paths:
                if model_path.exists():
                    model_files = list(model_path.glob("*.keras")) + list(model_path.glob("*.h5"))
                    if model_files:
                        model_found = True
                        break
            
            if not model_found:
                return False, "No model files found"
            
            # TODO: Add actual model loading check when model service is implemented
            return True, "Model available"
            
        except Exception as e:
            return False, f"Model check failed: {str(e)}"
    
    async def _check_disk_space(self) -> tuple[bool, str]:
        """Check available disk space."""
        try:
            disk_usage = psutil.disk_usage('/')
            free_percent = (disk_usage.free / disk_usage.total) * 100
            
            if free_percent < 10:
                return False, f"Low disk space: {free_percent:.1f}% free"
            elif free_percent < 20:
                return True, f"Disk space warning: {free_percent:.1f}% free"
            else:
                return True, f"Disk space OK: {free_percent:.1f}% free"
                
        except Exception as e:
            return False, f"Disk space check failed: {str(e)}"
    
    async def _check_memory_usage(self) -> tuple[bool, str]:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            if memory_percent > 90:
                return False, f"High memory usage: {memory_percent:.1f}%"
            elif memory_percent > 80:
                return True, f"Memory usage warning: {memory_percent:.1f}%"
            else:
                return True, f"Memory usage OK: {memory_percent:.1f}%"
                
        except Exception as e:
            return False, f"Memory check failed: {str(e)}"
    
    async def _check_database_connection(self) -> tuple[bool, str]:
        """Check database connection."""
        try:
            # TODO: Implement actual database connection check
            # For now, assume healthy if no explicit database is configured
            return True, "Database check skipped (not configured)"
            
        except Exception as e:
            return False, f"Database check failed: {str(e)}"
    
    async def _check_redis_connection(self) -> tuple[bool, str]:
        """Check Redis connection."""
        try:
            # TODO: Implement actual Redis connection check
            # For now, assume healthy if no explicit Redis is configured
            return True, "Redis check skipped (not configured)"
            
        except Exception as e:
            return False, f"Redis check failed: {str(e)}"


class PerformanceMonitor:
    """Monitor application performance metrics."""
    
    def __init__(self):
        self.request_times = []
        self.error_counts = {}
        self.start_time = time.time()
    
    def record_request(self, duration: float, endpoint: str, status_code: int):
        """Record a request for performance monitoring."""
        self.request_times.append({
            'duration': duration,
            'endpoint': endpoint,
            'status_code': status_code,
            'timestamp': time.time()
        })
        
        # Keep only last 1000 requests
        if len(self.request_times) > 1000:
            self.request_times = self.request_times[-1000:]
    
    def record_error(self, error_type: str):
        """Record an error occurrence."""
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        if not self.request_times:
            return {
                'average_response_time': 0,
                'requests_per_minute': 0,
                'error_rate': 0,
                'uptime_seconds': int(time.time() - self.start_time)
            }
        
        # Calculate average response time
        recent_requests = [
            req for req in self.request_times 
            if time.time() - req['timestamp'] < 300  # Last 5 minutes
        ]
        
        if recent_requests:
            avg_response_time = sum(req['duration'] for req in recent_requests) / len(recent_requests)
            requests_per_minute = len([
                req for req in recent_requests 
                if time.time() - req['timestamp'] < 60
            ])
            
            # Calculate error rate
            error_requests = [req for req in recent_requests if req['status_code'] >= 400]
            error_rate = len(error_requests) / len(recent_requests) * 100 if recent_requests else 0
        else:
            avg_response_time = 0
            requests_per_minute = 0
            error_rate = 0
        
        return {
            'average_response_time': avg_response_time,
            'requests_per_minute': requests_per_minute,
            'error_rate': error_rate,
            'uptime_seconds': int(time.time() - self.start_time),
            'total_requests': len(self.request_times),
            'error_counts': self.error_counts.copy()
        }


# Global instances
health_checker = HealthChecker()
performance_monitor = PerformanceMonitor()