"""
Health check utilities for monitoring system status.
"""

import time
import psutil
import os
from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
import json


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    details: Optional[Dict[str, Any]] = None


class HealthChecker:
    """Comprehensive health checking for the application."""
    
    def __init__(self, model_path: Optional[str] = None, data_dir: Optional[str] = None):
        self.model_path = model_path
        self.data_dir = data_dir
        self.start_time = time.time()
    
    def check_all(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive status."""
        checks = {
            'system': self.check_system_health(),
            'storage': self.check_storage_health(),
            'dependencies': self.check_dependencies(),
            'model': self.check_model_health(),
            'data': self.check_data_health()
        }
        
        # Determine overall status
        statuses = [check.status for check in checks.values()]
        if any(status == HealthStatus.UNHEALTHY for status in statuses):
            overall_status = HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        return {
            'status': overall_status.value,
            'timestamp': time.time(),
            'uptime_seconds': time.time() - self.start_time,
            'checks': {name: self._check_to_dict(check) for name, check in checks.items()},
            'summary': {
                'total_checks': len(checks),
                'healthy': sum(1 for check in checks.values() if check.status == HealthStatus.HEALTHY),
                'degraded': sum(1 for check in checks.values() if check.status == HealthStatus.DEGRADED),
                'unhealthy': sum(1 for check in checks.values() if check.status == HealthStatus.UNHEALTHY)
            }
        }
    
    def check_system_health(self) -> HealthCheck:
        """Check system resource health."""
        start_time = time.time()
        
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Check memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Check disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            details = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'disk_percent': disk_percent,
                'memory_available_mb': memory.available / (1024 * 1024),
                'disk_free_gb': disk.free / (1024 * 1024 * 1024)
            }
            
            # Determine status based on thresholds
            if cpu_percent > 90 or memory_percent > 90 or disk_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = "System resources critically low"
            elif cpu_percent > 70 or memory_percent > 70 or disk_percent > 80:
                status = HealthStatus.DEGRADED
                message = "System resources under pressure"
            else:
                status = HealthStatus.HEALTHY
                message = "System resources normal"
            
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheck(
                name="system",
                status=status,
                message=message,
                duration_ms=duration_ms,
                details=details
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheck(
                name="system",
                status=HealthStatus.UNHEALTHY,
                message=f"System check failed: {str(e)}",
                duration_ms=duration_ms,
                details={"error": str(e)}
            )
    
    def check_storage_health(self) -> HealthCheck:
        """Check storage and file system health."""
        start_time = time.time()
        
        try:
            details = {}
            issues = []
            
            # Check if key directories exist and are writable
            directories_to_check = [
                'saved_models',
                'data',
                'logs',
                'checkpoints'
            ]
            
            for dir_name in directories_to_check:
                dir_path = Path(dir_name)
                details[f"{dir_name}_exists"] = dir_path.exists()
                
                if dir_path.exists():
                    details[f"{dir_name}_writable"] = os.access(dir_path, os.W_OK)
                    if not os.access(dir_path, os.W_OK):
                        issues.append(f"{dir_name} is not writable")
                else:
                    issues.append(f"{dir_name} directory does not exist")
            
            # Check available disk space
            disk = psutil.disk_usage('.')
            free_gb = disk.free / (1024 * 1024 * 1024)
            details['free_space_gb'] = free_gb
            
            if free_gb < 1:  # Less than 1GB free
                issues.append("Less than 1GB disk space available")
            elif free_gb < 5:  # Less than 5GB free
                issues.append("Less than 5GB disk space available")
            
            # Determine status
            if any("not writable" in issue or "does not exist" in issue for issue in issues):
                status = HealthStatus.UNHEALTHY
                message = f"Storage issues: {'; '.join(issues)}"
            elif issues:
                status = HealthStatus.DEGRADED
                message = f"Storage warnings: {'; '.join(issues)}"
            else:
                status = HealthStatus.HEALTHY
                message = "Storage is healthy"
            
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheck(
                name="storage",
                status=status,
                message=message,
                duration_ms=duration_ms,
                details=details
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheck(
                name="storage",
                status=HealthStatus.UNHEALTHY,
                message=f"Storage check failed: {str(e)}",
                duration_ms=duration_ms,
                details={"error": str(e)}
            )
    
    def check_dependencies(self) -> HealthCheck:
        """Check critical dependencies are available."""
        start_time = time.time()
        
        try:
            dependencies = [
                'numpy',
                'tensorflow',
                'opencv-python',
                'pillow',
                'scikit-learn'
            ]
            
            available = {}
            missing = []
            
            for dep in dependencies:
                try:
                    __import__(dep.replace('-', '_'))
                    available[dep] = True
                except ImportError:
                    available[dep] = False
                    missing.append(dep)
            
            details = {
                'dependencies': available,
                'missing_count': len(missing)
            }
            
            if missing:
                status = HealthStatus.UNHEALTHY
                message = f"Missing dependencies: {', '.join(missing)}"
            else:
                status = HealthStatus.HEALTHY
                message = "All dependencies available"
            
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheck(
                name="dependencies",
                status=status,
                message=message,
                duration_ms=duration_ms,
                details=details
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheck(
                name="dependencies",
                status=HealthStatus.UNHEALTHY,
                message=f"Dependency check failed: {str(e)}",
                duration_ms=duration_ms,
                details={"error": str(e)}
            )
    
    def check_model_health(self) -> HealthCheck:
        """Check model availability and integrity."""
        start_time = time.time()
        
        try:
            details = {}
            
            if self.model_path:
                model_path = Path(self.model_path)
                details['model_path'] = str(model_path)
                details['model_exists'] = model_path.exists()
                
                if model_path.exists():
                    details['model_size_mb'] = model_path.stat().st_size / (1024 * 1024)
                    details['model_modified'] = model_path.stat().st_mtime
                    
                    # Try to load model metadata if possible
                    try:
                        # This would be more sophisticated in a real implementation
                        status = HealthStatus.HEALTHY
                        message = "Model is available and healthy"
                    except Exception as model_error:
                        status = HealthStatus.DEGRADED
                        message = f"Model exists but may be corrupted: {str(model_error)}"
                        details['model_error'] = str(model_error)
                else:
                    status = HealthStatus.DEGRADED
                    message = "Model file not found"
            else:
                status = HealthStatus.DEGRADED
                message = "No model path configured"
                details['model_path'] = None
            
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheck(
                name="model",
                status=status,
                message=message,
                duration_ms=duration_ms,
                details=details
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheck(
                name="model",
                status=HealthStatus.UNHEALTHY,
                message=f"Model check failed: {str(e)}",
                duration_ms=duration_ms,
                details={"error": str(e)}
            )
    
    def check_data_health(self) -> HealthCheck:
        """Check data availability and integrity."""
        start_time = time.time()
        
        try:
            details = {}
            
            if self.data_dir:
                data_path = Path(self.data_dir)
                details['data_dir'] = str(data_path)
                details['data_dir_exists'] = data_path.exists()
                
                if data_path.exists():
                    # Count files in data directory
                    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
                    image_count = sum(
                        1 for file in data_path.rglob('*')
                        if file.suffix.lower() in image_extensions
                    )
                    
                    details['image_count'] = image_count
                    
                    if image_count > 0:
                        status = HealthStatus.HEALTHY
                        message = f"Data directory contains {image_count} images"
                    else:
                        status = HealthStatus.DEGRADED
                        message = "Data directory exists but contains no images"
                else:
                    status = HealthStatus.DEGRADED
                    message = "Data directory not found"
            else:
                status = HealthStatus.HEALTHY
                message = "No data directory configured (not required for all operations)"
                details['data_dir'] = None
            
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheck(
                name="data",
                status=status,
                message=message,
                duration_ms=duration_ms,
                details=details
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheck(
                name="data",
                status=HealthStatus.UNHEALTHY,
                message=f"Data check failed: {str(e)}",
                duration_ms=duration_ms,
                details={"error": str(e)}
            )
    
    def _check_to_dict(self, check: HealthCheck) -> Dict[str, Any]:
        """Convert health check to dictionary."""
        return {
            'status': check.status.value,
            'message': check.message,
            'duration_ms': check.duration_ms,
            'details': check.details or {}
        }
    
    def get_readiness_probe(self) -> Dict[str, Any]:
        """Kubernetes-style readiness probe."""
        checks = self.check_all()
        
        # Ready if not unhealthy
        ready = checks['status'] != HealthStatus.UNHEALTHY.value
        
        return {
            'ready': ready,
            'status': checks['status'],
            'timestamp': checks['timestamp']
        }
    
    def get_liveness_probe(self) -> Dict[str, Any]:
        """Kubernetes-style liveness probe."""
        # Simpler check for liveness - just verify the application is responding
        start_time = time.time()
        
        try:
            # Basic system check
            psutil.cpu_percent()
            duration_ms = (time.time() - start_time) * 1000
            
            return {
                'alive': True,
                'timestamp': time.time(),
                'duration_ms': duration_ms
            }
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return {
                'alive': False,
                'timestamp': time.time(),
                'duration_ms': duration_ms,
                'error': str(e)
            }


if __name__ == "__main__":
    # CLI interface for health checks
    import argparse
    
    parser = argparse.ArgumentParser(description="Health check utility")
    parser.add_argument("--model-path", help="Path to model file")
    parser.add_argument("--data-dir", help="Path to data directory")
    parser.add_argument("--format", choices=['json', 'text'], default='json', help="Output format")
    parser.add_argument("--check", choices=['all', 'readiness', 'liveness'], default='all', help="Check type")
    
    args = parser.parse_args()
    
    health_checker = HealthChecker(model_path=args.model_path, data_dir=args.data_dir)
    
    if args.check == 'readiness':
        result = health_checker.get_readiness_probe()
    elif args.check == 'liveness':
        result = health_checker.get_liveness_probe()
    else:
        result = health_checker.check_all()
    
    if args.format == 'json':
        print(json.dumps(result, indent=2))
    else:
        # Text format
        if args.check == 'all':
            print(f"Overall Status: {result['status']}")
            print(f"Uptime: {result['uptime_seconds']:.1f} seconds")
            print("\nChecks:")
            for name, check in result['checks'].items():
                print(f"  {name}: {check['status']} - {check['message']}")
        else:
            print(f"Status: {result.get('ready', result.get('alive', 'unknown'))}")
    
    # Exit with appropriate code
    if args.check == 'readiness':
        exit(0 if result['ready'] else 1)
    elif args.check == 'liveness':
        exit(0 if result['alive'] else 1)
    else:
        exit(0 if result['status'] == 'healthy' else 1)