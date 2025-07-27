#!/usr/bin/env python3
"""
Health check script for Docker container.
"""

import sys
import requests
import time
from pathlib import Path


def check_api_health():
    """Check if the API is responding to health checks."""
    try:
        response = requests.get(
            "http://localhost:8080/health",
            timeout=5
        )
        if response.status_code == 200:
            health_data = response.json()
            if health_data.get("status") == "healthy":
                return True
        return False
    except Exception:
        return False


def check_model_availability():
    """Check if required model files are available."""
    model_paths = [
        "/app/saved_models",
        "/app/src"
    ]
    
    for path in model_paths:
        if not Path(path).exists():
            return False
    
    return True


def check_disk_space():
    """Check available disk space."""
    import shutil
    
    try:
        total, used, free = shutil.disk_usage("/app")
        free_percent = (free / total) * 100
        return free_percent > 10  # At least 10% free space
    except Exception:
        return False


def main():
    """Main health check function."""
    checks = [
        ("API Health", check_api_health),
        ("Model Availability", check_model_availability),
        ("Disk Space", check_disk_space),
    ]
    
    failed_checks = []
    
    for check_name, check_func in checks:
        try:
            if not check_func():
                failed_checks.append(check_name)
        except Exception as e:
            failed_checks.append(f"{check_name} (error: {str(e)})")
    
    if failed_checks:
        print(f"Health check failed: {', '.join(failed_checks)}")
        sys.exit(1)
    else:
        print("Health check passed")
        sys.exit(0)


if __name__ == "__main__":
    main()