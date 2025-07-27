"""
Metrics collection and export utilities.
"""

import time
import json
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import psutil
import os


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    inference_count: int = 0
    inference_time_total: float = 0.0
    inference_time_avg: float = 0.0
    inference_time_min: float = float('inf')
    inference_time_max: float = 0.0
    error_count: int = 0
    last_inference_time: Optional[float] = None
    model_load_time: Optional[float] = None
    model_memory_usage_mb: Optional[float] = None


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    disk_percent: float = 0.0
    disk_free_gb: float = 0.0
    load_average: Optional[List[float]] = None
    process_count: int = 0


@dataclass
class ApplicationMetrics:
    """Application-specific metrics."""
    uptime_seconds: float = 0.0
    requests_total: int = 0
    requests_per_minute: float = 0.0
    active_connections: int = 0
    cache_hit_rate: float = 0.0
    queue_size: int = 0


class MetricsCollector:
    """Comprehensive metrics collection and export."""
    
    def __init__(self, export_interval: int = 60):
        self.export_interval = export_interval
        self.start_time = time.time()
        self.model_metrics = ModelMetrics()
        self.system_metrics = SystemMetrics()
        self.app_metrics = ApplicationMetrics()
        
        # Thread-safe counters
        self._lock = threading.Lock()
        self._request_times = []
        self._inference_times = []
        
        # Custom metrics storage
        self._custom_metrics = {}
        self._custom_counters = {}
        self._custom_histograms = {}
    
    def record_inference(self, duration: float, success: bool = True):
        """Record an inference operation."""
        with self._lock:
            self.model_metrics.inference_count += 1
            self.model_metrics.last_inference_time = time.time()
            
            if success:
                self.model_metrics.inference_time_total += duration
                self.model_metrics.inference_time_avg = (
                    self.model_metrics.inference_time_total / 
                    self.model_metrics.inference_count
                )
                self.model_metrics.inference_time_min = min(
                    self.model_metrics.inference_time_min, duration
                )
                self.model_metrics.inference_time_max = max(
                    self.model_metrics.inference_time_max, duration
                )
                
                # Keep last 100 inference times for rate calculation
                self._inference_times.append((time.time(), duration))
                if len(self._inference_times) > 100:
                    self._inference_times.pop(0)
            else:
                self.model_metrics.error_count += 1
    
    def record_request(self, duration: float):
        """Record a request operation."""
        with self._lock:
            self.app_metrics.requests_total += 1
            
            # Keep last 100 request times for rate calculation
            self._request_times.append((time.time(), duration))
            if len(self._request_times) > 100:
                self._request_times.pop(0)
    
    def record_model_load(self, duration: float, memory_usage_mb: float):
        """Record model loading metrics."""
        with self._lock:
            self.model_metrics.model_load_time = duration
            self.model_metrics.model_memory_usage_mb = memory_usage_mb
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a custom counter."""
        key = f"{name}:{json.dumps(labels or {}, sort_keys=True)}"
        with self._lock:
            self._custom_counters[key] = self._custom_counters.get(key, 0) + value
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a custom gauge value."""
        key = f"{name}:{json.dumps(labels or {}, sort_keys=True)}"
        with self._lock:
            self._custom_metrics[key] = {
                'value': value,
                'timestamp': time.time(),
                'labels': labels or {}
            }
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a value in a custom histogram."""
        key = f"{name}:{json.dumps(labels or {}, sort_keys=True)}"
        with self._lock:
            if key not in self._custom_histograms:
                self._custom_histograms[key] = []
            
            self._custom_histograms[key].append({
                'value': value,
                'timestamp': time.time()
            })
            
            # Keep only last 1000 values
            if len(self._custom_histograms[key]) > 1000:
                self._custom_histograms[key] = self._custom_histograms[key][-1000:]
    
    def update_system_metrics(self):
        """Update system resource metrics."""
        try:
            # CPU metrics
            self.system_metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.system_metrics.memory_percent = memory.percent
            self.system_metrics.memory_used_mb = memory.used / (1024 * 1024)
            self.system_metrics.memory_available_mb = memory.available / (1024 * 1024)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.system_metrics.disk_percent = (disk.used / disk.total) * 100
            self.system_metrics.disk_free_gb = disk.free / (1024 * 1024 * 1024)
            
            # Load average (Unix systems)
            try:
                self.system_metrics.load_average = list(os.getloadavg())
            except (OSError, AttributeError):
                self.system_metrics.load_average = None
            
            # Process count
            self.system_metrics.process_count = len(psutil.pids())
            
        except Exception as e:
            print(f"Error updating system metrics: {e}")
    
    def update_application_metrics(self):
        """Update application-specific metrics."""
        current_time = time.time()
        self.app_metrics.uptime_seconds = current_time - self.start_time
        
        # Calculate requests per minute
        if self._request_times:
            recent_requests = [
                t for t, _ in self._request_times 
                if current_time - t <= 60
            ]
            self.app_metrics.requests_per_minute = len(recent_requests)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        self.update_system_metrics()
        self.update_application_metrics()
        
        with self._lock:
            # Calculate percentiles for inference times
            inference_percentiles = {}
            if self._inference_times:
                times = [duration for _, duration in self._inference_times]
                times.sort()
                n = len(times)
                if n > 0:
                    inference_percentiles = {
                        'p50': times[int(n * 0.5)],
                        'p90': times[int(n * 0.9)],
                        'p95': times[int(n * 0.95)],
                        'p99': times[int(n * 0.99)] if n >= 100 else times[-1]
                    }
            
            # Calculate histogram statistics
            histogram_stats = {}
            for key, values in self._custom_histograms.items():
                if values:
                    data = [v['value'] for v in values]
                    data.sort()
                    n = len(data)
                    histogram_stats[key] = {
                        'count': n,
                        'sum': sum(data),
                        'avg': sum(data) / n,
                        'min': min(data),
                        'max': max(data),
                        'p50': data[int(n * 0.5)] if n > 0 else 0,
                        'p90': data[int(n * 0.9)] if n > 0 else 0,
                        'p95': data[int(n * 0.95)] if n > 0 else 0,
                        'p99': data[int(n * 0.99)] if n >= 100 else data[-1] if data else 0
                    }
            
            return {
                'timestamp': time.time(),
                'model': asdict(self.model_metrics),
                'system': asdict(self.system_metrics),
                'application': asdict(self.app_metrics),
                'inference_percentiles': inference_percentiles,
                'custom_metrics': dict(self._custom_metrics),
                'custom_counters': dict(self._custom_counters),
                'histogram_stats': histogram_stats
            }
    
    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        metrics = self.get_all_metrics()
        lines = []
        
        # Model metrics
        model = metrics['model']
        lines.append(f'# HELP pneumonia_detector_inference_total Total number of inferences')
        lines.append(f'# TYPE pneumonia_detector_inference_total counter')
        lines.append(f'pneumonia_detector_inference_total {model["inference_count"]}')
        
        lines.append(f'# HELP pneumonia_detector_inference_duration_seconds Inference duration')
        lines.append(f'# TYPE pneumonia_detector_inference_duration_seconds summary')
        lines.append(f'pneumonia_detector_inference_duration_seconds_sum {model["inference_time_total"]}')
        lines.append(f'pneumonia_detector_inference_duration_seconds_count {model["inference_count"]}')
        
        if metrics['inference_percentiles']:
            for percentile, value in metrics['inference_percentiles'].items():
                quantile = float(percentile[1:]) / 100
                lines.append(f'pneumonia_detector_inference_duration_seconds{{quantile="{quantile}"}} {value}')
        
        lines.append(f'# HELP pneumonia_detector_errors_total Total number of errors')
        lines.append(f'# TYPE pneumonia_detector_errors_total counter')
        lines.append(f'pneumonia_detector_errors_total {model["error_count"]}')
        
        # System metrics
        system = metrics['system']
        lines.append(f'# HELP pneumonia_detector_cpu_percent CPU usage percentage')
        lines.append(f'# TYPE pneumonia_detector_cpu_percent gauge')
        lines.append(f'pneumonia_detector_cpu_percent {system["cpu_percent"]}')
        
        lines.append(f'# HELP pneumonia_detector_memory_percent Memory usage percentage')
        lines.append(f'# TYPE pneumonia_detector_memory_percent gauge')
        lines.append(f'pneumonia_detector_memory_percent {system["memory_percent"]}')
        
        lines.append(f'# HELP pneumonia_detector_memory_used_bytes Memory used in bytes')
        lines.append(f'# TYPE pneumonia_detector_memory_used_bytes gauge')
        lines.append(f'pneumonia_detector_memory_used_bytes {system["memory_used_mb"] * 1024 * 1024}')
        
        # Application metrics
        app = metrics['application']
        lines.append(f'# HELP pneumonia_detector_uptime_seconds Application uptime')
        lines.append(f'# TYPE pneumonia_detector_uptime_seconds counter')
        lines.append(f'pneumonia_detector_uptime_seconds {app["uptime_seconds"]}')
        
        lines.append(f'# HELP pneumonia_detector_requests_total Total number of requests')
        lines.append(f'# TYPE pneumonia_detector_requests_total counter')
        lines.append(f'pneumonia_detector_requests_total {app["requests_total"]}')
        
        # Custom counters
        for key, value in metrics['custom_counters'].items():
            name, labels_json = key.split(':', 1)
            labels = json.loads(labels_json)
            label_str = ','.join([f'{k}="{v}"' for k, v in labels.items()])
            label_str = f'{{{label_str}}}' if label_str else ''
            
            safe_name = name.replace('-', '_').replace('.', '_')
            lines.append(f'# TYPE pneumonia_detector_{safe_name} counter')
            lines.append(f'pneumonia_detector_{safe_name}{label_str} {value}')
        
        # Custom gauges
        for key, data in metrics['custom_metrics'].items():
            name, labels_json = key.split(':', 1)
            labels = json.loads(labels_json)
            label_str = ','.join([f'{k}="{v}"' for k, v in labels.items()])
            label_str = f'{{{label_str}}}' if label_str else ''
            
            safe_name = name.replace('-', '_').replace('.', '_')
            lines.append(f'# TYPE pneumonia_detector_{safe_name} gauge')
            lines.append(f'pneumonia_detector_{safe_name}{label_str} {data["value"]}')
        
        return '\n'.join(lines) + '\n'
    
    def export_json_format(self) -> str:
        """Export metrics in JSON format."""
        return json.dumps(self.get_all_metrics(), indent=2)
    
    def save_metrics(self, filepath: str, format: str = 'json'):
        """Save metrics to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'prometheus':
            content = self.export_prometheus_format()
        else:
            content = self.export_json_format()
        
        with open(filepath, 'w') as f:
            f.write(content)


# Global metrics collector instance
_metrics_collector = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def record_inference_time(duration: float, success: bool = True):
    """Convenience function to record inference time."""
    get_metrics_collector().record_inference(duration, success)


def record_request_time(duration: float):
    """Convenience function to record request time."""
    get_metrics_collector().record_request(duration)


def increment_counter(name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
    """Convenience function to increment a counter."""
    get_metrics_collector().increment_counter(name, value, labels)


def set_gauge(name: str, value: float, labels: Optional[Dict[str, str]] = None):
    """Convenience function to set a gauge."""
    get_metrics_collector().set_gauge(name, value, labels)


if __name__ == "__main__":
    # CLI interface for metrics
    import argparse
    
    parser = argparse.ArgumentParser(description="Metrics utility")
    parser.add_argument("--format", choices=['json', 'prometheus'], default='json', help="Output format")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--serve", action='store_true', help="Serve metrics on HTTP endpoint")
    parser.add_argument("--port", type=int, default=9090, help="Port for HTTP server")
    
    args = parser.parse_args()
    
    collector = get_metrics_collector()
    
    if args.serve:
        # Simple HTTP server for metrics (would use proper framework in production)
        import http.server
        import socketserver
        from urllib.parse import urlparse
        
        class MetricsHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/metrics':
                    self.send_response(200)
                    if args.format == 'prometheus':
                        self.send_header('Content-type', 'text/plain')
                    else:
                        self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    
                    if args.format == 'prometheus':
                        content = collector.export_prometheus_format()
                    else:
                        content = collector.export_json_format()
                    
                    self.wfile.write(content.encode())
                else:
                    self.send_error(404)
            
            def log_message(self, format, *args):
                pass  # Suppress default logging
        
        with socketserver.TCPServer(("", args.port), MetricsHandler) as httpd:
            print(f"Serving metrics on port {args.port}")
            print(f"Access metrics at: http://localhost:{args.port}/metrics")
            httpd.serve_forever()
    
    else:
        # Export metrics
        if args.format == 'prometheus':
            content = collector.export_prometheus_format()
        else:
            content = collector.export_json_format()
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(content)
            print(f"Metrics exported to {args.output}")
        else:
            print(content)