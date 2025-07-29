"""
Advanced Distributed Tracing Instrumentation - SDLC Enhancement
Comprehensive OpenTelemetry and Jaeger integration for ML systems observability.
"""

import os
import time
import functools
import logging
from typing import Dict, Any, Optional, Callable, Union, List
from contextlib import contextmanager
import threading
from dataclasses import dataclass, asdict

# OpenTelemetry imports
try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.trace import Status, StatusCode
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False
    # Provide mock implementations for environments without OpenTelemetry
    class MockTracer:
        def start_span(self, *args, **kwargs):
            return MockSpan()
    
    class MockSpan:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def set_attribute(self, key, value): pass
        def set_status(self, status): pass
        def add_event(self, name, attributes=None): pass
        def record_exception(self, exception): pass


@dataclass
class MLTraceAttributes:
    """ML-specific trace attributes for standardized observability."""
    model_name: str
    model_version: str
    input_shape: Optional[tuple] = None
    batch_size: Optional[int] = None
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    processing_time_ms: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    preprocessing_applied: Optional[List[str]] = None


class AdvancedTracer:
    """Advanced distributed tracing implementation with ML-specific instrumentation."""
    
    def __init__(self, service_name: str = "chest-xray-detector"):
        self.service_name = service_name
        self.tracer = None
        self.meter = None
        self.initialized = False
        self._setup_tracing()
        self._setup_metrics()
    
    def _setup_tracing(self):
        """Initialize distributed tracing with Jaeger and OpenTelemetry."""
        if not TRACING_AVAILABLE:
            logging.warning("OpenTelemetry not available, using mock tracer")
            self.tracer = MockTracer()
            return
        
        # Configure resource attributes
        resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: self.service_name,
            ResourceAttributes.SERVICE_VERSION: os.getenv("APP_VERSION", "2.0.0"),
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: os.getenv("ENVIRONMENT", "development"),
            "ml.model.framework": "tensorflow",
            "ml.task.type": "image_classification",
            "healthcare.domain": "radiology"
        })
        
        # Initialize tracer provider
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)
        
        # Configure Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name=os.getenv("JAEGER_AGENT_HOST", "localhost"),
            agent_port=int(os.getenv("JAEGER_AGENT_PORT", "6831")),
            collector_endpoint=os.getenv("JAEGER_COLLECTOR_ENDPOINT"),
        )
        
        # Add batch span processor
        span_processor = BatchSpanProcessor(
            jaeger_exporter,
            max_export_batch_size=512,
            export_timeout_millis=30000,
            schedule_delay_millis=5000
        )
        tracer_provider.add_span_processor(span_processor)
        
        # Get tracer instance
        self.tracer = trace.get_tracer(__name__)
        
        # Auto-instrument common libraries
        self._auto_instrument()
        
        self.initialized = True
        logging.info(f"Distributed tracing initialized for {self.service_name}")
    
    def _setup_metrics(self):
        """Initialize metrics collection with Prometheus integration."""
        if not TRACING_AVAILABLE:
            return
        
        # Configure Prometheus metric reader
        prometheus_reader = PrometheusMetricReader(port=8090)
        
        # Initialize meter provider
        meter_provider = MeterProvider(
            metric_readers=[prometheus_reader],
            resource=Resource.create({
                ResourceAttributes.SERVICE_NAME: self.service_name
            })
        )
        metrics.set_meter_provider(meter_provider)
        
        # Get meter instance
        self.meter = metrics.get_meter(__name__)
        
        # Create ML-specific metrics
        self._create_ml_metrics()
    
    def _create_ml_metrics(self):
        """Create ML-specific metrics for observability."""
        if not self.meter:
            return
        
        # Model inference metrics
        self.inference_duration_histogram = self.meter.create_histogram(
            name="model_inference_duration_seconds",
            description="Time spent on model inference",
            unit="s"
        )
        
        self.inference_counter = self.meter.create_counter(
            name="model_inference_total",
            description="Total number of model inferences"
        )
        
        self.model_accuracy_gauge = self.meter.create_up_down_counter(
            name="model_accuracy_gauge",
            description="Current model accuracy"
        )
        
        # Data processing metrics
        self.preprocessing_duration_histogram = self.meter.create_histogram(
            name="data_processing_duration_seconds",
            description="Time spent on data preprocessing",
            unit="s"
        )
        
        # System resource metrics
        self.gpu_memory_gauge = self.meter.create_up_down_counter(
            name="gpu_memory_usage_bytes",
            description="GPU memory usage"
        )
    
    def _auto_instrument(self):
        """Automatically instrument common libraries."""
        try:
            # Instrument HTTP requests
            RequestsInstrumentor().instrument()
            
            # Instrument logging
            LoggingInstrumentor().instrument(set_logging_format=True)
            
            logging.info("Auto-instrumentation completed")
        except Exception as e:
            logging.warning(f"Auto-instrumentation failed: {e}")
    
    @contextmanager
    def trace_span(self, 
                   operation_name: str, 
                   attributes: Optional[Dict[str, Any]] = None,
                   ml_attributes: Optional[MLTraceAttributes] = None):
        """Create a traced span with optional ML-specific attributes."""
        if not self.tracer:
            yield None
            return
        
        with self.tracer.start_as_current_span(operation_name) as span:
            try:
                # Set basic attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, str(value))
                
                # Set ML-specific attributes
                if ml_attributes:
                    ml_attrs = asdict(ml_attributes)
                    for key, value in ml_attrs.items():
                        if value is not None:
                            if isinstance(value, list):
                                span.set_attribute(f"ml.{key}", ",".join(map(str, value)))
                            else:
                                span.set_attribute(f"ml.{key}", str(value))
                
                # Set span status as OK initially
                span.set_status(Status(StatusCode.OK))
                
                yield span
                
            except Exception as e:
                # Record exception and set error status
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
    
    def trace_ml_inference(self, 
                          model_name: str, 
                          model_version: str,
                          input_shape: Optional[tuple] = None,
                          batch_size: Optional[int] = None):
        """Decorator for tracing ML model inference operations."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                ml_attrs = MLTraceAttributes(
                    model_name=model_name,
                    model_version=model_version,
                    input_shape=input_shape,
                    batch_size=batch_size
                )
                
                with self.trace_span(
                    f"ml_inference.{model_name}",
                    attributes={"operation.type": "ml_inference"},
                    ml_attributes=ml_attrs
                ) as span:
                    try:
                        result = func(*args, **kwargs)
                        
                        # Extract prediction information if available
                        if isinstance(result, dict):
                            if "prediction" in result:
                                span.set_attribute("ml.prediction", str(result["prediction"]))
                            if "confidence" in result:
                                span.set_attribute("ml.confidence", str(result["confidence"]))
                        
                        # Record processing time
                        processing_time_ms = (time.time() - start_time) * 1000
                        span.set_attribute("ml.processing_time_ms", str(processing_time_ms))
                        
                        # Record metrics
                        if self.inference_duration_histogram:
                            self.inference_duration_histogram.record(
                                time.time() - start_time,
                                {"model_name": model_name, "model_version": model_version}
                            )
                        
                        if self.inference_counter:
                            self.inference_counter.add(
                                1,
                                {"model_name": model_name, "model_version": model_version}
                            )
                        
                        return result
                        
                    except Exception as e:
                        span.add_event("inference_error", {
                            "error.type": type(e).__name__,
                            "error.message": str(e)
                        })
                        raise
            
            return wrapper
        return decorator
    
    def trace_data_processing(self, processing_type: str):
        """Decorator for tracing data processing operations."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                with self.trace_span(
                    f"data_processing.{processing_type}",
                    attributes={
                        "operation.type": "data_processing",
                        "processing.type": processing_type
                    }
                ) as span:
                    try:
                        result = func(*args, **kwargs)
                        
                        # Record processing time
                        processing_time = time.time() - start_time
                        span.set_attribute("processing.duration_ms", str(processing_time * 1000))
                        
                        # Record metrics
                        if self.preprocessing_duration_histogram:
                            self.preprocessing_duration_histogram.record(
                                processing_time,
                                {"processing_type": processing_type}
                            )
                        
                        return result
                        
                    except Exception as e:
                        span.add_event("processing_error", {
                            "error.type": type(e).__name__,
                            "error.message": str(e)
                        })
                        raise
            
            return wrapper
        return decorator
    
    def trace_api_endpoint(self, endpoint_name: str, method: str = "POST"):
        """Decorator for tracing API endpoint operations."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.trace_span(
                    f"api.{endpoint_name}",
                    attributes={
                        "http.method": method,
                        "http.route": endpoint_name,
                        "operation.type": "http_request"
                    }
                ) as span:
                    try:
                        result = func(*args, **kwargs)
                        
                        # Set response attributes if available
                        if isinstance(result, dict):
                            if "status_code" in result:
                                span.set_attribute("http.status_code", str(result["status_code"]))
                        
                        return result
                        
                    except Exception as e:
                        span.add_event("api_error", {
                            "error.type": type(e).__name__,
                            "error.message": str(e),
                            "http.status_code": "500"
                        })
                        raise
            
            return wrapper
        return decorator
    
    def add_custom_event(self, event_name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add a custom event to the current span."""
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.add_event(event_name, attributes or {})
    
    def set_custom_attribute(self, key: str, value: Any):
        """Set a custom attribute on the current span."""
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.set_attribute(key, str(value))
    
    def record_ml_metrics(self, 
                         model_name: str,
                         accuracy: Optional[float] = None,
                         inference_count: Optional[int] = None,
                         gpu_memory_mb: Optional[float] = None):
        """Record ML-specific metrics."""
        labels = {"model_name": model_name}
        
        if accuracy is not None and self.model_accuracy_gauge:
            self.model_accuracy_gauge.add(accuracy, labels)
        
        if inference_count is not None and self.inference_counter:
            self.inference_counter.add(inference_count, labels)
        
        if gpu_memory_mb is not None and self.gpu_memory_gauge:
            self.gpu_memory_gauge.add(gpu_memory_mb * 1024 * 1024, labels)  # Convert to bytes


# Global tracer instance
_global_tracer: Optional[AdvancedTracer] = None


def get_tracer() -> AdvancedTracer:
    """Get or create the global tracer instance."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = AdvancedTracer()
    return _global_tracer


def initialize_tracing(service_name: str = "chest-xray-detector") -> AdvancedTracer:
    """Initialize distributed tracing for the application."""
    global _global_tracer
    _global_tracer = AdvancedTracer(service_name)
    return _global_tracer


# Convenience decorators using global tracer
def trace_ml_inference(model_name: str, model_version: str, **kwargs):
    """Convenience decorator for ML inference tracing."""
    tracer = get_tracer()
    return tracer.trace_ml_inference(model_name, model_version, **kwargs)


def trace_data_processing(processing_type: str):
    """Convenience decorator for data processing tracing."""
    tracer = get_tracer()
    return tracer.trace_data_processing(processing_type)


def trace_api_endpoint(endpoint_name: str, method: str = "POST"):
    """Convenience decorator for API endpoint tracing."""
    tracer = get_tracer()
    return tracer.trace_api_endpoint(endpoint_name, method)


# Context manager for custom spans
@contextmanager
def custom_span(operation_name: str, attributes: Optional[Dict[str, Any]] = None):
    """Create a custom traced span."""
    tracer = get_tracer()
    with tracer.trace_span(operation_name, attributes) as span:
        yield span


# Example usage and integration helper
class TracingIntegration:
    """Helper class for integrating tracing into existing ML pipeline."""
    
    @staticmethod
    def instrument_tensorflow_model(model, model_name: str, model_version: str):
        """Instrument TensorFlow model with tracing (placeholder)."""
        # In a real implementation, this would wrap TensorFlow operations
        logging.info(f"Tracing instrumentation added to {model_name} v{model_version}")
        return model
    
    @staticmethod
    def instrument_data_pipeline(pipeline_steps: List[Callable]) -> List[Callable]:
        """Instrument data pipeline steps with tracing."""
        instrumented_steps = []
        for i, step in enumerate(pipeline_steps):
            instrumented_step = trace_data_processing(f"pipeline_step_{i}")(step)
            instrumented_steps.append(instrumented_step)
        return instrumented_steps
    
    @staticmethod
    def create_health_check_tracer():
        """Create a lightweight tracer for health checks."""
        @trace_api_endpoint("health_check", "GET")
        def health_check():
            return {"status": "healthy", "timestamp": time.time()}
        
        return health_check


if __name__ == "__main__":
    # Example usage
    tracer = initialize_tracing("chest-xray-detector-test")
    
    @trace_ml_inference("pneumonia-detector", "v2.1", batch_size=32)
    def example_inference(image_batch):
        time.sleep(0.1)  # Simulate inference
        return {"prediction": "pneumonia", "confidence": 0.87}
    
    @trace_data_processing("image_preprocessing")
    def example_preprocessing(images):
        time.sleep(0.05)  # Simulate preprocessing
        return images
    
    # Test the tracing
    with custom_span("test_pipeline") as span:
        preprocessed_images = example_preprocessing(["image1", "image2"])
        results = example_inference(preprocessed_images)
        tracer.add_custom_event("pipeline_completed", {"results_count": len(results)})
    
    print("Tracing instrumentation test completed")