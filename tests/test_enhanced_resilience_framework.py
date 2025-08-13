"""Comprehensive tests for the Enhanced Resilience Framework.

This module tests advanced error recovery, circuit breakers, and self-healing
capabilities for mission-critical medical AI inference systems.
"""

import asyncio
import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from dataclasses import dataclass

# Import the modules to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from enhanced_resilience_framework import (
        SystemState, CircuitState, HealthMetrics, CircuitBreaker,
        AdaptiveRetryStrategy, SelfHealingManager, MedicalAIResilienceFramework,
        PerformanceMonitor, AnomalyDetector, resilient_inference
    )
except ImportError:
    # Fallback for testing environment
    CircuitBreaker = Mock
    AdaptiveRetryStrategy = Mock
    SelfHealingManager = Mock
    HealthMetrics = Mock


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1.0,
            expected_exception=RuntimeError
        )
        
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        assert self.circuit_breaker.failure_threshold == 3
        assert self.circuit_breaker.recovery_timeout == 1.0
        assert self.circuit_breaker.state == CircuitState.CLOSED
        assert self.circuit_breaker.failure_count == 0
        
    def test_successful_execution_maintains_closed_state(self):
        """Test that successful executions keep circuit closed."""
        
        @self.circuit_breaker
        def successful_function():
            return "success"
            
        result = successful_function()
        assert result == "success"
        assert self.circuit_breaker.state == CircuitState.CLOSED
        assert self.circuit_breaker.failure_count == 0
        
    def test_failures_increment_count(self):
        """Test that failures increment the failure count."""
        
        @self.circuit_breaker
        def failing_function():
            raise RuntimeError("Test failure")
            
        # First failure
        with pytest.raises(RuntimeError):
            failing_function()
        assert self.circuit_breaker.failure_count == 1
        assert self.circuit_breaker.state == CircuitState.CLOSED
        
        # Second failure
        with pytest.raises(RuntimeError):
            failing_function()
        assert self.circuit_breaker.failure_count == 2
        assert self.circuit_breaker.state == CircuitState.CLOSED
        
    def test_circuit_opens_after_threshold_failures(self):
        """Test that circuit opens after reaching failure threshold."""
        
        @self.circuit_breaker
        def failing_function():
            raise RuntimeError("Test failure")
            
        # Reach failure threshold
        for _ in range(3):
            with pytest.raises(RuntimeError):
                failing_function()
                
        assert self.circuit_breaker.state == CircuitState.OPEN
        
        # Next call should be blocked
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            failing_function()
            
    def test_circuit_recovery_after_timeout(self):
        """Test circuit recovery after timeout period."""
        
        @self.circuit_breaker
        def function_that_recovers():
            if self.circuit_breaker.state == CircuitState.HALF_OPEN:
                return "recovered"
            raise RuntimeError("Still failing")
            
        # Trip the circuit
        for _ in range(3):
            with pytest.raises(RuntimeError):
                function_that_recovers()
                
        assert self.circuit_breaker.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        time.sleep(1.1)
        
        # Next call should attempt recovery
        result = function_that_recovers()
        assert result == "recovered"
        assert self.circuit_breaker.state == CircuitState.CLOSED


class TestAdaptiveRetryStrategy:
    """Test adaptive retry strategy."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.retry_strategy = AdaptiveRetryStrategy(
            max_attempts=3,
            base_delay=0.1,
            max_delay=1.0,
            exponential_base=2.0,
            jitter=False  # Disable jitter for predictable testing
        )
        
    def test_retry_strategy_initialization(self):
        """Test retry strategy initialization."""
        assert self.retry_strategy.max_attempts == 3
        assert self.retry_strategy.base_delay == 0.1
        assert self.retry_strategy.max_delay == 1.0
        assert self.retry_strategy.exponential_base == 2.0
        assert self.retry_strategy.jitter == False
        
    def test_successful_execution_no_retry(self):
        """Test that successful execution doesn't trigger retries."""
        
        def successful_function():
            return "success"
            
        start_time = time.time()
        result = self.retry_strategy.execute(successful_function)
        end_time = time.time()
        
        assert result == "success"
        assert end_time - start_time < 0.05  # Should be immediate
        
    def test_retry_with_eventual_success(self):
        """Test retries with eventual success."""
        
        attempt_count = 0
        def function_succeeds_on_third_attempt():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise RuntimeError(f"Failure {attempt_count}")
            return "success"
            
        result = self.retry_strategy.execute(function_succeeds_on_third_attempt)
        assert result == "success"
        assert attempt_count == 3
        
    def test_retry_exhaustion_raises_last_exception(self):
        """Test that retry exhaustion raises the last exception."""
        
        def always_failing_function():
            raise RuntimeError("Always fails")
            
        with pytest.raises(RuntimeError, match="Always fails"):
            self.retry_strategy.execute(always_failing_function)
            
    def test_exponential_backoff_delays(self):
        """Test exponential backoff delay calculation."""
        
        # Test delay calculation
        delay_0 = self.retry_strategy._calculate_delay(0)
        delay_1 = self.retry_strategy._calculate_delay(1)
        delay_2 = self.retry_strategy._calculate_delay(2)
        
        assert delay_0 == 0.1  # base_delay * 2^0
        assert delay_1 == 0.2  # base_delay * 2^1
        assert delay_2 == 0.4  # base_delay * 2^2
        
    def test_max_delay_limit(self):
        """Test that delay doesn't exceed max_delay."""
        
        # Test with large attempt number
        delay = self.retry_strategy._calculate_delay(10)
        assert delay <= self.retry_strategy.max_delay


class TestHealthMetrics:
    """Test health metrics functionality."""
    
    def test_health_metrics_initialization(self):
        """Test health metrics initialization."""
        metrics = HealthMetrics(
            cpu_usage=0.5,
            memory_usage=0.6,
            gpu_usage=0.3,
            inference_latency=1.5,
            error_rate=0.01,
            throughput=50.0,
            model_accuracy=0.95,
            data_quality_score=0.9
        )
        
        assert metrics.cpu_usage == 0.5
        assert metrics.memory_usage == 0.6
        assert metrics.gpu_usage == 0.3
        assert metrics.inference_latency == 1.5
        assert metrics.error_rate == 0.01
        assert metrics.throughput == 50.0
        assert metrics.model_accuracy == 0.95
        assert metrics.data_quality_score == 0.9
        
    def test_timestamp_default(self):
        """Test that timestamp defaults to current time."""
        before = time.time()
        metrics = HealthMetrics()
        after = time.time()
        
        assert before <= metrics.timestamp <= after


class TestSelfHealingManager:
    """Test self-healing manager functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.healing_manager = SelfHealingManager()
        
    def test_healing_manager_initialization(self):
        """Test healing manager initialization."""
        assert self.healing_manager.system_state == SystemState.HEALTHY
        assert len(self.healing_manager.healing_strategies) == 0
        assert len(self.healing_manager.metrics_history) == 0
        
    def test_register_healing_strategy(self):
        """Test registering healing strategies."""
        
        def test_strategy():
            return "healed"
            
        self.healing_manager.register_healing_strategy("test", test_strategy)
        assert "test" in self.healing_manager.healing_strategies
        assert self.healing_manager.healing_strategies["test"]() == "healed"
        
    def test_assess_healthy_system(self):
        """Test assessment of healthy system."""
        
        healthy_metrics = HealthMetrics(
            cpu_usage=0.3,
            memory_usage=0.4,
            error_rate=0.01,
            inference_latency=0.5,
            model_accuracy=0.95
        )
        
        state = self.healing_manager.assess_system_health(healthy_metrics)
        assert state == SystemState.HEALTHY
        
    def test_assess_degraded_system(self):
        """Test assessment of degraded system."""
        
        degraded_metrics = HealthMetrics(
            cpu_usage=0.8,
            memory_usage=0.7,
            error_rate=0.08,
            inference_latency=2.5,
            model_accuracy=0.88
        )
        
        state = self.healing_manager.assess_system_health(degraded_metrics)
        assert state == SystemState.DEGRADED
        
    def test_assess_critical_system(self):
        """Test assessment of critical system."""
        
        critical_metrics = HealthMetrics(
            cpu_usage=0.95,
            memory_usage=0.9,
            error_rate=0.15,
            inference_latency=6.0,
            model_accuracy=0.80
        )
        
        state = self.healing_manager.assess_system_health(critical_metrics)
        assert state == SystemState.CRITICAL
        
    def test_auto_heal_healthy_system(self):
        """Test auto-healing for healthy system."""
        
        success = self.healing_manager.auto_heal(SystemState.HEALTHY)
        assert success == True
        
    def test_auto_heal_with_strategies(self):
        """Test auto-healing with registered strategies."""
        
        strategy_called = False
        def test_strategy():
            nonlocal strategy_called
            strategy_called = True
            
        self.healing_manager.register_healing_strategy("test", test_strategy)
        
        # Auto-heal shouldn't call strategies for healthy system
        self.healing_manager.auto_heal(SystemState.HEALTHY)
        assert strategy_called == False
        
        # Auto-heal should call strategies for degraded system
        self.healing_manager.auto_heal(SystemState.DEGRADED)
        # Note: "test" strategy won't be called as it's not in the predefined strategy list


class TestPerformanceMonitor:
    """Test performance monitoring functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.monitor = PerformanceMonitor()
        
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        assert len(self.monitor.metrics_buffer) == 0
        assert self.monitor.anomaly_detector is not None
        
    def test_collect_metrics(self):
        """Test metrics collection."""
        
        with patch('psutil.cpu_percent', return_value=50.0), \\
             patch('psutil.virtual_memory') as mock_memory:
            
            mock_memory.return_value.percent = 60.0
            
            metrics = self.monitor.collect_metrics()
            
            assert isinstance(metrics, HealthMetrics)
            assert metrics.cpu_usage == 0.5  # 50% converted to 0.5
            assert metrics.memory_usage == 0.6  # 60% converted to 0.6
            
    def test_gpu_usage_fallback(self):
        """Test GPU usage fallback when not available."""
        
        gpu_usage = self.monitor._get_gpu_usage()
        assert gpu_usage == 0.0  # Should fallback to 0.0


class TestAnomalyDetector:
    """Test anomaly detection functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.detector = AnomalyDetector(window_size=10)
        
    def test_detector_initialization(self):
        """Test detector initialization."""
        assert self.detector.window_size == 10
        assert len(self.detector.metrics_window) == 0
        
    def test_insufficient_data_no_anomaly(self):
        """Test that insufficient data doesn't trigger anomalies."""
        
        for i in range(5):  # Less than minimum 10 required
            is_anomaly = self.detector.is_anomaly(1.0)
            assert is_anomaly == False
            
    def test_normal_values_no_anomaly(self):
        """Test that normal values don't trigger anomalies."""
        
        # Fill with consistent values
        for i in range(15):
            is_anomaly = self.detector.is_anomaly(1.0)
            
        # All values should be normal (no anomaly)
        assert is_anomaly == False
        
    def test_anomaly_detection(self):
        """Test anomaly detection for outlier values."""
        
        # Fill with normal values
        for i in range(15):
            self.detector.is_anomaly(1.0)
            
        # Add significant outlier
        is_anomaly = self.detector.is_anomaly(10.0)  # Significant outlier
        assert is_anomaly == True
        
    def test_window_size_limit(self):
        """Test that window size is properly limited."""
        
        # Add more values than window size
        for i in range(20):
            self.detector.is_anomaly(float(i))
            
        assert len(self.detector.metrics_window) == self.detector.window_size


class TestMedicalAIResilienceFramework:
    """Test the main resilience framework."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.framework = MedicalAIResilienceFramework()
        
    def test_framework_initialization(self):
        """Test framework initialization."""
        assert len(self.framework.circuit_breakers) == 0
        assert len(self.framework.retry_strategies) == 0
        assert self.framework.self_healing is not None
        assert self.framework.performance_monitor is not None
        
    def test_create_circuit_breaker(self):
        """Test circuit breaker creation."""
        
        cb = self.framework.create_circuit_breaker("test_cb", failure_threshold=5)
        
        assert "test_cb" in self.framework.circuit_breakers
        assert cb.failure_threshold == 5
        assert isinstance(cb, CircuitBreaker)
        
    def test_create_retry_strategy(self):
        """Test retry strategy creation."""
        
        strategy = self.framework.create_retry_strategy("test_retry", max_attempts=5)
        
        assert "test_retry" in self.framework.retry_strategies
        assert strategy.max_attempts == 5
        assert isinstance(strategy, AdaptiveRetryStrategy)
        
    def test_monitor_and_heal_healthy_system(self):
        """Test monitoring and healing for healthy system."""
        
        healthy_metrics = HealthMetrics(
            cpu_usage=0.3,
            memory_usage=0.4,
            error_rate=0.01,
            inference_latency=0.5,
            model_accuracy=0.95
        )
        
        # Should not raise any exceptions
        self.framework.monitor_and_heal(healthy_metrics)
        
    def test_monitor_and_heal_critical_system(self):
        """Test monitoring and healing for critical system."""
        
        critical_metrics = HealthMetrics(
            cpu_usage=0.95,
            memory_usage=0.9,
            error_rate=0.2,
            inference_latency=10.0,
            model_accuracy=0.75
        )
        
        # Should trigger healing strategies
        with patch.object(self.framework.self_healing, 'auto_heal', return_value=True) as mock_heal:
            self.framework.monitor_and_heal(critical_metrics)
            mock_heal.assert_called_once()


class TestResilientInferenceDecorator:
    """Test the resilient inference decorator."""
    
    def test_resilient_inference_decorator(self):
        """Test that decorator applies resilience measures."""
        
        @resilient_inference
        def mock_inference_function(data):
            return {"prediction": "normal", "confidence": 0.95}
            
        result = mock_inference_function("test_data")
        
        # Should return the expected result
        assert result["prediction"] == "normal"
        assert result["confidence"] == 0.95
        
    def test_resilient_inference_with_failures(self):
        """Test resilient inference with simulated failures."""
        
        call_count = 0
        
        @resilient_inference
        def mock_failing_inference(data):
            nonlocal call_count
            call_count += 1
            if call_count < 2:  # Fail first attempt
                raise RuntimeError("Inference failed")
            return {"prediction": "pneumonia", "confidence": 0.88}
            
        result = mock_failing_inference("test_data")
        
        # Should succeed after retry
        assert result["prediction"] == "pneumonia"
        assert result["confidence"] == 0.88
        assert call_count == 2  # Should have retried once


class TestIntegrationScenarios:
    """Integration tests for complete resilience scenarios."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.framework = MedicalAIResilienceFramework()
        
    def test_end_to_end_resilience_scenario(self):
        """Test complete end-to-end resilience scenario."""
        
        # Create circuit breaker and retry strategy
        cb = self.framework.create_circuit_breaker("inference", failure_threshold=2)
        retry = self.framework.create_retry_strategy("inference_retry", max_attempts=3)
        
        # Mock function that fails initially then succeeds
        attempt_count = 0
        def mock_inference():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count <= 1:
                raise RuntimeError("Temporary failure")
            return "success"
            
        # Apply circuit breaker
        protected_inference = cb(mock_inference)
        
        # Execute with retry
        result = retry.execute(protected_inference)
        
        assert result == "success"
        assert attempt_count == 2
        assert cb.state == CircuitState.CLOSED  # Should remain closed due to eventual success
        
    def test_system_degradation_and_recovery(self):
        """Test system degradation detection and recovery."""
        
        # Simulate system degradation over time
        metrics_sequence = [
            HealthMetrics(cpu_usage=0.3, memory_usage=0.3, error_rate=0.01, model_accuracy=0.95),  # Healthy
            HealthMetrics(cpu_usage=0.6, memory_usage=0.5, error_rate=0.03, model_accuracy=0.93),  # Starting to degrade
            HealthMetrics(cpu_usage=0.8, memory_usage=0.7, error_rate=0.08, model_accuracy=0.88),  # Degraded
            HealthMetrics(cpu_usage=0.95, memory_usage=0.9, error_rate=0.15, model_accuracy=0.82), # Critical
            HealthMetrics(cpu_usage=0.4, memory_usage=0.4, error_rate=0.02, model_accuracy=0.94),  # Recovered
        ]
        
        states = []
        for metrics in metrics_sequence:
            state = self.framework.self_healing.assess_system_health(metrics)
            states.append(state)
            self.framework.monitor_and_heal(metrics)
            
        # Verify state progression
        assert states[0] == SystemState.HEALTHY
        assert states[1] == SystemState.HEALTHY  # Still healthy
        assert states[2] == SystemState.DEGRADED
        assert states[3] == SystemState.CRITICAL
        assert states[4] == SystemState.HEALTHY  # Recovered
        
    def test_concurrent_resilience_operations(self):
        """Test resilience framework under concurrent load."""
        
        import threading
        import time
        
        results = []
        errors = []
        
        def worker_function(worker_id):
            try:
                # Create metrics for this worker
                metrics = HealthMetrics(
                    cpu_usage=0.3 + (worker_id % 3) * 0.2,
                    memory_usage=0.4 + (worker_id % 3) * 0.1,
                    error_rate=0.01 * (worker_id % 3),
                    model_accuracy=0.95 - (worker_id % 3) * 0.02
                )
                
                # Process with framework
                self.framework.monitor_and_heal(metrics)
                results.append(f"worker_{worker_id}_success")
                
            except Exception as e:
                errors.append(f"worker_{worker_id}_error: {e}")
                
        # Start multiple worker threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker_function, args=(i,))
            threads.append(thread)
            thread.start()
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5.0)
            
        # Verify results
        assert len(results) == 10
        assert len(errors) == 0


@pytest.mark.asyncio
async def test_async_resilience_operations():
    """Test resilience framework with async operations."""
    
    framework = MedicalAIResilienceFramework()
    
    async def async_inference():
        await asyncio.sleep(0.01)  # Simulate async work
        return {"prediction": "normal", "confidence": 0.92}
        
    # Test that framework works with async operations
    metrics = HealthMetrics(
        cpu_usage=0.5,
        memory_usage=0.4,
        error_rate=0.02,
        model_accuracy=0.94
    )
    
    # Monitor in parallel with async operation
    result = await async_inference()
    framework.monitor_and_heal(metrics)
    
    assert result["prediction"] == "normal"
    assert result["confidence"] == 0.92


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v", "--tb=short"])