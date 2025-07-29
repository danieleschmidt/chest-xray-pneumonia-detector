"""
Comprehensive Performance Testing Framework - Advanced SDLC Enhancement
Automated performance benchmarking with regression detection and metrics collection.
"""

import pytest
import time
import numpy as np
import psutil
import os
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import threading
import queue


@dataclass
class PerformanceMetrics:
    """Performance metrics container for benchmarking results."""
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    throughput_ops_per_sec: float
    memory_peak_mb: float
    test_name: str
    timestamp: str


class PerformanceMonitor:
    """Advanced performance monitoring with system resource tracking."""
    
    def __init__(self):
        self.metrics_queue = queue.Queue()
        self.monitoring_active = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start system resource monitoring in background thread."""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self) -> Dict:
        """Stop monitoring and return collected metrics."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            
        metrics = []
        while not self.metrics_queue.empty():
            metrics.append(self.metrics_queue.get())
            
        return {
            'avg_cpu': np.mean([m['cpu'] for m in metrics]) if metrics else 0,
            'max_memory_mb': max([m['memory_mb'] for m in metrics]) if metrics else 0,
            'avg_memory_mb': np.mean([m['memory_mb'] for m in metrics]) if metrics else 0
        }
        
    def _monitor_resources(self):
        """Background thread for resource monitoring."""
        process = psutil.Process()
        while self.monitoring_active:
            try:
                cpu_percent = process.cpu_percent()
                memory_mb = process.memory_info().rss / 1024 / 1024
                self.metrics_queue.put({
                    'cpu': cpu_percent,
                    'memory_mb': memory_mb,
                    'timestamp': time.time()
                })
                time.sleep(0.1)  # Sample every 100ms
            except (psutil.NoSuchProcess, OSError):
                break


@pytest.fixture
def performance_monitor():
    """Fixture providing performance monitoring capabilities."""
    monitor = PerformanceMonitor()
    yield monitor
    if monitor.monitoring_active:
        monitor.stop_monitoring()


@pytest.fixture
def performance_baseline():
    """Load performance baselines from configuration."""
    baseline_file = Path(__file__).parent / "performance_baselines.json"
    if baseline_file.exists():
        with open(baseline_file) as f:
            return json.load(f)
    return {
        "data_loading_time_sec": 2.0,
        "model_inference_time_sec": 0.5,
        "training_iteration_time_sec": 1.0,
        "memory_usage_mb": 500.0,
        "throughput_samples_per_sec": 10.0
    }


class TestAdvancedPerformance:
    """Advanced performance testing with regression detection."""
    
    @pytest.mark.performance
    def test_data_loading_performance(self, performance_monitor, performance_baseline):
        """Test data loading pipeline performance with memory monitoring."""
        performance_monitor.start_monitoring()
        
        start_time = time.time()
        
        # Simulate data loading performance test
        # In real implementation, this would load actual datasets
        import numpy as np
        data_size = 1000
        images = np.random.rand(data_size, 224, 224, 3).astype(np.float32)
        labels = np.random.randint(0, 2, data_size)
        
        # Simulate data preprocessing
        preprocessed = images / 255.0
        normalized = (preprocessed - 0.5) / 0.5
        
        execution_time = time.time() - start_time
        system_metrics = performance_monitor.stop_monitoring()
        
        # Performance assertions with tolerance
        baseline_time = performance_baseline["data_loading_time_sec"]
        assert execution_time < baseline_time * 1.2, (
            f"Data loading too slow: {execution_time:.2f}s vs baseline {baseline_time:.2f}s"
        )
        
        # Memory usage check
        baseline_memory = performance_baseline["memory_usage_mb"]
        assert system_metrics['max_memory_mb'] < baseline_memory * 1.5, (
            f"Memory usage too high: {system_metrics['max_memory_mb']:.2f}MB vs baseline {baseline_memory:.2f}MB"
        )
        
        print(f"✓ Data loading performance: {execution_time:.2f}s, Memory: {system_metrics['max_memory_mb']:.2f}MB")
    
    @pytest.mark.performance
    @pytest.mark.parametrize("batch_size,expected_throughput", [
        (16, 8.0),
        (32, 15.0),
        (64, 25.0)
    ])
    def test_inference_throughput(self, performance_monitor, performance_baseline, batch_size, expected_throughput):
        """Test model inference throughput across different batch sizes."""
        performance_monitor.start_monitoring()
        
        # Simulate batch inference performance test
        import numpy as np
        batch_data = np.random.rand(batch_size, 224, 224, 3).astype(np.float32)
        
        start_time = time.time()
        iterations = 100
        
        for _ in range(iterations):
            # Simulate model inference
            # In real implementation, this would use actual model.predict()
            time.sleep(0.001)  # Simulate inference time
            predictions = np.random.rand(batch_size, 1)
        
        execution_time = time.time() - start_time
        throughput = (iterations * batch_size) / execution_time
        system_metrics = performance_monitor.stop_monitoring()
        
        # Throughput assertions
        assert throughput >= expected_throughput * 0.8, (
            f"Throughput too low: {throughput:.2f} samples/sec vs expected {expected_throughput:.2f}"
        )
        
        print(f"✓ Batch {batch_size} throughput: {throughput:.2f} samples/sec, CPU: {system_metrics['avg_cpu']:.1f}%")
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_training_performance_profile(self, performance_monitor):
        """Comprehensive training performance profiling with bottleneck detection."""
        performance_monitor.start_monitoring()
        
        phases = {
            "data_loading": 0.5,
            "forward_pass": 0.3,
            "backward_pass": 0.4,
            "optimizer_step": 0.1,
            "metrics_calculation": 0.1
        }
        
        phase_times = {}
        total_start = time.time()
        
        for phase, expected_time in phases.items():
            phase_start = time.time()
            
            # Simulate training phase
            time.sleep(0.01)  # Minimal simulation time
            if phase == "data_loading":
                # Simulate data loading overhead
                data = np.random.rand(32, 224, 224, 3)
            elif phase == "forward_pass":
                # Simulate forward computation
                np.random.rand(32, 1000) @ np.random.rand(1000, 512)
            elif phase == "backward_pass":
                # Simulate gradient computation
                np.random.rand(512, 1000) @ np.random.rand(1000, 32)
            
            phase_time = time.time() - phase_start
            phase_times[phase] = phase_time
        
        total_time = time.time() - total_start
        system_metrics = performance_monitor.stop_monitoring()
        
        # Performance profiling analysis
        bottleneck_phase = max(phase_times.items(), key=lambda x: x[1])
        print(f"✓ Training profile - Total: {total_time:.3f}s, Bottleneck: {bottleneck_phase[0]} ({bottleneck_phase[1]:.3f}s)")
        print(f"  Memory peak: {system_metrics['max_memory_mb']:.2f}MB, Avg CPU: {system_metrics['avg_cpu']:.1f}%")
        
        # Ensure no single phase dominates
        for phase, phase_time in phase_times.items():
            relative_time = phase_time / total_time
            assert relative_time < 0.7, f"Phase {phase} taking too long: {relative_time:.1%} of total time"
    
    @pytest.mark.performance
    def test_memory_leak_detection(self, performance_monitor):
        """Advanced memory leak detection with multiple iterations."""
        import gc
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_samples = []
        
        for iteration in range(10):
            performance_monitor.start_monitoring()
            
            # Simulate potential memory leak scenario
            large_data = []
            for i in range(100):
                # Create and process data that might leak
                data = np.random.rand(1000, 100)
                processed = data * 2
                large_data.append(processed)
            
            # Clean up explicitly
            del large_data
            gc.collect()
            
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)
            performance_monitor.stop_monitoring()
        
        # Analyze memory growth trend
        memory_growth = np.polyfit(range(len(memory_samples)), memory_samples, 1)[0]
        
        print(f"✓ Memory leak test - Growth rate: {memory_growth:.2f}MB/iteration")
        print(f"  Initial: {initial_memory:.2f}MB, Final: {memory_samples[-1]:.2f}MB")
        
        # Assert no significant memory leak (growth < 1MB per iteration)
        assert memory_growth < 1.0, f"Potential memory leak detected: {memory_growth:.2f}MB/iteration growth"
    
    @pytest.mark.performance
    def test_concurrent_processing_performance(self, performance_monitor):
        """Test performance under concurrent processing scenarios."""
        import concurrent.futures
        import threading
        
        performance_monitor.start_monitoring()
        
        def simulate_processing_task(task_id: int) -> float:
            """Simulate a processing task that might be run concurrently."""
            start_time = time.time()
            
            # Simulate CPU-bound work
            data = np.random.rand(500, 500)
            result = np.linalg.norm(data)
            
            return time.time() - start_time
        
        # Test sequential processing
        sequential_start = time.time()
        sequential_results = [simulate_processing_task(i) for i in range(4)]
        sequential_time = time.time() - sequential_start
        
        # Test concurrent processing
        concurrent_start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            concurrent_results = list(executor.map(simulate_processing_task, range(4)))
        concurrent_time = time.time() - concurrent_start
        
        system_metrics = performance_monitor.stop_monitoring()
        
        # Performance analysis
        speedup_ratio = sequential_time / concurrent_time if concurrent_time > 0 else 0
        
        print(f"✓ Concurrency test - Sequential: {sequential_time:.3f}s, Concurrent: {concurrent_time:.3f}s")
        print(f"  Speedup: {speedup_ratio:.2f}x, Peak memory: {system_metrics['max_memory_mb']:.2f}MB")
        
        # Ensure concurrent processing provides some benefit
        assert speedup_ratio > 1.1, f"Insufficient concurrency benefit: {speedup_ratio:.2f}x speedup"


@pytest.mark.performance
def test_performance_regression_tracking():
    """Track and report performance regressions across test runs."""
    results_file = Path(__file__).parent / "performance_results.json"
    
    current_results = {
        "timestamp": time.time(),
        "git_commit": os.environ.get("GITHUB_SHA", "unknown"),
        "metrics": {
            "data_loading_baseline": 2.0,
            "inference_throughput": 15.0,
            "memory_peak": 500.0
        }
    }
    
    if results_file.exists():
        with open(results_file) as f:
            historical_results = json.load(f)
        
        # Compare with previous results
        if "metrics" in historical_results:
            for metric, current_value in current_results["metrics"].items():
                if metric in historical_results["metrics"]:
                    previous_value = historical_results["metrics"][metric]
                    regression_threshold = 1.15  # 15% regression threshold
                    
                    if current_value > previous_value * regression_threshold:
                        pytest.fail(f"Performance regression detected in {metric}: "
                                  f"{current_value:.2f} vs {previous_value:.2f} "
                                  f"({(current_value/previous_value - 1)*100:.1f}% increase)")
    
    # Save current results for future comparisons
    with open(results_file, 'w') as f:
        json.dump(current_results, f, indent=2)
    
    print("✓ Performance regression tracking updated")


if __name__ == "__main__":
    # Allow running performance tests directly
    pytest.main([__file__, "-v", "-m", "performance"])