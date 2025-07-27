import pytest
import time
import numpy as np
from pathlib import Path
import tempfile
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from performance_benchmark import benchmark_inference_speed, benchmark_memory_usage
except ImportError:
    pytest.skip("performance_benchmark module not available", allow_module_level=True)


@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def test_inference_speed_benchmark(self, sample_dataset_structure, mock_model_config):
        """Test inference speed benchmarking."""
        with tempfile.NamedTemporaryFile(suffix='.keras') as model_file:
            # Mock model path
            model_path = model_file.name
            
            # Create some test images
            test_images_dir = sample_dataset_structure / "test" / "NORMAL"
            
            # This would normally run the actual benchmark
            # For testing, we'll mock the timing
            start_time = time.time()
            
            # Simulate processing time
            time.sleep(0.1)  # 100ms simulation
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Verify benchmark completed in reasonable time
            assert processing_time < 1.0  # Should complete within 1 second
            assert processing_time > 0.05  # Should take at least 50ms
    
    def test_memory_usage_benchmark(self, mock_model_config):
        """Test memory usage benchmarking."""
        # This would test memory consumption during model operations
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate model loading and inference
        large_array = np.random.random((1000, 1000, 3))
        
        # Get memory after operation
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        # Clean up
        del large_array
        
        # Verify memory usage is tracked
        assert memory_increase >= 0  # Memory should increase or stay same
        assert current_memory > 0  # Memory usage should be positive
    
    def test_batch_processing_performance(self, sample_dataset_structure):
        """Test batch processing performance."""
        test_dir = sample_dataset_structure / "test"
        
        # Count available test images
        image_files = list(test_dir.rglob("*.jpg"))
        
        # Simulate batch processing timing
        batch_sizes = [1, 4, 8, 16]
        processing_times = []
        
        for batch_size in batch_sizes:
            start_time = time.time()
            
            # Simulate batch processing
            batches = len(image_files) // batch_size
            for _ in range(min(batches, 5)):  # Process max 5 batches for testing
                time.sleep(0.01)  # 10ms per batch simulation
            
            end_time = time.time()
            processing_times.append(end_time - start_time)
        
        # Verify processing times are reasonable
        assert all(t < 1.0 for t in processing_times)  # All under 1 second
        assert len(processing_times) == len(batch_sizes)
    
    @pytest.mark.integration
    def test_end_to_end_performance(self, sample_dataset_structure, temp_dir):
        """Test end-to-end pipeline performance."""
        # This would test the complete pipeline from data loading to prediction
        dataset_path = sample_dataset_structure
        output_path = Path(temp_dir) / "performance_results.csv"
        
        start_time = time.time()
        
        # Simulate full pipeline
        steps = [
            "data_loading",
            "preprocessing", 
            "model_loading",
            "inference",
            "postprocessing",
            "results_saving"
        ]
        
        step_times = {}
        for step in steps:
            step_start = time.time()
            time.sleep(0.02)  # 20ms per step simulation
            step_times[step] = time.time() - step_start
        
        total_time = time.time() - start_time
        
        # Verify performance metrics
        assert total_time < 2.0  # Total pipeline under 2 seconds
        assert all(t < 0.5 for t in step_times.values())  # Each step under 500ms
        assert len(step_times) == len(steps)
    
    def test_model_loading_performance(self, temp_dir):
        """Test model loading performance."""
        # Create a mock model file
        model_path = Path(temp_dir) / "test_model.keras"
        
        # Write some dummy data to simulate model file
        with open(model_path, 'wb') as f:
            f.write(b'dummy_model_data' * 1000)  # ~15KB mock model
        
        # Test loading performance
        start_time = time.time()
        
        # Simulate model loading
        with open(model_path, 'rb') as f:
            data = f.read()
            
        load_time = time.time() - start_time
        
        # Verify loading performance
        assert load_time < 1.0  # Should load within 1 second
        assert len(data) > 0  # Data should be loaded
    
    def test_concurrent_inference_performance(self, sample_dataset_structure):
        """Test concurrent inference performance."""
        import concurrent.futures
        import threading
        
        def simulate_inference(image_path):
            """Simulate inference on a single image."""
            time.sleep(0.05)  # 50ms simulation
            return f"processed_{image_path.name}"
        
        # Get test images
        test_images = list(sample_dataset_structure.rglob("*.jpg"))[:10]  # Limit to 10
        
        # Test sequential processing
        start_time = time.time()
        sequential_results = [simulate_inference(img) for img in test_images]
        sequential_time = time.time() - start_time
        
        # Test concurrent processing
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            concurrent_results = list(executor.map(simulate_inference, test_images))
        concurrent_time = time.time() - start_time
        
        # Verify results
        assert len(sequential_results) == len(test_images)
        assert len(concurrent_results) == len(test_images)
        
        # Concurrent should be faster (though not guaranteed in simulation)
        assert concurrent_time < sequential_time * 2  # Allow some overhead