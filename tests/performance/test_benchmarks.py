"""
Performance benchmarks and tests for the Chest X-Ray Pneumonia Detector.
"""

import time
import psutil
import pytest
import numpy as np
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock, patch


@pytest.mark.performance
@pytest.mark.slow
class TestInferencePerformance:
    """Test inference performance characteristics."""

    def test_single_image_inference_speed(self, sample_image: np.ndarray):
        """Test single image inference speed."""
        from src.performance_benchmark import benchmark_single_inference
        
        # Mock model for consistent testing
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.8]])
        
        with patch('tensorflow.keras.models.load_model', return_value=mock_model):
            start_time = time.time()
            
            # Run multiple inferences to get average
            num_iterations = 10
            for _ in range(num_iterations):
                result = benchmark_single_inference(
                    model=mock_model,
                    image=sample_image
                )
            
            end_time = time.time()
            avg_time = (end_time - start_time) / num_iterations
            
            assert avg_time < 0.1, f"Average inference time {avg_time:.3f}s too slow"
            assert result is not None

    def test_batch_inference_throughput(self, sample_dataset_structure):
        """Test batch inference throughput."""
        from src.performance_benchmark import benchmark_batch_inference
        
        _, _, test_dir = sample_dataset_structure
        
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.8], [0.2], [0.9], [0.1]])
        
        batch_sizes = [1, 4, 8, 16]
        results = {}
        
        for batch_size in batch_sizes:
            with patch('tensorflow.keras.models.load_model', return_value=mock_model):
                result = benchmark_batch_inference(
                    model_path="mock_model.keras",
                    data_dir=str(test_dir),
                    batch_size=batch_size,
                    num_samples=20
                )
                
                results[batch_size] = result
        
        # Verify throughput increases with batch size (up to a point)
        assert results[4]["throughput"] >= results[1]["throughput"]
        assert all(r["avg_latency"] > 0 for r in results.values())

    def test_memory_usage_during_inference(self, sample_image: np.ndarray):
        """Test memory usage during inference."""
        process = psutil.Process()
        
        # Get baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.8]])
        
        # Simulate multiple inferences
        batch_images = np.stack([sample_image] * 32)  # 32 image batch
        
        with patch('tensorflow.keras.models.load_model', return_value=mock_model):
            mock_model.predict(batch_images)
            
            # Check memory usage
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - baseline_memory
            
            # Memory increase should be reasonable for batch processing
            assert memory_increase < 500, f"Memory increased by {memory_increase:.2f}MB"

    def test_gpu_vs_cpu_performance(self, sample_image: np.ndarray):
        """Test GPU vs CPU performance if GPU is available."""
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            
            if not gpus:
                pytest.skip("No GPU available for performance comparison")
            
            # Create simple model for testing
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            # Prepare batch of images
            batch_images = np.stack([sample_image] * 16)
            batch_images = batch_images.astype(np.float32) / 255.0
            
            # Test CPU performance
            with tf.device('/CPU:0'):
                start_time = time.time()
                cpu_result = model.predict(batch_images, verbose=0)
                cpu_time = time.time() - start_time
            
            # Test GPU performance
            with tf.device('/GPU:0'):
                start_time = time.time()
                gpu_result = model.predict(batch_images, verbose=0)
                gpu_time = time.time() - start_time
            
            # GPU should be faster or at least comparable
            speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0
            
            assert speedup >= 0.8, f"GPU performance not better than CPU (speedup: {speedup:.2f}x)"
            
        except ImportError:
            pytest.skip("TensorFlow not available for GPU testing")


@pytest.mark.performance
class TestDataLoadingPerformance:
    """Test data loading and preprocessing performance."""

    def test_image_loading_speed(self, sample_dataset_structure):
        """Test image loading speed."""
        _, _, test_dir = sample_dataset_structure
        
        from src.data_loader import load_and_preprocess_image
        
        # Get all image files
        image_files = list(test_dir.rglob("*.jpg"))[:10]  # Test with first 10 images
        
        start_time = time.time()
        
        for img_path in image_files:
            processed_image = load_and_preprocess_image(
                str(img_path),
                target_size=(224, 224)
            )
            assert processed_image is not None
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_image = total_time / len(image_files)
        
        assert avg_time_per_image < 0.5, f"Image loading too slow: {avg_time_per_image:.3f}s per image"

    def test_data_generator_performance(self, sample_dataset_structure):
        """Test data generator performance."""
        train_dir, val_dir, _ = sample_dataset_structure
        
        from src.data_loader import create_data_generators
        
        # Create generators
        train_gen, val_gen = create_data_generators(
            train_dir=str(train_dir),
            val_dir=str(val_dir),
            batch_size=8,
            img_size=(224, 224),
            augment_training=True
        )
        
        # Test batch generation speed
        start_time = time.time()
        
        # Generate multiple batches
        num_batches = 5
        for i, batch in enumerate(train_gen):
            if i >= num_batches - 1:
                break
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_batch = total_time / num_batches
        
        assert avg_time_per_batch < 2.0, f"Batch generation too slow: {avg_time_per_batch:.3f}s per batch"

    def test_preprocessing_efficiency(self, sample_image: np.ndarray):
        """Test image preprocessing efficiency."""
        from src.image_utils import preprocess_medical_image
        
        # Test preprocessing multiple images
        num_images = 50
        images = [sample_image] * num_images
        
        start_time = time.time()
        
        for img in images:
            processed = preprocess_medical_image(
                img,
                target_size=(224, 224),
                normalize=True
            )
            assert processed is not None
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_image = total_time / num_images
        
        assert avg_time_per_image < 0.1, f"Preprocessing too slow: {avg_time_per_image:.3f}s per image"


@pytest.mark.performance
class TestModelPerformance:
    """Test model-related performance characteristics."""

    def test_model_loading_speed(self, mock_model_file: Path):
        """Test model loading speed."""
        mock_model = Mock()
        
        with patch('tensorflow.keras.models.load_model', return_value=mock_model) as mock_load:
            from src.model_registry import load_model
            
            start_time = time.time()
            
            # Load model multiple times
            for _ in range(5):
                model = load_model(str(mock_model_file))
                assert model is not None
            
            end_time = time.time()
            avg_loading_time = (end_time - start_time) / 5
            
            assert avg_loading_time < 1.0, f"Model loading too slow: {avg_loading_time:.3f}s"

    def test_model_compilation_speed(self):
        """Test model compilation speed."""
        try:
            import tensorflow as tf
            
            start_time = time.time()
            
            # Create and compile a simple model
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64, 3, activation='relu'),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            end_time = time.time()
            compilation_time = end_time - start_time
            
            assert compilation_time < 5.0, f"Model compilation too slow: {compilation_time:.3f}s"
            
        except ImportError:
            pytest.skip("TensorFlow not available for model compilation testing")

    def test_training_step_performance(self, sample_training_config: dict):
        """Test single training step performance."""
        try:
            import tensorflow as tf
            
            # Create simple model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer='adam', loss='binary_crossentropy')
            
            # Create dummy training data
            x_train = np.random.random((32, 784))
            y_train = np.random.randint(0, 2, (32, 1))
            
            start_time = time.time()
            
            # Single training step
            model.train_on_batch(x_train, y_train)
            
            end_time = time.time()
            step_time = end_time - start_time
            
            assert step_time < 1.0, f"Training step too slow: {step_time:.3f}s"
            
        except ImportError:
            pytest.skip("TensorFlow not available for training performance testing")


@pytest.mark.performance
class TestSystemResourceUsage:
    """Test system resource usage patterns."""

    def test_cpu_usage_monitoring(self):
        """Test CPU usage during operations."""
        import psutil
        import threading
        import time
        
        cpu_usage_samples = []
        
        def monitor_cpu():
            for _ in range(10):
                cpu_usage_samples.append(psutil.cpu_percent(interval=0.1))
        
        # Start monitoring
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # Simulate CPU-intensive operation
        for _ in range(1000000):
            _ = sum(range(100))
        
        monitor_thread.join()
        
        avg_cpu_usage = sum(cpu_usage_samples) / len(cpu_usage_samples)
        max_cpu_usage = max(cpu_usage_samples)
        
        # Ensure CPU usage is reasonable
        assert max_cpu_usage < 100, f"CPU usage too high: {max_cpu_usage}%"
        assert avg_cpu_usage > 0, "CPU usage should be > 0 during operations"

    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated operations."""
        import gc
        import psutil
        
        process = psutil.Process()
        
        # Get baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate repeated operations
        data_arrays = []
        for i in range(10):
            # Create and process arrays
            array = np.random.random((1000, 1000))
            processed = array * 2 + 1
            
            # Only keep every 5th array to simulate some retention
            if i % 5 == 0:
                data_arrays.append(processed)
            
            del array, processed
            
            if i % 3 == 0:
                gc.collect()
        
        # Final cleanup
        del data_arrays
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - baseline_memory
        
        # Memory increase should be minimal after cleanup
        assert memory_increase < 50, f"Potential memory leak: {memory_increase:.2f}MB increase"

    def test_disk_io_performance(self, tmp_path: Path):
        """Test disk I/O performance for model and data operations."""
        import pickle
        
        # Test writing performance
        large_data = np.random.random((1000, 1000))
        file_path = tmp_path / "test_data.pkl"
        
        start_time = time.time()
        
        with open(file_path, 'wb') as f:
            pickle.dump(large_data, f)
        
        write_time = time.time() - start_time
        
        # Test reading performance
        start_time = time.time()
        
        with open(file_path, 'rb') as f:
            loaded_data = pickle.load(f)
        
        read_time = time.time() - start_time
        
        assert write_time < 2.0, f"Disk write too slow: {write_time:.3f}s"
        assert read_time < 1.0, f"Disk read too slow: {read_time:.3f}s"
        assert np.array_equal(large_data, loaded_data), "Data integrity check failed"


@pytest.mark.performance
class TestScalabilityMetrics:
    """Test scalability characteristics."""

    def test_concurrent_inference_capacity(self, sample_image: np.ndarray):
        """Test system capacity for concurrent inferences."""
        import threading
        import queue
        
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.8]])
        
        results_queue = queue.Queue()
        
        def inference_worker():
            try:
                with patch('tensorflow.keras.models.load_model', return_value=mock_model):
                    # Simulate inference
                    result = mock_model.predict(np.expand_dims(sample_image, axis=0))
                    results_queue.put(("success", result))
            except Exception as e:
                results_queue.put(("error", str(e)))
        
        # Start multiple concurrent inference threads
        num_threads = 5
        threads = []
        
        start_time = time.time()
        
        for _ in range(num_threads):
            thread = threading.Thread(target=inference_worker)
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Collect results
        successful_inferences = 0
        failed_inferences = 0
        
        while not results_queue.empty():
            status, _ = results_queue.get()
            if status == "success":
                successful_inferences += 1
            else:
                failed_inferences += 1
        
        # Verify concurrent processing capability
        assert successful_inferences >= num_threads * 0.8, "Too many concurrent inference failures"
        assert total_time < 5.0, f"Concurrent processing too slow: {total_time:.3f}s"

    def test_batch_size_scaling(self, sample_image: np.ndarray):
        """Test performance scaling with different batch sizes."""
        mock_model = Mock()
        
        batch_sizes = [1, 2, 4, 8, 16]
        processing_times = {}
        
        for batch_size in batch_sizes:
            # Create batch
            batch_images = np.stack([sample_image] * batch_size)
            mock_model.predict.return_value = np.random.random((batch_size, 1))
            
            with patch('tensorflow.keras.models.load_model', return_value=mock_model):
                start_time = time.time()
                
                # Simulate batch processing
                result = mock_model.predict(batch_images)
                
                end_time = time.time()
                processing_times[batch_size] = end_time - start_time
        
        # Verify scaling characteristics
        # Processing time should not increase linearly with batch size
        for i in range(1, len(batch_sizes)):
            current_batch = batch_sizes[i]
            previous_batch = batch_sizes[i-1]
            
            time_ratio = processing_times[current_batch] / processing_times[previous_batch]
            batch_ratio = current_batch / previous_batch
            
            # Time ratio should be less than batch ratio (efficiency gains)
            assert time_ratio <= batch_ratio * 1.2, f"Poor batch scaling at size {current_batch}"