"""Tests for performance benchmarking functionality."""

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src.performance_benchmark import (
    benchmark_training,
    benchmark_inference,
    BenchmarkResults,
    memory_usage,
)


class TestBenchmarkResults:
    """Test BenchmarkResults dataclass."""

    def test_benchmark_results_creation(self):
        """Test creating BenchmarkResults with all fields."""
        results = BenchmarkResults(
            operation="training",
            total_time=120.5,
            avg_time_per_epoch=12.05,
            peak_memory_mb=1024.0,
            throughput_samples_per_sec=8.3,
            metadata={"epochs": 10, "batch_size": 32}
        )
        
        assert results.operation == "training"
        assert results.total_time == 120.5
        assert results.avg_time_per_epoch == 12.05
        assert results.peak_memory_mb == 1024.0
        assert results.throughput_samples_per_sec == 8.3
        assert results.metadata["epochs"] == 10

    def test_benchmark_results_to_dict(self):
        """Test converting BenchmarkResults to dictionary."""
        results = BenchmarkResults(
            operation="inference",
            total_time=5.2,
            peak_memory_mb=512.0,
            throughput_samples_per_sec=96.2
        )
        
        result_dict = results.to_dict()
        assert result_dict["operation"] == "inference"
        assert result_dict["total_time"] == 5.2
        assert result_dict["peak_memory_mb"] == 512.0
        assert result_dict["throughput_samples_per_sec"] == 96.2


class TestMemoryUsage:
    """Test memory usage tracking functionality."""

    @patch('psutil.virtual_memory')
    def test_memory_usage_context_manager(self, mock_memory):
        """Test memory usage tracking with context manager."""
        mock_memory.return_value.used = 1000000000  # 1GB initial
        
        with memory_usage() as tracker:
            mock_memory.return_value.used = 1500000000  # 1.5GB peak
            
        assert tracker.peak_memory_mb > 0
        assert tracker.initial_memory_mb > 0

    @patch('psutil.virtual_memory')
    def test_memory_usage_no_psutil(self, mock_memory):
        """Test memory usage tracking when psutil is not available."""
        mock_memory.side_effect = ImportError("psutil not available")
        
        with memory_usage() as tracker:
            pass
            
        assert tracker.peak_memory_mb == 0
        assert tracker.initial_memory_mb == 0


class TestBenchmarkTraining:
    """Test training benchmarking functionality."""

    @patch('src.performance_benchmark.create_simple_cnn')
    @patch('src.performance_benchmark.create_data_generators')
    @patch('src.performance_benchmark.time.time')
    def test_benchmark_training_simple_cnn(self, mock_time, mock_data_gen, mock_model):
        """Test benchmarking training with simple CNN."""
        # Mock time progression
        mock_time.side_effect = [0, 10, 20, 30]  # Start, epoch 1, epoch 2, end
        
        # Mock model
        mock_model_instance = MagicMock()
        mock_model_instance.fit.return_value.history = {
            'loss': [0.5, 0.3], 
            'val_loss': [0.6, 0.4]
        }
        mock_model.return_value = mock_model_instance
        
        # Mock data generators
        mock_train_gen = MagicMock()
        mock_val_gen = MagicMock()
        mock_train_gen.__len__.return_value = 10  # 10 batches
        mock_val_gen.__len__.return_value = 3     # 3 batches
        mock_data_gen.return_value = (mock_train_gen, mock_val_gen)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            results = benchmark_training(
                train_dir=temp_dir,
                val_dir=temp_dir,
                epochs=2,
                batch_size=32,
                use_dummy_data=True
            )
        
        assert results.operation == "training"
        assert results.total_time == 30
        assert results.avg_time_per_epoch == 15
        assert results.metadata["epochs"] == 2
        assert results.metadata["batch_size"] == 32

    @patch('src.performance_benchmark.create_transfer_learning_model')
    @patch('src.performance_benchmark.create_data_generators')
    def test_benchmark_training_transfer_learning(self, mock_data_gen, mock_model):
        """Test benchmarking training with transfer learning."""
        mock_model_instance = MagicMock()
        mock_model_instance.fit.return_value.history = {'loss': [0.4]}
        mock_model.return_value = mock_model_instance
        
        mock_train_gen = MagicMock()
        mock_val_gen = MagicMock()
        mock_train_gen.__len__.return_value = 5
        mock_val_gen.__len__.return_value = 2
        mock_data_gen.return_value = (mock_train_gen, mock_val_gen)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            results = benchmark_training(
                train_dir=temp_dir,
                val_dir=temp_dir,
                epochs=1,
                use_transfer_learning=True,
                use_dummy_data=True
            )
        
        assert results.operation == "training"
        assert results.metadata["use_transfer_learning"] is True


class TestBenchmarkInference:
    """Test inference benchmarking functionality."""

    def test_benchmark_inference_with_dummy_model(self):
        """Test benchmarking inference with a dummy model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dummy model file
            model_path = os.path.join(temp_dir, "model.keras")
            
            # Create test data directory structure
            data_dir = os.path.join(temp_dir, "test_data", "class1")
            os.makedirs(data_dir, exist_ok=True)
            
            # Create a dummy image file
            dummy_image_path = os.path.join(data_dir, "test.jpg")
            with open(dummy_image_path, 'w') as f:
                f.write("dummy")
            
            with patch('tensorflow.keras.models.load_model') as mock_load:
                mock_model = MagicMock()
                mock_model.predict.return_value = np.array([[0.7]])
                mock_load.return_value = mock_model
                
                with patch('src.performance_benchmark.predict_directory') as mock_predict:
                    mock_predict.return_value = pd.DataFrame({
                        'filepath': [dummy_image_path],
                        'prediction': [0.7]
                    })
                    
                    results = benchmark_inference(
                        model_path=model_path,
                        data_dir=os.path.join(temp_dir, "test_data"),
                        num_samples=1
                    )
            
            assert results.operation == "inference"
            assert results.total_time > 0
            assert results.throughput_samples_per_sec > 0
            assert results.metadata["num_samples"] == 1

    def test_benchmark_inference_multiclass(self):
        """Test benchmarking inference for multiclass model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.keras")
            data_dir = os.path.join(temp_dir, "test_data", "class1")
            os.makedirs(data_dir, exist_ok=True)
            
            dummy_image_path = os.path.join(data_dir, "test.jpg")
            with open(dummy_image_path, 'w') as f:
                f.write("dummy")
            
            with patch('tensorflow.keras.models.load_model'):
                with patch('src.performance_benchmark.predict_directory') as mock_predict:
                    mock_predict.return_value = pd.DataFrame({
                        'filepath': [dummy_image_path],
                        'prediction': [1],
                        'prob_0': [0.2],
                        'prob_1': [0.8]
                    })
                    
                    results = benchmark_inference(
                        model_path=model_path,
                        data_dir=os.path.join(temp_dir, "test_data"),
                        num_classes=2,
                        num_samples=1
                    )
            
            assert results.operation == "inference"
            assert results.metadata["num_classes"] == 2


class TestBenchmarkCLI:
    """Test command-line interface for benchmarking."""

    @patch('src.performance_benchmark.benchmark_training')
    def test_cli_training_benchmark(self, mock_benchmark):
        """Test CLI for training benchmark."""
        mock_benchmark.return_value = BenchmarkResults(
            operation="training",
            total_time=60.0,
            avg_time_per_epoch=20.0,
            peak_memory_mb=512.0,
            throughput_samples_per_sec=10.0
        )
        
        from src.performance_benchmark import main
        
        with patch('sys.argv', ['benchmark', 'training', '--epochs', '3']):
            with patch('builtins.print') as mock_print:
                main()
                
        mock_benchmark.assert_called_once()
        mock_print.assert_called()

    @patch('src.performance_benchmark.benchmark_inference')
    def test_cli_inference_benchmark(self, mock_benchmark):
        """Test CLI for inference benchmark."""
        mock_benchmark.return_value = BenchmarkResults(
            operation="inference",
            total_time=5.0,
            peak_memory_mb=256.0,
            throughput_samples_per_sec=20.0
        )
        
        from src.performance_benchmark import main
        
        with patch('sys.argv', ['benchmark', 'inference', '--model_path', 'model.keras']):
            with patch('builtins.print') as mock_print:
                main()
                
        mock_benchmark.assert_called_once()
        mock_print.assert_called()

    def test_cli_help(self):
        """Test CLI help functionality."""
        from src.performance_benchmark import main
        
        with patch('sys.argv', ['benchmark', '--help']):
            with pytest.raises(SystemExit):
                main()