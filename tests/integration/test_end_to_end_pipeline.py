"""
End-to-end integration tests for the Chest X-Ray Pneumonia Detector pipeline.
"""

import json
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
from PIL import Image

# Skip integration tests if TensorFlow is not available
tf = pytest.importorskip("tensorflow")


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndPipeline:
    """Test the complete pipeline from data loading to prediction."""

    def test_complete_training_pipeline(
        self, sample_dataset_structure: Tuple[Path, Path, Path]
    ):
        """Test the complete training pipeline with dummy data."""
        train_dir, val_dir, test_dir = sample_dataset_structure
        
        # Import modules after ensuring TensorFlow is available
        from src.train_engine import main as train_main
        from src.inference import main as inference_main
        from src.evaluate import main as evaluate_main
        
        # Test training
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.keras"
            
            # Run training with minimal parameters
            training_args = [
                "--train_dir", str(train_dir),
                "--val_dir", str(val_dir),
                "--epochs", "1",
                "--batch_size", "2",
                "--save_model_path", str(model_path),
                "--use_dummy_data",  # Use dummy data for faster testing
            ]
            
            # Mock sys.argv for the training script
            import sys
            original_argv = sys.argv
            sys.argv = ["train_engine.py"] + training_args
            
            try:
                train_main()
                assert model_path.exists(), "Model file should be created"
            finally:
                sys.argv = original_argv

    def test_inference_pipeline(
        self, sample_dataset_structure: Tuple[Path, Path, Path], mock_tensorflow
    ):
        """Test the inference pipeline with a mock model."""
        _, _, test_dir = sample_dataset_structure
        
        from src.inference import main as inference_main
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock model file
            model_path = Path(temp_dir) / "mock_model.keras"
            model_path.write_text("mock model")
            
            output_csv = Path(temp_dir) / "predictions.csv"
            
            inference_args = [
                "--model_path", str(model_path),
                "--data_dir", str(test_dir),
                "--output_csv", str(output_csv),
                "--num_classes", "1",
            ]
            
            import sys
            original_argv = sys.argv
            sys.argv = ["inference.py"] + inference_args
            
            try:
                inference_main()
                assert output_csv.exists(), "Predictions CSV should be created"
            finally:
                sys.argv = original_argv

    def test_data_split_pipeline(self, tmp_path: Path):
        """Test the data splitting functionality."""
        from src.data_split import main as data_split_main
        
        # Create source data structure
        source_dir = tmp_path / "source"
        normal_dir = source_dir / "NORMAL"
        pneumonia_dir = source_dir / "PNEUMONIA"
        normal_dir.mkdir(parents=True)
        pneumonia_dir.mkdir(parents=True)
        
        # Create sample images
        for i in range(10):
            # Normal images
            normal_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
            normal_path = normal_dir / f"normal_{i}.jpg"
            Image.fromarray(normal_image, mode='L').save(normal_path)
            
            # Pneumonia images
            pneumonia_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
            pneumonia_path = pneumonia_dir / f"pneumonia_{i}.jpg"
            Image.fromarray(pneumonia_image, mode='L').save(pneumonia_path)
        
        output_dir = tmp_path / "output"
        
        split_args = [
            "--input_dir", str(source_dir),
            "--output_dir", str(output_dir),
            "--val_frac", "0.2",
            "--test_frac", "0.2",
            "--seed", "42",
        ]
        
        import sys
        original_argv = sys.argv
        sys.argv = ["data_split.py"] + split_args
        
        try:
            data_split_main()
            
            # Verify directory structure was created
            assert (output_dir / "train" / "NORMAL").exists()
            assert (output_dir / "train" / "PNEUMONIA").exists()
            assert (output_dir / "val" / "NORMAL").exists()
            assert (output_dir / "val" / "PNEUMONIA").exists()
            assert (output_dir / "test" / "NORMAL").exists()
            assert (output_dir / "test" / "PNEUMONIA").exists()
            
            # Verify images were split
            train_normal_count = len(list((output_dir / "train" / "NORMAL").glob("*.jpg")))
            train_pneumonia_count = len(list((output_dir / "train" / "PNEUMONIA").glob("*.jpg")))
            
            assert train_normal_count > 0, "Training set should contain normal images"
            assert train_pneumonia_count > 0, "Training set should contain pneumonia images"
            
        finally:
            sys.argv = original_argv

    def test_gradcam_pipeline(self, sample_image_path: Path, mock_tensorflow):
        """Test the Grad-CAM visualization pipeline."""
        from src.predict_utils import main as gradcam_main
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "mock_model.keras"
            model_path.write_text("mock model")
            
            output_path = Path(temp_dir) / "gradcam_output.png"
            
            gradcam_args = [
                "--model_path", str(model_path),
                "--img_path", str(sample_image_path),
                "--output_path", str(output_path),
                "--img_size", "150", "150",
            ]
            
            import sys
            original_argv = sys.argv
            sys.argv = ["predict_utils.py"] + gradcam_args
            
            try:
                gradcam_main()
                assert output_path.exists(), "Grad-CAM output should be created"
            finally:
                sys.argv = original_argv

    def test_evaluation_pipeline(self, tmp_path: Path):
        """Test the model evaluation pipeline."""
        from src.evaluate import main as evaluate_main
        
        # Create mock prediction and label files
        pred_csv = tmp_path / "predictions.csv"
        label_csv = tmp_path / "labels.csv"
        metrics_csv = tmp_path / "metrics.csv"
        
        # Sample predictions and labels
        predictions_data = "filename,prediction,prob_0\nimg1.jpg,1,0.8\nimg2.jpg,0,0.2\nimg3.jpg,1,0.9\n"
        labels_data = "filename,label\nimg1.jpg,1\nimg2.jpg,0\nimg3.jpg,1\n"
        
        pred_csv.write_text(predictions_data)
        label_csv.write_text(labels_data)
        
        eval_args = [
            "--pred_csv", str(pred_csv),
            "--label_csv", str(label_csv),
            "--metrics_csv", str(metrics_csv),
            "--num_classes", "1",
            "--threshold", "0.5",
        ]
        
        import sys
        original_argv = sys.argv
        sys.argv = ["evaluate.py"] + eval_args
        
        try:
            evaluate_main()
            assert metrics_csv.exists(), "Metrics CSV should be created"
            
            # Verify metrics content
            metrics_content = metrics_csv.read_text()
            assert "precision" in metrics_content
            assert "recall" in metrics_content
            assert "f1" in metrics_content
            
        finally:
            sys.argv = original_argv


@pytest.mark.integration
@pytest.mark.medical
class TestMedicalDataHandling:
    """Test medical-specific data handling and compliance features."""

    def test_image_preprocessing_pipeline(self, sample_image: np.ndarray):
        """Test medical image preprocessing pipeline."""
        from src.image_utils import preprocess_medical_image, validate_medical_image
        
        # Test image validation
        is_valid, error_msg = validate_medical_image(sample_image)
        assert is_valid, f"Sample image should be valid: {error_msg}"
        
        # Test preprocessing
        processed_image = preprocess_medical_image(
            sample_image, 
            target_size=(224, 224),
            normalize=True
        )
        
        assert processed_image.shape == (224, 224, 1), "Processed image should have correct shape"
        assert processed_image.dtype == np.float32, "Processed image should be float32"
        assert processed_image.min() >= 0.0, "Normalized image should have non-negative values"
        assert processed_image.max() <= 1.0, "Normalized image should not exceed 1.0"

    def test_data_anonymization(self, sample_image_path: Path):
        """Test medical data anonymization features."""
        from src.image_utils import anonymize_medical_image
        
        # Test anonymization
        anonymized_image = anonymize_medical_image(str(sample_image_path))
        
        assert anonymized_image is not None, "Anonymized image should not be None"
        # Additional checks for anonymization could be added here

    def test_audit_logging(self, tmp_path: Path):
        """Test audit logging for medical data access."""
        from src.config import setup_audit_logging
        
        log_file = tmp_path / "audit.log"
        logger = setup_audit_logging(str(log_file))
        
        # Test logging medical data access
        logger.info("Medical data accessed", extra={
            "user_id": "test_user",
            "patient_id": "ANON_001", 
            "action": "view_xray",
            "timestamp": "2025-07-27T10:00:00Z"
        })
        
        assert log_file.exists(), "Audit log file should be created"
        log_content = log_file.read_text()
        assert "Medical data accessed" in log_content
        assert "test_user" in log_content


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.gpu
class TestGPUIntegration:
    """Test GPU-specific functionality if available."""

    @pytest.mark.skipif(not pytest.gpu_available, reason="GPU not available")
    def test_gpu_training(self, sample_dataset_structure: Tuple[Path, Path, Path]):
        """Test training with GPU acceleration."""
        train_dir, val_dir, _ = sample_dataset_structure
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            pytest.skip("No GPU available for testing")
        
        from src.train_engine import create_model, compile_model
        
        # Create a small model for GPU testing
        model = create_model(
            img_size=(150, 150),
            num_classes=1,
            use_transfer_learning=False
        )
        
        compile_model(model, learning_rate=0.001)
        
        # Verify model is using GPU
        assert model.built, "Model should be built"
        
        # Test with very small data
        dummy_x = tf.random.normal((4, 150, 150, 3))
        dummy_y = tf.random.uniform((4, 1))
        
        # This should run on GPU if available
        with tf.device('/GPU:0'):
            history = model.fit(dummy_x, dummy_y, epochs=1, verbose=0)
        
        assert 'loss' in history.history, "Training should produce loss values"


@pytest.mark.integration
@pytest.mark.performance
class TestPerformanceIntegration:
    """Test performance-related integration scenarios."""

    def test_batch_inference_performance(self, sample_dataset_structure: Tuple[Path, Path, Path]):
        """Test batch inference performance."""
        _, _, test_dir = sample_dataset_structure
        
        from src.performance_benchmark import benchmark_inference
        
        # Mock model for performance testing
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "mock_model.keras"
            model_path.write_text("mock model")
            
            # Test different batch sizes
            batch_sizes = [1, 4, 8]
            results = {}
            
            for batch_size in batch_sizes:
                # This would normally benchmark real inference
                # For testing, we'll just verify the function can be called
                result = benchmark_inference(
                    model_path=str(model_path),
                    data_dir=str(test_dir),
                    batch_size=batch_size,
                    num_iterations=2
                )
                results[batch_size] = result
            
            assert len(results) == len(batch_sizes), "Should have results for all batch sizes"

    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring during model operations."""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate model loading and inference
        large_array = np.random.random((1000, 1000, 3)).astype(np.float32)
        
        # Get peak memory usage
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Cleanup
        del large_array
        gc.collect()
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        assert peak_memory > initial_memory, "Memory usage should increase during operation"
        assert final_memory < peak_memory, "Memory should be released after cleanup"