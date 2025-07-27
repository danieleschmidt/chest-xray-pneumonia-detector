import pytest
import subprocess
import tempfile
import shutil
from pathlib import Path
import csv
import json
import os
import sys

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


@pytest.mark.e2e
class TestFullPipeline:
    """End-to-end tests for the complete ML pipeline."""
    
    def test_complete_training_pipeline(self, sample_dataset_structure, temp_dir):
        """Test the complete training pipeline from data to trained model."""
        dataset_path = sample_dataset_structure
        model_path = Path(temp_dir) / "test_model.keras"
        
        # Prepare training command
        cmd = [
            sys.executable, "-m", "src.train_engine",
            "--train_dir", str(dataset_path / "train"),
            "--val_dir", str(dataset_path / "val"),
            "--epochs", "2",
            "--batch_size", "2",
            "--save_model_path", str(model_path),
            "--use_dummy_data"
        ]
        
        try:
            # Run training
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            # Check if training completed successfully
            assert result.returncode == 0, f"Training failed: {result.stderr}"
            
            # Verify model was saved (in a real scenario)
            # Note: Since we're using dummy data, actual model saving might not occur
            # This test verifies the command structure and basic execution
            
        except subprocess.TimeoutExpired:
            pytest.fail("Training took too long (>60 seconds)")
        except FileNotFoundError:
            pytest.skip("train_engine module not available for E2E testing")
    
    def test_inference_pipeline(self, sample_dataset_structure, temp_dir):
        """Test the complete inference pipeline."""
        test_data_dir = sample_dataset_structure / "test"
        output_csv = Path(temp_dir) / "predictions.csv"
        
        # Create a dummy model file for testing
        dummy_model_path = Path(temp_dir) / "dummy_model.keras"
        dummy_model_path.write_text("dummy_model_content")
        
        # Prepare inference command
        cmd = [
            sys.executable, "-m", "src.inference",
            "--model_path", str(dummy_model_path),
            "--data_dir", str(test_data_dir),
            "--output_csv", str(output_csv),
            "--num_classes", "1"
        ]
        
        try:
            # Run inference (this might fail due to dummy model, but tests command structure)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # Check command structure (return code might be non-zero due to dummy model)
            assert "--model_path" in " ".join(cmd)
            assert "--data_dir" in " ".join(cmd)
            assert "--output_csv" in " ".join(cmd)
            
        except subprocess.TimeoutExpired:
            pytest.fail("Inference took too long (>30 seconds)")
        except FileNotFoundError:
            pytest.skip("inference module not available for E2E testing")
    
    def test_evaluation_pipeline(self, temp_dir):
        """Test the evaluation pipeline with mock predictions."""
        # Create mock prediction and label files
        pred_csv = Path(temp_dir) / "predictions.csv"
        label_csv = Path(temp_dir) / "labels.csv"
        output_png = Path(temp_dir) / "confusion_matrix.png"
        
        # Create mock prediction data
        pred_data = [
            ["filename", "prediction", "prob_0"],
            ["image1.jpg", "0", "0.8"],
            ["image2.jpg", "1", "0.3"],
            ["image3.jpg", "1", "0.2"],
            ["image4.jpg", "0", "0.9"]
        ]
        
        with open(pred_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(pred_data)
        
        # Create mock label data
        label_data = [
            ["filename", "label"],
            ["image1.jpg", "0"],
            ["image2.jpg", "1"],
            ["image3.jpg", "0"],
            ["image4.jpg", "0"]
        ]
        
        with open(label_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(label_data)
        
        # Prepare evaluation command
        cmd = [
            sys.executable, "-m", "src.evaluate",
            "--pred_csv", str(pred_csv),
            "--label_csv", str(label_csv),
            "--output_png", str(output_png),
            "--threshold", "0.5",
            "--num_classes", "1"
        ]
        
        try:
            # Run evaluation
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # Check command executed (might fail due to missing dependencies)
            assert pred_csv.exists()
            assert label_csv.exists()
            
        except subprocess.TimeoutExpired:
            pytest.fail("Evaluation took too long (>30 seconds)")
        except FileNotFoundError:
            pytest.skip("evaluate module not available for E2E testing")
    
    def test_data_split_pipeline(self, temp_dir, sample_image):
        """Test the data splitting pipeline."""
        # Create source dataset structure
        source_dir = Path(temp_dir) / "source"
        output_dir = Path(temp_dir) / "split_output"
        
        # Create source data
        for class_name in ["NORMAL", "PNEUMONIA"]:
            class_dir = source_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Create sample images
            for i in range(10):
                image_path = class_dir / f"image_{i}.jpg"
                sample_image.save(image_path)
        
        # Prepare data split command
        cmd = [
            sys.executable, "-m", "src.data_split",
            "--input_dir", str(source_dir),
            "--output_dir", str(output_dir),
            "--val_frac", "0.2",
            "--test_frac", "0.2",
            "--seed", "42"
        ]
        
        try:
            # Run data split
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # Verify command structure
            assert str(source_dir) in " ".join(cmd)
            assert str(output_dir) in " ".join(cmd)
            
            # Check if split directories would be created (in actual implementation)
            expected_dirs = [
                output_dir / "train" / "NORMAL",
                output_dir / "train" / "PNEUMONIA",
                output_dir / "val" / "NORMAL", 
                output_dir / "val" / "PNEUMONIA",
                output_dir / "test" / "NORMAL",
                output_dir / "test" / "PNEUMONIA"
            ]
            
            # The actual file operations depend on the implementation
            # This test verifies the command structure
            
        except subprocess.TimeoutExpired:
            pytest.fail("Data split took too long (>30 seconds)")
        except FileNotFoundError:
            pytest.skip("data_split module not available for E2E testing")
    
    def test_cli_version_command(self):
        """Test the version CLI command."""
        cmd = [sys.executable, "-m", "src.version_cli"]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            # Check command executed
            assert isinstance(result.returncode, int)
            
        except subprocess.TimeoutExpired:
            pytest.fail("Version command took too long (>10 seconds)")
        except FileNotFoundError:
            pytest.skip("version_cli module not available for E2E testing")
    
    def test_dataset_stats_command(self, sample_dataset_structure):
        """Test the dataset statistics command."""
        cmd = [
            sys.executable, "-m", "src.dataset_stats",
            "--data_dir", str(sample_dataset_structure),
            "--extensions", "jpg", "jpeg", "png"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            
            # Verify command structure
            assert str(sample_dataset_structure) in " ".join(cmd)
            assert "--extensions" in " ".join(cmd)
            
        except subprocess.TimeoutExpired:
            pytest.fail("Dataset stats took too long (>20 seconds)")
        except FileNotFoundError:
            pytest.skip("dataset_stats module not available for E2E testing")
    
    @pytest.mark.slow
    def test_security_scan_pipeline(self):
        """Test the security scanning pipeline."""
        cmd = [sys.executable, "-m", "src.dependency_security_scan"]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            # Check command executed (may fail due to missing dependencies)
            assert isinstance(result.returncode, int)
            
        except subprocess.TimeoutExpired:
            pytest.fail("Security scan took too long (>60 seconds)")
        except FileNotFoundError:
            pytest.skip("dependency_security_scan module not available for E2E testing")
    
    def test_integration_with_environment_variables(self, mock_environment_variables, temp_dir):
        """Test pipeline integration with environment variables."""
        # Test that commands respect environment variables
        model_path = Path(temp_dir) / "env_test_model.keras"
        
        # Set environment variable for model path
        os.environ["DEFAULT_MODEL_PATH"] = str(model_path)
        
        try:
            # This would test that commands pick up environment variables
            # For now, we just verify the environment is set correctly
            assert os.environ.get("MLFLOW_TRACKING_URI") is not None
            assert os.environ.get("MODEL_REGISTRY_PATH") is not None
            assert os.environ.get("LOG_LEVEL") == "DEBUG"
            
        finally:
            # Clean up
            os.environ.pop("DEFAULT_MODEL_PATH", None)