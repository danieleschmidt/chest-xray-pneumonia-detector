"""End-to-end integration tests for the full ML pipeline."""

import pytest
import tempfile
import os
import subprocess
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from unittest.mock import patch, Mock
import shutil

pytest.importorskip("tensorflow")


class TestEndToEndPipeline:
    """Test suite for complete train→predict→evaluate workflow."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for integration test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def dummy_dataset(self, temp_dir):
        """Create a minimal dummy dataset for testing."""
        # Create directory structure
        data_dir = os.path.join(temp_dir, 'data')
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        
        # Create class directories
        for split in ['train', 'val']:
            for class_name in ['NORMAL', 'PNEUMONIA']:
                class_dir = os.path.join(data_dir, split, class_name)
                os.makedirs(class_dir, exist_ok=True)
                
                # Create 3 dummy images per class/split (minimal viable dataset)
                for i in range(3):
                    img = Image.new('RGB', (64, 64), color='white' if class_name == 'NORMAL' else 'gray')
                    img_path = os.path.join(class_dir, f'{class_name.lower()}_{i}.jpg')
                    img.save(img_path)
        
        return {
            'data_dir': data_dir,
            'train_dir': train_dir,
            'val_dir': val_dir
        }

    @pytest.fixture
    def model_config(self, temp_dir):
        """Create minimal model configuration for fast testing."""
        return {
            'epochs': 2,
            'batch_size': 2,
            'img_size': (64, 64),
            'model_path': os.path.join(temp_dir, 'test_model.keras'),
            'checkpoint_path': os.path.join(temp_dir, 'test_checkpoint.keras'),
            'history_csv': os.path.join(temp_dir, 'training_history.csv'),
            'cm_path': os.path.join(temp_dir, 'confusion_matrix.png'),
            'plot_path': os.path.join(temp_dir, 'training_plot.png')
        }

    def test_minimal_pipeline_workflow(self, dummy_dataset, model_config, temp_dir):
        """Test minimal train→predict→evaluate workflow with dummy data."""
        # This test verifies that the basic pipeline can complete without errors
        
        # Step 1: Training Phase
        training_args = [
            sys.executable, '-m', 'src.train_engine',
            '--train_dir', dummy_dataset['train_dir'],
            '--val_dir', dummy_dataset['val_dir'],
            '--epochs', str(model_config['epochs']),
            '--batch_size', str(model_config['batch_size']),
            '--save_model_path', model_config['model_path'],
            '--checkpoint_path', model_config['checkpoint_path'],
            '--history_csv', model_config['history_csv'],
            '--cm_path', model_config['cm_path'],
            '--plot_path', model_config['plot_path'],
            '--use_dummy_data',  # Use built-in dummy data for speed
            '--seed', '42'
        ]
        
        # Mock MLflow to avoid external dependencies
        with patch('mlflow.start_run'), \
             patch('mlflow.log_params'), \
             patch('mlflow.log_metrics'), \
             patch('mlflow.log_artifacts'), \
             patch('mlflow.end_run'):
            
            # Run training
            result = subprocess.run(training_args, capture_output=True, text=True, timeout=120)
            
            # Verify training completed successfully
            assert result.returncode == 0, f"Training failed: {result.stderr}"
            
            # Verify model artifacts were created
            assert os.path.exists(model_config['model_path']), "Model file not created"
            assert os.path.exists(model_config['history_csv']), "Training history not saved"

    def test_inference_after_training(self, dummy_dataset, model_config, temp_dir):
        """Test inference on trained model."""
        # First create a minimal trained model (mock)
        with patch('tensorflow.keras.models.load_model') as mock_load_model:
            # Create a mock model
            mock_model = Mock()
            mock_model.predict.return_value = np.array([[0.7], [0.3], [0.8], [0.2], [0.9], [0.1]])
            mock_load_model.return_value = mock_model
            
            # Create predictions CSV path
            predictions_csv = os.path.join(temp_dir, 'predictions.csv')
            
            # Run inference
            inference_args = [
                sys.executable, '-m', 'src.inference',
                '--model_path', model_config['model_path'],
                '--data_dir', dummy_dataset['val_dir'],
                '--output_csv', predictions_csv,
                '--img_size', '64', '64',
                '--num_classes', '1'
            ]
            
            with patch('tensorflow.keras.preprocessing.image.ImageDataGenerator') as mock_datagen_class:
                # Mock the data generator
                mock_datagen = Mock()
                mock_generator = Mock()
                mock_generator.filepaths = [
                    'NORMAL/normal_0.jpg', 'NORMAL/normal_1.jpg', 'NORMAL/normal_2.jpg',
                    'PNEUMONIA/pneumonia_0.jpg', 'PNEUMONIA/pneumonia_1.jpg', 'PNEUMONIA/pneumonia_2.jpg'
                ]
                mock_datagen.flow_from_directory.return_value = mock_generator
                mock_datagen_class.return_value = mock_datagen
                
                result = subprocess.run(inference_args, capture_output=True, text=True, timeout=30)
                
                # Verify inference completed successfully
                assert result.returncode == 0, f"Inference failed: {result.stderr}"
                assert os.path.exists(predictions_csv), "Predictions CSV not created"
                
                # Verify predictions file has correct structure
                df = pd.read_csv(predictions_csv)
                assert 'filepath' in df.columns, "Predictions missing filepath column"
                assert 'prediction' in df.columns, "Predictions missing prediction column"
                assert len(df) == 6, f"Expected 6 predictions, got {len(df)}"

    def test_evaluation_after_inference(self, temp_dir):
        """Test evaluation on prediction results."""
        # Create mock predictions CSV
        predictions_data = pd.DataFrame({
            'filepath': ['NORMAL/img1.jpg', 'NORMAL/img2.jpg', 'PNEUMONIA/img3.jpg', 'PNEUMONIA/img4.jpg'],
            'prediction': [0.2, 0.3, 0.8, 0.9],
            'label': [0, 0, 1, 1]  # Ground truth labels
        })
        
        predictions_csv = os.path.join(temp_dir, 'test_predictions.csv')
        predictions_data.to_csv(predictions_csv, index=False)
        
        # Paths for evaluation outputs
        confusion_matrix_path = os.path.join(temp_dir, 'eval_confusion_matrix.png')
        metrics_csv = os.path.join(temp_dir, 'eval_metrics.csv')
        
        # Run evaluation
        evaluation_args = [
            sys.executable, '-m', 'src.evaluate',
            '--pred_csv', predictions_csv,
            '--output_png', confusion_matrix_path,
            '--metrics_csv', metrics_csv,
            '--threshold', '0.5',
            '--num_classes', '1'
        ]
        
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'):
            
            result = subprocess.run(evaluation_args, capture_output=True, text=True, timeout=30)
            
            # Verify evaluation completed successfully
            assert result.returncode == 0, f"Evaluation failed: {result.stderr}"
            assert os.path.exists(metrics_csv), "Metrics CSV not created"
            
            # Verify metrics file has correct structure
            metrics_df = pd.read_csv(metrics_csv)
            expected_metrics = ['precision', 'recall', 'f1', 'roc_auc']
            for metric in expected_metrics:
                assert metric in metrics_df.columns, f"Missing metric: {metric}"
            
            # Verify metrics are reasonable (not NaN, within valid ranges)
            for metric in expected_metrics:
                value = metrics_df[metric].iloc[0]
                assert not pd.isna(value), f"Metric {metric} is NaN"
                assert 0 <= value <= 1, f"Metric {metric} out of range: {value}"

    def test_data_split_utility(self, temp_dir):
        """Test the data splitting utility as part of preprocessing pipeline."""
        # Create source data directory
        source_dir = os.path.join(temp_dir, 'source_data')
        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_dir = os.path.join(source_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Create 10 dummy images per class
            for i in range(10):
                img = Image.new('RGB', (64, 64), color='white' if class_name == 'NORMAL' else 'gray')
                img_path = os.path.join(class_dir, f'{class_name.lower()}_{i}.jpg')
                img.save(img_path)
        
        # Output directory for split data
        output_dir = os.path.join(temp_dir, 'split_data')
        
        # Run data splitting
        split_args = [
            sys.executable, '-m', 'src.data_split',
            '--input_dir', source_dir,
            '--output_dir', output_dir,
            '--val_frac', '0.2',
            '--test_frac', '0.2',
            '--seed', '42'
        ]
        
        result = subprocess.run(split_args, capture_output=True, text=True, timeout=30)
        
        # Verify split completed successfully
        assert result.returncode == 0, f"Data split failed: {result.stderr}"
        
        # Verify directory structure was created
        for split in ['train', 'val', 'test']:
            for class_name in ['NORMAL', 'PNEUMONIA']:
                split_class_dir = os.path.join(output_dir, split, class_name)
                assert os.path.exists(split_class_dir), f"Missing directory: {split_class_dir}"
                
                # Verify images were split (approximately correct proportions)
                num_images = len(os.listdir(split_class_dir))
                if split == 'train':
                    assert 4 <= num_images <= 8, f"Train split has unexpected count: {num_images}"
                else:  # val or test
                    assert 1 <= num_images <= 3, f"{split} split has unexpected count: {num_images}"

    def test_grad_cam_visualization_pipeline(self, model_config, temp_dir):
        """Test Grad-CAM visualization as part of model interpretation pipeline."""
        # Create a test image
        test_image_path = os.path.join(temp_dir, 'test_image.jpg')
        img = Image.new('RGB', (150, 150), color='blue')
        img.save(test_image_path)
        
        # Output path for Grad-CAM overlay
        gradcam_output = os.path.join(temp_dir, 'gradcam_overlay.png')
        
        # Mock the entire Grad-CAM pipeline
        with patch('tensorflow.keras.models.load_model') as mock_load_model, \
             patch('src.predict_utils.generate_grad_cam') as mock_generate_grad_cam, \
             patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'):
            
            # Setup mocks
            mock_model = Mock()
            mock_load_model.return_value = mock_model
            mock_generate_grad_cam.return_value = np.random.rand(14, 14)
            
            # Run Grad-CAM generation
            gradcam_args = [
                sys.executable, '-m', 'src.predict_utils',
                '--model_path', model_config['model_path'],
                '--img_path', test_image_path,
                '--output_path', gradcam_output,
                '--img_size', '150', '150',
                '--last_conv_layer_name', 'conv_pw_13_relu'
            ]
            
            result = subprocess.run(gradcam_args, capture_output=True, text=True, timeout=30)
            
            # Verify Grad-CAM generation completed successfully
            assert result.returncode == 0, f"Grad-CAM generation failed: {result.stderr}"
            
            # Verify mocks were called appropriately
            mock_load_model.assert_called_once_with(model_config['model_path'])
            mock_generate_grad_cam.assert_called_once()

    def test_pipeline_error_handling(self, temp_dir):
        """Test that pipeline components handle errors gracefully."""
        # Test training with invalid data directory
        invalid_args = [
            sys.executable, '-m', 'src.train_engine',
            '--train_dir', '/nonexistent/path',
            '--val_dir', '/nonexistent/path',
            '--epochs', '1'
        ]
        
        result = subprocess.run(invalid_args, capture_output=True, text=True, timeout=30)
        
        # Should fail gracefully (non-zero exit code)
        assert result.returncode != 0, "Expected training to fail with invalid paths"
        assert result.stderr, "Expected error message in stderr"

    def test_version_consistency(self):
        """Test that version CLI works as part of the pipeline."""
        version_args = [sys.executable, '-m', 'src.version_cli']
        
        result = subprocess.run(version_args, capture_output=True, text=True, timeout=10)
        
        # Verify version command works
        assert result.returncode == 0, f"Version command failed: {result.stderr}"
        assert result.stdout.strip(), "Version output should not be empty"