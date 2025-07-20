"""Integration tests with synthetic medical data for realistic pipeline validation."""

import pytest
import tempfile
import os
import subprocess
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
from unittest.mock import patch, Mock
import shutil
import json
import time

pytest.importorskip("tensorflow")

from src.synthetic_medical_data_generator import (
    SyntheticMedicalDataGenerator,
    MedicalImageConfiguration,
    DatasetMetadata,
    generate_synthetic_chest_xray,
    create_synthetic_medical_dataset,
)


class TestSyntheticMedicalDataGenerator:
    """Test synthetic medical data generation functionality."""

    def test_medical_image_configuration_creation(self):
        """Test creating medical image configuration."""
        config = MedicalImageConfiguration(
            image_size=(224, 224),
            pathology_probability=0.3,
            noise_level=0.1,
            contrast_enhancement=True,
            add_anatomical_markers=True
        )
        
        assert config.image_size == (224, 224)
        assert config.pathology_probability == 0.3
        assert config.noise_level == 0.1
        assert config.contrast_enhancement is True

    def test_dataset_metadata_creation(self):
        """Test creating dataset metadata."""
        metadata = DatasetMetadata(
            total_images=1000,
            normal_count=700,
            pneumonia_count=300,
            image_format="PNG",
            image_size=(224, 224),
            generation_timestamp="2025-07-20T15:00:00Z",
            pathology_types=["bacterial_pneumonia", "viral_pneumonia"],
            quality_metrics={"avg_contrast": 0.75, "noise_level": 0.1}
        )
        
        assert metadata.total_images == 1000
        assert metadata.normal_count == 700
        assert len(metadata.pathology_types) == 2

    def test_generate_synthetic_chest_xray_normal(self):
        """Test generating normal chest X-ray."""
        config = MedicalImageConfiguration(
            image_size=(128, 128),
            pathology_probability=0.0  # Force normal
        )
        
        image = generate_synthetic_chest_xray(config, is_pathological=False)
        
        assert image.size == (128, 128)
        assert image.mode == "L"  # Grayscale for medical images

    def test_generate_synthetic_chest_xray_pathological(self):
        """Test generating pathological chest X-ray."""
        config = MedicalImageConfiguration(
            image_size=(128, 128),
            pathology_probability=1.0  # Force pathological
        )
        
        image = generate_synthetic_chest_xray(config, is_pathological=True)
        
        assert image.size == (128, 128)
        assert image.mode == "L"  # Grayscale for medical images

    def test_generate_different_image_sizes(self):
        """Test generating images with different sizes."""
        sizes = [(64, 64), (150, 150), (224, 224), (512, 512)]
        
        for size in sizes:
            config = MedicalImageConfiguration(image_size=size)
            image = generate_synthetic_chest_xray(config, is_pathological=False)
            assert image.size == size

    def test_contrast_enhancement_effect(self):
        """Test that contrast enhancement affects image properties."""
        config_normal = MedicalImageConfiguration(
            image_size=(128, 128),
            contrast_enhancement=False
        )
        config_enhanced = MedicalImageConfiguration(
            image_size=(128, 128),
            contrast_enhancement=True
        )
        
        image_normal = generate_synthetic_chest_xray(config_normal, is_pathological=False)
        image_enhanced = generate_synthetic_chest_xray(config_enhanced, is_pathological=False)
        
        # Enhanced image should have different pixel statistics
        normal_std = np.array(image_normal).std()
        enhanced_std = np.array(image_enhanced).std()
        
        # Enhanced image should have more contrast (different standard deviation)
        assert normal_std != enhanced_std


class TestSyntheticMedicalDatasetGeneration:
    """Test full synthetic medical dataset generation."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_create_small_synthetic_dataset(self, temp_dir):
        """Test creating a small synthetic medical dataset."""
        config = MedicalImageConfiguration(
            image_size=(64, 64),
            pathology_probability=0.4
        )
        
        dataset_path = create_synthetic_medical_dataset(
            output_dir=temp_dir,
            total_images=20,
            train_split=0.7,
            val_split=0.2,
            test_split=0.1,
            config=config
        )
        
        # Verify directory structure
        assert os.path.exists(os.path.join(dataset_path, "train", "NORMAL"))
        assert os.path.exists(os.path.join(dataset_path, "train", "PNEUMONIA"))
        assert os.path.exists(os.path.join(dataset_path, "val", "NORMAL"))
        assert os.path.exists(os.path.join(dataset_path, "val", "PNEUMONIA"))
        assert os.path.exists(os.path.join(dataset_path, "test", "NORMAL"))
        assert os.path.exists(os.path.join(dataset_path, "test", "PNEUMONIA"))
        
        # Verify metadata file exists
        metadata_path = os.path.join(dataset_path, "dataset_metadata.json")
        assert os.path.exists(metadata_path)
        
        # Verify metadata content
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        assert metadata["total_images"] == 20
        assert "generation_timestamp" in metadata

    def test_create_medium_synthetic_dataset(self, temp_dir):
        """Test creating a medium-sized synthetic dataset."""
        config = MedicalImageConfiguration(
            image_size=(150, 150),
            pathology_probability=0.3,
            noise_level=0.15,
            contrast_enhancement=True
        )
        
        dataset_path = create_synthetic_medical_dataset(
            output_dir=temp_dir,
            total_images=100,
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
            config=config
        )
        
        # Count images in each split
        train_normal = len(os.listdir(os.path.join(dataset_path, "train", "NORMAL")))
        train_pneumonia = len(os.listdir(os.path.join(dataset_path, "train", "PNEUMONIA")))
        
        total_train = train_normal + train_pneumonia
        assert 70 <= total_train <= 90  # Approximately 80% of 100 images

    def test_dataset_split_ratios(self, temp_dir):
        """Test that dataset splits follow specified ratios."""
        config = MedicalImageConfiguration(image_size=(64, 64))
        
        dataset_path = create_synthetic_medical_dataset(
            output_dir=temp_dir,
            total_images=100,
            train_split=0.6,
            val_split=0.2,
            test_split=0.2,
            config=config
        )
        
        # Count images in each split
        splits = ["train", "val", "test"]
        split_counts = {}
        
        for split in splits:
            normal_count = len(os.listdir(os.path.join(dataset_path, split, "NORMAL")))
            pneumonia_count = len(os.listdir(os.path.join(dataset_path, split, "PNEUMONIA")))
            split_counts[split] = normal_count + pneumonia_count
        
        total_images = sum(split_counts.values())
        assert total_images == 100
        
        # Check approximate ratios (allow for rounding)
        assert 55 <= split_counts["train"] <= 65  # ~60%
        assert 15 <= split_counts["val"] <= 25    # ~20%
        assert 15 <= split_counts["test"] <= 25   # ~20%

    def test_pathology_distribution(self, temp_dir):
        """Test that pathology distribution matches configuration."""
        config = MedicalImageConfiguration(
            image_size=(64, 64),
            pathology_probability=0.25  # 25% pathological
        )
        
        dataset_path = create_synthetic_medical_dataset(
            output_dir=temp_dir,
            total_images=100,
            train_split=1.0,  # All in training for easier counting
            val_split=0.0,
            test_split=0.0,
            config=config
        )
        
        normal_count = len(os.listdir(os.path.join(dataset_path, "train", "NORMAL")))
        pneumonia_count = len(os.listdir(os.path.join(dataset_path, "train", "PNEUMONIA")))
        
        total = normal_count + pneumonia_count
        pneumonia_ratio = pneumonia_count / total
        
        # Allow for some variance due to randomness
        assert 0.15 <= pneumonia_ratio <= 0.35  # Target 0.25 ± 0.10


class TestSyntheticMedicalDataIntegration:
    """Integration tests using synthetic medical datasets."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for integration tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def synthetic_dataset(self, temp_dir):
        """Create synthetic medical dataset for integration testing."""
        config = MedicalImageConfiguration(
            image_size=(150, 150),
            pathology_probability=0.3,
            noise_level=0.1,
            contrast_enhancement=True,
            add_anatomical_markers=True
        )
        
        dataset_path = create_synthetic_medical_dataset(
            output_dir=temp_dir,
            total_images=60,  # Small but sufficient for testing
            train_split=0.7,
            val_split=0.2,
            test_split=0.1,
            config=config
        )
        
        return {
            'dataset_path': dataset_path,
            'train_dir': os.path.join(dataset_path, 'train'),
            'val_dir': os.path.join(dataset_path, 'val'),
            'test_dir': os.path.join(dataset_path, 'test'),
            'metadata_path': os.path.join(dataset_path, 'dataset_metadata.json')
        }

    def test_data_loader_with_synthetic_dataset(self, synthetic_dataset):
        """Test data loader functionality with synthetic medical data."""
        from src.data_loader import create_data_generators
        
        train_gen, val_gen = create_data_generators(
            train_dir=synthetic_dataset['train_dir'],
            val_dir=synthetic_dataset['val_dir'],
            target_size=(150, 150),
            train_batch_size=4,
            val_batch_size=4
        )
        
        # Test that generators work
        assert train_gen.batch_size == 4
        assert val_gen.batch_size == 4
        assert train_gen.target_size == (150, 150)
        
        # Test batch generation
        train_batch = next(train_gen)
        val_batch = next(val_gen)
        
        assert train_batch[0].shape[1:] == (150, 150, 3)  # Batch of images
        assert len(train_batch[1]) == train_batch[0].shape[0]  # Labels match batch size

    def test_model_training_with_synthetic_data(self, synthetic_dataset):
        """Test model training pipeline with synthetic medical data."""
        from src.model_builder import create_simple_cnn
        from src.data_loader import create_data_generators
        
        # Create small model for testing
        model = create_simple_cnn(
            input_shape=(150, 150, 3),
            num_classes=1,
            learning_rate=0.01
        )
        
        # Create data generators
        train_gen, val_gen = create_data_generators(
            train_dir=synthetic_dataset['train_dir'],
            val_dir=synthetic_dataset['val_dir'],
            target_size=(150, 150),
            train_batch_size=4,
            val_batch_size=4
        )
        
        # Test training for 1 epoch
        history = model.fit(
            train_gen,
            epochs=1,
            validation_data=val_gen,
            verbose=0,
            steps_per_epoch=2,  # Minimal training
            validation_steps=1
        )
        
        assert 'loss' in history.history
        assert 'val_loss' in history.history
        assert len(history.history['loss']) == 1

    def test_inference_with_synthetic_data(self, synthetic_dataset):
        """Test inference pipeline with synthetic medical data."""
        from src.model_builder import create_simple_cnn
        from src.inference import predict_directory
        import tempfile
        
        # Create and save a simple model
        model = create_simple_cnn(input_shape=(150, 150, 3), num_classes=1)
        
        with tempfile.TemporaryDirectory() as model_dir:
            model_path = os.path.join(model_dir, 'test_model.keras')
            model.save(model_path)
            
            # Test inference
            predictions_df = predict_directory(
                model_path=model_path,
                data_dir=synthetic_dataset['test_dir'],
                img_size=(150, 150),
                num_classes=1
            )
            
            assert 'filepath' in predictions_df.columns
            assert 'prediction' in predictions_df.columns
            assert len(predictions_df) > 0

    def test_evaluation_with_synthetic_data(self, synthetic_dataset):
        """Test evaluation pipeline with synthetic medical data."""
        from src.evaluate import evaluate_predictions
        import tempfile
        
        # Create mock predictions CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as pred_file:
            pred_data = {
                'filepath': [
                    os.path.join(synthetic_dataset['test_dir'], 'NORMAL', 'normal_0.png'),
                    os.path.join(synthetic_dataset['test_dir'], 'PNEUMONIA', 'pneumonia_0.png')
                ],
                'prediction': [0.2, 0.8]  # Mock predictions
            }
            pd.DataFrame(pred_data).to_csv(pred_file.name, index=False)
            pred_file_path = pred_file.name
        
        # Create mock labels CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as label_file:
            label_data = {
                'filepath': [
                    os.path.join(synthetic_dataset['test_dir'], 'NORMAL', 'normal_0.png'),
                    os.path.join(synthetic_dataset['test_dir'], 'PNEUMONIA', 'pneumonia_0.png')
                ],
                'label': [0, 1]  # True labels
            }
            pd.DataFrame(label_data).to_csv(label_file.name, index=False)
            label_file_path = label_file.name
        
        try:
            metrics = evaluate_predictions(
                pred_csv=pred_file_path,
                label_csv=label_file_path,
                threshold=0.5,
                num_classes=1
            )
            
            assert 'accuracy' in metrics
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1_score' in metrics
        finally:
            os.unlink(pred_file_path)
            os.unlink(label_file_path)

    def test_end_to_end_pipeline_with_synthetic_data(self, synthetic_dataset):
        """Test complete end-to-end pipeline with synthetic medical data."""
        from src.model_builder import create_simple_cnn
        from src.data_loader import create_data_generators
        from src.inference import predict_directory
        import tempfile
        
        # Step 1: Train model
        model = create_simple_cnn(input_shape=(150, 150, 3), num_classes=1)
        
        train_gen, val_gen = create_data_generators(
            train_dir=synthetic_dataset['train_dir'],
            val_dir=synthetic_dataset['val_dir'],
            target_size=(150, 150),
            train_batch_size=4,
            val_batch_size=4
        )
        
        # Quick training
        model.fit(
            train_gen,
            epochs=1,
            validation_data=val_gen,
            verbose=0,
            steps_per_epoch=2,
            validation_steps=1
        )
        
        # Step 2: Save model and run inference
        with tempfile.TemporaryDirectory() as model_dir:
            model_path = os.path.join(model_dir, 'trained_model.keras')
            model.save(model_path)
            
            # Step 3: Predict on test data
            predictions_df = predict_directory(
                model_path=model_path,
                data_dir=synthetic_dataset['test_dir'],
                img_size=(150, 150),
                num_classes=1
            )
            
            # Step 4: Verify pipeline completed successfully
            assert len(predictions_df) > 0
            assert all(col in predictions_df.columns for col in ['filepath', 'prediction'])
            assert all(0 <= pred <= 1 for pred in predictions_df['prediction'])

    def test_synthetic_data_quality_metrics(self, synthetic_dataset):
        """Test quality metrics of synthetic medical data."""
        # Load metadata
        with open(synthetic_dataset['metadata_path'], 'r') as f:
            metadata = json.load(f)
        
        # Verify metadata structure
        assert 'total_images' in metadata
        assert 'normal_count' in metadata
        assert 'pneumonia_count' in metadata
        assert 'generation_timestamp' in metadata
        assert 'image_size' in metadata
        
        # Test image quality
        normal_dir = os.path.join(synthetic_dataset['train_dir'], 'NORMAL')
        pneumonia_dir = os.path.join(synthetic_dataset['train_dir'], 'PNEUMONIA')
        
        # Check that images exist and are readable
        normal_images = os.listdir(normal_dir)
        pneumonia_images = os.listdir(pneumonia_dir)
        
        assert len(normal_images) > 0
        assert len(pneumonia_images) > 0
        
        # Test loading a few images
        for img_name in normal_images[:2]:
            img_path = os.path.join(normal_dir, img_name)
            img = Image.open(img_path)
            assert img.size == (150, 150)
            assert img.mode in ['L', 'RGB']

    def test_large_scale_synthetic_dataset_performance(self, temp_dir):
        """Test performance with larger synthetic dataset."""
        config = MedicalImageConfiguration(
            image_size=(224, 224),
            pathology_probability=0.35
        )
        
        start_time = time.time()
        
        dataset_path = create_synthetic_medical_dataset(
            output_dir=temp_dir,
            total_images=200,  # Larger dataset
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
            config=config
        )
        
        generation_time = time.time() - start_time
        
        # Verify dataset was created efficiently
        assert generation_time < 60  # Should complete within 60 seconds
        
        # Verify all images were created
        total_created = 0
        for split in ['train', 'val', 'test']:
            for class_name in ['NORMAL', 'PNEUMONIA']:
                split_dir = os.path.join(dataset_path, split, class_name)
                total_created += len(os.listdir(split_dir))
        
        assert total_created == 200


class TestSyntheticDataCLIIntegration:
    """Test CLI integration with synthetic medical datasets."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for CLI tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_synthetic_data_generation_cli(self, temp_dir):
        """Test CLI for synthetic data generation."""
        # This would test a CLI command like:
        # python -m src.synthetic_medical_data_generator --output-dir temp_dir --total-images 50
        
        from src.synthetic_medical_data_generator import main
        
        with patch('sys.argv', [
            'synthetic_medical_data_generator',
            '--output-dir', temp_dir,
            '--total-images', '50',
            '--image-size', '128',
            '--pathology-probability', '0.3'
        ]):
            main()
        
        # Verify output
        dataset_dir = os.path.join(temp_dir, 'synthetic_medical_dataset')
        assert os.path.exists(dataset_dir)
        assert os.path.exists(os.path.join(dataset_dir, 'dataset_metadata.json'))

    def test_benchmark_with_synthetic_data(self, temp_dir):
        """Test performance benchmarking with synthetic medical data."""
        # Create synthetic dataset first
        config = MedicalImageConfiguration(image_size=(150, 150))
        dataset_path = create_synthetic_medical_dataset(
            output_dir=temp_dir,
            total_images=40,
            train_split=0.8,
            val_split=0.2,
            test_split=0.0,
            config=config
        )
        
        # Test with performance benchmark CLI
        from src.performance_benchmark import benchmark_training
        
        result = benchmark_training(
            train_dir=os.path.join(dataset_path, 'train'),
            val_dir=os.path.join(dataset_path, 'val'),
            epochs=1,
            batch_size=4,
            use_dummy_data=False  # Use real synthetic data
        )
        
        assert result.operation == "training"
        assert result.total_time > 0
        assert result.metadata["use_dummy_data"] is False