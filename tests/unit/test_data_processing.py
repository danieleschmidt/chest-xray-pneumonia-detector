"""
Unit tests for data processing modules.
"""

import numpy as np
import pytest
from pathlib import Path
from PIL import Image
from unittest.mock import Mock, patch, MagicMock


@pytest.mark.unit
class TestDataLoader:
    """Unit tests for data_loader module."""

    def test_load_and_preprocess_image(self, sample_image_path: Path):
        """Test loading and preprocessing a single image."""
        from src.data_loader import load_and_preprocess_image
        
        # Test loading with default parameters
        processed_image = load_and_preprocess_image(
            str(sample_image_path),
            target_size=(150, 150)
        )
        
        assert processed_image is not None
        assert processed_image.shape == (150, 150, 3)
        assert processed_image.dtype == np.float32
        assert 0.0 <= processed_image.min() <= processed_image.max() <= 1.0

    def test_load_image_with_invalid_path(self):
        """Test loading image with invalid path."""
        from src.data_loader import load_and_preprocess_image
        
        with pytest.raises((FileNotFoundError, ValueError)):
            load_and_preprocess_image("nonexistent_image.jpg")

    def test_data_generator_creation(self, sample_dataset_structure):
        """Test creation of data generators."""
        from src.data_loader import create_data_generators
        
        train_dir, val_dir, _ = sample_dataset_structure
        
        train_gen, val_gen = create_data_generators(
            train_dir=str(train_dir),
            val_dir=str(val_dir),
            batch_size=2,
            img_size=(150, 150),
            augment_training=True
        )
        
        assert train_gen is not None
        assert val_gen is not None
        
        # Test that generators produce batches
        train_batch = next(train_gen)
        assert len(train_batch) == 2  # X, y
        assert train_batch[0].shape[0] <= 2  # Batch size
        assert train_batch[0].shape[1:] == (150, 150, 3)  # Image dimensions

    def test_image_augmentation_parameters(self):
        """Test image augmentation parameter validation."""
        from src.data_loader import validate_augmentation_params
        
        # Valid parameters
        valid_params = {
            "rotation_range": 20,
            "brightness_range": (0.8, 1.2),
            "zoom_range": 0.1,
            "horizontal_flip": True
        }
        
        assert validate_augmentation_params(valid_params) is True
        
        # Invalid parameters
        invalid_params = {
            "rotation_range": -10,  # Should be positive
            "brightness_range": (1.5, 0.5),  # Invalid range
        }
        
        assert validate_augmentation_params(invalid_params) is False

    def test_class_weight_calculation(self, sample_dataset_structure):
        """Test automatic class weight calculation."""
        from src.data_loader import calculate_class_weights
        
        train_dir, _, _ = sample_dataset_structure
        
        class_weights = calculate_class_weights(str(train_dir))
        
        assert isinstance(class_weights, dict)
        assert 0 in class_weights  # NORMAL class
        assert 1 in class_weights  # PNEUMONIA class
        assert all(weight > 0 for weight in class_weights.values())


@pytest.mark.unit
class TestDataSplit:
    """Unit tests for data_split module."""

    def test_split_fractions_validation(self):
        """Test validation of split fractions."""
        from src.data_split import validate_split_fractions
        
        # Valid fractions
        assert validate_split_fractions(0.1, 0.1) is True
        assert validate_split_fractions(0.2, 0.3) is True
        
        # Invalid fractions
        assert validate_split_fractions(0.6, 0.6) is False  # Sum > 1
        assert validate_split_fractions(-0.1, 0.1) is False  # Negative
        assert validate_split_fractions(0.1, 1.1) is False  # > 1

    def test_file_distribution_calculation(self):
        """Test calculation of file distribution across splits."""
        from src.data_split import calculate_split_sizes
        
        total_files = 100
        val_frac = 0.1
        test_frac = 0.2
        
        train_size, val_size, test_size = calculate_split_sizes(
            total_files, val_frac, test_frac
        )
        
        assert train_size + val_size + test_size == total_files
        assert val_size == 10
        assert test_size == 20
        assert train_size == 70

    def test_directory_creation(self, tmp_path: Path):
        """Test creation of output directory structure."""
        from src.data_split import create_output_structure
        
        output_dir = tmp_path / "test_output"
        classes = ["NORMAL", "PNEUMONIA"]
        
        create_output_structure(str(output_dir), classes)
        
        # Check directory structure
        for split in ["train", "val", "test"]:
            for class_name in classes:
                expected_dir = output_dir / split / class_name
                assert expected_dir.exists()
                assert expected_dir.is_dir()

    def test_file_copying_vs_moving(self, tmp_path: Path):
        """Test file copying vs moving functionality."""
        from src.data_split import copy_or_move_file
        
        # Create source file
        source_file = tmp_path / "source.txt"
        source_file.write_text("test content")
        
        dest_file = tmp_path / "dest.txt"
        
        # Test copying
        copy_or_move_file(str(source_file), str(dest_file), move=False)
        assert source_file.exists()  # Original should still exist
        assert dest_file.exists()
        
        # Clean up for move test
        dest_file.unlink()
        
        # Test moving
        copy_or_move_file(str(source_file), str(dest_file), move=True)
        assert not source_file.exists()  # Original should be gone
        assert dest_file.exists()


@pytest.mark.unit
class TestImageUtils:
    """Unit tests for image_utils module."""

    def test_image_format_validation(self, sample_image: np.ndarray):
        """Test image format validation."""
        from src.image_utils import validate_image_format
        
        # Valid image
        assert validate_image_format(sample_image) is True
        
        # Invalid shapes
        assert validate_image_format(np.array([1, 2, 3])) is False  # 1D
        assert validate_image_format(np.random.random((10, 10, 10, 10))) is False  # 4D

    def test_image_normalization(self, sample_image: np.ndarray):
        """Test image normalization functions."""
        from src.image_utils import normalize_image, denormalize_image
        
        # Test normalization
        normalized = normalize_image(sample_image, method="min_max")
        assert 0.0 <= normalized.min() <= normalized.max() <= 1.0
        
        # Test Z-score normalization
        z_normalized = normalize_image(sample_image, method="z_score")
        assert abs(z_normalized.mean()) < 0.1  # Should be close to 0
        assert abs(z_normalized.std() - 1.0) < 0.1  # Should be close to 1
        
        # Test denormalization
        denormalized = denormalize_image(normalized, method="min_max", 
                                       original_min=sample_image.min(),
                                       original_max=sample_image.max())
        np.testing.assert_allclose(denormalized, sample_image, rtol=1e-5)

    def test_image_resizing(self, sample_image: np.ndarray):
        """Test image resizing functionality."""
        from src.image_utils import resize_image
        
        # Test resizing to different sizes
        resized_150 = resize_image(sample_image, (150, 150))
        assert resized_150.shape[:2] == (150, 150)
        
        resized_224 = resize_image(sample_image, (224, 224))
        assert resized_224.shape[:2] == (224, 224)
        
        # Test preserving aspect ratio
        resized_preserve = resize_image(sample_image, (200, 300), preserve_aspect=True)
        assert max(resized_preserve.shape[:2]) <= 300
        assert min(resized_preserve.shape[:2]) >= 200

    def test_image_quality_assessment(self, sample_image: np.ndarray):
        """Test image quality assessment functions."""
        from src.image_utils import assess_image_quality
        
        quality_metrics = assess_image_quality(sample_image)
        
        assert "sharpness" in quality_metrics
        assert "contrast" in quality_metrics
        assert "brightness" in quality_metrics
        assert all(isinstance(v, (int, float)) for v in quality_metrics.values())

    def test_medical_image_preprocessing(self, sample_image: np.ndarray):
        """Test medical-specific image preprocessing."""
        from src.image_utils import preprocess_medical_image
        
        processed = preprocess_medical_image(
            sample_image,
            target_size=(224, 224),
            normalize=True,
            enhance_contrast=True
        )
        
        assert processed.shape == (224, 224, 1)
        assert processed.dtype == np.float32
        assert 0.0 <= processed.min() <= processed.max() <= 1.0

    def test_histogram_equalization(self, sample_image: np.ndarray):
        """Test histogram equalization for medical images."""
        from src.image_utils import apply_histogram_equalization
        
        equalized = apply_histogram_equalization(sample_image)
        
        assert equalized.shape == sample_image.shape
        assert equalized.dtype == sample_image.dtype
        
        # Equalized image should have better contrast
        original_std = np.std(sample_image)
        equalized_std = np.std(equalized)
        assert equalized_std >= original_std * 0.8  # Allow some tolerance


@pytest.mark.unit
class TestDataValidation:
    """Unit tests for data validation functionality."""

    def test_medical_image_validation(self, sample_image: np.ndarray):
        """Test medical image validation rules."""
        from src.image_utils import validate_medical_image
        
        # Valid image
        is_valid, message = validate_medical_image(sample_image)
        assert is_valid is True
        assert message == ""
        
        # Too small image
        small_image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        is_valid, message = validate_medical_image(small_image)
        assert is_valid is False
        assert "too small" in message.lower()
        
        # Wrong data type
        float_image = sample_image.astype(np.float64)
        is_valid, message = validate_medical_image(float_image)
        # Should either be valid or provide guidance on conversion
        assert isinstance(is_valid, bool)

    def test_file_extension_validation(self):
        """Test file extension validation for medical images."""
        from src.image_utils import validate_file_extension
        
        # Valid extensions
        assert validate_file_extension("image.jpg") is True
        assert validate_file_extension("image.jpeg") is True
        assert validate_file_extension("image.png") is True
        assert validate_file_extension("image.dcm") is True
        assert validate_file_extension("image.dicom") is True
        
        # Invalid extensions
        assert validate_file_extension("document.txt") is False
        assert validate_file_extension("video.mp4") is False
        assert validate_file_extension("audio.wav") is False

    def test_batch_validation(self, sample_dataset_structure):
        """Test batch validation of image datasets."""
        from src.image_utils import validate_dataset
        
        train_dir, _, _ = sample_dataset_structure
        
        validation_report = validate_dataset(str(train_dir))
        
        assert "total_images" in validation_report
        assert "valid_images" in validation_report
        assert "invalid_images" in validation_report
        assert "class_distribution" in validation_report
        
        assert validation_report["total_images"] > 0
        assert validation_report["valid_images"] >= 0
        assert validation_report["invalid_images"] >= 0


@pytest.mark.unit
class TestConfigurationHandling:
    """Unit tests for configuration handling."""

    def test_config_loading(self):
        """Test configuration loading and validation."""
        from src.config import load_config, validate_config
        
        # Test default config
        config = load_config()
        assert isinstance(config, dict)
        assert "model" in config or "training" in config or len(config) > 0
        
        # Test config validation
        valid_config = {
            "model": {
                "input_shape": [224, 224, 3],
                "num_classes": 2
            },
            "training": {
                "batch_size": 32,
                "epochs": 10
            }
        }
        
        assert validate_config(valid_config) is True

    def test_environment_variable_handling(self):
        """Test environment variable configuration."""
        import os
        from src.config import get_env_config
        
        # Set test environment variables
        test_vars = {
            "MODEL_PATH": "/test/model.keras",
            "BATCH_SIZE": "16",
            "DEBUG": "true"
        }
        
        original_env = {}
        for key, value in test_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            env_config = get_env_config()
            
            assert env_config.get("MODEL_PATH") == "/test/model.keras"
            assert env_config.get("BATCH_SIZE") == "16"
            assert env_config.get("DEBUG") == "true"
            
        finally:
            # Restore original environment
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value