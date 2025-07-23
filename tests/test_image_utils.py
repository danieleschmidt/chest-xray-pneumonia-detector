"""Tests for centralized image loading utilities."""
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
import pytest

from src.image_utils import (
    load_single_image, 
    create_image_data_generator,
    create_inference_data_generator
)


class TestLoadSingleImage:
    """Test suite for load_single_image function."""
    
    def test_load_single_image_basic(self):
        """Test basic image loading functionality."""
        with patch('src.image_utils.image.load_img') as mock_load_img, \
             patch('src.image_utils.image.img_to_array') as mock_img_to_array:
            
            # Mock PIL image
            mock_img = MagicMock()
            mock_load_img.return_value = mock_img
            
            # Mock numpy array conversion
            mock_array = np.random.rand(150, 150, 3) * 255
            mock_img_to_array.return_value = mock_array
            
            # Test function
            result = load_single_image('test_image.jpg', (150, 150))
            
            # Verify calls
            mock_load_img.assert_called_once_with('test_image.jpg', target_size=(150, 150))
            mock_img_to_array.assert_called_once_with(mock_img)
            
            # Verify result shape and normalization
            assert result.shape == (1, 150, 150, 3)
            assert result.min() >= 0.0
            assert result.max() <= 1.0
    
    def test_load_single_image_normalization(self):
        """Test that images are properly normalized to [0, 1] range."""
        with patch('src.image_utils.image.load_img'), \
             patch('src.image_utils.image.img_to_array') as mock_img_to_array:
            
            # Create test array with values in [0, 255] range
            test_array = np.array([[[255, 128, 0]]], dtype=np.float32)
            mock_img_to_array.return_value = test_array
            
            result = load_single_image('test.jpg', (1, 1))
            
            # Check normalization
            expected = np.array([[[[1.0, 0.5019608, 0.0]]]])
            np.testing.assert_array_almost_equal(result, expected, decimal=6)
    
    def test_load_single_image_no_normalization(self):
        """Test loading without normalization."""
        with patch('src.image_utils.image.load_img'), \
             patch('src.image_utils.image.img_to_array') as mock_img_to_array:
            
            test_array = np.array([[[255, 128, 0]]], dtype=np.float32)
            mock_img_to_array.return_value = test_array
            
            result = load_single_image('test.jpg', (1, 1), normalize=False)
            
            # Should not be normalized
            expected = np.array([[[[255, 128, 0]]]])
            np.testing.assert_array_equal(result, expected)
    
    def test_load_single_image_file_not_found(self):
        """Test handling of non-existent files."""
        with patch('src.image_utils.image.load_img', side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                load_single_image('nonexistent.jpg', (150, 150))
    
    def test_load_single_image_invalid_format(self):
        """Test handling of invalid image formats."""
        with patch('src.image_utils.image.load_img', 
                   side_effect=Exception("cannot identify image file")):
            with pytest.raises(Exception):
                load_single_image('invalid.txt', (150, 150))


class TestCreateImageDataGenerator:
    """Test suite for create_image_data_generator function."""
    
    def test_create_training_generator_basic(self):
        """Test basic training data generator creation."""
        with patch('src.image_utils.ImageDataGenerator') as mock_datagen_class:
            mock_datagen = MagicMock()
            mock_generator = MagicMock()
            mock_datagen.flow_from_directory.return_value = mock_generator
            mock_datagen_class.return_value = mock_datagen
            
            result = create_image_data_generator(
                directory='train_dir',
                target_size=(150, 150),
                batch_size=32,
                class_mode='binary',
                augment=True
            )
            
            # Verify ImageDataGenerator created with augmentation
            mock_datagen_class.assert_called_once()
            call_args = mock_datagen_class.call_args[1]
            assert call_args['rescale'] == 1.0 / 255
            assert 'rotation_range' in call_args or 'horizontal_flip' in call_args
            
            # Verify flow_from_directory called
            mock_datagen.flow_from_directory.assert_called_once_with(
                'train_dir',
                target_size=(150, 150),
                batch_size=32,
                class_mode='binary',
                shuffle=True
            )
            
            assert result == mock_generator
    
    def test_create_validation_generator_no_augmentation(self):
        """Test validation generator creation without augmentation."""
        with patch('src.image_utils.ImageDataGenerator') as mock_datagen_class:
            mock_datagen = MagicMock()
            mock_generator = MagicMock()
            mock_datagen.flow_from_directory.return_value = mock_generator
            mock_datagen_class.return_value = mock_datagen
            
            result = create_image_data_generator(
                directory='val_dir',
                target_size=(150, 150),
                batch_size=32,
                class_mode='binary',
                augment=False
            )
            
            # Verify ImageDataGenerator created without augmentation
            mock_datagen_class.assert_called_once()
            call_args = mock_datagen_class.call_args[1]
            assert call_args == {'rescale': 1.0 / 255}
            
            # Verify flow_from_directory called with shuffle=False
            mock_datagen.flow_from_directory.assert_called_once_with(
                'val_dir',
                target_size=(150, 150),
                batch_size=32,
                class_mode='binary',
                shuffle=False
            )


class TestCreateInferenceDataGenerator:
    """Test suite for create_inference_data_generator function."""
    
    def test_create_inference_generator(self):
        """Test inference generator creation."""
        with patch('src.image_utils.ImageDataGenerator') as mock_datagen_class:
            mock_datagen = MagicMock()
            mock_generator = MagicMock()
            mock_generator.filepaths = ['image1.jpg', 'image2.jpg']
            mock_datagen.flow_from_directory.return_value = mock_generator
            mock_datagen_class.return_value = mock_datagen
            
            result = create_inference_data_generator(
                directory='test_dir',
                target_size=(224, 224),
                batch_size=16
            )
            
            # Verify ImageDataGenerator created with only rescaling
            mock_datagen_class.assert_called_once_with(rescale=1.0 / 255)
            
            # Verify flow_from_directory called with proper parameters
            mock_datagen.flow_from_directory.assert_called_once_with(
                'test_dir',
                target_size=(224, 224),
                class_mode=None,
                shuffle=False,
                batch_size=16
            )
            
            assert result == mock_generator
    
    def test_inference_generator_default_parameters(self):
        """Test inference generator with default parameters."""
        with patch('src.image_utils.ImageDataGenerator') as mock_datagen_class:
            mock_datagen = MagicMock()
            mock_generator = MagicMock()
            mock_datagen.flow_from_directory.return_value = mock_generator
            mock_datagen_class.return_value = mock_datagen
            
            create_inference_data_generator('test_dir')
            
            # Check default parameters used
            mock_datagen.flow_from_directory.assert_called_once_with(
                'test_dir',
                target_size=(150, 150),
                class_mode=None,
                shuffle=False,
                batch_size=32
            )