"""Tests for grad_cam module."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

# Skip tests if TensorFlow is unavailable
tf = pytest.importorskip("tensorflow")

from src.grad_cam import generate_grad_cam  # noqa: E402


class TestGenerateGradCam:
    """Test suite for generate_grad_cam function."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock TensorFlow model for testing."""
        model = Mock(spec=tf.keras.Model)
        
        # Mock the conv layer
        conv_layer = Mock()
        conv_layer.output = Mock()
        model.get_layer.return_value = conv_layer
        
        # Mock model inputs
        model.inputs = [Mock()]
        model.output = Mock()
        
        return model

    @pytest.fixture
    def sample_image(self):
        """Create a sample input image array."""
        return np.random.rand(1, 224, 224, 3).astype(np.float32)

    @pytest.fixture
    def mock_grad_model(self):
        """Create a mock gradient model."""
        grad_model = Mock()
        
        # Mock conv outputs (batch_size, height, width, channels)
        conv_outputs = tf.constant(np.random.rand(1, 14, 14, 512), dtype=tf.float32)
        
        # Mock predictions for binary classification
        predictions = tf.constant([[0.8]], dtype=tf.float32)
        
        grad_model.return_value = (conv_outputs, predictions)
        return grad_model, conv_outputs, predictions

    def test_generate_grad_cam_binary_classification(self, mock_model, sample_image):
        """Test Grad-CAM generation for binary classification."""
        with patch('tensorflow.keras.models.Model') as mock_keras_model, \
             patch('tensorflow.GradientTape') as mock_tape_class:
            
            # Setup mock gradient model
            grad_model = Mock()
            conv_outputs = tf.constant(np.random.rand(1, 14, 14, 512), dtype=tf.float32)
            predictions = tf.constant([[0.8]], dtype=tf.float32)  # Binary classification
            grad_model.return_value = (conv_outputs, predictions)
            mock_keras_model.return_value = grad_model
            
            # Setup mock gradient tape
            mock_tape = Mock()
            mock_tape_class.return_value.__enter__.return_value = mock_tape
            
            # Mock gradients
            mock_gradients = tf.constant(np.random.rand(1, 14, 14, 512), dtype=tf.float32)
            mock_tape.gradient.return_value = mock_gradients
            
            # Call the function
            result = generate_grad_cam(
                model=mock_model,
                image_array=sample_image,
                last_conv_layer_name="conv_layer",
                class_index=None
            )
            
            # Verify the result
            assert isinstance(result, np.ndarray)
            assert result.ndim == 2  # Should be 2D heatmap
            assert np.all(result >= 0) and np.all(result <= 1)  # Normalized between 0 and 1

    def test_generate_grad_cam_multiclass_classification(self, mock_model, sample_image):
        """Test Grad-CAM generation for multi-class classification."""
        with patch('tensorflow.keras.models.Model') as mock_keras_model, \
             patch('tensorflow.GradientTape') as mock_tape_class:
            
            # Setup mock gradient model
            grad_model = Mock()
            conv_outputs = tf.constant(np.random.rand(1, 14, 14, 512), dtype=tf.float32)
            predictions = tf.constant([[0.2, 0.8, 0.1]], dtype=tf.float32)  # Multi-class
            grad_model.return_value = (conv_outputs, predictions)
            mock_keras_model.return_value = grad_model
            
            # Setup mock gradient tape
            mock_tape = Mock()
            mock_tape_class.return_value.__enter__.return_value = mock_tape
            
            # Mock gradients
            mock_gradients = tf.constant(np.random.rand(1, 14, 14, 512), dtype=tf.float32)
            mock_tape.gradient.return_value = mock_gradients
            
            # Call the function
            result = generate_grad_cam(
                model=mock_model,
                image_array=sample_image,
                last_conv_layer_name="conv_layer",
                class_index=None
            )
            
            # Verify the result
            assert isinstance(result, np.ndarray)
            assert result.ndim == 2
            assert np.all(result >= 0) and np.all(result <= 1)

    def test_generate_grad_cam_invalid_layer_name(self, mock_model, sample_image):
        """Test Grad-CAM generation with invalid layer name."""
        # Setup model to raise ValueError when invalid layer requested
        mock_model.get_layer.side_effect = ValueError("No such layer: invalid_layer")
        
        with pytest.raises(ValueError) as exc_info:
            generate_grad_cam(
                model=mock_model,
                image_array=sample_image,
                last_conv_layer_name="invalid_layer",
                class_index=None
            )
        
        assert "Layer 'invalid_layer' not found in model" in str(exc_info.value)
        assert "Available layers:" in str(exc_info.value)

    def test_generate_grad_cam_layer_validation_lists_available_layers(self, sample_image):
        """Test that layer validation provides helpful layer names."""
        # Create a real model with known layer names for testing
        mock_model = Mock()
        mock_model.get_layer.side_effect = ValueError("No such layer: nonexistent")
        mock_model.layers = [
            Mock(name="input_layer"), 
            Mock(name="conv2d_1"), 
            Mock(name="conv2d_2"),
            Mock(name="output_layer")
        ]
        
        with pytest.raises(ValueError) as exc_info:
            generate_grad_cam(
                model=mock_model,
                image_array=sample_image,
                last_conv_layer_name="nonexistent",
                class_index=None
            )
        
        error_message = str(exc_info.value)
        assert "Layer 'nonexistent' not found in model" in error_message
        assert "Available layers:" in error_message
        assert "conv2d_1" in error_message
        assert "conv2d_2" in error_message

    def test_generate_grad_cam_explicit_class_index(self, mock_model, sample_image):
        """Test Grad-CAM generation with explicit class index."""
        with patch('tensorflow.keras.models.Model') as mock_keras_model, \
             patch('tensorflow.GradientTape') as mock_tape_class:
            
            # Setup mock gradient model
            grad_model = Mock()
            conv_outputs = tf.constant(np.random.rand(1, 14, 14, 512), dtype=tf.float32)
            predictions = tf.constant([[0.2, 0.8, 0.1]], dtype=tf.float32)
            grad_model.return_value = (conv_outputs, predictions)
            mock_keras_model.return_value = grad_model
            
            # Setup mock gradient tape
            mock_tape = Mock()
            mock_tape_class.return_value.__enter__.return_value = mock_tape
            
            # Mock gradients
            mock_gradients = tf.constant(np.random.rand(1, 14, 14, 512), dtype=tf.float32)
            mock_tape.gradient.return_value = mock_gradients
            
            # Call the function with explicit class index
            result = generate_grad_cam(
                model=mock_model,
                image_array=sample_image,
                last_conv_layer_name="conv_layer",
                class_index=1  # Explicit class index
            )
            
            # Verify the result
            assert isinstance(result, np.ndarray)
            assert result.ndim == 2
            assert np.all(result >= 0) and np.all(result <= 1)

    def test_generate_grad_cam_zero_gradients(self, mock_model, sample_image):
        """Test Grad-CAM generation when gradients are zero."""
        with patch('tensorflow.keras.models.Model') as mock_keras_model, \
             patch('tensorflow.GradientTape') as mock_tape_class:
            
            # Setup mock gradient model
            grad_model = Mock()
            conv_outputs = tf.constant(np.zeros((1, 14, 14, 512)), dtype=tf.float32)
            predictions = tf.constant([[0.5]], dtype=tf.float32)
            grad_model.return_value = (conv_outputs, predictions)
            mock_keras_model.return_value = grad_model
            
            # Setup mock gradient tape
            mock_tape = Mock()
            mock_tape_class.return_value.__enter__.return_value = mock_tape
            
            # Mock zero gradients
            mock_gradients = tf.constant(np.zeros((1, 14, 14, 512)), dtype=tf.float32)
            mock_tape.gradient.return_value = mock_gradients
            
            # Call the function
            result = generate_grad_cam(
                model=mock_model,
                image_array=sample_image,
                last_conv_layer_name="conv_layer",
                class_index=None
            )
            
            # Verify the result
            assert isinstance(result, np.ndarray)
            assert result.ndim == 2
            assert np.allclose(result, 0.0)  # Should be all zeros

    def test_generate_grad_cam_input_shapes(self, mock_model):
        """Test Grad-CAM generation with different input shapes."""
        test_cases = [
            (1, 224, 224, 3),  # Standard input
            (1, 150, 150, 3),  # Different resolution
            (1, 512, 512, 1),  # Grayscale
        ]
        
        for shape in test_cases:
            image_array = np.random.rand(*shape).astype(np.float32)
            
            with patch('tensorflow.keras.models.Model') as mock_keras_model, \
                 patch('tensorflow.GradientTape') as mock_tape_class:
                
                # Setup mocks
                grad_model = Mock()
                conv_height, conv_width = shape[1] // 16, shape[2] // 16  # Assume 16x downsampling
                conv_outputs = tf.constant(
                    np.random.rand(1, conv_height, conv_width, 512), 
                    dtype=tf.float32
                )
                predictions = tf.constant([[0.7]], dtype=tf.float32)
                grad_model.return_value = (conv_outputs, predictions)
                mock_keras_model.return_value = grad_model
                
                mock_tape = Mock()
                mock_tape_class.return_value.__enter__.return_value = mock_tape
                mock_gradients = tf.constant(
                    np.random.rand(1, conv_height, conv_width, 512), 
                    dtype=tf.float32
                )
                mock_tape.gradient.return_value = mock_gradients
                
                # Call the function
                result = generate_grad_cam(
                    model=mock_model,
                    image_array=image_array,
                    last_conv_layer_name="conv_layer",
                    class_index=None
                )
                
                # Verify the result
                assert isinstance(result, np.ndarray)
                assert result.ndim == 2
                assert result.shape == (conv_height, conv_width)

    def test_generate_grad_cam_normalization(self, mock_model, sample_image):
        """Test that Grad-CAM output is properly normalized."""
        with patch('tensorflow.keras.models.Model') as mock_keras_model, \
             patch('tensorflow.GradientTape') as mock_tape_class:
            
            # Setup mock gradient model with large values
            grad_model = Mock()
            conv_outputs = tf.constant(np.random.rand(1, 14, 14, 512) * 100, dtype=tf.float32)
            predictions = tf.constant([[0.9]], dtype=tf.float32)
            grad_model.return_value = (conv_outputs, predictions)
            mock_keras_model.return_value = grad_model
            
            # Setup mock gradient tape
            mock_tape = Mock()
            mock_tape_class.return_value.__enter__.return_value = mock_tape
            
            # Mock gradients with large values
            mock_gradients = tf.constant(np.random.rand(1, 14, 14, 512) * 50, dtype=tf.float32)
            mock_tape.gradient.return_value = mock_gradients
            
            # Call the function
            result = generate_grad_cam(
                model=mock_model,
                image_array=sample_image,
                last_conv_layer_name="conv_layer",
                class_index=None
            )
            
            # Verify normalization
            assert isinstance(result, np.ndarray)
            assert np.all(result >= 0) and np.all(result <= 1)
            assert np.max(result) <= 1.0  # Maximum should be 1 or less
            
    def test_generate_grad_cam_model_interaction(self, mock_model, sample_image):
        """Test that the function properly interacts with the model."""
        with patch('tensorflow.keras.models.Model') as mock_keras_model, \
             patch('tensorflow.GradientTape') as mock_tape_class:
            
            # Setup mocks
            grad_model = Mock()
            conv_outputs = tf.constant(np.random.rand(1, 14, 14, 512), dtype=tf.float32)
            predictions = tf.constant([[0.6]], dtype=tf.float32)
            grad_model.return_value = (conv_outputs, predictions)
            mock_keras_model.return_value = grad_model
            
            mock_tape = Mock()
            mock_tape_class.return_value.__enter__.return_value = mock_tape
            mock_gradients = tf.constant(np.random.rand(1, 14, 14, 512), dtype=tf.float32)
            mock_tape.gradient.return_value = mock_gradients
            
            layer_name = "test_conv_layer"
            
            # Call the function
            generate_grad_cam(
                model=mock_model,
                image_array=sample_image,
                last_conv_layer_name=layer_name,
                class_index=None
            )
            
            # Verify model interactions
            mock_model.get_layer.assert_called_once_with(layer_name)
            mock_keras_model.assert_called_once()
            grad_model.assert_called_once()
            mock_tape.gradient.assert_called_once()


def test_main_execution():
    """Test the main execution block."""
    import sys
    from io import StringIO
    from unittest.mock import patch
    
    # Capture stdout
    captured_output = StringIO()
    with patch.object(sys, 'stdout', captured_output):
        # Import the module to trigger the main block
        import importlib
        import src.grad_cam
        importlib.reload(src.grad_cam)
    
    # Verify output
    output = captured_output.getvalue()
    assert "grad_cam.py executed" in output
    assert "generate_grad_cam" in output