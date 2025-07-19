import sys
import subprocess
import pytest
import tempfile
import os
import numpy as np
from unittest.mock import patch, Mock, MagicMock
from PIL import Image

pytest.importorskip("tensorflow")

from src.predict_utils import load_image, display_grad_cam  # noqa: E402


class TestLoadImage:
    """Test suite for load_image function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_image_file(self, temp_dir):
        """Create a sample image file for testing."""
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        img_path = os.path.join(temp_dir, 'test_image.jpg')
        img.save(img_path)
        return img_path

    def test_load_image_basic_functionality(self, sample_image_file):
        """Test basic image loading functionality."""
        target_size = (150, 150)
        
        with patch('tensorflow.keras.preprocessing.image.load_img') as mock_load_img, \
             patch('tensorflow.keras.preprocessing.image.img_to_array') as mock_img_to_array:
            
            # Setup mocks
            mock_img = Mock()
            mock_load_img.return_value = mock_img
            mock_img_to_array.return_value = np.random.rand(150, 150, 3) * 255
            
            # Call the function
            result = load_image(sample_image_file, target_size)
            
            # Verify the result
            assert isinstance(result, np.ndarray)
            assert result.shape == (1, 150, 150, 3)  # Batch dimension added
            assert np.all(result >= 0) and np.all(result <= 1)  # Normalized to [0,1]
            
            # Verify mock calls
            mock_load_img.assert_called_once_with(sample_image_file, target_size=target_size)
            mock_img_to_array.assert_called_once_with(mock_img)

    def test_load_image_different_sizes(self, sample_image_file):
        """Test image loading with different target sizes."""
        test_sizes = [(224, 224), (512, 512), (64, 64)]
        
        for target_size in test_sizes:
            with patch('tensorflow.keras.preprocessing.image.load_img') as mock_load_img, \
                 patch('tensorflow.keras.preprocessing.image.img_to_array') as mock_img_to_array:
                
                mock_img = Mock()
                mock_load_img.return_value = mock_img
                mock_img_to_array.return_value = np.random.rand(*target_size, 3) * 255
                
                result = load_image(sample_image_file, target_size)
                
                assert result.shape == (1, *target_size, 3)
                mock_load_img.assert_called_once_with(sample_image_file, target_size=target_size)

    def test_load_image_normalization(self, sample_image_file):
        """Test that image values are properly normalized."""
        target_size = (100, 100)
        
        with patch('tensorflow.keras.preprocessing.image.load_img'), \
             patch('tensorflow.keras.preprocessing.image.img_to_array') as mock_img_to_array:
            
            # Mock image with known values
            test_array = np.full((100, 100, 3), 255, dtype=np.uint8)  # Max values
            mock_img_to_array.return_value = test_array
            
            result = load_image(sample_image_file, target_size)
            
            # After normalization, max should be 1.0
            assert np.allclose(result.max(), 1.0)
            assert np.allclose(result.min(), 1.0)  # All pixels were 255

    def test_load_image_file_not_found(self):
        """Test handling of non-existent image files."""
        non_existent_path = "/path/to/nonexistent/image.jpg"
        target_size = (150, 150)
        
        with patch('tensorflow.keras.preprocessing.image.load_img') as mock_load_img:
            mock_load_img.side_effect = FileNotFoundError("Image file not found")
            
            with pytest.raises(FileNotFoundError, match="Image file not found"):
                load_image(non_existent_path, target_size)

    def test_load_image_batch_dimension(self, sample_image_file):
        """Test that batch dimension is correctly added."""
        target_size = (150, 150)
        
        with patch('tensorflow.keras.preprocessing.image.load_img'), \
             patch('tensorflow.keras.preprocessing.image.img_to_array') as mock_img_to_array:
            
            # Mock 3D array (H, W, C)
            original_shape = (150, 150, 3)
            mock_img_to_array.return_value = np.random.rand(*original_shape) * 255
            
            result = load_image(sample_image_file, target_size)
            
            # Should add batch dimension at axis 0
            expected_shape = (1, *original_shape)
            assert result.shape == expected_shape


class TestDisplayGradCam:
    """Test suite for display_grad_cam function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_model(self):
        """Create a mock TensorFlow model."""
        model = Mock()
        return model

    @pytest.fixture
    def sample_image_file(self, temp_dir):
        """Create a sample image file for testing."""
        img = Image.new('RGB', (150, 150), color='blue')
        img_path = os.path.join(temp_dir, 'test_image.jpg')
        img.save(img_path)
        return img_path

    def test_display_grad_cam_basic_functionality(self, temp_dir, sample_image_file, mock_model):
        """Test basic Grad-CAM generation functionality."""
        model_path = os.path.join(temp_dir, 'model.keras')
        output_path = os.path.join(temp_dir, 'gradcam_output.png')
        
        # Mock heatmap from grad_cam
        mock_heatmap = np.random.rand(14, 14)
        
        with patch('tensorflow.keras.models.load_model') as mock_load_model, \
             patch('src.predict_utils.load_image') as mock_load_image, \
             patch('src.predict_utils.generate_grad_cam') as mock_generate_grad_cam, \
             patch('tensorflow.image.resize') as mock_tf_resize, \
             patch('tensorflow.keras.preprocessing.image.load_img') as mock_load_img, \
             patch('tensorflow.keras.preprocessing.image.img_to_array') as mock_img_to_array, \
             patch('matplotlib.pyplot.imshow') as mock_imshow, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.close') as mock_close:
            
            # Setup mocks
            mock_load_model.return_value = mock_model
            mock_load_image.return_value = np.random.rand(1, 150, 150, 3)
            mock_generate_grad_cam.return_value = mock_heatmap
            
            # Mock tensor resize
            resized_heatmap = np.random.rand(150, 150, 1) * 255
            mock_tensor = Mock()
            mock_tensor.numpy.return_value = resized_heatmap.astype(np.uint8)
            mock_tf_resize.return_value = mock_tensor
            
            # Mock original image loading
            mock_img = Mock()
            mock_load_img.return_value = mock_img
            mock_img_to_array.return_value = np.random.rand(150, 150, 3) * 255
            
            # Call the function
            display_grad_cam(
                model_path=model_path,
                img_path=sample_image_file,
                target_size=(150, 150),
                last_conv_layer_name="test_layer",
                output_path=output_path
            )
            
            # Verify function calls
            mock_load_model.assert_called_once_with(model_path)
            mock_load_image.assert_called_once_with(sample_image_file, (150, 150))
            mock_generate_grad_cam.assert_called_once_with(mock_model, mock_load_image.return_value, "test_layer")
            mock_savefig.assert_called_once_with(output_path, bbox_inches="tight")
            mock_close.assert_called_once()

    def test_display_grad_cam_custom_parameters(self, temp_dir, sample_image_file, mock_model):
        """Test Grad-CAM generation with custom parameters."""
        model_path = os.path.join(temp_dir, 'custom_model.keras')
        output_path = os.path.join(temp_dir, 'custom_output.png')
        custom_size = (224, 224)
        custom_layer = "custom_conv_layer"
        
        mock_heatmap = np.random.rand(7, 7)  # Smaller heatmap for 224x224
        
        with patch('tensorflow.keras.models.load_model') as mock_load_model, \
             patch('src.predict_utils.load_image') as mock_load_image, \
             patch('src.predict_utils.generate_grad_cam') as mock_generate_grad_cam, \
             patch('tensorflow.image.resize') as mock_tf_resize, \
             patch('tensorflow.keras.preprocessing.image.load_img'), \
             patch('tensorflow.keras.preprocessing.image.img_to_array'), \
             patch('matplotlib.pyplot.imshow'), \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.close'):
            
            # Setup mocks
            mock_load_model.return_value = mock_model
            mock_load_image.return_value = np.random.rand(1, *custom_size, 3)
            mock_generate_grad_cam.return_value = mock_heatmap
            
            resized_heatmap = np.random.rand(*custom_size, 1) * 255
            mock_tensor = Mock()
            mock_tensor.numpy.return_value = resized_heatmap.astype(np.uint8)
            mock_tf_resize.return_value = mock_tensor
            
            # Call the function with custom parameters
            display_grad_cam(
                model_path=model_path,
                img_path=sample_image_file,
                target_size=custom_size,
                last_conv_layer_name=custom_layer,
                output_path=output_path
            )
            
            # Verify custom parameters were used
            mock_load_image.assert_called_once_with(sample_image_file, custom_size)
            mock_generate_grad_cam.assert_called_once_with(mock_model, mock_load_image.return_value, custom_layer)
            mock_tf_resize.assert_called_once()
            mock_savefig.assert_called_once_with(output_path, bbox_inches="tight")

    def test_display_grad_cam_heatmap_processing(self, temp_dir, sample_image_file, mock_model):
        """Test heatmap processing and normalization."""
        model_path = os.path.join(temp_dir, 'model.keras')
        output_path = os.path.join(temp_dir, 'gradcam_output.png')
        
        # Create a specific heatmap for testing
        mock_heatmap = np.array([[0.0, 0.5], [0.8, 1.0]])  # 2x2 heatmap with known values
        
        with patch('tensorflow.keras.models.load_model') as mock_load_model, \
             patch('src.predict_utils.load_image') as mock_load_image, \
             patch('src.predict_utils.generate_grad_cam') as mock_generate_grad_cam, \
             patch('tensorflow.image.resize') as mock_tf_resize, \
             patch('tensorflow.keras.preprocessing.image.load_img'), \
             patch('tensorflow.keras.preprocessing.image.img_to_array'), \
             patch('matplotlib.pyplot.imshow'), \
             patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'):
            
            # Setup mocks
            mock_load_model.return_value = mock_model
            mock_load_image.return_value = np.random.rand(1, 150, 150, 3)
            mock_generate_grad_cam.return_value = mock_heatmap
            
            # Mock the tensor resize to return processed heatmap
            expected_uint8 = (mock_heatmap * 255).astype(np.uint8)
            resized_heatmap = np.expand_dims(expected_uint8, axis=2)  # Add channel dimension
            mock_tensor = Mock()
            mock_tensor.numpy.return_value = resized_heatmap
            mock_tf_resize.return_value = mock_tensor
            
            # Call the function
            display_grad_cam(
                model_path=model_path,
                img_path=sample_image_file,
                output_path=output_path
            )
            
            # Verify tensor resize was called with correct parameters
            mock_tf_resize.assert_called_once()
            args, kwargs = mock_tf_resize.call_args
            assert args[1] == (150, 150)  # target_size

    def test_display_grad_cam_model_loading_error(self, temp_dir, sample_image_file):
        """Test handling of model loading errors."""
        model_path = os.path.join(temp_dir, 'nonexistent_model.keras')
        output_path = os.path.join(temp_dir, 'output.png')
        
        with patch('tensorflow.keras.models.load_model') as mock_load_model:
            mock_load_model.side_effect = OSError("Model file not found")
            
            with pytest.raises(OSError, match="Model file not found"):
                display_grad_cam(
                    model_path=model_path,
                    img_path=sample_image_file,
                    output_path=output_path
                )

    def test_display_grad_cam_image_loading_error(self, temp_dir, mock_model):
        """Test handling of image loading errors."""
        model_path = os.path.join(temp_dir, 'model.keras')
        nonexistent_image = os.path.join(temp_dir, 'nonexistent.jpg')
        output_path = os.path.join(temp_dir, 'output.png')
        
        with patch('tensorflow.keras.models.load_model') as mock_load_model, \
             patch('src.predict_utils.load_image') as mock_load_image:
            
            mock_load_model.return_value = mock_model
            mock_load_image.side_effect = FileNotFoundError("Image file not found")
            
            with pytest.raises(FileNotFoundError, match="Image file not found"):
                display_grad_cam(
                    model_path=model_path,
                    img_path=nonexistent_image,
                    output_path=output_path
                )

    def test_display_grad_cam_output_file_creation(self, temp_dir, sample_image_file, mock_model):
        """Test that output file is properly created."""
        model_path = os.path.join(temp_dir, 'model.keras')
        output_path = os.path.join(temp_dir, 'test_output.png')
        
        mock_heatmap = np.random.rand(14, 14)
        
        with patch('tensorflow.keras.models.load_model') as mock_load_model, \
             patch('src.predict_utils.load_image') as mock_load_image, \
             patch('src.predict_utils.generate_grad_cam') as mock_generate_grad_cam, \
             patch('tensorflow.image.resize') as mock_tf_resize, \
             patch('tensorflow.keras.preprocessing.image.load_img'), \
             patch('tensorflow.keras.preprocessing.image.img_to_array'), \
             patch('matplotlib.pyplot.imshow'), \
             patch('matplotlib.pyplot.axis'), \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.close'):
            
            # Setup mocks
            mock_load_model.return_value = mock_model
            mock_load_image.return_value = np.random.rand(1, 150, 150, 3)
            mock_generate_grad_cam.return_value = mock_heatmap
            
            resized_heatmap = np.random.rand(150, 150, 1) * 255
            mock_tensor = Mock()
            mock_tensor.numpy.return_value = resized_heatmap.astype(np.uint8)
            mock_tf_resize.return_value = mock_tensor
            
            # Call the function
            display_grad_cam(
                model_path=model_path,
                img_path=sample_image_file,
                output_path=output_path
            )
            
            # Verify matplotlib savefig was called with correct path
            mock_savefig.assert_called_once_with(output_path, bbox_inches="tight")


def test_cli_help():
    """Test CLI help output."""
    result = subprocess.run(
        [sys.executable, "-m", "src.predict_utils", "--help"], capture_output=True
    )
    assert result.returncode == 0
    assert b"--model_path" in result.stdout
    assert b"--img_path" in result.stdout
    assert b"--last_conv_layer_name" in result.stdout
    assert b"--output_path" in result.stdout
    assert b"--img_size" in result.stdout


def test_main_function_integration():
    """Test the main function with mocked arguments."""
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, 'test_model.keras')
        img_path = os.path.join(temp_dir, 'test_image.jpg')
        output_path = os.path.join(temp_dir, 'test_gradcam.png')
        
        # Create a dummy image file
        img = Image.new('RGB', (150, 150), color='green')
        img.save(img_path)
        
        # Mock command line arguments
        test_args = [
            'src.predict_utils',
            '--model_path', model_path,
            '--img_path', img_path,
            '--last_conv_layer_name', 'test_layer',
            '--output_path', output_path,
            '--img_size', '224', '224'
        ]
        
        with patch('sys.argv', test_args), \
             patch('src.predict_utils.display_grad_cam') as mock_display_grad_cam:
            
            # Import and run main
            import importlib
            import src.predict_utils
            importlib.reload(src.predict_utils)
            
            # Verify display_grad_cam was called with correct arguments
            mock_display_grad_cam.assert_called_once_with(
                model_path=model_path,
                img_path=img_path,
                target_size=(224, 224),
                last_conv_layer_name='test_layer',
                output_path=output_path
            )
