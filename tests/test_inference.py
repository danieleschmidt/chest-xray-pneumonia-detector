import subprocess
import sys
import pytest
import tempfile
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock

pytest.importorskip("tensorflow")

from src.inference import predict_directory  # noqa: E402
from src.input_validation import ValidationError  # noqa: E402


class TestPredictDirectory:
    """Test suite for predict_directory function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_model_binary(self):
        """Create a mock binary classification model."""
        model = Mock()
        # Binary model outputs shape (batch_size, 1)
        model.predict.return_value = np.array([[0.8], [0.3], [0.9], [0.1]])
        return model

    @pytest.fixture
    def mock_model_multiclass(self):
        """Create a mock multi-class classification model."""
        model = Mock()
        # Multi-class model outputs shape (batch_size, num_classes)
        model.predict.return_value = np.array([
            [0.1, 0.8, 0.1],  # Class 1
            [0.7, 0.2, 0.1],  # Class 0
            [0.05, 0.05, 0.9],  # Class 2
            [0.6, 0.3, 0.1]   # Class 0
        ])
        return model

    @pytest.fixture
    def mock_image_generator(self):
        """Create a mock image data generator."""
        generator = Mock()
        generator.filepaths = [
            'class1/img1.jpg',
            'class2/img2.jpg', 
            'class1/img3.jpg',
            'class2/img4.jpg'
        ]
        return generator

    def test_binary_classification_prediction(self, temp_dir, mock_model_binary, mock_image_generator):
        """Test binary classification predictions."""
        model_path = os.path.join(temp_dir, 'model.keras')
        data_dir = os.path.join(temp_dir, 'data')
        
        with patch('tensorflow.keras.models.load_model') as mock_load_model, \
             patch('tensorflow.keras.preprocessing.image.ImageDataGenerator') as mock_datagen_class:
            
            # Setup mocks
            mock_load_model.return_value = mock_model_binary
            mock_datagen = Mock()
            mock_datagen.flow_from_directory.return_value = mock_image_generator
            mock_datagen_class.return_value = mock_datagen
            
            # Call the function
            result_df = predict_directory(
                model_path=model_path,
                data_dir=data_dir,
                img_size=(224, 224),
                num_classes=1
            )
            
            # Verify the result
            assert isinstance(result_df, pd.DataFrame)
            assert 'filepath' in result_df.columns
            assert 'prediction' in result_df.columns
            assert len(result_df) == 4
            
            # Verify model was loaded correctly
            mock_load_model.assert_called_once_with(model_path)
            
            # Verify data generator was set up correctly
            mock_datagen_class.assert_called_once_with(rescale=1.0/255)
            mock_datagen.flow_from_directory.assert_called_once_with(
                data_dir,
                target_size=(224, 224),
                class_mode=None,
                shuffle=False,
                batch_size=32
            )
            
            # Verify model prediction was called
            mock_model_binary.predict.assert_called_once_with(mock_image_generator)
            
            # Verify predictions are flattened for binary classification
            expected_preds = np.array([0.8, 0.3, 0.9, 0.1])
            np.testing.assert_array_equal(result_df['prediction'].values, expected_preds)

    def test_multiclass_classification_prediction(self, temp_dir, mock_model_multiclass, mock_image_generator):
        """Test multi-class classification predictions."""
        model_path = os.path.join(temp_dir, 'multiclass_model.keras')
        data_dir = os.path.join(temp_dir, 'data')
        
        with patch('tensorflow.keras.models.load_model') as mock_load_model, \
             patch('tensorflow.keras.preprocessing.image.ImageDataGenerator') as mock_datagen_class:
            
            # Setup mocks
            mock_load_model.return_value = mock_model_multiclass
            mock_datagen = Mock()
            mock_datagen.flow_from_directory.return_value = mock_image_generator
            mock_datagen_class.return_value = mock_datagen
            
            # Call the function
            result_df = predict_directory(
                model_path=model_path,
                data_dir=data_dir,
                img_size=(150, 150),
                num_classes=3
            )
            
            # Verify the result
            assert isinstance(result_df, pd.DataFrame)
            assert 'filepath' in result_df.columns
            assert 'prediction' in result_df.columns
            assert 'prob_0' in result_df.columns
            assert 'prob_1' in result_df.columns
            assert 'prob_2' in result_df.columns
            assert len(result_df) == 4
            
            # Verify predicted class indices
            expected_predictions = np.array([1, 0, 2, 0])  # argmax of each row
            np.testing.assert_array_equal(result_df['prediction'].values, expected_predictions)
            
            # Verify probabilities
            expected_probs = mock_model_multiclass.predict.return_value
            np.testing.assert_array_equal(result_df['prob_0'].values, expected_probs[:, 0])
            np.testing.assert_array_equal(result_df['prob_1'].values, expected_probs[:, 1])
            np.testing.assert_array_equal(result_df['prob_2'].values, expected_probs[:, 2])

    def test_custom_image_size(self, temp_dir, mock_model_binary, mock_image_generator):
        """Test prediction with custom image size."""
        model_path = os.path.join(temp_dir, 'model.keras')
        data_dir = os.path.join(temp_dir, 'data')
        custom_size = (512, 512)
        
        with patch('tensorflow.keras.models.load_model') as mock_load_model, \
             patch('tensorflow.keras.preprocessing.image.ImageDataGenerator') as mock_datagen_class:
            
            # Setup mocks
            mock_load_model.return_value = mock_model_binary
            mock_datagen = Mock()
            mock_datagen.flow_from_directory.return_value = mock_image_generator
            mock_datagen_class.return_value = mock_datagen
            
            # Call the function with custom image size
            predict_directory(
                model_path=model_path,
                data_dir=data_dir,
                img_size=custom_size,
                num_classes=1
            )
            
            # Verify custom image size was used
            mock_datagen.flow_from_directory.assert_called_once_with(
                data_dir,
                target_size=custom_size,
                class_mode=None,
                shuffle=False,
                batch_size=32
            )

    def test_filepath_preservation(self, temp_dir, mock_model_binary):
        """Test that file paths are correctly preserved in output."""
        model_path = os.path.join(temp_dir, 'model.keras')
        data_dir = os.path.join(temp_dir, 'data')
        
        # Create mock generator with specific file paths
        mock_generator = Mock()
        test_filepaths = [
            'normal/image1.png',
            'pneumonia/image2.jpg',
            'normal/image3.jpeg',
            'pneumonia/image4.tiff'
        ]
        mock_generator.filepaths = test_filepaths
        
        with patch('tensorflow.keras.models.load_model') as mock_load_model, \
             patch('tensorflow.keras.preprocessing.image.ImageDataGenerator') as mock_datagen_class:
            
            # Setup mocks
            mock_load_model.return_value = mock_model_binary
            mock_datagen = Mock()
            mock_datagen.flow_from_directory.return_value = mock_generator
            mock_datagen_class.return_value = mock_datagen
            
            # Call the function
            result_df = predict_directory(
                model_path=model_path,
                data_dir=data_dir,
                num_classes=1
            )
            
            # Verify file paths are preserved
            assert result_df['filepath'].tolist() == test_filepaths

    def test_empty_directory_handling(self, temp_dir):
        """Test handling of empty directory or no images."""
        model_path = os.path.join(temp_dir, 'model.keras')
        data_dir = os.path.join(temp_dir, 'empty_data')
        
        # Create mock generator with no files
        mock_generator = Mock()
        mock_generator.filepaths = []
        
        mock_model = Mock()
        mock_model.predict.return_value = np.array([]).reshape(0, 1)
        
        with patch('tensorflow.keras.models.load_model') as mock_load_model, \
             patch('tensorflow.keras.preprocessing.image.ImageDataGenerator') as mock_datagen_class:
            
            # Setup mocks
            mock_load_model.return_value = mock_model
            mock_datagen = Mock()
            mock_datagen.flow_from_directory.return_value = mock_generator
            mock_datagen_class.return_value = mock_datagen
            
            # Call the function
            result_df = predict_directory(
                model_path=model_path,
                data_dir=data_dir,
                num_classes=1
            )
            
            # Verify empty DataFrame is returned
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) == 0
            assert 'filepath' in result_df.columns
            assert 'prediction' in result_df.columns

    def test_model_loading_error(self, temp_dir):
        """Test handling of model loading errors."""
        model_path = os.path.join(temp_dir, 'nonexistent_model.keras')
        data_dir = os.path.join(temp_dir, 'data')
        
        with patch('tensorflow.keras.models.load_model') as mock_load_model:
            # Mock model loading to raise an error
            mock_load_model.side_effect = OSError("Model file not found")
            
            with pytest.raises(OSError, match="Model file not found"):
                predict_directory(
                    model_path=model_path,
                    data_dir=data_dir,
                    num_classes=1
                )

    def test_single_image_prediction(self, temp_dir, mock_model_binary):
        """Test prediction with single image."""
        model_path = os.path.join(temp_dir, 'model.keras')
        data_dir = os.path.join(temp_dir, 'data')
        
        # Create mock generator with single file
        mock_generator = Mock()
        mock_generator.filepaths = ['single/image.jpg']
        
        # Mock model to return single prediction
        mock_model_binary.predict.return_value = np.array([[0.75]])
        
        with patch('tensorflow.keras.models.load_model') as mock_load_model, \
             patch('tensorflow.keras.preprocessing.image.ImageDataGenerator') as mock_datagen_class:
            
            # Setup mocks
            mock_load_model.return_value = mock_model_binary
            mock_datagen = Mock()
            mock_datagen.flow_from_directory.return_value = mock_generator
            mock_datagen_class.return_value = mock_datagen
            
            # Call the function
            result_df = predict_directory(
                model_path=model_path,
                data_dir=data_dir,
                num_classes=1
            )
            
            # Verify single result
            assert len(result_df) == 1
            assert result_df['filepath'].iloc[0] == 'single/image.jpg'
            assert result_df['prediction'].iloc[0] == 0.75

    def test_large_batch_prediction(self, temp_dir):
        """Test prediction with large number of images."""
        model_path = os.path.join(temp_dir, 'model.keras')
        data_dir = os.path.join(temp_dir, 'data')
        
        # Create mock generator with many files
        num_images = 100
        mock_generator = Mock()
        mock_generator.filepaths = [f'class{i%2}/image{i}.jpg' for i in range(num_images)]
        
        # Create mock model with batch predictions
        mock_model = Mock()
        predictions = np.random.rand(num_images, 1)
        mock_model.predict.return_value = predictions
        
        with patch('tensorflow.keras.models.load_model') as mock_load_model, \
             patch('tensorflow.keras.preprocessing.image.ImageDataGenerator') as mock_datagen_class:
            
            # Setup mocks
            mock_load_model.return_value = mock_model
            mock_datagen = Mock()
            mock_datagen.flow_from_directory.return_value = mock_generator
            mock_datagen_class.return_value = mock_datagen
            
            # Call the function
            result_df = predict_directory(
                model_path=model_path,
                data_dir=data_dir,
                num_classes=1
            )
            
            # Verify all predictions are processed
            assert len(result_df) == num_images
            assert len(result_df['filepath'].unique()) == num_images
            
            # Verify predictions match
            np.testing.assert_array_equal(result_df['prediction'].values, predictions.flatten())


def test_inference_cli_help():
    """Test CLI help output."""
    result = subprocess.run(
        [sys.executable, "-m", "src.inference", "--help"], capture_output=True
    )
    assert result.returncode == 0
    assert b"--num_classes" in result.stdout
    assert b"--model_path" in result.stdout
    assert b"--data_dir" in result.stdout
    assert b"--img_size" in result.stdout
    assert b"--output_csv" in result.stdout


def test_main_function_integration():
    """Test the main function with mocked arguments."""
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, 'test_model.keras')
        data_dir = os.path.join(temp_dir, 'test_data')
        output_csv = os.path.join(temp_dir, 'test_predictions.csv')
        
        # Create test DataFrame that predict_directory would return
        test_df = pd.DataFrame({
            'filepath': ['img1.jpg', 'img2.jpg'],
            'prediction': [0.8, 0.2]
        })
        
        # Mock command line arguments
        test_args = [
            'src.inference',
            '--model_path', model_path,
            '--data_dir', data_dir,
            '--output_csv', output_csv,
            '--img_size', '224', '224',
            '--num_classes', '1'
        ]
        
        with patch('sys.argv', test_args), \
             patch('src.inference.predict_directory') as mock_predict, \
             patch('builtins.print') as mock_print:
            
            mock_predict.return_value = test_df
            
            from src.inference import main
            main()
            
            # Verify predict_directory was called with correct arguments
            mock_predict.assert_called_once_with(
                model_path, data_dir, (224, 224), num_classes=1
            )
            
            # Verify CSV was saved (mocked by returning test DataFrame)
            assert mock_print.call_count > 0
            printed_output = str(mock_print.call_args_list[-1])
            assert output_csv in printed_output


class TestInferenceInputValidation:
    """Test suite for input validation in inference module."""

    def test_invalid_model_path_raises_validation_error(self):
        """Test that invalid model path raises ValidationError."""
        with pytest.raises(ValidationError, match="Path traversal detected"):
            predict_directory(
                model_path="../../etc/passwd",
                data_dir="test_data",
                num_classes=1
            )

    def test_invalid_data_directory_raises_validation_error(self):
        """Test that invalid data directory raises ValidationError.""" 
        with patch('src.inference.validate_model_path') as mock_validate_model:
            mock_validate_model.return_value = "valid_model.keras"
            
            with pytest.raises(ValidationError, match="Path traversal detected"):
                predict_directory(
                    model_path="valid_model.keras",
                    data_dir="../../../sensitive_data",
                    num_classes=1
                )

    def test_empty_model_path_raises_validation_error(self):
        """Test that empty model path raises ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            predict_directory(
                model_path="",
                data_dir="test_data",
                num_classes=1
            )

    def test_null_byte_in_paths_raises_validation_error(self):
        """Test that null bytes in paths raise ValidationError."""
        with pytest.raises(ValidationError, match="null bytes"):
            predict_directory(
                model_path="model\x00.keras",
                data_dir="test_data",
                num_classes=1
            )

    def test_invalid_model_extension_raises_validation_error(self):
        """Test that invalid model file extension raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid file extension"):
            predict_directory(
                model_path="model.txt",
                data_dir="test_data",
                num_classes=1
            )

    @patch('src.inference.validate_model_path')
    @patch('src.inference.validate_directory_path')
    def test_main_function_validates_inputs(self, mock_validate_dir, mock_validate_model):
        """Test that main function validates inputs before processing."""
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model.keras')
            data_dir = os.path.join(temp_dir, 'test_data')
            output_csv = os.path.join(temp_dir, 'test_predictions.csv')
            
            # Setup validation mocks
            mock_validate_model.return_value = model_path
            mock_validate_dir.return_value = data_dir
            
            # Mock the prediction function to avoid actual model loading
            test_df = pd.DataFrame({
                'filepath': ['img1.jpg'],
                'prediction': [0.8]
            })
            
            test_args = [
                'src.inference',
                '--model_path', model_path,
                '--data_dir', data_dir,
                '--output_csv', output_csv,
                '--num_classes', '1'
            ]
            
            with patch('sys.argv', test_args), \
                 patch('src.inference.predict_directory') as mock_predict, \
                 patch('builtins.print'):
                
                mock_predict.return_value = test_df
                
                from src.inference import main
                main()
                
                # Verify validation functions were called
                mock_validate_model.assert_called_once_with(model_path, must_exist=True)
                mock_validate_dir.assert_called_once_with(data_dir, must_exist=True)

    def test_main_function_validation_error_handling(self):
        """Test that main function handles validation errors gracefully."""
        test_args = [
            'src.inference',
            '--model_path', '../../malicious.exe',
            '--data_dir', 'test_data',
            '--output_csv', 'output.csv',
            '--num_classes', '1'
        ]
        
        with patch('sys.argv', test_args), \
             patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit):
            
            from src.inference import main
            main()
            
            # Verify error message was printed
            assert mock_print.call_count > 0
            error_output = str(mock_print.call_args_list)
            assert "validation" in error_output.lower() or "error" in error_output.lower()
