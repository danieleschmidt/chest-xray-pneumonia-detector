"""Tests for model architecture validation."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

pytest.importorskip("tensorflow")
import tensorflow as tf

from src.model_builder import (
    create_simple_cnn,
    create_transfer_learning_model,
    create_cnn_with_attention,
    _squeeze_excite_block,
)
from src.model_architecture_validation import (
    ModelArchitectureValidator,
    ValidationResult,
    validate_model_architecture,
    validate_layer_structure,
    validate_output_shape,
    validate_parameter_count,
    get_model_summary,
)


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_validation_result_creation(self):
        """Test creating ValidationResult with all fields."""
        result = ValidationResult(
            model_name="simple_cnn",
            input_shape=(150, 150, 3),
            output_shape=(None, 1),
            total_params=123456,
            trainable_params=123456,
            layer_count=8,
            is_valid=True,
            validation_errors=[],
            layer_types=['Conv2D', 'MaxPooling2D', 'Dense'],
            metadata={'optimizer': 'Adam', 'loss': 'binary_crossentropy'}
        )
        
        assert result.model_name == "simple_cnn"
        assert result.input_shape == (150, 150, 3)
        assert result.output_shape == (None, 1)
        assert result.total_params == 123456
        assert result.is_valid is True
        assert len(result.validation_errors) == 0

    def test_validation_result_with_errors(self):
        """Test ValidationResult with validation errors."""
        errors = ["Output shape mismatch", "Layer count incorrect"]
        result = ValidationResult(
            model_name="test_model",
            input_shape=(224, 224, 3),
            output_shape=(None, 2),
            total_params=1000,
            trainable_params=1000,
            layer_count=5,
            is_valid=False,
            validation_errors=errors
        )
        
        assert result.is_valid is False
        assert len(result.validation_errors) == 2
        assert "Output shape mismatch" in result.validation_errors


class TestModelArchitectureValidator:
    """Test ModelArchitectureValidator class."""

    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = ModelArchitectureValidator()
        assert validator.validation_results == []

    def test_validate_simple_cnn_architecture(self):
        """Test validation of simple CNN architecture."""
        model = create_simple_cnn(input_shape=(150, 150, 3), num_classes=1)
        validator = ModelArchitectureValidator()
        
        result = validator.validate_model(model, expected_output_classes=1)
        
        assert result.is_valid is True
        assert result.model_name == "simple_cnn"
        assert result.input_shape == (150, 150, 3)
        assert result.output_shape[1] == 1  # Binary classification
        assert len(result.validation_errors) == 0
        assert 'Conv2D' in result.layer_types
        assert 'Dense' in result.layer_types

    def test_validate_transfer_learning_mobilenet_architecture(self):
        """Test validation of MobileNetV2 transfer learning architecture."""
        model = create_transfer_learning_model(
            input_shape=(224, 224, 3),
            num_classes=1,
            base_model_name="MobileNetV2"
        )
        validator = ModelArchitectureValidator()
        
        result = validator.validate_model(model, expected_output_classes=1)
        
        assert result.is_valid is True
        assert result.model_name == "transfer_learning_MobileNetV2"
        assert result.input_shape == (224, 224, 3)
        assert result.output_shape[1] == 1
        assert result.total_params > 100000  # MobileNetV2 has many parameters
        assert 'GlobalAveragePooling2D' in result.layer_types

    def test_validate_transfer_learning_vgg16_architecture(self):
        """Test validation of VGG16 transfer learning architecture."""
        model = create_transfer_learning_model(
            input_shape=(224, 224, 3),
            num_classes=3,
            base_model_name="VGG16"
        )
        validator = ModelArchitectureValidator()
        
        result = validator.validate_model(model, expected_output_classes=3)
        
        assert result.is_valid is True
        assert result.model_name == "transfer_learning_VGG16"
        assert result.output_shape[1] == 3  # Multi-class
        assert result.total_params > 1000000  # VGG16 has many parameters

    def test_validate_attention_cnn_architecture(self):
        """Test validation of CNN with attention architecture."""
        model = create_cnn_with_attention(
            input_shape=(150, 150, 3),
            num_classes=2
        )
        validator = ModelArchitectureValidator()
        
        result = validator.validate_model(model, expected_output_classes=2)
        
        assert result.is_valid is True
        assert result.model_name == "attention_cnn"
        assert result.output_shape[1] == 2
        assert 'GlobalAveragePooling2D' in result.layer_types  # From SE blocks
        assert 'Multiply' in result.layer_types  # From SE blocks

    def test_validate_model_with_mismatched_output_classes(self):
        """Test validation fails with mismatched output classes."""
        model = create_simple_cnn(input_shape=(150, 150, 3), num_classes=1)
        validator = ModelArchitectureValidator()
        
        result = validator.validate_model(model, expected_output_classes=3)
        
        assert result.is_valid is False
        assert any("Output classes mismatch" in error for error in result.validation_errors)

    def test_validate_model_with_wrong_input_shape(self):
        """Test validation with unexpected input shape."""
        model = create_simple_cnn(input_shape=(128, 128, 3), num_classes=1)
        validator = ModelArchitectureValidator()
        
        result = validator.validate_model(
            model, 
            expected_input_shape=(150, 150, 3),
            expected_output_classes=1
        )
        
        assert result.is_valid is False
        assert any("Input shape mismatch" in error for error in result.validation_errors)

    def test_validate_multiple_models(self):
        """Test validating multiple models."""
        validator = ModelArchitectureValidator()
        
        model1 = create_simple_cnn(input_shape=(150, 150, 3))
        model2 = create_cnn_with_attention(input_shape=(150, 150, 3))
        
        result1 = validator.validate_model(model1, expected_output_classes=1)
        result2 = validator.validate_model(model2, expected_output_classes=1)
        
        assert len(validator.validation_results) == 2
        assert result1.model_name != result2.model_name
        assert result1.is_valid is True
        assert result2.is_valid is True

    def test_generate_validation_report(self):
        """Test generating validation report."""
        validator = ModelArchitectureValidator()
        
        model = create_simple_cnn(input_shape=(150, 150, 3))
        validator.validate_model(model, expected_output_classes=1)
        
        report = validator.generate_report()
        
        assert "Model Architecture Validation Report" in report
        assert "simple_cnn" in report
        assert "PASSED" in report


class TestLayerStructureValidation:
    """Test layer structure validation functions."""

    def test_validate_layer_structure_simple_cnn(self):
        """Test layer structure validation for simple CNN."""
        model = create_simple_cnn(input_shape=(150, 150, 3))
        
        errors = validate_layer_structure(
            model,
            expected_layer_types=['Conv2D', 'MaxPooling2D', 'Flatten', 'Dense'],
            min_conv_layers=2,
            min_dense_layers=2
        )
        
        assert len(errors) == 0

    def test_validate_layer_structure_insufficient_layers(self):
        """Test layer structure validation with insufficient layers."""
        model = create_simple_cnn(input_shape=(150, 150, 3))
        
        errors = validate_layer_structure(
            model,
            min_conv_layers=5,  # More than model has
            min_dense_layers=5   # More than model has
        )
        
        assert len(errors) > 0
        assert any("Insufficient Conv2D layers" in error for error in errors)
        assert any("Insufficient Dense layers" in error for error in errors)

    def test_validate_layer_structure_missing_types(self):
        """Test layer structure validation with missing layer types."""
        model = create_simple_cnn(input_shape=(150, 150, 3))
        
        errors = validate_layer_structure(
            model,
            expected_layer_types=['Conv2D', 'LSTM', 'Dense']  # LSTM not in model
        )
        
        assert len(errors) > 0
        assert any("Missing expected layer type: LSTM" in error for error in errors)


class TestOutputShapeValidation:
    """Test output shape validation functions."""

    def test_validate_output_shape_correct(self):
        """Test output shape validation with correct shape."""
        model = create_simple_cnn(input_shape=(150, 150, 3), num_classes=1)
        
        errors = validate_output_shape(model, expected_classes=1)
        
        assert len(errors) == 0

    def test_validate_output_shape_incorrect(self):
        """Test output shape validation with incorrect shape."""
        model = create_simple_cnn(input_shape=(150, 150, 3), num_classes=1)
        
        errors = validate_output_shape(model, expected_classes=5)
        
        assert len(errors) > 0
        assert any("Output classes mismatch" in error for error in errors)

    def test_validate_output_shape_multiclass(self):
        """Test output shape validation for multiclass model."""
        model = create_cnn_with_attention(input_shape=(150, 150, 3), num_classes=3)
        
        errors = validate_output_shape(model, expected_classes=3)
        
        assert len(errors) == 0


class TestParameterCountValidation:
    """Test parameter count validation functions."""

    def test_validate_parameter_count_within_range(self):
        """Test parameter count validation within acceptable range."""
        model = create_simple_cnn(input_shape=(150, 150, 3))
        
        errors = validate_parameter_count(
            model,
            min_params=1000,
            max_params=10000000
        )
        
        assert len(errors) == 0

    def test_validate_parameter_count_too_few(self):
        """Test parameter count validation with too few parameters."""
        model = create_simple_cnn(input_shape=(150, 150, 3))
        
        errors = validate_parameter_count(
            model,
            min_params=99999999  # Much larger than model has
        )
        
        assert len(errors) > 0
        assert any("Too few parameters" in error for error in errors)

    def test_validate_parameter_count_too_many(self):
        """Test parameter count validation with too many parameters."""
        model = create_transfer_learning_model(
            input_shape=(224, 224, 3),
            base_model_name="VGG16"
        )
        
        errors = validate_parameter_count(
            model,
            max_params=1000  # Much smaller than VGG16 has
        )
        
        assert len(errors) > 0
        assert any("Too many parameters" in error for error in errors)


class TestSqueezeExciteBlock:
    """Test squeeze-and-excite block functionality."""

    def test_squeeze_excite_block_shape(self):
        """Test squeeze-and-excite block output shape."""
        # Create a dummy input tensor
        inputs = tf.keras.Input(shape=(32, 32, 64))
        
        # Apply squeeze-excite block
        output = _squeeze_excite_block(inputs, ratio=16)
        
        # Output should have same shape as input
        assert output.shape[-1] == inputs.shape[-1]
        assert output.shape[1:3] == inputs.shape[1:3]

    def test_squeeze_excite_block_different_ratios(self):
        """Test squeeze-excite block with different reduction ratios."""
        inputs = tf.keras.Input(shape=(32, 32, 64))
        
        # Test different ratios
        for ratio in [8, 16, 32]:
            output = _squeeze_excite_block(inputs, ratio=ratio)
            assert output.shape == inputs.shape


class TestUtilityFunctions:
    """Test utility functions for model validation."""

    def test_get_model_summary_simple_cnn(self):
        """Test getting model summary for simple CNN."""
        model = create_simple_cnn(input_shape=(150, 150, 3))
        
        summary = get_model_summary(model)
        
        assert summary['input_shape'] == (150, 150, 3)
        assert summary['output_shape'][1] == 1
        assert summary['total_params'] > 0
        assert summary['trainable_params'] > 0
        assert len(summary['layer_types']) > 0
        assert 'Conv2D' in summary['layer_types']

    def test_get_model_summary_transfer_learning(self):
        """Test getting model summary for transfer learning model."""
        model = create_transfer_learning_model(
            input_shape=(224, 224, 3),
            base_model_name="MobileNetV2"
        )
        
        summary = get_model_summary(model)
        
        assert summary['input_shape'] == (224, 224, 3)
        assert summary['total_params'] > 100000  # MobileNetV2 is large
        assert 'GlobalAveragePooling2D' in summary['layer_types']


class TestValidateModelArchitecture:
    """Test high-level validate_model_architecture function."""

    def test_validate_model_architecture_all_types(self):
        """Test validation of all model architecture types."""
        models_config = [
            {
                'type': 'simple_cnn',
                'input_shape': (150, 150, 3),
                'num_classes': 1
            },
            {
                'type': 'transfer_learning',
                'input_shape': (224, 224, 3),
                'num_classes': 2,
                'base_model_name': 'MobileNetV2'
            },
            {
                'type': 'attention_cnn',
                'input_shape': (150, 150, 3),
                'num_classes': 3
            }
        ]
        
        results = validate_model_architecture(models_config)
        
        assert len(results) == 3
        assert all(result.is_valid for result in results)
        assert results[0].model_name == "simple_cnn"
        assert results[1].model_name == "transfer_learning_MobileNetV2"
        assert results[2].model_name == "attention_cnn"

    def test_validate_model_architecture_with_failures(self):
        """Test validation with expected failures."""
        models_config = [
            {
                'type': 'simple_cnn',
                'input_shape': (150, 150, 3),
                'num_classes': 1,
                'expected_output_classes': 5  # Mismatch
            }
        ]
        
        results = validate_model_architecture(models_config)
        
        assert len(results) == 1
        assert results[0].is_valid is False
        assert len(results[0].validation_errors) > 0


class TestCLIInterface:
    """Test command-line interface for architecture validation."""

    @patch('builtins.print')
    def test_cli_validation_help(self, mock_print):
        """Test CLI help functionality."""
        from src.model_architecture_validation import main
        
        with patch('sys.argv', ['validate', '--help']):
            with pytest.raises(SystemExit):
                main()

    @patch('src.model_architecture_validation.validate_model_architecture')
    @patch('builtins.print')
    def test_cli_validation_run(self, mock_print, mock_validate):
        """Test CLI validation execution."""
        mock_validate.return_value = [
            ValidationResult(
                model_name="test_model",
                input_shape=(150, 150, 3),
                output_shape=(None, 1),
                total_params=1000,
                trainable_params=1000,
                layer_count=5,
                is_valid=True,
                validation_errors=[]
            )
        ]
        
        from src.model_architecture_validation import main
        
        with patch('sys.argv', ['validate', '--config', 'test_config.json']):
            main()
            
        mock_validate.assert_called_once()
        mock_print.assert_called()