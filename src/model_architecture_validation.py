"""Model architecture validation utilities for ensuring model correctness."""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Local imports
try:
    from .model_builder import (
        create_simple_cnn,
        create_transfer_learning_model,
        create_cnn_with_attention,
    )
except ImportError:
    try:
        from model_builder import (
            create_simple_cnn,
            create_transfer_learning_model,
            create_cnn_with_attention,
        )
    except ImportError:
        create_simple_cnn = None
        create_transfer_learning_model = None
        create_cnn_with_attention = None


@dataclass
class ValidationResult:
    """Container for model architecture validation results."""

    model_name: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[Optional[int], ...]
    total_params: int
    trainable_params: int
    layer_count: int
    is_valid: bool
    validation_errors: List[str]
    layer_types: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert validation result to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class ModelArchitectureValidator:
    """Validator for model architectures."""

    def __init__(self):
        self.validation_results: List[ValidationResult] = []

    def validate_model(
        self,
        model,
        expected_input_shape: Optional[Tuple[int, ...]] = None,
        expected_output_classes: Optional[int] = None,
        min_params: Optional[int] = None,
        max_params: Optional[int] = None,
        expected_layer_types: Optional[List[str]] = None,
        min_conv_layers: Optional[int] = None,
        min_dense_layers: Optional[int] = None,
    ) -> ValidationResult:
        """Validate a model's architecture.

        Parameters
        ----------
        model : tf.keras.Model
            The model to validate
        expected_input_shape : tuple, optional
            Expected input shape (height, width, channels)
        expected_output_classes : int, optional
            Expected number of output classes
        min_params : int, optional
            Minimum number of parameters
        max_params : int, optional
            Maximum number of parameters
        expected_layer_types : list, optional
            Expected layer types to be present
        min_conv_layers : int, optional
            Minimum number of convolutional layers
        min_dense_layers : int, optional
            Minimum number of dense layers

        Returns
        -------
        ValidationResult
            Validation results with pass/fail status and errors
        """
        errors = []

        # Get model summary
        summary = get_model_summary(model)

        # Determine model name based on architecture
        model_name = self._infer_model_name(model, summary)

        # Validate input shape
        if expected_input_shape is not None:
            input_errors = self._validate_input_shape(
                summary["input_shape"], expected_input_shape
            )
            errors.extend(input_errors)

        # Validate output shape
        if expected_output_classes is not None:
            output_errors = validate_output_shape(model, expected_output_classes)
            errors.extend(output_errors)

        # Validate parameter count
        if min_params is not None or max_params is not None:
            param_errors = validate_parameter_count(model, min_params, max_params)
            errors.extend(param_errors)

        # Validate layer structure
        layer_errors = validate_layer_structure(
            model, expected_layer_types, min_conv_layers, min_dense_layers
        )
        errors.extend(layer_errors)

        # Create validation result
        result = ValidationResult(
            model_name=model_name,
            input_shape=summary["input_shape"],
            output_shape=summary["output_shape"],
            total_params=summary["total_params"],
            trainable_params=summary["trainable_params"],
            layer_count=summary["layer_count"],
            is_valid=len(errors) == 0,
            validation_errors=errors,
            layer_types=summary["layer_types"],
            metadata=summary["metadata"],
        )

        self.validation_results.append(result)
        return result

    def _infer_model_name(self, model, summary: Dict[str, Any]) -> str:
        """Infer model name from architecture characteristics."""
        layer_types = summary["layer_types"]

        # Check for transfer learning (has base model layers)
        base_model_layers = ["MobileNetV2", "VGG16", "ResNet"]
        for base_name in base_model_layers:
            if any(base_name.lower() in layer.lower() for layer in layer_types):
                return f"transfer_learning_{base_name}"

        # Check for attention mechanism (Squeeze-Excite blocks)
        if "Multiply" in layer_types and "GlobalAveragePooling2D" in layer_types:
            return "attention_cnn"

        # Check for simple CNN
        if "Conv2D" in layer_types and "Flatten" in layer_types:
            return "simple_cnn"

        # Default fallback
        return "unknown_architecture"

    def _validate_input_shape(
        self, actual_shape: Tuple[int, ...], expected_shape: Tuple[int, ...]
    ) -> List[str]:
        """Validate input shape matches expectations."""
        errors = []

        if actual_shape != expected_shape:
            errors.append(
                f"Input shape mismatch: expected {expected_shape}, "
                f"got {actual_shape}"
            )

        return errors

    def generate_report(self) -> str:
        """Generate a human-readable validation report."""
        if not self.validation_results:
            return "No validation results available."

        report = ["Model Architecture Validation Report", "=" * 40, ""]

        passed_count = sum(1 for r in self.validation_results if r.is_valid)
        total_count = len(self.validation_results)

        report.append(f"Summary: {passed_count}/{total_count} models passed validation")
        report.append("")

        for result in self.validation_results:
            status = "PASSED" if result.is_valid else "FAILED"
            report.append(f"Model: {result.model_name} - {status}")
            report.append(f"  Input Shape: {result.input_shape}")
            report.append(f"  Output Shape: {result.output_shape}")
            report.append(
                f"  Parameters: {result.total_params:,} total, "
                f"{result.trainable_params:,} trainable"
            )
            report.append(f"  Layers: {result.layer_count}")

            if result.validation_errors:
                report.append("  Errors:")
                for error in result.validation_errors:
                    report.append(f"    - {error}")

            report.append("")

        return "\n".join(report)


def get_model_summary(model) -> Dict[str, Any]:
    """Extract comprehensive summary information from a model.

    Parameters
    ----------
    model : tf.keras.Model
        The model to summarize

    Returns
    -------
    dict
        Dictionary containing model summary information
    """
    # Basic shape information
    input_shape = model.input.shape[1:]  # Remove batch dimension
    output_shape = model.output.shape

    # Parameter counts
    total_params = model.count_params()
    trainable_params = sum(
        [tf.keras.backend.count_params(w) for w in model.trainable_weights]
    )

    # Layer information
    layer_types = []
    layer_count = len(model.layers)

    for layer in model.layers:
        layer_type = type(layer).__name__
        if layer_type not in layer_types:
            layer_types.append(layer_type)

    # Optimizer and loss information
    metadata = {}
    if hasattr(model, "optimizer") and model.optimizer:
        metadata["optimizer"] = type(model.optimizer).__name__
    if hasattr(model, "loss") and model.loss:
        if hasattr(model.loss, "__name__"):
            metadata["loss"] = model.loss.__name__
        else:
            metadata["loss"] = str(type(model.loss).__name__)

    return {
        "input_shape": input_shape,
        "output_shape": output_shape,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "layer_count": layer_count,
        "layer_types": layer_types,
        "metadata": metadata,
    }


def validate_layer_structure(
    model,
    expected_layer_types: Optional[List[str]] = None,
    min_conv_layers: Optional[int] = None,
    min_dense_layers: Optional[int] = None,
) -> List[str]:
    """Validate the layer structure of a model.

    Parameters
    ----------
    model : tf.keras.Model
        The model to validate
    expected_layer_types : list, optional
        Expected layer types that should be present
    min_conv_layers : int, optional
        Minimum number of Conv2D layers
    min_dense_layers : int, optional
        Minimum number of Dense layers

    Returns
    -------
    list
        List of validation error messages
    """
    errors = []

    # Get layer types
    layer_types = [type(layer).__name__ for layer in model.layers]
    unique_layer_types = list(set(layer_types))

    # Check for expected layer types
    if expected_layer_types:
        for expected_type in expected_layer_types:
            if expected_type not in unique_layer_types:
                errors.append(f"Missing expected layer type: {expected_type}")

    # Count specific layer types
    conv_count = layer_types.count("Conv2D")
    dense_count = layer_types.count("Dense")

    # Validate minimum layer counts
    if min_conv_layers and conv_count < min_conv_layers:
        errors.append(
            f"Insufficient Conv2D layers: expected at least {min_conv_layers}, "
            f"found {conv_count}"
        )

    if min_dense_layers and dense_count < min_dense_layers:
        errors.append(
            f"Insufficient Dense layers: expected at least {min_dense_layers}, "
            f"found {dense_count}"
        )

    return errors


def validate_output_shape(
    model,
    expected_classes: int,
) -> List[str]:
    """Validate the output shape of a model.

    Parameters
    ----------
    model : tf.keras.Model
        The model to validate
    expected_classes : int
        Expected number of output classes

    Returns
    -------
    list
        List of validation error messages
    """
    errors = []

    output_shape = model.output.shape
    actual_classes = output_shape[-1]

    if actual_classes != expected_classes:
        errors.append(
            f"Output classes mismatch: expected {expected_classes}, "
            f"got {actual_classes}"
        )

    return errors


def validate_parameter_count(
    model,
    min_params: Optional[int] = None,
    max_params: Optional[int] = None,
) -> List[str]:
    """Validate the parameter count of a model.

    Parameters
    ----------
    model : tf.keras.Model
        The model to validate
    min_params : int, optional
        Minimum number of parameters
    max_params : int, optional
        Maximum number of parameters

    Returns
    -------
    list
        List of validation error messages
    """
    errors = []

    total_params = model.count_params()

    if min_params and total_params < min_params:
        errors.append(
            f"Too few parameters: expected at least {min_params:,}, "
            f"got {total_params:,}"
        )

    if max_params and total_params > max_params:
        errors.append(
            f"Too many parameters: expected at most {max_params:,}, "
            f"got {total_params:,}"
        )

    return errors


def validate_model_architecture(
    models_config: List[Dict[str, Any]],
) -> List[ValidationResult]:
    """Validate multiple model architectures based on configuration.

    Parameters
    ----------
    models_config : list
        List of model configuration dictionaries

    Returns
    -------
    list
        List of ValidationResult objects
    """
    validator = ModelArchitectureValidator()
    results = []

    for config in models_config:
        model_type = config["type"]
        input_shape = config["input_shape"]
        num_classes = config.get("num_classes", 1)

        # Create model based on type
        if model_type == "simple_cnn":
            model = create_simple_cnn(
                input_shape=input_shape,
                num_classes=num_classes,
                learning_rate=config.get("learning_rate", 0.001),
                dropout_rate=config.get("dropout_rate", 0.0),
            )
        elif model_type == "transfer_learning":
            model = create_transfer_learning_model(
                input_shape=input_shape,
                num_classes=num_classes,
                base_model_name=config.get("base_model_name", "MobileNetV2"),
                learning_rate=config.get("learning_rate", 0.001),
                dropout_rate=config.get("dropout_rate", 0.0),
            )
        elif model_type == "attention_cnn":
            model = create_cnn_with_attention(
                input_shape=input_shape,
                num_classes=num_classes,
                learning_rate=config.get("learning_rate", 0.001),
                dropout_rate=config.get("dropout_rate", 0.0),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Validate model
        result = validator.validate_model(
            model,
            expected_input_shape=config.get("expected_input_shape"),
            expected_output_classes=config.get("expected_output_classes", num_classes),
            min_params=config.get("min_params"),
            max_params=config.get("max_params"),
            expected_layer_types=config.get("expected_layer_types"),
            min_conv_layers=config.get("min_conv_layers"),
            min_dense_layers=config.get("min_dense_layers"),
        )

        results.append(result)

    return results


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Model architecture validation for chest X-ray pneumonia detector"
    )

    parser.add_argument(
        "--config", type=str, help="JSON file with model configurations to validate"
    )
    parser.add_argument(
        "--output_json", type=str, help="Save validation results to JSON file"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    return parser


def main():
    """Main CLI entry point."""
    if not TF_AVAILABLE:
        print(
            "Error: TensorFlow is not available. Please install TensorFlow to use model validation."
        )
        sys.exit(1)

    parser = create_parser()
    args = parser.parse_args()

    if not args.config:
        # Default validation configuration
        models_config = [
            {"type": "simple_cnn", "input_shape": (150, 150, 3), "num_classes": 1},
            {
                "type": "transfer_learning",
                "input_shape": (224, 224, 3),
                "num_classes": 1,
                "base_model_name": "MobileNetV2",
            },
            {"type": "attention_cnn", "input_shape": (150, 150, 3), "num_classes": 1},
        ]
    else:
        with open(args.config, "r") as f:
            models_config = json.load(f)

    # Run validation
    print("Running model architecture validation...")
    results = validate_model_architecture(models_config)

    # Create validator for report generation
    validator = ModelArchitectureValidator()
    validator.validation_results = results

    # Print report
    print("\n" + validator.generate_report())

    # Print summary
    passed_count = sum(1 for r in results if r.is_valid)
    total_count = len(results)

    if passed_count == total_count:
        print(f"✅ All {total_count} model architectures validated successfully!")
    else:
        failed_count = total_count - passed_count
        print(
            f"❌ {failed_count} of {total_count} model architectures failed validation."
        )

    # Save to JSON if requested
    if args.output_json:
        output_data = {
            "summary": {
                "total_models": total_count,
                "passed": passed_count,
                "failed": total_count - passed_count,
            },
            "results": [result.to_dict() for result in results],
        }

        with open(args.output_json, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")

    # Exit with error code if any validations failed
    if passed_count < total_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
