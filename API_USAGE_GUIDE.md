# API Usage Guide

This guide provides examples for using the command line interfaces included with the project.

## 1. Minimal Training Pipeline

The `chest_xray_pneumonia_detector.pipeline` module offers a lightweight training entry point. Example:

```bash
python -m chest_xray_pneumonia_detector.pipeline \
    --train_dir data/train \
    --val_dir data/val \
    --epochs 5 \
    --batch_size 32 \
    --model_type transfer \
    --base_model_name MobileNetV2
```

## 2. Full Training Engine

For advanced options such as MLflow logging and data augmentation, use `src.train_engine`:

```bash
python -m src.train_engine \
    --train_dir data/train \
    --val_dir data/val \
    --epochs 10 \
    --rotation_range 20 \
    --brightness_range 0.7 1.3 \
    --checkpoint_path saved_models/best_model.keras
```

## 3. Grad-CAM Visualization

Generate a Grad-CAM overlay for a single image with `src.predict_utils`:

```bash
python -m src.predict_utils \
    --model_path saved_models/best_model.keras \
    --img_path path/to/image.jpg \
    --last_conv_layer_name conv_pw_13_relu \
    --output_path gradcam.png \
    --img_size 150 150
```

## 4. Batch Inference

Use `src.inference` to produce predictions for an entire directory of images:

```bash
python -m src.inference \
    --model_path saved_models/best_model.keras \
    --data_dir path/to/images \
    --output_csv predictions.csv \
    --num_classes 1
```

The resulting CSV lists each file path and the associated prediction.

## 5. Evaluate Predictions

Compute metrics from a predictions CSV using `src.evaluate`:

```bash
python -m src.evaluate \
    --pred_csv predictions.csv \
    --label_csv labels.csv \
    --output_png eval_confusion.png \
    --normalize_cm
```

This generates a confusion matrix image and prints precision, recall and F1-score.

## 6. Check Package Version

Print the installed version of the project using `version_cli`:

```bash
cxr-version  # or `python -m src.version_cli`
```

## 7. Dataset Statistics

Count the number of images per class in a dataset. You can restrict the scan to specific extensions if needed and store the results as CSV:

```bash
cxr-dataset-stats --input_dir path/to/data --json_output counts.json \
    --csv_output counts.csv \
    --extensions .jpeg .bmp \
    --plot_png counts.png
```

Use ``--sort_by count`` to sort results by descending count instead of the
default alphabetical order.

Extensions are case-insensitive and may be provided without a leading dot. By
default, output files list classes in alphabetical order for readability. The
command exits with an error if the input path does not exist or is not a directory.
Plotting requires the optional ``matplotlib`` dependency (install with ``pip install matplotlib``). The PNG output directory must exist.

## 8. Performance Benchmarking

Benchmark training and inference performance with timing, memory usage, and throughput metrics:

### Training Benchmark

```bash
cxr-benchmark training \
    --epochs 5 \
    --batch_size 32 \
    --use_transfer_learning \
    --base_model_name MobileNetV2 \
    --output_json training_benchmark.json \
    --verbose
```

### Inference Benchmark

```bash
cxr-benchmark inference \
    --model_path saved_models/best_model.keras \
    --data_dir path/to/test_data \
    --num_classes 1 \
    --num_samples 100 \
    --output_json inference_benchmark.json \
    --verbose
```

The benchmarking tool provides detailed metrics including:
- Total execution time
- Average time per epoch (for training)
- Peak memory usage
- Throughput (samples per second)
- Comprehensive metadata about the configuration

Results can be saved as JSON for further analysis or integration into CI/CD pipelines.

## 9. Model Architecture Validation

Validate model architectures to ensure they meet expected specifications and structural requirements:

### Default Validation (All Model Types)

```bash
cxr-validate --verbose --output_json validation_results.json
```

### Custom Configuration Validation

Create a JSON configuration file (e.g., `model_config.json`):

```json
[
  {
    "type": "simple_cnn",
    "input_shape": [150, 150, 3],
    "num_classes": 1,
    "min_conv_layers": 2,
    "min_dense_layers": 2
  },
  {
    "type": "transfer_learning",
    "input_shape": [224, 224, 3],
    "num_classes": 2,
    "base_model_name": "MobileNetV2",
    "min_params": 100000,
    "max_params": 10000000
  },
  {
    "type": "attention_cnn",
    "input_shape": [150, 150, 3],
    "num_classes": 3,
    "expected_layer_types": ["Conv2D", "GlobalAveragePooling2D", "Multiply"]
  }
]
```

Then run validation:

```bash
cxr-validate --config model_config.json --output_json validation_results.json --verbose
```

The validation tool checks:
- Input/output shape correctness
- Parameter count within expected ranges
- Layer structure and types
- Model compilation status
- Architecture-specific requirements (e.g., attention blocks, transfer learning base models)

Validation results include detailed error messages and can be exported as JSON for automated testing and CI/CD integration.
