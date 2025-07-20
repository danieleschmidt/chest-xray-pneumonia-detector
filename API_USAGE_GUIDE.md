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

## 10. Dependency Security Scanning

Scan project dependencies for known security vulnerabilities using industry-standard tools:

### Basic Security Scan

```bash
cxr-security-scan --verbose
```

### Scan Custom Requirements File

```bash
cxr-security-scan --requirements-file path/to/requirements.txt --output-json security_report.json
```

### CI/CD Integration (Fail on Vulnerabilities)

```bash
cxr-security-scan --fail-on-vulnerabilities --output-json security_results.json
```

### Advanced Options

```bash
cxr-security-scan \
    --requirements-file requirements.txt \
    --output-json vulnerability_report.json \
    --verbose \
    --fail-on-vulnerabilities
```

The security scanner provides:
- **Automatic tool detection**: Uses pip-audit (preferred) or safety as fallback
- **Vulnerability classification**: Critical, High, Medium, Low severity levels
- **Detailed reporting**: Package versions, vulnerability IDs, fix recommendations
- **JSON export**: Structured data for automated processing and CI/CD integration
- **Zero false positives**: Only reports confirmed vulnerabilities from official databases
- **Performance metrics**: Scan duration and tool information

**Tool Installation:**
```bash
# For comprehensive scanning, install security tools:
pip install pip-audit  # PyPA's official security scanner (recommended)
pip install safety     # Alternative security scanner
```

**Example Output:**
```
🔍 Scanning dependencies for security vulnerabilities...

=== Dependency Security Scan Results ===
Scan Tool: pip-audit
Scan Duration: 3.45 seconds
Total Vulnerabilities: 2
  Critical: 0
  High: 1
  Medium: 1
  Low: 0

=== Vulnerability Details ===
📦 Package: requests (2.25.0)
🔴 Vulnerability: PYSEC-2021-59
⚠️  Severity: HIGH
📝 Description: Inefficient Regular Expression Complexity
✅ Fixed in: 2.25.1
```

Security scanning integrates seamlessly with CI/CD pipelines and provides actionable intelligence for maintaining secure dependencies.

## 11. Synthetic Medical Data Generation

Generate realistic synthetic chest X-ray datasets for testing, development, and validation without privacy concerns:

### Basic Dataset Generation

```bash
cxr-generate-data --output-dir ./test_data --total-images 100 --image-size 224
```

### Advanced Dataset Configuration

```bash
cxr-generate-data \
    --output-dir ./synthetic_dataset \
    --total-images 500 \
    --image-size 224 \
    --pathology-probability 0.35 \
    --noise-level 0.15 \
    --train-split 0.8 \
    --val-split 0.1 \
    --test-split 0.1 \
    --contrast-enhancement \
    --anatomical-markers \
    --verbose
```

### Dataset Splits and Structure

```bash
# Generate dataset with custom splits
cxr-generate-data \
    --output-dir ./medical_test_data \
    --total-images 200 \
    --train-split 0.7 \
    --val-split 0.2 \
    --test-split 0.1 \
    --pathology-probability 0.4
```

**Generated Directory Structure:**
```
synthetic_medical_dataset/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── dataset_metadata.json
```

The synthetic data generator provides:
- **Realistic Medical Features**: Anatomical structures (ribcage, heart, spine, lung fields)
- **Pathological Variations**: Configurable pneumonia patterns and consolidations
- **Quality Control**: Noise, contrast, and brightness variations matching real X-rays
- **Comprehensive Metadata**: JSON metadata with quality metrics and dataset statistics
- **Privacy Safe**: No patient data, fully synthetic generation
- **Scalable**: Generate datasets from small test sets to large training corpora

**Key Parameters:**
- `--pathology-probability`: Controls ratio of pathological to normal images (0.0-1.0)
- `--noise-level`: Medical imaging noise simulation (0.0-1.0)
- `--image-size`: Output image dimensions (square format)
- `--contrast-enhancement`: Apply medical-grade contrast enhancement
- `--anatomical-markers`: Include detailed anatomical structures

**Use Cases:**
- Integration testing without privacy concerns
- Performance benchmarking with controlled datasets
- Algorithm validation with known ground truth
- CI/CD pipeline testing with reproducible data
- Development and debugging with realistic medical imagery

**Example Output:**
```
🔧 Configuration:
  Image Size: (224, 224)
  Pathology Probability: 0.35
  Noise Level: 0.15

Generating 400 images for train split...
Generating 50 images for val split...
Generating 50 images for test split...

✅ Synthetic medical dataset created successfully!
📍 Location: ./synthetic_dataset/synthetic_medical_dataset
📊 Total Images: 500
🫁 Normal: 325, 🦠 Pneumonia: 175
⏱️  Generation completed in 15.2 seconds
📈 Performance: 32.9 images/second
```
