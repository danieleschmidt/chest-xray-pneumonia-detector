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
    --extensions .jpeg .bmp

Extensions are case-insensitive and may be provided without a leading dot. Both
output files list classes in alphabetical order for readability. The command
exits with an error if the input path does not exist or is not a directory.
```
