# Chest X-Ray Pneumonia Detector

This project aims to build and evaluate a Convolutional Neural Network (CNN) to detect pneumonia from chest X-ray images. It will explore data augmentation, model architecture design, transfer learning, and model interpretability.

## Project Goals
- Implement a robust data loading and preprocessing pipeline for X-ray images.
- Design and train a custom CNN for pneumonia detection.
- Experiment with transfer learning using pre-trained models (e.g., VGG16, ResNet).
- Implement techniques for handling class imbalance if present in the dataset.
- Evaluate model performance using appropriate metrics (accuracy, precision, recall, F1-score, AUC-ROC).
- Explore model interpretability techniques like Grad-CAM.

## Tech Stack (Planned)
- Python
- TensorFlow / Keras or PyTorch
- OpenCV
- Scikit-learn
- Matplotlib / Seaborn
- Pandas, NumPy

## Initial File Structure
chest-xray-pneumonia-detector/
├── data/ # For image datasets (structure for train/val/test)
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
├── notebooks/
│   └── cnn_exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py # Image loading and augmentation
│   ├── model_builder.py # CNN model definitions
│   ├── train_engine.py # Training and evaluation loop
│   ├── predict_utils.py # Inference functions
│   └── grad_cam.py # For interpretability
├── tests/
│   ├── __init__.py
│   └── test_data_loader.py
├── saved_models/ # To store trained model weights
├── requirements.txt
├── .gitignore
└── README.md

## Dataset Split Utility

Prepare a train/validation/test directory structure from a single folder of
class subdirectories:

```bash
python -m src.data_split \
    --input_dir path/to/raw_dataset \
    --output_dir data \
    --val_frac 0.1 \
    --test_frac 0.1 \
    --seed 42 \
    --move
```

Fractions must sum to less than `1.0`. By default the script copies images into
`data/train`, `data/val` and `data/test` while preserving the original class
folder names. Use `--move` to move files instead of copying them.

## How to Contribute (and test Jules)
This project uses Jules, our Async Development Agent, for feature development and bug fixes. Create detailed issues for Jules to work on.

## Model Evaluation

`src/train_engine.py` now saves a validation confusion matrix to `reports/confusion_matrix_val.png` after training. The script also prints precision, recall, F1-score and ROC AUC on the validation set.

## Training CLI

Run the training loop directly from the terminal with custom directories and parameters:

```bash
python -m src.train_engine \
    --train_dir path/to/train \
    --val_dir path/to/val \
    --epochs 10 \
    --batch_size 32 \
    --num_classes 1 \
    --use_transfer_learning \
    --base_model_name MobileNetV2 \
    --learning_rate 0.001 \
    --dropout_rate 0.5 \
    --save_model_path saved_models/pneumonia_cnn_v1.keras \
    --checkpoint_path saved_models/best_pneumonia_cnn.keras \
    --resume_checkpoint saved_models/best_pneumonia_cnn.keras \
    --plot_path training_history.png \
    --cm_path reports/confusion_matrix_val.png \
    --rotation_range 20 \
    --brightness_range 0.7 1.3 \
    --contrast_range 0.0 \
    --zoom_range 0.2 \
    --random_flip horizontal \
    --mlflow_experiment pneumonia-detector \
    --mlflow_run_name test_run \
    --mlflow_tracking_uri http://localhost:5000 \
    --history_csv training_history.csv \
    --seed 42 \
    --early_stopping_patience 10 \
    --reduce_lr_factor 0.2 \
    --reduce_lr_patience 5 \
    --reduce_lr_min_lr 0.00001 \
    --class_weights 1.0 1.0
```

The optional `--seed` flag ensures reproducible training by setting NumPy,
Python and TensorFlow random seeds.

Use `--mlflow_tracking_uri` if your MLflow server is not running on the default
local URI.

The `--history_csv` option saves the raw training metrics to a CSV file for
further analysis.

`--early_stopping_patience` controls how many epochs with no
validation loss improvement are allowed before training stops.
`--reduce_lr_factor`, `--reduce_lr_patience` and `--reduce_lr_min_lr`
configure the learning rate schedule applied when the validation loss
plateaus.
`--learning_rate` sets the initial optimizer learning rate.
`--dropout_rate` sets the dropout rate applied before the final output layer.
`--class_weights` overrides automatic class weight calculation. Provide one
weight per class.
`--num_classes` determines how many output classes the model predicts. Use `1` for
binary classification or a higher value for multi-class problems.

The augmentation parameters (`--rotation_range`, `--brightness_range`,
`--contrast_range`, `--zoom_range`, `--random_flip`) control how training images
are randomly transformed to improve generalization.

Use `--use_dummy_data` (the default) to quickly test the pipeline with a small generated dataset.
`--use_attention_model` selects an architecture with Squeeze-and-Excitation blocks.
`--base_model_name` chooses the backbone for transfer learning.
`--trainable_base_layers` controls how many base layers are unfrozen during fine-tuning.
`--fine_tune_epochs` and `--fine_tune_lr` configure the second training stage.
`--save_model_path` and `--checkpoint_path` set where models are stored.
`--resume_checkpoint` lets you continue training from a previous checkpoint.
`--plot_path` and `--cm_path` change the artifact output locations.

## Grad-CAM Visualization

`src/predict_utils.py` now exposes a small command-line interface to create Grad-CAM overlays. It loads the model, processes the image and saves an overlay showing which regions contributed most to the prediction.

Example usage:

```bash
python -m src.predict_utils \
    --model_path saved_models/pneumonia_cnn_v1.keras \
    --img_path path/to/image.jpg \
    --last_conv_layer_name conv_pw_13_relu \
    --output_path gradcam.png \
    --img_size 150 150
```

## Batch Inference

`src/inference.py` provides a simple CLI for running a trained model on a directory of images.
It outputs a CSV file with predictions for each image.

```bash
python -m src.inference \
    --model_path saved_models/pneumonia_cnn_v1.keras \
    --data_dir path/to/images \
    --output_csv preds.csv \
    --num_classes 1
```
For multi-class models, set ``--num_classes`` accordingly. The resulting CSV
contains a ``prediction`` column with the predicted class index and additional
``prob_i`` columns with per-class probabilities.

## Evaluate Predictions

`src/evaluate.py` can compute precision, recall, F1-score and ROC AUC from a CSV of model predictions and optional labels. It also saves a confusion matrix plot.

```bash
python -m src.evaluate \
    --pred_csv preds.csv \
    --label_csv labels.csv \
    --output_png eval_confusion.png \
    --normalize_cm \
    --threshold 0.5 \
    --metrics_csv metrics.csv \
    --num_classes 1
```
The optional `--normalize_cm` flag normalizes the confusion matrix by the number of true samples in each class. Use `--threshold` to change the probability threshold used when converting predictions to labels. Set `--metrics_csv` to save the computed metrics as a CSV file. When evaluating multi-class predictions, specify `--num_classes` and ensure the CSV contains probability columns named `prob_0`, `prob_1`, etc.

## Experiment Tracking

`train_engine.py` now logs parameters, metrics and artifacts using MLflow. Run the script and then view the recorded runs with:

```bash
mlflow ui
```

## Attention-based Model

The training engine can optionally build a CNN incorporating Squeeze-and-Excitation
blocks for attention. Enable this architecture via the `--use_attention_model`
flag when running the training CLI.

## Additional CLI Examples

For a concise reference of the available command-line interfaces, see [API_USAGE_GUIDE.md](API_USAGE_GUIDE.md). It summarizes training, inference, Grad-CAM and evaluation commands.
The guide also shows how to print the package version using the `cxr-version` command (or `python -m src.version_cli`).
It now includes a dataset statistics tool for counting images per class using
`cxr-dataset-stats`. The command accepts a custom list of file extensions via
`--extensions` (case-insensitive and dot optional). Counts saved via
`--json_output` or `--csv_output` are sorted alphabetically by class name by default.
An optional `--sort_by count` flag sorts results by descending count instead.
`--plot_png` saves a bar chart visualizing the class distribution (requires `matplotlib`, install with `pip install matplotlib`).
The PNG's parent directory must already exist. The tool exits with an error if the provided path does not exist
or is not a directory.


## Continuous Integration

Every pull request triggers a GitHub Actions workflow that installs project dependencies and runs code quality checks. The pipeline executes `ruff` for linting, `bandit` for a security scan and `pytest` for the test suite.
