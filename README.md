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
    --use_transfer_learning \
    --base_model_name MobileNetV2
```

Use `--use_dummy_data` (the default) to quickly test the pipeline with a small generated dataset.

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
    --output_csv preds.csv
```

## Evaluate Predictions

`src/evaluate.py` can compute precision, recall, F1-score and ROC AUC from a CSV of model predictions and optional labels. It also saves a confusion matrix plot.

```bash
python -m src.evaluate \
    --pred_csv preds.csv \
    --label_csv labels.csv \
    --output_png eval_confusion.png
```

## Experiment Tracking

`train_engine.py` now logs parameters, metrics and artifacts using MLflow. Run the script and then view the recorded runs with:

```bash
mlflow ui
```

## Attention-based Model

The training engine can optionally build a CNN incorporating Squeeze-and-Excitation blocks for attention.
Set `USE_ATTENTION_MODEL = True` in `train_engine.py` to enable this architecture.
