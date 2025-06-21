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

## Training the Model

Run `train_engine.py` to train the classifier. Choose the model architecture with `--model_type` (defaults to `simple`):

```bash
python src/train_engine.py --model_type mobilenet
```

Available options are `simple`, `mobilenet`, and `vgg`.

## Visualizing Predictions with Grad-CAM

After training a model, generate a Grad-CAM heatmap overlay. The last convolutional layer can be supplied via `--layer`. If omitted, the script attempts to automatically detect it:

```bash
python src/grad_cam.py --model saved_models/pneumonia_cnn_v1.keras \
  --image path/to/xray.jpg --output overlay.png

# explicitly specify the layer if automatic detection fails
python src/grad_cam.py --model saved_models/pneumonia_cnn_v1.keras \
  --image path/to/xray.jpg --layer last_conv_layer_name --output overlay.png
```

Replace `last_conv_layer_name` with the name of your model's last convolutional layer.

Alternatively, you can call the helper in `src/predict_utils.py` which now also
prints the predicted probability. By default it saves an overlay image unless
`--no_overlay` is specified:

```bash
python src/predict_utils.py --model saved_models/pneumonia_cnn_v1.keras \
  --image path/to/xray.jpg

# disable Grad-CAM generation
python src/predict_utils.py --model saved_models/pneumonia_cnn_v1.keras \
  --image path/to/xray.jpg --no_overlay
```

The console output includes the predicted pneumonia probability and, when
enabled, the location of the saved overlay.

## Running the Prediction API

To serve the trained model as a simple Flask API, run:

```bash
python src/serve_model.py
```

Send a POST request to `http://localhost:5000/predict` with an image file using
`curl`:

```bash
curl -X POST -F file=@path/to/xray.jpg http://localhost:5000/predict
```

## Evaluating a Saved Model

After training, you can assess model performance on a directory of test images using `evaluate_model.py`.
You may specify the input image size and optionally save a confusion matrix plot:

```bash
python src/evaluate_model.py --model saved_models/pneumonia_cnn_v1.keras \
  --data path/to/test_dir --batch_size 16 \
  --target_size 150 150 --confusion_matrix cm.png
```

The script prints overall accuracy, a classification report, and a confusion matrix.
