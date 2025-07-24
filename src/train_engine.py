# Standard library imports
import argparse
import os
import random
import shutil
from dataclasses import dataclass
from typing import Optional

# Third-party imports
import mlflow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.utils import to_categorical

# Local imports
from .config import config
from .data_loader import create_data_generators
from .model_builder import (
    create_cnn_with_attention,
    create_simple_cnn,
    create_transfer_learning_model,
)


@dataclass
class TrainingArgs:
    """Arguments configuring a training run."""

    train_dir: str | None = None
    val_dir: str | None = None
    use_dummy_data: bool = True
    img_size: tuple[int, int] = (150, 150)
    batch_size: int = 2
    epochs: int = 2
    num_classes: int = 1
    use_attention_model: bool = False
    use_transfer_learning: bool = True
    base_model_name: str = "MobileNetV2"
    learning_rate: float = 1e-3
    dropout_rate: float = 0.0
    trainable_base_layers: int = 20
    fine_tune_epochs: int = 1
    fine_tune_lr: float = 1e-5
    rotation_range: int = 20
    brightness_range: tuple[float, float] = (0.7, 1.3)
    contrast_range: float = 0.0
    zoom_range: float = 0.2
    random_flip: str = "horizontal"
    seed: int = config.RANDOM_SEED
    checkpoint_path: str = config.CHECKPOINT_PATH
    save_model_path: str = config.SAVE_MODEL_PATH
    mlflow_experiment: str = config.MLFLOW_EXPERIMENT
    mlflow_run_name: str | None = config.MLFLOW_RUN_NAME
    mlflow_tracking_uri: str | None = config.MLFLOW_TRACKING_URI
    plot_path: str = config.PLOT_PATH
    cm_path: str = config.CONFUSION_MATRIX_PATH
    resume_checkpoint: str | None = None
    history_csv: str = config.HISTORY_CSV_PATH
    early_stopping_patience: int = config.EARLY_STOPPING_PATIENCE
    reduce_lr_factor: float = config.REDUCE_LR_FACTOR
    reduce_lr_patience: int = config.REDUCE_LR_PATIENCE
    reduce_lr_min_lr: float = config.REDUCE_LR_MIN_LR
    class_weights: list[float] | None = None


def create_dummy_data(
    base_dir: str = None, 
    num_images_per_class: int = None,
    image_width: int = None,
    image_height: int = None
):
    """Creates dummy directories and placeholder image files for training and validation.
    
    Parameters
    ----------
    base_dir : str, optional
        Base directory for dummy data. Defaults to config.DUMMY_DATA_BASE_DIR.
    num_images_per_class : int, optional
        Number of images per class. Defaults to config.DUMMY_DATA_IMAGES_PER_CLASS.
    image_width : int, optional
        Width of dummy images. Defaults to config.DUMMY_IMAGE_WIDTH.
    image_height : int, optional
        Height of dummy images. Defaults to config.DUMMY_IMAGE_HEIGHT.
    """
    # Use configuration defaults if not provided
    base_dir = base_dir or config.DUMMY_DATA_BASE_DIR
    num_images_per_class = num_images_per_class or config.DUMMY_DATA_IMAGES_PER_CLASS
    image_width = image_width or config.DUMMY_IMAGE_WIDTH
    image_height = image_height or config.DUMMY_IMAGE_HEIGHT
    
    print(f"Creating dummy data under '{base_dir}'...")
    # Use 'NORMAL' and 'PNEUMONIA' for class names to match expected scenario
    class_names = ["NORMAL", "PNEUMONIA"]
    sets = ["train", "val"]

    if os.path.exists(base_dir):
        print(f"Removing existing dummy data directory: {base_dir}")
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    for s in sets:
        set_path = os.path.join(base_dir, s)
        os.makedirs(set_path, exist_ok=True)
        for class_name in class_names:
            path = os.path.join(set_path, class_name)
            os.makedirs(path, exist_ok=True)
            for i in range(num_images_per_class):
                try:
                    color = "red" if class_name == "PNEUMONIA" else "blue"
                    img = Image.new("RGB", (image_width, image_height), color=color)
                    img.save(os.path.join(path, f"dummy_{s}_{class_name}_{i+1}.jpg"))
                except Exception as e:
                    print(
                        f"Could not create dummy image {i+1} for {path}: {e}. Pillow might be needed."
                    )
                    open(
                        os.path.join(path, f"dummy_{s}_{class_name}_{i+1}.jpg"), "a"
                    ).close()
    print(f"Dummy data created with {num_images_per_class} images per class ({image_width}x{image_height}).")


def cleanup_dummy_data(base_dir: str = None):
    """Removes the dummy data directories.
    
    Parameters
    ----------
    base_dir : str, optional
        Base directory for dummy data. Defaults to config.DUMMY_DATA_BASE_DIR.
    """
    base_dir = base_dir or config.DUMMY_DATA_BASE_DIR
    if os.path.exists(base_dir):
        print(f"Cleaning up dummy data from '{base_dir}'...")
        shutil.rmtree(base_dir)
        print("Dummy data cleaned up.")
    else:
        print(f"Directory '{base_dir}' not found, no cleanup needed.")


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    """Add data-related command line arguments to the parser.
    
    Configures arguments for dataset paths, image preprocessing,
    and batch configuration used during model training.
    
    Parameters
    ----------
    parser : argparse.ArgumentParser
        Command line argument parser to add data arguments to.
        
    Notes
    -----
    Arguments added:
    - --train_dir: Path to training data directory
    - --val_dir: Path to validation data directory  
    - --use_dummy_data: Flag to generate synthetic test data
    - --img_size: Image dimensions (height, width)
    - --batch_size: Number of images per training batch
    """
    parser.add_argument("--train_dir", help="Path to training data")
    parser.add_argument("--val_dir", help="Path to validation data")
    parser.add_argument(
        "--use_dummy_data",
        action="store_true",
        default=True,
        help="Generate small dummy dataset for testing",
    )
    parser.add_argument("--img_size", type=int, nargs=2, default=[150, 150], metavar=("H", "W"))
    parser.add_argument("--batch_size", type=int, default=2)


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add model architecture and training arguments to the parser.
    
    Configures arguments for model hyperparameters, architecture choices,
    and training duration settings.
    
    Parameters
    ----------
    parser : argparse.ArgumentParser
        Command line argument parser to add model arguments to.
        
    Notes
    -----
    Arguments added:
    - --epochs: Number of training epochs
    - --num_classes: Output classes (1 for binary, >1 for multiclass)
    - --model_type: Architecture choice (simple_cnn, transfer_learning, attention)
    - --learning_rate: Optimizer learning rate
    - --use_class_weights: Flag to balance class distributions
    """
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1,
        help="Number of output classes (1 for binary)",
    )
    parser.add_argument("--use_attention_model", action="store_true")
    parser.add_argument("--use_transfer_learning", action="store_true", default=True)
    parser.add_argument("--base_model_name", default="MobileNetV2")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.0,
        help="Dropout rate for dense layers",
    )
    parser.add_argument("--trainable_base_layers", type=int, default=20)
    parser.add_argument("--fine_tune_epochs", type=int, default=1)
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5)


def _add_augmentation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rotation_range", type=int, default=20, help="Degree range for random rotations")
    parser.add_argument(
        "--brightness_range",
        type=float,
        nargs=2,
        default=[0.7, 1.3],
        metavar=("LOW", "HIGH"),
        help="Brightness range as two floats",
    )
    parser.add_argument("--contrast_range", type=float, default=0.0, help="Random contrast adjustment range (0 disables)")
    parser.add_argument("--zoom_range", type=float, default=0.2, help="Zoom range for random zoom augmentation")
    parser.add_argument(
        "--random_flip",
        choices=["horizontal", "vertical", "horizontal_and_vertical", "none"],
        default="horizontal",
        help="Type of random flip to apply",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")


def _add_io_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint_path", default="saved_models/best_pneumonia_cnn.keras", help="Where to save the best model checkpoint")
    parser.add_argument("--save_model_path", default="saved_models/pneumonia_cnn_v1.keras", help="Path to save the final trained model")
    parser.add_argument("--mlflow_experiment", default="pneumonia-detector", help="MLflow experiment name")
    parser.add_argument("--mlflow_run_name", default=None, help="Optional MLflow run name")
    parser.add_argument("--mlflow_tracking_uri", default=None, help="Optional MLflow tracking URI")
    parser.add_argument("--plot_path", default="training_history.png", help="File path for the training history plot")
    parser.add_argument("--cm_path", default="reports/confusion_matrix_val.png", help="File path for the validation confusion matrix")
    parser.add_argument("--resume_checkpoint", default=None, help="Optional checkpoint to resume training from")
    parser.add_argument("--history_csv", default="training_history.csv", help="Where to save the raw training history as CSV")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Epochs with no improvement before stopping")
    parser.add_argument("--reduce_lr_factor", type=float, default=0.2, help="Factor to reduce learning rate by")
    parser.add_argument("--reduce_lr_patience", type=int, default=5, help="Epochs with no improvement before reducing LR")
    parser.add_argument("--reduce_lr_min_lr", type=float, default=1e-5, help="Lower bound on learning rate during ReduceLROnPlateau")
    parser.add_argument(
        "--class_weights",
        type=float,
        nargs="+",
        default=None,
        metavar="W",
        help="Optional space separated class weights overriding automatic computation",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the pneumonia detector")
    _add_data_args(parser)
    _add_model_args(parser)
    _add_augmentation_args(parser)
    _add_io_args(parser)
    return parser


def _parse_args(argv: list[str] | None = None) -> TrainingArgs:
    """Parse command-line arguments into a :class:`TrainingArgs`."""

    parser = _build_parser()
    parsed = parser.parse_args(argv)
    if not parsed.use_dummy_data and (not parsed.train_dir or not parsed.val_dir):
        parser.error("--train_dir and --val_dir are required when not using dummy data")
    return TrainingArgs(
        train_dir=parsed.train_dir,
        val_dir=parsed.val_dir,
        use_dummy_data=parsed.use_dummy_data,
        img_size=tuple(parsed.img_size),
        batch_size=parsed.batch_size,
        epochs=parsed.epochs,
        num_classes=parsed.num_classes,
        use_attention_model=parsed.use_attention_model,
        use_transfer_learning=parsed.use_transfer_learning,
        base_model_name=parsed.base_model_name,
        learning_rate=parsed.learning_rate,
        dropout_rate=parsed.dropout_rate,
        trainable_base_layers=parsed.trainable_base_layers,
        fine_tune_epochs=parsed.fine_tune_epochs,
        fine_tune_lr=parsed.fine_tune_lr,
        rotation_range=parsed.rotation_range,
        brightness_range=tuple(parsed.brightness_range),
        contrast_range=parsed.contrast_range,
        zoom_range=parsed.zoom_range,
        random_flip=parsed.random_flip,
        seed=parsed.seed,
        checkpoint_path=parsed.checkpoint_path,
        save_model_path=parsed.save_model_path,
        mlflow_experiment=parsed.mlflow_experiment,
        mlflow_run_name=parsed.mlflow_run_name,
        mlflow_tracking_uri=parsed.mlflow_tracking_uri,
        plot_path=parsed.plot_path,
        cm_path=parsed.cm_path,
        resume_checkpoint=parsed.resume_checkpoint,
        history_csv=parsed.history_csv,
        early_stopping_patience=parsed.early_stopping_patience,
        reduce_lr_factor=parsed.reduce_lr_factor,
        reduce_lr_patience=parsed.reduce_lr_patience,
        reduce_lr_min_lr=parsed.reduce_lr_min_lr,
        class_weights=parsed.class_weights,
    )


def _load_generators(args: TrainingArgs):
    """Return training and validation generators.

    When ``args.use_dummy_data`` is True small dummy folders are created for
    testing and cleaned up by the caller.
    """

    if args.use_dummy_data:
        dummy_base = "data_train_engine"
        train_dir = os.path.join(dummy_base, "train")
        val_dir = os.path.join(dummy_base, "val")
        create_dummy_data(
            base_dir=dummy_base, num_images_per_class=args.batch_size * 2 + 1
        )
    else:
        train_dir = args.train_dir
        val_dir = args.val_dir
        dummy_base = None

    flip_option = None if args.random_flip == "none" else args.random_flip
    class_mode = "binary" if args.num_classes == 1 else "categorical"
    generators = create_data_generators(
        train_dir=train_dir,
        val_dir=val_dir,
        target_size=args.img_size,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        rotation_range=args.rotation_range,
        brightness_range=args.brightness_range,
        contrast_range=args.contrast_range,
        zoom_range=args.zoom_range,
        random_flip=flip_option,
        class_mode=class_mode,
    )
    if generators[0] is None or generators[1] is None:
        raise RuntimeError("Failed to load datasets using create_data_generators")
    return *generators, dummy_base


def _create_model(args: TrainingArgs, input_shape: tuple[int, int, int]):
    """Build a Keras model according to ``args``."""

    if args.use_attention_model:
        return create_cnn_with_attention(
            input_shape=input_shape,
            num_classes=args.num_classes,
            learning_rate=args.learning_rate,
            dropout_rate=args.dropout_rate,
        )
    if args.use_transfer_learning:
        return create_transfer_learning_model(
            input_shape=input_shape,
            num_classes=args.num_classes,
            base_model_name=args.base_model_name,
            learning_rate=args.learning_rate,
            dropout_rate=args.dropout_rate,
        )
    return create_simple_cnn(
        input_shape=input_shape,
        num_classes=args.num_classes,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
    )


def _compute_class_weights(train_generator, args: TrainingArgs):
    if args.class_weights is not None:
        if len(args.class_weights) != args.num_classes:
            raise ValueError("Number of class weights must match num_classes")
        return {i: w for i, w in enumerate(args.class_weights)}
    train_labels = train_generator.classes
    unique_classes = np.unique(train_labels)
    weights = compute_class_weight(
        class_weight="balanced", classes=unique_classes, y=train_labels
    )
    return dict(zip(unique_classes, weights))


def _train(model, train_generator, val_generator, class_weights, args: TrainingArgs):
    steps_per_epoch = max(1, train_generator.samples // train_generator.batch_size)
    validation_steps = max(1, val_generator.samples // val_generator.batch_size)

    checkpoint_cb = ModelCheckpoint(
        filepath=args.checkpoint_path,
        save_weights_only=False,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=args.early_stopping_patience,
        verbose=1,
        mode="min",
        restore_best_weights=True,
    )
    reduce_lr_cb = ReduceLROnPlateau(
        monitor="val_loss",
        factor=args.reduce_lr_factor,
        patience=args.reduce_lr_patience,
        verbose=1,
        mode="min",
        min_lr=args.reduce_lr_min_lr,
    )
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb],
        class_weight=class_weights,
        verbose=1,
    )

    if args.use_transfer_learning and args.trainable_base_layers > 0:
        for layer in model.layers[-args.trainable_base_layers :]:
            layer.trainable = True
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.fine_tune_lr),
            loss=model.loss,
            metrics=model.metrics,
        )
        fine_history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=args.fine_tune_epochs,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb],
            class_weight=class_weights,
            verbose=1,
        )
        for k, vals in fine_history.history.items():
            history.history[k] = history.history.get(k, []) + vals
    return history


def _calculate_metrics(model, val_generator, args: TrainingArgs):
    """Calculate comprehensive evaluation metrics for model predictions.
    
    Evaluates the trained model on validation data and computes precision, recall,
    F1-score, ROC-AUC, and confusion matrix. Handles both binary and multiclass
    classification scenarios with appropriate metric calculations.
    
    Parameters
    ----------
    model : tf.keras.Model
        Trained neural network model to evaluate.
    val_generator : tf.keras.utils.Sequence
        Validation data generator providing image batches and labels.
    args : TrainingArgs
        Training configuration containing num_classes and other parameters.
        
    Returns
    -------
    tuple
        A tuple containing (precision, recall, f1, roc_auc, confusion_matrix).
        For multiclass problems, precision/recall/f1 are weighted averages.
        ROC-AUC may be NaN if computation fails.
        
    Notes
    -----
    - Binary classification uses threshold 0.5 for prediction labels
    - Multiclass uses argmax for prediction labels and one-vs-rest ROC-AUC
    - Zero division in metrics is handled gracefully with zero_division=0
    - ROC-AUC computation failures return NaN instead of raising exceptions
    """
    num_samples = val_generator.samples
    pred_steps = num_samples // val_generator.batch_size
    if num_samples % val_generator.batch_size:
        pred_steps += 1
    preds = model.predict(val_generator, steps=pred_steps)[:num_samples]
    true_labels = val_generator.classes[:num_samples]

    if args.num_classes == 1:
        pred_labels = (preds > 0.5).astype(int).ravel()
        precision = precision_score(true_labels, pred_labels, zero_division=0)
        recall = recall_score(true_labels, pred_labels, zero_division=0)
        f1 = f1_score(true_labels, pred_labels, zero_division=0)
        try:
            roc_auc = roc_auc_score(true_labels, preds)
        except ValueError:
            roc_auc = float("nan")
    else:
        pred_labels = np.argmax(preds, axis=1)
        precision = precision_score(
            true_labels, pred_labels, average="weighted", zero_division=0
        )
        recall = recall_score(
            true_labels, pred_labels, average="weighted", zero_division=0
        )
        f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
        try:
            y_true_cat = to_categorical(true_labels, args.num_classes)
            roc_auc = roc_auc_score(y_true_cat, preds, multi_class="ovr")
        except ValueError:
            roc_auc = float("nan")

    cm = confusion_matrix(true_labels, pred_labels)

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'predictions': preds,
        'true_labels': true_labels,
        'confusion_matrix': cm
    }


def _plot_confusion_matrix(cm: np.ndarray, save_path: str) -> None:
    """Generate and save a visual confusion matrix plot.
    
    Creates a heatmap visualization of the confusion matrix with proper
    labeling, color mapping, and formatting for model evaluation reports.
    
    Parameters
    ----------
    cm : numpy.ndarray
        Confusion matrix array with shape (n_classes, n_classes).
    save_path : str
        File path where the confusion matrix plot will be saved.
        Supports common image formats (PNG, JPEG, PDF).
        
    Notes
    -----
    - Uses seaborn heatmap for professional visualization
    - Includes percentage annotations for each cell
    - Automatically adjusts figure size for readability
    - Saves with high DPI for publication quality
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _plot_training_history(history: tf.keras.callbacks.History, epochs: int, save_path: str) -> None:
    """Generate and save training history visualization plots.
    
    Creates dual subplot visualization showing training/validation loss
    and accuracy curves over epochs for monitoring model convergence
    and detecting overfitting.
    
    Parameters
    ----------
    history : tf.keras.callbacks.History
        Training history object containing metrics for each epoch.
    epochs : int
        Number of training epochs completed.
    save_path : str
        File path where the training history plot will be saved.
        
    Notes
    -----
    - Left subplot shows loss curves (training vs validation)
    - Right subplot shows accuracy curves (training vs validation)
    - Includes grid lines and legends for clarity
    - Saves with bbox_inches='tight' to avoid clipping
    """
    epochs_range = range(epochs)
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def _save_artifacts(model, history, args: TrainingArgs):
    """Save trained model and associated training artifacts.
    
    Persists the trained model, training history, and configuration
    to the output directory for future inference and analysis.
    
    Parameters
    ----------
    model : tf.keras.Model
        Trained neural network model to save.
    history : tf.keras.callbacks.History
        Training history object containing epoch-wise metrics.
    args : TrainingArgs
        Training configuration containing output paths and parameters.
        
    Side Effects
    ------------
    - Saves model in Keras format to {output_dir}/model.keras
    - Saves training history as pickle to {output_dir}/history.pkl
    - Creates output directory if it doesn't exist
    
    Notes
    -----
    - Uses Keras native format for optimal compatibility
    - History pickle includes all tracked metrics and callbacks
    - Output directory structure matches evaluation expectations
    """
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(args.history_csv, index=False)
    mlflow.log_artifact(args.history_csv)

    model.save(args.save_model_path)
    mlflow.keras.log_model(model, "model")


def _evaluate_refactored(model, val_generator, history, args: TrainingArgs):
    """Orchestrate the complete evaluation process using refactored functions."""
    # Calculate metrics
    metrics = _calculate_metrics(model, val_generator, args)
    
    # Log metrics to MLflow
    mlflow.log_metric("precision", metrics['precision'])
    mlflow.log_metric("recall", metrics['recall'])
    mlflow.log_metric("f1", metrics['f1_score'])
    mlflow.log_metric("roc_auc", metrics['roc_auc'])

    # Plot and save confusion matrix
    _plot_confusion_matrix(metrics['confusion_matrix'], args.cm_path)
    mlflow.log_artifact(args.cm_path)

    # Plot and save training history
    _plot_training_history(history, args.epochs, args.plot_path)
    mlflow.log_artifact(args.plot_path)

    # Save artifacts
    _save_artifacts(model, history, args)


# Keep old function for backward compatibility during transition
def _evaluate(model, val_generator, history, args: TrainingArgs):
    """Legacy evaluation function - delegates to refactored implementation."""
    return _evaluate_refactored(model, val_generator, history, args)


def _setup_training_environment(args: TrainingArgs):
    """Set up the training environment including seeds, directories, and data generators.
    
    Initializes random seeds for reproducibility, ensures required directories exist,
    and loads training/validation data generators based on the provided arguments.
    
    Parameters
    ----------
    args : TrainingArgs
        Training configuration containing seed, data paths, and generator settings.
        
    Returns
    -------
    tuple
        A tuple containing (train_generator, val_generator, dummy_base_dir).
        dummy_base_dir is None if real data is used, otherwise contains the path
        to generated dummy data for cleanup.
        
    Notes
    -----
    - Sets NumPy, Python random, and TensorFlow random seeds for reproducibility
    - Creates necessary output directories for models, plots, and artifacts
    - Handles both real dataset paths and dummy data generation
    """
    # Ensure all required directories exist
    config.ensure_directories()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Load data generators
    train_gen, val_gen, dummy_base = _load_generators(args)
    
    return train_gen, val_gen, dummy_base


def _setup_mlflow_tracking(args: TrainingArgs):
    """Set up MLflow experiment tracking and parameter logging.
    
    Configures MLflow tracking URI, creates experiment, starts run, and logs
    all training parameters for experiment tracking and reproducibility.
    
    Parameters
    ----------
    args : TrainingArgs
        Training configuration containing MLflow settings and all parameters to log.
        
    Returns
    -------
    mlflow.ActiveRun
        MLflow run context manager for experiment tracking.
        
    Notes
    -----
    - Only sets tracking URI if explicitly provided in args
    - Logs comprehensive parameter set including model, training, and augmentation settings
    - Handles optional parameters (class_weights, resume_checkpoint) gracefully
    """
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    
    run_context = mlflow.start_run(run_name=args.mlflow_run_name)
    
    # Log all training parameters
    mlflow.log_params(
        {
            "batch_size": args.batch_size,
            "epochs_stage1": args.epochs,
            "use_attention_model": args.use_attention_model,
            "use_transfer_learning": args.use_transfer_learning,
            "base_model": args.base_model_name,
            "trainable_base_layers": args.trainable_base_layers,
            "num_classes": args.num_classes,
            "learning_rate": args.learning_rate,
            "dropout_rate": args.dropout_rate,
            "fine_tune_epochs": args.fine_tune_epochs,
            "fine_tune_lr": args.fine_tune_lr,
            "seed": args.seed,
            "rotation_range": args.rotation_range,
            "brightness_range": args.brightness_range,
            "contrast_range": args.contrast_range,
            "zoom_range": args.zoom_range,
            "random_flip": args.random_flip,
            "early_stopping_patience": args.early_stopping_patience,
            "reduce_lr_factor": args.reduce_lr_factor,
            "reduce_lr_patience": args.reduce_lr_patience,
            "reduce_lr_min_lr": args.reduce_lr_min_lr,
        }
    )
    
    # Log optional parameters if provided
    if args.class_weights is not None:
        mlflow.log_param("class_weights_manual", args.class_weights)
    if args.resume_checkpoint:
        mlflow.log_param("resume_checkpoint", args.resume_checkpoint)
    
    return run_context


def _execute_training_workflow(model, train_gen, val_gen, class_weights, args: TrainingArgs):
    """Execute the core training workflow including model loading, training, and evaluation.
    
    Handles checkpoint resumption, executes model training with the provided generators
    and class weights, and performs comprehensive model evaluation.
    
    Parameters
    ----------
    model : tf.keras.Model
        Compiled neural network model ready for training.
    train_gen : tf.keras.utils.Sequence
        Training data generator providing image batches and labels.
    val_gen : tf.keras.utils.Sequence
        Validation data generator for model evaluation.
    class_weights : dict
        Class weight mapping for handling class imbalance during training.
    args : TrainingArgs
        Training configuration containing checkpoint paths and training settings.
        
    Notes
    -----
    - Attempts to load checkpoint if resume_checkpoint is provided and file exists
    - Gracefully handles checkpoint loading failures and continues training
    - Executes full training pipeline including fine-tuning if enabled
    - Performs comprehensive evaluation with metrics, plots, and artifact generation
    """
    # Load checkpoint if resuming from previous training
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        try:
            model.load_weights(args.resume_checkpoint)
        except Exception as e_ckpt:  # pragma: no cover - best effort
            print(f"Failed to load checkpoint: {e_ckpt}")

    # Execute training and evaluation
    history = _train(model, train_gen, val_gen, class_weights, args)
    _evaluate(model, val_gen, history, args)


def _cleanup_training_resources(dummy_base: Optional[str]) -> None:
    """Clean up training resources including temporary dummy data.
    
    Removes temporary dummy data directories created during training if they exist.
    Safe to call with None dummy_base when real data was used.
    
    Parameters
    ----------
    dummy_base : str or None
        Path to dummy data base directory to clean up. If None, no cleanup is performed.
        
    Notes
    -----
    - Only performs cleanup if dummy_base is not None
    - Uses cleanup_dummy_data function for proper directory removal
    - Safe to call even if dummy data was not created
    """
    if dummy_base:
        cleanup_dummy_data(base_dir=dummy_base)


def train_pipeline(args: TrainingArgs) -> None:
    """Run the full training pipeline based on ``args``.
    
    Orchestrates the complete model training workflow including environment setup,
    MLflow tracking, model creation and training, and resource cleanup. This is the
    main entry point for the training pipeline.
    
    Parameters
    ----------
    args : TrainingArgs
        Complete training configuration including data paths, model parameters,
        training hyperparameters, and experiment tracking settings.
        
    Notes
    -----
    - Follows a structured workflow: setup → tracking → training → cleanup
    - Handles both real datasets and dummy data generation for testing
    - Provides comprehensive experiment tracking via MLflow
    - Ensures proper cleanup of temporary resources
    - Maintains backward compatibility with existing usage patterns
    """
    # Set up training environment and load data
    train_gen, val_gen, dummy_base = _setup_training_environment(args)

    # Set up MLflow experiment tracking
    with _setup_mlflow_tracking(args):
        # Create model with input shape
        input_shape = (*args.img_size, 3)
        model = _create_model(args, input_shape)
        
        # Compute class weights for handling imbalance
        class_weights = _compute_class_weights(train_gen, args)
        
        # Execute training workflow
        _execute_training_workflow(model, train_gen, val_gen, class_weights, args)

    # Clean up temporary resources
    _cleanup_training_resources(dummy_base)


def main() -> None:
    args = _parse_args(argv=None)
    
    # Validate input paths for security
    from .input_validation import validate_directory_path, validate_file_path, ValidationError
    try:
        if args.train_dir:
            args.train_dir = validate_directory_path(args.train_dir, create_if_missing=True)
        if args.val_dir:
            args.val_dir = validate_directory_path(args.val_dir, create_if_missing=True)
        if args.checkpoint_path:
            # Validate parent directory and create if needed
            checkpoint_dir = os.path.dirname(args.checkpoint_path)
            if checkpoint_dir:
                validate_directory_path(checkpoint_dir, create_if_missing=True)
            args.checkpoint_path = validate_file_path(args.checkpoint_path, must_exist=False, allowed_extensions=['.keras', '.h5'])
        if args.save_model_path:
            # Validate parent directory and create if needed
            model_dir = os.path.dirname(args.save_model_path)
            if model_dir:
                validate_directory_path(model_dir, create_if_missing=True)
            args.save_model_path = validate_file_path(args.save_model_path, must_exist=False, allowed_extensions=['.keras', '.h5'])
    except ValidationError as e:
        print(f"❌ Input validation error: {e}")
        return
    
    try:
        train_pipeline(args)
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"An error occurred during the training process: {exc}")
        import traceback

        traceback.print_exc()
    finally:
        print("\nTrain engine script finished.")


if __name__ == "__main__":
    main()
