import argparse
from dataclasses import dataclass
import tensorflow as tf
import random
import os
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)
import shutil  # For directory cleanup
from PIL import Image  # For creating dummy images
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight  # Added for class imbalance
from tensorflow.keras.utils import to_categorical
import mlflow

# Local imports
from .data_loader import create_data_generators
from .model_builder import (
    create_simple_cnn,
    create_transfer_learning_model,
    create_cnn_with_attention,
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
    seed: int = 42
    checkpoint_path: str = "saved_models/best_pneumonia_cnn.keras"
    save_model_path: str = "saved_models/pneumonia_cnn_v1.keras"
    mlflow_experiment: str = "pneumonia-detector"
    mlflow_run_name: str | None = None
    mlflow_tracking_uri: str | None = None
    plot_path: str = "training_history.png"
    cm_path: str = "reports/confusion_matrix_val.png"
    resume_checkpoint: str | None = None
    history_csv: str = "training_history.csv"
    early_stopping_patience: int = 10
    reduce_lr_factor: float = 0.2
    reduce_lr_patience: int = 5
    reduce_lr_min_lr: float = 1e-5
    class_weights: list[float] | None = None


def create_dummy_data(base_dir="data_train_engine", num_images_per_class=5):
    """Creates dummy directories and placeholder image files for training and validation."""
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
                    img = Image.new("RGB", (60, 30), color=color)
                    img.save(os.path.join(path, f"dummy_{s}_{class_name}_{i+1}.jpg"))
                except Exception as e:
                    print(
                        f"Could not create dummy image {i+1} for {path}: {e}. Pillow might be needed."
                    )
                    open(
                        os.path.join(path, f"dummy_{s}_{class_name}_{i+1}.jpg"), "a"
                    ).close()
    print("Dummy data created.")


def cleanup_dummy_data(base_dir="data_train_engine"):  # Updated base_dir
    """Removes the dummy data directories."""
    if os.path.exists(base_dir):
        print(f"Cleaning up dummy data from '{base_dir}'...")
        shutil.rmtree(base_dir)
        print("Dummy data cleaned up.")
    else:
        print(f"Directory '{base_dir}' not found, no cleanup needed.")


def _add_data_args(parser: argparse.ArgumentParser) -> None:
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
    """Calculate evaluation metrics for model predictions."""
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


def _plot_confusion_matrix(cm, save_path):
    """Plot and save confusion matrix."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _plot_training_history(history, epochs, save_path):
    """Plot and save training history."""
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
    """Save model and training artifacts."""
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


def train_pipeline(args: TrainingArgs) -> None:
    """Run the full training pipeline based on ``args``."""

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    train_gen, val_gen, dummy_base = _load_generators(args)

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name=args.mlflow_run_name):
        input_shape = (*args.img_size, 3)
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
        if args.class_weights is not None:
            mlflow.log_param("class_weights_manual", args.class_weights)
        if args.resume_checkpoint:
            mlflow.log_param("resume_checkpoint", args.resume_checkpoint)

        model = _create_model(args, input_shape)
        if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
            try:
                model.load_weights(args.resume_checkpoint)
            except Exception as e_ckpt:  # pragma: no cover - best effort
                print(f"Failed to load checkpoint: {e_ckpt}")

        class_weights = _compute_class_weights(train_gen, args)
        history = _train(model, train_gen, val_gen, class_weights, args)
        _evaluate(model, val_gen, history, args)

    if dummy_base:
        cleanup_dummy_data(base_dir=dummy_base)


def main() -> None:
    args = _parse_args(argv=None)
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
