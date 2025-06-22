import argparse
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
                    color = 'red' if class_name == "PNEUMONIA" else 'blue'
                    img = Image.new('RGB', (60, 30), color=color)
                    img.save(os.path.join(path, f"dummy_{s}_{class_name}_{i+1}.jpg"))
                except Exception as e:
                    print(f"Could not create dummy image {i+1} for {path}: {e}. Pillow might be needed.")
                    open(os.path.join(path, f"dummy_{s}_{class_name}_{i+1}.jpg"), 'a').close()
    print("Dummy data created.")

def cleanup_dummy_data(base_dir="data_train_engine"): # Updated base_dir
    """Removes the dummy data directories."""
    if os.path.exists(base_dir):
        print(f"Cleaning up dummy data from '{base_dir}'...")
        shutil.rmtree(base_dir)
        print("Dummy data cleaned up.")
    else:
        print(f"Directory '{base_dir}' not found, no cleanup needed.")


def main():
    parser = argparse.ArgumentParser(description="Train the pneumonia detector")
    parser.add_argument("--train_dir", help="Path to training data")
    parser.add_argument("--val_dir", help="Path to validation data")
    parser.add_argument("--use_dummy_data", action="store_true", default=True,
                        help="Generate small dummy dataset for testing")
    parser.add_argument("--img_size", type=int, nargs=2, default=[150, 150], metavar=("H", "W"))
    parser.add_argument("--batch_size", type=int, default=2)
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
    parser.add_argument(
        "--rotation_range",
        type=int,
        default=20,
        help="Degree range for random rotations",
    )
    parser.add_argument(
        "--brightness_range",
        type=float,
        nargs=2,
        default=[0.7, 1.3],
        metavar=("LOW", "HIGH"),
        help="Brightness range as two floats",
    )
    parser.add_argument(
        "--contrast_range",
        type=float,
        default=0.0,
        help="Random contrast adjustment range (0 disables)",
    )
    parser.add_argument(
        "--zoom_range",
        type=float,
        default=0.2,
        help="Zoom range for random zoom augmentation",
    )
    parser.add_argument(
        "--random_flip",
        choices=["horizontal", "vertical", "horizontal_and_vertical", "none"],
        default="horizontal",
        help="Type of random flip to apply",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="saved_models/best_pneumonia_cnn.keras",
        help="Where to save the best model checkpoint",
    )
    parser.add_argument(
        "--save_model_path",
        default="saved_models/pneumonia_cnn_v1.keras",
        help="Path to save the final trained model",
    )
    parser.add_argument(
        "--mlflow_experiment",
        default="pneumonia-detector",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--mlflow_run_name",
        default=None,
        help="Optional MLflow run name",
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        default=None,
        help="Optional MLflow tracking URI",
    )
    parser.add_argument(
        "--plot_path",
        default="training_history.png",
        help="File path for the training history plot",
    )
    parser.add_argument(
        "--cm_path",
        default="reports/confusion_matrix_val.png",
        help="File path for the validation confusion matrix",
    )
    parser.add_argument(
        "--resume_checkpoint",
        default=None,
        help="Optional checkpoint to resume training from",
    )
    parser.add_argument(
        "--history_csv",
        default="training_history.csv",
        help="Where to save the raw training history as CSV",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Epochs with no improvement before stopping",
    )
    parser.add_argument(
        "--reduce_lr_factor",
        type=float,
        default=0.2,
        help="Factor to reduce learning rate by",
    )
    parser.add_argument(
        "--reduce_lr_patience",
        type=int,
        default=5,
        help="Epochs with no improvement before reducing LR",
    )
    parser.add_argument(
        "--reduce_lr_min_lr",
        type=float,
        default=1e-5,
        help="Lower bound on learning rate during ReduceLROnPlateau",
    )
    parser.add_argument(
        "--class_weights",
        type=float,
        nargs='+',
        default=None,
        metavar='W',
        help="Optional space separated class weights overriding automatic computation",
    )
    args = parser.parse_args()

    IMG_HEIGHT, IMG_WIDTH = args.img_size
    IMAGE_SIZE_TUPLE = (IMG_HEIGHT, IMG_WIDTH)
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    NUM_CLASSES = args.num_classes

    USE_ATTENTION_MODEL = args.use_attention_model
    USE_TRANSFER_LEARNING = args.use_transfer_learning
    BASE_MODEL_NAME = args.base_model_name
    LEARNING_RATE = args.learning_rate
    TRAINABLE_BASE_LAYERS = args.trainable_base_layers
    DROPOUT_RATE = args.dropout_rate
    FINE_TUNE_EPOCHS = args.fine_tune_epochs
    FINE_TUNE_LR = args.fine_tune_lr
    MLFLOW_TRACKING_URI = args.mlflow_tracking_uri
    HISTORY_CSV = args.history_csv
    SEED = args.seed
    ROTATION_RANGE = args.rotation_range
    BRIGHTNESS_RANGE = args.brightness_range
    CONTRAST_RANGE = args.contrast_range
    ZOOM_RANGE = args.zoom_range
    RANDOM_FLIP = args.random_flip
    EARLY_STOPPING_PATIENCE = args.early_stopping_patience
    REDUCE_LR_FACTOR = args.reduce_lr_factor
    REDUCE_LR_PATIENCE = args.reduce_lr_patience
    REDUCE_LR_MIN_LR = args.reduce_lr_min_lr
    RESUME_CHECKPOINT = args.resume_checkpoint
    CLASS_WEIGHTS = args.class_weights
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    if args.use_dummy_data:
        dummy_data_base_dir = "data_train_engine"
        train_dir = os.path.join(dummy_data_base_dir, "train")
        val_dir = os.path.join(dummy_data_base_dir, "val")
        create_dummy_data(base_dir=dummy_data_base_dir, num_images_per_class=BATCH_SIZE * 2 + 1)
    else:
        if not args.train_dir or not args.val_dir:
            parser.error("--train_dir and --val_dir are required when not using dummy data")
        train_dir = args.train_dir
        val_dir = args.val_dir
        dummy_data_base_dir = None


    try:
        # --- Load Data using create_data_generators ---
        print("Loading training and validation data using create_data_generators...")
        flip_option = None if args.random_flip == 'none' else args.random_flip
        class_mode = 'binary' if NUM_CLASSES == 1 else 'categorical'
        train_generator, validation_generator = create_data_generators(
            train_dir=train_dir,
            val_dir=val_dir,
            target_size=IMAGE_SIZE_TUPLE,
            train_batch_size=BATCH_SIZE,
            val_batch_size=BATCH_SIZE,
            rotation_range=args.rotation_range,
            brightness_range=args.brightness_range,
            contrast_range=args.contrast_range,
            zoom_range=args.zoom_range,
            random_flip=flip_option,
            class_mode=class_mode
        )

        if train_generator is None or validation_generator is None:
            raise RuntimeError(
                "Failed to load datasets using create_data_generators. Check paths and data_loader.py."
            )

        if MLFLOW_TRACKING_URI:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(args.mlflow_experiment)
        with mlflow.start_run(run_name=args.mlflow_run_name):
            mlflow.log_param("batch_size", BATCH_SIZE)
            mlflow.log_param("epochs_stage1", EPOCHS)
            mlflow.log_param("use_attention_model", USE_ATTENTION_MODEL)
            mlflow.log_param("use_transfer_learning", USE_TRANSFER_LEARNING)
            mlflow.log_param("base_model", BASE_MODEL_NAME)
            mlflow.log_param("trainable_base_layers", TRAINABLE_BASE_LAYERS)
            mlflow.log_param("num_classes", NUM_CLASSES)
            mlflow.log_param("learning_rate", LEARNING_RATE)
            mlflow.log_param("dropout_rate", DROPOUT_RATE)
            mlflow.log_param("fine_tune_epochs", FINE_TUNE_EPOCHS)
            mlflow.log_param("fine_tune_lr", FINE_TUNE_LR)
            mlflow.log_param("seed", SEED)
            mlflow.log_param("rotation_range", args.rotation_range)
            mlflow.log_param("brightness_range", args.brightness_range)
            mlflow.log_param("contrast_range", args.contrast_range)
            mlflow.log_param("zoom_range", args.zoom_range)
            mlflow.log_param("random_flip", args.random_flip)
            mlflow.log_param("early_stopping_patience", EARLY_STOPPING_PATIENCE)
            mlflow.log_param("reduce_lr_factor", REDUCE_LR_FACTOR)
            mlflow.log_param("reduce_lr_patience", REDUCE_LR_PATIENCE)
            mlflow.log_param("reduce_lr_min_lr", REDUCE_LR_MIN_LR)
            if CLASS_WEIGHTS is not None:
                mlflow.log_param("class_weights_manual", CLASS_WEIGHTS)
            if RESUME_CHECKPOINT:
                mlflow.log_param("resume_checkpoint", RESUME_CHECKPOINT)

            # --- Model Creation ---
            # Input shape for CNN (height, width, channels)
            # ImageDataGenerator provides 3 channels (RGB) by default
            input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
            print(f"Using input shape for model: {input_shape}")
    
            if USE_ATTENTION_MODEL:
                print("Creating CNN model with attention...")
                model = create_cnn_with_attention(
                    input_shape=input_shape,
                    num_classes=NUM_CLASSES,
                    learning_rate=LEARNING_RATE,
                    dropout_rate=DROPOUT_RATE,
                )
            elif USE_TRANSFER_LEARNING:
                print(f"Creating transfer learning model with base {BASE_MODEL_NAME}...")
                model = create_transfer_learning_model(
                    input_shape=input_shape,
                    num_classes=NUM_CLASSES,
                    base_model_name=BASE_MODEL_NAME,
                    learning_rate=LEARNING_RATE,
                    dropout_rate=DROPOUT_RATE,
                )
            else:
                model = create_simple_cnn(
                    input_shape=input_shape,
                    num_classes=NUM_CLASSES,
                    learning_rate=LEARNING_RATE,
                    dropout_rate=DROPOUT_RATE,
                )
            print("\nModel Summary:")
            model.summary()

            if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
                print(f"Loading weights from {RESUME_CHECKPOINT} to resume training")
                try:
                    model.load_weights(RESUME_CHECKPOINT)
                except Exception as e_ckpt:
                    print(f"Failed to load checkpoint: {e_ckpt}")

            # --- Model Training ---
            print("\nStarting model training with generators...")
            
            # Calculate steps per epoch
            num_train_samples = train_generator.samples
            num_val_samples = validation_generator.samples
            
            steps_per_epoch = num_train_samples // train_generator.batch_size
            validation_steps = num_val_samples // validation_generator.batch_size
            
            # Ensure steps are at least 1, especially for small dummy datasets
            # Updated to use ternary operator for conciseness
            steps_per_epoch = train_generator.samples // train_generator.batch_size if train_generator.samples > 0 else 1
            validation_steps = validation_generator.samples // validation_generator.batch_size if validation_generator.samples > 0 else 1
            
            # --- Calculate Class Weights ---
            print("Calculating class weights...")
            if CLASS_WEIGHTS is not None:
                if len(CLASS_WEIGHTS) != NUM_CLASSES:
                    raise ValueError(
                        "Number of class weights must match num_classes"
                    )
                class_weights_dict = {
                    i: w for i, w in enumerate(CLASS_WEIGHTS)
                }
                print(f"Using provided class weights: {class_weights_dict}")
            else:
                train_labels = train_generator.classes
                unique_classes = np.unique(train_labels)

                class_weights_array = compute_class_weight(
                    class_weight='balanced',
                    classes=unique_classes,
                    y=train_labels
                )
                class_weights_dict = dict(zip(unique_classes, class_weights_array))

                print(f"Train generator class indices: {train_generator.class_indices}")
                print(f"Unique classes found by np.unique: {unique_classes}")
                print(f"Calculated class weights: {class_weights_dict}")
    
    
            # --- Callbacks ---
            # Ensure 'saved_models' directory exists for ModelCheckpoint
            os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    
            checkpoint_filepath = args.checkpoint_path
            model_checkpoint_callback = ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=False,
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                verbose=1
            )
    
            early_stopping_callback = EarlyStopping(
                monitor='val_loss',
                patience=EARLY_STOPPING_PATIENCE,
                verbose=1,
                mode='min',
                restore_best_weights=True
            )
    
            reduce_lr_callback = ReduceLROnPlateau(
                monitor='val_loss',
                factor=REDUCE_LR_FACTOR,
                patience=REDUCE_LR_PATIENCE,
                verbose=1,
                mode='min',
                min_lr=REDUCE_LR_MIN_LR
            )
            
            callbacks_list = [model_checkpoint_callback, early_stopping_callback, reduce_lr_callback]
    
            history = model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=EPOCHS,
                validation_data=validation_generator,
                validation_steps=validation_steps,
                callbacks=callbacks_list,
                class_weight=class_weights_dict, # Pass the class weights here
                verbose=1
            )
            print("Model training completed.")
    
            if USE_TRANSFER_LEARNING and TRAINABLE_BASE_LAYERS > 0:
                print("\nStarting fine-tuning stage...")
                for layer in model.layers[-TRAINABLE_BASE_LAYERS:]:
                    layer.trainable = True
    
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
                    loss=model.loss,
                    metrics=model.metrics,
                )
                history_fine = model.fit(
                    train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=FINE_TUNE_EPOCHS,
                    validation_data=validation_generator,
                    validation_steps=validation_steps,
                    callbacks=callbacks_list,
                    class_weight=class_weights_dict,
                    verbose=1,
                )
                for key, vals in history_fine.history.items():
                    history.history[key] = history.history.get(key, []) + vals
    
            # --- Save Final Model ---
            # This saves the model at the end of training, regardless of ModelCheckpoint's best
            final_model_save_path = args.save_model_path
            os.makedirs(os.path.dirname(final_model_save_path), exist_ok=True)
    
            model.save(final_model_save_path)
            print(f"Final model saved successfully to {final_model_save_path}")
    
            # --- Evaluate Model (Precision, Recall, F1-score) ---
            print("\nEvaluating model on validation set...")
            
            # Ensure validation_generator is reset if it was used before (e.g. in model.fit)
            # and shuffle is False for consistent order. The create_data_generators sets shuffle=False for val.
            # validation_generator.reset() # Not strictly necessary if shuffle=False and steps cover all data
    
            num_val_samples_for_pred = validation_generator.samples
            val_pred_steps = (num_val_samples_for_pred // validation_generator.batch_size) + \
                             (1 if num_val_samples_for_pred % validation_generator.batch_size else 0)
    
            val_predictions = model.predict(validation_generator, steps=val_pred_steps)
            val_predictions = val_predictions[:num_val_samples_for_pred]
            val_true_labels = validation_generator.classes[:num_val_samples_for_pred]

            if NUM_CLASSES == 1:
                pred_labels = (val_predictions > 0.5).astype(int).ravel()
                precision = precision_score(val_true_labels, pred_labels, zero_division=0)
                recall = recall_score(val_true_labels, pred_labels, zero_division=0)
                f1 = f1_score(val_true_labels, pred_labels, zero_division=0)
                try:
                    roc_auc = roc_auc_score(val_true_labels, val_predictions)
                except ValueError:
                    roc_auc = float('nan')
            else:
                pred_labels = np.argmax(val_predictions, axis=1)
                precision = precision_score(val_true_labels, pred_labels, average="weighted", zero_division=0)
                recall = recall_score(val_true_labels, pred_labels, average="weighted", zero_division=0)
                f1 = f1_score(val_true_labels, pred_labels, average="weighted", zero_division=0)
                try:
                    y_true_cat = to_categorical(val_true_labels, NUM_CLASSES)
                    roc_auc = roc_auc_score(y_true_cat, val_predictions, multi_class="ovr")
                except ValueError:
                    roc_auc = float('nan')

            cm = confusion_matrix(val_true_labels, pred_labels)

            print(f"Validation Precision: {precision:.4f}")
            print(f"Validation Recall: {recall:.4f}")
            print(f"Validation F1-score: {f1:.4f}")
            print(f"Validation ROC AUC: {roc_auc:.4f}")

            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("roc_auc", roc_auc)

            os.makedirs(os.path.dirname(args.cm_path), exist_ok=True)
            plt.figure(figsize=(4, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            cm_path = args.cm_path
            plt.tight_layout()
            plt.savefig(cm_path)
            plt.close()
            print(f"Confusion matrix saved to {cm_path}")
            mlflow.log_artifact(cm_path)
    
            # --- Plotting Training History ---
            print("\nGenerating training history plot...")
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            epochs_range = range(EPOCHS)
    
            plt.figure(figsize=(12, 6)) # Adjusted figsize for better layout
    
            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, acc, label='Training Accuracy')
            plt.plot(epochs_range, val_acc, label='Validation Accuracy')
            plt.legend(loc='lower right')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
    
            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, loss, label='Training Loss')
            plt.plot(epochs_range, val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            
            plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
            
            plot_save_path = args.plot_path
            os.makedirs(os.path.dirname(plot_save_path) or '.', exist_ok=True)
            try:
                plt.savefig(plot_save_path)
                print(f"Training history plot saved to {plot_save_path}")
                mlflow.log_artifact(plot_save_path)
            except Exception as e_plot:
                print(f"Error saving plot: {e_plot}")

            history_df = pd.DataFrame(history.history)
            history_csv_path = HISTORY_CSV
            try:
                history_df.to_csv(history_csv_path, index=False)
                print(f"Training history CSV saved to {history_csv_path}")
                mlflow.log_artifact(history_csv_path)
            except Exception as e_csv:
                print(f"Error saving history CSV: {e_csv}")

            mlflow.keras.log_model(model, "model")


    except Exception as e:
        print(f"An error occurred during the training process: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- Cleanup ---
        if dummy_data_base_dir:
            cleanup_dummy_data(base_dir=dummy_data_base_dir)

    print("\nTrain engine script finished.")


if __name__ == "__main__":
    main()
