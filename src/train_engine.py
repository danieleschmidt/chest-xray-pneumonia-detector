import argparse
import tensorflow as tf
import os
import mlflow  # Ensure MLflow is imported
import mlflow.keras  # Ensure MLflow Keras integration is imported
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import shutil  # For directory cleanup
from PIL import Image  # For creating dummy images
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight  # Added for class imbalance


# Attempt to import from src, assuming the script is run from the project root
# or the src directory is in PYTHONPATH
try:
    from data_loader import create_data_generators  # Updated import
    from model_builder import create_simple_cnn, create_transfer_learning_model
except ImportError:
    # Fallback for direct execution or if src is not in PYTHONPATH
    # This might happen if the script is run directly from the src directory
    # or if the IDE/environment doesn't automatically add 'src' to sys.path
    print("Attempting fallback import for data_loader and model_builder.")
    try:
        from .data_loader import load_images_from_directory
        from .model_builder import create_simple_cnn, create_transfer_learning_model
    except ImportError as e:
        print(f"Error importing modules. Make sure 'src' is in PYTHONPATH or run from project root.")
        print(f"Details: {e}")
        # You might want to exit here or raise the error depending on desired behavior
        raise

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
                    img = Image.new('RGB', (60, 30), color = ('red' if class_name == "class_a" else 'blue'))
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train pneumonia detection model")
    parser.add_argument(
        "--model_type",
        choices=["simple", "mobilenet", "vgg"],
        default="simple",
        help="Model architecture to use",
    )
    args = parser.parse_args()

    MODEL_TYPE = args.model_type

    # Define parameters
    IMG_WIDTH = 150
    IMG_HEIGHT = 150
    IMAGE_SIZE_TUPLE = (IMG_HEIGHT, IMG_WIDTH)  # For create_data_generators
    BATCH_SIZE = 2  # Keep batch size small for dummy data
    EPOCHS = 2  # Keep epochs low for a quick test run

    # Updated directory structure for dummy data
    dummy_data_base_dir = 'data_train_engine'
    train_dir = os.path.join(dummy_data_base_dir, 'train')
    val_dir = os.path.join(dummy_data_base_dir, 'val')

    # --- Setup ---
    # Create dummy data with enough images for batching
    create_dummy_data(base_dir=dummy_data_base_dir, num_images_per_class=BATCH_SIZE * 2 + 1) # Ensure enough for class weights

    # MLflow setup: Define experiment name
    mlflow_experiment_name = "PneumoniaDetectionTraining"
    mlflow.set_experiment(mlflow_experiment_name)
    print(f"MLflow experiment set to: {mlflow_experiment_name}")

    try:
      with mlflow.start_run() as run: # Start MLflow run
        print(f"MLflow Run ID: {run.info.run_id}")
        mlflow.log_param("image_width", IMG_WIDTH)
        mlflow.log_param("image_height", IMG_HEIGHT)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs_configured", EPOCHS)  # Log configured epochs
        mlflow.log_param("model_type", MODEL_TYPE)
        # Log more parameters as needed, e.g., learning rate if configured

        # --- Load Data using create_data_generators ---
        print("Loading training and validation data using create_data_generators...")
        train_generator, validation_generator = create_data_generators(
            train_dir=train_dir,
            val_dir=val_dir,
            target_size=IMAGE_SIZE_TUPLE,
            train_batch_size=BATCH_SIZE,
            val_batch_size=BATCH_SIZE, # Using same batch size for simplicity
            rotation_range=20,         # Example augmentation
            brightness_range=[0.7, 1.3],# Example augmentation
            zoom_range=0.2,            # Example augmentation
            random_flip='horizontal',  # Default is 'horizontal'
            class_mode='binary'        # For binary classification
        )

        if train_generator is None or validation_generator is None:
            raise RuntimeError("Failed to load datasets using create_data_generators. Check paths and data_loader.py.")

        # --- Model Creation ---
        # Input shape for CNN (height, width, channels)
        # ImageDataGenerator provides 3 channels (RGB) by default
        input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
        print(f"Using input shape for model: {input_shape}")

        if MODEL_TYPE == "simple":
            model = create_simple_cnn(input_shape=input_shape, num_classes=1)
        elif MODEL_TYPE == "mobilenet":
            model = create_transfer_learning_model(
                input_shape=input_shape,
                num_classes=1,
                base_model_name="MobileNetV2",
            )
        else:  # vgg
            model = create_transfer_learning_model(
                input_shape=input_shape,
                num_classes=1,
                base_model_name="VGG16",
            )

        print("\nModel Summary:")
        model.summary()

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
        train_labels = train_generator.classes
        unique_classes = np.unique(train_labels)
        
        class_weights_array = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=train_labels
        )
        class_weights_dict = {i : class_weights_array[i] for i, _ in enumerate(unique_classes)}
        
        print(f"Train generator class indices: {train_generator.class_indices}")
        print(f"Unique classes found by np.unique: {unique_classes}")
        print(f"Calculated class weights: {class_weights_dict}")


        # --- Callbacks ---
        # Ensure 'saved_models' directory exists for ModelCheckpoint
        os.makedirs('saved_models', exist_ok=True)

        checkpoint_filepath = os.path.join('saved_models', 'best_pneumonia_cnn.keras')
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
            patience=10,
            verbose=1,
            mode='min',
            restore_best_weights=True
        )

        reduce_lr_callback = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            verbose=1,
            mode='min',
            min_lr=0.00001
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

        # --- Save Final Model ---
        # This saves the model at the end of training, regardless of ModelCheckpoint's best
        final_model_save_path = 'saved_models/pneumonia_cnn_v1.keras'
        model.save(final_model_save_path)
        print(f"Final model saved successfully to {final_model_save_path}")

        # --- MLflow Model Registration ---
        # The 'checkpoint_filepath' variable holds the path to the best model saved by ModelCheckpoint
        best_model_path = checkpoint_filepath 

        if os.path.exists(best_model_path):
            print(f"Logging best model from {best_model_path} (using in-memory model with best weights) to MLflow...")
            
            # Since EarlyStopping(restore_best_weights=True) is used, 
            # the 'model' object in memory already has the best weights.
            mlflow.keras.log_model(
                keras_model=model, 
                artifact_path="keras_model", # Path within MLflow run's artifact store
                registered_model_name="pneumonia-detector" # Name for the registered model
            )
            print(f"Model registered with MLflow under name: pneumonia-detector")
            
            # Optional: Transition model to a specific stage (example)
            # try:
            #     client = mlflow.tracking.MlflowClient()
            #     model_versions = client.get_latest_versions(name="pneumonia-detector")
            #     if model_versions:
            #         latest_version = model_versions[0].version
            #         val_accuracy_threshold = 0.90 
            #         best_val_acc = max(history.history.get('val_accuracy', [0]))
            #         if best_val_acc >= val_accuracy_threshold:
            #             client.transition_model_version_stage(
            #                 name="pneumonia-detector",
            #                 version=latest_version,
            #                 stage="Staging", 
            #                 archive_existing_versions=True 
            #             )
            #             print(f"Model version {latest_version} transitioned to Staging (val_accuracy: {best_val_acc:.4f})")
            #         else:
            #             print(f"Model version {latest_version} not transitioned (val_accuracy: {best_val_acc:.4f} < {val_accuracy_threshold})")
            # except Exception as e_transition:
            #     print(f"Error during optional model stage transition: {e_transition}")
        else:
            print(f"Warning: Best model file not found at {best_model_path}. Skipping MLflow model registration.")
        # --- End MLflow Model Registration ---

        # --- Evaluate Model (Precision, Recall, F1-score) ---
        print("\nEvaluating model on validation set...")
        
        # Ensure validation_generator is reset if it was used before (e.g. in model.fit)
        # and shuffle is False for consistent order. The create_data_generators sets shuffle=False for val.
        # validation_generator.reset() # Not strictly necessary if shuffle=False and steps cover all data

        num_val_samples_for_pred = validation_generator.samples
        val_pred_steps = (num_val_samples_for_pred // validation_generator.batch_size) + \
                         (1 if num_val_samples_for_pred % validation_generator.batch_size else 0)

        val_predictions = model.predict(validation_generator, steps=val_pred_steps)
        val_predictions = val_predictions[:num_val_samples_for_pred] # Trim if steps caused over-prediction
        val_binary_predictions = (val_predictions > 0.5).astype(int).ravel() # Use ravel for (N,1) -> (N,)

        val_true_labels = validation_generator.classes[:num_val_samples_for_pred]
        
        if len(val_true_labels) == len(val_binary_predictions):
            precision = precision_score(val_true_labels, val_binary_predictions, zero_division=0)
            recall = recall_score(val_true_labels, val_binary_predictions, zero_division=0)
            f1 = f1_score(val_true_labels, val_binary_predictions, zero_division=0)
            print(f"Validation Precision: {precision:.4f}")
            print(f"Validation Recall: {recall:.4f}")
            print(f"Validation F1-score: {f1:.4f}")
        else:
            print(f"Could not calculate precision, recall, F1-score. Label count mismatch: True labels ({len(val_true_labels)}) vs Pred labels ({len(val_binary_predictions)}).")

        # --- Plotting Training History ---
        print("\nGenerating training history plot...")
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        actual_epochs = len(acc) # Number of epochs actually run due to EarlyStopping
        mlflow.log_param("epochs_run", actual_epochs) # Log actual epochs run
        epochs_range = range(actual_epochs)


        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        if acc: mlflow.log_metric("final_train_accuracy", acc[-1])
        if val_acc: mlflow.log_metric("final_val_accuracy", val_acc[-1])


        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if loss: mlflow.log_metric("final_train_loss", loss[-1])
        if val_loss: mlflow.log_metric("final_val_loss", val_loss[-1])
        
        plt.tight_layout()
        
        plot_save_path = 'training_history.png'
        try:
            plt.savefig(plot_save_path)
            print(f"Training history plot saved to {plot_save_path}")
            mlflow.log_artifact(plot_save_path, "plots") # Log plot to MLflow
        except Exception as e_plot:
            print(f"Error saving plot: {e_plot}")
        
        # Log evaluation metrics to MLflow
        if 'precision' in locals() and 'recall' in locals() and 'f1' in locals():
            mlflow.log_metric("val_precision", precision)
            mlflow.log_metric("val_recall", recall)
            mlflow.log_metric("val_f1_score", f1)
        
        print(f"MLflow Run completed. Check experiment '{mlflow_experiment_name}'")

    except Exception as e:
        print(f"An error occurred during the training process: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- Cleanup ---
        cleanup_dummy_data(base_dir=dummy_data_base_dir) # Use updated base_dir

    print("\nTrain engine script finished.")
