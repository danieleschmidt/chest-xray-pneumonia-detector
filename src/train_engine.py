import tensorflow as tf
import os
import shutil # For directory cleanup
from PIL import Image # For creating dummy images
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score


# Attempt to import from src, assuming the script is run from the project root
# or the src directory is in PYTHONPATH
try:
    from data_loader import load_images_from_directory
    from model_builder import create_simple_cnn
except ImportError:
    # Fallback for direct execution or if src is not in PYTHONPATH
    # This might happen if the script is run directly from the src directory
    # or if the IDE/environment doesn't automatically add 'src' to sys.path
    print("Attempting fallback import for data_loader and model_builder.")
    try:
        from .data_loader import load_images_from_directory
        from .model_builder import create_simple_cnn
    except ImportError as e:
        print(f"Error importing modules. Make sure 'src' is in PYTHONPATH or run from project root.")
        print(f"Details: {e}")
        # You might want to exit here or raise the error depending on desired behavior
        raise

def create_dummy_data(base_dir="data", num_images_per_class=3):
    """Creates dummy directories and placeholder image files for training and validation."""
    print(f"Creating dummy data under '{base_dir}'...")
    class_names = ["class_a", "class_b"]
    sets = ["train", "val"]
    for s in sets:
        for class_name in class_names:
            path = os.path.join(base_dir, s, class_name)
            os.makedirs(path, exist_ok=True)
            for i in range(num_images_per_class):
                try:
                    # Create a small, simple JPG image
                    img = Image.new('RGB', (60, 30), color = ('red' if class_name == "class_a" else 'blue'))
                    img.save(os.path.join(path, f"dummy_{s}_{class_name}_{i+1}.jpg"))
                except Exception as e:
                    print(f"Could not create dummy image {i+1} for {path}: {e}. Pillow might be needed.")
                    # Create an empty file as a fallback if PIL fails
                    open(os.path.join(path, f"dummy_{s}_{class_name}_{i+1}.jpg"), 'a').close()
    print("Dummy data created.")

def cleanup_dummy_data(base_dir="data"):
    """Removes the dummy data directories."""
    if os.path.exists(base_dir):
        print(f"Cleaning up dummy data from '{base_dir}'...")
        shutil.rmtree(base_dir)
        print("Dummy data cleaned up.")
    else:
        print(f"Directory '{base_dir}' not found, no cleanup needed.")


if __name__ == '__main__':
    # Define parameters
    IMG_WIDTH = 150
    IMG_HEIGHT = 150
    IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)
    BATCH_SIZE = 2 # Keep batch size small for dummy data
    # BUFFER_SIZE is used in dataset.shuffle, not directly here but good to define if expanding
    # BUFFER_SIZE = 1000 # Example, depends on dataset size
    EPOCHS = 2 # Keep epochs low for a quick test run

    train_dir_base = 'data' # Base directory for dummy data
    train_dir = os.path.join(train_dir_base, 'train')
    val_dir = os.path.join(train_dir_base, 'val')

    # --- Setup ---
    create_dummy_data(base_dir=train_dir_base, num_images_per_class=BATCH_SIZE * 2) # Ensure enough images for a few batches

    try:
        # --- Load Data ---
        print("Loading training data...")
        train_dataset = load_images_from_directory(train_dir,
                                                   target_size=IMAGE_SIZE,
                                                   batch_size=BATCH_SIZE)

        print("Loading validation data...")
        # Note: The current load_images_from_directory applies augmentation by default.
        # To have no augmentation for validation, data_loader.py would need modification.
        val_dataset = load_images_from_directory(val_dir,
                                                 target_size=IMAGE_SIZE,
                                                 batch_size=BATCH_SIZE)

        if train_dataset is None or val_dataset is None:
            raise RuntimeError("Failed to load datasets. Check data paths and data_loader.py.")

        # --- Model Creation ---
        # Determine input_shape from one batch of the dataset
        # (batch_size, height, width, channels)
        # We need (height, width, channels) for the model
        input_shape = None
        for images, _ in train_dataset.take(1):
            input_shape = images.shape[1:] # Get (height, width, channels)
            break
        
        if input_shape is None:
             # Fallback if dataset was empty or failed to load images
            print("Warning: Could not determine input_shape from dataset. Using default (IMG_HEIGHT, IMG_WIDTH, 3).")
            input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
        else:
            print(f"Determined input shape from data: {input_shape}")


        model = create_simple_cnn(input_shape=input_shape, num_classes=1)
        print("\nModel Summary:")
        model.summary()

        # --- Model Training ---
        print("\nStarting model training...")
        history = model.fit(
            train_dataset,
            epochs=EPOCHS,
            validation_data=val_dataset,
            verbose=1 # Set to 1 or 2 for more output, 0 for less
        )
        print("Model training completed.")

        # --- Save Model ---
        model_save_path = 'saved_models/pneumonia_cnn_v1.keras'
        model_save_dir = os.path.dirname(model_save_path)

        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
            print(f"Created directory: {model_save_dir}")

        model.save(model_save_path)
        print(f"Model saved successfully to {model_save_path}")

        # --- Evaluate Model (Precision, Recall, F1-score) ---
        print("\nEvaluating model on validation set...")
        val_predictions = model.predict(val_dataset)
        val_binary_predictions = (val_predictions > 0.5).astype(int)

        val_true_labels = []
        # Ensure val_dataset is not re-used if it's fully consumed by model.predict
        # Re-initialize if necessary, or ensure it can be iterated multiple times
        # For image_dataset_from_directory, it can be iterated multiple times.
        for _, labels in val_dataset:
            val_true_labels.extend(labels.numpy())
        val_true_labels = np.array(val_true_labels)
        
        # Check if there are enough samples to calculate metrics
        if len(val_true_labels) > 0 and len(val_binary_predictions) == len(val_true_labels):
            precision = precision_score(val_true_labels, val_binary_predictions, zero_division=0)
            recall = recall_score(val_true_labels, val_binary_predictions, zero_division=0)
            f1 = f1_score(val_true_labels, val_binary_predictions, zero_division=0)
            print(f"Validation Precision: {precision:.4f}")
            print(f"Validation Recall: {recall:.4f}")
            print(f"Validation F1-score: {f1:.4f}")
        else:
            print("Could not calculate precision, recall, F1-score. Check validation data and predictions.")


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
        
        plot_save_path = 'training_history.png'
        try:
            plt.savefig(plot_save_path)
            print(f"Training history plot saved to {plot_save_path}")
        except Exception as e_plot:
            print(f"Error saving plot: {e_plot}")


    except Exception as e:
        print(f"An error occurred during the training process: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- Cleanup ---
        cleanup_dummy_data(base_dir=train_dir_base)

    print("\nTrain engine script finished.")
