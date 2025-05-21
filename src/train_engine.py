import tensorflow as tf
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import shutil # For directory cleanup
from PIL import Image # For creating dummy images
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight # Added for class imbalance


# Attempt to import from src, assuming the script is run from the project root
# or the src directory is in PYTHONPATH
try:
    from data_loader import create_data_generators # Updated import
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
    # Define parameters
    IMG_WIDTH = 150
    IMG_HEIGHT = 150
    IMAGE_SIZE_TUPLE = (IMG_HEIGHT, IMG_WIDTH) # For create_data_generators
    BATCH_SIZE = 2 # Keep batch size small for dummy data
    EPOCHS = 2 # Keep epochs low for a quick test run

    # Updated directory structure for dummy data
    dummy_data_base_dir = 'data_train_engine'
    train_dir = os.path.join(dummy_data_base_dir, 'train')
    val_dir = os.path.join(dummy_data_base_dir, 'val')

    # --- Setup ---
    # Create dummy data with enough images for batching
    create_dummy_data(base_dir=dummy_data_base_dir, num_images_per_class=BATCH_SIZE * 2 + 1)


    try:
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

        model = create_simple_cnn(input_shape=input_shape, num_classes=1) # Assuming binary classification
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
        # The directory 'saved_models' is already ensured by the callback setup
        # model_save_dir = os.path.dirname(final_model_save_path)
        # if not os.path.exists(model_save_dir):
        #     os.makedirs(model_save_dir)
        #     print(f"Created directory: {model_save_dir}")

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
        cleanup_dummy_data(base_dir=dummy_data_base_dir) # Use updated base_dir

    print("\nTrain engine script finished.")
