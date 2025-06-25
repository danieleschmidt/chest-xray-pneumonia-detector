import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil
from PIL import Image  # For creating dummy images
import numpy as np  # For dummy image creation if needed, and for apply_contrast


# Helper function for contrast adjustment
def apply_contrast(x, contrast_range_param):
    """Applies random contrast adjustment to an image."""
    # Ensure the input is a float type, as expected by tf.image operations
    x = tf.image.convert_image_dtype(x, dtype=tf.float32)
    # Apply stateless random contrast
    return tf.image.stateless_random_contrast(
        x,
        lower=1.0 - contrast_range_param,
        upper=1.0 + contrast_range_param,
        seed=(
            np.random.randint(100),
            np.random.randint(100),
        ),  # Provide a seed tuple for stateless_random_contrast
    )


def create_data_generators(
    train_dir,
    val_dir,
    target_size=(150, 150),
    train_batch_size=32,
    val_batch_size=32,
    random_flip="horizontal",
    rotation_range=0,
    brightness_range=None,
    contrast_range=0.0,
    zoom_range=0.0,
    class_mode="binary",
):
    """
    Creates training and validation data generators using ImageDataGenerator.

    Args:
        train_dir (str): Path to the training data directory.
        val_dir (str): Path to the validation data directory.
        target_size (tuple): Desired size for the images (height, width).
        train_batch_size (int): Batch size for the training generator.
        val_batch_size (int): Batch size for the validation generator.
        random_flip (str): Type of random flipping: 'horizontal', 'vertical',
                           'horizontal_and_vertical', or None.
        rotation_range (int): Degree range for random rotations.
        brightness_range (list or tuple of 2 floats): Range for picking a brightness shift value.
                                                     e.g., [0.8, 1.2]
        contrast_range (float): Factor for random contrast adjustment.
                                If 0, no contrast adjustment. e.g., 0.2 for [0.8, 1.2]
        zoom_range (float or list/tuple of 2 floats): Range for random zoom.
                                                      If float, [1-zoom_range, 1+zoom_range].
        class_mode (str): 'binary', 'categorical', etc. Passed to flow_from_directory.


    Returns:
        tuple: (train_generator, validation_generator)
    """
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory '{train_dir}' not found.")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation directory '{val_dir}' not found.")

    # Training Data Generator
    train_datagen_args = {
        "rescale": 1.0 / 255,
        "rotation_range": rotation_range,
        "zoom_range": zoom_range,
        "brightness_range": brightness_range,
        "horizontal_flip": random_flip in ("horizontal", "horizontal_and_vertical"),
        "vertical_flip": random_flip in ("vertical", "horizontal_and_vertical"),
    }

    if contrast_range != 0.0:
        # Use a lambda that captures the current contrast_range value
        train_datagen_args["preprocessing_function"] = lambda x: apply_contrast(
            x, contrast_range_param=contrast_range
        )

    train_datagen = ImageDataGenerator(**train_datagen_args)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=train_batch_size,
        class_mode=class_mode,
        shuffle=True,
    )

    # Validation Data Generator - only rescaling
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=val_batch_size,
        class_mode=class_mode,
        shuffle=False,
    )

    return train_generator, validation_generator


def create_tf_datasets(train_dir, val_dir, image_size=(150, 150), batch_size=32):
    """Load datasets using ``image_dataset_from_directory`` with basic
    normalization.

    Parameters
    ----------
    train_dir : str
        Path to the training images directory.
    val_dir : str
        Path to the validation images directory.
    image_size : tuple, optional
        Image dimensions ``(height, width)``.
    batch_size : int, optional
        Number of images per batch.

    Returns
    -------
    tuple
        ``(train_ds, val_ds)`` TensorFlow ``Dataset`` objects or
        ``(None, None)`` if directories are missing.
    """

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError("Training or validation directory not found.")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    def _scale(images, labels):
        images = tf.cast(images, tf.float32) / 255.0
        return images, labels

    train_ds = train_ds.map(_scale)
    val_ds = val_ds.map(_scale)

    return train_ds, val_ds


def create_dummy_images_for_generator(base_dir, num_images_per_class=5):
    """Creates dummy directories and placeholder image files for ImageDataGenerator testing."""
    print(f"Creating dummy images for ImageDataGenerator under '{base_dir}'...")
    class_names = ["class_a", "class_b"]

    if os.path.exists(base_dir):  # Clean up if exists from a previous failed run
        print(f"Removing existing dummy data directory: {base_dir}")
        shutil.rmtree(base_dir)

    os.makedirs(base_dir, exist_ok=True)

    for class_name in class_names:
        path = os.path.join(base_dir, class_name)
        os.makedirs(path, exist_ok=True)
        for i in range(num_images_per_class):
            try:
                img = Image.new(
                    "RGB",
                    (200, 200),
                    color=("red" if class_name == "class_a" else "blue"),
                )
                img.save(os.path.join(path, f"img_{class_name}_{i}.png"))
            except Exception as e:
                print(
                    f"Could not create dummy image {i} for {path}: {e}. Pillow might be needed."
                )
                open(
                    os.path.join(path, f"img_{class_name}_{i}.png"), "a"
                ).close()  # Fallback
    print(f"Dummy images created in {base_dir}")


def cleanup_dummy_data_for_generator(base_dir):
    """Removes the dummy data directories created for ImageDataGenerator."""
    if os.path.exists(base_dir):
        print(f"Cleaning up dummy data from '{base_dir}'...")
        shutil.rmtree(base_dir)
        print(f"Dummy data '{base_dir}' cleaned up.")
    else:
        print(f"Directory '{base_dir}' not found, no cleanup needed.")


if __name__ == "__main__":
    # Define new dummy data directories
    dummy_train_dir = "data_generator_dummy/train_new"
    dummy_val_dir = "data_generator_dummy/val_new"

    # --- Setup Dummy Data ---
    print("--- Setting up new dummy data for create_data_generators ---")
    create_dummy_images_for_generator(
        dummy_train_dir, num_images_per_class=10
    )  # e.g. 10 images per class for train
    create_dummy_images_for_generator(
        dummy_val_dir, num_images_per_class=5
    )  # e.g. 5 images per class for val

    # Example parameters for create_data_generators
    target_size_example = (128, 128)
    batch_size_example = 3  # Small batch for easy inspection

    # --- Test create_data_generators ---
    print("\n--- Testing create_data_generators ---")
    contrast_range_example = 0.15  # Example contrast range

    train_gen, val_gen = create_data_generators(
        dummy_train_dir,
        dummy_val_dir,
        target_size=target_size_example,
        train_batch_size=batch_size_example,
        val_batch_size=batch_size_example,
        random_flip="horizontal_and_vertical",
        rotation_range=20,
        brightness_range=[0.7, 1.3],
        contrast_range=contrast_range_example,
        zoom_range=0.1,
        class_mode="binary",  # or 'categorical' if more classes
    )

    if train_gen and val_gen:
        print("\n--- Inspecting Training Generator ---")
        print(f"Train generator classes: {train_gen.classes}")
        print(f"Train generator class indices: {train_gen.class_indices}")
        for i in range(2):  # Take 2 batches
            try:
                image_batch_train, labels_batch_train = next(train_gen)
                print(f"Train Batch {i+1}:")
                print(
                    f"  Images shape: {image_batch_train.shape}, type: {image_batch_train.dtype}"
                )
                print(
                    f"  Labels shape: {labels_batch_train.shape}, type: {labels_batch_train.dtype}, values: {labels_batch_train}"
                )
                if image_batch_train.shape[0] > 0:  # if batch is not empty
                    print(
                        f"  Min/Max pixel values in images: {np.min(image_batch_train[0])}, {np.max(image_batch_train[0])}"
                    )

            except StopIteration:
                print(
                    "StopIteration: Not enough data for 2 batches in training generator."
                )
                break
            except Exception as e:
                print(f"Error iterating train_gen: {e}")
                break

        print("\n--- Inspecting Validation Generator ---")
        print(f"Validation generator classes: {val_gen.classes}")
        print(f"Validation generator class indices: {val_gen.class_indices}")
        for i in range(2):  # Take 2 batches
            try:
                image_batch_val, labels_batch_val = next(val_gen)
                print(f"Validation Batch {i+1}:")
                print(
                    f"  Images shape: {image_batch_val.shape}, type: {image_batch_val.dtype}"
                )
                print(
                    f"  Labels shape: {labels_batch_val.shape}, type: {labels_batch_val.dtype}, values: {labels_batch_val}"
                )
                if image_batch_val.shape[0] > 0:  # if batch is not empty
                    print(
                        f"  Min/Max pixel values in images: {np.min(image_batch_val[0])}, {np.max(image_batch_val[0])}"
                    )

            except StopIteration:
                print(
                    "StopIteration: Not enough data for 2 batches in validation generator."
                )
                break
            except Exception as e:
                print(f"Error iterating val_gen: {e}")
                break

        # Check if preprocessing_function was set for train_gen if contrast_range was used
        if (
            hasattr(train_gen.image_data_generator, "preprocessing_function")
            and train_gen.image_data_generator.preprocessing_function is not None
        ):
            print(
                "\nPreprocessing function is set on the training ImageDataGenerator (expected due to contrast_range)."
            )
        else:
            if (
                contrast_range_example == 0.0
            ):  # If contrast_range was 0, then no preprocessing function is expected
                print(
                    "\nPreprocessing function is NOT set on the training ImageDataGenerator (expected as contrast_range was 0)."
                )
            else:  # If contrast_range was not 0, but function is not set, it's an issue.
                print(
                    "\nWARNING: Preprocessing function IS NOT set on the training ImageDataGenerator (UNEXPECTED as contrast_range was non-zero)."
                )

    else:
        print("Failed to create data generators. Check directory paths or logs.")

    # --- Cleanup New Dummy Data ---
    print("\n--- Cleaning up new dummy data ---")
    cleanup_dummy_data_for_generator(
        "data_generator_dummy"
    )  # Clean up the parent directory

    print("\nData loader script finished.")
