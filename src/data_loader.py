"""Utilities for loading image datasets with optional augmentation."""

# Standard library imports
import os
import shutil
from typing import Tuple, Optional, List, Union, Callable

# Third-party imports
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import DirectoryIterator

# Local imports
from .image_utils import create_image_data_generator


# Helper function for contrast adjustment
def apply_contrast(x: tf.Tensor, contrast_range_param: float) -> tf.Tensor:
    """Applies random contrast adjustment to an image.

    Parameters
    ----------
    x : tf.Tensor
        Input image tensor with shape (height, width, channels).
    contrast_range_param : float
        Contrast adjustment range parameter. Creates contrast range of
        [1-contrast_range_param, 1+contrast_range_param].

    Returns
    -------
    tf.Tensor
        Image tensor with random contrast adjustment applied.
    """
    # Ensure the input is a float type, as expected by tf.image operations
    x = tf.image.convert_image_dtype(x, dtype=tf.float32)
    # Apply stateless random contrast
    return tf.image.stateless_random_contrast(
        x,
        lower=1.0 - contrast_range_param,
        upper=1.0 + contrast_range_param,
        seed=(
            np.random.randint(10000)
            & 0xFF,  # Efficient single random call with bit masking
            (np.random.randint(10000) >> 8) & 0xFF,
        ),  # Provide a seed tuple for stateless_random_contrast
    )


def create_data_generators(
    train_dir: str,
    val_dir: str,
    target_size: Tuple[int, int] = (150, 150),
    train_batch_size: int = 32,
    val_batch_size: int = 32,
    random_flip: Optional[str] = "horizontal",
    rotation_range: int = 0,
    brightness_range: Optional[List[float]] = None,
    contrast_range: float = 0.0,
    zoom_range: Union[float, List[float]] = 0.0,
    class_mode: str = "binary",
) -> Tuple[DirectoryIterator, DirectoryIterator]:
    """
    Creates training and validation data generators using centralized image utilities.

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
    # Validate input parameters
    if contrast_range < 0.0 or contrast_range > 1.0:
        raise ValueError(
            f"contrast_range must be between 0.0 and 1.0, got {contrast_range}. "
            f"Valid range creates contrast adjustment of [1-contrast_range, 1+contrast_range]."
        )
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(
            f"Training directory '{train_dir}' not found. "
            f"Please ensure the data directory exists and contains the expected structure."
        )
    if not os.path.exists(val_dir):
        raise FileNotFoundError(
            f"Validation directory '{val_dir}' not found. "
            f"Please ensure the data directory exists and contains the expected structure."
        )

    # Prepare custom augmentation parameters
    train_augmentation_params = {
        "rescale": 1.0 / 255,
        "rotation_range": rotation_range,
        "zoom_range": zoom_range,
        "brightness_range": brightness_range,
        "horizontal_flip": random_flip in ("horizontal", "horizontal_and_vertical"),
        "vertical_flip": random_flip in ("vertical", "horizontal_and_vertical"),
        "fill_mode": "nearest",
    }

    # Custom preprocessing function for contrast adjustment
    custom_preprocessing: Optional[Callable[[tf.Tensor], tf.Tensor]] = None
    if contrast_range != 0.0:

        def contrast_preprocessing(x: tf.Tensor) -> tf.Tensor:
            return apply_contrast(x, contrast_range_param=contrast_range)

        custom_preprocessing = contrast_preprocessing

    # Create training generator with augmentation
    train_generator = create_image_data_generator(
        directory=train_dir,
        target_size=target_size,
        batch_size=train_batch_size,
        class_mode=class_mode,
        augment=True,
        augmentation_params=train_augmentation_params,
        custom_preprocessing_function=custom_preprocessing,
    )

    # Create validation generator without augmentation
    validation_generator = create_image_data_generator(
        directory=val_dir,
        target_size=target_size,
        batch_size=val_batch_size,
        class_mode=class_mode,
        augment=False,
    )

    return train_generator, validation_generator


def create_tf_datasets(
    train_dir: str,
    val_dir: str,
    image_size: Tuple[int, int] = (150, 150),
    batch_size: int = 32,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
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
    tuple[tf.data.Dataset, tf.data.Dataset]
        ``(train_ds, val_ds)`` TensorFlow ``Dataset`` objects.

    Raises
    ------
    FileNotFoundError
        If training or validation directory does not exist.
    """

    if not os.path.exists(train_dir):
        raise FileNotFoundError(
            f"Training directory '{train_dir}' not found. "
            f"Please verify the path exists and contains subdirectories for each class."
        )
    if not os.path.exists(val_dir):
        raise FileNotFoundError(
            f"Validation directory '{val_dir}' not found. "
            f"Please verify the path exists and contains subdirectories for each class."
        )

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

    def _scale(images: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Normalize image pixel values to [0, 1] range."""
        images = tf.cast(images, tf.float32) / 255.0
        return images, labels

    train_ds = train_ds.map(_scale)
    val_ds = val_ds.map(_scale)

    return train_ds, val_ds


def create_dummy_images_for_generator(
    base_dir: str, num_images_per_class: int = 5
) -> None:
    """Creates dummy directories and placeholder image files for ImageDataGenerator testing.

    Parameters
    ----------
    base_dir : str
        Base directory path where dummy image structure will be created.
    num_images_per_class : int, default=5
        Number of dummy images to create per class.

    Notes
    -----
    Creates two classes: 'class_a' (red images) and 'class_b' (blue images).
    If PIL is not available, creates empty files as fallback.
    """
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
            except (IOError, OSError) as e:
                print(
                    f"Could not create dummy image {i} for class '{class_name}' in '{path}': {e}. "
                    f"This might be due to missing PIL/Pillow library or insufficient disk space. "
                    f"Creating placeholder file instead."
                )
                try:
                    open(
                        os.path.join(path, f"img_{class_name}_{i}.png"), "a"
                    ).close()  # Fallback
                except OSError as fallback_error:
                    print(f"Failed to create placeholder file: {fallback_error}")
                    raise
    print(f"Dummy images created in {base_dir}")


def cleanup_dummy_data_for_generator(base_dir: str) -> None:
    """Removes the dummy data directories created for ImageDataGenerator.

    Parameters
    ----------
    base_dir : str
        Base directory path containing dummy image structure to remove.
    """
    if os.path.exists(base_dir):
        print(f"Cleaning up dummy data from '{base_dir}'...")
        shutil.rmtree(base_dir)
        print(f"Dummy data '{base_dir}' cleaned up.")
    else:
        print(f"Directory '{base_dir}' not found, no cleanup needed.")
