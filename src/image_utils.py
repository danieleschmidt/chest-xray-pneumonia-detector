"""Centralized image loading and preprocessing utilities.

This module provides unified functions for image loading and data generator creation
to eliminate code duplication across the project. It standardizes image preprocessing
patterns used throughout the application.
"""
from typing import Tuple, Optional, Dict, Any, Union
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_single_image(
    img_path: str, 
    target_size: Tuple[int, int],
    normalize: bool = True
) -> np.ndarray:
    """Load and preprocess a single image for model inference.
    
    Loads an image from the specified path, resizes it to target dimensions,
    converts it to a numpy array with batch dimension, and optionally normalizes
    pixel values to the range [0, 1] for neural network input.
    
    Parameters
    ----------
    img_path : str
        Path to the image file to load. Supported formats include PNG, JPEG, BMP.
    target_size : tuple of int
        Target dimensions (height, width) to resize the image to.
    normalize : bool, default=True
        Whether to normalize pixel values to [0, 1] range.
        
    Returns
    -------
    numpy.ndarray
        Preprocessed image array with shape (1, height, width, channels).
        If normalize=True, pixel values are in [0, 1] range.
        If normalize=False, pixel values are in [0, 255] range.
        
    Raises
    ------
    FileNotFoundError
        If the image file does not exist at the specified path.
    PIL.UnidentifiedImageError
        If the file is not a valid image format.
        
    Examples
    --------
    >>> img_array = load_single_image('chest_xray.png', (150, 150))
    >>> img_array.shape
    (1, 150, 150, 3)
    >>> img_array.min(), img_array.max()
    (0.0, 1.0)
    
    >>> img_array = load_single_image('image.jpg', (224, 224), normalize=False)
    >>> img_array.min(), img_array.max()  # doctest: +SKIP
    (0.0, 255.0)
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    if normalize:
        img_array = img_array / 255.0
    
    return img_array


def create_image_data_generator(
    directory: str,
    target_size: Tuple[int, int] = (150, 150),
    batch_size: int = 32,
    class_mode: str = "binary",
    augment: bool = False,
    augmentation_params: Optional[Dict[str, Any]] = None,
    custom_preprocessing_function: Optional[callable] = None
) -> image.DirectoryIterator:
    """Create an ImageDataGenerator for training or validation data.
    
    Creates a standardized ImageDataGenerator with consistent preprocessing
    and optional data augmentation for training data.
    
    Parameters
    ----------
    directory : str
        Path to the directory containing image data organized in subdirectories.
    target_size : tuple of int, default=(150, 150)
        Target dimensions (height, width) to resize images to.
    batch_size : int, default=32
        Number of images to include in each batch.
    class_mode : str, default="binary"
        Type of classification problem ("binary", "categorical", "sparse", etc.).
    augment : bool, default=False
        Whether to apply data augmentation. Should be True for training data,
        False for validation data.
    augmentation_params : dict, optional
        Custom augmentation parameters. If None, uses default augmentation
        settings when augment=True.
    custom_preprocessing_function : callable, optional
        Custom preprocessing function to apply to images (e.g., for contrast adjustment).
        
    Returns
    -------
    tensorflow.keras.preprocessing.image.DirectoryIterator
        Configured data generator for the specified directory.
        
    Examples
    --------
    >>> # Create training generator with augmentation
    >>> train_gen = create_image_data_generator(
    ...     'data/train', augment=True
    ... )  # doctest: +SKIP
    
    >>> # Create validation generator without augmentation
    >>> val_gen = create_image_data_generator(
    ...     'data/val', augment=False
    ... )  # doctest: +SKIP
    """
    if augment:
        # Default augmentation parameters
        default_augmentation = {
            'rescale': 1.0 / 255,
            'rotation_range': 20,
            'width_shift_range': 0.2,
            'height_shift_range': 0.2,
            'shear_range': 0.2,
            'zoom_range': 0.2,
            'horizontal_flip': True,
            'fill_mode': 'nearest'
        }
        
        # Override with custom parameters if provided
        if augmentation_params:
            default_augmentation.update(augmentation_params)
        
        # Add custom preprocessing function if provided
        if custom_preprocessing_function:
            default_augmentation['preprocessing_function'] = custom_preprocessing_function
        
        datagen_params = default_augmentation
        shuffle = True
    else:
        # Validation/test data - only rescaling
        datagen_params = {'rescale': 1.0 / 255}
        shuffle = False
    
    datagen = ImageDataGenerator(**datagen_params)
    
    generator = datagen.flow_from_directory(
        directory,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=shuffle
    )
    
    return generator


def create_inference_data_generator(
    directory: str,
    target_size: Tuple[int, int] = (150, 150),
    batch_size: int = 32
) -> image.DirectoryIterator:
    """Create an ImageDataGenerator specifically for inference on unlabeled data.
    
    Creates a standardized generator for inference without labels, applying only
    rescaling preprocessing to maintain consistency with training preprocessing.
    
    Parameters
    ----------
    directory : str
        Path to directory containing images for inference.
    target_size : tuple of int, default=(150, 150)
        Target dimensions (height, width) to resize images to.
    batch_size : int, default=32
        Number of images to process in each batch.
        
    Returns
    -------
    tensorflow.keras.preprocessing.image.DirectoryIterator
        Configured data generator for inference with filepaths accessible
        via the .filepaths attribute.
        
    Examples
    --------
    >>> # Create inference generator
    >>> inf_gen = create_inference_data_generator('data/test')  # doctest: +SKIP
    >>> print(f"Found {inf_gen.samples} images")  # doctest: +SKIP
    >>> filepaths = inf_gen.filepaths  # Access to file paths  # doctest: +SKIP
    """
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    
    generator = datagen.flow_from_directory(
        directory,
        target_size=target_size,
        class_mode=None,  # No labels for inference
        shuffle=False,    # Maintain order for predictions
        batch_size=batch_size
    )
    
    return generator


# Backward compatibility aliases for transition period
def load_image(img_path: str, target_size: Tuple[int, int]) -> np.ndarray:
    """Backward compatibility alias for load_single_image.
    
    Deprecated: Use load_single_image instead.
    """
    return load_single_image(img_path, target_size, normalize=True)