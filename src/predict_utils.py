"""Generate Grad-CAM overlays for images using a trained model."""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

from .grad_cam import generate_grad_cam
from .image_utils import load_single_image


# Backward compatibility alias
def load_image(img_path, target_size):
    """Load and preprocess an image for model inference.
    
    Deprecated: Use image_utils.load_single_image instead.
    This function is maintained for backward compatibility.
    
    Parameters
    ----------
    img_path : str
        Path to the image file to load. Supported formats include PNG, JPEG, BMP.
    target_size : tuple of int
        Target dimensions (height, width) to resize the image to.
        
    Returns
    -------
    numpy.ndarray
        Preprocessed image array with shape (1, height, width, channels)
        and pixel values normalized to [0, 1].
        
    Raises
    ------
    FileNotFoundError
        If the image file does not exist at the specified path.
    PIL.UnidentifiedImageError
        If the file is not a valid image format.
        
    Examples
    --------
    >>> img_array = load_image('chest_xray.png', (150, 150))
    >>> img_array.shape
    (1, 150, 150, 3)
    >>> img_array.min(), img_array.max()
    (0.0, 1.0)
    """
    return load_single_image(img_path, target_size, normalize=True)


def display_grad_cam(
    model_path: str,
    img_path: str,
    target_size=(150, 150),
    last_conv_layer_name: str = "conv_pw_13_relu",
    output_path: str = "grad_cam_output.png",
):
    """Generate and save a Grad-CAM overlay for a given image.

    Parameters
    ----------
    model_path: str
        Path to a saved Keras model.
    img_path: str
        Path to the input image.
    target_size: tuple
        Size to which the image will be resized.
    last_conv_layer_name: str
        Name of the convolutional layer used for Grad-CAM.
    output_path: str
        Where to save the Grad-CAM overlay image.
    """

    model = tf.keras.models.load_model(model_path)
    img_array = load_single_image(img_path, target_size, normalize=True)

    heatmap = generate_grad_cam(model, img_array, last_conv_layer_name)

    heatmap = np.uint8(255 * heatmap)
    heatmap = np.expand_dims(heatmap, axis=2)
    heatmap = tf.image.resize(heatmap, target_size).numpy().astype("uint8")
    heatmap = np.squeeze(heatmap)

    # Use centralized image loading for consistency
    original_array = load_single_image(img_path, target_size, normalize=False)
    original = np.squeeze(original_array).astype("uint8")

    plt.imshow(original / 255.0)
    plt.imshow(heatmap, cmap="jet", alpha=0.4)
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Grad-CAM overlays")
    parser.add_argument("--model_path", required=True, help="Path to Keras model")
    parser.add_argument("--img_path", required=True, help="Path to input image")
    parser.add_argument(
        "--last_conv_layer_name",
        default="conv_pw_13_relu",
        help="Name of the convolutional layer for Grad-CAM",
    )
    parser.add_argument(
        "--output_path",
        default="grad_cam_output.png",
        help="Where to save the Grad-CAM overlay image",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        nargs=2,
        default=[150, 150],
        metavar=("HEIGHT", "WIDTH"),
        help="Size to which the image will be resized",
    )
    args = parser.parse_args()

    display_grad_cam(
        model_path=args.model_path,
        img_path=args.img_path,
        target_size=tuple(args.img_size),
        last_conv_layer_name=args.last_conv_layer_name,
        output_path=args.output_path,
    )
