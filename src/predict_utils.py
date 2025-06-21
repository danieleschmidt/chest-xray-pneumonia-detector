import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

from grad_cam import (
    generate_grad_cam,
    overlay_heatmap_on_image,
    find_last_conv_layer,
)


def predict_image(model, img_array):
    """Return the model's raw prediction for ``img_array``.

    Args:
        model (tf.keras.Model): Loaded Keras model.
        img_array (np.ndarray): Preprocessed single image of shape (H, W, C) with
            values in ``[0, 1]``.

    Returns:
        float: Probability of pneumonia as predicted by the model.
    """

    input_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(input_array, verbose=0)[0][0]
    return float(prediction)


def display_grad_cam(
    model_path,
    image_path,
    layer_name=None,
    output_path="grad_cam_result.png",
):
    """Generate and save a Grad-CAM overlay for an image using a saved model.

    Args:
        model_path (str): Path to a Keras model file.
        image_path (str): Path to the image to analyze.
        layer_name (str, optional): Name of the last convolutional layer to use.
            If ``None``, the layer is automatically detected.
        output_path (str): Path to save the resulting overlay image.

    Returns:
        str: Path to the saved overlay image.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = tf.keras.models.load_model(model_path)

    img = image.load_img(image_path, target_size=model.input_shape[1:3])
    img_array = image.img_to_array(img) / 255.0
    input_array = np.expand_dims(img_array, axis=0)

    if layer_name is None:
        layer_name = find_last_conv_layer(model)

    heatmap = generate_grad_cam(model, input_array, layer_name)
    overlay = overlay_heatmap_on_image(img_array, heatmap)

    plt.imsave(output_path, overlay)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Run prediction and optional Grad-CAM overlay"
    )
    parser.add_argument("--model", required=True, help="Path to Keras model")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument(
        "--layer",
        help="Name of last conv layer (auto if omitted)",
    )
    parser.add_argument(
        "--output",
        default="grad_cam_result.png",
        help="Where to save the overlay",
    )
    parser.add_argument(
        "--no_overlay",
        action="store_true",
        help="Skip Grad-CAM generation and only output the prediction",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    model = tf.keras.models.load_model(args.model)
    img = image.load_img(args.image, target_size=model.input_shape[1:3])
    img_array = image.img_to_array(img) / 255.0

    prob = predict_image(model, img_array)
    print(f"Predicted pneumonia probability: {prob:.4f}")

    if not args.no_overlay:
        layer = args.layer or find_last_conv_layer(model)
        heatmap = generate_grad_cam(model, np.expand_dims(img_array, 0), layer)
        overlay = overlay_heatmap_on_image(img_array, heatmap)
        plt.imsave(args.output, overlay)
        print(f"Grad-CAM saved to {args.output}")


if __name__ == "__main__":
    main()
