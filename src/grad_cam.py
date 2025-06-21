import argparse
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

def generate_grad_cam(model, image_array, last_conv_layer_name):
    """
    Generates a Grad-CAM heatmap for a given image and model.

    Args:
        model (tf.keras.Model): The trained Keras model.
        image_array (np.ndarray): A preprocessed image array, typically the output
                                  of `np.expand_dims(img_array, axis=0)` where
                                  img_array is a single image (e.g., HxWxC).
        last_conv_layer_name (str): The name of the last convolutional layer
                                    in the model to be used for Grad-CAM.

    Returns:
        np.ndarray: Normalized heatmap highlighting influential regions.
    """

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        if predictions.shape[-1] == 1:
            loss = predictions[:, 0]
        else:
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()


def find_last_conv_layer(model):
    """Return the name of the last convolutional layer in ``model``.

    This is useful when a layer name is not explicitly provided. The function
    searches the model's layers in reverse order and returns the first layer
    whose output has four dimensions (``rank == 4``), which typically
    corresponds to a convolutional feature map.
    """
    for layer in reversed(model.layers):
        try:
            if len(layer.output_shape) == 4:
                return layer.name
        except AttributeError:
            continue
    raise ValueError("No convolutional layer found in the model.")


def overlay_heatmap_on_image(img, heatmap, alpha=0.4, cmap="viridis"):
    """Overlay a heatmap onto an image.

    Args:
        img (np.ndarray): Original image as a NumPy array in the range [0, 1].
        heatmap (np.ndarray): Heatmap from :func:`generate_grad_cam`.
        alpha (float): Transparency factor for heatmap overlay.
        cmap (str): Matplotlib colormap to colorize the heatmap.

    Returns:
        np.ndarray: The image with the heatmap overlay applied.
    """

    if heatmap.max() == 0:
        heatmap_normalized = heatmap
    else:
        heatmap_normalized = heatmap / heatmap.max()

    cmap = plt.get_cmap(cmap)
    colored_heatmap = cmap(heatmap_normalized)
    colored_heatmap = colored_heatmap[..., :3]  # Drop alpha channel

    if img.max() > 1:
        img = img / 255.0

    overlay = colored_heatmap * alpha + img * (1 - alpha)
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype("uint8")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmap")
    parser.add_argument("--model", required=True, help="Path to Keras model")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument(
        "--layer",
        help="Name of the last convolutional layer for Grad-CAM. If omitted, the script attempts to detect it automatically.",
    )
    parser.add_argument(
        "--output",
        default="grad_cam_overlay.png",
        help="Path to save the overlay image",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    model = tf.keras.models.load_model(args.model)

    img = image.load_img(args.image, target_size=model.input_shape[1:3])
    img_array = image.img_to_array(img) / 255.0
    input_array = np.expand_dims(img_array, axis=0)

    layer_name = args.layer
    if not layer_name:
        try:
            layer_name = find_last_conv_layer(model)
            print(f"Automatically selected layer '{layer_name}' for Grad-CAM")
        except ValueError as e:
            raise RuntimeError(
                "Could not automatically determine the last convolutional layer. "
                "Please specify it with --layer"
            ) from e

    heatmap = generate_grad_cam(model, input_array, layer_name)
    overlay = overlay_heatmap_on_image(img_array, heatmap)

    plt.imsave(args.output, overlay)
    print(f"Grad-CAM overlay saved to {args.output}")
