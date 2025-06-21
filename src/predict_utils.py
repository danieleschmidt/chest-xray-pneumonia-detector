import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

from grad_cam import generate_grad_cam


def load_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


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
    img_array = load_image(img_path, target_size)

    heatmap = generate_grad_cam(model, img_array, last_conv_layer_name)

    heatmap = np.uint8(255 * heatmap)
    heatmap = np.expand_dims(heatmap, axis=2)
    heatmap = tf.image.resize(heatmap, target_size).numpy().astype("uint8")
    heatmap = np.squeeze(heatmap)

    original = image.load_img(img_path, target_size=target_size)
    original = image.img_to_array(original).astype("uint8")

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
