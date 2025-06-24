"""Grad-CAM utility functions."""

from typing import Optional

import numpy as np
import tensorflow as tf


def generate_grad_cam(
    model: tf.keras.Model,
    image_array: np.ndarray,
    last_conv_layer_name: str,
    class_index: Optional[int] = None,
) -> np.ndarray:
    """Generate a Grad-CAM heatmap for a given image.

    Parameters
    ----------
    model:
        A trained ``tf.keras`` model.
    image_array:
        Preprocessed image with shape ``(1, H, W, C)``.
    last_conv_layer_name:
        Name of the last convolutional layer to compute gradients from.
    class_index:
        Optional target class index. If ``None`` the predicted class is used.

    Returns
    -------
    np.ndarray
        2D heatmap array normalised between 0 and 1.
    """

    # Build a model that maps the input image to the activations of the last
    # conv layer as well as the final predictions
    conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        [model.inputs], [conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)

        if class_index is None:
            # If multi-class, take the index of the highest predicted score
            if predictions.shape[-1] > 1:
                class_index = tf.argmax(predictions[0])
            else:
                class_index = 0

        class_channel = predictions[:, class_index]

    # Compute the gradient of the predicted class with regard to the output
    # feature map of the last conv layer
    grads = tape.gradient(class_channel, conv_outputs)

    # Pool the gradients over all the axes leaving out the channel dimension
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the output feature map with the pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs * pooled_grads
    heatmap = tf.reduce_sum(heatmap, axis=-1)

    # Apply ReLU and normalise the heatmap between 0 and 1
    heatmap = tf.nn.relu(heatmap)
    if tf.reduce_max(heatmap) != 0:
        heatmap /= tf.reduce_max(heatmap)

    return heatmap.numpy()


if __name__ == "__main__":
    # Basic smoke test when executed directly
    print("grad_cam.py executed. 'generate_grad_cam' is ready for use.")
