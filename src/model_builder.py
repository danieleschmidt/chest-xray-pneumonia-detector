"""Model-building utilities for the pneumonia detection CNN."""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Reshape,
    multiply,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.applications import MobileNetV2, VGG16


def create_transfer_learning_model(
    input_shape,
    num_classes=1,
    base_model_name="MobileNetV2",
    trainable_base_layers=0,
    learning_rate=0.001,
    dropout_rate=0.0,
):
    """Create a transfer learning model with a configurable base.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input images, e.g. ``(height, width, channels)``.
    num_classes : int, optional
        Number of output classes. ``1`` assumes binary classification.
    base_model_name : str, optional
        Name of the keras application model to use. Currently supports
        ``"MobileNetV2"`` and ``"VGG16"``.
    trainable_base_layers : int, optional
        Number of layers at the end of the base model to unfreeze for
        fine-tuning. ``0`` keeps the base frozen.

    Returns
    -------
    tf.keras.Model
        A compiled Keras model ready for training.
    """

    if base_model_name == "MobileNetV2":
        base_model = MobileNetV2(
            include_top=False, input_shape=input_shape, weights="imagenet"
        )
    elif base_model_name == "VGG16":
        base_model = VGG16(
            include_top=False, input_shape=input_shape, weights="imagenet"
        )
    else:
        raise ValueError("Unsupported base_model_name. Use 'MobileNetV2' or 'VGG16'.")

    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)

    activation = "sigmoid" if num_classes == 1 else "softmax"
    loss = BinaryCrossentropy() if num_classes == 1 else CategoricalCrossentropy()

    outputs = Dense(num_classes, activation=activation)(x)
    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate), loss=loss, metrics=["accuracy"]
    )

    if trainable_base_layers > 0:
        for layer in base_model.layers[-trainable_base_layers:]:
            layer.trainable = True

    return model


def create_simple_cnn(
    input_shape, num_classes=1, learning_rate=0.001, dropout_rate=0.0
):
    """
    Builds a simple Keras Sequential CNN model for binary classification.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes. Defaults to 1 for binary
                           classification.

    Returns:
        tensorflow.keras.models.Sequential: A compiled Keras CNN model.
    """
    if num_classes != 1:
        raise ValueError(
            "This simple CNN is designed for binary classification (num_classes=1)."
        )

    model = Sequential(
        [
            # First Convolutional Block
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D((2, 2)),
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            # Flattening and Dense Layer
            Flatten(),
            Dense(
                128, activation="relu"
            ),  # Optional: an additional dense layer before the output
            (
                Dropout(dropout_rate)
                if dropout_rate > 0
                else tf.keras.layers.Activation("linear")
            ),
            Dense(
                num_classes, activation="sigmoid"
            ),  # Output layer for binary classification
        ]
    )

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=BinaryCrossentropy(),
        metrics=["accuracy"],  # Common metric to monitor
    )

    return model


def _squeeze_excite_block(inputs, ratio=16):
    """Squeeze-and-Excitation block used for attention."""
    filters = inputs.shape[-1]
    se = GlobalAveragePooling2D()(inputs)
    se = Dense(filters // ratio, activation="relu")(se)
    se = Dense(filters, activation="sigmoid")(se)
    se = Reshape((1, 1, filters))(se)
    return multiply([inputs, se])


def create_cnn_with_attention(
    input_shape, num_classes=1, learning_rate=0.001, dropout_rate=0.0
):
    """Build a simple CNN with Squeeze-and-Excitation attention blocks."""

    inputs = tf.keras.Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation="relu")(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = _squeeze_excite_block(x)

    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = _squeeze_excite_block(x)

    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = Dense(num_classes, activation=activation)(x)

    model = Model(inputs, outputs)
    loss = BinaryCrossentropy() if num_classes == 1 else CategoricalCrossentropy()
    model.compile(
        optimizer=Adam(learning_rate=learning_rate), loss=loss, metrics=["accuracy"]
    )

    return model
