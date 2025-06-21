import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
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
        raise ValueError(
            "Unsupported base_model_name. Use 'MobileNetV2' or 'VGG16'."
        )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)

    activation = "sigmoid" if num_classes == 1 else "softmax"
    loss = BinaryCrossentropy() if num_classes == 1 else CategoricalCrossentropy()

    outputs = Dense(num_classes, activation=activation)(x)
    model = Model(inputs, outputs)

    model.compile(optimizer=Adam(), loss=loss, metrics=["accuracy"])

    if trainable_base_layers > 0:
        for layer in base_model.layers[-trainable_base_layers:]:
            layer.trainable = True

    return model


def create_simple_cnn(input_shape, num_classes=1):
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
        raise ValueError("This simple CNN is designed for binary classification (num_classes=1).")

    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),

        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Flattening and Dense Layer
        Flatten(),
        Dense(128, activation='relu'), # Optional: an additional dense layer before the output
        Dense(num_classes, activation='sigmoid') # Output layer for binary classification
    ])

    # Compile the model
    model.compile(
        optimizer=Adam(),
        loss=BinaryCrossentropy(),
        metrics=['accuracy'] # Common metric to monitor
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


def create_cnn_with_attention(input_shape, num_classes=1):
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
    activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = Dense(num_classes, activation=activation)(x)

    model = Model(inputs, outputs)
    loss = BinaryCrossentropy() if num_classes == 1 else CategoricalCrossentropy()
    model.compile(optimizer=Adam(), loss=loss, metrics=["accuracy"])

    return model

if __name__ == '__main__':
    # This is an example of how to use the function
    img_height = 150
    img_width = 150
    channels = 3
    input_shape_example = (img_height, img_width, channels)

    print(f"Creating a simple CNN model with input shape: {input_shape_example}")
    cnn_model = create_simple_cnn(input_shape_example)

    print("\nModel Summary:")
    cnn_model.summary()

    # Example of how you might check the model's configuration
    print("\nModel Configuration (first few layers):")
    for layer in cnn_model.layers[:3]: # Print config for first 3 layers
        print(f"Layer: {layer.name}, Config: {layer.get_config()}")

    print(f"\nOptimizer: {cnn_model.optimizer.get_config()['name']}")
    print(f"Loss function: {cnn_model.loss}")

    # Test with a different number of classes (should raise ValueError)
    try:
        print("\nTesting create_simple_cnn with num_classes=2 (should fail for this binary setup):")
        create_simple_cnn(input_shape_example, num_classes=2)
    except ValueError as e:
        print(f"Caught expected error for create_simple_cnn: {e}")

    print("\n" + "="*50 + "\n")

    # --- Example for create_transfer_learning_model ---
    print("Creating a transfer learning model with MobileNetV2 base...")
    transfer_model_mobilenet = create_transfer_learning_model(input_shape_example, num_classes=1, base_model_name='MobileNetV2')
    print("\nMobileNetV2-based Model Summary:")
    transfer_model_mobilenet.summary()
    print(f"\nOptimizer: {transfer_model_mobilenet.optimizer.get_config()['name']}")
    print(f"Loss function: {transfer_model_mobilenet.loss}")


    print("\n" + "="*50 + "\n")
    print("Creating a transfer learning model with VGG16 base...")
    # VGG16 might require 3 channels, ensure input_shape_example is compatible or adjust
    if input_shape_example[2] != 3:
        print(f"Warning: VGG16 typically expects 3 channels. Current input shape: {input_shape_example}. Adjusting to 3 channels for VGG16 example.")
        input_shape_vgg = (input_shape_example[0], input_shape_example[1], 3)
    else:
        input_shape_vgg = input_shape_example
        
    transfer_model_vgg = create_transfer_learning_model(input_shape_vgg, num_classes=1, base_model_name='VGG16')
    print("\nVGG16-based Model Summary:")
    transfer_model_vgg.summary()
    print(f"\nOptimizer: {transfer_model_vgg.optimizer.get_config()['name']}")
    print(f"Loss function: {transfer_model_vgg.loss}")

    print("\n" + "="*50 + "\n")
    print("Creating a transfer learning model with MobileNetV2 base for multi-class (e.g., 5 classes)...")
    transfer_model_multiclass = create_transfer_learning_model(input_shape_example, num_classes=5, base_model_name='MobileNetV2')
    print("\nMobileNetV2-based Multi-class Model Summary:")
    transfer_model_multiclass.summary()
    print(f"\nOptimizer: {transfer_model_multiclass.optimizer.get_config()['name']}")
    print(f"Loss function: {transfer_model_multiclass.loss}")


    print("\n" + "="*50 + "\n")
    # Test with unsupported base model
    try:
        print("\nTesting with unsupported base model 'ResNet50':")
        create_transfer_learning_model(input_shape_example, num_classes=1, base_model_name='ResNet50')
    except ValueError as e:
        print(f"Caught expected error for unsupported base model: {e}")
