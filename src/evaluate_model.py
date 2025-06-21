import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt


def evaluate_model(
    model_path,
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    cm_path=None,
):
    """Evaluate a trained model on a directory of test images.

    Args:
        model_path (str): Path to the saved Keras model.
        test_dir (str): Directory containing test images organized in class folders.
        target_size (tuple): Image size expected by the model.
        batch_size (int): Batch size for the data generator.
        class_mode (str): 'binary' or 'categorical' depending on the model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    model = tf.keras.models.load_model(model_path)

    datagen = ImageDataGenerator(rescale=1.0 / 255)
    generator = datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=False,
    )

    predictions = model.predict(generator, verbose=1)
    if class_mode == "binary":
        y_pred = (predictions > 0.5).astype(int).ravel()
    else:
        y_pred = np.argmax(predictions, axis=1)

    y_true = generator.classes

    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    target_names = list(generator.class_indices.keys())
    print(classification_report(y_true, y_pred, target_names=target_names))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    if cm_path:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set(xticks=range(len(target_names)), yticks=range(len(target_names)))
        ax.set_xticklabels(target_names, rotation=45, ha="right")
        ax.set_yticklabels(target_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close(fig)
        print(f"Confusion matrix plot saved to {cm_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--model", required=True, help="Path to the Keras model")
    parser.add_argument("--data", required=True, help="Directory of test images")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=[150, 150],
        help="Input image size",
    )
    parser.add_argument(
        "--class_mode",
        choices=["binary", "categorical"],
        default="binary",
        help="Model output type",
    )
    parser.add_argument(
        "--confusion_matrix",
        help="Optional path to save confusion matrix plot",
    )
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        test_dir=args.data,
        target_size=tuple(args.target_size),
        batch_size=args.batch_size,
        class_mode=args.class_mode,
        cm_path=args.confusion_matrix,
    )


if __name__ == "__main__":
    main()
