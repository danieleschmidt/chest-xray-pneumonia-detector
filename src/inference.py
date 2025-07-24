"""Command-line interface for running a trained model on a directory of images."""

import argparse
import sys
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from .image_utils import create_inference_data_generator
from .input_validation import validate_model_path, validate_directory_path, ValidationError


def predict_directory(
    model_path: str,
    data_dir: str,
    img_size=(150, 150),
    num_classes: int = 1,
) -> pd.DataFrame:
    """Generate predictions for all images in a directory.

    Parameters
    ----------
    model_path : str
        Path to a saved Keras model.
    data_dir : str
        Directory containing images organised in subfolders. Only file paths are used;
        no label inference is performed.
    img_size : tuple, optional
        Image size expected by the model.

    Returns
    -------
    pandas.DataFrame
        For binary models, a DataFrame with columns ``filepath`` and ``prediction``.
        For multi-class models, the DataFrame includes ``prediction`` (predicted
        class index) and ``prob_i`` columns with class probabilities.
    """
    # Validate inputs for security
    validated_model_path = validate_model_path(model_path, must_exist=True)
    validated_data_dir = validate_directory_path(data_dir, must_exist=True)
    
    model = tf.keras.models.load_model(validated_model_path)

    generator = create_inference_data_generator(
        directory=validated_data_dir,
        target_size=img_size,
        batch_size=32
    )

    preds = model.predict(generator)

    if num_classes == 1:
        preds = preds.reshape(-1)
        df = pd.DataFrame({"filepath": generator.filepaths, "prediction": preds})
    else:
        pred_labels = preds.argmax(axis=1)
        data = {"filepath": generator.filepaths, "prediction": pred_labels}
        for i in range(num_classes):
            data[f"prob_{i}"] = preds[:, i]
        df = pd.DataFrame(data)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch inference on a directory of images"
    )
    parser.add_argument(
        "--model_path", required=True, help="Path to a saved Keras model"
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory with images organised in class folders",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        nargs=2,
        default=[150, 150],
        metavar=("HEIGHT", "WIDTH"),
        help="Image size expected by the model",
    )
    parser.add_argument(
        "--output_csv", default="predictions.csv", help="Where to save predictions"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1,
        help="Number of model output classes (1 for binary)",
    )
    args = parser.parse_args()

    try:
        # Validate inputs before processing
        validated_model_path = validate_model_path(args.model_path, must_exist=True)
        validated_data_dir = validate_directory_path(args.data_dir, must_exist=True)
        
        df = predict_directory(
            validated_model_path,
            validated_data_dir,
            tuple(args.img_size),
            num_classes=args.num_classes,
        )
        df.to_csv(args.output_csv, index=False)
        print(f"Saved predictions to {args.output_csv}")
        
    except ValidationError as e:
        print(f"Input validation error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
