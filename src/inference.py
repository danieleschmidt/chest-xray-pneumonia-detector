import argparse
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image


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
    model = tf.keras.models.load_model(model_path)

    datagen = image.ImageDataGenerator(rescale=1.0 / 255)
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        class_mode=None,
        shuffle=False,
        batch_size=32,
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
    parser = argparse.ArgumentParser(description="Batch inference on a directory of images")
    parser.add_argument("--model_path", required=True, help="Path to a saved Keras model")
    parser.add_argument("--data_dir", required=True, help="Directory with images organised in class folders")
    parser.add_argument(
        "--img_size",
        type=int,
        nargs=2,
        default=[150, 150],
        metavar=("HEIGHT", "WIDTH"),
        help="Image size expected by the model",
    )
    parser.add_argument("--output_csv", default="predictions.csv", help="Where to save predictions")
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1,
        help="Number of model output classes (1 for binary)",
    )
    args = parser.parse_args()

    df = predict_directory(
        args.model_path,
        args.data_dir,
        tuple(args.img_size),
        num_classes=args.num_classes,
    )
    df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
