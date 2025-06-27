"""Utility for splitting a dataset into train, val and test folders."""

import argparse
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_dataset(
    input_dir: str,
    output_dir: str,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    move: bool = False,
) -> None:
    """Split a dataset into train/val/test directories.

    Parameters
    ----------
    input_dir : str
        Path to a directory with subfolders per class containing images.
    output_dir : str
        Destination directory where ``train``, ``val`` and ``test`` folders will be created.
    val_frac : float, optional
        Fraction of images to use for validation.
    test_frac : float, optional
        Fraction of images to use for testing.
    seed : int, optional
        Random seed for reproducible splits.
    move : bool, optional
        If True, move files instead of copying them.
    """

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not 0 <= val_frac < 1:
        raise ValueError("val_frac must be between 0 and 1")
    if not 0 <= test_frac < 1:
        raise ValueError("test_frac must be between 0 and 1")
    if val_frac + test_frac >= 1:
        raise ValueError("val_frac + test_frac must be less than 1")

    class_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class folders found in {input_dir}")

    for subset in ["train", "val", "test"]:
        for class_dir in class_dirs:
            (output_path / subset / class_dir.name).mkdir(parents=True, exist_ok=True)

    for class_dir in class_dirs:
        images = list(class_dir.glob("*"))
        train_val, test = train_test_split(
            images, test_size=test_frac, random_state=seed
        )
        train, val = train_test_split(
            train_val, test_size=val_frac / (1 - test_frac), random_state=seed
        )
        splits = {"train": train, "val": val, "test": test}
        for split_name, files in splits.items():
            dest = output_path / split_name / class_dir.name
            for f in files:
                target = dest / f.name
                if move:
                    shutil.move(f, target)
                else:
                    shutil.copy2(f, target)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a dataset into train/val/test directories"
    )
    parser.add_argument(
        "--input_dir", required=True, help="Directory with class subfolders"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Where to create split dataset"
    )
    parser.add_argument(
        "--val_frac", type=float, default=0.1, help="Fraction for validation set"
    )
    parser.add_argument(
        "--test_frac", type=float, default=0.1, help="Fraction for test set"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them",
    )
    args = parser.parse_args()

    split_dataset(
        args.input_dir,
        args.output_dir,
        args.val_frac,
        args.test_frac,
        args.seed,
        args.move,
    )


if __name__ == "__main__":
    main()
