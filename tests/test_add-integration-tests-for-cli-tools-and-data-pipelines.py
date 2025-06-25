import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("tensorflow")

from src.data_loader import create_dummy_images_for_generator
from src.model_builder import create_simple_cnn


def test_cli_success(tmp_path):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    create_dummy_images_for_generator(str(train_dir), num_images_per_class=1)
    create_dummy_images_for_generator(str(val_dir), num_images_per_class=1)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "chest_xray_pneumonia_detector.pipeline",
            "--train_dir",
            str(train_dir),
            "--val_dir",
            str(val_dir),
            "--epochs",
            "1",
            "--batch_size",
            "1",
            "--img_size",
            "32",
            "32",
            "--model_type",
            "simple",
        ],
        capture_output=True,
    )
    assert result.returncode == 0


def test_bad_input(tmp_path):
    data_dir = tmp_path / "data"
    create_dummy_images_for_generator(str(data_dir), num_images_per_class=1)
    (data_dir / "class_a" / "extra.txt").write_text("hello")
    model = create_simple_cnn((32, 32, 3))
    model_path = tmp_path / "model.keras"
    model.save(model_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.inference",
            "--model_path",
            str(model_path),
            "--data_dir",
            str(data_dir),
            "--img_size",
            "32",
            "32",
        ],
        capture_output=True,
    )
    assert result.returncode == 0
    assert Path("predictions.csv").exists()
