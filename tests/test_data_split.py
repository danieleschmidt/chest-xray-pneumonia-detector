import subprocess
import sys
from pathlib import Path

from src.data_split import split_dataset


def test_data_split_cli_help():
    result = subprocess.run([sys.executable, "-m", "src.data_split", "--help"], capture_output=True)
    assert result.returncode == 0
    assert b"--val_frac" in result.stdout
    assert b"--move" in result.stdout


def test_split_dataset_counts(tmp_path):
    input_dir = tmp_path / "input"
    for cls in ["a", "b"]:
        class_dir = input_dir / cls
        class_dir.mkdir(parents=True)
        for i in range(10):
            (class_dir / f"img{i}.jpg").write_text("x")

    output_dir = tmp_path / "output"
    split_dataset(str(input_dir), str(output_dir), val_frac=0.2, test_frac=0.2, seed=123)

    for split in ["train", "val", "test"]:
        for cls in ["a", "b"]:
            path = output_dir / split / cls
            assert path.exists()

    train_count = len(list((output_dir / "train" / "a").glob("*.jpg")))
    val_count = len(list((output_dir / "val" / "a").glob("*.jpg")))
    test_count = len(list((output_dir / "test" / "a").glob("*.jpg")))

    assert train_count == 6  # 60% of 10
    assert val_count == 2    # 20% of 10
    assert test_count == 2   # 20% of 10


def test_split_dataset_move(tmp_path):
    input_dir = tmp_path / "input"
    for cls in ["a", "b"]:
        class_dir = input_dir / cls
        class_dir.mkdir(parents=True)
        for i in range(4):
            (class_dir / f"img{i}.jpg").write_text("x")

    output_dir = tmp_path / "output"
    split_dataset(str(input_dir), str(output_dir), val_frac=0.25, test_frac=0.25, seed=1, move=True)

    assert not any(input_dir.rglob("*.jpg"))
    moved_count = len(list((output_dir / "train" / "a").rglob("*.jpg")))
    assert moved_count == 2

