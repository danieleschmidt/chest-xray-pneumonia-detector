import subprocess
import sys
import pytest

from src.dataset_stats import count_images_per_class


def test_dataset_stats_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src.dataset_stats", "--help"], capture_output=True
    )
    assert result.returncode == 0
    assert b"input_dir" in result.stdout
    assert b"extensions" in result.stdout


def test_count_images_per_class(tmp_path) -> None:
    class_counts = {"a": 3, "b": 2}
    for cls, n in class_counts.items():
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for i in range(n):
            (cls_dir / f"img{i}.jpg").write_text("x")
    counts = count_images_per_class(str(tmp_path))
    assert counts == class_counts


def test_count_images_custom_extensions(tmp_path) -> None:
    cls_dir = tmp_path / "a"
    cls_dir.mkdir()
    (cls_dir / "img1.bmp").write_text("x")
    counts = count_images_per_class(str(tmp_path), extensions=[".bmp"])
    assert counts == {"a": 1}


def test_count_images_extension_normalization(tmp_path) -> None:
    cls_dir = tmp_path / "a"
    cls_dir.mkdir()
    (cls_dir / "img1.PNG").write_text("x")
    counts = count_images_per_class(str(tmp_path), extensions=["PNG"])
    assert counts == {"a": 1}


def test_dataset_stats_csv_output(tmp_path) -> None:
    (tmp_path / "cls").mkdir()
    (tmp_path / "cls" / "img.jpg").write_text("x")
    csv_path = tmp_path / "counts.csv"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.dataset_stats",
            "--input_dir",
            str(tmp_path),
            "--csv_output",
            str(csv_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert csv_path.read_text().strip().splitlines()[1] == "cls,1"


def test_dataset_stats_json_output(tmp_path) -> None:
    (tmp_path / "cls").mkdir()
    (tmp_path / "cls" / "img.jpg").write_text("x")
    json_path = tmp_path / "counts.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.dataset_stats",
            "--input_dir",
            str(tmp_path),
            "--json_output",
            str(json_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert json_path.read_text().strip() == "{\n  \"cls\": 1\n}"


def test_count_images_missing_dir(tmp_path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        count_images_per_class(str(missing))


def test_count_images_path_is_file(tmp_path) -> None:
    file_path = tmp_path / "file.txt"
    file_path.write_text("x")
    with pytest.raises(NotADirectoryError):
        count_images_per_class(str(file_path))


def test_count_images_no_classes(tmp_path) -> None:
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(ValueError):
        count_images_per_class(str(tmp_path))

