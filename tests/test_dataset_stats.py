import subprocess
import sys
import pytest

from src import dataset_stats
from src.dataset_stats import count_images_per_class


def test_dataset_stats_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src.dataset_stats", "--help"], capture_output=True
    )
    assert result.returncode == 0
    assert b"input_dir" in result.stdout
    assert b"extensions" in result.stdout
    assert b"--plot_png" in result.stdout
    assert b"--sort_by" in result.stdout


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


def test_dataset_stats_sort_by_count(tmp_path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "img1.jpg").write_text("x")
    (tmp_path / "a" / "img2.jpg").write_text("x")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "img.jpg").write_text("x")
    csv_path = tmp_path / "counts.csv"
    json_path = tmp_path / "counts.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.dataset_stats",
            "--input_dir",
            str(tmp_path),
            "--csv_output",
            str(csv_path),
            "--json_output",
            str(json_path),
            "--sort_by",
            "count",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    lines = csv_path.read_text().strip().splitlines()
    assert lines[1].startswith("a,")  # a has 2 images
    json_data = json_path.read_text().splitlines()
    assert json_data[1].strip().startswith('"a"')


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


def test_dataset_stats_plot_output(tmp_path) -> None:
    matplotlib = pytest.importorskip("matplotlib")  # noqa: F841
    (tmp_path / "cls").mkdir()
    (tmp_path / "cls" / "img.jpg").write_text("x")
    plot_path = tmp_path / "counts.png"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.dataset_stats",
            "--input_dir",
            str(tmp_path),
            "--plot_png",
            str(plot_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert plot_path.exists()


def test_dataset_stats_plot_missing_dep(tmp_path, monkeypatch, capsys) -> None:
    (tmp_path / "cls").mkdir()
    (tmp_path / "cls" / "img.jpg").write_text("x")
    def fake_import(name, *args, **kwargs):
        if name == "matplotlib.pyplot":
            raise ImportError
        return orig_import(name, *args, **kwargs)

    import builtins

    orig_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(SystemExit) as exc:
        dataset_stats.main([
            "--input_dir",
            str(tmp_path),
            "--plot_png",
            str(tmp_path / "out.png"),
        ])
    assert exc.value.code == 2
    assert "matplotlib is required for plotting" in capsys.readouterr().err


def test_dataset_stats_plot_missing_dir(tmp_path) -> None:
    matplotlib = pytest.importorskip("matplotlib")  # noqa: F841
    (tmp_path / "cls").mkdir()
    (tmp_path / "cls" / "img.jpg").write_text("x")
    missing_dir = tmp_path / "missing"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.dataset_stats",
            "--input_dir",
            str(tmp_path),
            "--plot_png",
            str(missing_dir / "out.png"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Output directory" in result.stderr


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

