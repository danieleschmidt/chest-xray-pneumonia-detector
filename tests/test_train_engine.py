import pytest
import subprocess
import sys

tf = pytest.importorskip("tensorflow")
from src.train_engine import create_dummy_data, cleanup_dummy_data


def test_dummy_data_creation_and_cleanup(tmp_path):
    base_dir = tmp_path / "dummy"
    create_dummy_data(base_dir=str(base_dir), num_images_per_class=1)
    assert (base_dir / "train").exists()
    cleanup_dummy_data(base_dir=str(base_dir))
    assert not base_dir.exists()


def test_train_engine_cli_help():
    result = subprocess.run([sys.executable, "-m", "src.train_engine", "--help"], capture_output=True)
    assert result.returncode == 0
