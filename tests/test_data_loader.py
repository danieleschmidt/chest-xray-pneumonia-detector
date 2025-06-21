import pytest

tf = pytest.importorskip("tensorflow")
from src.data_loader import create_data_generators


def test_missing_directories(tmp_path):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_gen, val_gen = create_data_generators(str(train_dir), str(val_dir))
    assert train_gen is None and val_gen is None
