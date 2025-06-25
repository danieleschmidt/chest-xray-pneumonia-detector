import pytest

pytest.importorskip("tensorflow")

from src.data_loader import (
    create_data_generators,
    create_dummy_images_for_generator,
)


def test_load_success(tmp_path):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    create_dummy_images_for_generator(str(train_dir), num_images_per_class=1)
    create_dummy_images_for_generator(str(val_dir), num_images_per_class=1)

    train_gen, val_gen = create_data_generators(
        str(train_dir),
        str(val_dir),
        target_size=(32, 32),
        train_batch_size=1,
        val_batch_size=1,
    )

    batch_x, batch_y = next(train_gen)
    assert batch_x.shape == (1, 32, 32, 3)
    assert batch_y.shape == (1,)
    assert val_gen.samples > 0


def test_invalid_path():
    with pytest.raises(FileNotFoundError):
        create_data_generators("nonexistent/train", "nonexistent/val")
