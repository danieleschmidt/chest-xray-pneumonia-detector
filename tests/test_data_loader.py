import pytest

tf = pytest.importorskip("tensorflow")
from src.data_loader import create_data_generators, create_tf_datasets  # noqa: E402


def test_missing_directories(tmp_path):
    with pytest.raises(FileNotFoundError):
        create_data_generators(str(tmp_path / "train"), str(tmp_path / "val"))


def test_create_tf_datasets(tmp_path):
    train_dir = tmp_path / "train/NORMAL"
    val_dir = tmp_path / "val/NORMAL"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    img_path = train_dir / "img0.png"
    tf.keras.preprocessing.image.save_img(img_path, tf.zeros((10, 10, 3)))
    img_path_val = val_dir / "img0.png"
    tf.keras.preprocessing.image.save_img(img_path_val, tf.zeros((10, 10, 3)))

    train_ds, val_ds = create_tf_datasets(
        str(tmp_path / "train"),
        str(tmp_path / "val"),
        image_size=(10, 10),
        batch_size=1,
    )
    batch = next(iter(train_ds))
    assert batch[0].shape == (1, 10, 10, 3)
    batch_val = next(iter(val_ds))
    assert batch_val[0].shape == (1, 10, 10, 3)
