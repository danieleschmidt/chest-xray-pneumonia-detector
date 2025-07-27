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


def test_contrast_range_validation(tmp_path):
    """Test contrast_range parameter validation in create_data_generators."""
    # Create minimal directory structure for tests
    train_dir = tmp_path / "train/NORMAL"
    val_dir = tmp_path / "val/NORMAL"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    
    # Test valid contrast_range values
    valid_values = [0.0, 0.1, 0.5, 1.0]
    for contrast_val in valid_values:
        try:
            create_data_generators(
                str(tmp_path / "train"),
                str(tmp_path / "val"),
                contrast_range=contrast_val
            )
        except ValueError as e:
            pytest.fail(f"Valid contrast_range {contrast_val} raised ValueError: {e}")
    
    # Test invalid contrast_range values (negative)
    with pytest.raises(ValueError, match=r"contrast_range must be between 0.0 and 1.0"):
        create_data_generators(
            str(tmp_path / "train"),
            str(tmp_path / "val"),
            contrast_range=-0.1
        )
    
    # Test invalid contrast_range values (greater than 1.0)
    with pytest.raises(ValueError, match=r"contrast_range must be between 0.0 and 1.0"):
        create_data_generators(
            str(tmp_path / "train"),
            str(tmp_path / "val"),
            contrast_range=1.5
        )
    
    # Test edge case with very large invalid value
    with pytest.raises(ValueError, match=r"contrast_range must be between 0.0 and 1.0"):
        create_data_generators(
            str(tmp_path / "train"),
            str(tmp_path / "val"),
            contrast_range=10.0
        )
