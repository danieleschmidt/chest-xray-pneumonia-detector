import pytest

pytest.importorskip("tensorflow")  # noqa: E402

from src.model_builder import (
    create_simple_cnn,
    create_transfer_learning_model,
)  # noqa: E402
from chest_xray_pneumonia_detector.pipeline import (
    TrainingConfig,
    run_training,
)  # noqa: E402
from src.data_loader import (  # noqa: E402
    create_dummy_images_for_generator,
    cleanup_dummy_data_for_generator,
)


def test_create_model():
    model1 = create_simple_cnn((32, 32, 3))
    assert hasattr(model1, "fit")
    assert model1.output_shape[-1] == 1

    model2 = create_transfer_learning_model((32, 32, 3))
    assert hasattr(model2, "fit")
    assert model2.output_shape[-1] == 1


def test_bad_config(tmp_path):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    create_dummy_images_for_generator(str(train_dir), num_images_per_class=1)
    create_dummy_images_for_generator(str(val_dir), num_images_per_class=1)
    config = TrainingConfig(
        train_dir=str(train_dir), val_dir=str(val_dir), model_type="invalid"
    )
    with pytest.raises(ValueError):
        run_training(config)
    cleanup_dummy_data_for_generator(str(train_dir))
    cleanup_dummy_data_for_generator(str(val_dir))
