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
    assert b"--seed" in result.stdout
    assert b"--mlflow_tracking_uri" in result.stdout
    assert b"--history_csv" in result.stdout
    assert b"--rotation_range" in result.stdout
    assert b"--early_stopping_patience" in result.stdout
    assert b"--reduce_lr_factor" in result.stdout
    assert b"--reduce_lr_patience" in result.stdout
    assert b"--reduce_lr_min_lr" in result.stdout
    assert b"--class_weights" in result.stdout
    assert b"--learning_rate" in result.stdout
    assert b"--dropout_rate" in result.stdout
    assert b"--num_classes" in result.stdout
    assert b"--mlflow_experiment" in result.stdout
    assert b"--mlflow_run_name" in result.stdout
    assert b"--checkpoint_path" in result.stdout
    assert b"--save_model_path" in result.stdout
    assert b"--plot_path" in result.stdout
    assert b"--cm_path" in result.stdout
    assert b"--fine_tune_epochs" in result.stdout
    assert b"--fine_tune_lr" in result.stdout
    assert b"--resume_checkpoint" in result.stdout
