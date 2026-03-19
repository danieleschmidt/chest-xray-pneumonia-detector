"""Smoke tests for the training pipeline."""
import pytest
from src.train import TrainConfig, train


def test_train_smoke():
    """Tiny training run — checks the pipeline doesn't crash."""
    cfg = TrainConfig(
        n_samples=40,
        img_size=32,
        batch_size=10,
        epochs=2,
        device="cpu",
    )
    history, model = train(cfg)

    assert "train_loss" in history
    assert "train_acc" in history
    assert "val_loss" in history
    assert "val_acc" in history
    assert "test_acc" in history
    assert len(history["train_loss"]) == 2


def test_train_accuracy_bounded():
    cfg = TrainConfig(n_samples=40, img_size=32, batch_size=10, epochs=1, device="cpu")
    history, _ = train(cfg)
    for acc in history["train_acc"] + history["val_acc"]:
        assert 0.0 <= acc <= 1.0


def test_train_loss_decreasing_trend():
    """Loss should generally decrease over a short run on this simple task."""
    cfg = TrainConfig(
        n_samples=200,
        img_size=64,
        batch_size=32,
        epochs=5,
        lr=1e-3,
        device="cpu",
        seed=0,
    )
    history, _ = train(cfg)
    # First epoch loss > last epoch loss (with high probability)
    assert history["train_loss"][0] >= history["train_loss"][-1] - 0.15
