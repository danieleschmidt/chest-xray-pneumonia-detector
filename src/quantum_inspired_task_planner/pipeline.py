from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf

# These modules are installed as top-level modules via ``pyproject.toml``.
from data_loader import create_data_generators
from model_builder import (
    create_simple_cnn,
    create_transfer_learning_model,
)


@dataclass
class TrainingConfig:
    """Configuration for a basic training run."""

    train_dir: str
    val_dir: str
    img_size: Tuple[int, int] = (150, 150)
    batch_size: int = 32
    epochs: int = 5
    num_classes: int = 1
    model_type: str = "transfer"  # "simple" or "transfer"
    base_model_name: str = "MobileNetV2"
    learning_rate: float = 1e-3
    dropout_rate: float = 0.0

    def input_shape(self) -> Tuple[int, int, int]:
        return (*self.img_size, 3)


def run_training(config: TrainingConfig) -> tf.keras.Model:
    """Run a minimal training pipeline using Keras."""

    train_gen, val_gen = create_data_generators(
        config.train_dir,
        config.val_dir,
        target_size=config.img_size,
        train_batch_size=config.batch_size,
        val_batch_size=config.batch_size,
        class_mode="binary" if config.num_classes == 1 else "categorical",
    )

    if train_gen is None or val_gen is None:
        raise RuntimeError("Failed to create data generators")

    if config.model_type == "simple":
        model = create_simple_cnn(
            input_shape=config.input_shape(),
            num_classes=config.num_classes,
            learning_rate=config.learning_rate,
            dropout_rate=config.dropout_rate,
        )
    elif config.model_type == "transfer":
        model = create_transfer_learning_model(
            input_shape=config.input_shape(),
            num_classes=config.num_classes,
            base_model_name=config.base_model_name,
            learning_rate=config.learning_rate,
            dropout_rate=config.dropout_rate,
        )
    else:
        raise ValueError("model_type must be 'simple' or 'transfer'")

    model.fit(
        train_gen,
        epochs=config.epochs,
        validation_data=val_gen,
    )

    return model


def _parse_args(args: list[str] | None = None) -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Run a basic training pipeline")
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, nargs=2, default=[150, 150])
    parser.add_argument(
        "--model_type",
        choices=["simple", "transfer"],
        default="transfer",
    )
    parser.add_argument("--base_model_name", default="MobileNetV2")
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parsed = parser.parse_args(args)
    return TrainingConfig(
        train_dir=parsed.train_dir,
        val_dir=parsed.val_dir,
        img_size=tuple(parsed.img_size),
        batch_size=parsed.batch_size,
        epochs=parsed.epochs,
        num_classes=parsed.num_classes,
        model_type=parsed.model_type,
        base_model_name=parsed.base_model_name,
        learning_rate=parsed.learning_rate,
        dropout_rate=parsed.dropout_rate,
    )


def main(args: list[str] | None = None) -> None:
    config = _parse_args(args)
    run_training(config)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
