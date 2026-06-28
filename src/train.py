"""Training entry point for CNN+LSTM steering-angle prediction.

This script loads the configured Udacity driving samples, builds temporal
CNN+LSTM input sequences, trains the steering-angle regression model, and saves
training artifacts for later evaluation, plotting, inference, and reporting.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from src.data import (
    load_config,
    load_driving_samples,
    split_train_validation_samples,
)
from src.model import build_model_from_config
from src.sequences import (
    build_sequence_index,
    create_sequence_generator_from_config,
    get_sequence_config,
)


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducible local training behavior.

    Full determinism is not guaranteed across all TensorFlow operations and
    hardware backends, but this reduces run-to-run variation.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_parent_directory(path: str | Path) -> None:
    """Create the parent directory for an output file if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_training_history(history: tf.keras.callbacks.History, path: str | Path) -> None:
    """Save Keras training history as readable JSON.

    Args:
        history: History object returned by model.fit.
        path: Destination JSON path.
    """
    ensure_parent_directory(path)

    serializable_history = {
        metric_name: [float(value) for value in metric_values]
        for metric_name, metric_values in history.history.items()
    }

    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(serializable_history, file, indent=2)


def build_training_artifacts(
    config: dict[str, Any],
    *,
    config_path: str | Path,
) -> tuple[
    tf.keras.Model,
    Any,
    Any,
    Any,
    Any,
]:
    """Build model, train/validation generators, and sequence indices.

    Args:
        config: Parsed project configuration.
        config_path: Path to the configuration file used for data loading.

    Returns:
        Tuple containing:
            model,
            train_generator,
            validation_generator,
            train_sequence_index,
            validation_sequence_index
    """
    sequence_config = get_sequence_config(config)
    split_config = config.get("split", {})

    samples = load_driving_samples(config_path)

    train_samples, validation_samples = split_train_validation_samples(
        samples,
        validation_ratio=float(split_config.get("validation_ratio", 0.2)),
        sequence_length=sequence_config["sequence_length"],
    )

    train_sequence_index = build_sequence_index(
        train_samples,
        sequence_length=sequence_config["sequence_length"],
        stride=sequence_config["stride"],
        target=sequence_config["target"],
    )

    validation_sequence_index = build_sequence_index(
        validation_samples,
        sequence_length=sequence_config["sequence_length"],
        stride=sequence_config["stride"],
        target=sequence_config["target"],
    )

    train_generator = create_sequence_generator_from_config(
        train_sequence_index,
        config,
        shuffle=True,
    )

    validation_generator = create_sequence_generator_from_config(
        validation_sequence_index,
        config,
        shuffle=False,
    )

    model = build_model_from_config(config)

    return (
        model,
        train_generator,
        validation_generator,
        train_sequence_index,
        validation_sequence_index,
    )


def train_model(config_path: str | Path) -> tf.keras.callbacks.History:
    """Train the CNN+LSTM steering-angle model from configuration.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Keras History object.
    """
    config = load_config(config_path)

    split_config = config.get("split", {})
    training_config = config.get("training", {})
    outputs_config = config.get("outputs", {})

    random_seed = int(split_config.get("random_seed", 42))
    epochs = int(training_config.get("epochs", 10))

    checkpoint_path = Path(
        outputs_config.get("checkpoint_path", "outputs/models/best_model.keras")
    )
    final_model_path = Path(
        outputs_config.get("final_model_path", "outputs/models/final_model.keras")
    )
    history_path = Path(
        outputs_config.get("history_path", "outputs/models/training_history.json")
    )

    ensure_parent_directory(checkpoint_path)
    ensure_parent_directory(final_model_path)
    ensure_parent_directory(history_path)

    set_random_seed(random_seed)

    (
        model,
        train_generator,
        validation_generator,
        train_sequence_index,
        validation_sequence_index,
    ) = build_training_artifacts(config, config_path=config_path)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    print("Training configuration")
    print(f"Epochs: {epochs}")
    print(f"Training sequences: {len(train_sequence_index)}")
    print(f"Validation sequences: {len(validation_sequence_index)}")
    print(f"Best checkpoint path: {checkpoint_path}")
    print(f"Final model path: {final_model_path}")
    print(f"Training history path: {history_path}")
    print()

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(final_model_path)
    save_training_history(history, history_path)

    print()
    print("Training artifacts saved")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Final model: {final_model_path}")
    print(f"Training history: {history_path}")

    return history


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the CNN+LSTM steering-angle regression model."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to project configuration YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run model training from the command line."""
    args = parse_args()
    train_model(args.config)


if __name__ == "__main__":
    main()