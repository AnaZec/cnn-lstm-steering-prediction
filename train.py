"""Training pipeline."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from dataset import load_config, prepare_data
from model import build_model


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _load_history(path: Path) -> dict[str, list[float]]:
    """Load an existing Keras history file, if one is available."""
    if not path.is_file():
        return {}

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return {
        name: [float(value) for value in values]
        for name, values in data.items()
        if isinstance(values, list)
    }


def _history_until_best_checkpoint(
    history: dict[str, list[float]],
) -> tuple[dict[str, list[float]], float | None]:
    """Trim history to the epoch represented by the best validation checkpoint."""
    val_loss = history.get("val_loss", [])
    if not val_loss:
        return history, None

    best_index = int(np.argmin(val_loss))
    trimmed = {
        name: values[: best_index + 1]
        for name, values in history.items()
    }
    return trimmed, float(val_loss[best_index])


def _merge_histories(
    previous: dict[str, list[float]],
    current: dict[str, list[float]],
) -> dict[str, list[float]]:
    """Append the metrics from a resumed fit call to the existing history."""
    merged: dict[str, list[float]] = {}
    for name in previous.keys() | current.keys():
        merged[name] = previous.get(name, []) + current.get(name, [])
    return merged


def train(config_path: str = "config.yaml", resume: bool = False) -> None:
    config = load_config(config_path)
    outputs = config["outputs"]
    _set_seed(int(config["split"]["random_seed"]))

    checkpoint = Path(outputs["checkpoint_path"])
    final_model = Path(outputs["final_model_path"])
    history_path = Path(outputs["history_path"])
    for path in (checkpoint, final_model, history_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    train_index, validation_index, train_generator, validation_generator = prepare_data(
        config
    )

    previous_history: dict[str, list[float]] = {}
    previous_best: float | None = None
    initial_epoch = 0

    if resume:
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Cannot resume training: checkpoint not found at {checkpoint}"
            )

        model = tf.keras.models.load_model(checkpoint)
        previous_history = _load_history(history_path)
        previous_history, previous_best = _history_until_best_checkpoint(
            previous_history
        )
        initial_epoch = max(
            (len(values) for values in previous_history.values()),
            default=0,
        )

        # If the history file is unavailable, evaluate the loaded best checkpoint so
        # ModelCheckpoint still knows which validation loss it must improve on.
        if previous_best is None:
            baseline: dict[str, Any] = model.evaluate(
                validation_generator,
                verbose=0,
                return_dict=True,
            )
            previous_best = float(baseline["loss"])

        mode = "RESUMED"
    else:
        model = build_model(config)
        mode = "NEW"

    additional_epochs = int(config["training"]["epochs"])
    final_epoch = initial_epoch + additional_epochs

    print(f"\n=== CNN + LSTM TRAINING ({mode}) ===")
    if resume:
        print(f"Loaded checkpoint:     {checkpoint}")
        print(f"Previous best val_loss: {previous_best:.6f}")
        print(f"Continuing from epoch: {initial_epoch + 1}")
    print(f"Training sequences:    {len(train_index)}")
    print(f"Validation sequences:  {len(validation_index)}")
    print(f"Model parameters:       {model.count_params():,}")
    print(f"Epochs this run:         {additional_epochs}")
    print()
    model.summary()

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        str(checkpoint),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    )

    # Prevent a resumed run from replacing the existing best checkpoint with a
    # worse model on its first epoch.
    if resume and previous_best is not None:
        checkpoint_callback.best = previous_best

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=3,
        baseline=previous_best if resume else None,
        restore_best_weights=True,
        verbose=1,
    )

    fit_history = model.fit(
        train_generator,
        validation_data=validation_generator,
        initial_epoch=initial_epoch,
        epochs=final_epoch,
        callbacks=[checkpoint_callback, early_stopping],
        verbose=1,
    )

    current_history = {
        name: [float(value) for value in values]
        for name, values in fit_history.history.items()
    }
    full_history = _merge_histories(previous_history, current_history)

    with history_path.open("w", encoding="utf-8") as file:
        json.dump(full_history, file, indent=2)

    # The checkpoint is always the best validation model across the original and
    # resumed runs. Save the same model as final_model for consistent downstream use.
    best_model = tf.keras.models.load_model(checkpoint)
    best_model.save(final_model)

    print("\nTraining complete.")
    print(f"Best model:     {checkpoint}")
    print(f"Training history: {history_path}")
