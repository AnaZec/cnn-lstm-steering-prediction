"""Training pipeline."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset import load_config, prepare_data
from model import build_model


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    outputs = config["outputs"]
    _set_seed(int(config["split"]["random_seed"]))

    train_index, validation_index, train_generator, validation_generator = prepare_data(
        config
    )
    model = build_model(config)

    checkpoint = Path(outputs["checkpoint_path"])
    final_model = Path(outputs["final_model_path"])
    history_path = Path(outputs["history_path"])
    for path in (checkpoint, final_model, history_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    print("\n=== CNN + LSTM TRAINING ===")
    print(f"Training sequences:   {len(train_index)}")
    print(f"Validation sequences: {len(validation_index)}")
    print(f"Model parameters:      {model.count_params():,}")
    print()
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint), monitor="val_loss", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
        ),
    ]

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=int(config["training"]["epochs"]),
        callbacks=callbacks,
        verbose=1,
    )

    model.save(final_model)
    with history_path.open("w", encoding="utf-8") as file:
        json.dump(
            {name: [float(v) for v in values] for name, values in history.history.items()},
            file,
            indent=2,
        )

    print("\nTraining complete.")
    print(f"Best model: {checkpoint}")
