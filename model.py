"""Compact TimeDistributed CNN + LSTM steering-angle regressor."""

from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow.keras import Model, layers


def build_model(config: dict[str, Any]) -> Model:
    """Build and compile the project's CNN+LSTM model."""
    image_cfg = config["preprocessing"]
    seq_cfg = config["sequences"]
    model_cfg = config["model"]

    input_shape = (
        int(seq_cfg["sequence_length"]),
        int(image_cfg["image_height"]),
        int(image_cfg["image_width"]),
        int(image_cfg["channels"]),
    )

    inputs = layers.Input(shape=input_shape, name="image_sequence")

    # The same CNN extracts spatial features independently from every frame.
    x = layers.TimeDistributed(
        layers.Conv2D(16, (5, 5), strides=(2, 2), activation="relu")
    )(inputs)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation="relu"))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation="relu"))(x)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)

    # LSTM combines the feature vectors across the temporal sequence.
    x = layers.LSTM(int(model_cfg["lstm_units"]))(x)
    x = layers.Dense(int(model_cfg["dense_units"]), activation="relu")(x)
    x = layers.Dropout(float(model_cfg["dropout_rate"]))(x)

    # Linear output keeps the original regression behaviour unchanged.
    outputs = layers.Dense(1, activation="linear", name="steering_angle")(x)

    model = Model(inputs, outputs, name="cnn_lstm_steering_regressor")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=float(config["training"]["learning_rate"])
        ),
        loss="mse",
        metrics=["mae"],
    )
    return model
