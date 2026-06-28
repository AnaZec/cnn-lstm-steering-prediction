"""CNN+LSTM model for steering-angle regression.

The model accepts a sequence of preprocessed front-camera frames and predicts
one floating-point steering-angle value.

Input shape:
    (sequence_length, image_height, image_width, channels)

Output shape:
    (1,)
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow.keras import Model, layers

from src.data import load_config


def get_model_input_shape(config: dict[str, Any]) -> tuple[int, int, int, int]:
    """Build the CNN+LSTM input shape from config.yaml.

    Args:
        config: Parsed project configuration.

    Returns:
        Tuple:
            (sequence_length, image_height, image_width, channels)

    Raises:
        ValueError: If any required dimension is invalid.
    """
    sequence_config = config.get("sequences", {})
    preprocessing_config = config.get("preprocessing", {})

    sequence_length = int(sequence_config.get("sequence_length", 5))
    image_height = int(preprocessing_config.get("image_height", 80))
    image_width = int(preprocessing_config.get("image_width", 160))
    channels = int(preprocessing_config.get("channels", 3))

    if sequence_length < 1:
        raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")

    if image_height < 1:
        raise ValueError(f"image_height must be >= 1, got {image_height}")

    if image_width < 1:
        raise ValueError(f"image_width must be >= 1, got {image_width}")

    if channels != 3:
        raise ValueError(
            "Only 3-channel RGB input is supported in this project. "
            f"Got channels={channels}."
        )

    return sequence_length, image_height, image_width, channels


def build_cnn_lstm_model(
    input_shape: tuple[int, int, int, int],
    *,
    learning_rate: float = 0.001,
    lstm_units: int = 64,
    dense_units: int = 32,
    dropout_rate: float = 0.2,
) -> Model:
    """Build and compile a compact CNN+LSTM steering-angle regression model.

    Args:
        input_shape: Model input shape without batch dimension:
            (sequence_length, image_height, image_width, channels).
        learning_rate: Adam optimizer learning rate.
        lstm_units: Number of LSTM units.
        dense_units: Number of hidden units in the regression head.
        dropout_rate: Dropout rate before the final regression output.

    Returns:
        Compiled Keras model.

    Raises:
        ValueError: If model hyperparameters are invalid.
    """
    if len(input_shape) != 4:
        raise ValueError(
            "input_shape must be "
            "(sequence_length, image_height, image_width, channels), "
            f"got {input_shape}"
        )

    sequence_length, image_height, image_width, channels = input_shape

    if sequence_length < 1 or image_height < 1 or image_width < 1:
        raise ValueError(f"Invalid input shape: {input_shape}")

    if channels != 3:
        raise ValueError(
            "Only 3-channel RGB input is supported in this project. "
            f"Got channels={channels}."
        )

    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")

    if lstm_units < 1:
        raise ValueError(f"lstm_units must be >= 1, got {lstm_units}")

    if dense_units < 1:
        raise ValueError(f"dense_units must be >= 1, got {dense_units}")

    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")

    inputs = layers.Input(shape=input_shape, name="image_sequence")

    x = layers.TimeDistributed(
        layers.Conv2D(16, kernel_size=(5, 5), strides=(2, 2), activation="relu"),
        name="td_conv_1",
    )(inputs)
    x = layers.TimeDistributed(
        layers.MaxPooling2D(pool_size=(2, 2)),
        name="td_pool_1",
    )(x)

    x = layers.TimeDistributed(
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        name="td_conv_2",
    )(x)
    x = layers.TimeDistributed(
        layers.MaxPooling2D(pool_size=(2, 2)),
        name="td_pool_2",
    )(x)

    x = layers.TimeDistributed(
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        name="td_conv_3",
    )(x)

    x = layers.TimeDistributed(
        layers.GlobalAveragePooling2D(),
        name="td_global_average_pool",
    )(x)

    x = layers.LSTM(lstm_units, name="lstm")(x)

    x = layers.Dense(dense_units, activation="relu", name="dense_regression")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)

    outputs = layers.Dense(1, activation="linear", name="steering_angle")(x)

    model = Model(inputs=inputs, outputs=outputs, name="cnn_lstm_steering_regressor")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"],
    )

    return model


def build_model_from_config(config: dict[str, Any]) -> Model:
    """Build a compiled CNN+LSTM model from config.yaml settings.

    Args:
        config: Parsed project configuration.

    Returns:
        Compiled Keras model.
    """
    input_shape = get_model_input_shape(config)
    training_config = config.get("training", {})

    learning_rate = float(training_config.get("learning_rate", 0.001))

    model_config = config.get("model", {})
    lstm_units = int(model_config.get("lstm_units", 64))
    dense_units = int(model_config.get("dense_units", 32))
    dropout_rate = float(model_config.get("dropout_rate", 0.2))

    return build_cnn_lstm_model(
        input_shape,
        learning_rate=learning_rate,
        lstm_units=lstm_units,
        dense_units=dense_units,
        dropout_rate=dropout_rate,
    )


def main() -> None:
    """Small CLI smoke test for model construction."""
    config = load_config("config.yaml")
    model = build_model_from_config(config)

    model.summary()

    print()
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Total parameters: {model.count_params()}")


if __name__ == "__main__":
    main()