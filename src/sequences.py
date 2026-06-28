"""Sequence construction utilities for CNN+LSTM steering-angle prediction.

This module converts ordered frame samples into fixed-length temporal sequences.
It keeps sequence construction deterministic and memory-conscious by building a
lightweight sequence index first, then loading/preprocessing image batches on
demand for Keras training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence

from src.data import (
    load_config,
    load_driving_samples,
    split_train_validation_samples,
)
from src.preprocessing import get_preprocessing_config, preprocess_image


def get_sequence_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract sequence settings from the project configuration.

    Args:
        config: Parsed project configuration dictionary.

    Returns:
        Dictionary containing sequence_length, stride, and target.

    Raises:
        ValueError: If the sequence settings are invalid.
    """
    sequence_config = config.get("sequences", {})

    sequence_length = int(sequence_config.get("sequence_length", 5))
    stride = int(sequence_config.get("stride", 1))
    target = str(sequence_config.get("target", "last_frame"))

    if sequence_length < 1:
        raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")

    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    if target != "last_frame":
        raise ValueError(
            "Only target='last_frame' is currently supported. "
            f"Got target={target!r}."
        )

    return {
        "sequence_length": sequence_length,
        "stride": stride,
        "target": target,
    }


def build_sequence_index(
    samples: pd.DataFrame,
    *,
    sequence_length: int,
    stride: int = 1,
    target: str = "last_frame",
) -> pd.DataFrame:
    """Build a lightweight sliding-window sequence index.

    The returned DataFrame does not contain image tensors. It contains ordered
    frame paths for each sequence and one steering-angle target per sequence.

    Args:
        samples: Ordered DataFrame with image_path and steering_angle columns.
        sequence_length: Number of consecutive frames per sequence.
        stride: Step size between consecutive sequence windows.
        target: Target assignment strategy. Currently only 'last_frame' is
            supported.

    Returns:
        DataFrame with columns:
            - start_index
            - end_index
            - frame_paths
            - target_steering_angle

    Raises:
        ValueError: If inputs are invalid or too few samples are provided.
    """
    required_columns = {"image_path", "steering_angle"}
    missing_columns = required_columns.difference(samples.columns)

    if missing_columns:
        raise ValueError(
            f"samples is missing required column(s): {sorted(missing_columns)}"
        )

    if sequence_length < 1:
        raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")

    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    if target != "last_frame":
        raise ValueError(
            "Only target='last_frame' is currently supported. "
            f"Got target={target!r}."
        )

    if len(samples) < sequence_length:
        raise ValueError(
            "Not enough samples to build one sequence. "
            f"Need at least {sequence_length}, got {len(samples)}."
        )

    rows: list[dict[str, Any]] = []

    for start_index in range(0, len(samples) - sequence_length + 1, stride):
        end_index = start_index + sequence_length
        window = samples.iloc[start_index:end_index]

        frame_paths = window["image_path"].tolist()
        target_steering_angle = float(window.iloc[-1]["steering_angle"])

        rows.append(
            {
                "start_index": start_index,
                "end_index": end_index - 1,
                "frame_paths": frame_paths,
                "target_steering_angle": target_steering_angle,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "start_index",
            "end_index",
            "frame_paths",
            "target_steering_angle",
        ],
    )


def preprocess_frame_sequence(
    frame_paths: list[str | Path],
    *,
    image_width: int,
    image_height: int,
    channels: int,
    normalize: bool,
) -> np.ndarray:
    """Preprocess one ordered list of frame paths into a sequence tensor.

    Args:
        frame_paths: Ordered frame paths belonging to one temporal sequence.
        image_width: Target image width.
        image_height: Target image height.
        channels: Number of image channels.
        normalize: Whether to normalize pixel values to [0, 1].

    Returns:
        Array with shape:
            (sequence_length, image_height, image_width, channels)
    """
    frames = [
        preprocess_image(
            frame_path,
            image_width=image_width,
            image_height=image_height,
            channels=channels,
            normalize=normalize,
        )
        for frame_path in frame_paths
    ]

    return np.stack(frames, axis=0)


class SteeringSequenceGenerator(Sequence):
    """Keras-compatible generator for CNN+LSTM steering-angle training.

    Each batch has:
        X shape: (batch_size, sequence_length, height, width, channels)
        y shape: (batch_size,)

    Frame order inside each sequence is preserved.
    """

    def __init__(
        self,
        sequence_index: pd.DataFrame,
        *,
        batch_size: int,
        image_width: int,
        image_height: int,
        channels: int,
        normalize: bool,
        shuffle: bool = False,
        random_seed: int = 42,
    ) -> None:
        """Initialize the sequence generator."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        required_columns = {"frame_paths", "target_steering_angle"}
        missing_columns = required_columns.difference(sequence_index.columns)

        if missing_columns:
            raise ValueError(
                "sequence_index is missing required column(s): "
                f"{sorted(missing_columns)}"
            )

        if sequence_index.empty:
            raise ValueError("sequence_index must not be empty.")

        self.sequence_index = sequence_index.reset_index(drop=True)
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height
        self.channels = channels
        self.normalize = normalize
        self.shuffle = shuffle
        self.rng = np.random.default_rng(random_seed)
        self.indices = np.arange(len(self.sequence_index))

        self.on_epoch_end()

    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return int(np.ceil(len(self.sequence_index) / self.batch_size))

    def __getitem__(self, batch_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return one batch of sequence tensors and steering targets."""
        start = batch_index * self.batch_size
        end = min(start + self.batch_size, len(self.sequence_index))

        batch_indices = self.indices[start:end]
        batch_rows = self.sequence_index.iloc[batch_indices]

        x_batch = []
        y_batch = []

        for _, row in batch_rows.iterrows():
            sequence = preprocess_frame_sequence(
                row["frame_paths"],
                image_width=self.image_width,
                image_height=self.image_height,
                channels=self.channels,
                normalize=self.normalize,
            )

            x_batch.append(sequence)
            y_batch.append(float(row["target_steering_angle"]))

        return np.stack(x_batch, axis=0), np.asarray(y_batch, dtype=np.float32)

    def on_epoch_end(self) -> None:
        """Shuffle sequence order between epochs if enabled."""
        if self.shuffle:
            self.rng.shuffle(self.indices)


def create_sequence_generator_from_config(
    sequence_index: pd.DataFrame,
    config: dict[str, Any],
    *,
    batch_size: int | None = None,
    shuffle: bool = False,
) -> SteeringSequenceGenerator:
    """Create a Keras-compatible sequence generator using config.yaml settings."""
    preprocessing = get_preprocessing_config(config)
    training_config = config.get("training", {})

    resolved_batch_size = int(
        batch_size if batch_size is not None else training_config.get("batch_size", 32)
    )

    split_config = config.get("split", {})
    random_seed = int(split_config.get("random_seed", 42))

    return SteeringSequenceGenerator(
        sequence_index,
        batch_size=resolved_batch_size,
        image_width=preprocessing["image_width"],
        image_height=preprocessing["image_height"],
        channels=preprocessing["channels"],
        normalize=preprocessing["normalize"],
        shuffle=shuffle,
        random_seed=random_seed,
    )


def main() -> None:
    """Smoke test sequence construction and one Keras-compatible batch."""
    config = load_config("config.yaml")
    sequence_config = get_sequence_config(config)

    samples = load_driving_samples("config.yaml")

    split_config = config.get("split", {})
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
        shuffle=False,
    )

    x_batch, y_batch = train_generator[0]

    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(validation_samples)}")
    print(f"Training sequences: {len(train_sequence_index)}")
    print(f"Validation sequences: {len(validation_sequence_index)}")
    print(f"X batch shape: {x_batch.shape}")
    print(f"y batch shape: {y_batch.shape}")
    print(f"X dtype: {x_batch.dtype}")
    print(f"y dtype: {y_batch.dtype}")
    print(f"X min: {x_batch.min():.6f}")
    print(f"X max: {x_batch.max():.6f}")


if __name__ == "__main__":
    main()