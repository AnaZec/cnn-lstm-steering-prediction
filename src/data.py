"""Data loading utilities for the CNN+LSTM steering-angle project.

This module is responsible only for reading the driving log CSV and resolving
valid center-camera image paths with numeric steering-angle labels.

It does not load image pixels, resize images, create sequences, train models,
or perform inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file is empty.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")

    return config


def _resolve_image_path(
    raw_path: str,
    *,
    csv_dir: Path,
    image_root: Path,
) -> Path | None:
    """Resolve an image path from the CSV to an existing local file.

    The Udacity CSV may contain:
    - absolute paths from another machine
    - paths relative to the CSV location
    - paths relative to the image root
    - only image filenames

    Args:
        raw_path: Image path string read from the CSV.
        csv_dir: Directory containing the driving log CSV.
        image_root: Configured image root directory.

    Returns:
        Resolved existing Path, or None if no candidate exists.
    """
    cleaned = str(raw_path).strip().replace("\\", "/")

    if not cleaned:
        return None

    path = Path(cleaned)

    candidates = []

    # Case 1: path is already absolute and valid.
    if path.is_absolute():
        candidates.append(path)

    # Case 2: path is relative to the CSV directory.
    candidates.append(csv_dir / path)

    # Case 3: path is relative to the configured image root.
    candidates.append(image_root / path)

    # Case 4: CSV contains an old absolute path or nested path from another
    # machine. Use only the filename under the configured image root.
    candidates.append(image_root / path.name)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    return None


def load_driving_samples(
    config_path: str | Path = "config.yaml",
    *,
    drop_invalid: bool = True,
) -> pd.DataFrame:
    """Load valid center-camera image samples and steering-angle labels.

    Args:
        config_path: Path to project configuration YAML.
        drop_invalid: If True, skip rows with missing images or invalid steering
            values. If False, raise ValueError when invalid rows are found.

    Returns:
        DataFrame with columns:
            - image_path: absolute resolved image path as string
            - steering_angle: steering angle as float

    Raises:
        FileNotFoundError: If the CSV or image root does not exist.
        KeyError: If configured columns are missing from the CSV.
        ValueError: If no valid samples are found, or invalid rows exist while
            drop_invalid is False.
    """
    config = load_config(config_path)
    dataset_config = config.get("dataset", {})

    csv_path = Path(dataset_config.get("driving_log_csv", ""))
    image_root = Path(dataset_config.get("image_root", ""))

    center_column = dataset_config.get("center_camera_column", "centercam")
    steering_column = dataset_config.get("steering_column", "steering_angle")

    if not csv_path.exists():
        raise FileNotFoundError(f"Driving log CSV not found: {csv_path}")

    if not image_root.exists():
        raise FileNotFoundError(f"Image root directory not found: {image_root}")

    has_header = bool(dataset_config.get("has_header", True))
    csv_columns = dataset_config.get("csv_columns")

    if has_header:
        df = pd.read_csv(csv_path)
    else:
        if not csv_columns:
            raise ValueError(
                "dataset.csv_columns must be configured when dataset.has_header is false."
            )
        df = pd.read_csv(csv_path, header=None, names=csv_columns)

    missing_columns = [
        column
        for column in (center_column, steering_column)
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            "Missing required column(s) in driving log CSV: "
            f"{missing_columns}. Available columns: {list(df.columns)}"
        )

    csv_dir = csv_path.parent
    valid_samples: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []

    for row_index, row in df.iterrows():
        raw_image_path = row[center_column]
        raw_steering = row[steering_column]

        resolved_path = _resolve_image_path(
            raw_image_path,
            csv_dir=csv_dir,
            image_root=image_root,
        )

        steering_angle = pd.to_numeric(raw_steering, errors="coerce")

        if resolved_path is None:
            invalid_rows.append(
                {
                    "row_index": row_index,
                    "reason": "missing_image",
                    "raw_image_path": raw_image_path,
                }
            )
            continue

        if pd.isna(steering_angle):
            invalid_rows.append(
                {
                    "row_index": row_index,
                    "reason": "invalid_steering_angle",
                    "raw_steering": raw_steering,
                }
            )
            continue

        steering_angle = float(steering_angle)

        valid_samples.append(
            {
                "image_path": str(resolved_path),
                "steering_angle": steering_angle,
            }
        )

    if invalid_rows and not drop_invalid:
        raise ValueError(f"Invalid driving log rows found: {invalid_rows[:10]}")

    samples = pd.DataFrame(valid_samples, columns=["image_path", "steering_angle"])

    if samples.empty:
        raise ValueError(
            "No valid driving samples found. Check CSV path, image_root, "
            "column names, and local dataset placement."
        )

    return samples


def split_train_validation_samples(
    samples: pd.DataFrame,
    *,
    validation_ratio: float,
    sequence_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ordered samples into train and validation sets without sequence leakage.

    The split is performed on raw ordered rows before sequence construction.
    This prevents overlapping source frames from appearing in both training and
    validation sequences.
    """
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(
            f"validation_ratio must be between 0 and 1, got {validation_ratio}"
        )

    if sequence_length < 1:
        raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")

    required_columns = {"image_path", "steering_angle"}
    missing_columns = required_columns.difference(samples.columns)

    if missing_columns:
        raise ValueError(
            f"samples is missing required column(s): {sorted(missing_columns)}"
        )

    total_samples = len(samples)

    if total_samples < 2 * sequence_length:
        raise ValueError(
            "Not enough samples to create train and validation sequences. "
            f"Need at least {2 * sequence_length}, got {total_samples}."
        )

    validation_size = int(round(total_samples * validation_ratio))
    validation_size = max(validation_size, sequence_length)

    train_size = total_samples - validation_size

    if train_size < sequence_length:
        raise ValueError(
            "Training split is too small to create at least one sequence. "
            f"train_size={train_size}, sequence_length={sequence_length}"
        )

    train_samples = samples.iloc[:train_size].reset_index(drop=True)
    validation_samples = samples.iloc[train_size:].reset_index(drop=True)

    return train_samples, validation_samples
    

def main() -> None:
    """Small CLI smoke test for the data loader and temporal split."""
    config = load_config("config.yaml")

    samples = load_driving_samples("config.yaml")

    split_config = config.get("split", {})
    sequence_config = config.get("sequences", {})

    validation_ratio = float(split_config.get("validation_ratio", 0.2))
    sequence_length = int(sequence_config.get("sequence_length", 5))

    train_samples, validation_samples = split_train_validation_samples(
        samples,
        validation_ratio=validation_ratio,
        sequence_length=sequence_length,
    )

    print(f"Loaded valid samples: {len(samples)}")
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(validation_samples)}")
    print()
    print("First training samples:")
    print(train_samples.head())
    print()
    print("First validation samples:")
    print(validation_samples.head())


if __name__ == "__main__":
    main()