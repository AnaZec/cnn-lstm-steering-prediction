"""Dataset loading, preprocessing and temporal sequence generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml
from tensorflow.keras.utils import Sequence


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _resolve_image_path(raw_path: str, csv_dir: Path, image_root: Path) -> Path | None:
    """Resolve Udacity CSV paths even if the CSV was created on another machine."""
    cleaned = str(raw_path).strip().replace("\\", "/")
    path = Path(cleaned)

    candidates = []
    if path.is_absolute():
        candidates.append(path)
    candidates.extend((csv_dir / path, image_root / path, image_root / path.name))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def load_samples(config: dict[str, Any]) -> pd.DataFrame:
    """Load center-camera paths and steering labels from driving_log.csv."""
    cfg = config["dataset"]
    csv_path = Path(cfg["driving_log_csv"])
    image_root = Path(cfg["image_root"])

    if not csv_path.is_file():
        raise FileNotFoundError(f"Driving log not found: {csv_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_root}")

    if cfg.get("has_header", False):
        raw = pd.read_csv(csv_path)
    else:
        raw = pd.read_csv(csv_path, header=None, names=cfg["csv_columns"])

    image_column = cfg.get("center_camera_column", "centercam")
    steering_column = cfg.get("steering_column", "steering_angle")

    rows = []
    for _, row in raw.iterrows():
        image_path = _resolve_image_path(row[image_column], csv_path.parent, image_root)
        steering = pd.to_numeric(row[steering_column], errors="coerce")
        if image_path is not None and not pd.isna(steering):
            rows.append((str(image_path), float(steering)))

    samples = pd.DataFrame(rows, columns=["image_path", "steering_angle"])
    if samples.empty:
        raise ValueError("No valid samples found. Check dataset paths in config.yaml.")
    return samples


def split_samples(
    samples: pd.DataFrame,
    validation_ratio: float,
    sequence_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal split BEFORE sequence construction to prevent frame leakage."""
    if len(samples) < 2 * sequence_length:
        raise ValueError("Dataset is too small for train and validation sequences.")

    validation_size = max(int(round(len(samples) * validation_ratio)), sequence_length)
    split_index = len(samples) - validation_size

    if split_index < sequence_length:
        raise ValueError("Training split is too small for one complete sequence.")

    train = samples.iloc[:split_index].reset_index(drop=True)
    validation = samples.iloc[split_index:].reset_index(drop=True)
    return train, validation


def build_sequence_index(
    samples: pd.DataFrame,
    sequence_length: int,
    stride: int = 1,
) -> pd.DataFrame:
    """Create sliding windows; each target is the steering angle of the last frame."""
    rows = []
    for start in range(0, len(samples) - sequence_length + 1, stride):
        window = samples.iloc[start : start + sequence_length]
        rows.append(
            {
                "start_index": start,
                "end_index": start + sequence_length - 1,
                "frame_paths": window["image_path"].tolist(),
                "target_steering_angle": float(window.iloc[-1]["steering_angle"]),
            }
        )

    if not rows:
        raise ValueError("Not enough frames to create a sequence.")
    return pd.DataFrame(rows)


def preprocess_image(path: str | Path, config: dict[str, Any]) -> np.ndarray:
    """Read image, convert BGR->RGB, resize and normalize to [0, 1]."""
    cfg = config["preprocessing"]
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(
        image,
        (int(cfg["image_width"]), int(cfg["image_height"])),
        interpolation=cv2.INTER_AREA,
    )

    if cfg.get("normalize", True):
        image = image.astype(np.float32) / 255.0
    return image


class SteeringSequenceGenerator(Sequence):
    """Keras generator returning batches shaped [B, T, H, W, C]."""

    def __init__(
        self,
        sequence_index: pd.DataFrame,
        config: dict[str, Any],
        shuffle: bool = False,
    ) -> None:
        super().__init__()
        self.data = sequence_index.reset_index(drop=True)
        self.config = config
        self.batch_size = int(config["training"]["batch_size"])
        self.shuffle = shuffle
        self.rng = np.random.default_rng(int(config["split"]["random_seed"]))
        self.indices = np.arange(len(self.data))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, batch_index: int) -> tuple[np.ndarray, np.ndarray]:
        start = batch_index * self.batch_size
        selected = self.indices[start : start + self.batch_size]

        x_batch, y_batch = [], []
        for _, row in self.data.iloc[selected].iterrows():
            frames = [preprocess_image(path, self.config) for path in row["frame_paths"]]
            x_batch.append(np.stack(frames))
            y_batch.append(float(row["target_steering_angle"]))

        return np.stack(x_batch), np.asarray(y_batch, dtype=np.float32)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indices)


def prepare_data(config: dict[str, Any]):
    """Build train/validation indices and generators used by training."""
    seq_cfg = config["sequences"]
    samples = load_samples(config)
    train_samples, validation_samples = split_samples(
        samples,
        validation_ratio=float(config["split"]["validation_ratio"]),
        sequence_length=int(seq_cfg["sequence_length"]),
    )

    train_index = build_sequence_index(
        train_samples,
        int(seq_cfg["sequence_length"]),
        int(seq_cfg.get("stride", 1)),
    )
    validation_index = build_sequence_index(
        validation_samples,
        int(seq_cfg["sequence_length"]),
        int(seq_cfg.get("stride", 1)),
    )

    return (
        train_index,
        validation_index,
        SteeringSequenceGenerator(train_index, config, shuffle=True),
        SteeringSequenceGenerator(validation_index, config, shuffle=False),
    )


def prepare_validation_data(config: dict[str, Any]):
    """Rebuild only the held-out validation data for evaluation/demo."""
    seq_cfg = config["sequences"]
    samples = load_samples(config)
    _, validation_samples = split_samples(
        samples,
        validation_ratio=float(config["split"]["validation_ratio"]),
        sequence_length=int(seq_cfg["sequence_length"]),
    )

    validation_index = build_sequence_index(
        validation_samples,
        int(seq_cfg["sequence_length"]),
        int(seq_cfg.get("stride", 1)),
    )
    generator = SteeringSequenceGenerator(validation_index, config, shuffle=False)
    return validation_index, generator
