"""Image preprocessing utilities for CNN+LSTM steering-angle prediction.

This module converts raw front-camera image files into consistently shaped
numeric arrays suitable for TensorFlow/Keras models.

Preprocessing is intentionally deterministic so the same function can be reused
for training, evaluation, and inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.data import load_config


def get_preprocessing_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract preprocessing settings from the project configuration.

    Args:
        config: Parsed project configuration dictionary.

    Returns:
        Dictionary containing image_width, image_height, channels, and normalize.

    Raises:
        ValueError: If required preprocessing values are invalid.
    """
    preprocessing_config = config.get("preprocessing", {})

    image_width = int(preprocessing_config.get("image_width", 160))
    image_height = int(preprocessing_config.get("image_height", 80))
    channels = int(preprocessing_config.get("channels", 3))
    normalize = bool(preprocessing_config.get("normalize", True))

    if image_width <= 0:
        raise ValueError(f"image_width must be positive, got {image_width}")

    if image_height <= 0:
        raise ValueError(f"image_height must be positive, got {image_height}")

    if channels != 3:
        raise ValueError(
            "Only 3-channel RGB images are supported in this project. "
            f"Got channels={channels}."
        )

    return {
        "image_width": image_width,
        "image_height": image_height,
        "channels": channels,
        "normalize": normalize,
    }


def preprocess_image(
    image_path: str | Path,
    *,
    image_width: int,
    image_height: int,
    channels: int = 3,
    normalize: bool = True,
) -> np.ndarray:
    """Load, color-convert, resize, and optionally normalize one image.

    Args:
        image_path: Path to the image file.
        image_width: Target image width.
        image_height: Target image height.
        channels: Number of expected channels. Only 3 is supported.
        normalize: If True, scale pixel values from [0, 255] to [0, 1].

    Returns:
        Preprocessed image array with shape:
            (image_height, image_width, channels)

        If normalize is True:
            dtype is float32 and values are in [0, 1].

        If normalize is False:
            dtype is uint8 and values are in [0, 255].

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: If the image cannot be read or the config is invalid.
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    if not path.is_file():
        raise FileNotFoundError(f"Image path is not a file: {path}")

    if image_width <= 0:
        raise ValueError(f"image_width must be positive, got {image_width}")

    if image_height <= 0:
        raise ValueError(f"image_height must be positive, got {image_height}")

    if channels != 3:
        raise ValueError(
            "Only 3-channel RGB images are supported in this project. "
            f"Got channels={channels}."
        )

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise ValueError(f"Failed to read image file: {path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(
        image_rgb,
        (image_width, image_height),
        interpolation=cv2.INTER_AREA,
    )

    if normalize:
        return resized.astype(np.float32) / 255.0

    return resized


def preprocess_image_from_config(
    image_path: str | Path,
    config_path: str | Path = "config.yaml",
) -> np.ndarray:
    """Preprocess one image using settings from config.yaml.

    Args:
        image_path: Path to the image file.
        config_path: Path to the project configuration file.

    Returns:
        Preprocessed image array.
    """
    config = load_config(config_path)
    preprocessing = get_preprocessing_config(config)

    return preprocess_image(
        image_path,
        image_width=preprocessing["image_width"],
        image_height=preprocessing["image_height"],
        channels=preprocessing["channels"],
        normalize=preprocessing["normalize"],
    )


def main() -> None:
    """Small CLI smoke test for image preprocessing."""
    from src.data import load_driving_samples

    config = load_config("config.yaml")
    preprocessing = get_preprocessing_config(config)

    samples = load_driving_samples("config.yaml")
    first_image_path = samples.iloc[0]["image_path"]

    image = preprocess_image(
        first_image_path,
        image_width=preprocessing["image_width"],
        image_height=preprocessing["image_height"],
        channels=preprocessing["channels"],
        normalize=preprocessing["normalize"],
    )

    print(f"Image path: {first_image_path}")
    print(f"Shape: {image.shape}")
    print(f"Dtype: {image.dtype}")
    print(f"Min pixel value: {image.min():.6f}")
    print(f"Max pixel value: {image.max():.6f}")


if __name__ == "__main__":
    main()