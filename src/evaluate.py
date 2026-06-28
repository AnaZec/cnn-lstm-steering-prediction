"""Evaluate the trained CNN+LSTM steering-angle regression model.

This script loads a saved Keras model, rebuilds the validation sequence data,
runs predictions, computes MAE and RMSE, and saves the metrics as JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from src.data import (
    load_config,
    load_driving_samples,
    split_train_validation_samples,
)
from src.sequences import (
    build_sequence_index,
    create_sequence_generator_from_config,
    get_sequence_config,
)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute MAE and RMSE for steering-angle regression.

    Args:
        y_true: Ground-truth steering angles.
        y_pred: Predicted steering angles.

    Returns:
        Dictionary containing mae and rmse.

    Raises:
        ValueError: If the input arrays are empty or have different shapes.
    """
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)

    if y_true.size == 0:
        raise ValueError("y_true must not be empty.")

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, "
            f"got {y_true.shape} and {y_pred.shape}."
        )

    errors = y_true - y_pred

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))

    return {
        "mae": mae,
        "rmse": rmse,
    }


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    """Save evaluation metrics as readable JSON."""
    metrics_path = Path(path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def build_validation_generator(
    config: dict[str, Any],
    *,
    config_path: str | Path,
):
    """Build the validation sequence generator used for evaluation.

    Args:
        config: Parsed project configuration.
        config_path: Path to config.yaml.

    Returns:
        Tuple containing:
            validation_generator,
            validation_sequence_index
    """
    sequence_config = get_sequence_config(config)
    split_config = config.get("split", {})

    samples = load_driving_samples(config_path)

    _, validation_samples = split_train_validation_samples(
        samples,
        validation_ratio=float(split_config.get("validation_ratio", 0.2)),
        sequence_length=sequence_config["sequence_length"],
    )

    validation_sequence_index = build_sequence_index(
        validation_samples,
        sequence_length=sequence_config["sequence_length"],
        stride=sequence_config["stride"],
        target=sequence_config["target"],
    )

    validation_generator = create_sequence_generator_from_config(
        validation_sequence_index,
        config,
        shuffle=False,
    )

    return validation_generator, validation_sequence_index


def evaluate_model(
    *,
    config_path: str | Path,
    model_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a saved CNN+LSTM model on the validation sequences.

    Args:
        config_path: Path to project config file.
        model_path: Optional path to saved Keras model. If omitted, the best
            checkpoint path from config.yaml is used.
        metrics_path: Optional output path for metrics JSON. If omitted, the
            evaluation metrics path from config.yaml is used.

    Returns:
        Metrics dictionary.
    """
    config = load_config(config_path)
    outputs_config = config.get("outputs", {})

    resolved_model_path = Path(
        model_path
        if model_path is not None
        else outputs_config.get("checkpoint_path", "outputs/models/best_model.keras")
    )

    resolved_metrics_path = Path(
        metrics_path
        if metrics_path is not None
        else outputs_config.get(
            "evaluation_metrics_path",
            "outputs/evaluation/evaluation_metrics.json",
        )
    )

    if not resolved_model_path.is_file():
        raise FileNotFoundError(
            "Saved model not found. Run training first.\n"
            f"Expected model path: {resolved_model_path}"
        )

    validation_generator, validation_sequence_index = build_validation_generator(
        config,
        config_path=config_path,
    )

    y_true = validation_sequence_index["target_steering_angle"].to_numpy(
        dtype=np.float32
    )

    model = tf.keras.models.load_model(resolved_model_path)

    y_pred = model.predict(validation_generator, verbose=1).reshape(-1)

    metrics = compute_regression_metrics(y_true, y_pred)

    result = {
        "model_path": str(resolved_model_path),
        "num_validation_sequences": int(len(validation_sequence_index)),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
    }

    save_metrics(result, resolved_metrics_path)

    print()
    print("Evaluation results")
    print(f"Model path: {resolved_model_path}")
    print(f"Validation sequences: {len(validation_sequence_index)}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"Metrics saved to: {resolved_metrics_path}")

    return result


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CNN+LSTM steering-angle model."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to project configuration YAML file.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Path to saved Keras model. "
            "Defaults to outputs.checkpoint_path from config.yaml."
        ),
    )
    parser.add_argument(
        "--metrics-output",
        default=None,
        help=(
            "Path to save evaluation metrics JSON. "
            "Defaults to outputs.evaluation_metrics_path from config.yaml."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run evaluation from the command line."""
    args = parse_args()

    evaluate_model(
        config_path=args.config,
        model_path=args.model,
        metrics_path=args.metrics_output,
    )


if __name__ == "__main__":
    main()