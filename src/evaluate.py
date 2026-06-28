"""Evaluate the trained CNN+LSTM steering-angle regression model.

This script loads a saved Keras model, rebuilds the validation sequence data,
runs predictions, computes MAE and RMSE, saves metrics as JSON, and generates
evaluation plots for reporting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
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
    """Compute MAE and RMSE for steering-angle regression."""
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


def plot_predicted_vs_true_steering(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: str | Path,
    *,
    max_points: int | None = 500,
) -> None:
    """Save a predicted-vs-true steering-angle plot.

    Args:
        y_true: Ground-truth steering angles.
        y_pred: Predicted steering angles.
        path: Output image path.
        max_points: Optional maximum number of validation points to plot.
            This keeps the plot readable for documentation and defense slides.
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

    if max_points is not None:
        if max_points < 1:
            raise ValueError(f"max_points must be positive, got {max_points}")

        y_true = y_true[:max_points]
        y_pred = y_pred[:max_points]

    sample_indices = np.arange(len(y_true))

    plot_path = Path(path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(sample_indices, y_true, label="True steering angle")
    plt.plot(sample_indices, y_pred, label="Predicted steering angle")
    plt.xlabel("Validation sequence index")
    plt.ylabel("Steering angle")
    plt.title("Predicted vs True Steering Angle")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def load_training_history(path: str | Path) -> dict[str, list[float]]:
    """Load saved Keras training history from JSON.

    Args:
        path: Path to the training history JSON file.

    Returns:
        Training history dictionary.

    Raises:
        FileNotFoundError: If the history file does not exist.
        ValueError: If required loss keys are missing.
    """
    history_path = Path(path)

    if not history_path.is_file():
        raise FileNotFoundError(
            "Training history file not found. Run training first.\n"
            f"Expected history path: {history_path}"
        )

    with history_path.open("r", encoding="utf-8") as file:
        history = json.load(file)

    required_keys = {"loss", "val_loss"}
    missing_keys = required_keys.difference(history)

    if missing_keys:
        raise ValueError(
            f"Training history is missing required key(s): {sorted(missing_keys)}"
        )

    return {
        metric_name: [float(value) for value in metric_values]
        for metric_name, metric_values in history.items()
    }


def plot_training_validation_loss(
    history: dict[str, list[float]],
    path: str | Path,
) -> None:
    """Save a training-vs-validation loss plot.

    Args:
        history: Training history containing loss and val_loss.
        path: Output plot path.
    """
    if "loss" not in history:
        raise ValueError("Training history must contain 'loss'.")

    if "val_loss" not in history:
        raise ValueError("Training history must contain 'val_loss'.")

    train_loss = np.asarray(history["loss"], dtype=np.float32)
    validation_loss = np.asarray(history["val_loss"], dtype=np.float32)

    if train_loss.size == 0:
        raise ValueError("Training loss must not be empty.")

    if validation_loss.size == 0:
        raise ValueError("Validation loss must not be empty.")

    if train_loss.shape != validation_loss.shape:
        raise ValueError(
            "Training loss and validation loss must have the same number of "
            f"epochs, got {train_loss.shape} and {validation_loss.shape}."
        )

    epochs = np.arange(1, len(train_loss) + 1)

    plot_path = Path(path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, marker="o", label="Training loss")
    plt.plot(epochs, validation_loss, marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def build_validation_generator(
    config: dict[str, Any],
    *,
    config_path: str | Path,
):
    """Build the validation sequence generator used for evaluation."""
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
    """Evaluate a saved CNN+LSTM model on validation sequences."""
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

    resolved_prediction_plot_path = Path(
        outputs_config.get(
            "prediction_plot_path",
            "outputs/plots/predicted_vs_true_steering.png",
        )
    )

    resolved_history_path = Path(
        outputs_config.get(
            "history_path",
            "outputs/models/training_history.json",
        )
    )

    resolved_loss_plot_path = Path(
        outputs_config.get(
            "loss_plot_path",
            "outputs/plots/training_validation_loss.png",
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

    plot_predicted_vs_true_steering(
        y_true,
        y_pred,
        resolved_prediction_plot_path,
    )

    history = load_training_history(resolved_history_path)

    plot_training_validation_loss(
        history,
        resolved_loss_plot_path,
    )

    result = {
        "model_path": str(resolved_model_path),
        "num_validation_sequences": int(len(validation_sequence_index)),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "prediction_plot_path": str(resolved_prediction_plot_path),
        "loss_plot_path": str(resolved_loss_plot_path),
    }

    save_metrics(result, resolved_metrics_path)

    print()
    print("Evaluation results")
    print(f"Model path: {resolved_model_path}")
    print(f"Validation sequences: {len(validation_sequence_index)}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"Metrics saved to: {resolved_metrics_path}")
    print(f"Prediction plot saved to: {resolved_prediction_plot_path}")
    print(f"Loss plot saved to: {resolved_loss_plot_path}")

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