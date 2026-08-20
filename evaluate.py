"""Held-out validation evaluation and report plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import tensorflow as tf

from dataset import load_config, prepare_validation_data


def _plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = min(500, len(y_true))
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(n), y_true[:n], label="True steering angle")
    plt.plot(np.arange(n), y_pred[:n], label="Predicted steering angle")
    plt.xlabel("Validation sequence")
    plt.ylabel("Steering angle")
    plt.title("Predicted vs True Steering Angle")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_loss(history_path: Path, output_path: Path) -> None:
    if not history_path.is_file():
        return
    with history_path.open("r", encoding="utf-8") as file:
        history = json.load(file)

    epochs = np.arange(1, len(history["loss"]) + 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["loss"], marker="o", label="Training loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate(config_path: str = "config.yaml") -> dict[str, float]:
    config = load_config(config_path)
    outputs = config["outputs"]
    model_path = Path(outputs["checkpoint_path"])

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}. Run training first.")

    validation_index, generator = prepare_validation_data(config)
    y_true = validation_index["target_steering_angle"].to_numpy(dtype=np.float32)

    model = tf.keras.models.load_model(model_path)
    y_pred = model.predict(generator, verbose=1).reshape(-1).astype(np.float32)

    errors = y_true - y_pred
    metrics = {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
    }

    metrics_path = Path(outputs["evaluation_metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    _plot_predictions(y_true, y_pred, Path(outputs["prediction_plot_path"]))
    _plot_loss(Path(outputs["history_path"]), Path(outputs["loss_plot_path"]))

    print("\n=== EVALUATION ===")
    print(f"Held-out validation sequences: {len(validation_index)}")
    print(f"MAE:  {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"Prediction plot: {outputs['prediction_plot_path']}")
    return metrics
