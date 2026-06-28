"""Inference pipeline for CNN+LSTM steering-angle prediction.

This script loads a saved trained model, builds fixed-length ordered image
sequences, runs steering-angle prediction, and saves predictions aligned with
the corresponding output frames.

Each prediction is aligned with the final frame of its input sequence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data import load_config, load_driving_samples
from src.sequences import (
    build_sequence_index,
    create_sequence_generator_from_config,
    get_sequence_config,
)


def build_inference_sequence_index(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    max_sequences: int | None = None,
) -> pd.DataFrame:
    """Build ordered fixed-length inference windows from center-camera samples.

    Args:
        config: Parsed project configuration.
        config_path: Path to config.yaml.
        max_sequences: Optional maximum number of sequences to keep. Useful for
            quick local smoke tests.

    Returns:
        DataFrame containing sequence frame paths, output frame alignment, and
        optional ground-truth steering angle from the dataset.
    """
    sequence_config = get_sequence_config(config)

    samples = load_driving_samples(config_path)

    sequence_index = build_sequence_index(
        samples,
        sequence_length=sequence_config["sequence_length"],
        stride=sequence_config["stride"],
        target=sequence_config["target"],
    )

    if max_sequences is not None:
        if max_sequences < 1:
            raise ValueError(f"max_sequences must be positive, got {max_sequences}")
        sequence_index = sequence_index.head(max_sequences).reset_index(drop=True)

    output_frame_indices = sequence_index["end_index"].astype(int)

    sequence_index = sequence_index.copy()
    sequence_index["output_frame_index"] = output_frame_indices
    sequence_index["output_image_path"] = [
        frame_paths[-1] for frame_paths in sequence_index["frame_paths"]
    ]

    return sequence_index


def run_inference(
    *,
    config_path: str | Path,
    model_path: str | Path | None = None,
    output_path: str | Path | None = None,
    max_sequences: int | None = None,
) -> pd.DataFrame:
    """Run steering-angle inference over ordered image sequences.

    Args:
        config_path: Path to project configuration YAML file.
        model_path: Optional path to saved Keras model. If omitted, the best
            checkpoint path from config.yaml is used.
        output_path: Optional CSV output path. If omitted, the inference output
            path from config.yaml is used.
        max_sequences: Optional maximum number of sequences for quick testing.

    Returns:
        DataFrame containing frame-aligned steering-angle predictions.

    Raises:
        FileNotFoundError: If the saved model does not exist.
    """
    config = load_config(config_path)
    outputs_config = config.get("outputs", {})

    resolved_model_path = Path(
        model_path
        if model_path is not None
        else outputs_config.get("checkpoint_path", "outputs/models/best_model.keras")
    )

    resolved_output_path = Path(
        output_path
        if output_path is not None
        else outputs_config.get(
            "inference_predictions_path",
            "outputs/inference/steering_predictions.csv",
        )
    )

    if not resolved_model_path.is_file():
        raise FileNotFoundError(
            "Saved model not found. Run training first.\n"
            f"Expected model path: {resolved_model_path}"
        )

    sequence_index = build_inference_sequence_index(
        config,
        config_path=config_path,
        max_sequences=max_sequences,
    )

    inference_generator = create_sequence_generator_from_config(
        sequence_index,
        config,
        shuffle=False,
    )

    model = tf.keras.models.load_model(resolved_model_path)

    predictions = model.predict(inference_generator, verbose=1).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float32)

    if len(predictions) != len(sequence_index):
        raise RuntimeError(
            "Prediction count does not match sequence count, "
            f"got {len(predictions)} predictions for {len(sequence_index)} sequences."
        )

    results = pd.DataFrame(
        {
            "sequence_index": np.arange(len(sequence_index), dtype=int),
            "start_frame_index": sequence_index["start_index"].astype(int),
            "end_frame_index": sequence_index["end_index"].astype(int),
            "output_frame_index": sequence_index["output_frame_index"].astype(int),
            "output_image_path": sequence_index["output_image_path"].astype(str),
            "predicted_steering_angle": predictions,
        }
    )

    if "target_steering_angle" in sequence_index.columns:
        results["true_steering_angle"] = sequence_index[
            "target_steering_angle"
        ].astype(float)

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(resolved_output_path, index=False)

    print()
    print("Inference complete")
    print(f"Model path: {resolved_model_path}")
    print(f"Sequences predicted: {len(results)}")
    print(f"Predictions saved to: {resolved_output_path}")
    print()
    print("First predictions:")
    print(results.head())

    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run CNN+LSTM steering-angle inference over image sequences."
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
        "--output",
        default=None,
        help=(
            "Path to save inference predictions CSV. "
            "Defaults to outputs.inference_predictions_path from config.yaml."
        ),
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Optional maximum number of sequences to process for quick tests.",
    )
    return parser.parse_args()


def main() -> None:
    """Run inference from the command line."""
    args = parse_args()

    run_inference(
        config_path=args.config,
        model_path=args.model,
        output_path=args.output,
        max_sequences=args.max_sequences,
    )


if __name__ == "__main__":
    main()