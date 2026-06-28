"""Threshold-based possible lane-change detection.

This module detects possible lane-change intervals from the predicted
steering-angle sequence.

The detector is intentionally simple and does not use visual lane markers,
segmentation, lane geometry, or real lane detection. It only uses predicted
steering angles produced by the CNN+LSTM inference pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import load_config


def get_lane_change_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract lane-change detector settings from config.yaml.

    Args:
        config: Parsed project configuration.

    Returns:
        Dictionary containing steering_threshold, smoothing_window, and
        min_duration_frames.

    Raises:
        ValueError: If any setting is invalid.
    """
    lane_change_config = config.get("lane_change", {})

    steering_threshold = float(lane_change_config.get("steering_threshold", 0.15))
    smoothing_window = int(lane_change_config.get("smoothing_window", 5))
    min_duration_frames = int(lane_change_config.get("min_duration_frames", 5))

    if steering_threshold <= 0.0:
        raise ValueError(
            f"steering_threshold must be positive, got {steering_threshold}"
        )

    if smoothing_window < 1:
        raise ValueError(f"smoothing_window must be >= 1, got {smoothing_window}")

    if min_duration_frames < 1:
        raise ValueError(
            f"min_duration_frames must be >= 1, got {min_duration_frames}"
        )

    return {
        "steering_threshold": steering_threshold,
        "smoothing_window": smoothing_window,
        "min_duration_frames": min_duration_frames,
    }


def smooth_steering_angles(
    steering_angles: np.ndarray,
    *,
    window_size: int,
) -> np.ndarray:
    """Smooth steering angles with a causal moving average.

    The value at index i uses the current and previous values only. This keeps
    the detector suitable for frame-ordered inference.

    Args:
        steering_angles: Predicted steering-angle sequence.
        window_size: Moving average window size.

    Returns:
        Smoothed steering-angle sequence.

    Raises:
        ValueError: If inputs are invalid.
    """
    angles = np.asarray(steering_angles, dtype=np.float32).reshape(-1)

    if angles.size == 0:
        raise ValueError("steering_angles must not be empty.")

    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")

    smoothed = np.empty_like(angles, dtype=np.float32)

    for index in range(len(angles)):
        start_index = max(0, index - window_size + 1)
        smoothed[index] = float(np.mean(angles[start_index : index + 1]))

    return smoothed


def mark_sustained_active_runs(
    active: np.ndarray,
    *,
    min_duration_frames: int,
) -> np.ndarray:
    """Keep only active runs lasting at least min_duration_frames.

    Args:
        active: Boolean sequence indicating threshold activation.
        min_duration_frames: Minimum consecutive active frames required.

    Returns:
        Boolean warning sequence.

    Raises:
        ValueError: If min_duration_frames is invalid.
    """
    active = np.asarray(active, dtype=bool).reshape(-1)

    if min_duration_frames < 1:
        raise ValueError(
            f"min_duration_frames must be >= 1, got {min_duration_frames}"
        )

    warning = np.zeros_like(active, dtype=bool)

    run_start: int | None = None

    for index, is_active in enumerate(active):
        if is_active and run_start is None:
            run_start = index

        is_run_ending = not is_active and run_start is not None
        is_last_index = index == len(active) - 1

        if is_run_ending or (is_last_index and run_start is not None):
            run_end = index if is_active and is_last_index else index - 1
            run_length = run_end - run_start + 1

            if run_length >= min_duration_frames:
                warning[run_start : run_end + 1] = True

            run_start = None

    return warning


def assign_lane_change_event_ids(warning: np.ndarray) -> np.ndarray:
    """Assign event IDs to consecutive warning intervals.

    Non-warning frames receive event ID 0. The first warning interval receives
    ID 1, the second receives ID 2, and so on.

    Args:
        warning: Boolean warning sequence.

    Returns:
        Integer event ID sequence.
    """
    warning = np.asarray(warning, dtype=bool).reshape(-1)

    event_ids = np.zeros(len(warning), dtype=np.int32)
    current_event_id = 0
    in_event = False

    for index, is_warning in enumerate(warning):
        if is_warning and not in_event:
            current_event_id += 1
            in_event = True

        if not is_warning:
            in_event = False
            continue

        event_ids[index] = current_event_id

    return event_ids


def detect_possible_lane_changes(
    predicted_steering_angles: np.ndarray,
    *,
    steering_threshold: float,
    smoothing_window: int,
    min_duration_frames: int,
) -> pd.DataFrame:
    """Detect possible lane-change warning frames from predicted steering angles.

    Rule:
        1. Smooth predicted steering angles with a causal moving average.
        2. Mark frames where abs(smoothed_angle) >= steering_threshold.
        3. Keep only threshold-active runs lasting at least min_duration_frames.
        4. Assign event IDs to sustained warning intervals.

    Args:
        predicted_steering_angles: Predicted steering-angle sequence.
        steering_threshold: Absolute smoothed steering threshold.
        smoothing_window: Causal moving-average window size.
        min_duration_frames: Minimum number of consecutive active frames.

    Returns:
        DataFrame with detector diagnostic columns and warning state.
    """
    angles = np.asarray(predicted_steering_angles, dtype=np.float32).reshape(-1)

    if angles.size == 0:
        raise ValueError("predicted_steering_angles must not be empty.")

    if steering_threshold <= 0.0:
        raise ValueError(
            f"steering_threshold must be positive, got {steering_threshold}"
        )

    smoothed = smooth_steering_angles(
        angles,
        window_size=smoothing_window,
    )

    abs_smoothed = np.abs(smoothed)
    threshold_active = abs_smoothed >= steering_threshold

    lane_change_warning = mark_sustained_active_runs(
        threshold_active,
        min_duration_frames=min_duration_frames,
    )

    event_ids = assign_lane_change_event_ids(lane_change_warning)

    return pd.DataFrame(
        {
            "predicted_steering_angle": angles,
            "smoothed_steering_angle": smoothed,
            "abs_smoothed_steering_angle": abs_smoothed,
            "lane_change_threshold_active": threshold_active,
            "lane_change_warning": lane_change_warning,
            "lane_change_event_id": event_ids,
        }
    )


def add_lane_change_warnings(
    predictions: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Add lane-change warning columns to an inference predictions table.

    Args:
        predictions: DataFrame containing predicted_steering_angle.
        config: Parsed project configuration.

    Returns:
        Copy of predictions with lane-change detector columns appended.

    Raises:
        ValueError: If predicted_steering_angle is missing.
    """
    if "predicted_steering_angle" not in predictions.columns:
        raise ValueError(
            "predictions must contain a 'predicted_steering_angle' column."
        )

    detector_config = get_lane_change_config(config)

    detector_output = detect_possible_lane_changes(
        predictions["predicted_steering_angle"].to_numpy(dtype=np.float32),
        steering_threshold=detector_config["steering_threshold"],
        smoothing_window=detector_config["smoothing_window"],
        min_duration_frames=detector_config["min_duration_frames"],
    )

    result = predictions.copy()

    columns_to_add = [
        "smoothed_steering_angle",
        "abs_smoothed_steering_angle",
        "lane_change_threshold_active",
        "lane_change_warning",
        "lane_change_event_id",
    ]

    for column in columns_to_add:
        result[column] = detector_output[column].to_numpy()

    return result


def run_lane_change_detection(
    *,
    config_path: str | Path,
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Run lane-change detection on saved inference predictions.

    Args:
        config_path: Path to project configuration YAML file.
        input_path: Optional input predictions CSV path. If omitted,
            outputs.inference_predictions_path from config.yaml is used.
        output_path: Optional output CSV path. If omitted,
            outputs.lane_change_predictions_path from config.yaml is used.

    Returns:
        Predictions table with lane-change detector columns.
    """
    config = load_config(config_path)
    outputs_config = config.get("outputs", {})

    resolved_input_path = Path(
        input_path
        if input_path is not None
        else outputs_config.get(
            "inference_predictions_path",
            "outputs/inference/steering_predictions.csv",
        )
    )

    resolved_output_path = Path(
        output_path
        if output_path is not None
        else outputs_config.get(
            "lane_change_predictions_path",
            "outputs/inference/lane_change_predictions.csv",
        )
    )

    if not resolved_input_path.is_file():
        raise FileNotFoundError(
            "Inference predictions file not found. Run inference first.\n"
            f"Expected input path: {resolved_input_path}"
        )

    predictions = pd.read_csv(resolved_input_path)

    result = add_lane_change_warnings(predictions, config)

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(resolved_output_path, index=False)

    warning_frames = int(result["lane_change_warning"].sum())
    event_count = int(result["lane_change_event_id"].max())

    print()
    print("Lane-change detection complete")
    print(f"Input predictions: {resolved_input_path}")
    print(f"Output predictions: {resolved_output_path}")
    print(f"Frames processed: {len(result)}")
    print(f"Warning frames: {warning_frames}")
    print(f"Detected warning intervals: {event_count}")

    return result


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run threshold-based possible lane-change detection on predicted "
            "steering angles."
        )
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to project configuration YAML file.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Path to inference predictions CSV. "
            "Defaults to outputs.inference_predictions_path from config.yaml."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Path to save lane-change output CSV. "
            "Defaults to outputs.lane_change_predictions_path from config.yaml."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run lane-change detection from the command line."""
    args = parse_args()

    run_lane_change_detection(
        config_path=args.config,
        input_path=args.input,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()