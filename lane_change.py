"""Simple lane-change warning heuristic based on predicted steering angles."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _causal_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    smoothed = np.empty_like(values, dtype=np.float32)
    for i in range(len(values)):
        smoothed[i] = np.mean(values[max(0, i - window + 1) : i + 1])
    return smoothed


def _sustained_runs(active: np.ndarray, minimum_length: int) -> np.ndarray:
    """Keep only threshold-active intervals long enough to count as warnings."""
    warning = np.zeros(len(active), dtype=bool)
    start = None

    for i in range(len(active) + 1):
        is_active = i < len(active) and active[i]
        if is_active and start is None:
            start = i
        elif not is_active and start is not None:
            if i - start >= minimum_length:
                warning[start:i] = True
            start = None
    return warning


def _event_ids(warning: np.ndarray) -> np.ndarray:
    ids = np.zeros(len(warning), dtype=np.int32)
    event_id = 0
    previous = False
    for i, active in enumerate(warning):
        if active and not previous:
            event_id += 1
        if active:
            ids[i] = event_id
        previous = bool(active)
    return ids


def add_lane_change_warnings(
    predictions: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Smooth predictions and mark sustained steering deviations."""
    cfg = config["lane_change"]
    angles = predictions["predicted_steering_angle"].to_numpy(dtype=np.float32)
    smoothed = _causal_moving_average(angles, int(cfg["smoothing_window"]))
    threshold_active = np.abs(smoothed) >= float(cfg["steering_threshold"])
    warning = _sustained_runs(threshold_active, int(cfg["min_duration_frames"]))

    result = predictions.copy()
    result["smoothed_steering_angle"] = smoothed
    result["lane_change_threshold_active"] = threshold_active
    result["lane_change_warning"] = warning
    result["lane_change_event_id"] = _event_ids(warning)
    return result
