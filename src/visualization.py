"""Visualization utilities for steering-angle inference outputs.

This module creates simple annotated image frames for documentation, reports,
and defense demos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _get_overlay_style(image: np.ndarray) -> tuple[float, int, int, int]:
    """Return font scale, text thickness, padding, and line spacing."""
    height, width = image.shape[:2]

    font_scale = max(0.35, min(width / 900.0, 0.55))
    text_thickness = max(1, int(round(width / 700.0)))
    padding = max(6, int(round(width / 80.0)))
    line_spacing = max(18, int(round(height / 8.0)))

    return font_scale, text_thickness, padding, line_spacing


def _as_bool(value: Any) -> bool:
    """Convert common boolean-like values to bool."""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}

    return bool(value)


def _draw_text_box(
    image: np.ndarray,
    lines: list[str],
    *,
    origin: tuple[int, int],
    background_color: tuple[int, int, int],
    text_color: tuple[int, int, int],
    font_scale: float,
    text_thickness: int,
    padding: int,
    line_spacing: int,
) -> None:
    """Draw a compact filled text box onto an image."""
    if not lines:
        return

    x, y = origin

    max_text_width = 0

    for text in lines:
        text_size, _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_thickness,
        )
        max_text_width = max(max_text_width, text_size[0])

    box_width = max_text_width + padding * 2
    box_height = padding * 2 + line_spacing * len(lines)

    image_height, image_width = image.shape[:2]

    box_width = min(box_width, image_width - x)
    box_height = min(box_height, image_height - y)

    cv2.rectangle(
        image,
        (x, y),
        (x + box_width, y + box_height),
        background_color,
        thickness=-1,
    )

    first_baseline_y = y + padding + 14

    for line_index, text in enumerate(lines):
        text_y = first_baseline_y + line_index * line_spacing

        cv2.putText(
            image,
            text,
            (x + padding, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            text_thickness,
            cv2.LINE_AA,
        )


def annotate_frame_with_steering(
    image_path: str | Path,
    predicted_steering_angle: float,
    *,
    true_steering_angle: float | None = None,
    lane_change_warning: bool = False,
) -> np.ndarray:
    """Load an image and overlay steering-angle and warning text.

    Args:
        image_path: Path to the source image frame.
        predicted_steering_angle: Predicted steering angle to display.
        true_steering_angle: Optional ground-truth steering angle.
        lane_change_warning: Whether to draw a lane-change warning banner.

    Returns:
        Annotated BGR image array suitable for saving with OpenCV.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image cannot be read.
    """
    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Failed to read image file: {path}")

    annotated = image.copy()

    font_scale, text_thickness, padding, line_spacing = _get_overlay_style(annotated)

    steering_lines = [
        f"Pred: {predicted_steering_angle:+.3f}",
    ]

    if true_steering_angle is not None:
        steering_lines.append(f"True: {true_steering_angle:+.3f}")

    _draw_text_box(
        annotated,
        steering_lines,
        origin=(0, 0),
        background_color=(0, 0, 0),
        text_color=(255, 255, 255),
        font_scale=font_scale,
        text_thickness=text_thickness,
        padding=padding,
        line_spacing=line_spacing,
    )

    if lane_change_warning:
        warning_text = "LANE CHANGE WARNING"

        warning_text_size, _ = cv2.getTextSize(
            warning_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_thickness,
        )

        image_height, image_width = annotated.shape[:2]
        warning_box_width = warning_text_size[0] + padding * 2
        warning_box_height = padding * 2 + line_spacing

        warning_x = max(0, image_width - warning_box_width)
        warning_y = max(0, image_height - warning_box_height)

        _draw_text_box(
            annotated,
            [warning_text],
            origin=(warning_x, warning_y),
            background_color=(0, 0, 255),
            text_color=(255, 255, 255),
            font_scale=font_scale,
            text_thickness=text_thickness,
            padding=padding,
            line_spacing=line_spacing,
        )

    return annotated


def save_annotated_frame(
    image_path: str | Path,
    output_path: str | Path,
    predicted_steering_angle: float,
    *,
    true_steering_angle: float | None = None,
    lane_change_warning: bool = False,
) -> None:
    """Save one annotated steering-angle frame."""
    annotated = annotate_frame_with_steering(
        image_path,
        predicted_steering_angle,
        true_steering_angle=true_steering_angle,
        lane_change_warning=lane_change_warning,
    )

    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(resolved_output_path), annotated)

    if not success:
        raise ValueError(f"Failed to write annotated frame: {resolved_output_path}")


def export_annotated_frames(
    predictions,
    output_dir: str | Path,
    *,
    max_frames: int | None = 50,
) -> list[Path]:
    """Export annotated frames from an inference prediction table.

    Args:
        predictions: DataFrame-like object containing output_image_path and
            predicted_steering_angle columns. If true_steering_angle exists, it
            is also displayed. If lane_change_warning exists, warning text is
            displayed only when active.
        output_dir: Directory where annotated frames should be written.
        max_frames: Optional maximum number of frames to export.

    Returns:
        List of written frame paths.
    """
    required_columns = {"output_image_path", "predicted_steering_angle"}
    missing_columns = required_columns.difference(predictions.columns)

    if missing_columns:
        raise ValueError(
            f"predictions is missing required column(s): {sorted(missing_columns)}"
        )

    if max_frames is not None and max_frames < 1:
        raise ValueError(f"max_frames must be positive, got {max_frames}")

    rows = predictions

    if max_frames is not None:
        rows = rows.head(max_frames)

    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    written_paths: list[Path] = []

    for row_number, (_, row) in enumerate(rows.iterrows()):
        true_steering_angle = None

        if "true_steering_angle" in row:
            raw_true_value = row["true_steering_angle"]

            if raw_true_value is not None:
                true_value = float(raw_true_value)

                if np.isfinite(true_value):
                    true_steering_angle = true_value

        lane_change_warning = False

        if "lane_change_warning" in row:
            lane_change_warning = _as_bool(row["lane_change_warning"])

        output_path = resolved_output_dir / f"annotated_frame_{row_number:05d}.jpg"

        save_annotated_frame(
            row["output_image_path"],
            output_path,
            float(row["predicted_steering_angle"]),
            true_steering_angle=true_steering_angle,
            lane_change_warning=lane_change_warning,
        )

        written_paths.append(output_path)

    return written_paths