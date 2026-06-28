"""Visualization utilities for steering-angle inference outputs.

This module creates simple annotated image frames for documentation, reports,
and defense demos.
"""

from __future__ import annotations

from pathlib import Path

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


def annotate_frame_with_steering(
    image_path: str | Path,
    predicted_steering_angle: float,
    *,
    true_steering_angle: float | None = None,
) -> np.ndarray:
    """Load an image and overlay compact steering-angle text.

    Args:
        image_path: Path to the source image frame.
        predicted_steering_angle: Predicted steering angle to display.
        true_steering_angle: Optional ground-truth steering angle.

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

    overlay_lines = [
        f"Pred: {predicted_steering_angle:+.3f}",
    ]

    if true_steering_angle is not None:
        overlay_lines.append(f"True: {true_steering_angle:+.3f}")

    font_scale, text_thickness, padding, line_spacing = _get_overlay_style(annotated)

    x = padding
    first_baseline_y = padding + 16

    max_text_width = 0
    total_text_height = padding * 2

    for text in overlay_lines:
        text_size, _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_thickness,
        )
        max_text_width = max(max_text_width, text_size[0])
        total_text_height += line_spacing

    box_width = max_text_width + padding * 2
    box_height = total_text_height

    box_width = min(box_width, annotated.shape[1])
    box_height = min(box_height, annotated.shape[0])

    # Dark compact background box for readability.
    cv2.rectangle(
        annotated,
        (0, 0),
        (box_width, box_height),
        (0, 0, 0),
        thickness=-1,
    )

    for line_index, text in enumerate(overlay_lines):
        text_y = first_baseline_y + line_index * line_spacing

        cv2.putText(
            annotated,
            text,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    return annotated


def save_annotated_frame(
    image_path: str | Path,
    output_path: str | Path,
    predicted_steering_angle: float,
    *,
    true_steering_angle: float | None = None,
) -> None:
    """Save one annotated steering-angle frame.

    Args:
        image_path: Source image path.
        output_path: Destination image path.
        predicted_steering_angle: Predicted steering angle.
        true_steering_angle: Optional true steering angle.
    """
    annotated = annotate_frame_with_steering(
        image_path,
        predicted_steering_angle,
        true_steering_angle=true_steering_angle,
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
            is also displayed.
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

        output_path = resolved_output_dir / f"annotated_frame_{row_number:05d}.jpg"

        save_annotated_frame(
            row["output_image_path"],
            output_path,
            float(row["predicted_steering_angle"]),
            true_steering_angle=true_steering_angle,
        )

        written_paths.append(output_path)

    return written_paths