"""Visual inference demo for CNN+LSTM steering-angle prediction."""

from __future__ import annotations

import os
from pathlib import Path

# Keep demo output readable on CPU-only Linux systems.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland":
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from dataset import SteeringSequenceGenerator, load_config, prepare_validation_data
from lane_change import add_lane_change_warnings


DISPLAY_WIDTH = 960
FRAME_HEIGHT = 480
FOOTER_HEIGHT = 125
BACKGROUND = (24, 24, 24)
TEXT = (235, 235, 235)
MUTED = (175, 175, 175)
PREDICTED = (0, 220, 255)
GROUND_TRUTH = (120, 210, 120)
WARNING = (80, 80, 230)


def _fit_frame(image: np.ndarray) -> np.ndarray:
    """Fit the source image into the demo area while preserving aspect ratio."""
    src_h, src_w = image.shape[:2]
    scale = min(DISPLAY_WIDTH / src_w, FRAME_HEIGHT / src_h)
    width = max(1, int(round(src_w * scale)))
    height = max(1, int(round(src_h * scale)))

    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    frame = np.full((FRAME_HEIGHT, DISPLAY_WIDTH, 3), BACKGROUND, dtype=np.uint8)
    x = (DISPLAY_WIDTH - width) // 2
    y = (FRAME_HEIGHT - height) // 2
    frame[y : y + height, x : x + width] = resized
    return frame


def _steering_x(value: float, x0: int, x1: int) -> int:
    value = float(np.clip(value, -1.0, 1.0))
    return int(round(x0 + (value + 1.0) * 0.5 * (x1 - x0)))


def _annotate_frame(row: pd.Series) -> np.ndarray:
    image = cv2.imread(str(row["output_image_path"]), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {row['output_image_path']}")

    predicted = float(row["predicted_steering_angle"])
    true = float(row["true_steering_angle"])
    smoothed = float(row["smoothed_steering_angle"])
    warning = bool(row["lane_change_warning"])

    frame = _fit_frame(image)
    canvas = np.full(
        (FRAME_HEIGHT + FOOTER_HEIGHT, DISPLAY_WIDTH, 3),
        BACKGROUND,
        dtype=np.uint8,
    )
    canvas[:FRAME_HEIGHT] = frame

    # Small numeric readout.
    cv2.putText(
        canvas,
        f"Predicted: {predicted:+.3f}",
        (35, FRAME_HEIGHT + 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        PREDICTED,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"True: {true:+.3f}",
        (245, FRAME_HEIGHT + 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        GROUND_TRUTH,
        2,
        cv2.LINE_AA,
    )

    # Simple left-right steering scale shared by prediction and ground truth.
    x0, x1 = 90, DISPLAY_WIDTH - 90
    y = FRAME_HEIGHT + 76
    center = (x0 + x1) // 2

    cv2.line(canvas, (x0, y), (x1, y), MUTED, 2, cv2.LINE_AA)
    cv2.line(canvas, (center, y - 8), (center, y + 8), MUTED, 1, cv2.LINE_AA)

    pred_x = _steering_x(predicted, x0, x1)
    true_x = _steering_x(true, x0, x1)
    cv2.circle(canvas, (pred_x, y), 7, PREDICTED, -1, cv2.LINE_AA)
    cv2.circle(canvas, (true_x, y), 8, GROUND_TRUTH, 2, cv2.LINE_AA)

    cv2.putText(
        canvas,
        "LEFT",
        (x0 - 18, FRAME_HEIGHT + 111),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        MUTED,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "RIGHT",
        (x1 - 25, FRAME_HEIGHT + 111),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        MUTED,
        1,
        cv2.LINE_AA,
    )

    # Keep the warning deliberately unobtrusive and show it only when active.
    if warning:
        direction = "LEFT" if smoothed < 0 else "RIGHT"
        label = f"Possible lane change: {direction}"
        (text_w, _), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
        )
        cv2.putText(
            canvas,
            label,
            (DISPLAY_WIDTH - text_w - 35, FRAME_HEIGHT + 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            WARNING,
            1,
            cv2.LINE_AA,
        )

    return canvas


def run_demo(
    config_path: str = "config.yaml",
    max_sequences: int = 200,
    show_window: bool = True,
    delay_ms: int = 50,
    save_video: bool = False,
) -> pd.DataFrame:
    """Run the trained model on held-out validation sequences and show results."""
    config = load_config(config_path)
    outputs = config["outputs"]
    model_path = Path(outputs["checkpoint_path"])

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}. Run 'python main.py train'.")

    validation_index, _ = prepare_validation_data(config)
    if max_sequences is not None:
        validation_index = validation_index.head(max_sequences).reset_index(drop=True)

    generator = SteeringSequenceGenerator(validation_index, config, shuffle=False)

    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"Running demo on {len(validation_index)} validation sequences...")
    predicted = model.predict(generator, verbose=0).reshape(-1).astype(np.float32)

    results = pd.DataFrame(
        {
            "sequence_index": np.arange(len(validation_index)),
            "output_image_path": [paths[-1] for paths in validation_index["frame_paths"]],
            "true_steering_angle": validation_index["target_steering_angle"].astype(float),
            "predicted_steering_angle": predicted,
        }
    )
    results = add_lane_change_warnings(results, config)

    output_csv = Path(outputs["inference_predictions_path"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)

    demo_dir = Path(outputs["demo_frames_dir"])
    demo_dir.mkdir(parents=True, exist_ok=True)

    if show_window:
        print("Press Q or Esc to exit.")
        cv2.namedWindow("CNN + LSTM Steering Prediction", cv2.WINDOW_AUTOSIZE)

    video_writer = None
    video_path = Path("outputs/demo/steering_demo.mp4")
    if save_video:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        fps = 1000.0 / max(delay_ms, 1)
        frame_size = (DISPLAY_WIDTH, FRAME_HEIGHT + FOOTER_HEIGHT)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, frame_size)
        if not video_writer.isOpened():
            raise RuntimeError(f"Could not create video: {video_path}")

    try:
        for i, row in results.iterrows():
            annotated = _annotate_frame(row)
            cv2.imwrite(str(demo_dir / f"frame_{i:05d}.jpg"), annotated)

            if video_writer is not None:
                video_writer.write(annotated)

            if show_window:
                cv2.imshow("CNN + LSTM Steering Prediction", annotated)
                key = cv2.waitKey(delay_ms) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        if video_writer is not None:
            video_writer.release()
        if show_window:
            cv2.destroyAllWindows()

    print(f"Predictions saved to: {output_csv}")
    print(f"Annotated frames saved to: {demo_dir}")
    if save_video:
        print(f"Demo video saved to: {video_path}")
    return results
