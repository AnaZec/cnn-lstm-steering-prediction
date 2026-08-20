"""Inference + lane-change warning + visual presentation demo."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from dataset import SteeringSequenceGenerator, load_config, prepare_validation_data
from lane_change import add_lane_change_warnings


def _annotate_frame(row: pd.Series) -> np.ndarray:
    image = cv2.imread(str(row["output_image_path"]), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {row['output_image_path']}")

    predicted = float(row["predicted_steering_angle"])
    true = float(row["true_steering_angle"])
    warning = bool(row["lane_change_warning"])

    # Compact steering visualization: center line + predicted horizontal offset.
    height, width = image.shape[:2]
    cx = width // 2
    bar_y = height - max(35, height // 12)
    max_offset = max(60, width // 4)
    px = int(np.clip(cx + predicted * max_offset, 0, width - 1))

    cv2.line(image, (cx, bar_y - 12), (cx, bar_y + 12), (255, 255, 255), 2)
    cv2.line(image, (cx - max_offset, bar_y), (cx + max_offset, bar_y), (255, 255, 255), 2)
    cv2.circle(image, (px, bar_y), max(6, width // 150), (0, 255, 255), -1)

    cv2.rectangle(image, (0, 0), (min(width, 430), 78), (0, 0, 0), -1)
    cv2.putText(
        image,
        f"Predicted steering: {predicted:+.3f}",
        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"True steering:      {true:+.3f}",
        (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
    )

    if warning:
        label = "POSSIBLE LANE CHANGE"
        cv2.rectangle(image, (0, height - 38), (width, height), (0, 0, 255), -1)
        cv2.putText(
            image,
            label,
            (12, height - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
            (255, 255, 255), 2, cv2.LINE_AA,
        )
    return image


def run_demo(
    config_path: str = "config.yaml",
    max_sequences: int = 200,
    show_window: bool = True,
    delay_ms: int = 50,
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
    model = tf.keras.models.load_model(model_path)
    predicted = model.predict(generator, verbose=1).reshape(-1).astype(np.float32)

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

    print("\n=== PRESENTATION DEMO ===")
    print("Data: held-out validation split")
    print(f"Sequences: {len(results)}")
    print(f"Lane-change warning frames: {int(results['lane_change_warning'].sum())}")
    print("Press Q or Esc to close the demo window.\n")

    for i, row in results.iterrows():
        frame = _annotate_frame(row)
        cv2.imwrite(str(demo_dir / f"frame_{i:05d}.jpg"), frame)

        if show_window:
            cv2.imshow("CNN + LSTM Steering Prediction", frame)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (ord("q"), 27):
                break

    if show_window:
        cv2.destroyAllWindows()

    print(f"Predictions: {output_csv}")
    print(f"Annotated frames: {demo_dir}")
    return results
