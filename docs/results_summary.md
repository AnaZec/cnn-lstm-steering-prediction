# Results Summary

This document summarizes the final results and generated artifacts for the CNN+LSTM steering-angle prediction project.

## Dataset and Split

The project uses the Udacity self-driving car behavioral cloning dataset with center-camera images only.

Configured input sequence:

- Sequence length: `5`
- Sequence stride: `1`
- Sequence target: `last_frame`
- Image shape: `(80, 160, 3)`
- Validation ratio: `0.2`

Local sample counts:

| Split | Raw frames | Sequences |
|---|---:|---:|
| Train | `2723` | `2719` |
| Validation | `681` | `677` |
| Total | `3404` | `3396` |

## Steering-Angle Evaluation

The trained CNN+LSTM model is evaluated on the validation sequence split.

| Metric | Value |
|---|---:|
| MAE | `TODO: copy value from outputs/evaluation/evaluation_metrics.json` |
| RMSE | `TODO: copy value from outputs/evaluation/evaluation_metrics.json` |

MAE reports the average absolute steering-angle error. RMSE penalizes larger steering-angle errors more strongly.

The predicted-vs-true plot shows that the model learns the broad steering trend, but its predictions are smoother than the true steering signal. Sharp steering spikes are often underestimated. This is expected for a compact CNN+LSTM model trained with a regression loss on a limited local course-project setup.

## Training Behavior

Training history artifact:

```text
outputs/models/training_history.json
```

The training and validation loss curves show whether the model continues to improve or starts to overfit. In the current run, validation loss decreases substantially and then plateaus, which suggests that further training may provide limited improvement without additional model or data changes.

## Generated Plots

The following plots are generated locally under `outputs/plots/`:

| Artifact | Path |
|---|---|
| Predicted vs true steering angle | `outputs/plots/predicted_vs_true_steering.png` |
| Training and validation loss | `outputs/plots/training_validation_loss.png` |

These plots can be used in the README, written report, or defense slides.

## Example Visual Output

Annotated demo frames are generated under:

```text
outputs/demo/annotated_frames
```

Example frame:

```text
outputs/demo/annotated_frames/annotated_frame_00000.jpg
```

The annotated output includes:

- the original center-camera frame
- predicted steering-angle overlay
- optional true steering-angle overlay when available
- lane-change warning overlay when the detector is active

## Inference and Lane-Change Warning Output

Inference predictions are saved to:

```text
outputs/inference/steering_predictions.csv
```

Local inference summary:

| Item | Value |
|---|---:|
| Prediction rows | `TODO: copy from outputs/inference/steering_predictions.csv` |
| Lane-change warning frames | `TODO: copy from lane_change_warning column` |
| Lane-change warning intervals | `TODO: copy max lane_change_event_id` |

The lane-change warning is based on the predicted steering-angle sequence. It is not based on lane markers, lane segmentation, lane boundaries, or real visual lane detection.

## Limitations

- The model is a compact course-project CNN+LSTM architecture, not a production autonomous-driving model.
- The dataset is limited and does not represent all road, weather, lighting, traffic, camera, and vehicle conditions.
- The validation split is temporal and useful for this project, but it is not equivalent to a fully independent real-world test set.
- The steering-angle target is predicted from image sequences, so prediction quality depends directly on image preprocessing, sequence length, dataset quality, and training time.
- The true steering signal contains abrupt frame-to-frame changes. The model predictions are smoother and may underestimate sharp steering peaks.
- Lane-change detection is heuristic. It is inferred from smoothed predicted steering-angle values, not from real lane detection.
- The lane-change warning should be interpreted as a possible steering-pattern warning, not as a verified lane-change event.
- If steering-angle predictions are weak or overly smooth, lane-change warning quality is also limited.
- The demo visualizations are intended for documentation and defense presentation, not for safety-critical use.
- Performance depends on local CPU/GPU availability. CPU-only training and inference may be slow.

## Reproducibility Commands

Train the model:

```bash
python -m src.train --config config.yaml
```

Evaluate the model and generate plots:

```bash
python -m src.evaluate --config config.yaml
```

Run inference and export annotated demo frames:

```bash
python -m src.infer \
  --config config.yaml \
  --max-sequences 200 \
  --export-demo-frames \
  --max-demo-frames 50
```