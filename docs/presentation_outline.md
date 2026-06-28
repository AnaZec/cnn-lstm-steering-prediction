# Defense Presentation Outline

This document outlines a short defense presentation for the CNN+LSTM steering-angle prediction and lane-change warning project.

## Slide 1 — Title and Project Goal

**Title:** CNN+LSTM Steering-Angle Prediction from Front-Camera Image Sequences

**Key points:**

* Goal: predict vehicle steering angle from short sequences of front-camera images.
* Input: ordered center-camera frames from the Udacity behavioral cloning dataset.
* Output: one predicted steering-angle value per image sequence.
* Additional demo feature: possible lane-change warning based on predicted steering-angle behavior.

**Speaker notes:**

This project focuses on sequence-based steering-angle prediction. The model does not control a vehicle; it predicts steering angle for a dataset-based course project and exports visual demo frames.

## Slide 2 — Dataset and Input Data

**Key points:**

* Dataset: Udacity behavioral cloning dataset.
* Main implementation uses the center camera.
* CSV contains image paths and steering-angle labels.
* Images are loaded from `data/udacity/IMG`.
* Steering-angle target is read from `driving_log.csv`.

**Suggested visual:**

* Small diagram: `driving_log.csv + IMG/ -> ordered samples`.
* Optional screenshot of dataset folder structure.

**Speaker notes:**

The project uses center-camera images only to keep the implementation focused and aligned with the required course scope.

## Slide 3 — Preprocessing and Sequence Construction

**Key points:**

* Images are loaded, resized, and normalized.
* Current configured image shape: `(80, 160, 3)`.
* Fixed-length sequences are built from consecutive frames.
* Current sequence length: `5`.
* Target strategy: steering angle from the last frame in the sequence.

**Example:**

```text
frames 0, 1, 2, 3, 4 -> steering angle from frame 4
frames 1, 2, 3, 4, 5 -> steering angle from frame 5
```

**Speaker notes:**

The sequence construction allows the model to use short-term temporal context instead of predicting from a single isolated image.

## Slide 4 — CNN+LSTM Model Architecture

**Key points:**

* CNN extracts spatial features from each frame.
* The same CNN is applied independently to each frame in the sequence.
* LSTM processes the ordered sequence of CNN feature vectors.
* Dense regression head outputs one steering-angle prediction.

**Suggested diagram:**

```text
frame sequence
-> TimeDistributed CNN
-> feature sequence
-> LSTM
-> Dense layers
-> predicted steering angle
```

**Speaker notes:**

The CNN handles visual feature extraction. The LSTM handles temporal ordering across the frame sequence. The final output is a continuous value, so this is a regression task.

## Slide 5 — Training Setup

**Key points:**

* Loss function: regression loss for steering-angle prediction.
* Metrics: MAE and RMSE during evaluation.
* Temporal train/validation split is used to reduce sequence leakage.
* Model checkpoints are saved during training.

**Command:**

```bash
python -m src.train --config config.yaml
```

**Generated outputs:**

```text
outputs/models/best_model.keras
outputs/models/final_model.keras
outputs/models/training_history.json
```

**Speaker notes:**

The project saves both the best checkpoint and the final model so evaluation and inference can reuse trained artifacts.

## Slide 6 — Evaluation Metrics and Results

**Key points:**

* Evaluation is performed on the validation sequence split.
* MAE measures average absolute steering-angle error.
* RMSE penalizes larger steering-angle errors more strongly.
* Predicted-vs-true plot shows general trend following.
* Predictions are smoother than the true steering signal and can underestimate sharp steering changes.

**Command:**

```bash
python -m src.evaluate --config config.yaml
```

**Generated outputs:**

```text
outputs/evaluation/evaluation_metrics.json
outputs/plots/predicted_vs_true_steering.png
outputs/plots/training_validation_loss.png
```

**Suggested visual:**

* `outputs/plots/predicted_vs_true_steering.png`
* `outputs/plots/training_validation_loss.png`

**Speaker notes:**

The model learns broad steering behavior, but sharp frame-to-frame steering spikes remain difficult. This is an important limitation to state clearly.

## Slide 7 — Inference and Visual Demo Output

**Key points:**

* Inference runs over ordered image sequences.
* Each prediction is aligned with the final frame of its input sequence.
* Predictions are saved to CSV.
* Demo frames can be exported with steering-angle overlays.

**Command:**

```bash
python -m src.infer \
  --config config.yaml \
  --max-sequences 200 \
  --export-demo-frames \
  --max-demo-frames 50
```

**Generated outputs:**

```text
outputs/inference/steering_predictions.csv
outputs/demo/annotated_frames/
```

**Suggested visual:**

* One annotated demo frame from `outputs/demo/annotated_frames/`.

**Speaker notes:**

The visual output is useful for defense because it shows the predicted steering angle directly on the original camera frame.

## Slide 8 — Lane-Change Warning Logic

**Key points:**

* Lane-change warning is based only on predicted steering angles.
* Predicted steering sequence is smoothed with a causal moving average.
* Absolute smoothed steering angle is compared with a threshold.
* Sustained threshold activation becomes a possible lane-change warning.
* Warning intervals receive `lane_change_event_id`.

**Logic:**

```text
predicted steering angles
-> smoothing
-> absolute value
-> threshold check
-> minimum duration filter
-> warning interval
```

**Important statement:**

This is not real lane detection. The project does not detect lane markers, lane boundaries, lane geometry, or vehicle position inside a lane.

**Speaker notes:**

The warning is a steering-pattern heuristic. It should be presented as a possible warning signal, not as verified lane-change detection.

## Slide 9 — Testing and Verification

**Key points:**

* Data loading smoke tests confirm dataset parsing and valid sample loading.
* Preprocessing checks confirm image resizing and normalization.
* Sequence generator checks confirm batch shape.
* Training command verifies model training and checkpoint output.
* Evaluation command verifies metrics and plot generation.
* Inference command verifies prediction CSV and annotated frame export.
* Lane-change logic is checked with prediction sequences and warning output columns.

**Example checked outputs:**

```text
outputs/models/
outputs/evaluation/
outputs/plots/
outputs/inference/
outputs/demo/annotated_frames/
```

**Speaker notes:**

The verification approach is practical and focused on the implemented pipeline. It confirms that each pipeline stage produces the expected artifact.

## Slide 10 — Limitations and Future Improvements

**Limitations:**

* Compact course-project model, not a production autonomous-driving system.
* Uses center-camera images only.
* Dataset is limited and does not cover all real-world driving conditions.
* Predictions are smoother than true steering labels.
* Sharp steering peaks may be underestimated.
* Lane-change warning depends directly on steering prediction quality.
* Lane-change warning is heuristic and not based on visual lane detection.
* CPU-only training and inference can be slow.

**Possible future improvements:**

* Train longer and tune hyperparameters.
* Use stronger CNN backbone.
* Add data augmentation.
* Use all three cameras with steering correction.
* Evaluate on a separate driving track or independent dataset.
* Add real lane-marker or lane-boundary detection only as a separate future extension.

**Speaker notes:**

The future work should not expand the required implementation scope. It should show awareness of limitations without claiming production readiness.

## Slide 11 — Final Summary

**Key points:**

* Implemented a complete CNN+LSTM steering-angle prediction pipeline.
* Built dataset loading, preprocessing, sequence generation, training, evaluation, and inference.
* Exported metrics, plots, prediction CSV files, and annotated demo frames.
* Added a simple possible lane-change warning based on predicted steering-angle sequences.
* Documented limitations clearly and avoided claiming real lane detection.

**Speaker notes:**

The final message should be that the project meets the course requirements: steering-angle prediction from image sequences, CNN+LSTM architecture, visual output, and warning logic based on predicted steering behavior.
