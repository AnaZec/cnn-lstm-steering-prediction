# Model Architecture and Lane-Change Detection Logic

This document explains the CNN+LSTM steering-angle model and the threshold-based possible lane-change detector used in this project.

## Steering-Angle Prediction Pipeline

The goal of the model is to predict one steering-angle value from a fixed-length sequence of front-camera images.

The processing pipeline is:

```text
ordered center-camera images
-> image preprocessing
-> fixed-length frame sequences
-> CNN feature extraction per frame
-> LSTM sequence processing
-> dense regression output
-> predicted steering angle
```

The model does not perform object detection, semantic segmentation, lane-marker detection, or geometric lane estimation.

## Input Representation

Each input sample is a sequence of preprocessed images.

For the current configuration:

```text
(sequence_length, image_height, image_width, channels)
= (5, 80, 160, 3)
```

A training batch therefore has shape:

```text
(batch_size, 5, 80, 160, 3)
```

Each sequence contains consecutive center-camera frames. The target steering angle is assigned from the final frame in the sequence.

Example:

```text
frames 0, 1, 2, 3, 4 -> target steering angle from frame 4
frames 1, 2, 3, 4, 5 -> target steering angle from frame 5
```

This keeps the input sequence and output target temporally aligned.

## Image Preprocessing

Before model input, each image is:

* loaded from disk
* converted to the expected color format
* resized to the configured image size
* normalized to a model-friendly numeric range

The same preprocessing function is reused for training, evaluation, and inference. This avoids a train/inference mismatch.

## CNN Feature Extraction

The CNN part of the model extracts spatial features from each image frame.

The CNN is applied independently to every frame in the input sequence using a sequence-aware wrapper such as `TimeDistributed`.

Conceptually:

```text
frame 0 -> CNN -> feature vector 0
frame 1 -> CNN -> feature vector 1
frame 2 -> CNN -> feature vector 2
frame 3 -> CNN -> feature vector 3
frame 4 -> CNN -> feature vector 4
```

The CNN layers learn local visual features useful for steering prediction, such as road shape, image perspective, curves, and visual context from the front camera.

The implementation intentionally uses a compact CNN instead of a large pretrained model. This keeps training practical for a local course-project environment.

## LSTM Sequence Processing

After CNN feature extraction, the model has a sequence of feature vectors.

The LSTM processes these feature vectors in temporal order:

```text
feature vector 0
-> feature vector 1
-> feature vector 2
-> feature vector 3
-> feature vector 4
-> LSTM output
```

The LSTM is used to model short-term temporal context across consecutive frames. This is useful because steering behavior is not determined only by a single isolated image; the recent frame sequence can provide smoother driving context.

## Steering-Angle Regression Output

The final part of the model is a dense regression head.

It outputs one floating-point value:

```text
predicted_steering_angle
```

This is a regression problem, not a classification problem. The model is trained with a regression loss and evaluated using regression metrics such as MAE and RMSE.

## Lane-Change Detection Input

The possible lane-change detector uses only the predicted steering-angle sequence.

Input:

```text
predicted_steering_angle[0]
predicted_steering_angle[1]
predicted_steering_angle[2]
...
```

The detector does not use:

* lane markers
* lane boundaries
* image segmentation
* lane geometry
* object detection
* real-world lane position

Therefore, the warning should be interpreted as a possible steering-pattern warning, not as verified visual lane-change detection.

## Steering-Angle Smoothing

Before thresholding, the predicted steering-angle sequence is smoothed.

The current implementation uses a causal moving average. This means each smoothed value uses only the current and previous predicted steering values.

For a smoothing window of 3:

```text
smoothed[0] = mean(predicted[0])
smoothed[1] = mean(predicted[0:2])
smoothed[2] = mean(predicted[0:3])
smoothed[3] = mean(predicted[1:4])
```

This keeps the output aligned with the original prediction sequence and handles short startup sequences without crashing.

Smoothing reduces noisy frame-by-frame warning changes.

## Threshold-Based Warning Logic

The detector applies a simple rule:

1. Smooth the predicted steering-angle sequence.
2. Compute the absolute smoothed steering angle.
3. Mark frames where the absolute smoothed steering angle exceeds the configured threshold.
4. Keep only sustained active intervals lasting at least the configured minimum duration.
5. Assign event IDs to detected warning intervals.

Conceptually:

```text
abs(smoothed_steering_angle) >= steering_threshold
```

for at least:

```text
min_duration_frames
```

consecutive frames.

The output includes per-frame warning state:

```text
lane_change_warning = true / false
```

and warning interval identifier:

```text
lane_change_event_id
```

Frames outside warning intervals receive event ID `0`.

## Visual Warning Overlay

During inference/demo export, the warning state can be drawn on output frames.

The frame visualization includes:

* predicted steering angle
* optional true steering angle when available
* lane-change warning text only when `lane_change_warning` is active

The warning overlay is visually distinct from the steering-angle text and does not claim to show real lane boundaries.

## Limitations

The implemented lane-change warning is heuristic.

Important limitations:

* The detector depends directly on steering prediction quality.
* If predicted steering angles are too smooth or inaccurate, warning quality is also limited.
* The method does not detect real lane markings or vehicle lane position.
* The method can miss lane changes that do not create a large predicted steering deviation.
* The method can produce warning intervals for sharp steering behavior that is not actually a lane change.
* The threshold and minimum duration are configurable and may need adjustment for different datasets or driving conditions.
* This is suitable for a course-project demonstration, not for safety-critical use.

## Summary

The CNN+LSTM model predicts steering angle from short image sequences. The CNN extracts spatial features from each frame, the LSTM processes the ordered feature sequence, and the dense regression head outputs one steering-angle value.

The lane-change warning logic is intentionally simple: it smooths the predicted steering-angle sequence and applies threshold-based sustained-deviation detection. It avoids any claim of real lane detection.
