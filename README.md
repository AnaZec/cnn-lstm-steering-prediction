# CNN + LSTM Steering-Angle Prediction

Implementation of steering-angle prediction from a temporal sequence of front-camera images using a combined **CNN + LSTM** architecture. The predicted steering signal is additionally processed to detect sustained steering deviations that may indicate a lane change.

The project is implemented in Python with TensorFlow/Keras and uses the **Udacity Self-Driving Car – Behavioural Cloning** dataset. The current solution uses the **center front camera** and its corresponding steering-angle labels.

## Implemented tasks

| Project requirement | Implementation |
| --- | --- |
| Predict steering angle from a sequence of camera images | TimeDistributed CNN + LSTM regression model |
| Extract spatial features from each frame | CNN applied independently to every frame |
| Process temporal information | LSTM over the sequence of CNN feature vectors |
| Display the predicted steering angle | OpenCV inference visualization |
| Detect possible lane changes from predicted steering | Moving-average smoothing + steering threshold + minimum duration |
| Signal a detected lane change | Visual warning in the inference window |
| Evaluate the solution | Held-out validation split, MAE, RMSE and comparison plots |

## Processing pipeline

```text
center-camera frames
        ↓
resize + RGB conversion + normalization
        ↓
sequences of 5 consecutive frames
        ↓
TimeDistributed CNN
        ↓
per-frame feature vectors
        ↓
LSTM
        ↓
steering-angle prediction
        ↓
smoothing + threshold + persistence check
        ↓
possible lane-change warning
```

## Dataset and preprocessing

Expected dataset structure:

```text
data/udacity/
├── driving_log.csv
└── IMG/
```

The CSV contains camera-image paths and vehicle measurements, including the steering angle. Steering labels in the dataset are continuous values in the range **[-1, 1]**.

Only the center-camera image is used by this implementation.

Each input image is:

- read with OpenCV;
- converted from BGR to RGB;
- resized to **160 × 80** pixels;
- normalized to the **[0, 1]** range.

The temporal order of the recorded samples is preserved.

### Sequence construction

The model receives **5 consecutive frames** with stride **1**:

```text
frame(t-4), frame(t-3), frame(t-2), frame(t-1), frame(t)
```

The target value is the steering angle associated with the final frame, `frame(t)`.

The ordered raw samples are split into training and validation partitions **before** overlapping sequences are generated. The default split is **80% training / 20% validation**. This prevents shared source frames from appearing in both partitions through overlapping sliding windows.

## Model architecture

```text
Input: 5 × 80 × 160 × 3
        ↓
TimeDistributed Conv2D(16, 5×5, stride 2, ReLU)
        ↓
TimeDistributed MaxPooling2D(2×2)
        ↓
TimeDistributed Conv2D(32, 3×3, ReLU)
        ↓
TimeDistributed MaxPooling2D(2×2)
        ↓
TimeDistributed Conv2D(64, 3×3, ReLU)
        ↓
TimeDistributed GlobalAveragePooling2D
        ↓
LSTM(64)
        ↓
Dense(32, ReLU)
        ↓
Dropout(0.2)
        ↓
Dense(1, linear)
        ↓
Predicted steering angle
```

`TimeDistributed` applies the same CNN to each frame. The CNN extracts spatial road-scene features, while the LSTM models how those features change across the five-frame sequence. The final dense layer performs single-value regression of the steering angle.

## Training

Default parameters are defined in `config.yaml`:

| Parameter | Value |
| --- | ---: |
| Batch size | 32 |
| Maximum epochs | 10 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss | Mean Squared Error (MSE) |
| Keras metric | Mean Absolute Error (MAE) |
| Early-stopping patience | 3 epochs |
| Random seed | 42 |

Training uses validation loss for both checkpoint selection and early stopping. The checkpoint with the lowest validation loss is saved as:

```text
outputs/models/best_model.keras
```

## Evaluation

Evaluation is performed on the held-out validation sequences using:

- **MAE** – mean absolute steering-angle error;
- **RMSE** – root mean squared steering-angle error.

Generated evaluation artifacts:

```text
outputs/evaluation/evaluation_metrics.json
outputs/plots/predicted_vs_true_steering.png
outputs/plots/training_validation_loss.png
```

The first plot compares predicted and ground-truth steering values on validation sequences. The second shows training and validation MSE across epochs.

## Lane-change detection

Lane-change detection is implemented as post-processing of the **predicted steering sequence** rather than as a separate neural network.

The algorithm:

1. smooths predicted steering values with a causal moving average;
2. checks whether the absolute smoothed steering value exceeds a threshold;
3. requires the condition to persist for a minimum number of consecutive frames;
4. marks the sustained interval as a possible lane-change event.

Default values:

| Parameter | Value |
| --- | ---: |
| Smoothing window | 5 frames |
| Steering threshold | 0.15 |
| Minimum duration | 5 frames |

The sign of the smoothed prediction indicates the direction of the steering deviation; the warning condition itself is based on the magnitude and duration of that deviation.

This is a steering-signal heuristic. It does not detect or track lane markings in the image.

## Running the project

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train the model:

```bash
python main.py train
```

Evaluate the best saved checkpoint:

```bash
python main.py evaluate
```

Run inference and visualization on held-out validation sequences:

```bash
python main.py demo
```

`python main.py` starts the same inference demo by default. Press **Q** or **Esc** to close the OpenCV window.

Run the complete workflow:

```bash
python main.py all
```

Optional demo arguments:

```bash
python main.py demo --frames 100
python main.py demo --delay 80
python main.py demo --no-window
```

## Generated outputs

```text
outputs/
├── models/
│   ├── best_model.keras
│   ├── final_model.keras
│   └── training_history.json
├── evaluation/
│   └── evaluation_metrics.json
├── plots/
│   ├── predicted_vs_true_steering.png
│   └── training_validation_loss.png
├── inference/
│   └── steering_predictions.csv
└── demo/
    └── annotated_frames/
```

## Project structure

```text
cnn-lstm-steering-prediction/
├── main.py          # single command-line entry point
├── dataset.py       # loading, preprocessing, split and sequence generation
├── model.py         # CNN+LSTM model definition
├── train.py         # training, checkpointing and history export
├── evaluate.py      # validation metrics and plots
├── lane_change.py   # steering-based lane-change heuristic
├── demo.py          # inference and OpenCV visualization
├── config.yaml      # experiment configuration
└── requirements.txt # Python dependencies
```

All dataset paths, model parameters, training parameters, lane-change thresholds and output paths are configured in `config.yaml`.
