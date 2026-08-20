# CNN + LSTM Steering-Angle Prediction

The project predicts vehicle steering angle from short sequences of front-camera images and raises a warning when the predicted steering sequence indicates a sustained possible lane change.

## Project structure

```text
cnn-lstm-steering-prediction/
├── main.py          # one entry point for the whole project
├── dataset.py       # CSV loading, preprocessing, temporal sequences
├── model.py         # TimeDistributed CNN + LSTM model
├── train.py         # model training
├── evaluate.py      # held-out validation metrics and plots
├── lane_change.py   # steering-based warning heuristic
├── demo.py          # inference + visual presentation
├── config.yaml      # project parameters
└── requirements.txt
```

This structure follows the project flow directly:

```text
camera frames
    ↓
dataset.py: load + preprocess + make sequences
    ↓
model.py: CNN features for each frame + LSTM over time
    ↓
train.py / evaluate.py
    ↓
demo.py: steering prediction + lane-change warning
```

## Setup

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the Udacity behavioral-cloning dataset here:

```text
data/udacity/
├── driving_log.csv
└── IMG/
```

## Clean presentation workflow

Train the model once before the defense:

```bash
python main.py train
```

Evaluate it on the held-out validation split:

```bash
python main.py evaluate
```

### Start the presentation demo

```bash
python main.py
```

`demo` is the default command, so this is equivalent to:

```bash
python main.py demo
```

The demo uses **held-out validation sequences**, predicts the steering angle, displays predicted and true steering values on the camera frame, and shows a `POSSIBLE LANE CHANGE` warning when the steering-based detector is active.

Press **Q** or **Esc** to stop the OpenCV demo window.

Useful options:

```bash
python main.py demo --frames 100
python main.py demo --delay 80
python main.py demo --no-window
```

Run the complete workflow if needed:

```bash
python main.py all
```

## Model

For every sequence of five frames:

1. `TimeDistributed` applies the same CNN to every frame.
2. The CNN extracts spatial road-image features.
3. `GlobalAveragePooling2D` converts every frame to a compact feature vector.
4. The LSTM processes those vectors in temporal order.
5. Dense layers regress one steering-angle value for the final frame.

The model uses MSE as the training loss and MAE as a Keras metric.

## Important evaluation detail

The temporal train/validation split is performed on the **raw ordered frames before overlapping sequences are created**. This prevents the same source frames from leaking into both training and validation sequences.

Evaluation reports:

- MAE
- RMSE
- predicted-vs-true steering plot
- training-vs-validation loss plot

## Lane-change warning

The warning is deliberately simple and uses only the predicted steering sequence:

1. causal moving-average smoothing;
2. absolute steering threshold;
3. the threshold must remain active for a minimum number of consecutive frames.

It is a steering-pattern heuristic for the assignment, not a computer-vision lane-marker detector.
