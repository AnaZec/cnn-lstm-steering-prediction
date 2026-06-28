## Setup and Run Instructions

This project predicts vehicle steering angle from short sequences of front-camera images using a CNN+LSTM model. The main implementation uses the **center camera** from the Udacity behavioral cloning dataset.

## Environment Setup

Python 3.11 is recommended.

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify that TensorFlow imports correctly:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

## Dataset Placement

The project expects the Udacity behavioral cloning dataset to be placed under:

```text
data/udacity/
```

Expected local structure:

```text
data/udacity/
├── driving_log.csv
└── IMG/
    ├── center_*.jpg
    ├── left_*.jpg
    └── right_*.jpg
```

The implementation uses the **center-camera image path** and steering-angle column from `driving_log.csv`.

The default dataset configuration is in `config.yaml`:

```yaml
dataset:
  driving_log_csv: "data/udacity/driving_log.csv"
  image_root: "data/udacity/IMG"
  camera: "center"
  center_camera_column: "centercam"
  steering_column: "steering_angle"
```

If your dataset is stored somewhere else, update only the dataset paths in `config.yaml`.

The `data/` directory is intentionally ignored by Git because the dataset is large and should not be committed.

## Training

Train the CNN+LSTM model:

```bash
python -m src.train --config config.yaml
```

Training saves model artifacts under:

```text
outputs/models/
```

Main generated files:

```text
outputs/models/best_model.keras
outputs/models/final_model.keras
outputs/models/training_history.json
```

The `outputs/` directory is ignored by Git because it contains generated artifacts.

## Evaluation

Evaluate the trained model on the validation split:

```bash
python -m src.evaluate --config config.yaml
```

Evaluation saves metrics and plots under:

```text
outputs/evaluation/
outputs/plots/
```

Main generated files:

```text
outputs/evaluation/evaluation_metrics.json
outputs/plots/predicted_vs_true_steering.png
outputs/plots/training_validation_loss.png
```

## Inference and Demo Frame Export

Run inference on ordered image sequences:

```bash
python -m src.infer --config config.yaml --max-sequences 200
```

Run inference and export annotated demo frames:

```bash
python -m src.infer \
  --config config.yaml \
  --max-sequences 200 \
  --export-demo-frames \
  --max-demo-frames 50
```

Inference saves predictions to:

```text
outputs/inference/steering_predictions.csv
```

Annotated demo frames are saved to:

```text
outputs/demo/annotated_frames/
```

The exported frames can include:

* predicted steering-angle overlay
* optional true steering-angle overlay
* lane-change warning overlay when active

## Lane-Change Warning

The lane-change warning is based on the predicted steering-angle sequence. The detector smooths predicted steering values and applies threshold-based sustained-deviation logic.

This project does **not** perform real lane detection. It does not detect lane markers, lane boundaries, lane geometry, or vehicle position within the lane.

The warning should be interpreted as a steering-pattern heuristic for demonstration purposes.
