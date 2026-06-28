# CNN-LSTM Steering Prediction

This project implements a CNN+LSTM-based pipeline for steering-angle prediction from front-camera driving images.

The project is developed as a local Python/TensorFlow course project.

## Dataset Configuration

The project is intended to use the Udacity Self-Driving Car Behavioral Cloning dataset.

Dataset files should be placed locally and referenced from `config.yaml`.

Expected dataset inputs:

```text
data/
└── udacity/
    ├── driving_log.csv
    └── IMG/
        ├── center_*.jpg
        ├── left_*.jpg
        └── right_*.jpg
```

The initial implementation uses the center front camera images and steering-angle labels.

## Environment Setup

Python 3.11 or newer is recommended for local development and training.

On Ubuntu, make sure virtual environment support is installed:

```bash
sudo apt update
sudo apt install python3-full python3-venv
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip and install the project dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify that the main dependencies import correctly:

```bash
python -c "import tensorflow as tf; import cv2; import pandas as pd; import yaml; import sklearn; import matplotlib; print('OK')"
```

Expected output:

```text
OK
```

TensorFlow may print informational CPU/GPU messages. These are acceptable as long as the command finishes successfully and prints `OK`.

## Dependencies

The project dependencies are listed in `requirements.txt`.

Main dependencies:

* TensorFlow/Keras: CNN+LSTM model implementation, training, inference, and model saving/loading
* NumPy: numerical arrays and tensor manipulation
* pandas: loading and processing the driving log CSV file
* OpenCV: image loading, resizing, preprocessing, and output-frame visualization
* Matplotlib: training and evaluation plots
* scikit-learn: train/validation splitting and regression metrics
* PyYAML: reading configuration from `config.yaml`
* tqdm: progress bars during data processing

## Notes

This project is intended for local course-project training and evaluation.

The dependency list intentionally does not include object-detection, segmentation, cloud deployment, TensorRT, ONNX, or TFLite packages.

## Results Summary

A compact final results summary for course submission and defense preparation is available in:

```text
docs/results_summary.md
```

The summary reports validation MAE/RMSE, references generated plots, references annotated demo output, and documents project limitations.
