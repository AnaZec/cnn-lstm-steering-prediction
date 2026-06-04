# CNN+LSTM Steering-Angle Prediction

This project predicts vehicle steering angle from a sequence of center-camera images using a compact CNN+LSTM model.

## Dataset Configuration

The project uses the Udacity Behavioral Cloning driving log and center-camera images. Local dataset paths are configured in `config.yaml`.

Update these values before running data loading, training, evaluation, or inference:

```yaml
dataset:
  driving_log_csv: "data/udacity/driving_log.csv"
  image_root: "data/udacity/IMG"
  camera: "center"
  center_camera_column: "center"
  steering_column: "steering"
```

`driving_log_csv` should point to the local Udacity driving log file, and `image_root` should point to the directory containing the camera image files.

The main implementation uses only the center camera. Left/right camera inputs, multi-camera fusion, object detection, semantic segmentation, and deployment-specific settings are outside the required scope.
