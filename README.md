# Real-Time UAV Detection and Tracking System

A comprehensive, real-time drone detection, tracking, and intrusion alert system. Built with YOLOv8, Streamlit, and OpenCV, this system provides a robust dashboard for monitoring airspace, analyzing threats, and viewing runtime performance metrics.

## Features

- **Live Video & Image Processing:** Supports webcam, video files, and static images for drone detection.
- **Advanced Tracking & Trajectory Prediction:** Real-time tracking of drones with flight trails and trajectory prediction.
- **Geofencing & Threat Zones:** Configurable Low, Medium, and High threat zones with automated approach warnings.
- **Heading & Distance Estimation:** Calculates drone heading (using Oriented Bounding Boxes - OBB), true rotation angle, and estimates distance from the camera.
- **Drone Type Classification:** Differentiates between various drone types (Large UAV, Fixed-wing, Hexacopter, Nano quad, Quadcopter).
- **Heatmap Generation:** Visualizes flight paths and highest activity areas using cumulative heatmaps.
- **Analytics Dashboard:** Live metrics and graphs for FPS, GPU VRAM usage, detection confidence, zone hits, and alert timelines.
- **Alert Logging:** Comprehensive logging of intrusion events with CSV export capabilities.

## Tech Stack

- **Computer Vision:** OpenCV, YOLOv8 (Ultralytics)
- **Frontend / Dashboard:** Streamlit
- **Data Manipulation:** NumPy, Pandas
- **Visualization:** Matplotlib

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Eternal-Spirit/Real-Time-UAV-Detection-and-Tracking-System.git
   cd Real-Time-UAV-Detection-and-Tracking-System
   ```

2. Install the required dependencies:
   ```bash
   # Use the provided installation script if applicable, or install via pip
   pip install -r requirements.txt
   ```
   *(Note: Ensure you have PyTorch installed with CUDA support for optimal performance).*

## Usage

Run the Streamlit dashboard:

```bash
streamlit run app.py
```

- **Tab 1: Live Feed** - Monitor live video streams, view active drones, active threat zones, and recent intrusion logs.
- **Tab 2: Analytics** - Review runtime performance (FPS, GPU usage) and threat analysis graphs (Heading, Zone hits, Model training curves).

## Configuration

System settings, model paths, and geofence zones can be customized in the `config.py` file.

- `MODEL_PATH`: Update this to point to your latest trained weights.
- `ZONES`: Adjust the pixel coordinates to match your camera's field of view.
- `CONFIDENCE` / `IOU_THRESHOLD`: Fine-tune the detection thresholds.

## License

This project is licensed under the MIT License.
