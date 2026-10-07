# Real-Time UAV Detection and Tracking System

A high-performance, real-time drone detection, tracking, and intrusion alert system optimized for high-resolution (up to 4K UHD) surveillance feeds. Built with YOLOv8, Streamlit, ByteTrack, PyTorch CUDA, and OpenCV, this system provides a robust dashboard for monitoring airspace, analyzing threats, and viewing runtime performance metrics.

## Key Highlights & Performance

- **High-Resolution 4K Optimization:** Implements pre-inference bilateral scaling (`3840x2160 -> 1080p/720p`) reducing memory pressure by ~75% and accelerating inference throughput from **~9 FPS to 35+ FPS** on modern CUDA GPUs.
- **CUDA FP16 Half-Precision Engine:** Fully hardware-accelerated pipeline with asynchronous memory streaming, automatic PyTorch CUDA cache clearing, and half-precision execution.
- **Oriented Bounding Box (OBB) & Heading Tracking:** Fine-tuned YOLOv8-OBB models detect rotated UAV orientations with precise polygon bounding boxes, enabling real-time heading angles and flight trajectory prediction.
- **ByteTrack Multi-Object Tracking:** Robust association through occlusions and rapid maneuvers with persistent track IDs and Kalman filter estimation.
- **Geofencing & Threat Zones:** Multi-tier threat classification (Low, Medium, High) with automatic perimeter breach alerts and CSV export.
- **Interactive Analytics Dashboard:** Real-time telemetry monitoring FPS, GPU VRAM usage, detection confidence, active threat zones, and flight heatmaps.

## Features

- **Live Video & Image Processing:** Supports webcam, video files (MP4, AVI, MKV), and static images.
- **Advanced Tracking & Trajectory Prediction:** Real-time tracking of drones with dynamic flight trails and speed/heading vector estimation.
- **Geofencing & Threat Zones:** Configurable polygonal boundary zones with intrusion timers and automated approach warnings.
- **Drone Classification:** Differentiates between UAV types (Nano quad, Quadcopter, Hexacopter, Fixed-wing, Large UAV).
- **Heatmap Generation:** Visualizes spatial flight paths and cumulative high-density airspace activity.
- **Analytics Dashboard:** Live metrics, confidence distributions, GPU load graphs, and alert logs.
- **CSV & Telemetry Export:** Structured event logs recording track ID, drone class, confidence, coordinates, heading, and threat level.

## Tech Stack

- **Computer Vision & AI:** YOLOv8 / YOLOv8-OBB (Ultralytics), OpenCV, ByteTrack, PyTorch (CUDA)
- **Frontend / Dashboard:** Streamlit
- **Data & Telemetry:** NumPy, Pandas, Matplotlib

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Eternal-Spirit/Real-Time-UAV-Detection-and-Tracking-System.git
   cd Real-Time-UAV-Detection-and-Tracking-System
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure PyTorch is installed with CUDA support for hardware acceleration).*

## Usage

Run the Streamlit dashboard:

```bash
streamlit run app.py
```

- **Tab 1: Live Feed** - Monitor live video streams, adjust confidence/IOU thresholds, toggle OBB tracking, and view real-time intrusion alerts.
- **Tab 2: Analytics** - Inspect real-time FPS benchmarks, GPU VRAM utilization, heading rose plots, and zone hit distribution.

## Model Checkpoints

The repository includes ready-to-run fine-tuned model weights:
- `models/best.pt`: YOLOv8 standard bounding box model fine-tuned for small and fast-moving UAVs.
- `models/obb_best.pt`: YOLOv8-OBB oriented bounding box model for precision rotation and heading estimation.

## Configuration

System settings, model paths, and geofence zones can be customized in `config.py`:
- `MODEL_PATH` / `OBB_MODEL_PATH`: Model weight filepaths.
- `INFERENCE_RESOLUTION`: Downscaling dimensions for high-resolution (4K) inputs.
- `CONFIDENCE` / `IOU_THRESHOLD`: Detection and NMS thresholds.
- `ZONES`: Geofence polygon boundary coordinates.

## License

This project is licensed under the MIT License.
