# ============================================================
#  config.py  —  Drone Detection System
# ============================================================

# ── Model ────────────────────────────────────────────────────
MODEL       = 'yolov8s.pt'       # upgraded from yolov8n.pt
MODEL_PATH  = 'models/best.pt'   # update this after retraining

# ── Detection ─────────────────────────────────────────────────
CONFIDENCE      = 0.40
IOU_THRESHOLD   = 0.45
IMG_SIZE        = 640

# ── CLAHE preprocessing ───────────────────────────────────────
CLAHE_ENABLED   = True
CLAHE_CLIP      = 2.0
CLAHE_GRID      = (8, 8)

# ── Training ──────────────────────────────────────────────────
DATA_YAML       = 'drone.yaml'
EPOCHS          = 30
BATCH_SIZE      = 8              # lowered to 8 to prevent OOM
DEVICE          = 0              # RTX 4060 = device 0
WORKERS         = 0              # set to 0 to prevent WinError 1455 pagefile error
PROJECT         = 'runs/train'
RUN_NAME        = 'yolov8s_clahe_v1'

# ── Tracking ──────────────────────────────────────────────────
MAX_AGE         = 30
MIN_HITS        = 3
IOU_TRACK       = 0.3

# ── Output paths ──────────────────────────────────────────────
SNAPSHOT_DIR    = 'outputs/snapshots'
LOG_PATH        = 'outputs/intrusion_log.csv'

import numpy as np

# ── Geofence Zones ────────────────────────────────────────────
# 3 concentric polygon zones drawn over the camera frame.
# Coordinates are (x, y) pixel points for a 640×480 frame.
# Adjust these points to match your actual camera view / map.

GEOFENCE_ENABLED = True

ZONES = {
    'LOW': np.array([
        [0, 0], [640, 0], [640, 120], [0, 120]
    ], dtype=np.int32),

    'MEDIUM': np.array([
        [0, 120], [640, 120], [640, 240], [0, 240]
    ], dtype=np.int32),

    'HIGH': np.array([
        [0, 240], [640, 240], [640, 360], [0, 360]
    ], dtype=np.int32),
}

# Translucent overlay colours per zone (BGR format)
ZONE_COLORS = {
    'HIGH':   (0,   0,   255),   # red
    'MEDIUM': (0,  165,  255),   # orange
    'LOW':    (0,  255,  255),   # yellow
}

# Border colours (brighter versions for zone outlines)
ZONE_BORDER_COLORS = {
    'HIGH':   (60,  60,  255),
    'MEDIUM': (60, 185, 255),
    'LOW':    (60, 255, 255),
}

# How many consecutive frames a drone must be in a zone
# before the alert fires. HIGH fires instantly (1 frame).
ZONE_FRAMES = {
    'HIGH':   1,
    'MEDIUM': 3,
    'LOW':    5,
}

# Zone threat score used in combined threat formula (Phase 7+)
ZONE_THREAT_SCORE = {
    'HIGH':   50,
    'MEDIUM': 30,
    'LOW':    10,
}

# ── Trail / prediction toggles ────────────────────────────────
# These are the defaults. Streamlit sidebar can override them
# at runtime by passing show_trail/show_prediction/show_arrow
# directly into process_frame().

TRAIL_ENABLED      = True    # draw fading trail lines
PREDICTION_ENABLED = True    # draw orange prediction dots
ARROW_ENABLED      = True    # draw direction arrow

# ── Heatmap ───────────────────────────────────────────────────
HEATMAP_ENABLED      = False
HEATMAP_ALPHA        = 0.35     # overlay opacity (0.0 – 1.0)
HEATMAP_RADIUS       = 20       # Gaussian splat radius in pixels
HEATMAP_UPDATE_EVERY = 1        # update accumulator every N frames
HEATMAP_RENDER_EVERY = 1        # re-render overlay every N frames
                                 # set to 5–10 for big performance gain
                                 # on lower-end hardware

# Frame size — must match your camera output
FRAME_HEIGHT = 360
FRAME_WIDTH  = 640

import os

# ── Alert system ──────────────────────────────────────────────
SNAPSHOT_DIR       = 'outputs/snapshots'    # where snapshots are saved
ALERT_COOLDOWN     = 5.0                    # seconds between alerts per drone
                                             # raise to 10.0 if too noisy

ALERT_SOUND_ENABLED = False                  # set False to disable all sounds

# Beep frequencies in Hz (Windows winsound)
ALERT_BEEP_HIGH    = 1200   # urgent / high-pitched
ALERT_BEEP_MEDIUM  =  800   # warning
ALERT_BEEP_LOW     =  500   # notice
BEEP_DURATION_MS   =  350   # beep length in milliseconds

# Snapshot trigger: which zones save a snapshot
# Disabled completely to absolutely prevent any disk I/O lag
SNAPSHOT_ZONES = set()

# ── Stats & graphs ────────────────────────────────────────────
STATS_ENABLED        = False
GPU_POLL_INTERVAL    = 2.0    # seconds between nvidia-smi polls
GRAPH_REFRESH_FRAMES = 999999 # completely disabled to prevent matplotlib lag spikes

# Path to YOLOv8 training results CSV for model training curves
# Using the active document path the user had open
TRAINING_RESULTS_CSV = 'runs/detect/runs/train/yolov8s_clahe_v12/results.csv'

# ── Heading / angle ───────────────────────────────────────────
HEADING_ENABLED      = True
APPROACH_CONE_DEG    = 60.0   # degrees either side for "approaching" warning
MIN_MOVEMENT_PX      = 2      # min pixel displacement to register movement
SHOW_COMPASS         = True   # draw mini compass rose in frame corner
COMPASS_POSITION     = (40, 40)   # (x, y) pixel position of compass centre
COMPASS_RADIUS       = 22         # compass radius in pixels

# ── Phase 8: Drone Type ───────────────────────────────────────
# BGR colours for each drone type label drawn on frame
DRONE_TYPE_COLORS = {
    'Large UAV':  (0,   0,   255),   # Red
    'Fixed-wing': (0,   69,  255),   # Orange-red
    'Hexacopter': (0,   165, 255),   # Orange
    'Nano quad':  (255, 255, 0  ),   # Cyan
    'Quadcopter': (200, 200, 0  ),   # Yellow-green
}

# ── Phase 9: OBB True Rotation Angle ──────────────────────────
MODEL_PATH_OBB  = "models/obb_best.pt"   # set after training completes
USE_OBB         = True                # set False to fall back to Phase 7 heading (until trained)

# ── Phase 11: Distance Estimation ──────────────────────────
KNOWN_DRONE_WIDTH_M = 0.5        # Assumed real-world drone width in meters
FOCAL_LENGTH_PX     = 800        # Estimated camera focal length in pixels
