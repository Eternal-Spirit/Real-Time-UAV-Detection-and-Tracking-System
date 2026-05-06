import cv2
import math
import numpy as np
from ultralytics import YOLO

# Load OBB model once at module level
_obb_model = None

def load_obb_model(weights_path='models/obb_best.pt'):
    """Call once at startup to load the OBB model into memory."""
    global _obb_model
    _obb_model = YOLO(weights_path)
    print(f"OBB model loaded: {weights_path}")

def track_with_angle(frame, conf_thresh=0.40):
    """
    Run YOLOv8-OBB inference on frame and uses model.track() to get persistent track IDs.
    Returns a list of dicts, one per detection:
        {
            'cx':    float,          # centre x (pixels)
            'cy':    float,          # centre y (pixels)
            'w':     float,          # box width (pixels)
            'h':     float,          # box height (pixels)
            'angle': float,          # rotation angle in degrees (0–360)
            'conf':  float,          # detection confidence
            'pts':   np.ndarray,     # (4,1,2) int32 — rotated box corners
            'track_id': int or None
        }

    Falls back to empty list if OBB model is not loaded.
    """
    if _obb_model is None:
        print("WARNING: OBB model not loaded.")
        return []

    results = _obb_model.track(frame, persist=True,
                               conf=conf_thresh, verbose=False)[0]
    detections = []

    if results.obb is None:
        return detections

    track_ids = (results.obb.id.cpu().numpy().astype(int)
                 if results.obb.id is not None
                 else [None] * len(results.obb))

    for box, track_id in zip(results.obb, track_ids):
        cx, cy, w, h, angle_rad = box.xywhr[0].tolist()
        conf      = float(box.conf[0])
        angle_deg = math.degrees(angle_rad) % 360

        rect = ((cx, cy), (w, h), math.degrees(angle_rad))
        pts  = cv2.boxPoints(rect).astype(np.int32)

        detections.append({
            'cx':       cx,
            'cy':       cy,
            'w':        w,
            'h':        h,
            'angle':    angle_deg,
            'conf':     conf,
            'pts':      pts,
            'track_id': int(track_id) if track_id is not None else None,
        })

    return detections
