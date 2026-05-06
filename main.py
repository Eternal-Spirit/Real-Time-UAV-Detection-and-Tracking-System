"""
main.py — Phase 9 changes:
  - Import USE_OBB, MODEL_PATH_OBB
  - Import load_obb_model, track_with_angle
  - Unify standard YOLO and OBB detections
"""

import cv2
import time
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

from config import (
    MODEL_PATH, CONFIDENCE, IMG_SIZE, ZONE_FRAMES,
    HEATMAP_ENABLED, HEATMAP_ALPHA,
    HEATMAP_UPDATE_EVERY, HEATMAP_RENDER_EVERY,
    SNAPSHOT_ZONES, HEADING_ENABLED,
    SHOW_COMPASS, COMPASS_POSITION, COMPASS_RADIUS,
    DRONE_TYPE_COLORS,                           # ← Phase 8
    MODEL_PATH_OBB, USE_OBB,                     # ← Phase 9
    KNOWN_DRONE_WIDTH_M, FOCAL_LENGTH_PX,        # Distance estimation
)
from utils import (
    get_threat_level, draw_zones,
    update_trail, draw_trail, draw_prediction, draw_direction_arrow,
    heatmap,
    get_heading, get_approach_warnings,
    draw_heading_overlay, draw_mini_compass,
    classify_drone_type,                         # ← Phase 8
    is_heading_toward_zone_obb, CARDINAL_8,      # ← Phase 9
)
from logger import trigger_alert
from stats  import stats
from preprocessing import preprocess
from inference import load_obb_model, track_with_angle

model              = YOLO(MODEL_PATH)
if USE_OBB:
    load_obb_model(MODEL_PATH_OBB)

zone_frame_counter = defaultdict(lambda: defaultdict(int))
latest_zone        = {}
_frame_count       = 0
_last_heading      = None    # track last heading for compass


def process_frame(frame: np.ndarray,
                  show_trail:      bool = True,
                  show_prediction: bool = True,
                  show_arrow:      bool = True,
                  show_heatmap:    bool = True,
                  show_heading:    bool = True) -> tuple[np.ndarray, list]:
    """
    Full pipeline for one frame — Phases 1–9.
    """
    global _frame_count, _last_heading
    _frame_count += 1
    _t0 = time.perf_counter()

    # Resize any high-res video down to 360p to absolutely maximize FPS.
    # The heatmap is disabled and this will run extremely fast.
    frame = cv2.resize(frame, (640, 360))
    
    frame = preprocess(frame)
    clean_frame = frame.copy()  # Save clean frame for YOLO

    if show_heatmap:
        if _frame_count % HEATMAP_RENDER_EVERY == 0:
            frame = heatmap.get_overlay(frame, alpha=HEATMAP_ALPHA)

    frame = draw_zones(frame)

    # ── Unified Detection Logic ────────────────────────────────────
    unified_boxes = []

    if USE_OBB:
        obb_results = track_with_angle(clean_frame, conf_thresh=CONFIDENCE)
        for obb in obb_results:
            track_id = obb['track_id'] if obb['track_id'] is not None else -1
            
            pts = obb['pts']
            x1 = int(pts[:, 0].min())
            x2 = int(pts[:, 0].max())
            y1 = int(pts[:, 1].min())
            y2 = int(pts[:, 1].max())
            
            unified_boxes.append({
                'track_id': track_id,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'cx': obb['cx'], 'cy': obb['cy'],
                'conf': obb['conf'],
                'angle_deg': obb['angle'],
                'pts': pts
            })
    else:
        results = model.track(
            clean_frame,
            persist = True,
            conf    = CONFIDENCE,
            imgsz   = IMG_SIZE,
            verbose = False,
        )
        if results[0].boxes is not None:
            for box in results[0].boxes:
                track_id = int(box.id[0]) if box.id is not None else -1
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                unified_boxes.append({
                    'track_id': track_id,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'cx': cx, 'cy': cy,
                    'conf': float(box.conf[0]),
                    'angle_deg': None,
                    'pts': None
                })

    detections = []

    for box_data in unified_boxes:
        track_id  = box_data['track_id']
        x1, y1    = box_data['x1'], box_data['y1']
        x2, y2    = box_data['x2'], box_data['y2']
        cx, cy    = box_data['cx'], box_data['cy']
        conf      = box_data['conf']
        angle_deg = box_data['angle_deg']
        pts       = box_data['pts']
        
        # Calculate distance
        bbox_width = max(1.0, float(x2 - x1))
        distance_m = (KNOWN_DRONE_WIDTH_M * FOCAL_LENGTH_PX) / bbox_width

        # Phase 4 — heatmap update
        if show_heatmap and _frame_count % HEATMAP_UPDATE_EVERY == 0:
            zone_now = get_threat_level(cx, cy)
            weight   = {'HIGH': 2.0, 'MEDIUM': 1.5, 'LOW': 1.0}.get(zone_now, 1.0)
            heatmap.update(cx, cy, weight=weight)

        # Phase 3 — trail (MUST happen before get_heading)
        update_trail(track_id, cx, cy)

        # Phase 7 — heading (Trajectory based, ignores synthetic OBB angle)
        heading = None
        approach_warnings = []
        
        if HEADING_ENABLED:
            heading = get_heading(track_id)
            if heading is not None:
                approach_warnings = get_approach_warnings(
                    track_id, cx, cy, get_threat_level(cx, cy)
                )
                
        if heading is not None:
            _last_heading = heading
        zone        = get_threat_level(cx, cy)
        alert_fired = False
        latest_zone[track_id] = zone

        if zone is not None:
            zone_frame_counter[track_id][zone] += 1
            for other in ['HIGH', 'MEDIUM', 'LOW']:
                if other != zone:
                    zone_frame_counter[track_id][other] = 0
            if zone_frame_counter[track_id][zone] >= ZONE_FRAMES[zone]:
                alert_fired = True
        else:
            for z in ['HIGH', 'MEDIUM', 'LOW']:
                zone_frame_counter[track_id][z] = 0

        # Phase 8 — drone type classification
        dtype, desc, threat_mod = classify_drone_type(
            x1, y1, x2, y2,
            frame_h=frame.shape[0],
            frame_w=frame.shape[1]
        )
        type_color = DRONE_TYPE_COLORS.get(dtype, (200, 200, 200))

        # Phase 5 — alert
        alert_entry = None
        if alert_fired and zone in SNAPSHOT_ZONES:
            alert_entry = trigger_alert(
                frame=frame, track_id=track_id,
                zone=zone, conf=conf, cx=cx, cy=cy,
                drone_type=dtype, type_threat=threat_mod,
            )

        # Phase 3 — draw trail
        if show_trail:
            frame = draw_trail(frame, track_id)
        if show_prediction:
            frame = draw_prediction(frame, track_id, steps=10)

        # Bounding box (Phase 9 rotated vs standard)
        box_color = _zone_box_color(zone)
        if pts is not None:
            cv2.polylines(frame, [pts], True, box_color, 2)
        else:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
            
        label = f'ID:{track_id}'
        if zone:
            label += f' | {zone}'
        if alert_fired:
            label += ' !'
        label += f' | {distance_m:.1f}m'
        if heading is not None:
            label += f" | {heading['speed_px']:.0f}px/s"
            
        cv2.putText(frame, label,
                    (int(x1), int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    box_color, 1, cv2.LINE_AA)

        # Phase 8 — draw type badge below bounding box
        cv2.putText(frame, f"TYPE: {dtype}", (int(x1), int(y2) + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, type_color, 1, cv2.LINE_AA)
        cv2.putText(frame, desc, (int(x1), int(y2) + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, type_color, 1, cv2.LINE_AA)

        # Phase 7/9 — heading overlay (replaces Phase 3 arrow)
        if show_heading and heading is not None:
            frame = draw_heading_overlay(
                frame, track_id, cx, cy,
                heading, approach_warnings,
            )
        elif show_arrow:
            # Fallback: Phase 3 simple arrow if heading disabled
            frame = draw_direction_arrow(frame, track_id, cx, cy)

        detections.append({
            'track_id':         track_id,
            'cx':               cx,
            'cy':               cy,
            'conf':             conf,
            'zone':             zone,
            'alert_fired':      alert_fired,
            'alert_entry':      alert_entry,
            'heading':          heading,
            'approach_warnings': approach_warnings,
            'drone_type':       dtype,
            'type_desc':        desc,
            'type_threat':      threat_mod,
            'distance_m':       distance_m,
        })

    # Phase 7/9 — mini compass
    if show_heading and SHOW_COMPASS and _last_heading is not None:
        frame = draw_mini_compass(
            frame, _last_heading,
            pos    = COMPASS_POSITION,
            radius = COMPASS_RADIUS,
        )

    # Phase 6 — stats update
    elapsed = time.perf_counter() - _t0
    stats.update(fps=1.0 / max(elapsed, 1e-6), detections=detections)

    return frame, detections


def _zone_box_color(zone):
    return {
        'HIGH':   (60,  60, 255),
        'MEDIUM': (60, 185, 255),
        'LOW':    (60, 255, 255),
    }.get(zone, (0, 255, 0))


# ── Standalone run ────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',        default='0')
    parser.add_argument('--no-trail',      action='store_true')
    parser.add_argument('--no-prediction', action='store_true')
    parser.add_argument('--no-arrow',      action='store_true')
    parser.add_argument('--no-heatmap',    action='store_true')
    parser.add_argument('--no-heading',    action='store_true')
    args = parser.parse_args()

    src = int(args.source) if args.source == '0' else args.source
    cap = cv2.VideoCapture(src)
    print('[INFO] Press Q to quit | R to reset heatmap')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated, dets = process_frame(
            frame,
            show_trail      = not args.no_trail,
            show_prediction = not args.no_prediction,
            show_arrow      = not args.no_arrow,
            show_heatmap    = not args.no_heatmap,
            show_heading    = not args.no_heading,
        )

        cv2.imshow('Drone Detection — Phase 9', annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            heatmap.reset()

    cap.release()
    cv2.destroyAllWindows()
