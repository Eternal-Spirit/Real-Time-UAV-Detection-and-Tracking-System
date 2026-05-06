"""
logger.py — Alert logger, snapshot writer, and sound system.

Responsibilities:
  - Write annotated snapshot JPEGs to outputs/snapshots/
  - Play a beep sound on alert (Windows winsound, fallback silent)
  - Maintain an in-memory alert log (list of dicts)
  - Enforce per-drone cooldown so alerts don't spam
  - Expose get_log() for Streamlit and CSV export
"""

import cv2
import os
import time
import threading
from datetime import datetime
from pathlib import Path

from config import (
    SNAPSHOT_DIR,
    ALERT_COOLDOWN,
    ALERT_SOUND_ENABLED,
    ALERT_BEEP_HIGH,
    ALERT_BEEP_MEDIUM,
    ALERT_BEEP_LOW,
    BEEP_DURATION_MS,
)


# ── Directory setup ───────────────────────────────────────────
Path(SNAPSHOT_DIR).mkdir(parents=True, exist_ok=True)


# ── State ─────────────────────────────────────────────────────
_last_alert_time: dict[int, float] = {}   # track_id → timestamp
_alert_log:       list[dict]       = []   # full session log


# ── Sound (non-blocking) ──────────────────────────────────────

def _play_beep(frequency: int, duration_ms: int) -> None:
    """
    Plays a beep in a background thread so it never blocks the
    video pipeline. Uses winsound on Windows; silently skips on
    Linux/macOS (add pygame.mixer there if needed).
    """
    def _beep():
        try:
            import winsound
            winsound.Beep(frequency, duration_ms)
        except Exception:
            pass    # non-Windows or winsound not available — silent

    threading.Thread(target=_beep, daemon=True).start()


def play_alert_sound(zone: str) -> None:
    """
    Plays a zone-appropriate beep tone:
      HIGH   → high-pitched urgent beep
      MEDIUM → mid-pitched warning beep
      LOW    → low-pitched notice beep

    Args:
        zone : 'HIGH' | 'MEDIUM' | 'LOW'
    """
    if not ALERT_SOUND_ENABLED:
        return

    freq = {
        'HIGH':   ALERT_BEEP_HIGH,
        'MEDIUM': ALERT_BEEP_MEDIUM,
        'LOW':    ALERT_BEEP_LOW,
    }.get(zone, ALERT_BEEP_MEDIUM)

    _play_beep(freq, BEEP_DURATION_MS)


# ── Snapshot ──────────────────────────────────────────────────

def save_snapshot(frame, track_id: int, zone: str,
                  conf: float, drone_type: str = None) -> str | None:
    """
    Saves an annotated snapshot of the current frame to disk.

    Filename format:
        outputs/snapshots/20240503_143022_id7_HIGH_Hexacopter.jpg

    A red banner with alert info is stamped onto the image so
    saved snapshots are self-documenting without needing a log.

    Args:
        frame      : BGR frame at the moment of alert
        track_id   : tracker ID of the alerting drone
        zone       : zone string ('HIGH' | 'MEDIUM' | 'LOW')
        conf       : detection confidence (0.0–1.0)
        drone_type : string drone type label (optional)

    Returns:
        Saved file path as string, or None if write failed
    """
    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    dtype    = (drone_type or 'unknown').replace(' ', '_')
    filename = f"{ts}_id{track_id}_{zone}_{dtype}.jpg"
    filepath = os.path.join(SNAPSHOT_DIR, filename)

    # Copy frame immediately so main thread can continue without corrupting data
    frame_copy = frame.copy()

    def _save_task():
        # Stamp an alert banner onto the frame
        stamped = _stamp_alert_banner(frame_copy, track_id, zone, conf, ts)
        try:
            cv2.imwrite(filepath, stamped)
        except Exception as e:
            print(f'[WARN] Snapshot save failed: {e}')

    # Dispatch to background thread to prevent FPS lag spikes on disk I/O
    threading.Thread(target=_save_task, daemon=True).start()
    return filepath


def _stamp_alert_banner(frame, track_id: int, zone: str,
                        conf: float, ts: str):
    """
    Draws a translucent coloured banner at the top of the frame
    with alert metadata. Makes snapshots self-documenting.
    """
    banner_color = {
        'HIGH':   (0,   0,  200),
        'MEDIUM': (0,  130, 200),
        'LOW':    (0,  200, 200),
    }.get(zone, (0, 0, 200))

    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), banner_color, -1)
    frame = cv2.addWeighted(frame, 0.35, overlay, 0.65, 0)

    text = (f"ALERT  Zone:{zone}  "
            f"ID:{track_id}  "
            f"Conf:{conf:.0%}  "
            f"Time:{ts}")
    cv2.putText(frame, text, (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return frame


# ── Main trigger function ─────────────────────────────────────

def trigger_alert(frame,
                  track_id:  int,
                  zone:      str,
                  conf:      float = 0.0,
                  cx:        float = 0.0,
                  cy:        float = 0.0,
                  drone_type: str = None,
                  type_threat: str = None) -> dict | None:
    """
    Central alert handler. Call this when zone_frame_counter
    threshold is reached (already computed in main.py Phase 2).

    Enforces per-drone cooldown — same drone won't spam alerts.
    Saves snapshot, plays sound, logs the event.

    Args:
        frame       : current annotated BGR frame
        track_id    : drone tracker ID
        zone        : 'HIGH' | 'MEDIUM' | 'LOW'
        conf        : detection confidence
        cx, cy      : drone centre position (logged for analysis)
        drone_type  : drone type label
        type_threat : threat modifier from type

    Returns:
        Alert dict if alert fired, None if still in cooldown
    """
    now = time.time()
    last = _last_alert_time.get(track_id, 0.0)

    if now - last < ALERT_COOLDOWN:
        return None     # still in cooldown — do nothing

    # Update cooldown timer
    _last_alert_time[track_id] = now

    # Save snapshot
    snapshot_path = save_snapshot(frame, track_id, zone, conf, drone_type)

    # Play sound (non-blocking)
    play_alert_sound(zone)

    # Build log entry
    entry = {
        'timestamp':     datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'track_id':      track_id,
        'zone':          zone,
        'drone_type':    drone_type or 'Unknown',
        'type_threat':   type_threat or 'None',
        'confidence':    f'{conf:.0%}',
        'position':      f'({int(cx)}, {int(cy)})',
        'snapshot':      snapshot_path or 'save failed',
    }
    _alert_log.append(entry)

    print(f'[ALERT] Zone:{zone}  ID:{track_id}  '
          f'Conf:{conf:.0%}  Snapshot:{snapshot_path}')

    return entry


# ── Log access ────────────────────────────────────────────────

def get_log() -> list[dict]:
    """Returns full alert log (newest first) for Streamlit/CSV."""
    return list(reversed(_alert_log))


def get_log_count() -> int:
    return len(_alert_log)


def clear_log() -> None:
    """Clears the in-memory alert log. Does NOT delete snapshots."""
    _alert_log.clear()
    _last_alert_time.clear()


def get_recent_snapshots(n: int = 5) -> list[str]:
    """
    Returns file paths of the N most recent saved snapshots
    that actually exist on disk.

    Args:
        n : number of recent snapshots to return

    Returns:
        List of valid file paths, most recent first
    """
    recent = [
        e['snapshot'] for e in reversed(_alert_log)
        if e.get('snapshot') and os.path.exists(e['snapshot'])
    ]
    return recent[:n]
