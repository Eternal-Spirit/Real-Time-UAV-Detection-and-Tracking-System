import cv2
import numpy as np
import math
from collections import defaultdict, deque
from config import (
    ZONES, ZONE_COLORS, ZONE_BORDER_COLORS,
    ZONE_FRAMES, GEOFENCE_ENABLED
)


_current_w = 640
_current_h = 360
_active_zones = {k: v.copy() for k, v in ZONES.items()}

def update_zone_dimensions(frame_w: int, frame_h: int) -> None:
    """
    Scales the nominal 640x360 geofence zones to match the current frame resolution.
    Also ensures the heatmap accumulator matches the frame size.
    """
    global _current_w, _current_h, _active_zones
    if frame_w == _current_w and frame_h == _current_h:
        return
    _current_w = frame_w
    _current_h = frame_h
    scale_x = frame_w / 640.0
    scale_y = frame_h / 360.0
    _active_zones = {}
    for level, pts in ZONES.items():
        scaled_pts = pts.copy().astype(np.float32)
        scaled_pts[:, 0] *= scale_x
        scaled_pts[:, 1] *= scale_y
        _active_zones[level] = scaled_pts.astype(np.int32)
    if 'heatmap' in globals() and (heatmap.width != frame_w or heatmap.height != frame_h):
        heatmap.resize(frame_w, frame_h)


def get_threat_level(cx: float, cy: float) -> str | None:
    """
    Returns the threat zone ('HIGH', 'MEDIUM', 'LOW') the drone
    centre point (cx, cy) is currently inside, or None if outside
    all zones.

    Checks from innermost (HIGH) outward so a drone in HIGH is
    not also counted as MEDIUM.

    Args:
        cx, cy : drone bounding-box centre in pixel coordinates

    Returns:
        'HIGH' | 'MEDIUM' | 'LOW' | None
    """
    if not GEOFENCE_ENABLED:
        return None

    pt = (int(cx), int(cy))

    for level in ['HIGH', 'MEDIUM', 'LOW']:   # innermost first
        result = cv2.pointPolygonTest(_active_zones[level], pt, measureDist=False)
        if result >= 0:          # 0 = on edge, 1 = inside
            return level

    return None                  # outside all zones


def draw_zones(frame: np.ndarray) -> np.ndarray:
    """
    Draws all three geofence zones onto the frame as semi-transparent
    filled polygons with solid borders and zone labels.

    Args:
        frame : BGR image (will NOT be modified in place — a copy is used)

    Returns:
        Annotated BGR image
    """
    if not GEOFENCE_ENABLED:
        return frame

    h, w = frame.shape[:2]
    update_zone_dimensions(w, h)
    overlay = frame.copy()

    # Draw fills (outermost first so inner zones paint on top)
    for level in ['LOW', 'MEDIUM', 'HIGH']:
        cv2.fillPoly(overlay, [_active_zones[level]], ZONE_COLORS[level])

    # Blend overlay with original — alpha=0.15 keeps it subtle
    frame = cv2.addWeighted(frame, 0.85, overlay, 0.15, 0)

    # Draw solid borders on top of the blend
    for level in ['LOW', 'MEDIUM', 'HIGH']:
        cv2.polylines(
            frame,
            [_active_zones[level]],
            isClosed=True,
            color=ZONE_BORDER_COLORS[level],
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    # Zone labels — positioned at top-left corner of each polygon bbox
    for level in ['LOW', 'MEDIUM', 'HIGH']:
        pts   = _active_zones[level]
        lx    = int(pts[:, 0].min()) + 6
        ly    = int(pts[:, 1].min()) + 18
        color = ZONE_BORDER_COLORS[level]
        cv2.putText(frame, f'ZONE {level}', (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return frame

# ── Trail storage ─────────────────────────────────────────────
# Stores last 30 centre-point positions per track ID.
# maxlen=30 automatically drops oldest point when full.
# This same dict is reused by Phase 7 (heading angle) — do NOT
# create a second one there.

trail_history = defaultdict(lambda: deque(maxlen=30))


def update_trail(track_id: int, cx: float, cy: float) -> None:
    """
    Appends the current drone centre to its trail history.
    Call once per frame per tracked drone, after getting cx/cy
    from the bounding box.

    Args:
        track_id : integer tracker ID from YOLOv8 .track()
        cx, cy   : bounding-box centre in pixel coordinates
    """
    trail_history[track_id].append((int(cx), int(cy)))


def draw_trail(frame: np.ndarray, track_id: int) -> np.ndarray:
    """
    Draws a fading trail of lines connecting the drone's past positions.
    Most recent segment is brightest; oldest is nearly invisible.
    Colour fades from warm gold (old) to electric neon cyan (recent).

    Args:
        frame    : BGR image to draw on
        track_id : tracker ID whose trail to draw

    Returns:
        Annotated frame
    """
    trail = list(trail_history[track_id])
    n = len(trail)

    if n < 2:
        return frame

    for i in range(1, n):
        progress = i / n
        # Color gradient: warm gold (old) -> vivid neon cyan (recent)
        b = int(255 * progress)
        g = int(200 + 55 * progress)
        r = int(60 * (1.0 - progress))

        thickness = 2 if i < n // 2 else 3
        cv2.line(frame, trail[i - 1], trail[i], (b, g, r),
                 thickness, cv2.LINE_AA)

        # Draw glowing waypoint dots along the trail
        if i % 4 == 0 or i == n - 1:
            radius = 2 if i < n - 1 else 4
            cv2.circle(frame, trail[i], radius, (b, g, r), -1, cv2.LINE_AA)

    return frame


def draw_prediction(frame: np.ndarray, track_id: int,
                    steps: int = 12) -> np.ndarray:
    """
    Extrapolates the drone's trajectory using recent motion velocity
    and renders a projected trajectory vector line with waypoint nodes
    and a forward directional arrow.

    Args:
        frame    : BGR image to draw on
        track_id : tracker ID to predict for
        steps    : how many frames ahead to predict (default 12)

    Returns:
        Annotated frame
    """
    trail = list(trail_history[track_id])

    if len(trail) < 2:
        return frame

    # Velocity vector from last 2 or 3 points
    if len(trail) >= 3:
        dx = (trail[-1][0] - trail[-3][0]) / 2.0
        dy = (trail[-1][1] - trail[-3][1]) / 2.0
    else:
        dx = float(trail[-1][0] - trail[-2][0])
        dy = float(trail[-1][1] - trail[-2][1])

    speed = math.hypot(dx, dy)

    # Skip prediction if drone is stationary (hovering)
    if speed < 0.6:
        return frame

    h, w = frame.shape[:2]
    curr_x, curr_y = trail[-1]
    prev_pt = (int(curr_x), int(curr_y))

    for s in range(1, steps + 1):
        px = int(curr_x + dx * s)
        py = int(curr_y + dy * s)

        # Don't draw outside frame bounds
        if not (0 <= px < w and 0 <= py < h):
            break

        curr_pt = (px, py)
        # Draw dashed projected trajectory line
        if s % 2 == 1:
            cv2.line(frame, prev_pt, curr_pt, (0, 165, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, curr_pt, 3, (0, 215, 255), -1, cv2.LINE_AA)
        prev_pt = curr_pt

    # Arrow head at end of predicted trajectory vector
    if prev_pt != (int(curr_x), int(curr_y)):
        cv2.arrowedLine(frame, (int(curr_x), int(curr_y)), prev_pt,
                        (0, 165, 255), 2, cv2.LINE_AA, tipLength=0.18)

    return frame


def draw_direction_arrow(frame: np.ndarray, track_id: int,
                         cx: float, cy: float,
                         length: int = 45) -> np.ndarray:
    """
    Draws a cyan arrowedLine from the drone centre pointing in
    the direction of current movement (computed from last 3 trail
    points for stability).

    Args:
        frame    : BGR image to draw on
        track_id : tracker ID
        cx, cy   : current drone centre position
        length   : arrow length in pixels (default 45)

    Returns:
        Annotated frame
    """
    trail = list(trail_history[track_id])

    # Need at least 3 points to compute a stable direction
    if len(trail) < 3:
        return frame

    # Average velocity over last 3 points for stability
    dx = trail[-1][0] - trail[-3][0]
    dy = trail[-1][1] - trail[-3][1]

    if abs(dx) < 1 and abs(dy) < 1:
        return frame    # hovering — no direction to draw

    angle_rad = math.atan2(dy, dx)
    ex = int(cx + length * math.cos(angle_rad))
    ey = int(cy + length * math.sin(angle_rad))

    cv2.arrowedLine(
        frame,
        (int(cx), int(cy)),
        (ex, ey),
        color     = (255, 255, 0),   # cyan
        thickness = 2,
        line_type = cv2.LINE_AA,
        tipLength = 0.30,
    )

    return frame


def clear_trail(track_id: int) -> None:
    """
    Clears trail history for a specific track.
    Call when a track is permanently lost (optional cleanup).
    """
    if track_id in trail_history:
        del trail_history[track_id]


class DroneHeatmap:
    """
    Accumulates drone position data across frames and renders it
    as a colour heatmap overlay on the video feed.

    How it works:
      - Every time a drone is detected, update(cx, cy) paints a
        Gaussian "splat" onto a float32 accumulator array.
      - get_overlay(frame) normalises the accumulator, converts it
        to a JET colourmap image, and blends it onto the frame.
      - Hot spots (red) = areas with the most drone activity.
      - Cold spots (blue) = areas with occasional drone presence.
      - reset() clears the accumulator back to zero.

    The accumulator is NEVER reset automatically — it grows across
    the entire session so you see cumulative activity. Use the
    Streamlit "Reset heatmap" button to clear it manually.
    """

    def __init__(self, height: int = 480, width: int = 640,
                 radius: int = 20):
        """
        Args:
            height, width : must match your camera frame size.
                            Default 480×640. Change to match your
                            actual resolution (e.g. 720, 1280).
            radius        : Gaussian splat radius in pixels.
                            Larger = smoother/more spread heatmap.
                            Smaller = more precise hotspot mapping.
        """
        self.height  = height
        self.width   = width
        self.radius  = radius
        self.acc     = np.zeros((height, width), dtype=np.float32)
        self._kernel_size = radius * 2 + 1     # must be odd

    def update(self, cx: float, cy: float,
               weight: float = 1.0) -> None:
        """
        Adds a Gaussian splat centred at (cx, cy) to the accumulator.
        Call once per detection per frame.

        Args:
            cx, cy : drone bounding-box centre in pixel coordinates
            weight : multiplier for this detection's contribution.
                     Pass a higher value (e.g. 2.0) for HIGH-zone
                     detections to make threat areas redder faster.
        """
        px = int(np.clip(cx, 0, self.width  - 1))
        py = int(np.clip(cy, 0, self.height - 1))

        # Paint a filled circle on a blank mask
        mask = np.zeros((self.height, self.width), dtype=np.float32)
        cv2.circle(mask, (px, py), self.radius, weight, -1)

        # Blur into a smooth Gaussian splat
        mask = cv2.GaussianBlur(
            mask,
            (self._kernel_size, self._kernel_size),
            sigmaX=0,
        )

        self.acc += mask

    def get_overlay(self, frame: np.ndarray,
                    alpha: float = 0.35) -> np.ndarray:
        """
        Blends the heatmap onto the frame.

        Args:
            frame : BGR image (not modified in place)
            alpha : heatmap opacity  0.0 = invisible, 1.0 = full
                    Default 0.35 — subtle but visible

        Returns:
            Blended BGR frame
        """
        if self.acc.max() == 0:
            return frame    # nothing accumulated yet

        # Normalise accumulator to 0–255 uint8
        norm = cv2.normalize(
            self.acc, None,
            alpha=0, beta=255,
            norm_type=cv2.NORM_MINMAX,
        ).astype(np.uint8)

        # Apply JET colourmap  (blue→green→yellow→red)
        heatmap_bgr = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

        # Resize if frame size differs from accumulator size
        if heatmap_bgr.shape[:2] != frame.shape[:2]:
            heatmap_bgr = cv2.resize(
                heatmap_bgr,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        return cv2.addWeighted(frame, 1 - alpha, heatmap_bgr, alpha, 0)

    def get_snapshot(self) -> np.ndarray:
        """
        Returns a standalone heatmap image (no video frame blend).
        Useful for the Streamlit sidebar heatmap panel.

        Returns:
            BGR heatmap image same size as accumulator
        """
        if self.acc.max() == 0:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        norm = cv2.normalize(
            self.acc, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    def reset(self) -> None:
        """Clears all accumulated heatmap data."""
        self.acc.fill(0)

    def resize(self, width: int, height: int) -> None:
        """Resizes the accumulator grid to match updated frame dimensions."""
        if self.width == width and self.height == height:
            return
        if self.acc.max() > 0:
            self.acc = cv2.resize(self.acc, (width, height), interpolation=cv2.INTER_LINEAR)
        else:
            self.acc = np.zeros((height, width), dtype=np.float32)
        self.width = width
        self.height = height

    @property
    def total_detections(self) -> int:
        """Rough count of total drone-frame hits accumulated."""
        return int(self.acc.sum() / max(self.radius, 1))


# ── Module-level singleton ────────────────────────────────────
# Import this instance everywhere so all modules share one heatmap.
# Usage:  from utils import heatmap
from config import FRAME_HEIGHT, FRAME_WIDTH
heatmap = DroneHeatmap(height=FRAME_HEIGHT, width=FRAME_WIDTH, radius=20)

import math

# ── Constants ─────────────────────────────────────────────────

CARDINAL_8 = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

# How many degrees either side of a straight line counts as
# "heading toward" a zone. 60° is a wide cone — tighten to 30°
# for stricter "approaching" logic.
APPROACH_CONE_DEG = 60.0

# Minimum pixel displacement per 3-frame window to consider
# the drone moving (not hovering). Raise if arrow flickers.
MIN_MOVEMENT_PX = 2


# ── Core heading calculation ──────────────────────────────────

def get_heading(track_id: int) -> dict | None:
    """
    Computes current heading angle from trail_history.

    Uses the displacement between trail[-3] and trail[-1]
    (a 2-step average) for stability — single-frame velocity
    is too noisy at typical drone speeds.

    NOTE: trail_history is defined in this same file (Phase 3).
    update_trail() must be called before get_heading() each frame.

    Args:
        track_id : tracker ID

    Returns:
        dict with keys:
            angle_deg  : float  0–360°  (0=East, 90=North, clockwise)
            angle_nav  : float  0–360°  navigation convention
                                        (0=North, 90=East, clockwise)
            cardinal   : str    one of CARDINAL_8
            dx, dy     : float  raw pixel velocity components
            speed_px   : float  pixels/frame speed estimate
        or None if not enough history or drone is hovering
    """
    trail = list(trail_history[track_id])

    if len(trail) < 2:
        return None

    if len(trail) >= 3:
        dx = (trail[-1][0] - trail[-3][0]) / 2.0
        dy = (trail[-1][1] - trail[-3][1]) / 2.0
    else:
        dx = float(trail[-1][0] - trail[-2][0])
        dy = float(trail[-1][1] - trail[-2][1])

    speed = math.hypot(dx, dy)

    if speed < 0.8:
        return None    # hovering — no meaningful heading

    # Screen-space angle (y-axis points DOWN in OpenCV)
    # atan2(-dy, dx) flips y so North = up on screen
    angle_screen = math.degrees(math.atan2(-dy, dx)) % 360

    # Navigation convention: 0 = North (up), 90 = East, clockwise
    # Convert from math convention (0=East, CCW) to nav (0=North, CW)
    angle_nav = (90.0 - angle_screen) % 360

    # 8-point cardinal — bucket by 45° starting at N
    cardinal_idx = int((angle_nav + 22.5) / 45) % 8
    cardinal     = CARDINAL_8[cardinal_idx]

    return {
        'angle_deg':  angle_screen,    # screen-space (used for drawing)
        'angle_nav':  angle_nav,       # navigation convention (used for display)
        'cardinal':   cardinal,
        'dx':         dx,
        'dy':         dy,
        'speed_px':   speed,           # per-frame speed estimate
    }


# ── Zone approach detection ───────────────────────────────────

def _zone_centroid(zone_name: str) -> tuple[float, float]:
    """Returns the pixel centroid of a zone polygon."""
    poly = _active_zones.get(zone_name, ZONES.get(zone_name))
    pts = poly.astype(np.float32)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def is_heading_toward_zone(track_id: int,
                            cx: float, cy: float,
                            zone_name: str) -> bool:
    """
    Returns True if the drone is heading toward the given zone
    within APPROACH_CONE_DEG degrees.
    """
    heading = get_heading(track_id)
    if heading is None:
        return False

    # Vector from drone → zone centroid
    zx, zy     = _zone_centroid(zone_name)
    to_zone_dx = zx - cx
    to_zone_dy = zy - cy

    # Heading vector
    hdx = heading['dx']
    hdy = heading['dy']

    # Angle between the two vectors (dot product method)
    dot     = hdx * to_zone_dx + hdy * to_zone_dy
    mag_h   = math.hypot(hdx, hdy)
    mag_z   = math.hypot(to_zone_dx, to_zone_dy)

    if mag_h < 1e-6 or mag_z < 1e-6:
        return False

    cos_angle = dot / (mag_h * mag_z)
    cos_angle = max(-1.0, min(1.0, cos_angle))   # clamp for acos safety
    angle_between = math.degrees(math.acos(cos_angle))

    return angle_between < APPROACH_CONE_DEG


def get_approach_warnings(track_id: int,
                           cx: float, cy: float,
                           current_zone: str | None) -> list[str]:
    """
    Returns a list of zone names the drone is currently heading
    toward but NOT yet inside. Used to generate early warnings.
    """
    warnings = []
    for zone_name in ['HIGH', 'MEDIUM', 'LOW']:
        if zone_name == current_zone:
            continue    # already inside — not a warning
        if is_heading_toward_zone(track_id, cx, cy, zone_name):
            warnings.append(zone_name)

    # Only return the innermost approaching zone to avoid noise
    priority = ['HIGH', 'MEDIUM', 'LOW']
    for p in priority:
        if p in warnings:
            return [p]
    return []


# ── Drawing ───────────────────────────────────────────────────

# Cardinal compass colours
CARDINAL_COLORS = {
    'N':  (255, 255,   0),   # cyan-yellow (north = toward top)
    'NE': (  0, 200, 255),
    'E':  (  0, 165, 255),   # orange
    'SE': (  0, 100, 255),
    'S':  (  0,   0, 255),   # red (toward bottom)
    'SW': (100,   0, 255),
    'W':  (200,   0, 255),
    'NW': (255,   0, 200),
}


def draw_heading_overlay(frame: np.ndarray,
                          track_id: int,
                          cx: float, cy: float,
                          heading: dict,
                          approach_warnings: list[str]) -> np.ndarray:
    """
    Draws the full heading overlay for one drone:
      1. Heading arrow (coloured by cardinal direction)
      2. Cardinal label next to arrowhead (e.g. "NE")
      3. Angle badge below drone ID (e.g. "047°")
      4. Approach warning flash (if heading toward inner zone)
    """
    if heading is None:
        return frame

    angle_screen = heading['angle_deg']
    cardinal     = heading['cardinal']
    speed        = heading['speed_px']
    arrow_color  = CARDINAL_COLORS.get(cardinal, (0, 255, 255))

    # ── 1. Heading arrow ──────────────────────────────────────
    # Length scales with speed — faster drone = longer arrow
    arrow_len = int(np.clip(30 + speed * 3, 30, 80))
    rad = math.radians(angle_screen)
    ex  = int(cx + arrow_len * math.cos(rad))
    ey  = int(cy - arrow_len * math.sin(rad))    # subtract: y-axis is flipped

    cv2.arrowedLine(
        frame,
        (int(cx), int(cy)),
        (ex, ey),
        color     = arrow_color,
        thickness = 2,
        line_type = cv2.LINE_AA,
        tipLength = 0.28,
    )

    # ── 2. Cardinal label at arrowhead ────────────────────────
    label_x = ex + (8 if ex > cx else -24)
    label_y = ey + (14 if ey > cy else -6)
    cv2.putText(frame, cardinal,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                arrow_color, 1, cv2.LINE_AA)

    # ── 3. Angle badge ────────────────────────────────────────
    angle_nav = heading['angle_nav']
    badge_txt = f"{angle_nav:.0f}deg {cardinal} | {speed:.0f}px/f"
    cv2.putText(frame, badge_txt,
                (int(cx) - 20, int(cy) + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                arrow_color, 1, cv2.LINE_AA)

    # ── 4. Approach warning ───────────────────────────────────
    if approach_warnings:
        warn_zone  = approach_warnings[0]
        warn_color = {'HIGH': (60,60,255), 'MEDIUM': (60,185,255),
                      'LOW':  (60,255,255)}.get(warn_zone, (0,255,0))
        warn_txt   = f">> {warn_zone}"
        cv2.putText(frame, warn_txt,
                    (int(cx) - 20, int(cy) + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    warn_color, 1, cv2.LINE_AA)

        # Flashing border: thin rectangle around bbox area
        flash_thickness = 2 if (int(cx) // 10) % 2 == 0 else 1
        pad = 6
        # We don't have x1/y1 here so draw a circle instead
        cv2.circle(frame, (int(cx), int(cy)),
                   int(math.hypot(30, 30)) + pad,
                   warn_color, flash_thickness, cv2.LINE_AA)

    return frame


def draw_mini_compass(frame: np.ndarray,
                       heading: dict,
                       pos: tuple[int, int] = (30, 30),
                       radius: int = 22) -> np.ndarray:
    """
    Draws a small compass rose in the corner of the frame showing
    the most recent drone's heading direction.
    """
    if heading is None:
        return frame

    cx_c, cy_c = pos
    angle_nav  = heading['angle_nav']
    cardinal   = heading['cardinal']

    # Background circle
    cv2.circle(frame, (cx_c, cy_c), radius + 2,
               (30, 30, 30), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx_c, cy_c), radius,
               (80, 80, 80), 1, cv2.LINE_AA)

    # Cardinal tick marks
    for i, label in enumerate(CARDINAL_8):
        tick_angle = math.radians(i * 45 - 90)  # 0=N at top
        tx = int(cx_c + (radius - 4) * math.cos(tick_angle))
        ty = int(cy_c + (radius - 4) * math.sin(tick_angle))
        color = (255, 255, 255) if label in ('N', 'S', 'E', 'W') \
                else (120, 120, 120)
        cv2.circle(frame, (tx, ty), 1, color, -1)

    # N label
    cv2.putText(frame, 'N',
                (cx_c - 4, cy_c - radius + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                (200, 200, 200), 1, cv2.LINE_AA)

    # Heading needle
    needle_rad = math.radians(angle_nav - 90)  # convert nav to screen angle
    nx = int(cx_c + (radius - 5) * math.cos(needle_rad))
    ny = int(cy_c + (radius - 5) * math.sin(needle_rad))
    arrow_color = CARDINAL_COLORS.get(cardinal, (0, 255, 255))
    cv2.arrowedLine(frame, (cx_c, cy_c), (nx, ny),
                    arrow_color, 2, cv2.LINE_AA, tipLength=0.4)

    return frame


# ── Phase 8: Drone Type Classification ────────────────────────

def classify_drone_type(x1, y1, x2, y2, frame_h=480, frame_w=640):
    """
    Classifies drone type from bounding box geometry.
    Returns: (type_label, description, threat_modifier)
    threat_modifier is 'HIGH', 'MEDIUM', or 'LOW' — can be used to
    escalate/de-escalate the zone-based threat level if needed.

    Replace with Phase 10 (EfficientNet-B0) for ~85% accuracy.
    Calling interface is identical — just swap the import.
    """
    bw = x2 - x1
    bh = y2 - y1
    area_ratio = (bw * bh) / (frame_w * frame_h)
    aspect = bw / max(bh, 1)

    if area_ratio > 0.08:
        return 'Large UAV',  'Commercial/Military', 'HIGH'
    elif aspect > 2.5:
        return 'Fixed-wing', 'Long-range UAV',      'HIGH'
    elif aspect > 1.5:
        return 'Hexacopter', 'Heavy-lift quad',      'MEDIUM'
    elif area_ratio < 0.005:
        return 'Nano quad',  'Consumer micro',       'LOW'
    else:
        return 'Quadcopter', 'Consumer UAV',         'MEDIUM'


# ── Phase 9: OBB True Rotation Angle ──────────────────────────

def bbox_to_obb_label(cls, cx, cy, bw, bh):
    """
    Converts a normalised axis-aligned bbox to OBB label format.
    Angle is estimated from aspect ratio (landscape=0°, portrait=90°).
    """
    angle = 0.0 if bw >= bh else 90.0
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {angle:.1f}"

def is_heading_toward_zone_obb(angle_deg, cx, cy, zone_name, threshold_deg=60):
    """
    Returns True if the drone's heading points within threshold_deg of
    the centroid of the named zone. Works with Phase 9 angles.
    """
    zone_poly = _active_zones.get(zone_name, ZONES.get(zone_name))
    if zone_poly is None:
        return False
    # Compute zone centroid
    zx = int(zone_poly[:, 0].mean())
    zy = int(zone_poly[:, 1].mean())
    # Angle from drone to zone centroid
    to_zone = math.degrees(math.atan2(-(zy - cy), zx - cx)) % 360
    diff = abs(angle_deg - to_zone) % 360
    if diff > 180:
        diff = 360 - diff
    return diff < threshold_deg
