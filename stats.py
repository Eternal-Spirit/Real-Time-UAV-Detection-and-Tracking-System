"""
stats.py — Runtime statistics collector + graph renderer.

Tracks per-session:
  - FPS history
  - GPU VRAM usage history
  - Confidence score distribution
  - Zone hit counts over time
  - Alert frequency timeline
  - Detection count per frame

All data lives in memory. Graphs are rendered with matplotlib
and returned as Streamlit-compatible images.
"""

import time
import subprocess
import threading
from collections import defaultdict, deque
from datetime import datetime

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')                # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec


# ── Data stores ───────────────────────────────────────────────

class SessionStats:
    """
    Collects and stores all runtime metrics for the current session.
    One global instance is shared across app.py and main.py.
    """

    def __init__(self, max_history: int = 300):
        """
        Args:
            max_history : number of frames of rolling history to keep.
                          300 frames ≈ 10 seconds at 30fps.
        """
        self.max_history = max_history

        # Rolling time-series (deque auto-drops oldest)
        self.fps_history     = deque(maxlen=max_history)
        self.gpu_history     = deque(maxlen=max_history)   # VRAM used MB
        self.det_history     = deque(maxlen=max_history)   # detections/frame
        self.conf_history    = deque(maxlen=max_history)   # individual confs
        self.timestamps      = deque(maxlen=max_history)   # datetime strings

        # Cumulative counters
        self.zone_hits       = defaultdict(int)    # zone → total frame-hits
        self.alert_timeline  = []                  # list of (time_str, zone)
        self.total_frames    = 0
        self.total_dets      = 0
        self.session_start   = datetime.now()

        # GPU polling (background thread)
        self._gpu_thread_running = False
        self._latest_gpu_mb      = 0
        self._latest_gpu_total   = 0

    # ── Update methods ────────────────────────────────────────

    def update(self, fps: float, detections: list) -> None:
        """
        Call once per frame with current FPS and detection list.

        Args:
            fps        : frames per second this frame
            detections : list of detection dicts from process_frame()
        """
        now = datetime.now().strftime('%H:%M:%S')

        self.fps_history.append(fps)
        self.gpu_history.append(self._latest_gpu_mb)
        self.det_history.append(len(detections))
        self.timestamps.append(now)
        self.total_frames += 1
        self.total_dets   += len(detections)

        for d in detections:
            self.conf_history.append(d['conf'])
            if d['zone']:
                self.zone_hits[d['zone']] += 1
            if d['alert_fired']:
                self.alert_timeline.append((now, d['zone']))

        if not hasattr(self, '_heading_buffer'):
            self._heading_buffer = []
        self._heading_buffer.extend(detections)
        if len(self._heading_buffer) > 10000:
            self._heading_buffer = self._heading_buffer[-5000:]

    def start_gpu_polling(self, interval: float = 2.0) -> None:
        """
        Starts a background thread that polls nvidia-smi every
        `interval` seconds. Non-blocking — never affects FPS.
        """
        if self._gpu_thread_running:
            return
        self._gpu_thread_running = True

        def _poll():
            while self._gpu_thread_running:
                try:
                    out = subprocess.check_output(
                        ['nvidia-smi',
                         '--query-gpu=memory.used,memory.total',
                         '--format=csv,noheader,nounits'],
                        text=True, timeout=2,
                    )
                    used, total = out.strip().split(',')
                    self._latest_gpu_mb    = int(used.strip())
                    self._latest_gpu_total = int(total.strip())
                except Exception:
                    pass
                time.sleep(interval)

        threading.Thread(target=_poll, daemon=True).start()

    def stop_gpu_polling(self) -> None:
        self._gpu_thread_running = False

    # ── Summary properties ────────────────────────────────────

    @property
    def avg_fps(self) -> float:
        return float(np.mean(self.fps_history)) if self.fps_history else 0.0

    @property
    def current_fps(self) -> float:
        return self.fps_history[-1] if self.fps_history else 0.0

    @property
    def gpu_used_mb(self) -> int:
        return self._latest_gpu_mb

    @property
    def gpu_total_mb(self) -> int:
        return self._latest_gpu_total

    @property
    def gpu_pct(self) -> float:
        if self._latest_gpu_total == 0:
            return 0.0
        return self._latest_gpu_mb / self._latest_gpu_total * 100

    @property
    def session_duration(self) -> str:
        delta = datetime.now() - self.session_start
        m, s  = divmod(int(delta.total_seconds()), 60)
        h, m  = divmod(m, 60)
        return f'{h:02d}:{m:02d}:{s:02d}'

    def reset(self) -> None:
        self.fps_history.clear()
        self.gpu_history.clear()
        self.det_history.clear()
        self.conf_history.clear()
        self.timestamps.clear()
        self.zone_hits.clear()
        self.alert_timeline.clear()
        self.total_frames  = 0
        self.total_dets    = 0
        self.session_start = datetime.now()


# ── Graph renderers ───────────────────────────────────────────

DARK_BG   = '#0E1117'
PANEL_BG  = '#1A1D27'
TEXT_CLR  = '#FAFAFA'
GRID_CLR  = '#2A2D3A'
CYAN      = '#00D4FF'
ORANGE    = '#FF8C00'
RED       = '#FF4444'
GREEN     = '#00CC66'
YELLOW    = '#FFD700'

ZONE_CLR  = {'HIGH': RED, 'MEDIUM': ORANGE, 'LOW': YELLOW}


def _base_fig(figsize=(10, 3.5)):
    """Creates a dark-themed matplotlib figure."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_CLR, labelsize=9)
    ax.spines[:].set_color(GRID_CLR)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.title.set_color(TEXT_CLR)
    ax.grid(True, color=GRID_CLR, linewidth=0.5, alpha=0.6)
    return fig, ax


def fig_to_numpy(fig) -> np.ndarray:
    """Converts a matplotlib figure to an RGB numpy array for Streamlit."""
    fig.canvas.draw()
    # Matplotlib 3.8+ compatibility
    rgba = np.asarray(fig.canvas.buffer_rgba())
    # Extract RGB channels from RGBA
    img = rgba[..., :3].copy()
    plt.close(fig)
    return img


def plot_fps(stats: SessionStats) -> np.ndarray:
    """Rolling FPS line chart."""
    fig, ax = _base_fig()
    data = list(stats.fps_history)
    if not data:
        plt.close(fig)
        return np.zeros((200, 600, 3), dtype=np.uint8)

    ax.plot(data, color=CYAN, linewidth=1.5, alpha=0.9)
    ax.axhline(y=30, color=GREEN,  linewidth=1, linestyle='--', alpha=0.6, label='30 fps target')
    ax.axhline(y=stats.avg_fps, color=ORANGE, linewidth=1,
               linestyle=':', alpha=0.8, label=f'avg {stats.avg_fps:.1f} fps')
    ax.fill_between(range(len(data)), data, alpha=0.15, color=CYAN)
    ax.set_title('FPS — Rolling History', fontsize=11)
    ax.set_ylabel('Frames / sec')
    ax.set_xlabel('Frame')
    ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_CLR,
              framealpha=0.8)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    fig.tight_layout()
    return fig_to_numpy(fig)


def plot_gpu(stats: SessionStats) -> np.ndarray:
    """Rolling GPU VRAM usage area chart."""
    fig, ax = _base_fig()
    data  = list(stats.gpu_history)
    total = stats.gpu_total_mb or 8192    # default assume 8 GB

    if not data or max(data) == 0:
        ax.text(0.5, 0.5, 'nvidia-smi not available',
                transform=ax.transAxes, ha='center', va='center',
                color=TEXT_CLR, fontsize=10)
        ax.set_title('GPU VRAM Usage', fontsize=11)
        fig.tight_layout()
        return fig_to_numpy(fig)

    ax.fill_between(range(len(data)), data, alpha=0.4, color=ORANGE)
    ax.plot(data, color=ORANGE, linewidth=1.5)
    ax.axhline(y=total * 0.85, color=RED, linewidth=1,
               linestyle='--', alpha=0.7, label='85% limit')
    ax.set_title(f'GPU VRAM  ({stats.gpu_used_mb} / {total} MB  '
                 f'{stats.gpu_pct:.0f}%)', fontsize=11)
    ax.set_ylabel('VRAM used (MB)')
    ax.set_xlabel('Frame')
    ax.set_ylim(0, total * 1.05)
    ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_CLR)
    fig.tight_layout()
    return fig_to_numpy(fig)


def plot_detections(stats: SessionStats) -> np.ndarray:
    """Detection count per frame bar chart."""
    fig, ax = _base_fig()
    data = list(stats.det_history)

    if not data:
        plt.close(fig)
        return np.zeros((200, 600, 3), dtype=np.uint8)

    colors = [GREEN if d == 0 else ORANGE if d == 1 else RED for d in data]
    ax.bar(range(len(data)), data, color=colors, width=1.0, alpha=0.8)
    ax.set_title('Detections per Frame', fontsize=11)
    ax.set_ylabel('Count')
    ax.set_xlabel('Frame')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    return fig_to_numpy(fig)


def plot_confidence(stats: SessionStats) -> np.ndarray:
    """Confidence score distribution histogram."""
    fig, ax = _base_fig()
    data = list(stats.conf_history)

    if not data:
        ax.text(0.5, 0.5, 'No detections yet',
                transform=ax.transAxes, ha='center', va='center',
                color=TEXT_CLR, fontsize=10)
        ax.set_title('Confidence Distribution', fontsize=11)
        fig.tight_layout()
        return fig_to_numpy(fig)

    n, bins, patches = ax.hist(data, bins=20, range=(0, 1),
                                color=CYAN, alpha=0.75, edgecolor=DARK_BG)

    # Colour bins by confidence level
    for patch, left in zip(patches, bins[:-1]):
        if left >= 0.7:
            patch.set_facecolor(GREEN)
        elif left >= 0.5:
            patch.set_facecolor(CYAN)
        elif left >= 0.3:
            patch.set_facecolor(ORANGE)
        else:
            patch.set_facecolor(RED)

    mean_conf = float(np.mean(data))
    ax.axvline(x=mean_conf, color=YELLOW, linewidth=1.5,
               linestyle='--', label=f'mean {mean_conf:.2f}')
    ax.set_title('Detection Confidence Distribution', fontsize=11)
    ax.set_xlabel('Confidence score')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_CLR)
    fig.tight_layout()
    return fig_to_numpy(fig)


def plot_zone_hits(stats: SessionStats) -> np.ndarray:
    """Zone hit frequency bar chart."""
    fig, ax = _base_fig(figsize=(6, 3.5))

    zones  = ['HIGH', 'MEDIUM', 'LOW', 'Outside']
    hits   = [
        stats.zone_hits.get('HIGH',   0),
        stats.zone_hits.get('MEDIUM', 0),
        stats.zone_hits.get('LOW',    0),
        max(0, stats.total_dets
            - sum(stats.zone_hits.values())),
    ]
    colors = [RED, ORANGE, YELLOW, GREEN]

    bars = ax.bar(zones, hits, color=colors, alpha=0.85,
                  edgecolor=DARK_BG, linewidth=0.5)

    # Value labels on top of bars
    for bar, val in zip(bars, hits):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5, str(val),
                    ha='center', va='bottom',
                    color=TEXT_CLR, fontsize=9)

    ax.set_title('Zone Hit Frequency', fontsize=11)
    ax.set_ylabel('Frame-hit count')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    return fig_to_numpy(fig)


def plot_alert_timeline(stats: SessionStats) -> np.ndarray:
    """Alert events plotted on a time axis as coloured markers."""
    fig, ax = _base_fig(figsize=(10, 2.8))

    if not stats.alert_timeline:
        ax.text(0.5, 0.5, 'No alerts fired this session',
                transform=ax.transAxes, ha='center', va='center',
                color=TEXT_CLR, fontsize=10)
        ax.set_title('Alert Timeline', fontsize=11)
        fig.tight_layout()
        return fig_to_numpy(fig)

    zone_y = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
    labels_drawn = set()

    for i, (ts, zone) in enumerate(stats.alert_timeline):
        y     = zone_y.get(zone, 1)
        color = ZONE_CLR.get(zone, CYAN)
        label = zone if zone not in labels_drawn else ''
        ax.scatter(i, y, color=color, s=80, zorder=5,
                   label=label, alpha=0.9)
        labels_drawn.add(zone)

    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['LOW', 'MEDIUM', 'HIGH'],
                       color=TEXT_CLR, fontsize=9)
    ax.set_xlabel('Alert index')
    ax.set_title(f'Alert Timeline  ({len(stats.alert_timeline)} total)',
                 fontsize=11)
    ax.set_ylim(0, 4)
    ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_CLR,
              loc='upper right')
    fig.tight_layout()
    return fig_to_numpy(fig)


def plot_model_training(results_csv_path: str) -> np.ndarray:
    """
    Reads YOLOv8 results.csv from your training run and plots:
      - box_loss (train + val)
      - mAP50
      - mAP50-95
      - Precision + Recall

    Args:
        results_csv_path : path to runs/train/yolov8s_clahe_v1/results.csv

    Returns:
        RGB numpy image of the 2×2 subplot grid
    """
    import pandas as pd
    import os

    if not os.path.exists(results_csv_path):
        fig, ax = _base_fig()
        ax.text(0.5, 0.5,
                f'Training results not found:\n{results_csv_path}\n\n'
                'Run training first.',
                transform=ax.transAxes, ha='center', va='center',
                color=TEXT_CLR, fontsize=9, linespacing=1.8)
        ax.set_title('Model Training Curves', fontsize=11)
        fig.tight_layout()
        return fig_to_numpy(fig)

    df = pd.read_csv(results_csv_path)
    df.columns = df.columns.str.strip()     # remove whitespace from headers

    fig = plt.figure(figsize=(12, 7))
    fig.patch.set_facecolor(DARK_BG)
    gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    def styled_ax(pos):
        ax = fig.add_subplot(pos)
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_CLR, labelsize=8)
        ax.spines[:].set_color(GRID_CLR)
        ax.xaxis.label.set_color(TEXT_CLR)
        ax.yaxis.label.set_color(TEXT_CLR)
        ax.title.set_color(TEXT_CLR)
        ax.grid(True, color=GRID_CLR, linewidth=0.5, alpha=0.6)
        return ax

    epochs = df['epoch'] if 'epoch' in df.columns else range(len(df))

    # Plot 1 — Box loss
    ax1 = styled_ax(gs[0, 0])
    if 'train/box_loss' in df.columns:
        ax1.plot(epochs, df['train/box_loss'], color=CYAN,
                 label='train', linewidth=1.5)
    if 'val/box_loss' in df.columns:
        ax1.plot(epochs, df['val/box_loss'], color=ORANGE,
                 label='val', linewidth=1.5, linestyle='--')
    ax1.set_title('Box Loss', fontsize=10)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=TEXT_CLR)

    # Plot 2 — mAP
    ax2 = styled_ax(gs[0, 1])
    if 'metrics/mAP50(B)' in df.columns:
        ax2.plot(epochs, df['metrics/mAP50(B)'], color=GREEN,
                 label='mAP@0.50', linewidth=1.5)
    if 'metrics/mAP50-95(B)' in df.columns:
        ax2.plot(epochs, df['metrics/mAP50-95(B)'], color=YELLOW,
                 label='mAP@0.50:0.95', linewidth=1.5, linestyle='--')
    ax2.set_title('mAP Scores', fontsize=10)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mAP')
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=TEXT_CLR)

    # Plot 3 — Precision & Recall
    ax3 = styled_ax(gs[1, 0])
    if 'metrics/precision(B)' in df.columns:
        ax3.plot(epochs, df['metrics/precision(B)'], color=CYAN,
                 label='Precision', linewidth=1.5)
    if 'metrics/recall(B)' in df.columns:
        ax3.plot(epochs, df['metrics/recall(B)'], color=ORANGE,
                 label='Recall', linewidth=1.5)
    ax3.set_title('Precision & Recall', fontsize=10)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.set_ylim(0, 1.05)
    ax3.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=TEXT_CLR)

    # Plot 4 — Obj / cls loss
    ax4 = styled_ax(gs[1, 1])
    for col, color, label in [
        ('train/cls_loss', RED,    'train cls loss'),
        ('val/cls_loss',   YELLOW, 'val cls loss'),
        ('train/dfl_loss', CYAN,   'train dfl loss'),
    ]:
        if col in df.columns:
            ax4.plot(epochs, df[col], color=color,
                     label=label, linewidth=1.3, alpha=0.85)
    ax4.set_title('Classification & DFL Loss', fontsize=10)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=TEXT_CLR)

    return fig_to_numpy(fig)


def plot_heading_history(stats_obj) -> np.ndarray:
    """
    Polar histogram of all heading angles seen this session.
    Shows which directions drones have been flying most.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(5, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax  = fig.add_subplot(111, projection='polar')
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    ax.spines['polar'].set_color(GRID_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)

    angles_deg = [d.get('heading', {}).get('angle_nav')
                  for d in getattr(stats_obj, '_heading_buffer', [])
                  if d.get('heading')]
    angles_deg = [a for a in angles_deg if a is not None]

    if not angles_deg:
        ax.text(0, 0, 'No heading\ndata yet',
                ha='center', va='center',
                color=TEXT_CLR, fontsize=9)
        ax.set_title('Heading Distribution', color=TEXT_CLR, fontsize=10)
        plt.tight_layout()
        return fig_to_numpy(fig)

    bins   = np.linspace(0, 2 * np.pi, 17)
    angles = np.radians(angles_deg)
    counts, _ = np.histogram(angles, bins=bins)
    width  = bins[1] - bins[0]
    bars   = ax.bar(bins[:-1], counts, width=width, align='edge',
                    alpha=0.8, color=CYAN, edgecolor=DARK_BG)

    # Colour bars by quadrant
    for bar, angle in zip(bars, bins[:-1]):
        deg = np.degrees(angle) % 360
        if deg < 90:   bar.set_facecolor(GREEN)
        elif deg < 180: bar.set_facecolor(CYAN)
        elif deg < 270: bar.set_facecolor(ORANGE)
        else:           bar.set_facecolor(RED)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title('Heading Distribution', color=TEXT_CLR,
                 fontsize=10, pad=12)
    ax.set_xticklabels(['N','NE','E','SE','S','SW','W','NW'],
                        color=TEXT_CLR, fontsize=8)
    plt.tight_layout()
    return fig_to_numpy(fig)


# ── Module-level singleton ────────────────────────────────────
stats = SessionStats(max_history=300)
