"""
app.py — Phase 7 complete dashboard.
Two pages via st.tabs:
  Tab 1 — Live Feed  (video + status + alert log + approach warnings)
  Tab 2 — Analytics  (all graphs + heading polar chart + model curves)
"""

import os
import time
import cv2
import tempfile
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    MODEL_PATH, CONFIDENCE, IMG_SIZE, ZONE_FRAMES,
    HEATMAP_ENABLED, HEATMAP_ALPHA,
    HEATMAP_UPDATE_EVERY, HEATMAP_RENDER_EVERY,
    SNAPSHOT_ZONES, GRAPH_REFRESH_FRAMES,
    TRAINING_RESULTS_CSV,
)
from utils   import heatmap
from logger  import get_log, get_recent_snapshots, clear_log, get_log_count
from main    import process_frame
from stats   import (
    stats,
    plot_fps, plot_gpu, plot_detections, plot_confidence,
    plot_zone_hits, plot_alert_timeline, plot_model_training,
    plot_heading_history,
)


# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title = 'Drone Intrusion Detection',
    page_icon  = '🛸',
    layout     = 'wide',
)

# Start GPU polling background thread
stats.start_gpu_polling()

# ── Session state ─────────────────────────────────────────────
if 'running' not in st.session_state:
    st.session_state.running      = False
if 'frame_count' not in st.session_state:
    st.session_state.frame_count  = 0

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.image('https://img.icons8.com/fluency/48/drone.png', width=40)
    st.title('Drone Detection')
    st.caption('Phase 7 — Full Dashboard')
    st.markdown('---')

    # ── Source ────────────────────────────────────────────────
    st.subheader('📹 Video Source')
    source_input = st.text_input('Video URL or Camera ID', value='0')
    uploaded_video = st.file_uploader("Upload Video (.mp4, .avi)", type=['mp4', 'avi', 'mov'])

    st.markdown('---')
    st.subheader('🖼️ Photo Source')
    uploaded_image = st.file_uploader("Upload Image to Test (.jpg, .png)", type=['jpg', 'jpeg', 'png'])

    # ── Overlay controls ──────────────────────────────────────
    st.markdown('---')
    st.subheader('🎨 Overlays')
    show_trail      = st.checkbox('Flight trail',          value=True)
    show_prediction = st.checkbox('Trajectory prediction', value=True)
    show_heatmap    = st.checkbox('Heatmap tracking (Warning: Lowers FPS)', value=False)
                                 
    # ── Heading controls ──────────────────────────────────────
    st.markdown('---')
    st.subheader('🧭 Heading')
    show_heading   = st.checkbox('Show OBB Heading Arrow',   value=True)
    show_compass   = st.checkbox('Show compass rose',        value=True)
    approach_alert = st.checkbox('Approach warnings',        value=True)

    # ── Reset buttons ─────────────────────────────────────────
    st.markdown('---')
    if st.button('🗑️ Clear Heatmap & Log'):
        heatmap.reset()
        clear_log()
        stats.reset()

    # ── Live metrics bar ──────────────────────────────────────
    st.markdown('---')
    st.subheader('📊 Live Metrics')
    m1, m2 = st.columns(2)
    fps_metric   = m1.empty()
    gpu_metric   = m2.empty()
    m3, m4 = st.columns(2)
    det_metric   = m3.empty()
    alert_metric = m4.empty()


# ── Main tabs ─────────────────────────────────────────────────
tab_live, tab_analytics = st.tabs(['🎥 Live Feed', '📈 Analytics'])


# ────────────────────────────────────────────────────────────
# TAB 1 — LIVE FEED
# ────────────────────────────────────────────────────────────
with tab_live:
    col_video, col_status = st.columns([3, 1])

    with col_video:
        st.subheader('Live Feed')
        frame_ph = st.empty()

    with col_status:
        st.subheader('Active Drones')
        status_ph = st.empty()
        st.subheader('Zone Counts')
        zone_ph   = st.empty()

    st.markdown('---')
    col_log_hdr, col_csv = st.columns([4, 1])
    with col_log_hdr:
        log_title_ph = st.empty()
    with col_csv:
        csv_ph = st.empty()
    log_ph = st.empty()

    st.markdown('---')
    st.subheader('📍 Cumulative Heatmap')
    col_hm, col_hm_stats = st.columns([2, 1])
    with col_hm:
        heatmap_ph = st.empty()
    with col_hm_stats:
        heatmap_stats_ph = st.empty()


# ────────────────────────────────────────────────────────────
# TAB 2 — ANALYTICS
# ────────────────────────────────────────────────────────────
with tab_analytics:
    st.subheader('Runtime Performance')
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fps_graph_ph = st.empty()
    with col_g2:
        gpu_graph_ph = st.empty()

    col_g3, col_g4 = st.columns(2)
    with col_g3:
        det_graph_ph  = st.empty()
    with col_g4:
        conf_graph_ph = st.empty()

    st.markdown('---')
    st.subheader('Threat Analysis')
    col_g5, col_g6 = st.columns(2)
    with col_g5:
        # Replaced zone graph with the new polar heading chart
        heading_graph_ph = st.empty()
    with col_g6:
        alert_graph_ph = st.empty()

    st.markdown('---')
    st.subheader('Zone Hits')
    col_g7, col_g8 = st.columns(2)
    with col_g7:
        zone_graph_ph = st.empty()

    st.markdown('---')
    st.subheader('Model Training Curves')
    st.caption(f'Source: {TRAINING_RESULTS_CSV}')
    training_graph_ph = st.empty()

    # Render training curves once on load (static — not per frame)
    training_img = plot_model_training(TRAINING_RESULTS_CSV)
    training_graph_ph.image(training_img, use_container_width=True)


# ────────────────────────────────────────────────────────────
# MAIN VIDEO LOOP
# ────────────────────────────────────────────────────────────
if uploaded_image is not None:
    bytes_data = uploaded_image.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    annotated, detections = process_frame(
        cv2_img,
        show_trail      = show_trail,
        show_prediction = show_prediction,
        show_arrow      = True,
        show_heatmap    = show_heatmap,
        show_heading    = show_heading,
    )
    
    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    frame_ph.image(rgb, channels='RGB', use_container_width=True)
    
    status_html = ''
    BADGE = {'HIGH': 'background:#FF4444;color:white', 'MEDIUM': 'background:#FF8C00;color:white', 'LOW': 'background:#FFD700;color:black', None: 'background:#2E7D32;color:white'}
    
    for d in detections:
        b = BADGE.get(d['zone'], BADGE[None])
        z = d['zone'] or 'SAFE'
        h = d.get('heading')
        heading_html = f'<span style="color:#00D4FF;font-size:11px">&nbsp; {h["angle_nav"]:.0f}° {h["cardinal"]}</span>' if h else ''
        dist = d.get('distance_m')
        dist_html = f'<span style="color:#00FF00;font-size:11px;margin-left:5px;">Dist:{dist:.1f}m</span>' if dist else ''
        status_html += f'<div style="margin-bottom:10px;padding:6px 8px;border-left:3px solid #2A2D3A"><b>Photo Detection</b>&nbsp;<span style="{b};padding:2px 8px;border-radius:10px;font-size:12px">{z}</span>{heading_html}{dist_html}</div>'
        
    if not detections:
        status_html = '<span style="color:#aaa;font-size:13px">No drones detected in photo</span>'
        
    status_ph.markdown(status_html, unsafe_allow_html=True)
    st.stop()

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    src = tfile.name
else:
    src = int(source_input) if source_input.strip() == '0' else source_input.strip()

cap = cv2.VideoCapture(src)

zone_hit_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
frame_count     = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.warning('Video stream ended or camera not found.')
        break

    frame_count += 1
    st.session_state.frame_count = frame_count

    # ── Run pipeline ──────────────────────────────────────────
    annotated, detections = process_frame(
        frame,
        show_trail      = show_trail,
        show_prediction = show_prediction,
        show_arrow      = False,
        show_heatmap    = show_heatmap,
        show_heading    = show_heading,
    )

    # ── Display feed ──────────────────────────────────────────
    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    frame_ph.image(rgb, channels='RGB', use_container_width=True)

    # ── Update zone hit counts ────────────────────────────────
    for d in detections:
        if d['zone']:
            zone_hit_counts[d['zone']] += 1

    # ── Status panel ──────────────────────────────────────────
    status_html = ''
    BADGE = {
        'HIGH':   'background:#FF4444;color:white',
        'MEDIUM': 'background:#FF8C00;color:white',
        'LOW':    'background:#FFD700;color:black',
        None:     'background:#2E7D32;color:white',
    }
    
    for d in detections:
        b    = BADGE.get(d['zone'], BADGE[None])
        z    = d['zone'] or 'SAFE'
        h    = d.get('heading')
        warn = d.get('approach_warnings', [])

        # Cardinal badge
        if h:
            nav_angle = h['angle_nav']
            cardinal  = h['cardinal']
            speed     = h['speed_px']
            heading_html = (
                f'<span style="color:#00D4FF;font-size:11px">'
                f'&nbsp; {nav_angle:.0f}° {cardinal} '
                f'({speed:.0f}px/f)</span>'
            )
        else:
            heading_html = '<span style="color:#555;font-size:11px"> hovering</span>'

        dist = d.get('distance_m')
        dist_html = f'<span style="color:#00FF00;font-size:11px;margin-left:5px;">Dist:{dist:.1f}m</span>' if dist else ''

        # Approach warning badge
        warn_html = ''
        if warn and approach_alert:
            warn_html = (
                f'<span style="background:#FF4444;color:white;'
                f'padding:1px 6px;border-radius:8px;font-size:11px">'
                f'⚠ → {warn[0]}</span>'
            )

        status_html += (
            f'<div style="margin-bottom:10px;padding:6px 8px;'
            f'border-left:3px solid #2A2D3A">'
            f'<b>Drone #{d["track_id"]}</b>&nbsp;'
            f'<span style="{b};padding:2px 8px;border-radius:10px;'
            f'font-size:12px">{z}</span>'
            f'{heading_html}{dist_html} {warn_html}'
            f'</div>'
        )

    if not detections:
        status_html = '<span style="color:#aaa;font-size:13px">No drones detected</span>'

    status_ph.markdown(status_html, unsafe_allow_html=True)

    zone_html = ''.join([
        f'<div style="display:flex;justify-content:space-between;'
        f'margin-bottom:5px">'
        f'<span style="color:{c};font-weight:bold">{z}</span>'
        f'<span style="color:#ccc">{zone_hit_counts[z]}</span></div>'
        for z, c in [('HIGH','#FF4444'),('MEDIUM','#FF8C00'),('LOW','#FFD700')]
    ])
    zone_ph.markdown(zone_html, unsafe_allow_html=True)

    # ── Alert log + CSV export ────────────────────────────────
    log_entries = get_log()
    log_title_ph.markdown(f'### 🚨 Alert Log  `{len(log_entries)} events`')

    if log_entries:
        df_log = pd.DataFrame(log_entries)
        cols = ['timestamp','track_id','zone','drone_type','type_threat','confidence','position','snapshot']
        cols = [c for c in cols if c in df_log.columns]
        log_ph.dataframe(
            df_log[cols],
            use_container_width=True, hide_index=True,
        )
        csv_ph.download_button(
            label     = '⬇️ Export CSV',
            data      = df_log.to_csv(index=False).encode(),
            file_name = f'intrusion_log_{int(time.time())}.csv',
            mime      = 'text/csv',
            key       = f'csv_{frame_count}',
        )
    else:
        log_ph.info('No alerts this session.')

    # ── Sidebar live metrics ──────────────────────────────────
    fps_metric.metric('FPS',    f'{stats.current_fps:.1f}',
                      f'avg {stats.avg_fps:.1f}')
    gpu_metric.metric('VRAM',   f'{stats.gpu_used_mb} MB',
                      f'{stats.gpu_pct:.0f}%')
    det_metric.metric('Dets',   len(detections))
    alert_metric.metric('Alerts', get_log_count())

    # ── Graphs + heatmap (every N frames) ────────────────────
    if frame_count % GRAPH_REFRESH_FRAMES == 0:

        # Heatmap panel
        snap = heatmap.get_snapshot()
        heatmap_ph.image(
            cv2.cvtColor(snap, cv2.COLOR_BGR2RGB),
            caption='Red = highest activity',
            use_container_width=True,
        )
        heatmap_stats_ph.markdown(
            f'**Hits:** {heatmap.total_detections}  \n'
            f'**Peak:** {heatmap.acc.max():.1f}  \n'
            f'**Session:** {stats.session_duration}'
        )

        # Sidebar snapshots & Drone Type Bar Chart
        recent = get_recent_snapshots(n=3)
        snapshot_sidebar.empty() # clear placeholder
        with snapshot_sidebar.container():
            for path in recent:
                img = cv2.imread(path)
                if img is not None:
                    st.image(
                        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                        caption=os.path.basename(path),
                        use_container_width=True,
                    )
            
            # Phase 8: Drone Type distribution
            if log_entries:
                df_logs = pd.DataFrame(log_entries)
                if 'drone_type' in df_logs.columns:
                    st.markdown('---')
                    st.subheader("Drone Types Detected")
                    st.bar_chart(df_logs['drone_type'].value_counts())

        # Analytics tab graphs
        fps_graph_ph.image(plot_fps(stats),         use_container_width=True)
        gpu_graph_ph.image(plot_gpu(stats),         use_container_width=True)
        det_graph_ph.image(plot_detections(stats),  use_container_width=True)
        conf_graph_ph.image(plot_confidence(stats), use_container_width=True)
        zone_graph_ph.image(plot_zone_hits(stats),  use_container_width=True)
        alert_graph_ph.image(plot_alert_timeline(stats), use_container_width=True)
        heading_graph_ph.image(plot_heading_history(stats), use_container_width=True)

cap.release()

if frame_count > 0:
    with tab_live:
        st.markdown("---")
        st.subheader("🔥 Final Flight Trajectory (Standalone)")
        snap = heatmap.get_snapshot()
        rgb_snap = cv2.cvtColor(snap, cv2.COLOR_BGR2RGB)
        st.image(rgb_snap, use_container_width=True, caption="Heatmap on pure black background")
