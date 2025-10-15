"""
Live Video Display Component
Shows video frames with bounding boxes in real-time
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
import queue
from utils.websocket_client import WebSocketClient


def render_live_video_section(job_id: str, api_base_url: str):
    """
    Render live video display with bounding boxes.
    
    Args:
        job_id: Job ID to monitor
        api_base_url: Base URL of API
    """
    
    st.subheader("🎥 Live Video Feed")
    
    # Create placeholder for video frame
    video_placeholder = st.empty()
    
    # Create placeholder for track info
    info_placeholder = st.empty()
    
    # Initialize WebSocket connection state
    if 'ws_connected' not in st.session_state:
        st.session_state['ws_connected'] = False
    if 'latest_frame_data' not in st.session_state:
        st.session_state['latest_frame_data'] = None
    if 'ws_client' not in st.session_state:
        st.session_state['ws_client'] = None
    if 'ws_queue' not in st.session_state:
        st.session_state['ws_queue'] = queue.Queue()
    
    # Connection status
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.session_state['ws_connected']:
            st.success("🟢 WebSocket Connected - Receiving real-time updates")
        else:
            st.warning("🟡 Connecting to WebSocket...")
    
    with col2:
        if st.button("🔌 Reconnect"):
            # Safely disconnect existing client
            try:
                if st.session_state['ws_client'] is not None:
                    st.session_state['ws_client'].disconnect()
            except Exception:
                pass
            # Reset state
            st.session_state['ws_connected'] = False
            st.session_state['latest_frame_data'] = None
            st.session_state['ws_client'] = None
            st.session_state['ws_queue'] = queue.Queue()
            st.rerun()
    
    # Display instructions
    with st.expander("ℹ️ How it works", expanded=False):
        st.write("""
        **Real-time Video Streaming:**
        - WebSocket connection to backend
        - Frames streamed as they're processed
        - Bounding boxes drawn for vehicles
        - Track IDs and speeds displayed
        - Violations highlighted in red
        
        **Color Coding:**
        - 🟢 Green: Normal vehicles
        - 🔴 Red: Speed violations
        - 🟡 Yellow: Plates detected
        """)
    
    # Establish WebSocket connection if not connected
    if not st.session_state['ws_connected']:
        # Capture a plain Queue reference to avoid touching Streamlit in background thread
        _ws_qref = st.session_state.get('ws_queue')

        def _on_message(data: dict):
            # Do NOT call Streamlit APIs from background thread. Only enqueue data.
            try:
                if _ws_qref is not None:
                    _ws_qref.put_nowait(data)
            except Exception:
                # If queue not available, drop frame rather than touching Streamlit
                pass

        def _on_error(err):
            # Errors can be reflected on next UI refresh by lack of frames
            return

        def _on_close():
            return

        # Create and connect client if not already set
        if st.session_state['ws_client'] is None:
            try:
                ws_client = WebSocketClient(job_id=job_id, base_url=api_base_url)
                ws_client.connect(on_message=_on_message, on_error=_on_error, on_close=_on_close)
                st.session_state['ws_client'] = ws_client
            except Exception:
                pass

    # Drain queue and update latest_frame_data in main thread
    try:
        q = st.session_state.get('ws_queue')
        latest = None
        if q is not None:
            while not q.empty():
                latest = q.get_nowait()
        if latest is not None:
            st.session_state['latest_frame_data'] = latest
            st.session_state['ws_connected'] = True
    except Exception:
        pass

    # Display latest frame if available
    if st.session_state.get('latest_frame_data'):
        display_frame_with_boxes(
            st.session_state['latest_frame_data'],
            video_placeholder,
            info_placeholder
        )
    else:
        with video_placeholder.container():
            st.info("⏳ Waiting for video frames...")
            st.write("Frames will appear here as video processes")


def display_frame_with_boxes(frame_data: dict, video_placeholder, info_placeholder):
    """Display frame with bounding boxes and track information."""
    
    tracks = frame_data.get('tracks', [])
    frame_number = frame_data.get('frame_number', 0)
    progress = frame_data.get('progress', 0)
    stats = frame_data.get('stats', {})
    # Prefer 'frame_base64' if provided by backend, else fall back to 'frame'
    frame_base64 = frame_data.get('frame_base64') or frame_data.get('frame')
    
    with video_placeholder.container():
        # Display the actual video frame if available
        if frame_base64:
            import base64
            from PIL import Image
            import io
            
            # Decode base64 to image
            frame_bytes = base64.b64decode(frame_base64)
            image = Image.open(io.BytesIO(frame_bytes))
            
            # Display image
            st.image(image, caption=f"Frame {frame_number} | Progress: {progress:.1f}%")
        else:
            st.info(f"**Frame:** {frame_number} | **Progress:** {progress:.1f}%")
            st.write("Waiting for frame data...")
        
        # Display track information
        if tracks:
            st.write(f"**Active Tracks:** {len(tracks)}")
            
            # Show tracks in columns
            cols = st.columns(min(len(tracks), 4))
            for i, track in enumerate(tracks[:4]):
                with cols[i]:
                    track_id = track.get('track_id')
                    vehicle_class = track.get('class', 'unknown')
                    confidence = track.get('confidence', 0) * 100
                    
                    st.metric(
                        f"🚗 ID: {track_id}",
                        vehicle_class,
                        f"{confidence:.0f}% conf"
                    )
    
    # Display statistics
    with info_placeholder.container():
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Detections", stats.get('detections', 0))
        with col2:
            st.metric("Tracked", stats.get('tracked', 0))
        with col3:
            st.metric("Plates", stats.get('plates', 0))
        with col4:
            violations = stats.get('violations', 0)
            if violations > 0:
                st.metric("⚠️ Violations", violations)
            else:
                st.metric("✅ Violations", 0)


def _render_metadata(tracks: list):
    """Render a compact metadata view of current tracks when no image is available."""
    if tracks:
        st.write(f"**Active Tracks:** {len(tracks)}")
        cols = st.columns(min(len(tracks), 4))
        for i, track in enumerate(tracks[:4]):  # Show max 4
            with cols[i]:
                track_id = track.get('track_id')
                vehicle_class = track.get('class', 'unknown')
                confidence = track.get('confidence', 0) * 100
                st.metric(
                    f"🚗 ID: {track_id}",
                    vehicle_class,
                    f"{confidence:.0f}% conf"
                )
    else:
        st.info("No vehicles detected in current frame")


def draw_bounding_boxes(frame: np.ndarray, tracks: list) -> np.ndarray:
    """
    Draw bounding boxes on frame.
    
    Args:
        frame: Video frame (numpy array)
        tracks: List of track dictionaries with bbox info
    
    Returns:
        Frame with bounding boxes drawn
    """
    
    # Color scheme
    colors = {
        'car': (0, 255, 0),      # Green
        'truck': (0, 255, 255),  # Yellow
        'bus': (255, 0, 0),      # Blue
        'motorcycle': (0, 0, 255),  # Red
        'bicycle': (255, 0, 255)    # Magenta
    }
    
    violation_color = (0, 0, 255)  # Red for violations
    
    for track in tracks:
        bbox = track.get('bbox', [])
        if len(bbox) != 4:
            continue
        
        x1, y1, x2, y2 = map(int, bbox)
        track_id = track.get('track_id', 0)
        vehicle_class = track.get('class', 'vehicle')
        confidence = track.get('confidence', 0)
        is_violation = track.get('is_violation', False)
        
        # Choose color
        color = violation_color if is_violation else colors.get(vehicle_class, (255, 255, 255))
        
        # Draw bounding box
        thickness = 3 if is_violation else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label background
        label = f"ID:{track_id} {vehicle_class} {confidence:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        
        # Draw label text
        cv2.putText(
            frame, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        
        # Draw violation warning if applicable
        if is_violation:
            warning = "⚠️ SPEEDING"
            cv2.putText(
                frame, warning, (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, violation_color, 2
            )
    
    return frame


def numpy_to_base64(frame: np.ndarray) -> str:
    """Convert numpy frame to base64 string."""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image
    pil_img = Image.fromarray(frame_rgb)
    # Convert to bytes
    buff = io.BytesIO()
    pil_img.save(buff, format="JPEG", quality=85)
    # Encode to base64
    img_str = base64.b64encode(buff.getvalue()).decode()
    return img_str